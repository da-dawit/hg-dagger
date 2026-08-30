#!/usr/bin/env bash
# GR00T N1.7-3B -- HG-DAgger iteration 1, warm-started from the 16D 30k checkpoint.
#
# Differs from 05_train_80k.sh in three ways, each with a reason:
#
#   WARM START, NOT RESUME.  checkpoint-30000 holds WEIGHTS ONLY -- no optimizer.pt, no DeepSpeed
#   global_step*/, and its cosine LR is annealed to 3.0e-13. There is nothing to resume from. This
#   is a fresh schedule over pretrained weights, so the LR is 3e-5 rather than 1e-4: at 1e-4 a
#   converged checkpoint gets kicked out of its basin in the first few hundred steps.
#
#   MIXTURE, NOT ONE PATH.  Without an explicit ratio, factory.py:77 weights datasets by SIZE.
#   The DAgger set is 43,686 frames against the original's 55,701, so it would silently take 44%
#   of the mix. The path@ratio syntax patched into launch_finetune.py sets it deliberately.
#
#   ~3k STEPS, NOT 80k.  Combined train is 99,387 frames; at global batch 128 one epoch is ~776
#   steps, so 3000 is ~3.9 epochs. groot35 converged at epoch ~2.07. 80k would be catastrophic
#   overfitting onto 13 correction episodes. Save every 250 and pick by held-out grasp error --
#   checkpoint_sweep.py exists for exactly this.
set -euo pipefail

ROOT=${ROOT:-$HOME/screwing_train}
export GR00T_SQUARE_CROP=1
export GR00T_ROI_JSON=${GR00T_ROI_JSON:-$ROOT/eval/roi_hybrid.json}

BASE=${BASE:-$ROOT/runs/16d_30k/checkpoint-30000}   # the checkpoint that grasps on hardware
ORIG=${ORIG:-$ROOT/datasets/screwing35_follower_train}
HUMAN=${HUMAN:-$ROOT/datasets/dagger_human}          # 39 eps: the operator's corrections ONLY
AUTO=${AUTO:-$ROOT/datasets/dagger_auto}             # 52 eps: the policy's own rollout frames
VAL=${VAL:-$ROOT/datasets/screwing35_follower_val}
OUT=${OUT:-$ROOT/runs/groot_dagger_it1}
GR=${GR:-$ROOT/Isaac-GR00T}
PATCH=${PATCH:-$ROOT/vastai/patches/crop_to_square.patch}
STEPS=${STEPS:-6000}
LR=${LR:-3e-5}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
BATCH=${BATCH:-128}
SAVE_STEPS=${SAVE_STEPS:-250}
SAVE_LIMIT=${SAVE_LIMIT:-24}
MASTER_PORT=${MASTER_PORT:-29500}
#NVIDIA's FAQ strongly recommends colour jitter for a fixed scene with few episodes -- ours is one
#taped rectangle. The 30k run used NONE. Mild values only: this is a regulariser against repeating
#39 corrections 8.5x, not a domain-randomisation exercise. JITTER='' disables it.
#tyro takes KEY VALUE pairs here, not a JSON string -- a JSON string is rejected at parse time.
JITTER=${JITTER:-"brightness 0.2 contrast 0.2 saturation 0.2 hue 0.03"}

#Ilia's 0.60 original / 0.30 DAgger / 0.10 replay, at the DATASET level because Isaac-GR00T has
#no per-frame loss weighting (no sample_weight anywhere in the loss path). export_dagger_split.py
#did the frame-level work up front: dagger_human is the 39 intervention runs, dagger_auto is the
#policy's own frames between them, and the handover pauses plus a 2s pre-intervention window were
#dropped entirely rather than zero-weighted.
#
#Without these ratios factory.py:77 weights by SIZE, and the corrections are 6,765 frames against
#the original's 55,701 -- they would land at 11% of the mix instead of 30%.
for _d in "$ORIG" "$HUMAN" "$AUTO"; do
  [ -d "$_d" ] || { echo "missing dataset: $_d"; echo "run eval/export_dagger_split.py first"; exit 1; }
done
MIX="${ORIG}@0.60:${HUMAN}@0.30:${AUTO}@0.10"
echo "mixture: 0.60 original / 0.30 dagger_human / 0.10 dagger_auto"
python3 - "$ORIG" "$HUMAN" "$AUTO" <<'PYX'
import json,sys,pathlib
for tag,p in zip(("original","human","auto"),sys.argv[1:]):
    i=json.loads((pathlib.Path(p)/"meta"/"info.json").read_text())
    print(f"         {tag:<9} {i['total_episodes']:3d} eps  {i['total_frames']:6d} frames")
PYX


# ---------------------------------------------------------------- preflight: disk and VRAM
#Each GR00T N1.7-3B checkpoint is ~6.5 GB. save_steps 250 over 3000 steps = 12 of them = 78 GB,
#and a rented box that runs out of disk at step 2750 has wasted the whole rental. Check first.
_CKPT_GB=7
_N_CKPT=$(( (STEPS + SAVE_STEPS - 1) / SAVE_STEPS ))
[ "$_N_CKPT" -gt "$SAVE_LIMIT" ] && _N_CKPT=$SAVE_LIMIT
_NEED=$(( _CKPT_GB * _N_CKPT + 10 ))
_FREE=$(df -BG --output=avail "$(dirname "$OUT")" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "disk: need ~${_NEED} GB (${_N_CKPT} checkpoints x ${_CKPT_GB} GB + slack), have ${_FREE:-?} GB"
if [ -n "$_FREE" ] && [ "$_FREE" -lt "$_NEED" ]; then
  echo "  NOT ENOUGH DISK. Either raise SAVE_STEPS, lower SAVE_LIMIT, or get a bigger box."
  echo "  Do not start a run that will die at step $(( SAVE_STEPS * (_FREE / _CKPT_GB) ))."
  exit 1
fi

#Training loads the model in FP32 -- launch_finetune.py:110 hardcodes load_bf16=False, and bf16=True
#only turns on autocast. So per GPU, before activations:
#    params fp32 12.6 GB (ZeRO-2 does NOT shard these) + grads/N + AdamW moments/N
#On 2 GPUs that is ~22 GB, leaving ~17 GB on a 40 GB A100 and ~57 GB on an 80 GB card.
#Measured: this exact config ran at global batch 128 on 2xH100-80GB. 40 GB cards are UNTESTED here.
_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "VRAM: ${_VRAM:-?} MiB/GPU x ${NUM_GPUS}"
if [ -n "$_VRAM" ] && [ "$_VRAM" -lt 60000 ] && [ "$BATCH" -gt 64 ]; then
  echo "  WARNING: ${_VRAM} MiB/GPU with global batch $BATCH ($((BATCH / NUM_GPUS))/GPU)."
  echo "  The 30k run used 2x80GB. If the smoke step OOMs, drop to BATCH=64 and raise STEPS to"
  echo "  $(( STEPS * 2 )) to keep the same number of epochs."
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT"
else
  LAUNCH="python3"
fi
echo "launcher: $LAUNCH   ($NUM_GPUS GPU(s), global batch $BATCH -> $((BATCH / NUM_GPUS))/GPU)"

# ---------------------------------------------------------------- 0. the CropToSquare patch
if ! grep -q "class CropToSquare" "$GR/gr00t/model/gr00t_n1d7/image_augmentations.py"; then
  echo "[0/5] applying crop_to_square.patch"
  ( cd "$GR" && git apply "$PATCH" ) || { echo "PATCH FAILED -- the run would letterbox"; exit 1; }
fi
grep -q "class CropToSquare" "$GR/gr00t/model/gr00t_n1d7/image_augmentations.py" \
  || { echo "CropToSquare missing after patch"; exit 1; }
[ -f "$GR00T_ROI_JSON" ] || { echo "GR00T_ROI_JSON not found: $GR00T_ROI_JSON"; exit 1; }

#the mixture syntax is a local patch too. Without it every path lands in ONE spec and the ratios
#above are silently ignored -- the failure is invisible in the logs.
grep -q '"@" in _entry' "$GR/gr00t/experiment/launch_finetune.py" \
  || { echo "launch_finetune.py lacks the path@ratio patch -- ratios would be IGNORED"; exit 1; }
echo "[0/5] CropToSquare + path@ratio both present"

# ---------------------------------------------------------------- 1. base checkpoint sanity
[ -d "$BASE" ] || { echo "base checkpoint not found: $BASE"; exit 1; }
python3 - "$BASE" <<'PY'
import json, sys, pathlib
b = pathlib.Path(sys.argv[1])
st = json.loads((b / "trainer_state.json").read_text())
print(f"[1/5] warm start from step {st.get('global_step')}, "
      f"last LR {st['log_history'][-1].get('learning_rate'):.2e}")
s = json.loads((b / "experiment_cfg" / "dataset_statistics.json").read_text())
def walk(o, p=""):
    for k, v in (o.items() if isinstance(o, dict) else []):
        if isinstance(v, dict) and "mean" in v:
            print(f"        {p}/{k}: dim={len(v['mean'])}")
        elif isinstance(v, dict):
            walk(v, p + "/" + k)
walk(s)
PY

# ---------------------------------------------------------------- 2. gate BOTH datasets
echo "[2/5] dataset gate -- refuses on any degenerate normaliser among DECLARED dims"
for d in "$ORIG" "$HUMAN" "$AUTO"; do
  echo "    --- $(basename "$d")"
  python3 "$ROOT/eval/verify_dataset.py" --root "$d" --lead 5 --isaac
done

# ---------------------------------------------------------------- 3. smoke: 20 steps
if [ "${SMOKE:-1}" = "1" ]; then
  echo "[3/5] 20-step smoke -- config, mixture, one batch, fwd/bwd, loss"
  ( cd "$GR" && $LAUNCH gr00t/experiment/launch_finetune.py \
      --base-model-path "$BASE" \
      --dataset-path    "$MIX" \
      --embodiment-tag  NEW_EMBODIMENT \
      --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_16d_config.py \
      --num-gpus "$NUM_GPUS" \
      --output-dir /tmp/groot_dagger_smoke \
      --max-steps 20 --save-steps 1000 --global-batch-size "$BATCH" \
      --dataloader-num-workers ${WORKERS:-16} \
      --learning-rate "$LR" \
      --no-tune-llm --no-tune-visual \
      --tune-projector --tune-diffusion-model \
      --color-jitter-params $JITTER \
      --state-dropout-prob 0.2 )
  echo "    smoke OK -- read the step rate above and multiply by $STEPS for the ETA"
fi

# ---------------------------------------------------------------- 4. the run
echo "[4/5] launching $STEPS steps on $NUM_GPUS GPU(s), lr $LR"
cd "$GR"
$LAUNCH gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path    "$MIX" \
  --embodiment-tag  NEW_EMBODIMENT \
  --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_16d_config.py \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUT" \
  --max-steps "$STEPS" \
  --global-batch-size "$BATCH" \
  --dataloader-num-workers ${WORKERS:-16} \
  --learning-rate "$LR" \
  --weight-decay 1e-5 \
  --warmup-ratio 0.03 \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --state-dropout-prob 0.2 \
  --color-jitter-params $JITTER \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_LIMIT" \
  --save-only-model \
  --use-wandb --wandb-project screwing_hil

# ---------------------------------------------------------------- 5. pick by held-out error
#NO IN-TRAINING EVAL IS POSSIBLE. factory.py's build() hard-asserts eval_strategy == "no" --
#"Sharded dataset does not support evaluation sets". Setting it kills the run at startup. So the
#checkpoint MUST be picked afterwards, and loss cannot do it: 39 corrections are easy to memorise,
#so train loss keeps falling while the arm gets worse.
echo "[5/5] done. Pick the checkpoint by MEASUREMENT, not by final loss:"
echo "    python3 $ROOT/eval/checkpoint_sweep.py --run $OUT --dataset $VAL"
echo "  Training loss keeps falling while the arm gets worse -- 13 correction episodes are easy"
echo "  to memorise. The number that matters is grasp error on the held-out set."
