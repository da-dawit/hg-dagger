#!/usr/bin/env bash
# GR00T N1.7-3B -- 16D screwing, 80k steps.
#
# Every flag below is verified present in gr00t.configs.finetune_config.FinetuneConfig (31 fields).
# Flags that do NOT exist and must never be added:
#   --state-gaussian-noise-std   the field has no consumer anywhere in the source
#   --crop-fraction              read from the MODEL config, patched below
#   --use-percentiles            same
#   --tune-top-llm-layers        Isaac path reads it from the model config only (N1.7 ships 0)
#
# The base checkpoint's config.json is loaded via start_from_checkpoint and WINS over the CLI for
# any field FinetuneConfig does not declare. That is why steps 1-2 patch a local copy.
set -euo pipefail

#PER-CAMERA CROP (hybrid). LetterBoxPad pads every view to square with black bars: 44% of every
#model input on our cameras. But the three cameras are not alike, measured on the held-out episodes:
#
#  head    a STATIC context view. 50% of all its motion energy -- 59% during "Grab the driver" --
#          lies in the side bands a centred square crop would delete. The cheapest box keeping 90%
#          of the motion still costs 36% padding. So there is nothing worth cutting: leave it
#          letterboxed, full field of view.
#  wrists  MOVING close-ups whose upper half is ceiling (88% of the frame at the grasp instant).
#          Crop bottom-anchored to square: 0% padding, and the ceiling is gone. These are the
#          0.73-1.08 mm/px cameras where the grasp precision actually lives.
#
#Each camera's crop must be SQUARE, but the squares need NOT match: the trailing
#SmallestMaxSize(256) normalises every one to 256x256 before _get_vlm_inputs stacks the views.
#Verified: 672x672 + 240x240 + 240x240 stacks without error.
#
#  mean black share   44.0% -> 14.7%
#  workspace content  18.6% -> 40.3%  (2.17x), entirely from the wrists; head unchanged
#
#roi_hybrid.json expresses "letterbox the head" as an ROI covering the full frame -- cropping to it
#is a no-op and padding it to square IS LetterBoxPad. The ROI is stored as FRACTIONS keyed by aspect
#ratio, so it survives a camera resolution change (an earlier version keyed on exact pixels and
#would have silently fallen back to a different crop).
export GR00T_SQUARE_CROP=1

ROOT=${ROOT:-$HOME/screwing_train}
export GR00T_ROI_JSON=${GR00T_ROI_JSON:-$ROOT/eval/roi_hybrid.json}
HF_SNAP=${HF_SNAP:-}                       # path to the nvidia/GR00T-N1.7-3B snapshot
BASE=$ROOT/base_model_patched
DATA=${DATA:-$ROOT/datasets/screwing35_follower_train}
VAL=${VAL:-$ROOT/datasets/screwing35_follower_val}
OUT=${OUT:-$ROOT/runs/groot35_16d_80k}
GR=${GR:-$ROOT/Isaac-GR00T}
PATCH=${PATCH:-$ROOT/vastai/patches/crop_to_square.patch}
STEPS=${STEPS:-80000}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
MASTER_PORT=${MASTER_PORT:-29500}

#MULTI-GPU MUST GO THROUGH torchrun. With bare `python`, HF Trainer falls back to DataParallel and
#dies with "module must have its parameters and buffers on device cuda:0 but found one of them on
#device: cpu". DeepSpeed ZeRO-2 also only engages when num_gpus > 1 (experiment.py:197), and
#experiment.py:204 splits global_batch_size across the GPUs, so 64 on 4 cards is 16/GPU.
if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT"
else
  LAUNCH="python3"
fi
echo "launcher: $LAUNCH   ($NUM_GPUS GPU(s), global batch 64 -> $((64 / NUM_GPUS))/GPU)"

# ---------------------------------------------------------------- 0. the CropToSquare patch
if ! grep -q "class CropToSquare" "$GR/gr00t/model/gr00t_n1d7/image_augmentations.py"; then
  echo "[0/6] applying crop_to_square.patch (a fresh clone of the fork does not have it)"
  ( cd "$GR" && git apply "$PATCH" ) || { echo "PATCH FAILED -- do not train, the run would letterbox"; exit 1; }
fi
grep -q "class CropToSquare" "$GR/gr00t/model/gr00t_n1d7/image_augmentations.py" \
  || { echo "CropToSquare missing after patch"; exit 1; }
[ -f "$GR00T_ROI_JSON" ] || { echo "GR00T_ROI_JSON not found: $GR00T_ROI_JSON"; exit 1; }
echo "[0/6] CropToSquare present; per-camera crop from $(basename "$GR00T_ROI_JSON")"
python3 - "$GR00T_ROI_JSON" <<'PYX'
import json,sys
for k,v in json.load(open(sys.argv[1])).items():
    ch,cw=v["height"],v["width"]; m=max(ch,cw)
    print(f"        {k.split('.')[-1]:<16} {cw}x{ch} -> {m}x{m}  padding {1-(ch*cw)/(m*m):5.1%}")
PYX

# ---------------------------------------------------------------- 1. base model copy
if [ ! -d "$BASE" ]; then
  [ -n "$HF_SNAP" ] || { echo "set HF_SNAP to the GR00T-N1.7-3B snapshot dir"; exit 1; }
  echo "[1/6] copying base model (deref symlinks -- the HF cache stores blobs)"
  mkdir -p "$BASE"; cp -rL "$HF_SNAP"/. "$BASE"/
fi

# ---------------------------------------------------------------- 2. patch the model config
echo "[2/6] patching base config: use_percentiles=False, crop_fraction=1.0"
python3 - "$BASE" <<'PY'
import json, sys, pathlib
base = pathlib.Path(sys.argv[1])
PATCH = {
    "use_percentiles": False,   # q01/q99 clipped 0.5-35% of each dim's range on our data
    "crop_fraction": 1.0,       # FractionalRandomCrop(1.0) == identity -> random crop OFF.
                                # The wrist cameras put the workspace in the lower band of the
                                # frame; a 0.95 crop can remove up to 45% of it at the grasp.
                                # image_augmentations.py:210 requires 0.0 < crop_fraction <= 1.0.
}
for name in ("config.json", "processor_config.json"):
    p = base / name
    if not p.exists():
        continue
    c = json.loads(p.read_text())
    tgt = c["processor_kwargs"] if (name == "processor_config.json" and "processor_kwargs" in c) else c
    for k, v in PATCH.items():
        tgt[k] = v
    p.write_text(json.dumps(c, indent=2))
    print(f"    {name}: " + ", ".join(f"{k}={v}" for k, v in PATCH.items()))
PY

# ---------------------------------------------------------------- 3. gate the dataset
echo "[3/6] dataset gate (refuses on any degenerate normaliser among DECLARED dims)"
python3 "$ROOT/eval/verify_dataset.py" --root "$DATA" --val-root "$VAL" --lead 5 --isaac

# ---------------------------------------------------------------- 4. smoke: 20 steps
if [ "${SMOKE:-1}" = "1" ]; then
  echo "[4/6] 20-step smoke -- config loads, dataset loads, one batch, fwd/bwd, loss"
  ( cd "$GR" && $LAUNCH gr00t/experiment/launch_finetune.py \
      --base-model-path "$BASE" \
      --dataset-path    "$DATA" \
      --embodiment-tag  NEW_EMBODIMENT \
      --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_16d_config.py \
      --num-gpus "$NUM_GPUS" \
      --output-dir /tmp/groot_smoke \
      --max-steps 20 --save-steps 1000 --global-batch-size 64 \
      --dataloader-num-workers ${WORKERS:-16} \
      --no-tune-llm --no-tune-visual \
      --tune-projector --tune-diffusion-model \
      --state-dropout-prob 0.2 )
  echo "    smoke OK -- read the tqdm step rate above, multiply by $STEPS for the real ETA"
fi

# ---------------------------------------------------------------- 5. the run
echo "[5/6] launching $STEPS steps on $NUM_GPUS GPU(s)"
cd "$GR"
$LAUNCH gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path    "$DATA" \
  --embodiment-tag  NEW_EMBODIMENT \
  --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_16d_config.py \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUT" \
  --max-steps "$STEPS" \
  --global-batch-size 64 \
  --dataloader-num-workers ${WORKERS:-16} \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --state-dropout-prob 0.2 \
  --save-steps 5000 \
  --save-total-limit 20 \
  --save-only-model \
  --use-wandb --wandb-project screwing_hil
