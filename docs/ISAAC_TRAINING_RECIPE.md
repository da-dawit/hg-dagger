# GR00T N1.7 retrain on the ISAAC path (ROBOTIS fork)

**Re-audited 2026-08-25 from the installed fork, not from the LeRobot audit.**
Source: `ROBOTIS-GIT/Isaac-GR00T-n1.7` @ `e81d02b`, local at
`cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T`, including
`examples/CYCLO/ffw_sg2_rev1` — the modality config for our exact robot.

**The earlier `TRAINING_RECIPE.md` was verified against LeRobot's GR00T port. Nothing in it
transfers to this path without re-checking.** This file supersedes it for Isaac.

---

## 1. Two stacks, and we are switching

| | training | inference |
|---|---|---|
| what we ran before | LeRobot `lerobot_train`, v3.0 dataset, chunk 40 | `aiworker_deploy/groot_policy.py` -> LeRobot |
| ROBOTIS / Cyclo | Isaac `gr00t/experiment/launch_finetune.py` | Cyclo groot container -> Isaac, TensorRT |

Both are internally consistent. Moving training to Isaac means the deployed inference should move
with it, or the checkpoint must be verified to load in both — **not yet checked.**

---

## 2. Verified correct already

Our dataset's 22-D layout matches ROBOTIS's `modality.json` slice-for-slice:

| modality | slice | our columns |
|---|---|---|
| `arm_left` | [0,8) | `arm_l_joint1..7` + `gripper_l_joint1` |
| `arm_right` | [8,16) | `arm_r_joint1..7` + `gripper_r_joint1` |
| `head` | [16,18) | `head_joint1`, `head_joint2` |
| `lift` | [18,19) | `lift_joint` |
| `odometry` | [19,22) | `linear_x`, `linear_y`, `angular_z` |

This also answers the "drop the unused dims" question: **do not drop them.** They are structural
modality keys in ROBOTIS's config, and removing them means diverging from the config ROBOTIS
maintains for this robot.

`FinetuneConfig` defaults that match what we already ran: `learning_rate 1e-4`,
`weight_decay 1e-5`, `warmup_ratio 0.05`, `tune_llm False`, `tune_visual False`,
`tune_projector True`, `tune_diffusion_model True`, `state_dropout_prob 0.2`.
Normalisation is the same too: `use_percentiles False`, `clip_outliers True` (min-max to [-1,1]).

---

## 3. Blockers — resolve before renting

### 3a. ~~`meta/modality.json` is missing~~ RESOLVED -- installed in both datasets

Required by this path. ROBOTIS's README:

```bash
cp examples/CYCLO/ffw_sg2_rev1/modality.json <dataset_path>/meta/modality.json
```

### 3b. Validation is IMPOSSIBLE on this path, not merely unexposed

`launch_finetune.py` parses only `FinetuneConfig` via `tyro.cli(...)`, so `eval_strategy` and
`val_dataset_path` are not flags. But exposing them would not help. The first line of
`DatasetFactory.build()` is:

```python
assert self.config.training.eval_strategy == "no", (
    "Sharded dataset does not support evaluation sets"
)
```

**Isaac's sharded dataset cannot produce an eval set at all.** Any value other than `"no"` aborts
the run. NVIDIA's `finetune_new_embodiment.md` says to pass `--eval-strategy steps --eval-steps
500`; against this fork that is not merely a missing flag, it is an assertion failure.

### 3b-fix. Evaluate checkpoints POST-HOC instead -- and it is better

We already have the harness, and it measures the thing that actually matters:

```bash
python3 hil_dagger/eval/rollout_mae.py --ckpt <ckpt>/pretrained_model --episode 1 \
        --root datasets/screwing35_follower_val
python3 hil_dagger/eval/drift_probe.py --ckpt <ckpt>/pretrained_model
```

An in-training eval loss is flow-matching velocity MSE. What decides whether the robot works is the
**systematic Cartesian offset**, which MSE cannot see -- a policy drifting 50 mm consistently to one
side scores the same as one randomly off by 50 mm, and only the first is visible on the robot.
So the loss we cannot have is the weaker signal anyway.

**Val episode 1 == original episode 32** (verified byte-identical), so every baseline still applies.

To make post-hoc selection possible, the run must keep enough checkpoints:

| flag | value | why |
|---|---|---|
| `--save-steps` | **500** | the previous run's eval bottomed near step 1600; coarse saves would miss it |
| `--save-total-limit` | **12** | the default 5 keeps only the last 5, i.e. exactly the overfitted end |
| `--save-only-model` | **true** | drops optimizer state. Our 014400 `pretrained_model` is 12 GB; a full checkpoint with optimizer is ~3x that, and 12 of those will not fit |

**Budget the disk before renting:** 12 x ~12 GB = ~145 GB of checkpoints.

### 3c. ~~The validation set is a SEPARATE dataset~~ RESOLVED -- split built and verified

`SingleDatasetConfig.val_dataset_path: Optional[str] = None`. Unlike LeRobot's `eval_split=0.1`,
Isaac will not hold episodes out for you. The 35 episodes must be **physically split at conversion
time** — e.g. 31 train / 4 val, keeping the same held-out episodes (31-34) so results stay
comparable with the measurements already taken on episode 32.

### 3d. ~~Dataset format~~ RESOLVED -- both datasets converted to v2.1 and load in Isaac

ROBOTIS's README says "Use a LeRobot **v2.x** dataset". Ours is v3.0. I could **not** verify a
version check in the loader code, so I do not know whether v3.0 fails outright or silently
mis-parses. `Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py` exists for this.
**Unresolved — test on 2 episodes before committing.**

---

## 4. DECIDED: action horizon 16 — and what it propagates into

Carrying over ROBOTIS's `delta_indices=list(range(16))`. The base checkpoint declares
`action_horizon = 40`, so we deliberately supervise 16 of the 40 the head can emit.

**This choice does not stay inside the training config.** `seam_blend()` cross-fades
`min(seam, horizon, horizon - execute_steps)` waypoints, so when `execute_steps >= horizon` the
overlap is zero and the cross-fade **silently disappears** — every re-plan seam executes raw.
Verified numerically:

| horizon | execute_steps | seam_blend | waypoints actually blended |
|---|---|---|---|
| 40 | 25 | 20 | 15 — the current, working setup |
| **16** | **25 (today's spec)** | 20 | **0 — no blending at all** |
| 16 | 16 | 20 | 0 |
| **16** | **10** | **6** | **6 — correct** |

So a 16-step model loaded with today's `spec.EXECUTE_STEPS = 25` would reintroduce exactly the
seam jerk this session removed, without any error.

### The coherent set for horizon 16

| constant | 40-step (now) | **16-step (new model)** | why |
|---|---|---|---|
| action horizon | 40 | **16** | ROBOTIS CYCLO config |
| `spec.EXECUTE_STEPS` | 25 | **10** | keeps the 62.5% executed / 37.5% overlap ratio tuned at 40/25 |
| `--seam-blend` | 20 | **6** | must be <= horizon - execute_steps |
| `dagger_aggregate.CHUNK` | 40 | **16** | a training sample is one horizon; done |

**Do not change `spec.EXECUTE_STEPS` until the 16-step checkpoint exists** — it would break the
model currently deployed. `control_math.check_horizon()` now prints the required values on the
first re-plan of every run, so a mismatch announces itself instead of degrading silently.

A side effect worth knowing: at ~21 Hz, executing 10 waypoints is **0.48 s per plan** versus 1.19 s
at 40/25. Inference must therefore complete ~2.5x faster to stay ahead of the buffer. That makes
TensorRT a requirement of this choice rather than an optimisation — which is consistent with
ROBOTIS shipping the two together, and argues the 16 is deliberate rather than carried over from
the SO100 tutorial.

Good news for data collection: the minimum useful takeover drops from 40 frames (1.33 s) to
16 (0.53 s).

---

## 4b. Previously open, now closed

`examples/CYCLO/ffw_sg2_rev1_config.py` sets `delta_indices=list(range(16))` — a 16-step horizon.
But the base `nvidia/GR00T-N1.7-3B` `config.json` declares **`action_horizon = 40`**, and LeRobot's
port uses 40. We trained at 40.

*Inferred, not confirmed:* the 16 matches NVIDIA's SO100 tutorial verbatim and may be carried over.
It may equally be deliberate — a shorter horizon is cheaper under TensorRT.

It matters concretely: a model supervised on 16 steps but executed with `execute_steps=25` runs
past its trained region. **Ask ROBOTIS before committing a run.**

---

## 5. What does NOT change

**The label fix is pipeline-independent.** `action[t] = observation.state[t+5]` is a
conversion-time data fix and is required on either path.

I verified that Isaac's relative-action mode would *not* have fixed it:
`launch_finetune.py:94` sets `use_relative_action = True`, but that only applies to modality keys
whose `ActionConfig.rep == RELATIVE`, and ROBOTIS sets **all five keys to `ABSOLUTE`** — so it is a
no-op for this robot. And even with RELATIVE it would not help: the conversion is
`relative = action[t+h] - state[t]`, so a constant leader/follower offset survives the subtraction.

---

## 6. Augmentation -- verified from the code, and one doc error

**Random crop is already on, for free.** `use_albumentations_transforms` defaults to `True`
(`configs/model/gr00t_n1d7.py:62`) and the TRAIN transform is:

```python
LetterBoxPad()                        # pads to square -- PRESERVES aspect ratio
A.SmallestMaxSize(max_size)
FractionalRandomCrop(crop_fraction)   # <-- random crop, train only; eval uses a CENTER crop
A.SmallestMaxSize(max_size)
```

Two consequences. It is genuine train-time augmentation, since eval takes the centre crop. And
`LetterBoxPad` means **Isaac preserves aspect ratio** where LeRobot's torch path squashed our
424x240 wrists to square -- Isaac's preprocessing is simply better for these cameras.

**DOC ERROR.** `FinetuneConfig.color_jitter_params` says *"If None, applying the default color
jitter augmentation from the pretrained model."* The code says otherwise:

```python
if color_jitter_params is not None:
    train_transform_list.append(A.ColorJitter(...))
```

`None` means **no jitter at all**. It has to be passed explicitly. Believing the docstring would
have shipped a run with no colour augmentation while thinking it had some.

```
--color-jitter-params '{"brightness":0.4,"contrast":0.4,"saturation":0.4,"hue":0.1}'
```

Those are NVIDIA's own values from that docstring's example.

**`random_rotation_angle`: OFF** (operator decision, 2026-08-25). The cameras are rigidly mounted,
so large rotations teach a variation that does not exist; `A.Rotate` also fills the corners, adding
a black-border artifact. Rotation is also the one geometric axis that has already cost a debugging
cycle here. The augmentation that attacks the actual failure -- memorising a pose instead of
looking -- is the random crop, and that is already on.

## 7. The command

Everything below is verified: the flags exist in `FinetuneConfig` (which is what `tyro.cli` parses),
and both datasets load in Isaac's own `LeRobotEpisodeLoader`.

```bash
# smoke test FIRST -- 1 step, proves config+data+forward+loss+backward
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <DATA>/screwing35_follower_train \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_config.py \
  --num-gpus 1 --output-dir /tmp/smoke \
  --max-steps 1 --global-batch-size 1 --dataloader-num-workers 0

# the real run
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <DATA>/screwing35_follower_train \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_config.py \
  --num-gpus $NUM_GPUS \
  --output-dir <OUT>/groot35_follower \
  --max-steps 10000 \
  --global-batch-size 64 \
  --dataloader-num-workers 8 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --state-dropout-prob 0.2 \
  --color-jitter-params '{"brightness":0.4,"contrast":0.4,"saturation":0.4,"hue":0.1}' \
  --save-steps 500 \
  --save-total-limit 12 \
  --save-only-model \
  --use-wandb --wandb-project screwing_hil
```

Not passed, deliberately: `--random-rotation-angle` (off, see 6), and anything eval-related
(impossible, see 3b). `tune_llm` / `tune_visual` / `tune_projector` / `tune_diffusion_model` are
left at their defaults because those defaults already match what we want.

**Step count.** 55,701 train frames / 64 global = 870 steps per epoch, so NVIDIA's default 10,000
is ~11.5 epochs. The previous run's eval bottomed near step 1,600 -- but that was with the broken
labels, so it is not a reliable guide any more. `--save-steps 500` covers the early region either
way and the choice is made post-hoc.

---

## 8. Judge it on the right number

Run on **val episode 1**, which is byte-identical to original episode 32:

| | baseline from 014400 |
|---|---|
| action MAE | 0.0611 rad (3.50 deg) |
| **Cartesian error at the driver grasp** | **54.0 mm** |
| **systematic fraction** | **~1.00** |
| natural spread between demonstrations | 27 mm |

Success is the **systematic fraction collapsing**, not MAE dropping.

---

## 9. What I could not verify

- whether an Isaac-trained checkpoint loads in `aiworker_deploy/groot_policy.py`, which is LeRobot
- the per-dimension loss decomposition (needs a forward pass)
- that the fix improves the MODEL. Everything so far shows the labels are now correct and
  self-consistent; only the retrain and `drift_probe.py` can show the policy is better.
