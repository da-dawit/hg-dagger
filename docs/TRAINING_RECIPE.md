# GR00T N1.7-3B retrain recipe — verified against NVIDIA and LeRobot

**Verified 2026-08-25 from the installed code, not from memory.**
Sources: `Isaac-GR00T/gr00t/configs/finetune_config.py` (NVIDIA's official `FinetuneConfig`),
`Isaac-GR00T/getting_started/finetune_new_embodiment.md` (our exact case), and
`lerobot/policies/groot/{configuration_groot,groot_n1_7,processor_groot}.py`.

---

## 0. The data fix comes first. Nothing else matters until it lands.

The last model reached **within 1.1× of the noise floor in its own labels** (4.79° error against a
4.45° floor). It did not undertrain — it learned contradictory labels almost perfectly. No training
setting can beat that floor.

```bash
python3 hil_dagger/eval/verify_dataset.py --root <new dataset>
```

It exits non-zero and names the problem. **Do not launch training until it passes.**
Run it bare -- piping into `tail`/`grep` returns the pipe's exit code, not the script's.

### The freeze is PER ARM

The operator freezes ONE arm to hold a steady camera view of the work site while the other arm
keeps demonstrating -- right arm parked over the screw, left arm placing the bolt into the hole.

| | share of frames |
|---|---|
| neither arm frozen | 91.20% |
| LEFT frozen, right still demonstrating | 3.52% |
| RIGHT frozen, left still demonstrating | 5.15% |
| both frozen | 0.13% |
| **at least one frozen** | **8.80%** |

**Do not discard these frames.** 8.67% of the dataset has one arm frozen *while the other performs
the most precise part of the task*; dropping them would bias the data against exactly those moments.

`action[t] = observation.state[t+k]` handles this per JOINT with no per-arm logic -- on those
frames it asks 0.030 deg of the frozen arm and preserves 0.486 deg on the working one. The leader
label asks 4.015 deg of the FROZEN arm, more than the 3.591 deg it asks of the working one.

---

## 0b. Apply the relabel to the 16 CONTROLLED dims only

Verified: `observation.state[t+k]` is right for the arms and grippers and **wrong for the other
six**. The dataset is 22-D; the robot is commanded on 16 (`spec.MODEL_JOINTS`).

| dims | action range today | state range | decision |
|---|---|---|---|
| 14 arm joints | — | — | **relabel** to `state[t+5]` |
| `gripper_{l,r}_joint1` | 2.299 / 2.235 | 1.049 / 1.278 | **relabel** — they carry the clutch offset too (10.8 deg by subtask 2) |
| `head_joint1`, `head_joint2` | 0.079 / 0.121 | 0.0031 / 0.0031 | **leave as-is** |
| `lift_joint` | 0.0105 | **0.00003** | **leave as-is** |
| `linear_x`, `linear_y`, `angular_z` | **0.000** | 0.032 / 0.008 / 0.054 | **leave as-is — must stay exactly 0** |

Two reasons, both measured:

- **The base velocities are a clean 0 in the action and noise in the state.** The base never moves.
  Relabelling them turns three exactly-zero dims into three noise channels the model must predict.
- **`lift_joint` has a state range of 3e-5 rad.** Min-max normalisation stretches whatever range
  exists to [-1, 1], so a near-constant dim becomes amplified encoder noise. The action's range is
  350x larger and therefore less degenerate. Same argument for the head joints.

None of the six is commanded on the robot, so leaving them alone changes nothing about control.

---

## 1. Already correct — do NOT change these

Every one matches NVIDIA's `FinetuneConfig` defaults exactly. Changing them is a regression.

| setting | ours | NVIDIA | note |
|---|---|---|---|
| `tune_llm` | False | False | backbone frozen |
| `tune_visual` | False | False | vision encoder frozen |
| `tune_projector` | True | True | |
| `tune_diffusion_model` | True | True | |
| `tune_vlln` | True | True | LeRobot-specific, its default |
| `tune_top_llm_layers` | 0 | 0 | an earlier run used 4 (an N1.5 value) — that was the bug, now fixed |
| `optimizer_lr` | 1e-4 | 1e-4 | |
| `weight_decay` | 1e-5 | 1e-5 | |
| `warmup_ratio` | 0.05 | 0.05 | |
| `embodiment_tag` | new_embodiment | NEW_EMBODIMENT | correct for a custom robot |
| `chunk_size` | 40 | `N1_7_NATIVE_ACTION_HORIZON` = 40 | |
| `state_dropout_prob` | **0.2** | 0.2 | applied by the model automatically; the `0.0` in `policy_preprocessor.json` is a different field |
| image geometry | 256 target / 230 crop | LeRobot N1.7 defaults | |

**`state_dropout_prob` deserves a note**: it randomly zeroes the proprioceptive features during
training, forcing the policy to use vision instead of replaying a memorised trajectory. It is on,
at NVIDIA's value. Do not turn it off.

---

## 2. Change these three

### a. Augmentation — ON (was completely off)

Verified: `dataset.image_transforms.enable` was `false`, **and** the torch image path has no random
crop (`processor_groot.py:2113` samples a random crop position only under `use_albumentations`,
which is not a LeRobot config field). So the last run had **zero image augmentation of any kind**.

NVIDIA's `color_jitter_params` docstring: *"If None, applying the default color jitter augmentation
from the pretrained model."* — their recipe augments by default. Ours did not.

```
--dataset.image_transforms.enable=true
```

Gives ColorJitter (brightness/contrast/saturation/hue), SharpnessJitter, and RandomAffine (±5°,
0.05 translate). With 35 episodes of one fixed scene this is the counter-pressure against
memorising a pose instead of looking at it.

### b. Let the schedule finish

Last run set `steps=26110` and stopped at 14400, leaving the cosine LR at **~43% of peak** — the
model was never annealed. Set `steps` to what you will actually run.

### c. Evaluate often enough to pick the checkpoint by measurement

`eval_split=0.1` holds out the **last** `ceil(n*0.1)` episodes per task (`factory.py:164`,
episode-level, no frame leakage). Keep `eval_steps == save_freq` so every checkpoint has a number.

---

## 3. The command

Batch 24/GPU is verified to fit on 80 GB H100s. On 2 GPUs that is **48 global** — between NVIDIA's
doc example (32) and their config default (64).

```bash
# 55,701 train frames / 48 global = 1,160 steps per epoch
# NVIDIA's default 10,000 steps at batch 64 is ~11.5 epochs; the equivalent here is ~13,000

accelerate launch --num_processes=$NUM_GPUS -m lerobot_train \
  --dataset.repo_id=dawity/screwing35_followerlabel \
  --dataset.root="$DATA_ROOT" \
  --dataset.eval_split=0.1 \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.tune_top_llm_layers=0 \
  --policy.chunk_size=40 \
  --policy.use_bf16=true \
  --policy.device=cuda \
  --steps=13000 \
  --batch_size=24 \
  --save_freq=1000 \
  --eval_steps=1000 \
  --num_workers=8 \
  --seed=1000 \
  --log_freq=10 \
  --env_eval_freq=-1 \
  --save_checkpoint=true \
  --wandb.enable=true --wandb.project=screwing_hil --wandb.disable_artifact=true \
  --job_name=groot35_fixed --output_dir="$ROOT/runs/groot35_fixed"
```

Every flag above was checked to exist in `TrainPipelineConfig` / `DatasetConfig` / `GrootConfig`.

**Do not** pass `--policy.tune_top_llm_layers=4`. It is an N1.5 value, adds ~201 M trainable
parameters, and was used by an earlier run.

---

## 4. Two things that made the last run slow

Both already fixed in our scripts, but they matter if anything is rebuilt:

- `gradient_as_bucket_view=True` in the DDP wrapper — saved 6.5 GB and was the difference between
  OOM and running at all.
- `find_unused_parameters=False` — +8%.

---

## 5. Confirm the fix worked — do not settle for "MAE went down"

MAE is unsigned and hides a directional offset. Re-run the exact harness on the same held-out
episode:

```bash
python3 hil_dagger/eval/rollout_mae.py  --ckpt <new>/pretrained_model --episode 32
python3 hil_dagger/eval/drift_probe.py  --ckpt <new>/pretrained_model
```

The number that matters is the **systematic Cartesian offset at the driver grab**, currently
**~48 mm** with `|bias|/|error| ≈ 1.00`. Success is that collapsing toward the natural
episode-to-episode spread (27 mm), not merely a lower MAE.

Baselines from checkpoint 014400 on held-out episode 32:

| | value |
|---|---|
| overall action MAE | 0.0611 rad (3.50°) |
| Cartesian error, driver grab | 54.0 mm mean |
| systematic fraction | ~1.00 (entirely directional) |
| worst joint | `arm_r_joint6`, 10.95° |
