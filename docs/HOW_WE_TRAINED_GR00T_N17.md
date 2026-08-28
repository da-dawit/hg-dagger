# How we trained GR00T N1.7-3B — complete record

**Run:** `groot35_follower` · **Date:** 2026-08-25/26 · **Result:** `checkpoint-30000`
**Every value below is read from the produced checkpoint**, not from the command we believe we ran.
Where the two disagree, that is called out — it happened twice.

---

## 1. What the model is

| | |
|---|---|
| base | `nvidia/GR00T-N1.7-3B` (gated on HF) |
| VLM backbone | `nvidia/Cosmos-Reason2-2B` (gated), truncated at `select_layer: 16` |
| total params | **3,144,016,000** |
| trainable | **1,620,515,968 (51.54%)** — action head + projector + VLLN |
| DiT params | 1,091,722,240 |
| self-attn transformer | 201,433,088 |
| dtype | bfloat16, `backbone_trainable_params_fp32: True` |

Frozen: LLM and visual encoder. Trained: projector, diffusion (action) head, VLLN.

---

## 2. Data pipeline — every step

### 2.1 Source

35 teleop episodes, 62,663 frames, 30 fps, robot `ffw_sg2_rev1`, recorded through Cyclo
Intelligence → rosbag → `cyclo_data/converter/to_lerobot_v30.py` → LeRobot **v3.0**.

### 2.2 The relabel — the reason this run exists

`action` held the **leader's** absolute joint positions. The per-arm freeze permanently
desynchronises leader from follower, so the target drifted out of the frame we command. Measured on
episode 32 by forward kinematics: **0.3 mm before the first freeze, 157 mm by the driver grasp.**

```bash
python3 hil_dagger/eval/relabel_follower.py \
  --src datasets/screwing35_subtask \
  --dst datasets/screwing35_follower \
  --lead 5
```

- `action[t] = observation.state[t + 5]`, tail-clamped at each episode end
- **k = 5** is the measured leader→follower tracking lag (median 5, mean 5.0 over 35 episodes;
  per-arm left 5, right 4.3; identical in free motion and in the final extension)
- Applied to the **16 controlled dims only** (14 arm joints + 2 grippers)
- `head_joint1/2`, `lift_joint`, `linear_x/y`, `angular_z` left untouched — `state` is worse for all
  six (base velocities are a clean 0.000 in action and jitter in state; `lift_joint`'s state range
  is 3e-5 rad, which min-max would stretch to full scale)

Effect, Cartesian gap between the label and the robot's own frame:

| subtask | before | after |
|---|---|---|
| Grab bolt | 1.2 mm | 1.2 mm |
| Place in hole | 41.3 mm | 1.5 mm |
| Grab driver | 96.8 mm | 7.1 mm |
| Screw in | 87.4 mm | 6.7 mm |
| Home | 100.3 mm | 4.2 mm |

Label consistency across episodes: **4.04° SD → 0.12°**.

### 2.3 Split — physical, because Isaac has no `eval_split`

```bash
python3 hil_dagger/eval/split_train_val.py \
  --src datasets/screwing35_follower \
  --train-out datasets/screwing35_follower_train \
  --val-out   datasets/screwing35_follower_val \
  --val-episodes 31 32 33 34
```

- train **31 episodes / 55,701 frames**, val **4 / 6,962**
- val episodes are the same four LeRobot held out, so baselines stay comparable
- episodes renumbered from 0 in each output; mapping in `meta/split_provenance.json`
  (**val ep 1 == original ep 32**, verified byte-identical)

### 2.4 v3.0 → v2.1, because Isaac reads v2.x

Isaac's `lerobot_episode_loader.py` reads `episodes.jsonl` / `tasks.jsonl` — v2.x files.

Run ROBOTIS's own `scripts/lerobot_conversion/convert_v3_to_v2.py` inside
`robotis/lerobot-zenoh:1.3.2-amd64`, with a three-function shim (`load_info`, `write_info`,
`load_tasks`) because the script targets a LeRobot version neither our fork nor 0.5.2 provides.

**Then three fixes the converter does not do:**

1. **`modality.json` is dropped** — copy `examples/CYCLO/ffw_sg2_rev1/modality.json` into `meta/`
2. **`stats.json` lacks `q01`/`q99`** — regenerate with Isaac's own `gr00t.data.stats.generate_stats`
3. **Videos are misaligned by up to 240 frames** — see below

### 2.5 The video misalignment — silent and severe

`convert_v3_to_v2.py` splits the concatenated video with `-ss` **before** `-i` and `-c copy`. That
is input seeking: ffmpeg cannot cut mid-GOP, so the output carries extra **leading** frames —
measured **+11 to +240**, exactly `video_frames − parquet_rows` every time.

Isaac indexes video by **frame number** (`get_frames_by_indices`), no timestamp compensation. So row
*i* pairs with video frame *i*, which is really episode frame *i − offset*. On episode 1 that is
**240 frames = 8 seconds**: every image matched to an action from eight seconds later.

```bash
python3 hil_dagger/eval/fix_episode_videos.py --v21 <v2.1 dataset> --v30 <v3.0 sibling>
```

Re-extracts with `-ss` **after** `-i` (frame-accurate), re-encodes libx264 crf 18. Verified frame
counts equal parquet rows and frame 0 pixel-correct (residual 0.65–0.88 is x264 noise; a one-frame
offset measures ~37).

### 2.6 Gate before spending GPU time

```bash
python3 hil_dagger/eval/verify_dataset.py --root <train> --isaac --val-root <val>
```

Nine checks, all must pass: action frame, sustained freezes, label consistency, tasks index,
camera shapes, `modality.json` slices, train/val overlap by original id, dataset version, video
alignment.

---

## 3. Environment — every pin

The fork requires **exactly Python 3.10** and **transformers 4.57.3**. On transformers 5.x,
`PretrainedConfig` became a dataclass and the fork **cannot import at all**:

```
TypeError: non-default argument 'diffusion_model_cfg' follows default argument
           'attend_text_every_n_blocks'
```

| | |
|---|---|
| python | 3.10.20 (deadsnakes; image shipped 3.12) |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1 |
| transformers | **4.57.3** |
| deepspeed | 0.17.6 |
| numpy | 1.26.4 |
| albumentations | 1.4.18 |
| **flash-attn** | **NOT installed** — see below |
| repo | `ROBOTIS-GIT/Isaac-GR00T-n1.7` @ `e81d02b` |

**flash-attn was deliberately dropped.** `pip install -e .` fails on it — its build backend imports
torch, which pip's build isolation hides, and the failure aborts the entire install:

```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
```

It is optional: `qwen3_backbone.py:66` catches `ImportError` and falls back to **sdpa**. The
checkpoint records `use_flash_attention: True` but flash-attn was absent, so **sdpa ran**.

---

## 4. Hardware

2× A100-SXM4-80GB (vast.ai), 192 cores, 629 GB RAM, 666 GB disk, CUDA 12.8, driver 580.173.02.
~31.5 GB VRAM per GPU in use. **1.14 it/s.**

**Check FREE VRAM, not TOTAL.** A previous 4× A100-40GB box had 20 GB per card held by another
tenant, visible only as `Used: 20109 MiB` against "No running processes found". It survived a
reboot. `00_check_instance.sh` rejects that before uploading.

---

## 5. The launch command — verbatim

**Multi-GPU requires `torchrun`.** With bare `python`, HF Trainer falls back to DataParallel and
dies with `module must have its parameters and buffers on device cuda:0 but found one of them on
device: cpu`.

```bash
export HF_TOKEN=...            # nvidia/GR00T-N1.7-3B and Cosmos-Reason2-2B are gated
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY=...

torchrun --nproc_per_node=2 --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path ~/screwing_train/base_model_nopct \
  --dataset-path    ~/screwing_train/datasets/screwing35_follower_train \
  --embodiment-tag  NEW_EMBODIMENT \
  --modality-config-path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_config.py \
  --num-gpus 2 \
  --output-dir ~/screwing_train/runs/groot35_follower \
  --max-steps 40000 \
  --global-batch-size 64 \
  --dataloader-num-workers 8 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --warmup-ratio 0.05 \
  --state-dropout-prob 0.2 \
  --color-jitter-params brightness 0.4 contrast 0.4 saturation 0.4 hue 0.1 \
  --save-steps 1000 \
  --save-total-limit 15 \
  --save-only-model \
  --use-wandb --wandb-project screwing_hil
```

**`--color-jitter-params` takes space-separated pairs, not JSON.** tyro's signature is
`{None}|{[STR FLOAT [STR FLOAT ...]]}`; passing JSON is a parse error that prints help and exits.

**DeepSpeed ZeRO-2 engages automatically** when `num_gpus > 1` and `use_ddp` is false
(`experiment.py:197`). It is not a CLI flag. Config: `gr00t/configs/deepspeed/zero2_config.json`.

---

## 6. Two CLI arguments that were silently overridden

The base checkpoint's `config.json` is loaded via `start_from_checkpoint` and **wins over the CLI**.

### 6.1 `use_percentiles` — we worked around it

NVIDIA ships `use_percentiles: True` in **both** `config.json` and `processor_config.json`. It is
**not a `FinetuneConfig` field**, so `tyro` never exposes it, and editing
`gr00t/configs/model/gr00t_n1d7.py` has no effect.

Workaround: copy the base model, patch both JSONs, point `--base-model-path` at the copy.

```bash
cp -rL <hf snapshot>/. ~/screwing_train/base_model_nopct/
# set "use_percentiles": false in config.json AND processor_config.json
```

Confirmed in the checkpoint: **`use_percentiles: False`**.

Why it mattered: q01/q99 caps what the model can ever command. Measured on our data it removed
0.5–35% of each dimension's range (worst: `arm_r_joint2` 35.3%, `gripper_r_joint1` 32.2%) —
although the driver-grasp poses themselves were **not** clipped (0.00–0.29%).

### 6.2 `color_jitter_params` — we did NOT work around it

```
CLI passed      {brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1}
model config    {brightness 0.3, contrast 0.4, hue 0.08, saturation 0.5}   <- base model's
processor cfg   {brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1}    <- CLI's
```

The two saved configs disagree. `setup.py:161` reads `self.model_config.color_jitter_params`, so
**the base model's values were applied and the CLI value was inert.** Colour jitter did run — at
NVIDIA's pretrained settings, not ours.

Also note `FinetuneConfig.color_jitter_params`' docstring claims `None` applies a default jitter.
The code is `if color_jitter_params is not None:` — **`None` means no jitter at all.**

---

## 7. Every effective parameter (read from `checkpoint-30000`)

### 7.1 Tuning flags — all at NVIDIA defaults

| | value |
|---|---|
| `tune_llm` | False |
| `tune_visual` | False |
| `tune_projector` | True |
| `tune_diffusion_model` | True |
| `tune_vlln` | True |
| `tune_linear` | True |
| `tune_top_llm_layers` | **0** (an earlier run used 4 — an N1.5 value; that was a bug) |

### 7.2 Optimisation

| | value |
|---|---|
| optimizer | adamw_torch |
| learning rate | 1e-4, cosine |
| weight decay | 1e-5 |
| warmup ratio | 0.05 |
| global batch | 64 (32/GPU × 2) |
| max steps | 40,000 (reached 30,000 — the box died) |
| dataloader workers | 8 |
| precision | bf16, `model_params_fp32` for trainable |

### 7.3 Action / flow matching

| | value |
|---|---|
| `action_horizon` (model native) | 40 |
| **supervised horizon** | **16** (`delta_indices=range(16)` in the CYCLO modality config) |
| verified chunk returned at inference | **T = 16** |
| `max_action_dim` / `max_state_dim` | 132 (ours is 22, padded; loss is masked) |
| `num_inference_timesteps` | 4 |
| `noise_beta_alpha` / `beta` / `s` | 1.5 / 1.0 / 0.999 |
| `num_timestep_buckets` | 1000 |
| `rtc_ramp_rate` | 6.0 |

### 7.4 Normalisation

| | value |
|---|---|
| `use_percentiles` | **False** (patched; NVIDIA ships True) |
| `clip_outliers` | True |
| `use_mean_std` | False |
| `use_relative_action` | True — **a no-op**: only applies to keys marked `RELATIVE`, and our five are all `ABSOLUTE` |
| `apply_sincos_state_encoding` | False |
| `state_dropout_prob` | **0.2** — zeroes proprioception at random so vision must carry it |
| `state_gaussian_noise_std` | 0.0 |

### 7.5 Vision

| | value |
|---|---|
| `use_albumentations` | True → random-crop augmentation is ON in train, centre crop at eval |
| `image_target_size` | [256, 256] |
| `image_crop_size` | [230, 230] |
| `crop_fraction` | 0.95 |
| `shortest_image_edge` | 256 |
| `letter_box_transform` | False |
| `random_rotation_angle` | **0** (operator decision) |
| `random_history_crop` | True |
| cameras | `cam_left_head` 376×672, `cam_left_wrist` 424×240, `cam_right_wrist` 424×240 |

### 7.6 Architecture

`backbone_embedding_dim` 2048 · `hidden_size` 1024 · `max_seq_len` 1024 · `select_layer` 16 ·
`attn_dropout` 0.2 · `use_alternate_vl_dit` True · `use_vl_self_attention` True · `use_vlln` True ·
`add_pos_embed` True · `use_future_tokens` False · `reproject_vision` False ·
`max_num_embodiments` 32

### 7.7 Modality layout (`examples/CYCLO/ffw_sg2_rev1/modality.json`)

| modality | slice | columns |
|---|---|---|
| `arm_left` | [0,8) | `arm_l_joint1..7` + `gripper_l_joint1` |
| `arm_right` | [8,16) | `arm_r_joint1..7` + `gripper_r_joint1` |
| `head` | [16,18) | `head_joint1`, `head_joint2` |
| `lift` | [18,19) | `lift_joint` |
| `odometry` | [19,22) | `linear_x`, `linear_y`, `angular_z` |

Action rep: `ABSOLUTE`, type `NON_EEF`, format `DEFAULT`, for all five keys.
Language: `annotation.human.primitive_instruction` ← `task_index`.

### 7.8 The five instruction strings — byte-exact

```
0  Grab the orange bolt
1  Place the orange bolt into the hole
2  Grab the driver.
3  Screw in the bolt by pushing down.
4  Go back to home after done.
```

0 and 1 have **no** trailing period; 2, 3, 4 **do**. Conditioning is per frame.

---

## 8. What actually happened

| | |
|---|---|
| steps completed | 30,000 of 40,000 (instance destroyed) |
| loss | 1.2445 → min 0.0041 → 0.0056 at step 30,000 |
| LR at 30,000 | 1.61e-05 (cosine, mid-decay) |
| checkpoints saved | every 1,000, limit 15, model-only (**6.44 GiB each**) |
| pushed to HF | 10000, 20000, 30000 → `dawity/groot_screwing35` under `follower/` |
| lost | checkpoint-40000 |

`trainer_state.json` reports `epoch: 0.75`, which is **meaningless here** — Isaac's sharded dataset
defines an epoch by `num_shards_per_epoch` (default 1e5), not by a pass over the data. The real
figure is 30,000 × 64 / 55,701 ≈ **34 passes**.

**No validation loss exists.** `DatasetFactory.build()` asserts `eval_strategy == "no"` — Isaac's
sharded dataset cannot produce an eval set. NVIDIA's docs suggest `--eval-strategy steps`; against
this fork that is an assertion failure, not a missing flag. Checkpoints are therefore selected
**post-hoc**.

---

## 9. Results, on held-out episode 32

Open-loop, Cartesian error of the **right** wrist at the driver grasp:

| checkpoint | error | systematic (\|bias\|/\|err\|) |
|---|---|---|
| 014400 (old, broken labels) | 54.0 mm | 1.00 |
| 10000 | 28.4 mm | 0.87 |
| 20000 | 26.5 mm | 0.77 |
| **30000** | **21.4 mm** | **0.48** |

Natural spread between demonstrations: 27 mm.

**Caveats that matter:**

- Open-loop — every chunk starts from the *human's* observation, so this cannot show error
  accumulation, which is what closed-loop control actually suffers from.
- The **left** arm, which does the bolt grasp, is worse: 22.4 mm bias at the grasp instant, and it
  did **not** improve from 10k → 30k (21.3 → 22.4). Policy error ÷ demo spread = **1.16**, i.e. at
  the data's own noise floor.
- On the robot the policy approaches correctly then **stalls short** of the bolt. Seven mechanisms
  were tested offline (time lag, correctable bias, camera rotation, control accumulation, lead
  time, chunk length, untrained steps) and **none explains it**. The inference pipeline audits
  clean. Unresolved.

---

## 10. Reproducing this

```
hil_dagger/eval/
  relabel_follower.py      action[t] = observation.state[t+k], 16 controlled dims
  split_train_val.py       physical train/val split with provenance
  fix_episode_videos.py    frame-accurate re-extraction after v3->v2
  verify_dataset.py        the 9-check gate
  isaac_drift_probe.py     signed Cartesian drift, per subtask, --arm l|r
  grasp_bias.py            bias at the grasp instant, working arm per phase
  closed_loop_probe.py     state-feedback closed loop
  vastai/00..04            instance check, upload, prepare, smoke, train
```

**Order:** relabel → split → convert to v2.1 → `modality.json` → `generate_stats` → fix videos →
gate → smoke test → train.
