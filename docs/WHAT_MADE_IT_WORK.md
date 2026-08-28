# The run that worked — `16d_30k/checkpoint-30000`

Date: 2026-08-28. **Verified on the physical robot**: grasps succeed. Screwing-down does not yet.
This is the strongest evidence level we have reached — every prior claim in this project was
offline, teacher-forced, or open-loop.

Checkpoint: `dawity/groot_screwing35` → `16d_30k/checkpoint-30000` (public).
Integrity: 1031/1031 tensors readable, `global_step 30000`, LR annealed to 3.04e-13.

---

## 1. What actually changed, ranked by likely contribution

Five things changed at once, so **none of this is individually attributable**. Ranked by the
size of the defect each corrected, not by confidence.

### 1.1 Six of 22 state dimensions were noise (the largest defect)

`modality.json` declared `head [16,18)`, `lift [18,19)`, `odometry [19,22)` alongside the arms.
Measured on `screwing35_follower_train`:

| dim | q99−q01 | normalised state range |
|---|---|---|
| `head_joint1` / `head_joint2` | **0.0** | **divide by zero** |
| `lift_joint` | 1.0e-5 | ±3.0 |
| `linear_x` | 1.4e-3 | **±23** |
| `linear_y` | 2.5e-4 | **±32** |
| `angular_z` | 2.5e-3 | **±25** |

On the action side `linear_x/y/angular_z` had `q01 = q99 = std = 0` — the model was asked to
regress three constants through a zero-width normaliser. **27% of the proprioceptive input was
noise or undefined**, while deployment (`spec_sg2.MODEL_JOINTS`) only ever used 16 dims. The 16
controlled dims normalise cleanly (−1.4 … +1.9).

Now: `arm_left [0,8)` + `arm_right [8,16)`. `verify_dataset.py` gained **check 10 (NORMALISER)**,
which refuses any *declared* dim with `q99−q01 ≤ 0` or `std ≤ 0`. Regression-verified: 5 failures
against the old modality, 0 against the new.

### 1.2 Per-camera crop replacing whole-frame letterboxing

`LetterBoxPad` padded every view to square: **44% of every model input was black**. The three
cameras are not alike, and one rule was wrong for both:

- **head** — static context view. **50% of its motion energy** (59% during the driver grasp) lies
  in the side bands a centred square crop deletes. The cheapest box retaining 90% of motion still
  costs 36% padding. **Verdict: do not crop. Keep the letterbox, keep the field of view.**
- **wrists** — moving close-ups whose upper half is ceiling (**88% of the frame at the grasp
  instant**, measured over all 374 frames of the driver grasp). **Verdict: crop bottom-anchored to
  square.** These are the 0.73–1.08 mm/px cameras where grasp precision lives.

```
cam_left_head    672x376 (full frame) -> 672x672   44.0% padding   full FoV kept
cam_left_wrist   240x240 rows 184-424 -> 240x240    0.0% padding   ceiling gone
cam_right_wrist  240x240 rows 184-424 -> 240x240    0.0% padding   ceiling gone

mean black share    44.0% -> 14.7%
workspace content   18.6% -> 40.3%   (2.17x, entirely from the wrists)
```

**Key implementation fact:** each crop must be square, but the squares need **not** match.
`_get_vlm_inputs` stacks views with `torch.stack`, which needs identical dims — but the trailing
`SmallestMaxSize(256)` normalises every square to 256×256 first. Verified: 672² + 240² + 240²
stacks without error. This is what makes per-camera treatment possible at all.

`roi_hybrid.json` expresses "letterbox the head" as an ROI covering the full frame — cropping to
it is a no-op and padding it to square *is* `LetterBoxPad`. One code path, two behaviours.

**The ROI is stored as fractions keyed by aspect ratio**, so it survives a camera resolution
change. It does **not** survive an aspect-ratio change — swapping a camera for a different aspect
invalidates this config and requires retraining.

### 1.3 Instructions now name the working arm

Per NVIDIA's bimanual guidance ("provided the training data is distinct or annotated"). Byte-exact,
and verified **sha1-identical** across the training source (`tasks.jsonl`) and the deploy sources
(`subtasks.parquet`, `tasks.parquet` — which read *different files*):

```
0  Grab the orange bolt with the left arm                     (left,  85% of travel)
1  Place the orange bolt into the hole with the left arm       (left,  69%)
2  Grab the driver with the right arm                          (right, 90%)
3  Screw in the bolt by pushing down with the right arm        (right, 76%)
4  Return both arms to home                                    (both,  48/52)
```

Arm assignment confirmed by measured joint travel, not assumed.

### 1.4 Two-fold data exposure

68.9 epochs vs 34.5 (batch 128 × 30k steps vs batch 64 × 30k). Final training loss **0.0015** vs
0.0056.

### 1.5 A properly annealed schedule

The previous run used a cosine targeting **40,000** steps and stopped at 30,000, leaving the LR at
**1.61e-5 — 16% of peak**. That is why it looked "still improving": it was mid-anneal, not
converged. This run targeted 30,000 and finished at **3.04e-13**.

---

## 2. Training configuration, verbatim

```
model      GR00T N1.7-3B from base (NOT resumed - 22->16 resizes the action head,
           and --save-only-model means there was never an optimizer state)
hardware   2x H100 80GB SXM3, NV18 NVLink, 208 cores
launcher   torchrun --nproc_per_node=2 --master_port=29500
           (bare python falls back to DataParallel and dies immediately)

--base-model-path        base_model_patched          # use_percentiles=False, crop_fraction=1.0
--dataset-path           screwing35_follower_train
--embodiment-tag         NEW_EMBODIMENT
--modality-config-path   examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_16d_config.py
--num-gpus               2
--max-steps              30000
--global-batch-size      128      (64/GPU)
--dataloader-num-workers 24
--learning-rate          1e-4     cosine
--weight-decay           1e-5
--warmup-ratio           0.05
--state-dropout-prob     0.2
--save-steps             2500
--save-total-limit       30
--save-only-model
--use-wandb --wandb-project screwing_hil

env  GR00T_SQUARE_CROP=1
     GR00T_ROI_JSON=eval/roi_hybrid.json
```

**Frozen / trained** (all four are `FinetuneConfig` defaults — the 30k run's command did not pass
them either, and `tyro` rejects `--flag false` syntax):

```
tune_llm             False        tune_projector        True
tune_visual          False        tune_diffusion_model  True
tune_top_llm_layers  0            tune_vlln             True
```

**Not CLI-settable** — these live in the model config and must be patched into a copy of the base
checkpoint before launch:

```
use_percentiles  False   # q01/q99 clipped 0.5-35% of each dim's range on our data
crop_fraction    1.0     # random crop OFF
```

**Does not exist at all:** `state_gaussian_noise_std` appears nowhere in the Isaac-GR00T source.
Do not pass it.

**Throughput:** 1.13 it/s without flash-attn, **1.53 it/s with it** (`pip install flash-attn
--no-build-isolation`, after `sed -i '/flash-attn==/d' pyproject.toml` to get the install through).
30,000 steps in ~5.4 h.

**Environment:** Python **3.10** exactly (3.11+ fails on GR00T's dataclasses:
`non-default argument 'diffusion_model_cfg' follows default argument`), torch 2.7.1+cu126,
transformers 4.57.3, torchcodec **0.4.0** (gr00t pins it; 0.5 warns).

---

## 3. Dataset

```
train   31 episodes  55,701 frames   30.9 min @ 30 Hz
val      4 episodes   6,962 frames   (original episodes 31-34)
dims    16 controlled (arm_l 7 + gripper_l, arm_r 7 + gripper_r)
label   action[t] = observation.state[t+5]
horizon 16 steps (0.53 s)
```

**`k=5` verified optimal**: all **35 of 35** episodes independently minimise
`|leader_action[t] - follower_state[t+k]|` at k=5 (167 ms — the follower's tracking lag).
Penalty vs the optimum: 0.0%.

**Audit results:** 0 grasp retries in 35 episodes (every grasp closes once, never reopens),
0 discontinuities >15°/frame, subtask durations sd 0.8–1.4 s, train/val disjoint by content hash.

---

## 4. Measured results

Held-out episodes 31–34, open-loop, identical protocol, each model evaluated with the
preprocessing it was trained on:

| | previous (22-dim, letterbox, 34.5 ep) | **this (16-dim, hybrid, 68.9 ep)** | change |
|---|---|---|---|
| Grab bolt (left) | 16.3 mm | **14.4 mm** | −12% |
| Grab driver (right) | 22.7 mm | **13.3 mm** | **−41%** |
| systematic fraction (left) | 0.56 | **0.38** | |
| systematic fraction (right) | 0.87 | **0.40** | |

The **systematic-fraction collapse on the right arm (0.87 → 0.40)** is the qualitatively important
part: previously almost all the error was a fixed, uncorrectable bias — the signature of stalling
in the same place every attempt. Now most of it is variance. The two arms are also balanced
(14.4 / 13.3) where the right was previously 40% worse.

**Attention** (`eval/attn_map.py`): both wrists at **0.0% padding attention** across all 293
frames of the episode; head ~51% (letterboxed by design).

---

## 5. Honest limits

**Screwing-down does not work yet.** Grasps succeed; the final subtask does not.

**The offline numbers are open-loop and teacher-forced.** Every predicted chunk starts from the
human's recorded observation, so they cannot show error accumulation. They were never sufficient
evidence — the robot run is.

**Five changes at once.** Nothing here is individually attributable. If you need attribution, the
cheap experiment is a 5k-step A/B on one variable at a time.

**Attention maps are weak evidence.** Normalised entropy ≈0.98 — near-uniform in aggregate.
Individual heads are sharp (one block reached 0.0012), but averaging 8 blocks × 4 denoising steps
flattens it. The error numbers are the claim; the heatmaps are illustration.

**Known and unfixed: 4.8% of training frames are dead time** at subtask starts (~1.2 s of
motionless arm, 8.9% of "Grab the driver" frames, all in the first quarter). Deliberately left in —
the per-subtask instructions allow nudging past it. Cutting it would inject up to 6.17° teleports
and require video re-encoding.

**Untaken measurement:** `--trace` on one failed screwing attempt. It is the only thing that
separates "the policy commands short" from "the arm does not arrive", and it has never been run.
That is the obvious next step for the screwing subtask.
