# Dataset audit + the 80k run

Date 2026-08-27. VERIFIED = measured here. INFERRED = reasoned. UNKNOWN = not established.

---

## 1. What the audit found

All 35 episodes, recovered to their **original** ids by sha1 of `observation.state`
(the relabel rewrote `action`, so action hashes cannot be used for this).

### Clean — no action needed

| Check | Result |
|---|---|
| **Grasp retries** | **0 of 35 episodes.** Every grasp closes exactly once and never reopens. No misgrabs in the demos. |
| Gripper semantics | HIGH = closed. Confirmed on all 5 subtasks (0.087→0.992 bolt, 0.204→0.830 driver). |
| Discontinuities | **0** frames jump >15° in one 33 ms tick. |
| Subtask durations | sd 0.8–1.4 s on 7–18 s subtasks. Worst outlier z=+3.20 (ep22, driver grasp, 15.4 s vs 12.8 mean). Tight. |
| Action frame `k=5` | **35 of 35 episodes independently pick k=5**; pooled minimum exactly at 5; 0.0% penalty vs optimum. See §2. |
| Train/val split | disjoint, **val = original episodes 31–34**. 55,701 + 6,962 = 62,663 = the full set. |
| Video↔parquet | 0 disagreements. |

### Fixed in this session

**The 22-dim modality was feeding the model garbage.** `modality.json` declared
`head [16,18)`, `lift [18,19)`, `odometry [19,22)` alongside the two arms, and the
normalisers for those dims are degenerate:

| dim | q99−q01 | normalised state range | |
|---|---|---|---|
| `head_joint1` | 0.0 | **DIV/0** | zero range |
| `head_joint2` | 0.0 | **DIV/0** | zero range |
| `lift_joint` | 1.0e-5 | ±3.0 | out of range |
| `linear_x` | 1.4e-3 | **±23** | sensor noise, amplified |
| `linear_y` | 2.5e-4 | **±32** | sensor noise, amplified |
| `angular_z` | 2.5e-3 | **±25** | sensor noise, amplified |

On the action side `linear_x/y/angular_z` have `q01 = q99 = 0, std = 0` — the model
was asked to regress three constants through a zero-width normaliser.

**6 of 22 state inputs (27%) were noise or undefined.** The 16 controlled dims are
all healthy (normalised −1.4…+1.9). Deployment is 16D anyway, so those six bought
nothing.

This is verbatim the divergence mode openpi's own troubleshooting table describes:
> "Certain dimensions that are rarely used can end up with very small q01, q99, or
> std values, leading to huge states and actions after normalization."

**Action taken:** `modality.json` in both splits reduced to `arm_left [0,8)` +
`arm_right [8,16)`. Originals backed up to `modality.json.bak-20260827-122154`.

**`meta/split_provenance.json` was missing** from both splits, so the gate could not
verify the split and refused. Regenerated from content hashes; both splits now carry
the original→new mapping.

**`verify_dataset.py` gained check 10 (NORMALISER)**, which refuses any *declared*
dim whose q99−q01 or std is ≤ 0. Verified by regression: run against the old 22-dim
`modality.json` it reports **5 problems and exits 1**; against the new 16-dim one it
reports 0 and exits 0. The gate now passes end-to-end.

### Found, NOT fixed — decide before the run

**(a) Dead time at subtask boundaries — 4.72% of frames.**

Sustained stalls (working arm AND its gripper < 1°/s for ≥ 0.5 s):

| subtask | stalled | in first 25% | gripper idle too |
|---|---|---|---|
| Grab the orange bolt | 0.00% | — | — |
| Place the bolt in the hole | 9.13% | 1121 of 1315 | 68.4% |
| **Grab the driver.** | **9.71%** | **1291 of 1307 (98.8%)** | **95.6%** |
| Screw in the bolt | 1.73% | 192 of 335 | 83.0% |
| Go back to home | 17.14% | (1161 in *last* 25%) | 97.8% |

This is the ROBOTIS teleop slow-start, reproducing at **every** subtask boundary in
**all 35 episodes** — roughly 1.2 s of motionless arm each time. Note there are
**zero** stalls in the last 25% of either grasp subtask, so the grasping itself is
clean; the dead time is pure handover latency.

**Why cutting is not free.** Deleting a dead run creates a discontinuity equal to the
pose drift across it:

| jump cap | frames recovered | runs | worst induced jump |
|---|---|---|---|
| 0.25° | 1,173 (1.87%) | 42 | 0.250° |
| 0.50° | 1,643 (2.62%) | 60 | 0.500° |
| 1.00° | 2,236 (3.57%) | 80 | 1.000° |
| uncapped | 2,956 (4.72%) | 99 | **6.167°** |

Typical per-frame motion in this dataset is **0.054°**, so even the 0.25° cap injects
a ~5-frame teleport. Cutting also changes episode lengths, which forces video
re-encoding — the exact operation that previously produced a 240-frame
(8-second) misalignment on episode 1.

**Recommendation: do not cut for this run.** The right fix is a *sampling weight*
(exclude dead frames as chunk start points, keep them as chunk content) — that has
zero discontinuity risk. Whether Isaac-GR00T's loader supports per-frame weights is
**UNKNOWN**; I do not have the GR00T source on this machine to check. Failing that,
fix it at the recorder.

**(b) Stale camera frames — 0.50% of camera-frames.**

`meta/frame_reuse.parquet` records duplicated (dropped-and-repeated) camera frames.
Restricted to the three cameras we actually train on:

| camera | stale | rate |
|---|---|---|
| `cam_left_head` | 669 | 1.07% |
| `cam_left_wrist` | 136 | 0.22% |
| `cam_right_wrist` | 129 | 0.21% |

**83.4% occur while the arm is moving > 2°/s** — frozen image, moving arm, which is
wrong supervision for visual servoing. Rate is low enough not to block; not fixable
without re-recording. (The table also lists `cam_right_head` at 1186, but that camera
was dropped during canonicalisation and is not in the training set.)

**(c) Wrist camera framing — the one I would fix before anything else.**

Measured over all 374 frames of the driver grasp in held-out ep32:

| phase of reach | workspace pixels | ceiling/wall | workspace centroid |
|---|---|---|---|
| 0–17% | 22.1% | 77.9% | 0.63 |
| 50–67% | 22.1% | 77.9% | 0.68 |
| **83–100%** | **11.5%** | **88.5%** | **0.76** |

The wrist camera tilts up as the arm extends: useful content halves and migrates into
the bottom quarter of the frame, at exactly the moment precision matters. No training
setting fixes this. **INFERRED** (not proven) that this is the better explanation for
"approaches correctly, stalls short" than the seven mechanisms already eliminated.

---

## 2. The k=5 relabel was correct

Derived from the **original** dataset where `action` is still the leader, on the first
25% of each episode (before freezes desynchronise leader from follower):

```
k= 0  0.02415 rad (1.384 deg)
k= 3  0.02039 rad (1.168 deg)
k= 5  0.01896 rad (1.086 deg)   <== minimum, and what we used
k= 7  0.02105 rad (1.206 deg)
k=10  0.02475 rad (1.418 deg)
```

**All 35 of 35 episodes independently pick k=5.** 5 frames at 30 fps = 167 ms, the
follower's tracking lag. Penalty for using 5 instead of the optimum: **0.0%**.

---

## 3. Augmentation — what was actually applied

Read from `checkpoint-30000`, not from the launch command:

| setting | value | note |
|---|---|---|
| `use_albumentations` | `True` | random crop in train, centre crop at eval |
| `crop_fraction` | 0.95 | |
| `image_crop_size` | `[230, 230]` | |
| `color_jitter_params` | `{brightness 0.3, contrast 0.4, hue 0.08, saturation 0.5}` | **the base model's values.** Our CLI passed `{0.4, 0.4, 0.4, 0.1}` and it was silently overridden — `setup.py:161` reads `model_config`, and `start_from_checkpoint` loads the base `config.json` over the CLI. |
| `random_rotation_angle` | **0** | no rotation augmentation at all |
| `state_gaussian_noise_std` | **0.0** | no proprioceptive noise at all |

(`noise_beta_alpha/beta` and `noise_s` are flow-matching schedule parameters, not
image augmentation.)

**For the 80k run.** With 31 episodes I would keep augmentation conservative and
change one thing only: add mild state noise. The measured reason is in §4 — a
zero-motion policy already scores 98.1% against absolute labels, and state noise is
the standard counter to the model leaning on proprioception instead of vision.

- `--state-gaussian-noise-std 0.01` (≈0.57°, well under the 1.09° label lag)
- leave `random_rotation_angle 0` — the wrist framing problem is spatial; rotating it
  makes the useful 11% harder to find, not easier
- keep crop at 0.95
- to actually control colour jitter you must patch a **local copy** of the base
  model's `config.json`; passing it on the CLI does nothing

---

## 4. The absolute-label measurement (context for any retrain)

Over horizon 16, across all train episodes:

- median dynamic-range compression, absolute → delta-from-state: **29.1%**
- mean |action[t+k] − state[t]| = 0.0151 rad against a mean q99−q01 output range of
  0.7957 rad
- **a policy that outputs `state[t]` verbatim — commanding zero motion — is already
  98.1% "correct" in normalised absolute action space**

The entire motion signal lives in the last ~2% of the output range. GR00T's
`use_relative_action: True` was a no-op because all five action keys are `ABSOLUTE`.
This is **INFERRED** to interact badly with the 4.72% dead-time frames (§1a), which
are explicit supervision for zero motion. Not proven.

---

## 5. The 80k run

### What changes vs the 30k run

1. **16 action/state dims, not 22.** This resizes the action head, so
   `checkpoint-30000` **cannot be resumed** — this is a fresh fine-tune from the
   GR00T N1.7 base. That is what we want.
2. **The schedule must target 80k.** The 30k run used a cosine set for **40,000**
   steps; at step 30,000 the LR was already annealed to **1.61e-5, 16% of peak**.
   That is why "still improving at 30k" is not evidence of convergence — and equally,
   80k is a *new schedule*, not an extension.
3. `--state-gaussian-noise-std 0.01`.

### Arithmetic

| | 30k run | 80k run |
|---|---|---|
| steps × batch / frames | 30,000 × 64 / 55,701 = **34.5 passes** | 80,000 × 64 / 55,701 = **91.9 passes** |
| warmup (ratio 0.05) | 2,000 | 4,000 |
| checkpoints at `save_steps 5000` | 6 | **16 × 6.44 GiB = 103 GiB** |

Watch the disk. `--save-only-model` keeps each checkpoint at 6.44 GiB instead of ~12.

### Validation — the gap that must be closed

Isaac's `DatasetFactory.build()` asserts `eval_strategy == "no"`; **there is no
validation loop**. 92 passes over 31 episodes with no held-out curve is how you
overfit invisibly.

`screwing35_follower_val` (4 episodes, original 31–34, v2.1, now with 16-dim
modality) exists precisely for this. Run `grasp_bias.py` / `isaac_drift_probe.py` on
every saved checkpoint, **both arms, all four episodes** — not just ep32 and not just
the right wrist, which is how I previously reported a 2.5× improvement on the arm
that does not do the failing grasp.

The one number that answers your professor: the right arm went 28.4 → 26.5 → 21.4 mm
across 10k/20k/30k **while the LR was being annealed**, and the systematic fraction
0.87 → 0.77 → 0.48. That genuinely does not look converged. The left arm was flat
(21.3 → 22.4) at ~1.16× the demonstrations' own spread — a label-information limit
that more steps cannot move.

**So: expect 80k to help the right arm and not the left.** If the left arm is the one
that has to grab the bolt first, step count is not its bottleneck.

---

## 6. Gate status

```
$ python3 verify_dataset.py --root .../screwing35_follower_train \
      --val-root .../screwing35_follower_val --lead 5 --isaac
1. ACTION FRAME       mean |action[t] - state[t+5]| = 0.000 deg
2. FREEZE FRAMES      0 of 55701 (0.00%)
3. LABEL CONSISTENCY  worst across-episode SD = 0.12 deg
4. TASKS.JSONL (v2.1) real strings
5. CAMERA SHAPES      376x672, 424x240, 424x240
6. MODALITY.JSON      arm_left [0,8) ok   arm_right [8,16) ok   declared 16 of 22
10. NORMALISER        0 declared dim(s) with a degenerate range
8. DATASET VERSION    v2.1
7. TRAIN/VAL SPLIT    train 31 ep, val 4 ep, overlap 0
9. VIDEO ALIGNMENT    0 disagreements
All checks passed. Safe to train.       EXIT=0
```

---

## 7. What I did NOT verify

- what Isaac-GR00T does at runtime with a degenerate normaliser (clamp / NaN /
  pass-through) — the GR00T source is not on this machine
- whether its loader supports per-frame sampling weights (needed for §1a)
- the 10k/20k/30k numbers in §5 are read from `HOW_WE_TRAINED_GR00T_N17.md`, which
  recorded them when they were measured; I could not re-run them here because GR00T
  is not importable on this machine
- that 80k actually helps — no run has been launched
- whether the wrist-camera framing (§1c) is the cause of the stall; it is the best
  candidate, not a proven one
