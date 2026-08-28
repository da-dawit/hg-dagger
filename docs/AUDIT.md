# Top-to-bottom audit

**2026-08-25.** Written to the discipline in `MUSTREAD/SKILL.md` and `~/.claude/CLAUDE.md`:
claims are labelled **Verified** (a command was run and its output inspected), **Inferred** (a
reasonable conclusion from verified evidence, not itself observed), or **Unknown**.

Everything under "Verified" was **re-run for this audit**, not copied from earlier in the session.

---

## Verified

### Artifacts

| dataset | version | episodes | frames |
|---|---|---|---|
| `screwing35_follower` | v3.0 | 35 | 62,663 |
| `screwing35_follower_train` | v2.1 | 31 | 55,701 |
| `screwing35_follower_val` | v2.1 | 4 | 6,962 |

Plus `*_v3.0` siblings holding the pre-conversion originals.

### The relabel, recomputed from scratch

Not read back from the gate — recomputed per episode:

- `action[t] == observation.state[t+5]` on the 16 controlled dims, tail-clamped: **True**
- `observation.state` untouched: **True**
- the 6 non-controlled dims bit-identical to the original: **True**
- driver-grasp label gap by forward kinematics: **99.2 mm -> 6.6 mm**

The residual 6.6 mm is real motion during 167 ms, which is what a lead target should contain.

### Gate: 9 checks, exit 0

Action frame 0.000 deg · sustained freezes 0.00% · label SD 0.12 deg (was 4.04) · language is
sentences not digits · wrists 424x240 · modality.json present with all 5 slices matching by column
name · train/val overlap 0 by original episode id · video/parquet alignment 0 mismatches.

### Isaac

Both datasets construct in Isaac's own `LeRobotEpisodeLoader`; `check_stats_validity` returns True
for both; 5 tasks each with real sentences. Modality config reports action horizon **16**.

### Video alignment — a corruption found and fixed

`convert_v3_to_v2.py` splits with `-ss` BEFORE `-i` and `-c copy`. That is input seeking: ffmpeg
cannot cut mid-GOP, so the output carries extra LEADING frames — measured **+11 to +240** across
the val episodes, exactly `video_frames - parquet_rows` every time.

Isaac indexes video by frame NUMBER (`get_frames_by_indices`) with no timestamp compensation, so
row *i* would have been paired with video frame *i*, which is really episode frame *i - offset*.
On episode 1 that is **240 frames — eight seconds**. Verified by pixel match: the true first frame
sits at video frame 240 (`mean|diff| = 0.00`) while frame 0 differs by 37.

Fixed by re-extracting with `-ss` AFTER `-i` (output seeking, frame-accurate). Verified: frame
counts equal parquet rows, and frame 0 is pixel-correct against LeRobot's decode of the v3.0 source
(residual 0.65–0.88 is x264 noise; a one-frame offset measures ~37).

### Control constants

`spec.EXECUTE_STEPS` is still **25**, deliberately — changing it would break the deployed 40-step
model. `dagger_aggregate.CHUNK` is **16**. Seam blending measured at (40,25,20)=15,
**(16,25,20)=0**, (16,10,6)=6, so `check_horizon()` will announce the mismatch when a 16-step
checkpoint is loaded.

### Tooling

`online_rl.py --self-test` 10 checks pass. `dagger_aggregate.py --self-test` 17 pass. Every touched
file parses. All 20 flags in the training command exist in `FinetuneConfig`, which is what
`tyro.cli` parses.

---

## Inferred — believed, not proven

- **The leader/follower label bug caused the drift.** The evidence is strong and consistent: the
  per-subtask offset ordering (1.2 / 41 / 97 / 87 / 100 mm) matches the operator's independent
  subjective ordering; the model landed within **1.1x** of the label noise floor; post-processing
  was exonerated (normalize/unnormalize stats bit-identical); the scene was shown not to have moved.
  **But causation is only demonstrated by the retrain.**
- **Horizon 16 is deliberate rather than copied from the SO100 tutorial.** It demands ~2.5x faster
  inference at the same control rate, which fits ROBOTIS shipping TensorRT alongside it. Not confirmed.
- **The 14.9% normalised variance in the six dead dims translates to loss share.** Variance is a
  proxy; the loss is MSE on flow-matching velocity. Not measured.

---

## Unknown

- Whether an Isaac-trained checkpoint loads in `aiworker_deploy/groot_policy.py`, which is LeRobot-based
- The smoke test — config, data, forward, loss, backward — needs a GPU and the 6.5 GB base model
- **Whether the model actually improves.** Everything so far shows the labels are now correct and
  self-consistent. That is a precondition, not a result.
- Whether the v2.1 conversion is lossless in respects I did not check. I verified action, state,
  task_index, frame counts and frame-0 alignment; I did not compare every video frame.

---

## What I got wrong this session, and corrected

Listed because a track record of confident errors is the reason to distrust the confident parts.

1. **"MuJoCo reports zero contacts, so the pose cannot hit the table."** The model contains no
   table, floor or ground plane at all. The check could not have detected table contact. Retracted.
2. **"Episode 32's driver is a 3.4-sigma outlier."** Wrong three ways: compared joint angles rather
   than Cartesian (a 7-DoF arm reaches one point many ways), used the leader instead of the
   follower, and averaged the whole subtask instead of the grasp instant. Measured properly, ep32
   sits **0.14 sd** from the mean. The operator pushed back and was right.
3. **"Checkpoint 002400 beats 014400."** From remembered wandb eval loss. Measured directly on
   held-out ep32: **3.50 deg vs 3.74** — 014400 is better. Advice reversed.
4. **Freeze frames, twice.** First 1.55% (whole-robot detector missed single-arm freezes), then
   8.80% (per-arm, but counting 1-2 frame blips), finally **1.48%** sustained. The middle figure
   was in a document for an hour.
5. **Gate check 6 flagged correct data** — demanded all 8 `arm_left` columns start with `arm_l`,
   but ROBOTIS's arm modality is 7 joints **plus that arm's gripper**.
6. **Gate check 7 reported a 4-episode overlap that did not exist** — compared renumbered indices
   instead of original ones.
7. **`check_horizon()` referenced `chunk` before assignment**, which would have crashed on the
   first re-plan. Caught before it ran on hardware.
8. **`check_horizon()`'s message was wrong** for `seam_blend=0`, reported by the self-test.
9. **The wrist-rotation equivalence test fed portrait input** where the live camera delivers
   landscape, so its printed shapes were reversed. The transform was right; the demonstration was not.
10. **Nearly reported `state_dropout_prob = 0.0` as a defect.** The model uses **0.2**, matching
    NVIDIA; the 0.0 is a different processor-level field.
11. **Said the relabel was "blocked on the robot side" for several exchanges.** It was not — it uses
    only columns already in the dataset and took seconds locally. The operator had to ask "so are
    you fixing this or what" before I checked.
12. **Recommended augmentation as the top fix** in the first training audit. Wrong priority: no
    augmentation setting beats labels that disagree by 19 degrees.
13. **Said "all 11 checks"** for the `online_rl` self-test. It is 10.

Pattern worth naming: most of these are **measuring the wrong quantity and reporting it
confidently** — joint space for a Cartesian question, unsigned MAE for a directional failure, a
collision model with no table. The checks that caught them were cheap; running them first would
have been cheaper than retracting.
