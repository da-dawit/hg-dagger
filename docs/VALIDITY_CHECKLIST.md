# What has to be true for the training to be VALID

Not "what is left to do" — **what must be demonstrated**, and the command that demonstrates it.
Every item below corresponds to a failure that completes training without any error.

Rule for this list: an item is not done because someone believes it is done. It is done when the
stated check prints the stated result.

---

## A. The labels mean what we think

| # | must be true | check | now |
|---|---|---|---|
| A1 | `action` is in the **follower's** frame | gate check 1 -> `<= 1.0 deg` | **FAIL 3.42** |
| A2 | a sustained freeze commands no motion | gate check 2 -> `~0%` | **FAIL 1.48%** |
| A3 | the same phase carries the same offset in every episode | gate check 3 -> `<= 1.0 deg` | **FAIL 4.04** |
| A4 | language is in the parquet INDEX, and row order == `task_index` | gate check 4 | pass |
| A5 | wrists are 424x240 portrait | gate check 5 | pass |

```bash
python3 hil_dagger/eval/verify_dataset.py --root <train> --isaac --val-root <val>
```

**A3 is the one that decides whether the model can succeed at all.** The last run reached within
1.1x of the floor set by A3 (4.79 deg error against a 4.45 deg floor). It was not undertrained; the
labels contradicted each other. If A3 still fails, a retrain buys nothing.

---

## B. Isaac can actually read it

| # | must be true | check | now |
|---|---|---|---|
| B1 | `meta/modality.json` present in **both** datasets | gate check 6 | **FAIL missing** |
| B2 | its slices hold the columns they claim | gate check 6 (per-key) | blocked on B1 |
| B3 | train and val share **no episodes** | gate check 7 | no val set yet |
| B4 | val episodes are exactly `[31,32,33,34]` | gate check 7 | no val set yet |
| B5 | a v3.0 dataset loads, or is converted to v2.x | smoke run below | **UNKNOWN** |

B2 matters because the slices are **positional**. `arm_left [0,8)` is not checked against column
names anywhere in Isaac — a shifted dataset feeds the wrong joints into the wrong modality and
trains happily.

B4 is not pedantry: every baseline in this repo was measured on **episode 32**. A different val set
means the retrain cannot be compared to the 54 mm / `|bias|/|error| = 1.00` numbers.

---

## C. The run is not blind

| # | must be true | check |
|---|---|---|
| C1 | validation actually runs | an eval loss appears in the log |
| C2 | checkpoints are chosen by measurement, not by step count | eval curve exists |

`TrainingConfig.eval_strategy` defaults to `"no"`, and `launch_finetune.py` parses only
`FinetuneConfig` via `tyro.cli(...)` — so `eval_strategy`, `eval_steps` and `val_dataset_path`
**cannot be set from the command line.** Either patch the launcher or write our own.

Training blind is how the last run reached epoch 12 while its best checkpoint was at step ~1600.

---

## D. The smoke test passes before renting

Per SKILL.md §16 — cheap checks before an expensive one. ROBOTIS's own smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base_model_path nvidia/GR00T-N1.7-3B \
  --dataset_path <2-episode train set> \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_config.py \
  --num_gpus 1 --output_dir /tmp/smoke \
  --max_steps 1 --global_batch_size 1 --dataloader_num_workers 0
```

Must demonstrate, in order: config loads, dataset loads (**answers B5**), one batch passes,
forward runs, loss computes, backward runs. A 2-episode conversion also gives a measured
per-episode transcode cost to multiply by 35.

---

## E. Deployment matches what was trained

| # | must be true | why |
|---|---|---|
| E1 | horizon 16 -> `spec.EXECUTE_STEPS = 10` | at 25, `horizon - execute_steps <= 0` and the seam cross-fade **silently disappears** |
| E2 | `--seam-blend 6` | must be `<= horizon - execute_steps` |
| E3 | `dagger_aggregate.CHUNK = 16` | a training sample is one horizon | 

E1/E2 are **not** to be applied until the 16-step checkpoint exists — they would break the
currently deployed 40-step model. `control_math.check_horizon()` prints the required values on the
first re-plan of every run, so a mismatch announces itself.

E3 is already done.

---

## F. The result is judged on the right number

`rollout_mae.py` and `drift_probe.py` on **held-out episode 32**, against these baselines from
checkpoint 014400:

| | baseline |
|---|---|
| action MAE | 0.0611 rad (3.50 deg) |
| **Cartesian error at the driver grab** | **54.0 mm** |
| **systematic fraction** `\|bias\|/\|error\|` | **~1.00** |
| natural spread between demonstrations | 27 mm |

**Success is the systematic fraction collapsing, not MAE dropping.** MAE is unsigned and hides a
directional offset — a policy that drifts 50 mm consistently to one side can score the same as one
that is randomly off by 50 mm, and only the first is the failure you can see on the robot.

---

## Current status

Ready: the diagnosis and its evidence, the gate (checks 1-8), the `k=5` validation, the Isaac
recipe, `check_horizon()`, `CHUNK=16`, and the eval harness with baselines recorded.

Blocking, all on the robot side and none of it needing a GPU:

1. **A1/A2/A3** — the converter relabel, `action[t] = observation.state[t+5]` for the 16 controlled
   dims only. Nothing else can be judged until this lands.
2. **B1** — copy `modality.json` into `meta/`.
3. **B3/B4** — split the conversion into two physical datasets, val = episodes 31-34.
4. **B5** — the 2-episode smoke run.
5. **C1** — patch validation into the launcher.

Rent after 1-5, not before.
