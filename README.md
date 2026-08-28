# Fine-tuning GR00T N1.7-3B on a ROBOTIS AI Worker

Documentation and tooling from fine-tuning NVIDIA's GR00T N1.7-3B for a bimanual
bolt-insertion-and-screwing task on the ROBOTIS AI Worker (FFW-SG2), using Isaac-GR00T.

Everything here is written from **values read out of the produced checkpoints**, not from the
commands we believed we ran. That distinction mattered: several settings we thought we had set
were silently overridden.

---

## Start here

| doc | what it covers |
|---|---|
| **[WHAT_MADE_IT_WORK.md](docs/WHAT_MADE_IT_WORK.md)** | The run that first succeeded on hardware, and the five changes that got it there |
| [HOW_WE_TRAINED_GR00T_N17.md](docs/HOW_WE_TRAINED_GR00T_N17.md) | Every argument and effective parameter of an earlier full run |
| [DATASET_AUDIT_AND_80K.md](docs/DATASET_AUDIT_AND_80K.md) | Auditing demonstrations: dead time, grasp retries, degenerate normalisers |
| [DEPLOY_16D_30K.md](docs/DEPLOY_16D_30K.md) | Deploying a checkpoint, and the two failures that are silent |
| [ISAAC_TRAINING_RECIPE.md](docs/ISAAC_TRAINING_RECIPE.md) | Isaac-GR00T specifics: sharding, augmentation, the config-precedence trap |
| [PI05_OPENPI_PLAN.md](docs/PI05_OPENPI_PLAN.md) | A parallel evaluation of π₀.₅ / openpi for the same robot |

---

## Findings that cost us time, so they might save you some

**The base checkpoint's `config.json` overrides your CLI arguments.** Any field that is not
declared in `FinetuneConfig` is read from the base model instead. We lost runs to this three
times — `use_percentiles`, `crop_fraction`, and `color_jitter_params` were all silently ignored
from the command line. The workaround is to patch a local copy of the base model.

**`FinetuneConfig` has exactly 31 fields.** Anything else you pass is either a parse error or
quietly inert. `state_gaussian_noise_std` appears **nowhere** in the Isaac-GR00T source despite
being present in saved configs. Validate flags against the dataclass, not against the docs.

**Multi-GPU requires `torchrun`.** With bare `python`, HF Trainer falls back to DataParallel and
dies with `module must have its parameters and buffers on device cuda:0`.

**Degenerate normalisers are silent and destructive.** Our `modality.json` declared 22 dimensions;
six of them had `q99 − q01 ≈ 0`, so normalisation amplified pure sensor noise to ±23/±32 or divided
by zero. **27% of the proprioceptive input was garbage for an entire training run.**
[`verify_dataset.py`](scripts/verify_dataset.py) now refuses any declared dimension whose
normaliser is degenerate.

**Video conversion silently misaligns frames.** Using `-ss` before `-i` with `-c copy` prepends
frames; Isaac indexes video by frame number, so images and actions disagree with no error. One
episode was off by 240 frames (8 seconds). Check 9 of the gate detects it.

**Cameras are not interchangeable.** Whole-frame letterboxing put 44% black in every model input.
But the right fix differs per camera: our static head camera loses 50% of its motion energy to a
crop and must keep its full field of view, while the wrist cameras are 88% ceiling at the grasp
instant and should be cropped hard. Crops must be square, but the squares need not match — the
trailing resize normalises before views are stacked.

---

## Layout

```
docs/     the write-ups above
scripts/  dataset preparation, gating, and evaluation
configs/  per-camera crop config, the crop patch, 16-dim modality config, LaTeX results table
math/     proofs and error analyses for design decisions
```

Notable scripts:

- **`verify_dataset.py`** — ten checks that must pass before spending GPU time: action frame,
  sustained freezes, label consistency, task strings, camera shapes, modality slices, train/val
  overlap by original episode id, dataset version, video alignment, normaliser degeneracy.
- **`attn_map.py`** — action→image cross-attention over an episode, with a selectivity readout.
  Use the entropy number: ours was ≈0.98 (near-uniform in aggregate), which makes such maps much
  weaker evidence than they look.
- **`checkpoint_sweep.py`** — held-out Cartesian error per checkpoint via forward kinematics.
  Isaac has no validation loop (`DatasetFactory.build()` asserts `eval_strategy == "no"`), so
  something like this is the only defence against invisible overfitting.
- **`split_dagger.py`** — splits an HG-DAgger recording into human / autonomous / dropped sets,
  since Isaac has no per-frame loss weighting.

---

## Caveats

Numbers here are from **one robot, one task, 31 demonstrations**. Offline metrics are open-loop
and teacher-forced: they cannot show error accumulation, which is what closed-loop control
actually suffers from. The only claim we make about real-world behaviour is the one we observed on
the physical robot, and it is stated as such.

Not affiliated with NVIDIA. GR00T N1.7 is NVIDIA's; Isaac-GR00T is theirs and ROBOTIS's fork.
