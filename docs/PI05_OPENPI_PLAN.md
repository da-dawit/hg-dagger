# Training π₀.₅ on the AI Worker — research findings

Date: 2026-08-27. Everything below is labelled **VERIFIED** (read from installed
code / cloned repo / our own dataset), **INFERRED** (reasoned from verified
facts), or **UNKNOWN**.

Sources actually read:
- `github.com/Physical-Intelligence/openpi` @ `215abfb` (2026-08-25) — cloned, read
- `github.com/IliaLarchenko/lehome_solution` @ `ca73828` (2026-08-16) — cloned, read
- `github.com/huggingface/lerobot` — our checkout `8a74e0a` + fetched `origin/main` `bf31dd7`
- our datasets under `/home/robotis/robot_aiworker/datasets/`

---

## 0. The headline

**VERIFIED:** Ilia's solution is **openpi (JAX/Flax)**, not LeRobot's PyTorch
π₀.₅. His `pyproject.toml` pins `jax[cuda12]==0.5.3`, `flax==0.10.2`,
`orbax-checkpoint==0.11.13`; `openpi` is a path dependency. LeRobot appears only
as a hardware/dataset library.

**VERIFIED:** openpi pins LeRobot at commit `0cf8648` (2025-05-28), whose
`CODEBASE_VERSION = "v2.1"`.

**That is the single luckiest fact in this document:** we already converted
`screwing35` to **v2.1** for Isaac. `screwing35_follower_train` (31 ep / 55,701
frames) and `screwing35_follower_val` (4 ep / 6,962 frames) are both v2.1 on
disk right now. **No reconversion is needed.**

---

## 1. What Ilia actually built

| Fact | Value |
|---|---|
| Competition | LeHome Challenge 2026 (ICRA), bimanual garment folding |
| Result | **1st of 62** in sim round, **2nd** in the real-world final |
| Robot | LeRobot SO-ARM101 ×2, 6 DoF/arm = **12 dims** |
| Cameras | overhead RealSense D435 + 2 wrist USB (3 total) |
| Base model | π₀.₅ via openpi, **heavily modified** (`PiModified`) |
| Hardware | H200 for training, RTX PRO 6000 for rollouts, 500+ GB disk |
| Tech report | arXiv:2606.27163 |
| Checkpoints | `IliaLarchenko/lehome_sim`, `IliaLarchenko/lehome_real` |

**VERIFIED:** `PiModified` is *not* stock π₀.₅. He added: FAST-token auxiliary
loss, knowledge insulation, KV-cache cross-layer transform, correlated flow
noise, advantage embedding (RL conditioning) with CFG, and six auxiliary heads
(success / checkpoint / garment-type / completion / time-to-completion /
keypoint-distance) plus two world-model heads. `use_success_head`,
`use_advantage_embedding`, `cfg_scale` etc. are all his own fields in
`pi_modified_config.py`.

**Do not try to reproduce `PiModified`.** It is competition code, coupled to
garment folding and to his Isaac Sim RL flywheel. The value is in the *ideas*,
listed next.

---

## 2. Nine things in his repo that map directly onto our failures

### 2.1 Delta joint actions — the biggest one

**VERIFIED**, `configs/train_real_bc.yaml`: `use_delta_joint_actions: true`.

**VERIFIED**, openpi `config.py:LeRobotAlohaDataConfig`: the field defaults to
`True` for ALOHA, applied as

```python
delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
data_transforms = data_transforms.push(
    inputs=[_transforms.DeltaActions(delta_action_mask)],
    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
)
```

**VERIFIED**, `openpi/transforms.py:DeltaActions.__call__`:
```python
actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
```
Every action in the chunk has the **current state at t=0** subtracted. It is
delta-from-current-state, *not* frame-to-frame delta. Gripper dims stay
absolute.

**VERIFIED**, openpi `LeRobotLiberoDataConfig` comment:
> "pi0 models are trained on delta actions (relative to the first state in each
> action chunk). IF your data has `absolute` actions (e.g. target joint angles)
> you can uncomment the following line to convert the actions to delta actions."

**Our data is absolute target joint angles.** We trained GR00T on absolute
(`use_relative_action: True` was a no-op because all five action keys are
`ABSOLUTE`). π₀/π₀.₅ base models were pre-trained in delta space.

For our 16 controlled dims the mask is `make_bool_mask(7, -1, 7, -1)` —
7 arm_l delta, gripper_l absolute, 7 arm_r delta, gripper_r absolute.
**VERIFIED** against `make_bool_mask`'s docstring and implementation.

**INFERRED:** this is *plausibly* relevant to the stall. In absolute space a
model that regresses toward the training mean undershoots a reach; in delta
space it predicts a displacement, which does not have that bias. **But I have
not tested this, and there is a counter-argument:** near the goal the true
deltas are small, so a delta model can also collapse to ~0 and stall. Do not
treat this as a diagnosis.

### 2.2 Real-Time Chunking by inpainting — replaces our `seam_blend` hack

**VERIFIED**, `pi_modified.py:1445-1765`: inference passes `inpaint_targets`,
`inpaint_mask`, `inpaint_lengths` into the flow-matching sampler. The first
`actions_to_keep` steps of the new chunk are *pinned* to the previously
committed actions during denoising, with precomputed correction matrices, up to
`time_threshold_inpaint=0.3`.

His deployed setting (`README`, real robot):
```
--actions_to_execute 5 --actions_to_keep 1 --execute_in_n_steps 5 --num_steps 10
```
i.e. horizon 30, execute 5, overlap-pin 1, 10 denoising steps.

**VERIFIED**, `eval_wrapper.py`: raises if `actions_to_execute + actions_to_keep
> action_horizon`.

Our GR00T equivalent was `seam_blend`, a linear cross-fade we bolted on, which
silently did nothing whenever `execute_steps >= chunk_len`. Inpainting is the
principled version — the model itself produces a chunk that already starts
where the robot is going.

### 2.3 `initial_stuck` — the ROBOTIS slow-start problem, solved in the sampler

**VERIFIED**, `configs/train_real_bc.yaml`:
```yaml
initial_stuck:
  movement_threshold: 3.0
  position_std_factor: 3.0
  min_position_range: 5.0
  min_leading_frames: 5
  mid_weight: 0.2
  trailing_weight: 0.3
  apply_to_buckets: [primary_real, home_real]
```
> "Suppress sampling of frames stuck near the initial home position."

This is exactly the thing you asked me to remove from our recordings months
ago. He does not delete the frames — he **downweights** them in the sampler
(0.2 / 0.3). That is strictly better than cutting, because the episode stays
temporally intact for chunk reads.

### 2.4 DAgger per-frame weighting with a pre-intervention ramp

**VERIFIED**, `configs/train_real_bc.yaml` + `data_loader.py:1009-1018`:
```yaml
dagger_sampling:
  human_weight: 2.0
  auto_weight: 0.3
  pre_intervention_min_weight: 0.0
  pre_intervention_window_seconds: 5.0
```
- human frames → weight 2.0
- autonomous frames far from an intervention → 0.3
- autonomous frames inside the 5 s **before** a human takeover → linear ramp
  from 0.3 down to **0.0**

Docstring, verbatim:
> "teach the policy on the auto frames it produced, EXCEPT the ones immediately
> leading up to a human takeover — those represent failure modes the policy
> itself should not be encouraged to repeat."

This is the correct HG-DAgger weighting and we did not have it.

### 2.5 `task_is_policy` — a per-frame column, exactly as I concluded for Cyclo

**VERIFIED**, `record_real_dagger.py:489-493`:
```python
features["task_is_policy"] = {"dtype": "float32", "shape": (1,), "names": ["is_policy"]}
```
written per frame as `1.0` (autonomous) / `0.0` (human).

This confirms the earlier finding that intervention labels must be a **per-frame
dataset feature**, not Cyclo segments (segments are the subtask mechanism and
would corrupt our five language labels).

### 2.6 Leader mirroring during autonomous — fixes our desync

**VERIFIED**, `record_real_dagger.py`: while the policy drives, he calls
`_mirror_to_leaders(teleop, obs)` every frame, so the leader arms track the
follower. On `SPACE` he runs `_freeze_position(robot, teleop, mode_switch_delay)`,
toggles leader torque, then clears `action_queue` and calls
`policy_client.reset_session()`.

This means the human can grab the leaders at any moment with **no jump** — which
is precisely the leader/follower desynchronisation that corrupted our
`screwing35` action labels (0.3 mm → 157 mm drift by the driver grasp).

### 2.7 `speed_factor` — time-axis resampling per data source

**VERIFIED**, `data_loader.py:DataSourceConfig`:
> ">1.0 means source is slower than primary — grabs more native frames and
> compresses into action_horizon (larger per-step deltas). Cubic-resamples onto
> the 30-action grid."

He uses `dagger_human_speed_factor: 2.0`, `dagger_auto_speed_factor: 2.0`,
sim replays at `0.65`. A direct lever on "the policy moves too timidly".

### 2.8 Augmentation far stronger than ours

**VERIFIED**, `train_real_bc.yaml:train_aug` — brightness 0.25, contrast 0.30,
saturation 0.30, hue 0.03, **per-camera independent colour**, per-channel gain
0.10, gamma ±0.10, Gaussian blur σ0.8 @ p0.30, additive noise 0.012, crop 0.92,
rotate ±3°, zoom ±0.05, translate ±10 px, cutout p0.30 area 0.10, **camera
dropout p0.05**, **state dropout p0.05**, state noise 0.02, calibration scale
noise 0.02.

His stated reason for state noise: *"pushes the model to rely on images over
potentially-unreliable proprioception."*

Ours was colour jitter only — and (as documented in `HOW_WE_TRAINED_GR00T_N17.md`)
the CLI values were silently overridden by the base checkpoint anyway.

### 2.9 A completion/progress head

**VERIFIED:** `completion_loss_weight: 0.1`, target = `frame_position /
episode_length`. An auxiliary signal for "how far through the task am I".
Cheap, and directly aimed at a policy that does not know it has not finished
reaching.

---

## 3. openpi: the actual recipe for a new embodiment

**VERIFIED** from `README.md` and `src/openpi/training/config.py`.

### 3.1 Three transform layers

| Layer | Applies to | Purpose |
|---|---|---|
| `repack_transforms` | **dataset only** | rename our LeRobot keys → the keys our inference client will send |
| `data_transforms` | dataset **and** inference | `<Robot>Inputs` / `<Robot>Outputs` — pack into the model's slots |
| `model_transforms` | both | tokenise prompt/state. **Do not touch.** |

Optional 4th: `DeltaActions` / `AbsoluteActions` pushed onto `data_transforms`.

### 3.2 Hard constraints

**VERIFIED**, `models/model.py:40-42` and `pi0_config.py:71-78` — exactly
**three image slots**, hardcoded:
```
base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
```
Unused slots are filled with `np.zeros_like(base_image)` and masked
`np.False_` (for π₀/π₀.₅; π₀-FAST masks `True`).

**Our rig has exactly three cameras.** Clean 1:1 mapping:
- `observation.images.rgb.cam_left_head` → `base_0_rgb`
- `observation.images.rgb.cam_left_wrist` → `left_wrist_0_rgb`
- `observation.images.rgb.cam_right_wrist` → `right_wrist_0_rgb`

**VERIFIED**, `pi0_config.py:25-41`:
- `action_dim = 32` (state and actions are padded to 32; π₀.₅ was pre-trained at 32)
- `action_horizon = 50` default — **free to change**; `pi05_libero` uses 10,
  `pi05_droid_finetune` uses 16
- `max_token_len = 200` when `pi05=True` (48 otherwise)
- `discrete_state_input` defaults to `pi05` — π₀.₅ discretises the state into tokens

**VERIFIED**, `config.py:187`: `use_quantile_norm = model_config.model_type !=
ModelType.PI0` → **π₀.₅ uses quantile (q01/q99) normalisation**, not mean/std.

### 3.3 The Inputs class we need to write

Template is `src/openpi/policies/libero_policy.py`. Ours, `aiworker_policy.py`:

```python
CTRL = list(range(16))          #arm_l 0-6, gripper_l 7, arm_r 8-14, gripper_r 15

@dataclasses.dataclass(frozen=True)
class AIWorkerInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb":        _parse_image(data["observation/head"]),
                "left_wrist_0_rgb":  _parse_image(data["observation/left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,     #all three are real
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

@dataclasses.dataclass(frozen=True)
class AIWorkerOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :16])}
```

**Open decision:** feed 16 dims (controlled only) or all 22. Our 22 includes
`head_joint1/2`, `lift_joint`, `linear_x/y`, `angular_z` — six dims that are
constant or near-constant. **VERIFIED** from the openpi troubleshooting table:
> "Certain dimensions that are rarely used can end up with very small q01, q99,
> or std values, leading to huge states and actions after normalization."

**Recommendation: slice to the 16 controlled dims.** Feeding six dead dims into
quantile normalisation is exactly the documented divergence mode.

### 3.4 The TrainConfig

Modelled on `pi05_libero` (**VERIFIED**, `config.py:744`):

```python
TrainConfig(
    name="pi05_aiworker",
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16),
    data=LeRobotAIWorkerDataConfig(
        repo_id="screwing35_follower_train",
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
    ),
    batch_size=64,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    action_sequence_keys=("action",),     #NOT the ("actions",) default
)
```

**`action_sequence_keys` is a real trap.** **VERIFIED**: `DataConfig` defaults to
`("actions",)`, but LeRobot datasets store the column as `action` (singular).
`LeRobotAlohaDataConfig` overrides it to `("action",)`. We must too.

**VERIFIED**, `data_loader.py:141-146`: the loader builds
`delta_timestamps={key: [t/fps for t in range(action_horizon)]}` — the chunk is
read at the dataset's native fps (30 Hz for us). Horizon 16 = 0.53 s.

**VERIFIED**, `data_loader.py:148-149`: `prompt_from_task=True` wraps the dataset
in `PromptFromLeRobotTask(dataset_meta.tasks)`, which raises if `task_index` is
missing and looks the string up in the tasks dict.

**Checked our data:** `screwing35_follower_train/meta/tasks.jsonl` contains five
real strings ("Grab the orange bolt", "Place the orange bolt into the hole",
"Grab the driver.", "Screw in the bolt by pushing down.", "Go back to home after
done."). **The `tasks.parquet` index trap does not apply here** — this v2.1
dataset uses `tasks.jsonl`. Verified by reading the file.

### 3.5 Pointing openpi at our local dataset

**VERIFIED**, `data_loader.py:139-143`: openpi constructs
`LeRobotDatasetMetadata(repo_id)` and `LeRobotDataset(repo_id, ...)` with **no
`root=` argument**.

**VERIFIED**, pinned LeRobot `lerobot_dataset.py:90`:
```python
self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
```
and `constants.py:39`: `HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path))`.

**VERIFIED**, `LeRobotDatasetMetadata.__init__`: it calls `load_metadata()` first
and only falls back to `pull_from_repo()` on `FileNotFoundError`.

So:
```bash
export HF_LEROBOT_HOME=/home/robotis/robot_aiworker/datasets
# repo_id="screwing35_follower_train" resolves to
#   /home/robotis/robot_aiworker/datasets/screwing35_follower_train
```
**No HF upload needed, no network at train time.**

### 3.6 Commands

```bash
# norm stats first — training errors out without them
uv run scripts/compute_norm_stats.py --config-name pi05_aiworker

# JAX (recommended — this is the mainline path)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_aiworker --exp-name=screw_v1 --overwrite

# PyTorch alternative
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_aiworker --exp_name screw_v1

# serve
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_aiworker --policy.dir=checkpoints/pi05_aiworker/screw_v1/30000
```

**I have not run any of these.** They are transcribed from the openpi README and
the config dataclass field names, which I did read.

### 3.7 Hardware

**VERIFIED**, openpi README requirements table:

| Mode | Memory | Example |
|---|---|---|
| Inference | > 8 GB | RTX 4090 |
| Fine-tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-tuning (full) | **> 70 GB** | A100 80GB / H100 |

Multi-GPU via `fsdp_devices=<n>`. **VERIFIED**: "the current training script does
not yet support multi-node training."

LoRA path (**VERIFIED**, `config.py:693-697`):
```python
freeze_filter=pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
).get_freeze_filter(),
ema_decay=None,   #"Turn off EMA for LoRA finetuning."
```
Variants **VERIFIED** in `gemma.py:55`: `dummy | gemma_300m | gemma_300m_lora |
gemma_2b | gemma_2b_lora`.

The 2×A100-80GB box we used for GR00T is sufficient for full fine-tuning.

### 3.8 PyTorch path — extra care

**VERIFIED**, README: PyTorch π₀/π₀.₅ is validated on LIBERO, but **no** π₀-FAST,
**no** mixed precision, **no** FSDP, **no** LoRA, **no** EMA. It needs
`transformers 4.53.2` plus a patch that **overwrites files in the transformers
package**, with this warning verbatim:
> "With the default uv link mode (hardlink), this will permanently affect the
> transformers library in your uv cache … could even propagate to other
> projects."

Given that, and that Ilia's proven path is JAX: **use JAX.**

---

## 4. What openpi does NOT give us

**VERIFIED:** `grep -niE "val_|valid|eval_|test_split" scripts/train.py` returns
only `_load_weights_and_validate` (a weight-shape check). **There is no
validation loop.** Same blind spot as Isaac.

Mitigation: we already built the offline probes — `isaac_drift_probe.py`,
`grasp_bias.py`, `closed_loop_probe.py`, and the held-out
`screwing35_follower_val` (4 episodes, v2.1, byte-identical provenance recorded
in `meta/split_provenance.json`). They need a new adapter for openpi's
`policy.infer()` API but the measurement logic carries over unchanged.

---

## 5. LeRobot moved too (you were right)

Our checkout `8a74e0a` (2026-07-06, v0.6.1) is **179 commits behind**
`origin/main` `bf31dd7` (2026-08-25). π₀.₅ changed substantially:

| Commit | Date | Change |
|---|---|---|
| `15b19df` | 2026-08-21 | **feat(pi05): add optional training-time RTC** |
| `8ae2354` | 2026-08-12 | **feat(pi05): add optional visual and proprioceptive MEM** |
| `5361e02` | 2026-07-20 | refactor(pi05): use shared VLA components |
| `6ac9536` | 2026-07-30 | fix(rollout): reject incompatible RTC policies |

Diff: `modeling_pi05.py` +649/-, new `memory.py` (250 lines),
`configuration_pi05.py` +45.

New config fields (**VERIFIED** from the diff):
```python
use_visual_memory: bool = False
use_proprioceptive_memory: bool = False
memory_frames: int = 6
memory_stride: int = 30                  #frames; 6 obs 1 s apart at 30 fps
memory_temporal_attention_every: int = 4
rtc_training_max_delay: int = 0          #0 disables trained RTC
```
`tokenizer_max_length` was **removed**.

Both features are relevant to us — MEM (arXiv 2603.03596) gives short-horizon
observation history, and training-time RTC is the trained version of the
inpainting trick Ilia hand-rolled. **They exist only in LeRobot's PyTorch port,
not in openpi.** That is the one real argument against the openpi path.
`memory_stride=30` is correct for our 30 fps data as-is.

---

## 6. Honest assessment: will this fix the stall?

You told me to stop agreeing and to push back. So:

**Switching GR00T → π₀.₅ does not, by itself, address what the evidence points
at.**

What we established about the GR00T failure:
- offline on held-out ep32, the **left** arm (which does the bolt grasp) was at
  **12.8 mm vs a demo spread of 11.0 mm — ratio 1.16**, i.e. essentially at the
  data's own noise floor
- 10k → 30k did not improve it (21.3 → 22.4 mm)
- seven mechanisms tested and rejected: time lag, correctable bias, camera
  rotation, control accumulation, lead time, chunk length, untrained steps
- the one thing never measured: **the robot's own images at close range**

A model that is near-optimal offline but fails online is the textbook signature
of **closed-loop distribution shift**, and a different architecture does not fix
distribution shift. **DAgger does.** So the parts of this plan most likely to
actually move the needle are, in order:

1. **HG-DAgger with the pre-intervention weight ramp** (§2.4) — trains on states
   the policy actually visits, and explicitly refuses to reinforce the frames
   that preceded a takeover
2. **Delta joint actions** (§2.1) — removes absolute-position mean-reversion, and
   matches how π₀.₅ base was pre-trained
3. **RTC/inpainting at inference** (§2.2) — removes the seam discontinuity our
   `seam_blend` never actually smoothed
4. **`initial_stuck` downweighting + stronger augmentation** (§2.3, §2.8)

π₀.₅ is a reasonable base to do all four on — it is better pre-trained than
GR00T for this, the openpi codebase is smaller and more legible than Isaac's, and
we have a proven reference implementation. But if we swap the architecture and
change nothing else, **I would not expect the stall to disappear**, and I am not
going to tell you otherwise.

**UNKNOWN and worth settling cheaply before spending money:** whether the robot
commands short or arrives short. That is one `--trace` run on one failed grasp.
It costs nothing and it decides whether this is a policy problem at all.

---

## 7. Risks and open questions

1. **Wrist cameras are portrait.** `cam_left_wrist` / `cam_right_wrist` are
   `[3, 424, 240]` — 424 tall × 240 wide. openpi's `resize_with_pad` letterboxes
   to 224×224, so a portrait frame becomes ~127×224 of content with ~43% of the
   canvas as black padding. The head cam `[3, 376, 672]` is landscape and
   letterboxes to ~224×125. **UNKNOWN:** how much this costs. π₀.₅ base was
   pre-trained on letterboxed images so it is not out-of-distribution, but we are
   throwing away resolution on the *only* cameras with the precision for this
   task (0.73–1.08 mm/px on the wrists vs 1.54 mm/px on the head).
2. **31 episodes / 55,701 frames is small** for a 3B VLA. Ilia had the organizer
   dataset plus his own teleop plus DAgger plus sim replays.
3. **Six dead dims** in our 22-dim vectors — slice to 16 (§3.3).
4. **No validation loop** (§4) — we must drive our own probes.
5. **The 16-dim slice must be consistent** between training transform, norm
   stats, and the deployed client. This is exactly the class of mismatch that
   produced the 157 mm label drift last time.
6. **`gs://` download** — openpi pulls `pi05_base` from Google Cloud Storage into
   `~/.cache/openpi` (override with `OPENPI_DATA_HOME`). Needs disk headroom on
   the vast.ai box; budget for it in the prep script.

---

## 8. What I have NOT done

- not installed openpi, not run `uv sync`
- not run `compute_norm_stats.py` or any training
- not downloaded `pi05_base`
- not written `aiworker_policy.py` or `LeRobotAIWorkerDataConfig`
- not verified our v2.1 dataset actually loads under openpi's pinned LeRobot
  (static file-layout check only)
- not read Ilia's tech report (arXiv:2606.27163) — code only
- not measured whether letterboxing the portrait wrist cams costs accuracy
