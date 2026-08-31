# Setup

Author: Dawit Chun
2026-08-14

## Scope

**Two methods only:** HG-DAgger against the gate-as-potential method proved in
`docs/HIL_THEORY.md`. Online RL, real robot, Meta Quest 3 for human intervention.

**The base policy is OpenPI pi-0.5, and it is held constant across both arms of the comparison.**
The experiment varies only what the human's signal is used for -- HG-DAgger clones the intervened
actions, ours uses the same actions plus the intervention labels as a potential on top of the servo
reward. Same policy, same demonstrations, same operator, same episode budget.

What the two-method scope does remove is RLinf's *second* model: there is no expert model, no beta
action-mixing between a model expert and the student. Human intervention alone decides, which is
what RLinf's real-world DAgger config already does (it deliberately omits `rollout.expert_model`).

Still ours rather than RLinf's:

* **Action space is 16D joint**, not tcp_rot6d 20D. Every demonstration, checkpoint, safety
  threshold and the FK-based reward already assume it. pi-0.5 is fine-tuned to that action head and
  its normalisation stats; the `realworld_dual_franka_*` configs are reference, not targets.

## Hosts

| | this machine | robot PC (Orin) |
| --- | --- | --- |
| hardware | RTX 4090, 24 GB | Jetson Orin |
| ROS | none natively (Docker only) | ROS 2 Jazzy in Docker, Zenoh RMW |
| role | training, gate fitting, the harness; talks to the robot over Zenoh as `infer.py` does | VR stack, bringup, motion control |
| runs | pi-0.5 training and inference, gate fitting, the harness, RLinf | `robotis_vuer`, `ffw_bringup`, `cyclo_motion_controller` |

## Cloned here

`python3 scripts/setup_repos.py` prints the plan; `--run` performs it. Both are shallow clones.

* **`external/RLinf`** -- reference for baseline fidelity, and the source of the DAgger algorithm
  layer we reuse.
* **`external/robotis_applications`** (branch `jazzy`) -- `robotis_vuer`, read while writing the
  Quest shim. Must **also** be cloned on the robot PC, where it is actually launched.

## OpenPI and pi-0.5

Required. `models--lerobot--pi05_base` (14 GB) is already cached locally at
`~/.cache/huggingface/hub` -- it survived the disk cleanup deliberately.

Two practical constraints worth settling before the SFT run:

* **Fine-tuning compute.** pi-0.5 is ~3B parameters. This machine is a single RTX 4090 (24 GB),
  which is tight for full fine-tuning; expect LoRA or partial fine-tuning here, or rent H100s the
  way the act_prior runs were done. Decide before collecting, because it affects nothing about the
  data but everything about the schedule.
* **Inference rate.** A 3B VLA cannot run a forward pass per control tick. It does not need to:
  the deployed receding-horizon setup already predicts a chunk and executes 25 waypoints of it, so
  the policy forward runs about once per 0.8 s while the loop holds 30 Hz. The existing
  chunk-and-execute architecture is exactly what a large policy needs. Inference runs on this
  machine and commands the robot over Zenoh, as `infer.py` already does.

## Needed on the robot PC, not here

* `ai_worker` -- `ffw_bringup`, hardware interface
* `cyclo_control` -- motion control, launched `controller_type:=vr`
* `robotis_applications` on `jazzy` -- to launch `robotis_vuer`

## The port surface, measured

RLinf splits the VR integration in two, and only one half is device-specific:

| file | lines | device-specific? | what we do |
| --- | --- | --- | --- |
| `rlinf/envs/realworld/common/pico/pico_expert.py` | 549 | **yes** -- ZeroMQ transport, relative controller motion to normalised deltas | **replace** with `QuestExpert`, a ROS 2 subscriber |
| `rlinf/envs/realworld/common/wrappers/pico_intervention.py` | 625 | mostly no -- `gym.ActionWrapper`, per-arm composition, `hold_current_when_inactive`, `intervene_flag`, `intervene_action` | **reuse the logic**, joint-space variant of the dual-arm class |

So the device change is **one class in one file**. Everything the replacement needs is already on
ROS topics -- see `QUEST_VS_PICO.md` for the verified field-by-field mapping, including that the
per-arm squeeze topics publish outside the deadman gate, so per-arm intervention needs no change to
`robotis_vuer`.

`DualFrankaTcpPicoIntervention` is tcp_rot6d specific; our variant composes 16D joint actions
instead. The per-arm composition logic it contains is what we keep.

## HG-DAgger baseline settings to mirror

Verified line by line against `external/RLinf/examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml`
on 2026-08-14, so the baseline is faithful rather than invented:

| setting | line | value | ours |
| --- | --- | --- | --- |
| `only_save_expert` | 78 | `True` | same |
| `online_lerobot.enabled` | 80 | `True` | same |
| `only_success` | 81 | `True` | same |
| `robot_type` | 82 | `"dual_FR3"` | `"ffw_sg2"` |
| `fps` | 83 | `10` | **30 -- see below** |
| `finalize_interval` | 85 | `1` | same |
| `rolling_lerobot_window_size` | 87 | `50000` | same |
| `min_frames` | 88 | `1` | same |
| `lerobot_num_workers` | 89 | `0` | same |
| `smooth_intervene` | 103 | `True` | same |
| `use_spacemouse` | 107 | `False` | same |
| `use_pico` | 110 | `True` (train) / `False` (eval) | Quest, same semantics |
| `keyboard_reward_wrapper` | 114 | `eval_control` | same |
| `hand` | 119 | `"dual"` | same |
| `hold_current_when_inactive` | 122 | `False` | same |
| `control_trigger` | 123 | `"grip"` | same |

`rollout.expert_model` and a DAgger `beta` are **absent** from the file -- the only `beta` present is
`adam_beta1/2`, the optimiser. So "human intervention alone decides, with no model expert and no
action mixing" is confirmed from the config itself.

### The one deliberate deviation: fps 10 -> 30

RLinf writes its online dataset at 10 fps. The AI Worker control loop, every existing demonstration,
and the VR squeeze stream all run at 30. Recording at 10 would discard two thirds of the
intervention labels.

It does not bias the comparison, because **both methods read the same dataset**. HG-DAgger consumes
the intervened actions and ours consumes the same actions plus the labels; whatever rate is chosen
applies identically to both. Recording at the native 30 simply avoids throwing away data that one
method can use and the other cannot -- which is a property of the methods, not a handicap imposed on
one of them.

Documented here so it is a stated choice rather than an unexplained difference from the reference.

## What blocks the first robot session

One read-only discovery run, commanding nothing:

1. What `cyclo_motion_controller` publishes for the **arms** under `controller_type:=vr`, and on
   which topic. The gripper path is known from source; the arm path is not.
2. Whether `robotis_vuer`'s gripper publishers can be disabled, making it a pure reporter. If so
   the single-writer mux gets much simpler.
3. Rates and QoS of the squeeze and pose topics against our 30 Hz loop.

## Safety

Unchanged and non-negotiable. Dry run before live, always. `infer.py` stays the single writer and
keeps its velocity and acceleration clamps, start-pose refusal, and pre-rollout stillness check.
The both-hands VR deadman is retained as a hard interlock.
