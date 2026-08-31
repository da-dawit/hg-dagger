# Replacing PICO with Meta Quest 3

Source-verified against `robotis_vuer/robotis_vuer/vr_publisher_sg2.py` (main branch, 1426 lines)
and the RLinf PICO documentation.

Author: Dawit Chun
2026-08-14

## Verdict

**Yes, and the Quest side is better instrumented than the PICO path assumes.** Everything RLinf's
`PicoIntervention` consumes is already published by `robotis_vuer` -- it arrives over ROS 2 / Zenoh
instead of ZeroMQ, so the port is a subscriber shim, not a rewrite.

The real work is somewhere else entirely, and it is described in section 3.

## 1. What RLinf's PICO path needs, and where Quest provides it

RLinf's `PicoIntervention` subscribes to a ZeroMQ stream carrying headset pose, per-controller
position/quaternion, grip value, trigger value and button states, and converts relative controller
motion into normalised end-effector deltas. It is **PICO-specific, not abstracted** -- the class is
named for the device and configured under `env.*.pico`.

Every field it needs has a `robotis_vuer` equivalent:

| RLinf / PICO field | AI Worker topic | type | verified |
| --- | --- | --- | --- |
| per-controller grip (`control_trigger: "grip"`) | `/vr_controller/left_squeeze`, `/vr_controller/right_squeeze` | `Float32` | yes |
| controller / wrist pose | `/l_wrist_pose`, `/r_wrist_pose` | `PoseStamped` | yes |
| arm posture beyond the wrist | `/l_elbow_pose`, `/r_elbow_pose`, `/l_shoulder_pose`, `/r_shoulder_pose` | `PoseStamped` | yes |
| gripper open/close buttons | published as gripper `JointTrajectory` directly (see 3) | `JointTrajectory` | yes |
| A / B button combinations | tracked internally (`both_a_buttons_pressed_prev`, `both_b_buttons_pressed_prev`) | — | yes |
| base motion | `/cmd_vel` | `Twist` | yes |
| robot feedback | subscribes `/joint_states` | — | yes |

Transport is the only substantive difference: ZeroMQ (`ipc://` or `tcp://`) on the PICO side,
ROS 2 over Zenoh here. A `QuestIntervention` class that subscribes to the topics above and exposes
the same fields is a drop-in replacement for `PicoIntervention`.

## 2. Per-arm intervention is available WITHOUT modifying robotis_vuer

This resolves an open question in AIWORKER_DAGGER_PORT.md section 4.2, where the concern was that
AI Worker's both-hands deadman would block RLinf's per-arm gating.

The deadman is:

```python
def can_publish_goal_pose(self):
    """Safety gate for goal_pose topics."""
    return (
        self.vr_publishing_enabled and
        self.left_squeeze_value >= self.goal_pose_squeeze_threshold and
        self.right_squeeze_value >= self.goal_pose_squeeze_threshold
    )
```

Both squeezes must exceed `goal_pose_squeeze_threshold` (default 0.8) before any wrist, elbow or
shoulder pose is published. That is the interlock, and it should stay.

**But `/vr_controller/left_squeeze` and `/vr_controller/right_squeeze` are published
unconditionally**, outside that gate. So the per-arm squeeze value is readable at all times even
though the pose gate needs both hands.

Therefore: keep the both-hands deadman as a hard safety interlock, and read the two squeeze topics
to decide **which arm** is being corrected. Per-arm HG-DAgger gating with no change to
`robotis_vuer` and no weakening of an existing safety property.

## 3. The real problem: robotis_vuer is a COMMANDER, not a reporter

RLinf's architecture assumes the VR consumer only *reports* controller state, and the environment
composes the action vector. That assumption does not hold here.

`vr_publisher_sg2.py` publishes the grippers **directly to the robot's command topics**:

```
/leader/joint_trajectory_command_broadcaster_left/joint_trajectory
/leader/joint_trajectory_command_broadcaster_right/joint_trajectory
```

Those are **the same two topics `infer.py` publishes to** (`spec_sg2.py`: `CMD_LEFT`, `CMD_RIGHT`).
It also drives `/cmd_vel` for the base, and the head and lift through
`/leader/joystick_controller_{left,right}/joint_trajectory`.

Meanwhile the arms travel a different route: the wrist / elbow / shoulder `PoseStamped` topics feed
`cyclo_motion_controller`, which generates arm trajectories. (Those publishers are named `*_rviz_pub`
but they are gated by `can_publish_goal_pose()`, so they are references, not visualisation.)

So during HG-DAgger there are up to **three writers** to the same command topics: `infer.py`
(arm and gripper), `robotis_vuer` (gripper), and `cyclo_motion_controller` (arm). Nothing arbitrates
between them.

**This is the engineering problem, and it is not about VR at all.** The fix is the one already
recommended: make `infer.py` the single writer, subscribing to the VR-derived commands and choosing
per arm which source reaches the robot, logging the choice as `intervene_flag`. `infer.py` holds
every safety guard we have -- velocity and acceleration clamps, start-pose refusal, pre-rollout
stillness check -- so it is the correct place to concentrate authority.

## 4. What still has to be discovered on the robot

1. What `cyclo_motion_controller` publishes for the **arms** under `controller_type:=vr`, and on
   which topic. The gripper path is now known from source; the arm path is not.
2. Whether `robotis_vuer` can be run with its gripper publishers disabled, so it becomes a pure
   reporter. If a launch parameter allows this, the mux gets much simpler.
3. Rates and QoS of the squeeze and pose topics against our 30 Hz control loop.

All three are answerable read-only, in one session, commanding nothing.

## 5. Practical notes carried over from the VR guide

* Default pose (arms straight down) puts the hands outside the camera field of view. A bent-elbow
  ready pose is required, and it must then match the `q_home` our start-pose guard uses.
* The robot needs internet on the WAN port for the whole VR session.
* Wired is strongly preferred; USB-C-to-Ethernet from the Quest into the **LAN** port if wireless
  is insufficient.
* Tape over the Quest proximity sensor stops the session pausing when the headset is off the head.
* Activation is left-X + right-A together, followed by a 3-second alignment check and a 5-second
  slow start. Match your arm posture to the robot's first.

---

## Appendix: confirmed on hardware, 2026-08-14

Section 2 argued from source that the per-arm squeeze topics publish outside the deadman gate.
Measured on the robot, with `vr.launch.py model:=sg2` running and the robot NOT brought up:

* `/vr_controller/left_squeeze` reported **0.2586 while the right grip was untouched** -- below the
  0.8 gate, so the value is genuinely ungated. Per-arm intervention needs no change to
  `robotis_vuer`, and the both-hands interlock stays intact.
* Full squeeze reaches **1.0**, so the whole range above the 0.8 threshold is available.
* The signal is **analog (0.0-1.0), not binary**. We may set our own intervention threshold
  independently of the ROBOTIS safety gate, and press depth is available as a confidence signal.
* The topic publishes **continuously, including at rest** -- the stream of 0.0 values is exactly the
  ~95% of non-intervention timesteps our method trains on and HG-DAgger discards.
* Topics are visible **across containers**, so the Zenoh graph is shared between the
  `robotis_applications` and `ai_worker` containers.

Also established, and it matters for procedure: the Quest **proximity sensor must be taped**. With
the headset off the face the session pauses, the ROS node keeps publishing its last value, and the
squeeze reads a constant 0.0 -- which looks exactly like a broken topic rather than a paused
session. That cost one debugging cycle.

Safety state during this test: `ros2 node list` showed only `/cyclo_manager`, and
`ros2 topic info` on the gripper command topic returned "Unknown topic" -- no publishers, no
subscribers. Nothing could move.

---

## Appendix 2: live graph discovery, 2026-08-14

Captured with the robot up, `motion_controller controller_type:=vr` running, teleop working.

### The command path -- section 3 confirmed, with the arm route now known

`/leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory`:

    Publisher count: 2
      vr_controller      RELIABLE     KEEP_LAST(10)     <- cyclo VR controller, ARM trajectories
      vr_publisher_sg2   BEST_EFFORT  KEEP_LAST(1)      <- robotis_vuer, GRIPPER
    Subscription count: 1
      arm_l_controller   RELIABLE     KEEP_LAST(42)     <- ros2_control

So **arm and gripper converge on one topic pair**, consumed by `arm_l_controller` /
`arm_r_controller`. `infer.py` publishing there would be the **third** publisher. The single-writer
design is confirmed necessary -- and simplified, because there is only one topic pair to arbitrate
rather than separate arm and gripper routes.

### A QoS incompatibility worth testing

`vr_publisher_sg2` publishes **BEST_EFFORT** into a subscriber that is **RELIABLE**. Under standard
ROS 2 QoS matching that pair is incompatible and no connection is formed, so those messages would be
silently dropped.

Falsifiable prediction: **the VR grippers should not respond during teleop.** If they do respond,
the reasoning above is wrong somewhere -- possibly `rmw_zenoh` matches differently from the DDS
default -- and that is worth knowing before we rely on QoS behaviour anywhere else.

### Existing safety machinery we should use rather than reinvent

* `/reference_checker` node and `/reference_diverged` topic -- a divergence check already exists.
* `/arm_l_controller/speed_scaling_input`, same for the right -- a **runtime speed scaling input**.
  A global speed lever we can drive without touching our own clamps.
* `/leader/.../raw_joint_trajectory` alongside the processed topic -- worth understanding which is
  which before muxing.
* Richer reference topics than the source showed: `/l_goal_pose`, `/l_subgoal_pose`,
  `/l_gripper_pose` and the right-hand equivalents.

### Depth is available

`camera_left` and `camera_right` are RealSense units publishing `depth/image_rect_raw` and
`extrinsics/depth_to_color`, and there is a **ZED stereo camera** as well. That answers the open
question in OKAMI_NOTE.md section "Hard parts, honestly" item 2: metric scale for object
trajectories is available without building a stereo pipeline.

### The VR ready pose is not the old q_home

Measured ready pose vs the two-bottle task's `q_home`: **max difference 71.6 deg (1.250 rad)**,
driven mostly by `joint1` (-67 deg L, -72 deg R) and `joint6` (+60 deg both). Hands sit 30 cm
further forward and 9 cm higher (left hand x +0.376 vs +0.079, z -0.203 vs -0.295).

Two consequences:

1. **Old policies cannot start from the VR ready pose.** `infer.py` will refuse -- 1.250 rad against
   a 0.30 tolerance -- and that refusal is correct, not a nuisance. Do not raise the tolerance to
   get past it; the policy has never seen this pose.
2. The ready pose puts both hands at y = +0.185 / -0.160, **inside the bimanual overlap band**
   (y in [-0.23, +0.23]) computed from joint limits. Convenient for a two-handed task: the arms
   start where they can already reach a shared object.
