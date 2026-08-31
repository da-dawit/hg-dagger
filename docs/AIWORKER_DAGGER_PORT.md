# Human-Gated DAgger on the ROBOTIS AI Worker

Porting the RLinf dual-arm HG-DAgger pipeline to AI Worker (FFW-SG2) with Meta Quest 3,
and a two-way comparison of HG-DAgger against our gate-as-potential method.

Author: Dawit Chun
2026-08-14

---

## 1. What this document is for

Two separate questions, and the first one matters more:

1. **Can the RLinf HG-DAgger pipeline actually run on AI Worker?** RLinf's guide is written for two
   Franka arms, Robotiq grippers, a PICO headset and a ZeroMQ bridge. None of that is our hardware.
   The question is which parts are hardware-independent, which must be rewritten, and where the
   genuine blockers are.
2. **Does our method beat HG-DAgger?** Only two arms in the comparison, by design. See section 6.

It is also written so someone who has not been inside either stack can follow what the system is,
because that was asked for directly.

---

## 2. The two systems, explained

### 2.1 What HG-DAgger is

Human-Gated DAgger (Kelly et al., 2019). The robot executes its own policy. A human watches, and
when the robot is about to do something wrong, the human takes over for a few seconds, then hands
control back. Those human corrections are added to the training set and the policy is retrained.
Repeat.

The idea it fixes: plain behaviour cloning only ever sees states a *human* visited. As soon as the
robot drifts somewhere a human never went, it has no idea what to do, and the error compounds.
Letting the human correct the robot *in the states the robot actually reaches* is the fix.

What RLinf adds on top is engineering, not algorithm: per-arm intervention, a foot pedal for
success/failure marking, automatic writing of successful episodes to a LeRobot dataset, and a
sampler that trains only on action chunks that were entirely human-controlled
(`only_save_expert: True`).

### 2.2 What the AI Worker VR system is

Five pieces, in the order data flows:

| piece | what it does |
| --- | --- |
| **Meta Quest 3** | The headset and two controllers. Reports hand poses and button states. |
| **Vuer** | A browser-based VR client, served over HTTPS at `https://{pc_ip}:8012?ws=wss://{pc_ip}:8012`. The Quest's browser opens this page; "Enter VR" fixes the VR coordinate origin at your current physical position. |
| **`robotis_vuer`** | ROS 2 node, launched with `vr model:=sg2`. Publishes the VR reference poses into the ROS graph. |
| **`cyclo_motion_controller`** | The motion-control layer. Launched with `controller_type:=vr`, it consumes VR references and generates the arm trajectories the robot follows. This is where Cartesian VR poses become joint motion. |
| **`ffw_bringup`** | Brings the robot up (`ffw_sg2_follower_ai.launch.py`) and holds the hardware interface. |

Transport is **Zenoh**, not DDS: AI Worker uses `rmw_zenoh` since version 2.0.0, so `rmw_zenohd`
must be running before anything else. Default `ROS_DOMAIN_ID` inside the container is 30.

Two safety behaviours are built into the VR path and both matter to us:

* **Deadman.** VR references are published *only* while **both** controller squeeze/grip buttons are
  held. Release either one and teleoperation stops publishing.
* **Aligned activation.** Pressing left-X and right-A together activates the controller. The system
  then compares controller poses against the robot's actual wrist poses; if they are close enough it
  starts after 3 seconds, followed by a 5-second slow-start ramp. The documentation is explicit that
  you should match your arm posture to the robot's before activating.

---

## 3. Component mapping: RLinf to AI Worker

| RLinf (dual Franka + PICO) | AI Worker equivalent | status |
| --- | --- | --- |
| `vr_data_publisher` (XRoboToolkit) | `robotis_vuer` / `vr.launch.py model:=sg2` | **exists** |
| PICO controllers over ZeroMQ (`zmq_addr`) | Meta Quest 3 over Vuer, HTTPS/WSS port 8012 | **exists**, different transport |
| `pico.hand: "dual"` | Quest reports both controllers natively | **exists** |
| `control_trigger: "grip"` (deadman) | both squeeze buttons held | **exists**, same semantics |
| calibration `button: "trigger"` | left-X + right-A together | **exists**, different buttons |
| libfranka / franky arm control | `ffw_bringup` + `cyclo_motion_controller` | **exists** |
| Robotiq grippers | SG2 grippers (`gripper_l_joint1`, `gripper_r_joint1`) | **exists** |
| RealSense / Lumos by serial | AI Worker's 3 cameras (scene + 2 wrist) | **exists** |
| foot pedal via `RLINF_KEYBOARD_DEVICE=/dev/input/eventXX` | any USB pedal or keyboard exposes an evdev node | **trivial**, RLinf reads raw evdev |
| Ray cluster, 2-3 nodes | robot Orin + one GPU box | **needs setup** |
| `realworld_dual_franka_tcp_rot6d` env worker | — | **must be written** |
| action space `tcp_rot6d`, 20D | AI Worker native: **16D joint** (14 arm + 2 gripper) | **decided: joint space** |
| OpenPI pi-0.5 | `act_prior` / ACT, already trained here | **substitute** |
| online LeRobot writer, `intervene_flag`, `only_save_expert` | hardware-agnostic algorithm layer | **reusable as-is** |

**The headline:** RLinf's *algorithm* layer is reusable and its *environment* layer is not. Nothing
above is a blocker in the sense of "impossible"; the work concentrates in one place, described next.

---

## 4. The three real gaps

### 4.1 Two writers to the same joints (the main engineering problem)

During HG-DAgger, the policy and the human must both be able to command the arms, switching
per-arm, at 30 Hz. Today those are two independent paths:

* policy: `infer.py` publishes joint trajectories to the leader command topics
* human: `cyclo_motion_controller` consumes VR references and generates arm trajectories

Running both at once means two publishers commanding the same joints. That is unsafe and will not
arbitrate itself.

**Recommendation: extend `infer.py` rather than add an arbiter node.** `infer.py` already holds
every safety guard we have — velocity and acceleration clamps, the start-pose refusal, and the
pre-rollout stillness check. Keeping it as the *single writer* to the robot means there is exactly
one place that can command motion. It subscribes to whatever `cyclo_motion_controller` produces
and, per arm, either forwards the policy's chunk or the human's command, recording which at every
timestep. That record **is** the `intervene_flag`, and it is the data our method consumes.

**Discovery task, and it gates the design:** find which topic `cyclo_motion_controller` publishes
when `controller_type:=vr`, and whether it emits joint trajectories or Cartesian targets. If joint,
the mux is straightforward. If Cartesian only, we need inverse kinematics in the loop or we must
mux upstream of the controller. This is question #1 for the first session on the robot.

### 4.2 Per-arm intervention versus a both-hands deadman

RLinf's DAgger uses per-arm gating: hold the left grip and only the left arm is replaced, while the
right continues under policy control. AI Worker's VR publisher requires **both** squeeze buttons to
publish anything at all.

These conflict. Options, in order of preference:

1. Keep the both-hands deadman as a hard safety interlock, and use a *separate* per-arm signal
   (a controller button, or the pedal) to choose which arm is being corrected. Safety behaviour
   is unchanged.
2. Modify `robotis_vuer` to publish per-arm. More faithful to RLinf, but it weakens an existing
   safety property, so it needs deliberate sign-off.

Option 1 is recommended. It costs one button and changes no safety behaviour.

### 4.3 The environment worker

RLinf's `realworld_dual_franka_tcp_rot6d` talks to libfranka directly. An AI Worker equivalent must
present the same interface to RLinf (reset, step, observation, `intervene_action`, `intervene_flag`)
while speaking ROS 2 / Zenoh underneath. This is the bulk of the port and is ordinary work, not
research risk.

### 4.4 Non-blocking notes

* **Initial pose.** The AI Worker default pose is arms straight down, which puts the hands outside
  the camera field of view. A bent-elbow "ready" pose is required before VR is usable. This must be
  set once and then used consistently, because our policies and our `q_home` start-pose guard are
  defined against the demonstrated start pose.
* **Network.** VR quality depends heavily on the link. Wired is recommended; if wireless is
  insufficient, host Vuer on the robot PC and give the Quest a USB-C-to-Ethernet adapter into the
  **LAN** port, not WAN.
* **Internet.** The VR publisher requires the robot to be online for the whole session (WAN port).
* **Proximity sensor.** The Quest pauses when it thinks it has been removed; tape over the sensor
  between the lenses avoids this.

---

## 5. Verified environment facts

Checked on the user PC on 2026-08-14:

* This machine is x86_64 at `10.42.0.1` (robot reachable at `10.42.0.22`, which is what
  `infer.py --router-ip` already uses) and also carries `192.168.6.101`, the same subnet as the
  Orin address `192.168.6.2` used in the VR documentation.
* `robotis_applications`, `ai_worker` and `cyclo_control` are **not** present on this machine. They
  live on the robot PC. Nothing in this document has been executed against the robot.
* AI Worker native action space confirmed 16D from 96,003 demonstration frames: 14 arm joints plus
  `gripper_l_joint1`, `gripper_r_joint1`.

---

## 6. The comparison study

Two methods only.

| | HG-DAgger | ours (gate as potential) |
| --- | --- | --- |
| what the human supplies | corrective actions while intervening | the same corrections, **plus** the intervention decision at every timestep |
| fraction of human output used | ~5% (intervened steps only) | **100%** — the ~95% of non-intervention steps are labels too |
| reward source | none; pure imitation | servo stall (physical) + gate logit used as a **potential** |
| training signal | supervised loss on expert-only chunks | RL on `r_phys + gamma*Phi(s') - Phi(s)` |
| can the human leave? | no | **yes, once the gate classifier is fitted** |

### 6.1 The design property that makes this cheap and fair

**One data collection serves both methods.** HG-DAgger consumes the intervened *actions*; ours
consumes the same actions plus the intervention *labels*, which are recorded anyway. So a single
human session is replayed offline into both pipelines. Neither method gets more human time than the
other, by construction, and there is no scheduling confound.

### 6.2 Held constant

Same robot, same task, same initial policy (behaviour cloning on the same demonstrations), same
operator, same episode budget, same evaluation protocol.

### 6.3 Metrics

**Primary: success rate against human-minutes.** Human attention is the resource the two methods
spend differently, so it is the correct x-axis. Not wall-clock, not environment steps.

Secondary:

* episodes to reach a fixed success threshold
* **query rate** actually observed. Our theory needs the human to be a *thresholded* oracle firing
  on a small fraction of steps; the predicted operating point is 4-9% (HIL_THEORY.md sec 8). If the
  measured rate is far higher, assumption A1 is wrong and we report that.
* **scaling with horizon** — in simulation the advantage grew with task length (baseline never
  solved past 8 stages; ours 15 to 40 episodes across a 4x horizon increase). On hardware, the
  number of objects deposited is the horizon knob, so the same curve is measurable.

### 6.4 What must be logged, and it is more than the current pipeline records

1. **Both-arm joint states at full rate**, for the proprioceptive reward and for FK.
2. **The intervention flag at every timestep, per arm** — including the non-intervention steps.
   This is the single most important addition. It is 95% of the data our method uses and the
   existing pipeline does not record it at all.
3. Episode outcome (success/failure) from the pedal.

---

## 7. Bring-up order

Each step is checkable before the next begins.

1. **Read-only discovery on the robot.** Bring up Zenoh, `ffw_bringup`, and
   `cyclo_motion_controller controller_type:=vr`, then list topics and resolve: what does the VR
   controller publish, joint or Cartesian, at what rate. **Nothing is commanded.** This answers 4.1.
2. **VR teleoperation as documented**, unmodified, to confirm the stack works end to end and to set
   the bent-elbow ready pose.
3. **Logging.** Record joint states, VR references, and the deadman/per-arm intervention state
   together, time-aligned. No policy in the loop yet.
4. **Demonstrations** for the chosen task, via VR.
5. **Behaviour cloning** to produce the shared initial policy for both comparison arms.
6. **Single-writer mux in `infer.py`**, dry-run first, with the arbitration logged but the human
   path disabled. Verify the flag is recorded correctly before it can move anything.
7. **HG-DAgger round 0.** Live, short episodes.
8. **Offline replay into both methods**, then evaluation.

---

## 8. Safety requirements

These are not optional and several were learned the hard way.

* **Dry run before live, always.** No command that moves the arms is issued from code that has not
  been executed in a non-publishing mode first.
* **Single writer.** Exactly one process may command the arms at any time. This is the reason for
  the recommendation in 4.1.
* `infer.py`'s existing guards stay in force: velocity and acceleration clamps, start-pose refusal,
  and the pre-rollout stillness check (refuses to plan from a moving arm; thresholds derived from
  the demonstrations at 1443x separation between a resting and a working arm).
* The both-hands deadman is retained as a hard interlock.
* Aligned activation and the 5-second slow start are retained.
* First live sessions run 1-2 episodes, not 30.

---

## 9. Open questions

1. What does `cyclo_motion_controller` publish under `controller_type:=vr` — joint or Cartesian?
   (Blocks 4.1. Read-only to answer.)
2. Can per-arm intervention be signalled without weakening the both-hands deadman? (4.2)
3. Which foot pedal or input device, and its `/dev/input/event*` node?
4. Does Ray need to run on the robot Orin, or can the env worker be a thin ROS bridge with training
   entirely on the GPU box? The latter is preferable and probably sufficient.
5. Does the VR-generated motion respect the same velocity limits our policy path enforces? If not,
   the human can move the robot faster than the policy is ever allowed to.
