# HG-DAgger: what runs off-robot, and what it needs from the robot side

For the agent working on the robot. Written 2026-08-20.

## The short version

A workstation process (`hil_dagger/online_rl.py`) drives the follower with a GR00T policy over
zenoh, and records every control tick for later retraining. The operator watches, and when the
policy is about to fail he takes over with the leader arms and corrects it. Those corrections
become new training data. That is HG-DAgger.

Nothing new is installed on the robot. The workstation is a **zenoh client** talking to the
robot's existing graph.

## What the workstation publishes, and when

It writes to exactly the topics the leader already writes to:

    /leader/joint_trajectory_command_broadcaster_left/joint_trajectory
    /leader/joint_trajectory_command_broadcaster_right/joint_trajectory
      joints: arm_{l,r}_joint1..7 + gripper_{l,r}_joint1   (8 per arm, gripper interleaved LAST)

Timing: one JointTrajectory per control tick at ~21-30 Hz, single point, absolute joint positions,
rate-limited to <= 0.6 rad/s and <= 2.0 rad/s^2 before publishing.

**It publishes ONLY while the policy is driving.** The moment the operator takes over it goes
silent -- it does not command the robot at all during an intervention. That is deliberate: the
takeover is fail-safe, so if the workstation process dies mid-intervention it was already silent.

It also reads `/joint_states` and three camera topics (ZED left eye, both wrist RealSense).

## The three states we record, per tick

     0  policy drove and the operator let it     <- the majority; only our method uses these
     1  operator drove                           <- the expert action HG-DAgger trains on
    -1  FROZEN: operator repositioning the leader while the follower holds still

The -1 state exists **because of the leader-side freeze**. While frozen the follower is stationary
and the leader's motion is not a demonstration of anything. If those ticks were recorded as
intervention, the policy would learn "when the human takes over, hold still" -- the opposite of the
correction being made. They are excluded from training entirely.

## WHAT WE NEED FROM THE ROBOT SIDE

### 1. `/arm_freeze/status` -- message type and exact semantics  [BLOCKING]

We must subscribe to it to label ticks -1. Please provide:

    ros2 topic info /arm_freeze/status
    ros2 topic echo /arm_freeze/status --once

Specifically we need:
  - the message TYPE (std_msgs/String? a custom msg? which package?)
  - the exact FIELD names and the exact VALUE strings for frozen vs free
  - whether it is published continuously (at what rate) or only on state CHANGE
  - whether left and right are separate fields or one combined string

If it is latched/transient-local, say so -- a subscriber that joins late otherwise sees nothing
until the next change, and would mislabel every tick until the operator toggles.

The operator freezes and releases BOTH arms together, so a single combined state is sufficient.

### 2. Does the leader publish continuously, or only while engaged?  [BLOCKING]

Our process refuses to publish if another node is publishing to those command topics, because
three uncoordinated writers on a joint command topic produces motion nobody asked for.

  - If the leader node publishes ONLY while the operator is actively driving (and is silent while
    frozen and while the policy drives), we are fine as-is.
  - If it publishes CONTINUOUSLY, we need either a way to gate it, or an explicit guarantee about
    who wins, before we run this live.

Please describe the actual behaviour rather than the intent.

### 3. What holds the follower while an arm is frozen?

We assume the JointTrajectoryController simply holds its last commanded point when nothing new
arrives. Please confirm, and say whether anything continues publishing a hold pose during freeze.

### 4. Re-engagement on release

The operator reports that release "re-zeroes at the current pose, no jump". We depend on that: a
step discontinuity at release would be recorded as an expert action and would train the policy to
lunge. Please confirm the offset is applied on the LEADER side and that the follower receives a
continuous trajectory across the release.

### 5. Camera reliability  [IMPORTANT, not blocking]

`cam_right_wrist` (/camera_right/camera_right/color/image_rect_raw) intermittently times out --
observed as "timeout waiting for frame after 2000.0ms" then an all-zero frame, after ~200 good
steps. The other two are stable. If both wrist cameras share a USB controller, bandwidth contention
is the likely cause. This aborts a run mid-motion (safely -- we stop publishing -- but abruptly).

## What we do NOT need

  - nothing installed or launched on the robot by us
  - no changes to the controllers or the leader
  - no new topics published for our benefit, beyond the freeze status that already exists

## Open issue from the operator

"it doesn't drive" -- the policy is not moving the follower. Most likely one of:
  - our process refusing to publish because other publishers are active (see item 2); it prints
    `[writer] OTHER PUBLISHERS ARE ACTIVE` and the node names when this happens
  - the run started without `--live` (dry run publishes nothing by design)
  - the start-pose guard refusing before the loop begins
The console output distinguishes these; please capture it if the robot side is investigating.
