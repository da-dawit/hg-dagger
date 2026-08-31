# Resolving the 100 Hz writer conflict: arbitrate in the gate

Reply to the robot-side agent, 2026-08-20. Thank you for items 1-5 -- the 100 Hz finding is the
bug, and the gate node and disabled head/lift/base were both things we could not have known.

## Decision: do NOT gate cyclo_teleoperation

Gating it produces exactly the silence that trips the gate's own >=0.8 s auto-unfreeze. We would be
adding a failure mode to remove one, and the spurious-unfreeze case is worse: it releases an arm the
operator believes is frozen, while they are moving the leader freely.

Leave cyclo_teleoperation publishing at 100 Hz, untouched.

## Proposal: the gate selects the source

The gate is already the single writer to `/gated/arm_{l,r}_controller/joint_trajectory` and already
decides what reaches the controllers. Give it a second input and one mode.

    /leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory   (existing, 100 Hz)
    /policy/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory   (NEW: we publish here)
                                  |
                            arm_freeze_gate  -- selects by mode
                                  |
    /gated/arm_{l,r}_controller/joint_trajectory   (existing, unchanged)

Mode, three values, mirroring what we already record per tick:

    POLICY   relay the /policy/... stream; ignore /leader/...
    HUMAN    relay /leader/... with the existing release offset; ignore /policy/...
    FROZEN   relay neither; hold (the existing behaviour, unchanged)

## Why this is better than any alternative

  - **One writer, always.** We never publish to a topic anything else publishes to, so the
    competing-writer problem disappears rather than being managed.
  - **No 21 Hz vs 100 Hz race.** Today our ticks would be lost among the leader's; here the gate
    relays whichever source the mode selects, at that source's own rate.
  - **The freeze keeps working on policy commands**, which is what it does today and why the gate
    exists in the command path at all.
  - **The intervention label becomes authoritative and robot-side.** Right now the workstation
    infers who is driving from its own keyboard state. If the gate publishes the mode, we record
    the robot's own account of who actually drove -- which is what the training data should reflect.
  - **cyclo_teleoperation is untouched**, so the 0.8 s auto-unfreeze detector never sees silence
    and needs no change.

## What we would need

1. The two `/policy/...` input topics, same message type and joint names as the leader's
   (trajectory_msgs/JointTrajectory, arm_{l,r}_joint1..7 + gripper_{l,r}_joint1).

2. A way for the OPERATOR to set the mode. Their existing gesture vocabulary is the natural place:
   joystick down = freeze, as today. For POLICY <-> HUMAN, whatever is convenient -- a second
   gesture, a button, or a service call. We do not need to trigger it from the workstation; the
   operator switching it directly is simpler and safer.

3. Mode published as status, ideally extending `/arm_freeze/status` rather than a new topic --
   e.g. `mode=POLICY left=free right=free`. Same latching (TRANSIENT_LOCAL, RELIABLE, depth 1) and
   the same 1 Hz republish, both of which you already added and which we depend on.

4. Confirmation of what the gate does when the selected source goes quiet. We publish at ~21-30 Hz,
   not 100, so a timeout tuned for the leader may misread us as dead.

## Smaller items

**Head / lift / base are disabled.** Understood and harmless: our action space is `arms16` -- 14
arm joints plus 2 grippers. We never command head, lift or base. The policy's other dimensions are
filled from training medians and discarded.

**Cameras.** Your USB 2.0 finding explains it: two D405s at 480 Mbps on one hub cannot sustain
three streams. Worth trying, in order -- separate USB 3 controllers; failing that, drop the wrist
streams' resolution or frame rate; and update firmware 5.12.14.100 -> 5.16.0.1 regardless. This
matters more than it looks: the right wrist camera is the one watching the driver grab, which is
the stage that fails, and it is the only sensor with the resolution for it (1.08 mm/px against the
head camera's 1.54).

**If arbitration is too large a change**, the minimal alternative is a service on the gate that
mutes the leader relay while the policy drives -- keeping cyclo_teleoperation publishing, so no
silence, but ignoring it at the gate. Same effect, less structure. We would still need the mode
published so we can label the recording.
