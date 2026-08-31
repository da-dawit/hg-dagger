# HG-DAgger data collection: where we landed

For the robot-side agent, 2026-08-20. Summarises the workstation side so both halves agree.

## Settled

**Topics.** We publish to `/policy/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory`
at ~18-21 Hz (measured on the robot: steady 18 Hz, occasional 174 ms gaps when inference runs long).
We are the sole writer there. `spec_sg2.py` was changed from `/leader/...` to `/policy/...`.
Confirmed working end to end -- `ros2 topic hz` shows our traffic arriving.

**Freeze status.** We subscribe to `/arm_freeze/status` and parse
`mode=POLICY left=free right=FROZEN`. Latching (TRANSIENT_LOCAL, depth 1) and the 1 Hz republish
both matter to us: a late subscriber must see the current state, not wait for a toggle.

**Three-state recording.** Every control tick is labelled from the ROBOT'S account, not ours:

     0  policy drove, operator let it     -> only our method trains on these
     1  operator drove                    -> the expert action HG-DAgger trains on
    -1  FROZEN                            -> excluded from both

The -1 case exists because while frozen the follower holds still and the leader's motion is not a
demonstration. Recording those as intervention would teach the policy that "human takes over" means
"stop moving".

**Everything is recorded, then filtered at training time.** Nothing is discarded during collection.

## Still needed from you

1. **Allow the mode toggle while frozen** -- the one blocking change. See GATE_MODE_SWITCH_BUG.md
   and its addendum. Without it there is no safe path from POLICY to HUMAN, so a takeover is not
   possible and HG-DAgger cannot run at all.

2. **`initial_mode:=POLICY`** so the operator does not have to toggle in at startup, ideally with a
   short ignore-window on `/policy/` input at gate start-up to cover the "restart the launch while
   inference is already running" case.

3. **Label the ease-in** in `/arm_freeze/status` (e.g. `easing=true`). During ease-in the action we
   record is not what produced the next state, so those ticks must be excluded.

4. **Log the release gap** `|policy_target - held_pose|` per joint, so we can tell whether the right
   arm dropping on release is the ease-in traversing a large gap (our problem -- we keep predicting
   while frozen and drift) or the ease-in not being applied (yours).

## Operator constraints that shaped the design

He runs this **alone, with both hands on the leader arms**. There is no free hand for the keyboard
while teleoperating. Hands are free only while the POLICY is driving, and after an episode ends. So:

  - recording **auto-starts**; no keypress to begin
  - the episode ends on a step budget or one keypress, and success/failure is asked for
    **afterwards**, blocking, once the leader is put down
  - the only in-run key is **ENTER**, which advances the subtask stage. The policy is conditioned on
    five per-frame sentences and cannot advance itself -- left on stage 0 it grabs the bolt and
    stops. ENTER needs no aiming, and stage changes fall while the policy is driving anyway.

Everything else is joysticks, which is why the arbitration you built matters so much: it moved the
whole control surface onto the device already in his hands.

## Intended session flow

    run online_rl.py            recording starts immediately
    LEFT joystick               policy drives      (or startup default, if item 2 lands)
    ...policy goes wrong...
    RIGHT freeze -> LEFT to HUMAN -> RIGHT release      operator drives from the held pose
    ...operator corrects...
    RIGHT freeze -> LEFT to POLICY -> RIGHT release     policy continues
    ENTER at each stage change
    'c' when the task completes, or the step budget expires
    answer y/n
    next episode

Target ~10 successful episodes, then aggregate and retrain warm-started from the current checkpoint,
~2,500 steps.

## Not blocking, still true

`cam_right_wrist` times out intermittently (both D405s on one USB 2.0 hub). Deprioritised by the
operator for now. Worth knowing it is the camera watching the driver grab -- the stage that fails --
and the only sensor with the resolution for it (1.08 mm/px against the head camera's 1.54).
