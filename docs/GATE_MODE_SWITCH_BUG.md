# POLICY -> HUMAN has no clutch, and there is no safe path for a takeover

Robot side, 2026-08-20. Two observations from the operator, one root cause.

## Observed

1. **Switching POLICY -> HUMAN with the left joystick makes the follower jump to the leader's
   pose** -- the operator reports it "goes back to the home position", which is wherever the leader
   arm happens to be parked. This is a full-speed move to an arbitrary pose, unannounced.

2. On release from freeze back into POLICY, the right arm moves sharply downward. Possibly just the
   ease-in traversing a large gap, but see the question at the end.

## Root cause

The clutch offset is computed only on **release from freeze**:
`offset = held_pose - leader_pose`.

There is no equivalent on a **mode switch**. So POLICY -> HUMAN starts relaying the leader's
absolute joint targets immediately, and the follower jumps from wherever the policy left it to
wherever the leader is.

## Why this blocks HG-DAgger

The entire method is: policy drives, and the moment it is about to fail the operator takes over and
corrects it *from that pose*. The takeover has to be continuous -- a jump at the handover is
recorded as an expert action and would train the policy to lunge.

Right now there is no safe path from POLICY to HUMAN:

    POLICY -> left joystick -> HUMAN            follower jumps to the leader   (observed)
    POLICY -> freeze -> toggle                  refused while frozen           (by design)
    POLICY -> freeze -> release                 returns to POLICY, not HUMAN

## What we are asking for

**Apply the same clutch on a mode switch into HUMAN.** At the moment POLICY -> HUMAN is requested:

    offset := current_follower_pose - current_leader_pose

then relay `leader + offset` exactly as the freeze-release path already does. The first point after
the switch equals the pose the policy left, and the operator drives on from there with no
discontinuity. This is the behaviour item 4 of the previous reply already implements for
freeze-release; it just needs to fire on the mode transition too.

If that is easy, everything else follows and no other change is needed.

**Alternative, if the offset is awkward to compute on that transition:** allow the mode toggle
*while frozen*, and recompute the offset on release according to whichever mode is then active. The
operator would do: freeze -> toggle to HUMAN -> release, and release already has the clutch. That
was refused deliberately, so this is only worth doing if the direct fix is harder.

## Question about the ease-in

On release into POLICY the right arm dropped noticeably. Two possibilities we cannot distinguish
from our side:

  - the ease-in is working as designed and simply has a long way to travel, because our policy kept
    predicting while frozen and its target drifted from the held pose; or
  - the ease-in is not being applied on that path.

Could you log the gap at release -- `|policy_target - held_pose|` per joint -- so we can tell? If
it is the first, we can hold the last pre-freeze command instead of a fresh prediction while frozen,
which is a change on our side, not yours.

## Still outstanding from before

The ease-in ticks need labelling in `/arm_freeze/status` (e.g. `easing=true`), so we can exclude
them from training. During ease-in the recorded action is not what produced the next state.

---

# ADDENDUM -- the operator's proposal is better. Please do this instead.

Disregard the "apply a clutch on the mode switch" request above. The operator proposes using
**freeze as the transition point**, and it is a better design than what we asked for:

    policy running
      RIGHT joystick   freeze -- arm holds
      LEFT  joystick   switch POLICY -> HUMAN        <-- currently REFUSED, this is the only change
      RIGHT joystick   release -- existing clutch applies, operator drives from the held pose
      ... operator corrects ...
      RIGHT joystick   freeze
      LEFT  joystick   switch HUMAN -> POLICY
      RIGHT joystick   release -- existing ease-in, policy continues

Why this is better than a clutch on the mode switch:

  - **No new offset logic.** Freeze-release already computes and applies the clutch correctly, for
    both directions. Reuse it rather than adding a second path that has to stay consistent with it.
  - **Every transition happens with the arm stationary.** Switching source mid-motion is the
    dangerous case; this removes it entirely instead of making it survivable.
  - **It matches how the operator already thinks about the device** -- right stick stops the arm,
    left stick decides who drives next. One gesture vocabulary, no special cases.

**The single change needed: allow the mode toggle while frozen**, and on release apply the
transition appropriate to whichever mode is then active (clutch offset into HUMAN, ease-in into
POLICY). It was refused deliberately because "switching source underneath a freeze would make the
release ambiguous" -- but the ambiguity disappears if release simply reads the current mode.

## Also please consider: `initial_mode:=POLICY`

The operator would like the policy to be driving from the start rather than having to toggle in.
You already offer `-p initial_mode:=POLICY`. Our earlier advice was to keep the HUMAN default, but
in practice the robot cannot move at launch anyway -- nothing publishes to /policy/... until the
workstation process starts. The one real hazard is restarting the follower launch while inference
is already running, which would begin relaying immediately.

If it is easy, a short ignore-window at startup (e.g. discard /policy/ input for the first 2 s
after the gate comes up) would remove that hazard and let POLICY be the default safely.
