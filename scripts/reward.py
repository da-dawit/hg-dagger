#!/usr/bin/env python3
"""Reward from proprioception alone: no vision, no human labelling, no reward classifier.

Author: Dawit Chun

WHY THIS IS POSSIBLE. The grippers are position-controlled. Commanding one past an object's width
makes it stall against the object, so the difference between the COMMANDED and the ACHIEVED joint
angle reports whether something is between the jaws. Measured on 90 demonstrations, restricted to
frames settled at least 15 steps so a gripper still travelling is not mistaken for one stalling:

    right gripper   median stall gap per episode 0.331 rad, fires on 90/90 episodes
    left  gripper   median stall gap per episode 0.137 rad, fires on 90/90
    control (commanded OPEN)   median gap 0.0009 rad

Two to three orders of magnitude of separation from the open-command control. This is what makes an
offline-RL reward available without a learned success classifier, and therefore without the
reward-hacking failure mode a classifier brings.

WHAT THE REWARD IS. Two events per gripper, both directly observable:

    grasp   a gripper transitions into a sustained stall            +1
    place   a sustained stall ends, and does not resume             +1

A full two-object episode therefore scores 4. Nothing else is rewarded.

WHY NOT A DENSE SHAPED REWARD. "One point per timestep an object is held" is denser and easy to
write, and it is maximised by grasping early and NEVER RELEASING. Rewarding the release event
instead makes the intended behaviour the optimum. Distance-to-target shaping was also rejected: it
needs object positions, which are exactly what proprioception cannot see.

LIMITATION, and it is the important one. Every demonstration succeeds, so this file can be shown to
FIRE on success but not to DISCRIMINATE success from failure -- the negative class does not exist in
demonstration data. That measurement requires policy rollouts, which is the reason to collect them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Commanded gripper levels, from gripper_levels.json over 96,003 demonstration frames. These are
# ACTION values, not observed states: the file's clusters were fitted on what the operator
# commanded, and a position-controlled gripper commanded past an object stalls short of the target.
GRIP_LO = np.array([0.31994176, 0.28772816])
GRIP_HI = np.array([0.86143696, 1.05165064])
GRIP_MID = (GRIP_LO + GRIP_HI) / 2.0

SETTLE = 15          # steps a close command must persist before a gap counts as a stall (0.5 s)
# rad. Chosen from the data, not picked: the commanded-OPEN control sits at 0.0009, and the 11
# demonstrations that a 0.02 cutoff missed have left-gripper gaps of 0.0101-0.0200 -- real holds
# with a softer grip, not noise. 0.010 is 11x the control and recovers all 90 episodes. Every
# demonstration succeeds, so 90/90 detection is the correct answer here; a threshold that misses
# 11 of them is measuring the threshold rather than the grip.
STALL_MIN = 0.010
HOLD_MIN = 30        # steps a stall must last to count as a real hold (1.0 s)


def stall_gap(state, action, n_grip=2, arm_dims=14):
    """Per-gripper (commanded - achieved) while the command says closed, else 0."""
    state = np.asarray(state, float)
    action = np.asarray(action, float)
    out = np.zeros((len(state), n_grip))
    for g in range(n_grip):
        closed = action[:, arm_dims + g] > GRIP_MID[g]
        out[closed, g] = (action[closed, arm_dims + g] - state[closed, arm_dims + g])
    return out


def held_mask(state, action, n_grip=2, arm_dims=14, settle=SETTLE, stall_min=STALL_MIN):
    """True where a gripper is stalled against an object, having been commanded closed long enough.

    The settle window is what separates a stall from a gripper still travelling toward its target:
    an unobstructed gripper closes within a few steps and the gap decays to ~0, while a stall
    persists for as long as the object is held.
    """
    gap = stall_gap(state, action, n_grip, arm_dims)
    action = np.asarray(action, float)
    held = np.zeros_like(gap, bool)
    for g in range(n_grip):
        closed = action[:, arm_dims + g] > GRIP_MID[g]
        run = 0
        for t in range(len(closed)):
            run = run + 1 if closed[t] else 0
            held[t, g] = (run >= settle) and (gap[t, g] > stall_min)
    return held


def episode_events(state, action, n_grip=2, arm_dims=14, hold_min=HOLD_MIN):
    """Grasp and place events per gripper, from the held mask.

    A hold shorter than `hold_min` is not counted: a momentary stall is as likely to be the jaws
    brushing the object as gripping it, and rewarding it would teach the policy to tap rather than
    to grasp.
    """
    held = held_mask(state, action, n_grip, arm_dims)
    events = []
    for g in range(n_grip):
        h = held[:, g].astype(int)
        starts = np.flatnonzero(np.diff(h) == 1) + 1
        ends = np.flatnonzero(np.diff(h) == -1) + 1
        for s in starts:
            e = ends[ends > s]
            e = int(e[0]) if len(e) else len(h)          # still held at episode end
            if e - s < hold_min:
                continue
            events.append(dict(gripper=g, t_grasp=int(s), t_place=int(e) if e < len(h) else None,
                               duration=int(e - s)))
    return events


def episode_reward(state, action, n_grip=2, arm_dims=14, gamma_free=True):
    """Per-timestep reward array. +1 on a grasp event, +1 on a place event, 0 elsewhere."""
    r = np.zeros(len(state))
    for ev in episode_events(state, action, n_grip, arm_dims):
        r[ev["t_grasp"]] += 1.0
        if ev["t_place"] is not None:
            r[ev["t_place"]] += 1.0
    return r


def main():
    ap = argparse.ArgumentParser(description="label reward on a demonstration or rollout set")
    ap.add_argument("--arrays", required=True,
                    help="npz with episode/state/action arrays (see offline_rl/README.md)")
    ap.add_argument("--arm-dims", type=int, default=14)
    ap.add_argument("--n-grip", type=int, default=2)
    ap.add_argument("--out", default=None, help="write per-frame rewards to this npz")
    a = ap.parse_args()

    d = np.load(a.arrays)
    EP, ST, AC = d["episode"], d["state"], d["action"]
    eps = sorted(set(EP.tolist()))
    print(f"{len(eps)} episodes, {len(EP)} frames\n")

    totals, per_ep_r = [], {}
    n_grasp = n_place = 0
    for e in eps:
        m = EP == e
        s, ac = ST[m], AC[m]
        evs = episode_events(s, ac, a.n_grip, a.arm_dims)
        r = episode_reward(s, ac, a.n_grip, a.arm_dims)
        per_ep_r[e] = r
        totals.append(r.sum())
        n_grasp += len(evs)
        n_place += sum(1 for ev in evs if ev["t_place"] is not None)

    t = np.array(totals)
    print(f"reward per episode: mean {t.mean():.2f}  min {t.min():.0f}  max {t.max():.0f}")
    for v in sorted(set(t.tolist())):
        print(f"    {int(v)} points: {int((t == v).sum())} episodes")
    print(f"\ngrasp events {n_grasp}, of which {n_place} were followed by a place")
    print(f"expected for a two-object task, fully successful: 4 points per episode")
    print(f"episodes scoring the full 4: {(t >= 4).sum()}/{len(t)} = {100*(t>=4).mean():.0f}%")

    if a.out:
        np.savez_compressed(a.out, episode=EP,
                            reward=np.concatenate([per_ep_r[e] for e in eps]))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
