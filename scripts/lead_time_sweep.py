"""Would follower_state[t+k] be a better action label than the leader's raw joint position?

Author: Dawit Chun

Pure array indexing over the existing dataset -- no GPU, no retrain, no robot.

WHAT IS WRONG WITH THE CURRENT LABEL. `action` is the LEADER's absolute joint position. The leader
freeze lets the operator lock the follower and move their own hand freely; after each freeze the two
sit at different joint configurations PERMANENTLY, because the teleop re-references rather than
snapping the follower. Measured on episode 32 that gap is 0.3 mm before the first freeze and 157 mm
by the driver grasp. The model learns the leader's frame faithfully and we then command the
FOLLOWER with it.

WHAT WE WANT INSTEAD. action[t] = follower_state[t+k]: "be here k frames from now". Same frame we
command, so a freeze cannot desynchronise it, and during a freeze it correctly reads "stay put".

THREE THINGS DECIDE k:
  1. it must be REACHABLE          -- true by construction; the leader label is not
  2. it must LEAD enough to drive  -- k too small and the target is the current pose
  3. it must stay inside the clamps -- MAX_VEL 0.6, MAX_ACC 2.0, or the limiter rewrites it anyway
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
import spec_sg2 as spec

R = "/home/robotis/robot_aiworker/datasets/screwing35_subtask"
info = json.load(open(f"{R}/meta/info.json"))
FPS = info["fps"]; DT = 1.0 / FPS
names = info["features"]["observation.state"]["names"]
ARM = [i for i, n in enumerate(names) if n.startswith(("arm_l", "arm_r"))]

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/data/**/*.parquet", recursive=True))])
eps = []
for e, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    eps.append((int(e),
                np.stack(g["observation.state"].to_numpy())[:, ARM],
                np.stack(g["action"].to_numpy())[:, ARM],
                g.subtask_index.to_numpy()))

def frozen_mask(S, A):
    """follower standing still while the leader moves -- the freeze/clutch signature."""
    f = np.r_[0, np.abs(np.diff(S, axis=0)).max(axis=1) * FPS]
    l = np.r_[0, np.abs(np.diff(A, axis=0)).max(axis=1) * FPS]
    return (f < 0.02) & (l > 0.05)

print("=== 0. natural leader->follower lag, measured BEFORE any freeze (subtask 0) ===")
best = []
for e, S, A, sub in eps:
    m = sub == 0
    if m.sum() < 60: continue
    s, a = S[m], A[m]
    err = [np.abs(s[k:] - a[:len(a) - k if k else None]).mean() for k in range(0, 16)]
    best.append(int(np.argmin(err)))
print(f"  per-episode best lag: median {int(np.median(best))} frames "
      f"({int(np.median(best))/FPS*1000:.0f} ms), mean {np.mean(best):.1f}")

print("\n=== 1. is the CURRENT label reachable? does the follower ever arrive at leader[t]? ===")
res = []
for e, S, A, sub in eps:
    d = [np.abs(S[t:t+30] - A[t]).max() for t in range(0, len(A) - 30, 25)]
    res.append(np.degrees(np.mean(d)))
print(f"  closest the follower gets to the leader's target within 1s: "
      f"{np.mean(res):.2f} deg (mean over 35 episodes)")
print("  -> a target the robot never reaches. follower[t+k] is reachable by definition.")

print("\n=== 2. sweep k ===")
print(f"  {'k':>3}{'lead (ms)':>11}{'motion NORMAL':>15}{'motion FROZEN':>15}"
      f"{'>MAX_VEL':>10}{'>MAX_ACC':>10}")
for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15):
    mv = ma = tot = 0
    nrm, frz = [], []
    for e, S, A, sub in eps:
        fz = frozen_mask(S, A)
        tgt = S[k:]                       #action[t] = follower[t+k]
        cur = S[:len(S) - k]
        step = np.abs(tgt - cur)
        v = step / (k * DT)
        acc = np.abs(np.diff(v, axis=0)) / DT
        m = fz[:len(cur)]
        nrm.append(np.degrees(step[~m]).mean()); frz.append(np.degrees(step[m]).mean() if m.any() else np.nan)
        mv += (v > spec.MAX_VEL).sum(); ma += (acc > spec.MAX_ACC).sum(); tot += v.size
    print(f"  {k:>3}{k/FPS*1000:>11.0f}{np.nanmean(nrm):>14.3f}d{np.nanmean(frz):>14.3f}d"
          f"{100*mv/tot:>9.2f}%{100*ma/tot:>9.2f}%")
print("\n  'motion NORMAL' = how far the target sits ahead of the current pose during normal teleop")
print("  'motion FROZEN' = the same during a freeze; MUST be ~0, i.e. the label says 'hold still'")

print("\n=== 3. what the CURRENT leader label does in those same two regimes ===")
nrm, frz = [], []
for e, S, A, sub in eps:
    fz = frozen_mask(S, A); step = np.abs(A - S)
    nrm.append(np.degrees(step[~fz]).mean()); frz.append(np.degrees(step[fz]).mean() if fz.any() else np.nan)
print(f"  leader:  normal {np.nanmean(nrm):.3f} deg   FROZEN {np.nanmean(frz):.3f} deg")
print("  -> during a freeze the leader label commands real motion the robot must not make.")
