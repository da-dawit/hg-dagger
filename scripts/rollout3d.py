"""3-D Cartesian rollout: where the policy's gripper actually goes, against the human's.

Author: Dawit Chun

Joint-space error is not the question. A 7-DoF arm reaches the same point many ways, so a policy
can be several degrees off per joint and land in exactly the right place -- or be a degree off and
miss by 4 cm. This runs the policy open-loop over a held-out episode, pushes BOTH the prediction
and the ground truth through forward kinematics, and plots the two end-effector paths in the
robot's own frame.

Reads the subtask labels, so the screwing phase can be looked at on its own -- that is the part
that still fails on hardware, and a Cartesian plot says whether the policy stops short, overshoots,
or arrives and does the wrong thing once there.

    python3 rollout3d.py --ckpt <checkpoint> --dataset <val set> --episode 1 [--subtask 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G = "/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T"
MJ = "/home/robotis/robotis_mujoco_menagerie/robotis_ffw/ffw_sg2.xml"
SUB = ["Grab bolt (L)", "Place in hole (L)", "Grab driver (R)", "Screw in (R)", "Home"]
COL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
WORKING = {0: "l", 1: "l", 2: "r", 3: "r", 4: "r"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--stride", type=int, default=6, help="frames between policy queries")
    ap.add_argument("--subtask", type=int, default=-1, help="-1 = whole episode")
    ap.add_argument("--out", default="rollout3d.png")
    ap.add_argument("--spin", action="store_true", help="also write a rotating mp4")
    ap.add_argument("--dump", default=None,
                    help="write the human path, the policy path and the per-point "
                         "subtask index to this .npz, so the figure can be redrawn "
                         "without running inference again")
    a = ap.parse_args()

    sys.path.insert(0, G)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.eval.open_loop_eval import parse_observation_gr00t, parse_action_gr00t

    ds = Path(a.dataset)
    info = json.loads((ds / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    df = pd.read_parquet(next(iter(sorted(glob.glob(
        str(ds / "data" / "**" / f"episode_{a.episode:06d}.parquet"), recursive=True))))
    ).sort_values("frame_index")
    gt = np.stack(df["action"].to_numpy())
    sub = (df["subtask_index"].to_numpy() if "subtask_index" in df.columns
           else np.zeros(len(df), int))

    m = mujoco.MjModel.from_xml_path(MJ)
    d = mujoco.MjData(m)
    import spec_sg2 as spec  # noqa: E402  (needs aiworker_deploy on the path)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in spec.MODEL_JOINTS}
    TIP = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"arm_{WORKING[s]}_link7")
           for s in range(5)}

    def fk(vec, tip):
        for j in spec.MODEL_JOINTS:
            d.qpos[qadr[j]] = vec[names.index(j)]
        mujoco.mj_kinematics(m, d)
        return d.xpos[tip].copy()

    pol = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=a.ckpt,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    mc = pol.get_modality_config()
    akeys = mc["action"].modality_keys
    loader = LeRobotEpisodeLoader(dataset_path=str(ds), modality_configs=mc,
                                  video_backend="torchcodec")
    traj = loader[a.episode]

    #extract_step_data reads t .. t+action_horizon, so the last horizon frames of an episode
    #cannot be queried. grasp_bias.py guards this the same way.
    horizon = int(json.loads((Path(a.ckpt) / "config.json").read_text()).get("action_horizon", 40))
    idx = np.arange(0, max(0, len(df) - horizon - 1), a.stride)
    if a.subtask >= 0:
        idx = idx[sub[idx] == a.subtask]
    P, H, S = [], [], []
    for t in idx:
        step = extract_step_data(traj, int(t), mc, "new_embodiment")
        obs = {f"state.{k}": v for k, v in step.states.items()}
        for k, v in step.images.items():
            obs[f"video.{k}"] = np.array(v)
        obs["annotation.human.primitive_instruction"] = step.text
        raw, _ = pol.get_action(parse_observation_gr00t(obs, mc))
        ch = parse_action_gr00t(raw)
        pred = np.concatenate([np.atleast_2d(ch[f"action.{k}"])[0] for k in akeys])
        s = int(sub[t])
        full = np.array(gt[t], dtype=float)
        full_pred = full.copy()
        full_pred[:len(pred)] = pred          # the 16 declared dims; the rest stay at ground truth
        P.append(fk(full_pred, TIP[s])); H.append(fk(full, TIP[s])); S.append(s)
        print(f"\r  {len(P)}/{len(idx)}", end="", flush=True)
    P, H, S = np.array(P) * 1000, np.array(H) * 1000, np.array(S)
    print()

    err = np.linalg.norm(P - H, axis=1)
    print(f"\n  episode {a.episode}, {len(P)} queries")
    print(f"  {'phase':<18}{'n':>5}{'mean mm':>10}{'max mm':>9}{'dx':>8}{'dy':>8}{'dz':>8}")
    for s in sorted(set(S.tolist())):
        k = S == s
        b = (P[k] - H[k]).mean(axis=0)
        print(f"  {SUB[s]:<18}{k.sum():>5}{err[k].mean():>10.1f}{err[k].max():>9.1f}"
              f"{b[0]:>+8.1f}{b[1]:>+8.1f}{b[2]:>+8.1f}")
    print("  (+x forward, +y left, +z up; negative dx at a grasp = stops SHORT)")

    if a.dump:
        np.savez_compressed(a.dump, human=H, policy=P, subtask=S, err=err)
        print(f"  dumped paths to {a.dump}")

    fig = plt.figure(figsize=(15.4, 6.2))
    for n, (el, az, ttl) in enumerate([(22, -60, "perspective"), (90, -90, "top-down (x-y)"),
                                       (0, -90, "side (x-z)")]):
        ax = fig.add_subplot(1, 3, n + 1, projection="3d")
        ax.plot(*H.T, color="0.25", lw=2.4, label="human (ground truth)", zorder=3)
        for s in sorted(set(S.tolist())):
            k = S == s
            ax.plot(*P[k].T, color=COL[s], lw=2.0, label=f"policy · {SUB[s]}", zorder=4)
        for i in range(0, len(P), max(1, len(P) // 45)):   # error whiskers
            ax.plot(*np.stack([H[i], P[i]]).T, color="#C44E52", lw=0.7, alpha=0.55, zorder=2)
        ax.scatter(*H[0], c="k", s=45, marker="o", zorder=5)
        ax.scatter(*H[-1], c="k", s=55, marker="X", zorder=5)
        ax.view_init(elev=el, azim=az)
        ax.set_title(ttl, fontsize=10)
        #labelpad: matplotlib places an axis label relative to the axis line,
        #not to the tick text beside it, so on the edge-on views the label lands
        #on top of its own numbers.
        ax.set_xlabel("x (mm)", labelpad=10)
        ax.set_ylabel("y (mm)", labelpad=10 if n != 1 else 14)
        ax.set_zlabel("z (mm)", labelpad=10 if n != 2 else 18)
        #Panels 2 and 3 are edge-on views, so one axis collapses to a line and
        #its tick labels pile up on top of each other in the corner. Drop the
        #axis that is degenerate in each view; the other two carry the plot.
        if n == 1:                                  #top-down: z is edge-on
            ax.set_zticks([]); ax.set_zlabel("")
        elif n == 2:                                #side: y is edge-on
            ax.set_yticks([]); ax.set_ylabel("")
        #zoom < 1 shrinks the drawn box inside the axes rectangle. Without it
        #the outermost z tick label is cut in half by the axes edge, which
        #widening the figure does not fix because tight_layout repacks it.
        ax.set_box_aspect(None, zoom=0.84)
        allpts = np.vstack([H, P])
        c, r = allpts.mean(0), (allpts.max(0) - allpts.min(0)).max() / 2 + 10
        for setter, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setter(c[ci] - r, c[ci] + r)
        if n == 0:
            ax.legend(fontsize=7, loc="upper left")
    ph = "whole episode" if a.subtask < 0 else SUB[a.subtask]
    fig.suptitle(f"{Path(a.ckpt).name} · episode {a.episode} · {ph} · "
                 f"mean {err.mean():.1f} mm", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=145)
    print(f"\n  wrote {a.out}")

    if a.spin:
        import matplotlib.animation as anim
        f2 = plt.figure(figsize=(7, 6.5)); ax = f2.add_subplot(111, projection="3d")
        ax.plot(*H.T, color="0.25", lw=2.4, label="human")
        for s in sorted(set(S.tolist())):
            k = S == s
            ax.plot(*P[k].T, color=COL[s], lw=2.0, label=SUB[s])
        allpts = np.vstack([H, P]); c, r = allpts.mean(0), (allpts.max(0) - allpts.min(0)).max() / 2 + 10
        for setter, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setter(c[ci] - r, c[ci] + r)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.legend(fontsize=8, loc="upper left")
        mp4 = str(Path(a.out).with_suffix(".mp4"))
        anim.FuncAnimation(f2, lambda i: ax.view_init(elev=20, azim=i * 3),
                           frames=120, interval=50).save(mp4, writer="ffmpeg", dpi=130)
        print(f"  wrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
