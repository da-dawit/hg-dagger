"""Where does the policy stop, at the GRASP MOMENT, relative to the human?

Author: Dawit Chun

The per-phase average dilutes the thing that decides success. "Grab the orange bolt" is ~7 s of
approach and one instant of closing; a 20 mm average over the whole phase says little about whether
the gripper arrives at the bolt. This measures the last N frames of the phase -- the settle and
close -- and reports the SIGNED offset, because "stops short" is a direction, not a magnitude.

Measures whichever arm actually does the work in that phase: LEFT for subtasks 0/1, RIGHT for 2/3.
An earlier version of this harness measured the right wrist throughout and reported a small,
meaningless number for the left-arm phases.
"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

G = "/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T"
sys.path.insert(0, G); sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
SUB = ["Grab bolt", "Place in hole", "Grab driver", "Screw in", "Home"]
WORKING_ARM = {0: "l", 1: "l", 2: "r", 3: "r", 4: "r"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--v30", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val_v3.0")
    ap.add_argument("--traj-id", type=int, default=1)
    ap.add_argument("--tail", type=int, default=60, help="frames at the end of a phase = the settle+close")
    ap.add_argument("--horizon", type=int, default=16)
    a = ap.parse_args()

    import torch, mujoco, spec_sg2 as spec
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.eval.open_loop_eval import parse_observation_gr00t, parse_action_gr00t
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=a.ckpt,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    mc = policy.get_modality_config(); akeys = mc["action"].modality_keys
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc, video_backend="torchcodec")
    traj = loader[a.traj_id]
    names = json.loads((Path(a.dataset)/"meta"/"info.json").read_text())["features"]["action"]["names"]
    gt = np.stack(pd.read_parquet(Path(a.dataset)/"data"/"chunk-000"/f"episode_{a.traj_id:06d}.parquet"
                                  ).sort_values("frame_index")["action"].to_numpy())
    v30 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(a.v30)/"data/**/*.parquet"), recursive=True))])
    sub = v30[v30.episode_index == a.traj_id].sort_values("frame_index").subtask_index.to_numpy()

    m = mujoco.MjModel.from_xml_path("/home/robotis/robotis_mujoco_menagerie/robotis_ffw/ffw_sg2.xml")
    d = mujoco.MjData(m)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in spec.MODEL_JOINTS}
    TIPS = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"arm_{WORKING_ARM[s]}_link7") for s in range(5)}
    def fk(v, tip):
        for j in spec.MODEL_JOINTS: d.qpos[qadr[j]] = v[names.index(j)]
        mujoco.mj_kinematics(m, d); return d.xpos[tip].copy()

    n = min(len(gt), len(sub))
    # the last `tail` frames of each phase
    windows = {}
    for s in range(5):
        idx = np.where(sub[:n] == s)[0]
        if len(idx): windows[s] = idx[-a.tail:]

    out = {}
    for s, idx in windows.items():
        V = []
        for t in idx[::8]:
            if t + a.horizon >= n: continue
            step = extract_step_data(traj, int(t), mc, "new_embodiment")
            obs = {f"state.{k}": v for k, v in step.states.items()}
            for k, v in step.images.items(): obs[f"video.{k}"] = np.array(v)
            obs["annotation.human.primitive_instruction"] = step.text
            raw, _ = policy.get_action(parse_observation_gr00t(obs, mc))
            ch = parse_action_gr00t(raw)
            pred = np.concatenate([np.atleast_1d(np.atleast_1d(ch[f"action.{k}"])[0]) for k in akeys])
            V.append(fk(pred, TIPS[s]) - fk(gt[t], TIPS[s]))
        if V: out[s] = 1000 * np.stack(V)

    print(f"\n[ckpt] {Path(a.ckpt).name}   last {a.tail} frames of each phase")
    print(f"  {'phase':<16}{'arm':>4}{'dx':>8}{'dy':>8}{'dz':>8}{'|bias|':>9}{'systematic':>12}")
    for s in sorted(out):
        V = out[s]; b = V.mean(axis=0); nb = np.linalg.norm(b)
        me = np.linalg.norm(V, axis=1).mean()
        print(f"  {SUB[s]:<16}{WORKING_ARM[s].upper():>4}{b[0]:>+8.1f}{b[1]:>+8.1f}{b[2]:>+8.1f}"
              f"{nb:>9.1f}{nb/max(me,1e-9):>12.2f}")
    print("  (+x forward, +y left, +z up. Negative dx at the grasp = stops SHORT.)")


if __name__ == "__main__":
    main()
