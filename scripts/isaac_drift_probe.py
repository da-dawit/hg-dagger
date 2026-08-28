"""Cartesian drift of an ISAAC-trained GR00T checkpoint, per subtask, on a held-out episode.

Author: Dawit Chun

Same measurement as drift_probe.py, different loader. The LeRobot harness cannot read an Isaac
checkpoint (sharded model-0000N-of-0000M + processor_config.json vs LeRobot's single
model.safetensors + policy_preprocessor.json), and Isaac is the right target anyway: Cyclo's
groot-zenoh container -- and the TensorRT path -- are Isaac-based.

WHY CARTESIAN AND SIGNED. MAE is unsigned, so a policy sitting 50 mm to one side scores the same as
one randomly off by 50 mm; only the first shows on the robot as drift. And joint error is
misleading on a 7-DoF arm because many joint solutions reach the same point. So: forward kinematics
on the predicted chunk, then the SIGNED offset of the right wrist, per subtask.

Baseline from LeRobot checkpoint 014400 on this same episode (the broken-label model):
    Cartesian error at the driver grasp   54.0 mm
    systematic fraction |bias|/|error|    ~1.00  (entirely directional)
    natural spread between demonstrations 27 mm
"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

G = "/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T"
sys.path.insert(0, G)
sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")

SUBTASKS = ["Grab bolt", "Place in hole", "Grab driver", "Screw in", "Home"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--v30", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val_v3.0",
                    help="the v3.0 sibling, which still carries subtask_index per frame")
    ap.add_argument("--traj-id", type=int, default=1, help="val ep 1 == original ep 32")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--arm", choices=("l","r"), default="r",
                    help="WHICH ARM. Subtask 0/1 are LEFT-arm work; 2/3 are RIGHT. Measuring the\n"
                         "idle arm reports a small number that means nothing about the phase.")
    a = ap.parse_args()

    import torch, mujoco
    import spec_sg2 as spec
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.eval.open_loop_eval import parse_observation_gr00t, parse_action_gr00t
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=a.ckpt,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    mc = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc,
                                  video_backend="torchcodec")
    traj = loader[a.traj_id]
    akeys = mc["action"].modality_keys

    names = json.loads((Path(a.dataset) / "meta" / "info.json").read_text())["features"]["action"]["names"]
    gt = np.stack(pd.read_parquet(
        Path(a.dataset) / "data" / "chunk-000" / f"episode_{a.traj_id:06d}.parquet"
    ).sort_values("frame_index")["action"].to_numpy())

    #subtask labels live in the v3.0 sibling; the v2.1 conversion drops the column
    v30 = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(str(Path(a.v30) / "data/**/*.parquet"), recursive=True))])
    sub = v30[v30.episode_index == a.traj_id].sort_values("frame_index").subtask_index.to_numpy()

    m = mujoco.MjModel.from_xml_path("/home/robotis/robotis_mujoco_menagerie/robotis_ffw/ffw_sg2.xml")
    d = mujoco.MjData(m)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in spec.MODEL_JOINTS}
    #arm_r_link7, NOT a finger: a finger's position also moves with the gripper opening, which
    #would contaminate the arm's Cartesian error.
    TIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"arm_{a.arm}_link7")

    def fk(v22):
        for j in spec.MODEL_JOINTS:
            d.qpos[qadr[j]] = v22[names.index(j)]
        mujoco.mj_kinematics(m, d)
        return d.xpos[TIP].copy()

    n = min(len(gt), len(sub))
    print(f"[ckpt] {Path(a.ckpt).name}")
    print(f"[data] traj {a.traj_id} ({n} frames), stride {a.stride}, horizon {a.horizon}\n")

    per = {s: [] for s in range(5)}
    jerr = {s: [] for s in range(5)}
    for t in range(0, n - a.horizon, a.stride):
        step = extract_step_data(traj, t, mc, "new_embodiment")
        obs = {}
        for k, v in step.states.items():
            obs[f"state.{k}"] = v
        for k, v in step.images.items():
            obs[f"video.{k}"] = np.array(v)
        obs["annotation.human.primitive_instruction"] = step.text
        raw, _ = policy.get_action(parse_observation_gr00t(obs, mc))
        chunk = parse_action_gr00t(raw)          #open_loop_eval.py:194 does this too
        for j in range(0, a.horizon, 4):
            pred = np.concatenate(
                [np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[j]) for k in akeys], axis=0)
            g = gt[t + j]
            per[int(sub[t + j])].append(fk(pred) - fk(g))
            jerr[int(sub[t + j])].append(np.degrees(np.abs(pred - g)).mean())
        print(f"\r  frame {t}/{n}", end="", flush=True)

    print(f"\n\n=== SIGNED Cartesian offset of the {'LEFT' if a.arm=='l' else 'RIGHT'} wrist, per subtask (mm) ===")
    print(f"  {'subtask':<16}{'dx':>8}{'dy':>8}{'dz':>8}{'|bias|':>9}{'mean|err|':>11}{'systematic':>12}{'joint':>8}")
    for s in range(5):
        if not per[s]:
            continue
        V = 1000 * np.stack(per[s])
        bias = V.mean(axis=0)
        nb = np.linalg.norm(bias)
        me = np.linalg.norm(V, axis=1).mean()
        print(f"  {SUBTASKS[s]:<16}{bias[0]:>+8.1f}{bias[1]:>+8.1f}{bias[2]:>+8.1f}"
              f"{nb:>9.1f}{me:>11.1f}{nb/max(me,1e-9):>12.2f}{np.mean(jerr[s]):>7.2f}°")
    print("\n  'systematic' = |bias|/mean|err|. 1.00 means every prediction is off the same way")
    print("  (that is drift). Near 0 means the error is random scatter.")
    print("  BASELINE 014400 at the driver grasp: 54.0 mm, systematic ~1.00, demo spread 27 mm")


if __name__ == "__main__":
    main()
