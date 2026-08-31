"""Does the deployed control loop fall BEHIND the demonstration? State-feedback closed loop.

Author: Dawit Chun

isaac_drift_probe.py is open-loop: every chunk is predicted from the HUMAN's observation, so it
measures agreement with the demonstration and structurally cannot show error accumulating. The
operator reports the arm "stops short of grabbing", which is exactly what accumulation looks like
and exactly what that probe cannot detect.

This closes the STATE half of the loop: the policy is fed the state the control loop actually
reached, not the human's. Images still come from the demonstration -- we have no simulator - so
this isolates the CONTROL path (clamp + receding horizon + lead time) from visual covariate shift.
If the commanded wrist falls progressively behind the demonstrated one here, the cause is in the
control loop and is fixable without more data.
"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

G = "/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T"
sys.path.insert(0, G); sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
SUBTASKS = ["Grab bolt", "Place in hole", "Grab driver", "Screw in", "Home"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--v30", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val_v3.0")
    ap.add_argument("--traj-id", type=int, default=1)
    ap.add_argument("--execute-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--hz", type=float, default=21.0, help="infer.py default is FPS*speed = 30*0.7")
    ap.add_argument("--max-vel", type=float, default=None)
    ap.add_argument("--max-acc", type=float, default=None)
    ap.add_argument("--no-clamp", action="store_true", help="bypass the rate limiter entirely")
    a = ap.parse_args()

    import torch, mujoco, control_math as cm, spec_sg2 as spec
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.eval.open_loop_eval import parse_observation_gr00t, parse_action_gr00t
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    mv = a.max_vel if a.max_vel is not None else spec.MAX_VEL
    ma = a.max_acc if a.max_acc is not None else spec.MAX_ACC
    dt = 1.0 / a.hz

    policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=a.ckpt,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    mc = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc,
                                  video_backend="torchcodec")
    traj = loader[a.traj_id]
    akeys = mc["action"].modality_keys
    names = json.loads((Path(a.dataset)/"meta"/"info.json").read_text())["features"]["action"]["names"]
    gt = np.stack(pd.read_parquet(Path(a.dataset)/"data"/"chunk-000"/f"episode_{a.traj_id:06d}.parquet"
                                  ).sort_values("frame_index")["action"].to_numpy())
    v30 = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(str(Path(a.v30)/"data/**/*.parquet"), recursive=True))])
    sub = v30[v30.episode_index == a.traj_id].sort_values("frame_index").subtask_index.to_numpy()

    m = mujoco.MjModel.from_xml_path("/home/robotis/robotis_mujoco_menagerie/robotis_ffw/ffw_sg2.xml")
    d = mujoco.MjData(m)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in spec.MODEL_JOINTS}
    TIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_r_link7")
    def fk(v):
        for j in spec.MODEL_JOINTS: d.qpos[qadr[j]] = v[names.index(j)]
        mujoco.mj_kinematics(m, d); return d.xpos[TIP].copy()

    n = min(len(gt), len(sub))
    q = gt[0].copy()                    # commanded state, fed back
    v = np.zeros(len(names))
    lag, t = {s: [] for s in range(5)}, 0
    print(f"[ckpt] {Path(a.ckpt).name}   execute={a.execute_steps}/{a.horizon}  hz={a.hz}  "
          f"clamp={'OFF' if a.no_clamp else f'vel {mv} acc {ma}'}")
    while t < n - a.horizon:
        step = extract_step_data(traj, t, mc, "new_embodiment")
        obs = {f"state.{k}": val for k, val in step.states.items()}
        for k, val in step.images.items(): obs[f"video.{k}"] = np.array(val)
        obs["annotation.human.primitive_instruction"] = step.text
        # STATE FEEDBACK: overwrite the human's state with the one we actually reached
        for k in list(obs):
            if k.startswith("state."):
                key = k.split(".", 1)[1]
                sl = mc["state"].modality_keys
                if key in sl:
                    lo = sum({"arm_left":8,"arm_right":8,"head":2,"lift":1,"odometry":3}[x]
                             for x in sl[:sl.index(key)])
                    w = {"arm_left":8,"arm_right":8,"head":2,"lift":1,"odometry":3}[key]
                    obs[k] = q[lo:lo+w][None, :].astype(np.float32)
        raw, _ = policy.get_action(parse_observation_gr00t(obs, mc))
        chunk = parse_action_gr00t(raw)
        for j in range(min(a.execute_steps, a.horizon)):
            tgt = np.concatenate([np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[j]) for k in akeys])
            if a.no_clamp:
                q = tgt.copy()
            else:
                q, v = cm.clamp_step(tgt, q, v, dt, mv, ma, spec.ARM_DIMS)
            if t + j < n:
                lag[int(sub[t+j])].append(np.linalg.norm(fk(q) - fk(gt[t+j])) * 1000)
        t += a.execute_steps
        print(f"\r  frame {t}/{n}", end="", flush=True)

    print("\n\n=== how far the COMMANDED wrist is from the demonstrated one (mm) ===")
    print(f"  {'subtask':<16}{'mean':>9}{'end of phase':>15}")
    for s in range(5):
        if lag[s]:
            L = np.array(lag[s])
            print(f"  {SUBTASKS[s]:<16}{L.mean():>9.1f}{L[-20:].mean():>15.1f}")
    allL = np.concatenate([np.array(lag[s]) for s in range(5) if lag[s]])
    print(f"\n  overall {allL.mean():.1f} mm   |   open-loop reference was ~21 mm at the driver grasp")
    print("  A number that GROWS across phases means the control loop is falling behind.")


if __name__ == "__main__":
    main()
