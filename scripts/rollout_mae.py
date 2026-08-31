"""Open-loop rollout of a GR00T checkpoint on a HELD-OUT episode, with a per-timestep MAE trace.

Author: Dawit Chun

WHAT THIS MEASURES, AND WHAT IT DOES NOT.
At frame t the policy is given the episode's REAL observation and asked for its 40-step action
chunk; that chunk is compared against the actions the human actually produced over the same 40
frames. That is an open-loop, teacher-forced comparison: the observation always comes from the
human's trajectory, never from where the policy would have driven the arm. It therefore measures
"does the policy agree with the demonstration from states the demonstration visited" and CANNOT
measure error accumulation, which is the failure mode closed-loop control actually suffers from.
A hardware run remains the only evidence of real behaviour.

THE EPISODE MUST BE HELD OUT. lerobot's factory.py:164 holds out the LAST ceil(n*eval_split)
episodes per task, so with 35 episodes and eval_split 0.1 that is 31..34. Episode 0 is training
data AND an outlier (0.3275 rad from the median start); rolling out on it has produced flattering
numbers here before.

The control constants come from spec_sg2.py so this cannot drift from what is deployed.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/robotis/robot_omy/lerobot/src")
sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
sys.path.insert(0, "/home/robotis/robot_aiworker/hil_dagger")

EVAL_EPISODES = [31, 32, 33, 34]          #the held-out split for this dataset


def main():
    import spec_sg2 as spec
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="/home/robotis/robot_aiworker/datasets/screwing35_subtask")
    ap.add_argument("--episode", type=int, default=32)
    ap.add_argument("--execute-steps", type=int, default=spec.EXECUTE_STEPS)
    ap.add_argument("--max-vel", type=float, default=spec.MAX_VEL)
    ap.add_argument("--max-acc", type=float, default=spec.MAX_ACC)
    ap.add_argument("--speed", type=float, default=0.7, help="infer.py's default; hz = FPS*speed")
    ap.add_argument("--out", default="/home/robotis/robot_aiworker/hil_dagger/eval")
    ap.add_argument("--allow-training-episode", action="store_true")
    a = ap.parse_args()

    if a.episode not in EVAL_EPISODES and not a.allow_training_episode:
        raise SystemExit(
            f"episode {a.episode} is TRAINING data. Held-out episodes are {EVAL_EPISODES}.\n"
            f"Rolling out on training data reports a number that means nothing about "
            f"generalisation. Pass --allow-training-episode only to compare the two deliberately.")

    hz = spec.FPS * a.speed
    dt = 1.0 / hz
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import control_math as cm
    from groot_policy import GrootPolicy

    print(f"[data] episode {a.episode} from {a.root}")
    ds = LeRobotDataset(repo_id="local", root=a.root, episodes=[a.episode])
    n = len(ds)
    info = json.loads((Path(a.root) / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    arm = [i for i, nm in enumerate(names) if nm.startswith(("arm_l", "arm_r"))]
    cams = [f"observation.images.rgb.{c}" for c in spec.CAMERA_ORDER]

    print(f"[data] {n} frames | hz {hz:.1f} | execute_steps {a.execute_steps} | "
          f"max_vel {a.max_vel} | max_acc {a.max_acc}")

    gt_action = np.stack([ds[i]["action"].numpy() for i in range(n)])
    gt_state = np.stack([ds[i]["observation.state"].numpy() for i in range(n)])

    print(f"[policy] loading {a.ckpt}")
    policy = GrootPolicy(a.ckpt, camera_keys=cams, robot_joints=names, names_from=a.root)

    H = 40
    starts = list(range(0, n - H, a.execute_steps))
    per_step_mae, per_joint_abs, exec_q, replan_at = [], [], [], []
    q_prev, v_prev = gt_state[0].copy(), np.zeros(len(names))

    for k, t in enumerate(starts):
        item = ds[t]
        imgs = [item[c].permute(1, 2, 0).numpy() for c in cams]     #CHW -> HWC, already [0,1]
        chunk = np.asarray(policy.predict_chunk(imgs, gt_state[t], item["task"]), dtype=np.float64)

        gt = gt_action[t:t + H]
        err = np.abs(chunk[:len(gt), arm] - gt[:, arm])
        per_step_mae.append(err.mean(axis=1))                       #(H,) MAE per horizon step
        per_joint_abs.append(err.mean(axis=0))                      #(len(arm),) per joint
        replan_at.append(len(exec_q))

        #the deployed limiter, so the rendered motion is what the robot would be commanded
        for w in range(min(a.execute_steps, len(chunk))):
            q_prev, v_prev = cm.clamp_step(chunk[w], q_prev, v_prev, dt,
                                           a.max_vel, a.max_acc, spec.ARM_DIMS)
            exec_q.append(q_prev.copy())
        print(f"\r  chunk {k+1}/{len(starts)}  frame {t:>5}  "
              f"MAE {per_step_mae[-1].mean():.4f} rad", end="", flush=True)

    per_step_mae = np.stack(per_step_mae)
    per_joint = np.stack(per_joint_abs).mean(axis=0)
    exec_q = np.stack(exec_q)
    np.savez(out / f"mae_ep{a.episode}.npz", per_step_mae=per_step_mae, per_joint=per_joint,
             exec_q=exec_q, starts=np.array(starts), replan_at=np.array(replan_at),
             gt_action=gt_action, gt_state=gt_state,
             arm_names=np.array([names[i] for i in arm]), hz=hz)
    print(f"\n\n[MAE] overall {per_step_mae.mean():.4f} rad  "
          f"({np.degrees(per_step_mae.mean()):.2f} deg)")
    print(f"[MAE] horizon step 1 {per_step_mae[:, 0].mean():.4f} -> "
          f"step {H} {per_step_mae[:, -1].mean():.4f} rad")
    print(f"[out] {out}/mae_ep{a.episode}.npz")


if __name__ == "__main__":
    main()
