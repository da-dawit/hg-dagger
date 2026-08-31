"""Is the drift the POLICY's, or something post-processing does to it?

Author: Dawit Chun

MAE cannot answer this: it is unsigned, so a joint that is consistently 3 deg to one side scores
the same as one that is randomly +/-3 deg. Drift is a SIGNED bias, so that is what this measures.

Three quantities are compared against the human's action at the same frame:
  raw       the chunk exactly as the model emitted it, after the checkpoint's own unnormalize
  clamped   after clamp_step -- the velocity/acceleration limiter that runs on the robot
  state     where the arm actually was, as a reference for what the human did next

If the bias is already in `raw`, post-processing is innocent and the model learned it. If `raw` is
centred and `clamped` is not, the limiter is introducing it.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/robotis/robot_omy/lerobot/src")
sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")


def main():
    import spec_sg2 as spec, control_math as cm
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="/home/robotis/robot_aiworker/datasets/screwing35_subtask")
    ap.add_argument("--episode", type=int, default=32)
    ap.add_argument("--subtask", type=int, default=2)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default="/home/robotis/robot_aiworker/hil_dagger/eval/drift_ep32.npz")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from groot_policy import GrootPolicy

    ds = LeRobotDataset(repo_id="local", root=a.root, episodes=[a.episode])
    n = len(ds)
    info = json.loads((Path(a.root) / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    arm = [i for i, nm in enumerate(names) if nm.startswith(("arm_l", "arm_r"))]
    cams = [f"observation.images.rgb.{c}" for c in spec.CAMERA_ORDER]

    sub = np.array([int(ds[i]["subtask_index"]) for i in range(n)])
    gt_a = np.stack([ds[i]["action"].numpy() for i in range(n)])
    gt_s = np.stack([ds[i]["observation.state"].numpy() for i in range(n)])

    policy = GrootPolicy(a.ckpt, camera_keys=cams, robot_joints=names, names_from=a.root)
    dt = 1.0 / (spec.FPS * 0.7)
    H = 40

    rows = []
    for st in sorted(set(sub.tolist())):
        idx = [t for t in np.where(sub == st)[0] if t + H < n][::a.stride]
        raw_b, cl_b = [], []
        for t in idx:
            item = ds[t]
            imgs = [item[c].permute(1, 2, 0).numpy() for c in cams]
            chunk = np.asarray(policy.predict_chunk(imgs, gt_s[t], item["task"]), dtype=np.float64)
            gt = gt_a[t:t + H]
            raw_b.append((chunk[:len(gt)] - gt)[:, arm])          #SIGNED, raw model output
            #clamp exactly as deployed, starting from the true pose so only the limiter differs
            q, v = gt_s[t].copy(), np.zeros(len(names))
            cq = []
            for w in range(len(gt)):
                q, v = cm.clamp_step(chunk[w], q, v, dt, spec.MAX_VEL, spec.MAX_ACC, spec.ARM_DIMS)
                cq.append(q.copy())
            cl_b.append((np.stack(cq) - gt)[:, arm])
            print(f"\r  subtask {st}  {len(raw_b)}/{len(idx)}", end="", flush=True)
        rows.append((st, np.concatenate(raw_b), np.concatenate(cl_b)))
    print()

    an = [names[i] for i in arm]
    np.savez(a.out, **{f"raw_{st}": r for st, r, _ in rows},
             **{f"cl_{st}": c for st, _, c in rows}, arm_names=np.array(an))

    for st, raw, cl in rows:
        tag = "  <-- driver grab" if st == a.subtask else ""
        print(f"\n=== subtask {st}{tag} ===")
        print(f"  {'joint':<14}{'raw bias':>10}{'raw |err|':>11}{'clamp bias':>12}{'bias/|err|':>12}")
        for j, nm in enumerate(an):
            b, e = np.degrees(raw[:, j].mean()), np.degrees(np.abs(raw[:, j]).mean())
            cb = np.degrees(cl[:, j].mean())
            ratio = abs(b) / e if e > 1e-9 else 0.0
            flag = "  SYSTEMATIC" if ratio > 0.5 else ""
            print(f"  {nm:<14}{b:>+10.2f}{e:>11.2f}{cb:>+12.2f}{ratio:>12.2f}{flag}")


if __name__ == "__main__":
    main()
