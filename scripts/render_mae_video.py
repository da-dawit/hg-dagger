"""Render the rollout as a video: MuJoCo on the left, the MAE trace scrolling on the right.

Author: Dawit Chun

Written because a number alone does not tell you WHERE the policy disagrees with the human. The
MAE trace is aligned to the same frame the 3D view is showing, so a spike is attributable to a
moment in the task rather than to the episode as a whole.

H.264 ("avc1") only -- mp4v produces a file that plays in VLC and in no browser, which has already
cost one round of "I cannot see the video".
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
sys.path.insert(0, "/home/robotis/robot_aiworker/hil_dagger")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2, preview, spec_sg2 as spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3, help="render every Nth executed waypoint")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--title", default="GR00T N1.7-3B vs held-out episode 32")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    exec_q, per_step, hz = d["exec_q"], d["per_step_mae"], float(d["hz"])
    starts, arm_names = d["starts"], [str(x) for x in d["arm_names"]]

    #chunk MAE -> a value per EXECUTED waypoint, so the trace and the 3D view share a clock
    es = int(np.ceil(len(exec_q) / len(per_step)))
    mae_t = np.repeat(per_step.mean(axis=1), es)[:len(exec_q)]

    #MODEL_JOINTS is 16-D; the dataset is 22-D. Map BY NAME, never positionally.
    import json
    info = json.loads(Path("/home/robotis/robot_aiworker/datasets/screwing35_subtask/meta/info.json").read_text())
    names22 = info["features"]["action"]["names"]
    idx16 = [names22.index(j) for j in spec.MODEL_JOINTS]
    q16 = exec_q[:, idx16]

    print(f"[render] {len(exec_q)} waypoints -> every {a.stride} -> {len(exec_q)//a.stride} frames")
    r = preview.replay(q16[::a.stride], fps=hz / a.stride, frames_out=True,
                       width=a.width, height=a.height)
    frames3d = r["frames"]
    mae_s = mae_t[::a.stride][:len(frames3d)]
    print(f"[render] {len(frames3d)} 3D frames; compositing the MAE trace")

    out_frames = []
    x = np.arange(len(mae_s)) * a.stride / hz
    lo, hi = 0.0, float(max(mae_t.max(), 0.12)) * 1.15
    for i in range(len(frames3d)):
        fig, ax = plt.subplots(figsize=(a.width / 100, a.height / 100), dpi=100)
        ax.plot(x, np.degrees(mae_s), lw=1.2, color="#888", label="MAE (all frames)")
        ax.plot(x[:i + 1], np.degrees(mae_s[:i + 1]), lw=2.0, color="#c0392b")
        ax.axvline(x[i], color="#c0392b", lw=1.0, alpha=0.6)
        ax.scatter([x[i]], [np.degrees(mae_s[i])], s=36, color="#c0392b", zorder=5)
        ax.axhline(np.degrees(mae_t.mean()), color="#2980b9", ls="--", lw=1.0,
                   label=f"mean {np.degrees(mae_t.mean()):.2f} deg")
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(np.degrees(lo), np.degrees(hi))
        ax.set_xlabel("time (s)"); ax.set_ylabel("action MAE vs human (deg)")
        ax.set_title(f"t = {x[i]:5.1f}s    MAE = {np.degrees(mae_s[i]):.2f} deg", fontsize=10)
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        if buf.shape[:2] != frames3d[i].shape[:2]:
            buf = cv2.resize(buf, (frames3d[i].shape[1], frames3d[i].shape[0]))
        out_frames.append(np.hstack([frames3d[i], buf]))
        if i % 25 == 0:
            print(f"\r  frame {i}/{len(frames3d)}", end="", flush=True)

    print()
    preview.write_video(out_frames, a.out, fps=hz / a.stride)

    #a static summary alongside the video: where the error lives, per joint and per horizon step
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=110)
    axes[0].plot(np.arange(len(mae_t)) / hz, np.degrees(mae_t), lw=1.0, color="#c0392b")
    axes[0].axhline(np.degrees(mae_t.mean()), color="#2980b9", ls="--",
                    label=f"mean {np.degrees(mae_t.mean()):.2f} deg")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("MAE (deg)")
    axes[0].set_title("MAE over the episode"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(1, per_step.shape[1] + 1), np.degrees(per_step.mean(axis=0)),
                 marker="o", ms=3, color="#8e44ad")
    axes[1].set_xlabel("step within the 40-action chunk"); axes[1].set_ylabel("MAE (deg)")
    axes[1].set_title("MAE vs horizon step"); axes[1].grid(alpha=0.3)

    pj = np.degrees(d["per_joint"])
    order = np.argsort(pj)[::-1]
    axes[2].barh([arm_names[i] for i in order][::-1], pj[order][::-1], color="#16a085")
    axes[2].set_xlabel("MAE (deg)"); axes[2].set_title("MAE by joint"); axes[2].grid(alpha=0.3, axis="x")
    fig.suptitle(a.title, fontsize=12)
    fig.tight_layout()
    png = str(Path(a.out).with_suffix(".png"))
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
