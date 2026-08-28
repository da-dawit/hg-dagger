"""Pick a per-camera ROI by hand, the HIL-SERL way, without rebuilding the dataset.

Author: Dawit Chun

lerobot/rl/crop_dataset_roi.py does the same selection but then re-encodes the whole dataset to
128x128. We only want the coordinates: the crop is applied inside GR00T's image transform, so no
video is touched (re-encoding is what put an 8-second offset into episode 1 once).

The wrist cameras MOVE, so a fixed ROI has to cover the workspace for the whole subtask, not one
instant. Each camera is therefore shown as a temporal AVERAGE over the subtask -- ghosting marks
everything the workspace passes through. Draw a box that contains all of it.

    python3 pick_roi.py --subtask 2 --out roi.json
      drag a box   'c' confirm   'r' redo   ESC skip this camera

Needs a display (it opens OpenCV windows).
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/robotis/robot_omy/lerobot/src")
SUB = ["Grab bolt", "Place in hole", "Grab driver", "Screw in", "Home"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--v30", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val_v3.0")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--subtask", type=int, default=-1,
                    help="-1 (default) = blend the WHOLE episode. The ROI is fixed for every "
                         "subtask, so it must cover left-arm work (bolt) AND right-arm work "
                         "(driver); blending one subtask only shows half the workspace.")
    ap.add_argument("--samples", type=int, default=48, help="frames blended into the composite")
    ap.add_argument("--out", default="roi.json")
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="substrings of camera names to (re)draw, e.g. --cameras head. "
                         "Anything not listed keeps its existing box from --out.")
    a = ap.parse_args()

    import cv2, glob, pandas as pd
    from lerobot.rl.crop_dataset_roi import select_rect_roi

    #opencv-python-headless has no GUI backend and cv2.namedWindow dies with an unhelpful
    #"rebuild the library" error. Say so up front instead.
    if "GUI:                           NONE" in cv2.getBuildInformation():
        raise SystemExit(
            "this interpreter has opencv-python-HEADLESS (no GUI). Use conda base:\n"
            "  /home/robotis/miniconda3/bin/python pick_roi.py ...\n"
            "(cv2 5.0.0 there is built with QT5)")

    info = json.loads((Path(a.dataset) / "meta" / "info.json").read_text())
    cams = [k for k in info["features"] if info["features"][k]["dtype"] == "video"]
    v30 = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(str(Path(a.v30) / "data/**/*.parquet"), recursive=True))])
    sub = v30[v30.episode_index == a.episode].sort_values("frame_index").subtask_index.to_numpy()
    if a.subtask < 0:
        idx = np.arange(len(sub)); label = "WHOLE EPISODE (all 5 subtasks)"
    else:
        idx = np.where(sub == a.subtask)[0]; label = f"subtask {a.subtask} '{SUB[a.subtask]}'"
    frames = np.linspace(idx.min(), idx.max(), a.samples).astype(int)
    print(f"episode {a.episode}, {label}: frames {idx.min()}..{idx.max()}, blending {len(frames)}")

    rois = {}
    if Path(a.out).exists():
        rois = json.loads(Path(a.out).read_text())          #keep boxes we are not redrawing
        print(f"merging into existing {a.out} ({len(rois)} camera(s) already set)")
    if a.cameras:
        cams = [c for c in cams if any(t in c for t in a.cameras)]
        print(f"redrawing only: {[c.replace('observation.images.rgb.','') for c in cams]}")
    for c in cams:
        C, H, W = info["features"][c]["shape"]
        vp = os.path.join(a.dataset, info["video_path"].format(
            episode_chunk=0, video_key=c, episode_index=a.episode))
        sel = "+".join(f"eq(n\\,{f})" for f in frames)
        o = subprocess.run(["ffmpeg", "-v", "error", "-i", vp, "-vf", f"select='{sel}'",
                            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True)
        arr = np.frombuffer(o.stdout, np.uint8)
        k = arr.size // (W * H * 3)
        comp = arr[:k * W * H * 3].reshape(k, H, W, 3).astype(np.float32).mean(0).astype(np.uint8)

        short = c.replace("observation.images.rgb.", "")
        print(f"\n--- {short}  {W}x{H}  ({k} frames averaged)")
        print("    drag a box that covers the workspace for the WHOLE subtask; 'c' confirm, 'r' redo, ESC skip")
        cv2.imwrite(f"roi_composite_{short}.png", cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        roi = select_rect_roi(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        if roi is None:
            print(f"    skipped {short}"); continue
        top, left, h, w = roi
        rois[c] = dict(top=int(top), left=int(left), height=int(h), width=int(w), H=int(H), W=int(W))
        print(f"    ROI top={top} left={left} h={h} w={w}   "
              f"({100*h*w/(H*W):.0f}% of the frame)")

    Path(a.out).write_text(json.dumps(rois, indent=2))
    print(f"\nwrote {a.out}")
    print("apply it with:  GR00T_ROI_JSON=%s GR00T_SQUARE_CROP=1 python3 attn_map.py ..." % a.out)


if __name__ == "__main__":
    main()
