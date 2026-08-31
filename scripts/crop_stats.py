#!/usr/bin/env python3
"""Measure what a square crop would keep or discard, per camera.

Produces the numbers quoted in the image-preparation section of the manual.
Run it again whenever the cell is rearranged or a camera is remounted; the
values are properties of one scene, not constants.

Two metrics, and they are not interchangeable:

  motion energy   sum of |frame - previous frame|, collapsed onto one axis.
                  Valid only for a FIXED camera, where frame-to-frame change
                  is task activity. On a wrist camera the background sweeps
                  past the lens and dominates, so the number means nothing.

  spatial detail  sum of gradient magnitude, collapsed onto one axis. Valid
                  for a moving camera, since it is computed per frame and
                  does not depend on what the camera did between frames.

Usage:
    python3 crop_stats.py --dataset <lerobot root> [--episode 0]
                          [--span 1827 159] [--stride 4]
"""
import argparse, glob, os, sys
import numpy as np
import cv2


def video_paths(root, episode):
    """Every camera stream for one episode, keyed by camera name."""
    pat = os.path.join(root, "videos", "**",
                       f"episode_{episode:06d}.mp4")
    out = {}
    for p in glob.glob(pat, recursive=True):
        #.../observation.images.rgb.cam_left_head/episode_000000.mp4
        cam = os.path.basename(os.path.dirname(p)).split(".")[-1]
        out[cam] = p
    return dict(sorted(out.items()))


def accumulate(path, axis, start, count, stride, mode):
    """Collapse a per-frame measure onto one axis, summed over the episode.

    axis 0 keeps columns, axis 1 keeps rows.
    """
    cap = cv2.VideoCapture(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev, acc, k, used = None, None, 0, 0
    while k < count:
        ok, f = cap.read()
        if not ok:
            break
        if k % stride == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if mode == "motion":
                if prev is not None:
                    d = np.abs(g - prev).sum(axis=axis)
                    acc = d if acc is None else acc + d
                    used += 1
                prev = g
            else:
                gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                d = np.hypot(gx, gy).sum(axis=axis)
                acc = d if acc is None else acc + d
                used += 1
        k += 1
    cap.release()
    if acc is None:
        raise RuntimeError(f"no frames read from {path}")
    return acc, used


def report(cam, path, start, count, stride):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n{cam}  {w}x{h}, {n} frames")
    if w == h:
        print("  already square; nothing to decide")
        return

    if w > h:
        #landscape: a centred square crop trims the sides
        lo = (w - h) // 2
        hi = lo + h
        pad = (w - h) / w
        m, used = accumulate(path, 0, start, count, stride, "motion")
        out = (m[:lo].sum() + m[hi:].sum()) / m.sum()
        d, _ = accumulate(path, 0, start, count, stride, "detail")
        out_d = (d[:lo].sum() + d[hi:].sum()) / d.sum()
        print(f"  letterbox to {w}x{w}: {pad:.1%} padding")
        print(f"  centred square crop keeps columns {lo}..{hi}, "
              f"discarding {(w - h) / w:.1%} of the width")
        print(f"    motion energy discarded : {out:.1%}   ({used} frames)")
        print(f"    spatial detail discarded: {out_d:.1%}")
        print("  motion energy is the metric to use if this camera is fixed")
    else:
        #portrait: a bottom-anchored square crop trims the top
        cut = h - w
        pad = (h - w) / h
        d, used = accumulate(path, 1, start, count, stride, "detail")
        out_d = d[:cut].sum() / d.sum()
        print(f"  letterbox to {h}x{h}: {pad:.1%} padding")
        print(f"  bottom-anchored crop keeps rows {cut}..{h}, "
              f"discarding {cut / h:.1%} of the frame")
        print(f"    spatial detail discarded: {out_d:.1%}   ({used} frames)")
        print("  motion energy is NOT reported: if this camera moves with the")
        print("  arm, frame-to-frame change is mostly background sweeping past")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot dataset root")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--span", type=int, nargs=2, metavar=("START", "COUNT"),
                    help="restrict to a frame range, e.g. one intervention")
    ap.add_argument("--stride", type=int, default=4,
                    help="sample every Nth frame (default 4)")
    a = ap.parse_args()

    paths = video_paths(a.dataset, a.episode)
    if not paths:
        sys.exit(f"no episode {a.episode} videos under {a.dataset}/videos")

    start, count = (a.span if a.span else (0, 1 << 30))
    scope = (f"frames {start}..{start + count}" if a.span else "whole episode")
    print(f"{a.dataset}  episode {a.episode}  {scope}  stride {a.stride}")

    for cam, p in paths.items():
        report(cam, p, start, count, a.stride)

    print("\nThese are measurements of one episode in one scene. Quote them "
          "with the scene, or regenerate them.")


if __name__ == "__main__":
    main()
