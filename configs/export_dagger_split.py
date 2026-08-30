"""Write the three-way HG-DAgger split as real LeRobot v2.1 datasets.

Author: Dawit Chun

split_dagger.py computes the split and stops. This writes it.

Isaac-GR00T has no per-frame loss weighting, so Ilia's human-2.0 / auto-0.3 scheme has to be
reproduced at the DATASET level via DatasetSpec.mix_ratio. That means the human frames and the
autonomous frames must become separate datasets on disk.

CONTIGUITY, and why episodes multiply. LeRobot indexes video per episode and the policy consumes
40-frame action chunks, so a kept span must be a CONTIGUOUS run written as its own episode.
Splicing two corrections into one episode would invent a transition that never happened. Each of
the 39 interventions becomes one episode of dagger_human; the gaps between them become episodes
of dagger_auto.

VIDEO IS THE RISK. Frames are cut with ffmpeg's `select` filter on frame index, not by timestamp:
`-ss` seeks to the nearest keyframe and silently shifts everything after it, which is what put an
8-second offset into episode 1 once. After writing, verify_dataset.py's video-alignment check
(check 9) must report 0 disagreements on both outputs before either is trained on.

    python3 export_dagger_split.py --src <dagger15> --out-dir <datasets/> [--pre-window 2.0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_COL = "task_is_policy"
CHUNK = 40          #GR00T N1.7 action horizon; a run shorter than this yields zero training samples


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def cut_video(src: Path, dst: Path, a: int, b: int) -> None:
    """Frames [a, b) of src -> dst, re-encoded. Frame-indexed, never timestamp-seeked."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"select='between(n\\,{a}\\,{b - 1})',setpts=N/FRAME_RATE/TB",
           "-vsync", "0", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-an", str(dst)]
    subprocess.run(cmd, check=True)


def write_split(src: Path, out: Path, spans: dict[int, list[tuple[int, int]]],
                info: dict, tasks: list[dict], modality: dict, video_keys: list[str]) -> int:
    """spans: {source_episode_index: [(a, b), ...]} -> one output episode per span."""
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)

    ep_rows, n_out, n_frames, gidx = [], 0, 0, 0
    for sei in sorted(spans):
        sdf = pd.read_parquet(
            src / info["data_path"].format(episode_chunk=sei // info["chunks_size"],
                                           episode_index=sei)).sort_values("frame_index")
        for (a, b) in spans[sei]:
            seg = sdf.iloc[a:b].copy().reset_index(drop=True)
            seg["episode_index"] = n_out
            seg["frame_index"] = np.arange(len(seg), dtype="int64")
            seg["index"] = np.arange(gidx, gidx + len(seg), dtype="int64")
            seg["timestamp"] = seg["frame_index"] / info["fps"]
            seg = seg.drop(columns=[POLICY_COL])          #the flag is consumed here, not trained on
            seg.to_parquet(out / "data" / "chunk-000" / f"episode_{n_out:06d}.parquet", index=False)
            for vk in video_keys:
                ok = modality["video"][vk].get("original_key", f"observation.images.{vk}")
                cut_video(src / info["video_path"].format(episode_chunk=sei // info["chunks_size"],
                                                          video_key=ok, episode_index=sei),
                          out / info["video_path"].format(episode_chunk=0, video_key=ok,
                                                          episode_index=n_out), a, b)
            tmap = {t["task_index"]: t["task"] for t in tasks}
            ep_rows.append({"episode_index": n_out,
                            "tasks": [tmap[i] for i in sorted(seg.task_index.unique())],
                            "length": len(seg),
                            "source_episode": sei, "source_frames": [int(a), int(b)]})
            gidx += len(seg); n_frames += len(seg); n_out += 1

    ni = dict(info)
    ni.update(total_episodes=n_out, total_frames=n_frames, total_videos=n_out * len(video_keys),
              total_chunks=1, splits={"train": f"0:{n_out}"})
    ni["features"] = {k: v for k, v in info["features"].items() if k != POLICY_COL}
    (out / "meta" / "info.json").write_text(json.dumps(ni, indent=4))
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for r in ep_rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps({"task_index": t["task_index"], "task": t["task"]}) + "\n")
    json.dump(modality, open(out / "meta" / "modality.json", "w"), indent=2)
    shutil.copy(out / "meta" / "modality.json", out / "modality.json")

    #stats must be recomputed on THIS subset -- reusing the parent's q01/q99 would normalise the
    #human frames against a distribution that is 85% policy motion
    big = pd.concat([pd.read_parquet(p) for p in sorted((out / "data" / "chunk-000").glob("*.parquet"))])
    stats = json.loads((src / "meta" / "stats.json").read_text())
    for feat in ("observation.state", "action"):
        M = np.stack(big[feat].to_numpy()).astype(np.float64)
        stats[feat] = dict(q01=np.percentile(M, 1, axis=0).tolist(),
                           q99=np.percentile(M, 99, axis=0).tolist(),
                           mean=M.mean(0).tolist(), std=M.std(0).tolist(),
                           min=M.min(0).tolist(), max=M.max(0).tolist())
    json.dump(stats, open(out / "meta" / "stats.json", "w"))
    return n_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pre-window", type=float, default=2.0)
    ap.add_argument("--min-run", type=int, default=CHUNK)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src, outd = Path(a.src), Path(a.out_dir)
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = [json.loads(l) for l in open(src / "meta" / "tasks.jsonl")]
    modality = json.loads((src / "meta" / "modality.json").read_text())
    vks = list(modality["video"])
    W = int(round(a.pre_window * info["fps"]))

    human_spans, auto_spans, tally = {}, {}, {"h": 0, "a": 0, "drop": 0, "short": 0}
    for e in (json.loads(l) for l in open(src / "meta" / "episodes.jsonl")):
        ei = e["episode_index"]
        df = pd.read_parquet(src / info["data_path"].format(
            episode_chunk=ei // info["chunks_size"], episode_index=ei)).sort_values("frame_index")
        pol = np.asarray([float(np.ravel(v)[0]) for v in df[POLICY_COL]], dtype=float)
        excluded = pol < -0.5
        human = (pol >= -0.5) & (pol < 0.5)
        drop = excluded.copy()
        for s in np.where(human[1:] & ~human[:-1])[0] + 1:
            drop[max(0, s - W):s] = True
        hs = [(x, y) for x, y in runs_of(human & ~drop) if y - x >= a.min_run]
        tally["short"] += len([1 for x, y in runs_of(human & ~drop) if y - x < a.min_run])
        as_ = [(x, y) for x, y in runs_of((pol >= 0.5) & ~drop) if y - x >= a.min_run]
        if hs:
            human_spans[ei] = hs
        if as_:
            auto_spans[ei] = as_
        tally["h"] += sum(y - x for x, y in hs)
        tally["a"] += sum(y - x for x, y in as_)
        tally["drop"] += int(drop.sum())

    nh = sum(len(v) for v in human_spans.values())
    na = sum(len(v) for v in auto_spans.values())
    print(f"  dagger_human   {nh:3d} episodes  {tally['h']:6d} frames  "
          f"{sum(max(0, (y - x) - CHUNK + 1) for v in human_spans.values() for x, y in v):5d} windows")
    print(f"  dagger_auto    {na:3d} episodes  {tally['a']:6d} frames")
    print(f"  dropped                     {tally['drop']:6d} frames "
          f"(-1 pauses + {a.pre_window}s pre-window)")
    if tally["short"]:
        print(f"  {tally['short']} human run(s) shorter than {a.min_run} frames -> discarded "
              f"(they yield zero training windows)")
    if a.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    for name, spans in (("dagger_human", human_spans), ("dagger_auto", auto_spans)):
        n = write_split(src, outd / name, spans, info, tasks, modality, vks)
        print(f"  wrote {outd / name}  ({n} episodes)")
    print("\n  NOW RUN verify_dataset.py --isaac ON BOTH. Video was re-encoded; check 9 "
          "(video alignment) is the one that catches a bad cut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
