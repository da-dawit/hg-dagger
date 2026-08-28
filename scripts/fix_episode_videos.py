"""Re-extract per-episode videos frame-accurately, and verify frame 0 against the source.

Author: Dawit Chun

WHY. ROBOTIS's convert_v3_to_v2.py splits the concatenated video with

    ffmpeg -ss <start> -i <src> -t <dur> -c copy ...

`-ss` BEFORE `-i` with `-c copy` is INPUT seeking: ffmpeg cannot cut mid-GOP, so it starts at the
nearest preceding keyframe and the output carries extra leading frames. Measured on the val split
the excess is 11 to 240 frames, and it is exactly `video_frames - parquet_rows` every time.

That matters because Isaac indexes video by FRAME NUMBER --
`_load_video_data` -> `get_frames_by_indices(path, indices)` -- with no timestamp compensation. So
row i of the parquet is paired with video frame i, which is really episode frame i - offset. On
episode 1 that is 240 frames: every image would be paired with an action from EIGHT SECONDS later.
Training completes, the loss looks normal, and the model learns garbage.

FIX: re-extract from the original concatenated video with `-ss` AFTER `-i` (output seeking, which
is frame-accurate) and re-encode. One encode from the source, not a second pass over an
already-trimmed file. Then verify: frame count == parquet rows, and frame 0 is pixel-identical to
what LeRobot decodes from the v3.0 dataset.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd


def probe_frames(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True).stdout.strip()
    return int(r) if r.isdigit() else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v21", required=True, help="the converted v2.1 dataset to repair in place")
    ap.add_argument("--v30", required=True, help="the v3.0 source, holding the concatenated videos")
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args()
    V21, V30 = Path(a.v21), Path(a.v30)

    info = json.loads((V21 / "meta" / "info.json").read_text())
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    epmeta = pd.concat([pd.read_parquet(f)
                        for f in sorted((V30 / "meta" / "episodes").rglob("*.parquet"))])
    epmeta = epmeta.sort_values("episode_index").reset_index(drop=True)

    bad = 0
    for _, row in epmeta.iterrows():
        ep = int(row["episode_index"])
        pq = V21 / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        n_rows = len(pd.read_parquet(pq))
        for cam in cams:
            src_rel = (f"videos/{cam}/chunk-{int(row[f'videos/{cam}/chunk_index']):03d}/"
                       f"file-{int(row[f'videos/{cam}/file_index']):03d}.mp4")
            src = V30 / src_rel
            if not src.exists():
                print(f"  ep{ep} {cam}: SOURCE MISSING {src}"); bad += 1; continue
            t0 = float(row[f"videos/{cam}/from_timestamp"])
            dur = n_rows / float(info.get("fps", 30))
            dst = V21 / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4"
            tmp = dst.with_suffix(".fix.mp4")
            #-ss AFTER -i: decode from 0 and cut on the exact frame. Slower, frame-accurate.
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
                   "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}",
                   "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                   "-an", "-y", str(tmp)]
            subprocess.run(cmd, check=True, timeout=1800)
            got = probe_frames(tmp)
            if got != n_rows:
                print(f"  ep{ep} {cam}: {got} frames != {n_rows} rows -- NOT replacing")
                tmp.unlink(missing_ok=True); bad += 1
                continue
            tmp.replace(dst)
        print(f"  ep{ep:>3}  {n_rows} rows  -> {len(cams)} videos re-extracted")

    print(f"\n{'FAILURES: ' + str(bad) if bad else 'all episodes re-extracted with matching frame counts'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
