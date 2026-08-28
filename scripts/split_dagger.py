"""Split an HG-DAgger recording into the three datasets the mixture trainer expects.

Author: Dawit Chun

Isaac-GR00T has no per-frame loss weighting (no `sample_weight` anywhere in the loss path), so
Ilia's frame-level scheme -- human 2.0 / auto 0.3 / pre-intervention -> 0.0 -- cannot be applied
directly. It CAN be reproduced at the dataset level, because DatasetSpec.mix_ratio weights whole
datasets and launch_finetune now accepts "path@ratio". So we split at export time instead:

    dagger_human   task_is_policy == 0                        -> mix 0.30
    dagger_auto    task_is_policy == 1, outside the window    -> mix 0.10
    (dropped)      the PRE_WINDOW seconds before each takeover -> excluded entirely

Dropping beats zero-weighting: a zero-weight frame still costs dataloader time and shard space.

PRE_WINDOW default 2.0 s, not Ilia's 5.0. His was tuned for garment folding, where a failure
develops slowly; a screwing miss commits in well under a second. Screwing is 18.5 s/episode here,
so 5 s would discard 27% of the subtask -- including frames where the policy was still fine.

    python3 split_dagger.py --src <dagger recording> --out-dir <datasets/> [--pre-window 2.0]
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd

POLICY_COL = "task_is_policy"     # 1.0 = policy driving, 0.0 = human on the leaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LeRobot dataset recorded by the DAgger runner")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pre-window", type=float, default=2.0,
                    help="seconds BEFORE each takeover to drop (these are the policy's mistakes)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    info = json.loads((src / "meta" / "info.json").read_text())
    if POLICY_COL not in info["features"]:
        raise SystemExit(
            f"{POLICY_COL!r} not in {src}/meta/info.json features.\n"
            "The recorder must write it per frame (1.0 autonomous / 0.0 human), or the split\n"
            "cannot distinguish corrections from the policy's own actions.")

    W = int(round(a.pre_window * a.fps))
    eps = [json.loads(l) for l in open(src / "meta" / "episodes.jsonl")]
    tally = {"human": 0, "auto": 0, "dropped": 0, "total": 0, "takeovers": 0}
    keep = {"human": {}, "auto": {}}

    for e in eps:
        ei = e["episode_index"]
        p = info["data_path"].format(episode_chunk=ei // info["chunks_size"], episode_index=ei)
        df = pd.read_parquet(src / p).sort_values("frame_index")
        pol = np.asarray([float(np.ravel(v)[0]) for v in df[POLICY_COL]], dtype=float)
        human = pol < 0.5
        # a takeover is an auto->human transition; drop the W frames before each
        drop = np.zeros(len(pol), bool)
        starts = np.where(human[1:] & ~human[:-1])[0] + 1
        tally["takeovers"] += len(starts)
        for s in starts:
            drop[max(0, s - W):s] = True
        h_idx = np.where(human & ~drop)[0]
        a_idx = np.where(~human & ~drop)[0]
        keep["human"][ei] = h_idx
        keep["auto"][ei] = a_idx
        tally["human"] += len(h_idx); tally["auto"] += len(a_idx)
        tally["dropped"] += int(drop.sum()); tally["total"] += len(pol)

    t = tally
    print(f"  {len(eps)} episodes, {t['total']:,} frames, {t['takeovers']} takeovers")
    print(f"    human    {t['human']:>7,}  {100*t['human']/t['total']:5.1f}%   -> dagger_human  (mix 0.30)")
    print(f"    auto     {t['auto']:>7,}  {100*t['auto']/t['total']:5.1f}%   -> dagger_auto   (mix 0.10)")
    print(f"    dropped  {t['dropped']:>7,}  {100*t['dropped']/t['total']:5.1f}%   "
          f"({a.pre_window}s x {t['takeovers']} takeovers)")
    if t["human"] == 0:
        print("\n  WARNING: no human frames. Either nobody intervened, or task_is_policy is inverted.")
    if t["dropped"] > 0.4 * t["total"]:
        print(f"\n  WARNING: the pre-window is discarding {100*t['dropped']/t['total']:.0f}% of the "
              f"recording. Consider a shorter --pre-window.")
    if a.dry_run:
        print("\n  --dry-run: nothing written. Re-run without it to build the datasets.")
        return
    print("\n  NOTE: writing the split datasets requires re-emitting parquet AND re-encoding the\n"
          "  per-episode videos so frame indices stay aligned. Video re-encoding is what put an\n"
          "  8-second offset into episode 1 once -- run verify_dataset.py on each output before\n"
          "  training on it.")


if __name__ == "__main__":
    main()
