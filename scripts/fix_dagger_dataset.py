"""Make an HG-DAgger recording trainable by the 16D screwing checkpoint.

Author: Dawit Chun

The recorder writes a dataset that LOOKS clean -- 13 episodes, v2.1 layout, subtask_index varying
with real per-frame counts -- and is still not trainable. Four things are wrong, and only the first
one is dangerous, because it fails silently and produces a converging run on wrong labels.

  1  LANGUAGE.  modality.json maps annotation.human.primitive_instruction to a column, and
     lerobot_episode_loader.py:380 resolves it as tasks_map[column]. The recorder writes
     task_index = 0 for every frame and a single generic line in tasks.jsonl, so all 43,686
     frames read "Screw the orange bolt into the hole using the driver." The base checkpoint was
     trained on FIVE per-phase instructions. The phase is already in the data as subtask_index --
     nothing maps it to language. Fixed by task_index <- subtask_index plus the five-line
     tasks.jsonl the base run used, verbatim.

     LANG_KEYS = ["task", "sub_task"] (loader:65) is a DIFFERENT path that reads episodes.jsonl.
     Our key is annotation.human.primitive_instruction, which is not in it, so the
     subtask_instructions field in episodes.jsonl is NOT what the model sees. Do not "fix" that
     field and assume it took.

  2  MODALITY.  No modality.json at all; Isaac cannot train without one. The recording also has
     FOUR cameras and the checkpoint knows three -- cam_right_head has never been seen by this
     model and must not be declared.

  3  FRAME_INDEX.  Absent. Standard LeRobot v2.1 column; its absence breaks tooling that sorts by
     it, verify_dataset.py included.

  4  STATS.  meta/stats.json carries only max/mean/min/std. GR00T normalises on q01/q99 -- the
     exact fields whose degenerate range wrecked the first 30k run. Recomputed here from the data.

Idempotent: safe to re-run. Backs up whatever it overwrites and never deletes.

    python3 fix_dagger_dataset.py --root <dataset> --reference <the dataset the ckpt trained on>
"""
import argparse
import glob
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

REF_DEFAULT = "/home/robotis/robot_aiworker/datasets/screwing35_follower_val"


def _backup(p: Path) -> None:
    """Copy aside once per run, never overwrite an existing .orig."""
    if p.exists() and not p.with_suffix(p.suffix + ".orig").exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".orig"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the DAgger dataset to repair, in place")
    ap.add_argument("--reference", default=REF_DEFAULT,
                    help="dataset the base checkpoint trained on; supplies tasks.jsonl + modality.json")
    a = ap.parse_args()
    root, ref = Path(a.root), Path(a.reference)

    for p in (root / "meta" / "info.json", ref / "meta" / "tasks.jsonl", ref / "meta" / "modality.json"):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    parquets = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not parquets:
        print(f"no parquets under {root}/data")
        return 1
    print(f"{root.name}: {len(parquets)} episodes")

    #-- 1. language -------------------------------------------------------------------------
    tasks = [json.loads(l) for l in open(ref / "meta" / "tasks.jsonl")]
    tmap = {t["task_index"]: t["task"] for t in tasks}
    _backup(root / "meta" / "tasks.jsonl")
    with open(root / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps({"task_index": t["task_index"], "task": t["task"]}) + "\n")
    print(f"  tasks.jsonl   <- {len(tasks)} per-phase instructions from {ref.name}")

    #-- 1b + 3. task_index and frame_index --------------------------------------------------
    ref_cols = list(pd.read_parquet(
        sorted(glob.glob(str(ref / "data" / "**" / "*.parquet"), recursive=True))[0]).columns)
    backup_dir = root / f"data_orig_{time.strftime('%Y%m%d-%H%M%S')}"
    rows, relabelled, framed = [], 0, 0
    for f in parquets:
        df = pd.read_parquet(f)
        if "subtask_index" not in df.columns:
            print("  !! no subtask_index -- cannot recover the phase. Stopping.")
            return 1
        need_task = not df["task_index"].equals(df["subtask_index"].astype("int64"))
        need_frame = "frame_index" not in df.columns
        #back up only when something is actually about to change, or a no-op re-run
        #quietly fills the disk with identical copies
        if (need_task or need_frame) and not backup_dir.exists():
            shutil.copytree(root / "data", backup_dir)
        if need_task:
            df["task_index"] = df["subtask_index"].astype("int64")
            relabelled += 1
        if need_frame:
            df.insert(1, "frame_index", range(len(df)))
            framed += 1
        if need_task or need_frame:
            df["frame_index"] = df["frame_index"].astype("int64")
            df = df[[c for c in ref_cols if c in df.columns]
                    + [c for c in df.columns if c not in ref_cols]]
            df.to_parquet(f, index=False)
        rows.append({"episode_index": int(df.episode_index.iloc[0]),
                     "tasks": [tmap[i] for i in sorted(df.task_index.unique())],
                     "length": len(df)})
    print(f"  task_index    <- subtask_index ({relabelled} episode(s) changed)")
    print(f"  frame_index   <- added to {framed} episode(s)")
    if backup_dir.exists():
        print(f"  originals     -> {backup_dir.name}/")

    _backup(root / "meta" / "episodes.jsonl")
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: r["episode_index"]):
            f.write(json.dumps(r) + "\n")
    print(f"  episodes.jsonl<- {len(rows)} episodes, phases listed per episode")

    #-- 2. modality -------------------------------------------------------------------------
    mod = json.load(open(ref / "meta" / "modality.json"))
    feats = json.load(open(root / "meta" / "info.json"))["features"]
    have = {k.split(".")[-1] for k in feats if k.startswith("observation.images")}
    want = set(mod["video"])
    missing = want - have
    if missing:
        print(f"  !! reference wants cameras this recording lacks: {sorted(missing)}")
        return 1
    _backup(root / "meta" / "modality.json")
    json.dump(mod, open(root / "meta" / "modality.json", "w"), indent=2)
    shutil.copy(root / "meta" / "modality.json", root / "modality.json")
    print(f"  modality.json <- video {sorted(want)}")
    if have - want:
        print(f"                   excluded {sorted(have - want)} (the checkpoint has never seen it)")

    #-- 4. stats ----------------------------------------------------------------------------
    _backup(root / "meta" / "stats.json")
    stats = json.load(open(root / "meta" / "stats.json"))
    big = pd.concat([pd.read_parquet(f) for f in parquets])
    for feat in ("observation.state", "action"):
        M = np.stack(big[feat].to_numpy()).astype(np.float64)
        stats.setdefault(feat, {})
        stats[feat].update(
            q01=np.percentile(M, 1, axis=0).tolist(), q99=np.percentile(M, 99, axis=0).tolist(),
            mean=M.mean(0).tolist(), std=M.std(0).tolist(),
            min=M.min(0).tolist(), max=M.max(0).tolist())
    json.dump(stats, open(root / "meta" / "stats.json", "w"))
    print("  stats.json    <- q01/q99 recomputed on the actual frames")

    #-- the gate that the first 30k run needed and did not have -----------------------------
    declared = sorted({i for g in mod["state"].values() for i in range(g["start"], g["end"])})
    bad = []
    for feat in ("observation.state", "action"):
        q01 = np.asarray(stats[feat]["q01"]); q99 = np.asarray(stats[feat]["q99"])
        sd = np.asarray(stats[feat]["std"])
        for i in declared:
            if i < len(q01) and ((q99[i] - q01[i]) <= 0 or sd[i] <= 0):
                bad.append(f"{feat}[{i}]")
    if bad:
        print(f"\nDEGENERATE declared dims -- do NOT train: {bad}")
        return 1

    lang = big["subtask_index"].map(tmap).value_counts()
    print(f"\n  declared dims {declared[0]}..{declared[-1]}, none degenerate")
    print(f"  {len(big)} frames now carry {len(lang)} distinct instructions:")
    for s, n in lang.sort_index().items():
        print(f"     {n:6d}  {s}")
    print("\nfixed. Re-run verify_dataset.py --isaac before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
