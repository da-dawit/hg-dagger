"""Split a LeRobot v3.0 dataset into two PHYSICAL datasets, for Isaac.

Author: Dawit Chun

Isaac has no `eval_split`. `SingleDatasetConfig.val_dataset_path` takes a separate dataset path, so
unlike LeRobot -- which holds out the last ceil(n*eval_split) episodes per task automatically -- the
split here has to exist on disk.

All episodes share ONE video file per camera, each located by from_timestamp/to_timestamp. Both
outputs therefore HARD-LINK the same video files: no re-encode, no extra disk, and the timestamps
keep pointing at the right segments.

WHICH EPISODES. Default val = [31, 32, 33, 34], the same four LeRobot held out, so results stay
comparable with every baseline already measured on episode 32. Episode indices are renumbered from
0 in each output (LeRobot indexes episodes positionally); the mapping is printed and written to
meta/split_provenance.json so "val episode 1" can still be identified as original 32.
"""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd


def stats_of(a):
    return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist()}


def write_split(S, D, keep, label):
    D.mkdir(parents=True)
    info = json.loads((S / "meta" / "info.json").read_text())

    #videos and anything else that is bit-identical -> hard-link
    for p in S.rglob("*"):
        rel = p.relative_to(S)
        if rel.parts and rel.parts[0] in ("data", "meta"):
            continue
        q = D / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, q)
            except OSError:
                shutil.copy2(p, q)

    remap = {int(e): i for i, e in enumerate(sorted(keep))}

    frames = []
    for f in sorted((S / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        df = df[df.episode_index.isin(keep)].copy()
        if len(df):
            frames.append((f.relative_to(S), df))
    assert frames, f"{label}: no rows matched {sorted(keep)}"

    total = 0
    for rel, df in frames:
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        df["episode_index"] = df["episode_index"].map(remap).astype("int64")
        df["index"] = np.arange(total, total + len(df), dtype="int64")
        total += len(df)
        q = D / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(q, index=False)

    #meta
    (D / "meta").mkdir(exist_ok=True)
    for p in (S / "meta").rglob("*"):
        rel = p.relative_to(S / "meta")
        q = D / "meta" / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)

    off = 0
    for f in sorted((D / "meta" / "episodes").rglob("*.parquet")):
        ep = pd.read_parquet(f)
        ep = ep[ep.episode_index.isin(keep)].copy()
        ep = ep.sort_values("episode_index").reset_index(drop=True)
        ep["episode_index"] = ep["episode_index"].map(remap).astype("int64")
        #these index into the dataset's own frame range, which the renumbering just changed
        lens = ep["length"].to_numpy()
        ep["dataset_from_index"] = np.cumsum(np.r_[0, lens])[:-1] + off
        ep["dataset_to_index"] = np.cumsum(lens) + off
        ep.to_parquet(f, index=False)

    allA = np.concatenate([np.stack(pd.read_parquet(f)["action"].to_numpy())
                           for f in sorted((D / "data").rglob("*.parquet"))]).astype(np.float32)
    allS = np.concatenate([np.stack(pd.read_parquet(f)["observation.state"].to_numpy())
                           for f in sorted((D / "data").rglob("*.parquet"))]).astype(np.float32)
    stats = json.loads((D / "meta" / "stats.json").read_text())
    stats["action"] = stats_of(allA)
    stats["observation.state"] = stats_of(allS)
    (D / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    info["total_episodes"] = len(keep)
    info["total_frames"] = int(total)
    (D / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    (D / "meta" / "split_provenance.json").write_text(json.dumps(
        {"source": str(S), "split": label,
         "original_episode_index_to_new": {str(k): v for k, v in remap.items()}}, indent=2))

    print(f"[{label:<5}] {len(keep)} episodes, {total} frames -> {D}")
    print(f"        original -> new: "
          f"{ {k: v for k, v in list(remap.items())[:3]} }{' ...' if len(remap) > 3 else ''}")
    return remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--val-episodes", type=int, nargs="+", default=[31, 32, 33, 34])
    a = ap.parse_args()
    S = Path(a.src)
    for d in (a.train_out, a.val_out):
        if Path(d).exists():
            raise SystemExit(f"{d} exists -- refusing to overwrite.")
    df = pd.concat([pd.read_parquet(f) for f in sorted((S / "data").rglob("*.parquet"))])
    allep = set(int(x) for x in df.episode_index.unique())
    val = set(a.val_episodes)
    missing = val - allep
    if missing:
        raise SystemExit(f"val episodes {sorted(missing)} are not in {S}")
    train = allep - val
    write_split(S, Path(a.train_out), train, "train")
    write_split(S, Path(a.val_out), val, "val")
    print(f"\noverlap: {sorted(train & val)}  (must be empty)")


if __name__ == "__main__":
    main()
