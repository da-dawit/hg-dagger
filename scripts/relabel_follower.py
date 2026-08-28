"""Rewrite `action` into the FOLLOWER's frame: action[t] = observation.state[t+k].

Author: Dawit Chun

NO RECONVERSION IS NEEDED. The relabel uses only columns the dataset already contains, so the
rosbags, the video transcode and the whole cyclo_data pipeline are irrelevant to it. Videos are
hard-linked, not copied.

WHY. `action` is the LEADER's absolute joint position. The leader freeze locks the follower while
the operator's hand keeps moving; on release the teleop re-references instead of snapping, so from
then on the two sit at permanently different joint configurations. Measured on episode 32 the gap
is 0.3 mm before the first freeze and 157 mm by the driver grasp. The model learns the leader's
frame faithfully and we then command the FOLLOWER with it.

SCOPE: the 16 CONTROLLED dims only (14 arms + 2 grippers). The other six are left exactly as they
are, because `state` is worse for all of them:
  linear_x / linear_y / angular_z   action is a clean 0.000; state is jitter (the base never moves)
  lift_joint                        state range 3e-5 rad; min-max would stretch encoder noise to
                                    full scale. The action's range is 350x larger.
  head_joint1 / head_joint2         same argument, smaller magnitude
They are never commanded on the robot (spec.MODEL_JOINTS is 16-D), so leaving them changes nothing
about control.

k=5 is not tuned: it is the measured leader->follower tracking lag on subtask 0, BEFORE any freeze
can corrupt it (median 5, mean 5.0 over 35 episodes; per-arm left 5, right 4.3).
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

CTRL_PREFIX = ("arm_l", "arm_r", "gripper_l", "gripper_r")


def stats_of(a):
    return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--lead", type=int, default=5)
    ap.add_argument("--modality", default="/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/"
                                          "policy/groot/Isaac-GR00T/examples/CYCLO/ffw_sg2_rev1/"
                                          "modality.json")
    a = ap.parse_args()
    S, D = Path(a.src), Path(a.dst)
    if D.exists():
        raise SystemExit(f"{D} already exists -- refusing to overwrite. Remove it first.")

    info = json.loads((S / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    assert names == info["features"]["observation.state"]["names"], \
        "action and state column order differ; the relabel would mix joints"
    CTRL = [i for i, n in enumerate(names) if n.startswith(CTRL_PREFIX)]
    print(f"[scope] relabelling {len(CTRL)} of {len(names)} dims: "
          f"{[names[i] for i in CTRL[:3]]} ... {[names[i] for i in CTRL[-2:]]}")
    print(f"[scope] left untouched: {[names[i] for i in range(len(names)) if i not in CTRL]}")

    #tree: everything except data/ and meta/ is bit-identical, so hard-link it
    D.mkdir(parents=True)
    print("[tree] hard-linking videos ...")
    for p in (S).rglob("*"):
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

    #data ------------------------------------------------------------------
    per_ep = {}
    n_files = 0
    for f in sorted((S / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        out = []
        for e, g in df.groupby("episode_index", sort=True):
            g = g.sort_values("frame_index").copy()
            st = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
            ac = np.stack(g["action"].to_numpy()).astype(np.float64)
            new = ac.copy()
            idx = np.minimum(np.arange(len(st)) + a.lead, len(st) - 1)   #tail clamps to final pose
            new[:, CTRL] = st[idx][:, CTRL]
            g["action"] = list(new.astype(np.float32))
            per_ep[int(e)] = stats_of(new.astype(np.float32))
            out.append(g)
        df2 = pd.concat(out).sort_values("index")
        q = D / f.relative_to(S)
        q.parent.mkdir(parents=True, exist_ok=True)
        df2.to_parquet(q, index=False)
        n_files += 1
    print(f"[data] rewrote {n_files} parquet file(s), {len(per_ep)} episodes")

    #meta ------------------------------------------------------------------
    (D / "meta").mkdir(exist_ok=True)
    for p in (S / "meta").rglob("*"):
        rel = p.relative_to(S / "meta")
        q = D / "meta" / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)

    #global action stats must match the new column, or normalisation is computed from the old one
    allA = np.concatenate([np.stack(pd.read_parquet(f)["action"].to_numpy())
                           for f in sorted((D / "data").rglob("*.parquet"))]).astype(np.float32)
    stats = json.loads((D / "meta" / "stats.json").read_text())
    stats["action"] = stats_of(allA)
    (D / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[meta] recomputed global action stats over {len(allA)} frames")

    #per-episode stats live in meta/episodes/*.parquet as stats/action/<stat> columns
    for f in sorted((D / "meta" / "episodes").rglob("*.parquet")):
        ep = pd.read_parquet(f)
        for stat in ("mean", "std", "min", "max"):
            col = f"stats/action/{stat}"
            if col in ep.columns:
                ep[col] = [np.asarray(per_ep[int(e)][stat], dtype=np.float32)
                           for e in ep["episode_index"]]
        ep.to_parquet(f, index=False)
        print(f"[meta] refreshed per-episode action stats in {f.name}")

    #modality.json -- required by the Isaac path
    m = Path(a.modality)
    if m.exists():
        shutil.copy2(m, D / "meta" / "modality.json")
        print(f"[meta] installed modality.json from {m.parent.name}/")
    else:
        print(f"[meta] WARNING modality.json not found at {m}")

    print(f"\nwrote {D}")


if __name__ == "__main__":
    main()
