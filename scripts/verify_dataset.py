"""Refuse to spend a training run on a dataset whose labels are broken.

Author: Dawit Chun

Every check here corresponds to something that has already cost us a real training run. None of them
raise an error during training -- the run completes, the loss looks fine, and the damage only shows
up as a robot that drifts.

  1  ACTION FRAME     `action` must be in the FOLLOWER's frame. It was the LEADER's absolute joint
                      position, and the leader freeze permanently desynchronises the two: measured
                      0.3 mm before the first freeze and 157 mm by the driver grasp on episode 32.
                      The model learns the leader faithfully and we then command the follower.
  2  FREEZE FRAMES    frames where the label commands motion the robot did not make. 5.2% of the
                      last dataset. They teach "command motion, nothing happens".
  3  LABEL CONSISTENCY the same task phase must not carry a different offset in each episode. It
                      varied by 19 deg, which is an irreducible error floor no amount of training
                      can beat -- the last model landed within 1.1x of it.
  4  TASKS.PARQUET    language lives in the INDEX. Cyclo writes it as a column, so the model reads
                      the literal strings "0".."1".. and it fails silently.
  5  CAMERA SHAPES    wrists are portrait 424x240 in training. A transposed frame is a valid array.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

FPS = 30.0
MIN_FREEZE_RUN = 15      #frames; 0.5 s at 30 Hz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lead", type=int, default=5, help="k, if action == follower_state[t+k]")
    ap.add_argument("--eval-episodes", type=int, nargs="*", default=[31, 32, 33, 34])
    ap.add_argument("--val-root", default=None,
                    help="the SEPARATE validation dataset. Isaac has no eval_split -- it takes "
                         "val_dataset_path -- so the split is physical and must be checked for "
                         "episode overlap.")
    ap.add_argument("--isaac", action="store_true",
                    help="also apply the Isaac-path checks: meta/modality.json present and its "
                         "slices matching the action names, and the codebase version.")
    a = ap.parse_args()
    R = Path(a.root)
    info = json.loads((R / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    ARM = [i for i, n in enumerate(names) if n.startswith(("arm_l", "arm_r"))]
    #FREEZE IS PER ARM, NOT PER ROBOT.
    #
    #The operator freezes ONE arm to hold a steady camera view of the work site while the other arm
    #keeps demonstrating -- right arm parked over the screw, left arm placing the bolt. Taking the
    #max joint speed across BOTH arms hides that completely: the working arm keeps the frame out of
    #the frozen bucket. Measured, that under-reported freezes as 1.55% when the true figure is
    #8.80%, of which 8.67% is single-arm.
    #
    #These frames must NOT be discarded. The working arm is demonstrating the most precise part of
    #the task, and dropping them would bias the dataset against exactly those moments.
    ARMS = {side: [i for i, n in enumerate(names) if n.startswith(f"arm_{side}")]
            for side in ("l", "r")}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(R / "data/**/*.parquet"),
                                                                 recursive=True))])
    fails, warns = [], []
    print(f"verifying {R}\n{'='*74}")

    #1 + 2 + 3 -------------------------------------------------------------
    gaps, frozen, offs = [], 0, {}
    total = 0
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        S = np.stack(g["observation.state"].to_numpy())[:, ARM]
        A = np.stack(g["action"].to_numpy())[:, ARM]
        any_frozen = np.zeros(len(S), dtype=bool)
        for side, idx in ARMS.items():
            Sa = np.stack(g["observation.state"].to_numpy())[:, idx]
            Aa = np.stack(g["action"].to_numpy())[:, idx]
            f = np.r_[0, np.abs(np.diff(Sa, axis=0)).max(axis=1) * FPS]
            l = np.r_[0, np.abs(np.diff(Aa, axis=0)).max(axis=1) * FPS]
            fz = (f < 0.02) & (l > 0.05)
            #ONLY SUSTAINED RUNS COUNT. 92% of raw detections are 1-2 frame blips where the
            #follower momentarily reads zero velocity while the leader jitters -- they are not
            #freezes and they do not create the permanent offset. Requiring half a second leaves
            #the real ones: measured 350 frames in runs >= 30, max run 67 frames (2.23 s).
            d = np.diff(np.r_[0, fz.astype(int), 0])
            for a0, b0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if b0 - a0 >= MIN_FREEZE_RUN:
                    any_frozen[a0:b0] = True
        frozen += int(any_frozen.sum()); total += len(any_frozen)
        #if the label is follower[t+k] the gap is one k-step of real motion, not a static offset
        gaps.append(np.degrees(np.abs(A[:-a.lead] - S[a.lead:]).mean()))
        for st, gg in g.groupby("subtask_index"):
            s = np.stack(gg["observation.state"].to_numpy())[:, ARM]
            aa = np.stack(gg["action"].to_numpy())[:, ARM]
            offs.setdefault(int(st), []).append(np.degrees((aa - s).mean(axis=0)))

    gap = float(np.mean(gaps))
    print(f"1. ACTION FRAME       mean |action[t] - state[t+{a.lead}]| = {gap:6.3f} deg")
    if gap > 1.0:
        fails.append(f"action is NOT follower[t+{a.lead}] (gap {gap:.2f} deg). Still the leader "
                     f"topic? See the conversion fix: action[t] = observation.state[t+{a.lead}].")
    else:
        print(f"   -> consistent with follower[t+{a.lead}]")

    pct = 100 * frozen / max(total, 1)
    print(f"2. FREEZE FRAMES      {frozen} of {total} ({pct:.2f}%) -- at least one arm frozen "
          f"in a sustained freeze (>={MIN_FREEZE_RUN}f) while the label commands motion, PER ARM")
    if pct > 0.2:
        fails.append(f"{pct:.1f}% of frames are frozen-but-commanded. With action = follower[t+k] "
                     f"this must be near zero.")

    worst = max((np.stack(v).std(axis=0).mean(), k) for k, v in offs.items())
    print(f"3. LABEL CONSISTENCY  worst across-episode SD = {worst[0]:.2f} deg (subtask {worst[1]})")
    if worst[0] > 1.0:
        fails.append(f"the same task phase carries a {worst[0]:.2f} deg different offset per episode. "
                     f"That is an error floor the model cannot beat.")

    #4 ---------------------------------------------------------------------
    #v3.0 keeps tasks in meta/tasks.parquet (sentence in the INDEX); v2.1 in meta/tasks.jsonl
    #(sentence under the "task" key). Isaac reads the jsonl; LeRobot reads the parquet.
    tj, tp = R / "meta" / "tasks.jsonl", R / "meta" / "tasks.parquet"
    if tj.exists():
        rows = [json.loads(l) for l in tj.read_text().splitlines() if l.strip()]
        idx = [str(r["task"]) for r in rows[:5]]
        label = "TASKS.JSONL (v2.1)"
    elif tp.exists():
        t = pd.read_parquet(tp)
        idx = [str(x) for x in t.index[:5]]
        label = "TASKS.PARQUET"
    else:
        idx, label = [], "TASKS -- NEITHER FILE"
        fails.append("neither meta/tasks.jsonl nor meta/tasks.parquet exists")
    print(f"4. {label:<18} first two = {idx[:2]}")
    if idx and all(x.isdigit() for x in idx):
        fails.append("the task strings are digits. In v3.0 that means they were written to a COLUMN "
                     "instead of the parquet INDEX; in v2.1 it means the conversion carried that "
                     "through. Either way the model is conditioned on \"0\",\"1\",\"2\".")

    #5 ---------------------------------------------------------------------
    print("5. CAMERA SHAPES")
    for k, v in info["features"].items():
        if "image" not in k:
            continue
        c, h, w = v["shape"]
        tag = ""
        if "wrist" in k and (h, w) != (424, 240):
            tag = "  <-- expected 424x240 portrait"
            fails.append(f"{k} is {h}x{w}, not the 424x240 the checkpoint was trained on.")
        print(f"   {k.split('.')[-1]:<18} {h}x{w}{tag}")

    #6 + 7 + 8: the Isaac path ------------------------------------------------
    if a.isaac:
        mj = R / "meta" / "modality.json"
        print(f"6. MODALITY.JSON      {'present' if mj.exists() else 'MISSING'}")
        if not mj.exists():
            fails.append("meta/modality.json is missing. Isaac requires it: cp "
                         "examples/CYCLO/ffw_sg2_rev1/modality.json <dataset>/meta/")
        else:
            #the slices are positional -- a mismatch silently feeds the wrong joints to a modality
            spec = json.loads(mj.read_text()).get("action", {})
            #ROBOTIS's arm_* modalities are 8 wide: SEVEN joints plus that arm's gripper.
            #An earlier version of this check demanded all 8 start with "arm_l"/"arm_r" and
            #flagged a correct dataset -- the gripper is the 8th element by design.
            expect = {
                "arm_left":  [f"arm_l_joint{i}" for i in range(1, 8)] + ["gripper_l_joint1"],
                "arm_right": [f"arm_r_joint{i}" for i in range(1, 8)] + ["gripper_r_joint1"],
                "head":      ["head_joint1", "head_joint2"],
                "lift":      ["lift_joint"],
                "odometry":  ["linear_x", "linear_y", "angular_z"],
            }
            #Which modalities are PRESENT is a deliberate choice -- we train 16D (the two arms)
            #because head/lift/odometry are degenerate here. So do not demand every key exists;
            #demand that whatever IS declared maps to the right columns, and that no declared
            #dim has a degenerate normaliser. The 30k run shipped all 22 and fed the model
            #linear_y at +-32 of pure sensor noise because nothing checked this.
            for key in spec:
                if key not in expect:
                    fails.append(f"modality.json declares unknown action key '{key}'"); continue
                want_cols = expect[key]
                lo, hi = spec[key]["start"], spec[key]["end"]
                got = names[lo:hi]
                ok = got == want_cols
                print(f"   {key:<11} [{lo:>2},{hi:>2})  {'ok' if ok else 'MISMATCH'}")
                if not ok:
                    fails.append(f"modality.json slice {key} [{lo},{hi}) holds {got}, expected "
                                 f"{want_cols}. The slices are POSITIONAL -- Isaac never checks "
                                 f"them against column names, so a mismatch feeds the wrong joints "
                                 f"to that modality silently.")
            declared = sorted({i for k in spec for i in range(spec[k]["start"], spec[k]["end"])})
            print(f"   declared dims: {len(declared)} of {len(names)}")

            #10. every DECLARED dim must have a usable normaliser, in state and in action.
            stats_p = Path(R) / "meta" / "stats.json"
            if stats_p.exists():
                stt = json.loads(stats_p.read_text())
                bad = []
                for feat in ("observation.state", "action"):
                    if feat not in stt: continue
                    import numpy as _np
                    q01 = _np.asarray(stt[feat]["q01"], dtype=float).ravel()
                    q99 = _np.asarray(stt[feat]["q99"], dtype=float).ravel()
                    sd  = _np.asarray(stt[feat]["std"], dtype=float).ravel()
                    for i in declared:
                        rng = float(q99[i] - q01[i])
                        if rng <= 0 or float(sd[i]) <= 0:
                            bad.append(f"{feat}:{names[i]} q99-q01={rng:.3e} std={float(sd[i]):.3e}")
                print(f"10. NORMALISER        {len(bad)} declared dim(s) with a degenerate range")
                for b in bad:
                    fails.append(f"{b} -- zero/near-zero normaliser on a DECLARED dim. This is the "
                                 f"documented divergence mode: normalising by ~0 turns sensor noise "
                                 f"into huge inputs. Drop the dim from modality.json or fix the data.")

        ver = info.get("codebase_version", "?")
        print(f"8. DATASET VERSION    {ver}")
        if str(ver).startswith("v3"):
            warns.append(f"codebase_version is {ver}. ROBOTIS's CYCLO README asks for a LeRobot "
                         f"v2.x dataset. I could not find a version check in the Isaac loader, so "
                         f"I do not know whether v3.0 loads or mis-parses. Settle it with a "
                         f"2-episode smoke run before converting all 35.")

    if a.val_root:
        V = Path(a.val_root)
        vdf = pd.concat([pd.read_parquet(f) for f in
                         sorted(glob.glob(str(V / "data/**/*.parquet"), recursive=True))])
        #COMPARE ORIGINAL EPISODE IDS, NOT RENUMBERED ONES.
        #
        #split_train_val.py renumbers each output from 0 because LeRobot indexes episodes
        #positionally, so train holds 0..30 and val 0..3 and a naive comparison reports four
        #"overlapping" episodes that are in fact completely different. meta/split_provenance.json
        #records the original -> new mapping for exactly this.
        def original_ids(root, frame):
            prov = Path(root) / "meta" / "split_provenance.json"
            if prov.exists():
                m = json.loads(prov.read_text())["original_episode_index_to_new"]
                return set(int(k) for k in m)
            return set(int(x) for x in frame.episode_index.unique())

        tr_eps = original_ids(R, df)
        va_eps = original_ids(V, vdf)
        overlap = tr_eps & va_eps
        print(f"7. TRAIN/VAL SPLIT    train {len(tr_eps)} ep, val {len(va_eps)} ep, "
              f"overlap {len(overlap)}  (original episode ids)")
        if overlap:
            fails.append(f"episodes {sorted(overlap)} appear in BOTH datasets. Isaac will not "
                         f"hold anything out for you; the split is physical.")
        if va_eps != set(a.eval_episodes):
            warns.append(f"val episodes {sorted(va_eps)} != {a.eval_episodes}. Results will not "
                         f"be comparable with the baselines already measured on episode 32.")
    elif a.isaac:
        warns.append("no --val-root given. Isaac's eval_strategy defaults to \"no\" and "
                     "val_dataset_path is not exposed by launch_finetune's CLI, so without a "
                     "separate val dataset AND a launcher patch the run trains blind.")

    #9: video/parquet alignment (v2.1 layouts only) --------------------------
    v21 = R / "data" / "chunk-000"
    if (R / "meta" / "tasks.jsonl").exists() and v21.exists():
        import subprocess
        mism = []
        for pq in sorted(v21.glob("episode_*.parquet")):
            rows = len(pd.read_parquet(pq))
            ep = pq.stem.split("_")[1]
            for vd in sorted((R / "videos" / "chunk-000").glob("*")):
                mp4 = vd / f"episode_{ep}.mp4"
                if not mp4.exists():
                    mism.append((mp4.name, vd.name, "MISSING", rows)); continue
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
                    capture_output=True, text=True).stdout.strip()
                n = int(out) if out.isdigit() else -1
                if n != rows:
                    mism.append((mp4.name, vd.name, n, rows))
        print(f"9. VIDEO ALIGNMENT    {len(mism)} of the per-episode videos disagree with their parquet")
        if mism:
            for nm, cam, n, rows in mism[:4]:
                print(f"   {cam.split('.')[-1]:<16} {nm}  video {n} vs rows {rows}")
            fails.append(
                f"{len(mism)} video(s) have a different frame count from their parquet. "
                f"convert_v3_to_v2.py splits with `-ss` BEFORE `-i` and `-c copy`, which cannot cut "
                f"mid-GOP, so the output carries extra LEADING frames. Isaac indexes video by frame "
                f"NUMBER with no timestamp compensation, so every image would be paired with a later "
                f"action -- measured up to 240 frames (8 s). Fix with "
                f"hil_dagger/eval/fix_episode_videos.py")

    #summary ---------------------------------------------------------------
    n = df.episode_index.nunique()
    tr = [e for e in df.episode_index.unique() if e not in a.eval_episodes]
    print(f"\n   {len(df)} frames, {n} episodes ({len(tr)} train / {len(a.eval_episodes)} held out)")
    print("="*74)
    for w in warns:
        print(f"WARN  {w}")
    if fails:
        print(f"\nREFUSING: {len(fails)} problem(s) that a training run will not reveal.\n")
        for i, f in enumerate(fails, 1):
            print(f"  {i}. {f}")
        return 1
    print("\nAll checks passed. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
