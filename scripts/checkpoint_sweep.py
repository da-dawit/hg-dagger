"""Held-out grasp error vs training step -- the validation curve Isaac will not give you.

Author: Dawit Chun

Isaac's DatasetFactory.build() asserts eval_strategy == "no", so a run produces no held-out
number at all. At 80k steps that is 92 passes over 31 episodes with nothing to say whether it
is still learning or memorising.

This loads each checkpoint ONCE and measures the same quantity grasp_bias.py does -- signed
Cartesian offset of the working arm over the last `tail` frames of each subtask, by forward
kinematics on arm_{l,r}_link7 -- across EVERY held-out episode, then reports it per arm as a
curve over steps.

Both arms, all episodes. An earlier harness measured the right wrist throughout and reported a
2.5x improvement on the arm that does not do the failing grasp.

  python3 checkpoint_sweep.py --runs ~/runs/groot35_follower --out sweep.json
"""
import argparse, json, sys, glob, re, time
from pathlib import Path
import numpy as np
import pandas as pd

G = "/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T"
sys.path.insert(0, G); sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
SUB = ["Grab bolt", "Place in hole", "Grab driver", "Screw in", "Home"]
WORKING_ARM = {0: "l", 1: "l", 2: "r", 3: "r", 4: "r"}
#subtask 4 is measured on the right arm for continuity with the earlier numbers, but joint
#travel is 48/52 L/R -- it is genuinely bimanual, so read phase 4 with that in mind.
GRASP_PHASES = (0, 2)   #the two that decide success: bolt (left) and driver (right)


def discover(runs):
    out = []
    for p in glob.glob(str(Path(runs) / "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m and (Path(p) / "config.json").exists():
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="directory holding checkpoint-*")
    ap.add_argument("--ckpts", nargs="*", default=None, help="explicit checkpoint paths instead")
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--v30", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val_v3.0")
    ap.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--tail", type=int, default=60)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out", default="checkpoint_sweep.json")
    a = ap.parse_args()

    if a.ckpts:
        cks = sorted((int(re.search(r"(\d+)$", c).group(1)), c) for c in a.ckpts)
    else:
        if not a.runs: raise SystemExit("pass --runs or --ckpts")
        cks = discover(a.runs)
    if not cks: raise SystemExit(f"no checkpoints under {a.runs}")
    print(f"{len(cks)} checkpoints: {[s for s, _ in cks]}")
    print(f"{len(a.episodes)} held-out episodes: {a.episodes}\n")

    import torch, mujoco, spec_sg2 as spec
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.eval.open_loop_eval import parse_observation_gr00t, parse_action_gr00t
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    names = json.loads((Path(a.dataset) / "meta" / "info.json").read_text())["features"]["action"]["names"]
    v30 = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(str(Path(a.v30) / "data/**/*.parquet"), recursive=True))])

    m = mujoco.MjModel.from_xml_path("/home/robotis/robotis_mujoco_menagerie/robotis_ffw/ffw_sg2.xml")
    d = mujoco.MjData(m)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in spec.MODEL_JOINTS}
    TIPS = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"arm_{WORKING_ARM[s]}_link7") for s in range(5)}
    #MODEL_JOINTS are the 16 controlled dims and every one indexes into names[0:16], so this is
    #correct for BOTH the old 22-dim checkpoints and the new 16-dim ones.
    JIDX = [names.index(j) for j in spec.MODEL_JOINTS]
    assert max(JIDX) < 16, f"MODEL_JOINTS reaches index {max(JIDX)}; expected all < 16"

    def fk(v, tip):
        for j, i in zip(spec.MODEL_JOINTS, JIDX): d.qpos[qadr[j]] = v[i]
        mujoco.mj_kinematics(m, d); return d.xpos[tip].copy()

    results = {}
    for step, ck in cks:
        t0 = time.time()
        policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=ck,
                             device="cuda" if torch.cuda.is_available() else "cpu")
        mc = policy.get_modality_config(); akeys = mc["action"].modality_keys
        loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc,
                                      video_backend="torchcodec")
        per_phase = {s: [] for s in range(5)}
        for ep in a.episodes:
            traj = loader[ep]
            gt = np.stack(pd.read_parquet(
                Path(a.dataset) / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
            ).sort_values("frame_index")["action"].to_numpy())
            sub = v30[v30.episode_index == ep].sort_values("frame_index").subtask_index.to_numpy()
            n = min(len(gt), len(sub))
            for s in range(5):
                idx = np.where(sub[:n] == s)[0]
                if not len(idx): continue
                for t in idx[-a.tail:][::a.stride]:
                    if t + a.horizon >= n: continue
                    st = extract_step_data(traj, int(t), mc, "new_embodiment")
                    obs = {f"state.{k}": v for k, v in st.states.items()}
                    for k, v in st.images.items(): obs[f"video.{k}"] = np.array(v)
                    obs["annotation.human.primitive_instruction"] = st.text
                    raw, _ = policy.get_action(parse_observation_gr00t(obs, mc))
                    ch = parse_action_gr00t(raw)
                    pred = np.concatenate([np.atleast_1d(np.atleast_1d(ch[f"action.{k}"])[0]) for k in akeys])
                    per_phase[s].append(1000 * (fk(pred, TIPS[s]) - fk(gt[t], TIPS[s])))

        row = {}
        for s in range(5):
            if not per_phase[s]: continue
            V = np.stack(per_phase[s]); b = V.mean(0)
            row[s] = {"arm": WORKING_ARM[s], "n": len(V),
                      "bias_mm": [float(x) for x in b],
                      "bias_norm_mm": float(np.linalg.norm(b)),
                      "mean_err_mm": float(np.linalg.norm(V, axis=1).mean()),
                      "systematic": float(np.linalg.norm(b) / max(np.linalg.norm(V, axis=1).mean(), 1e-9))}
        results[step] = row
        del policy; torch.cuda.empty_cache()
        print(f"  [{step:>7}] {time.time()-t0:5.0f}s  " +
              "  ".join(f"{SUB[s]}={row[s]['mean_err_mm']:.1f}mm" for s in GRASP_PHASES if s in row))
        Path(a.out).write_text(json.dumps(results, indent=2))   #checkpoint the results as we go

    print(f"\n{'step':>8}" + "".join(f"{SUB[s]+' ('+WORKING_ARM[s].upper()+')':>22}" for s in GRASP_PHASES))
    print(f"{'':>8}" + "".join(f"{'err':>11}{'syst':>11}" for _ in GRASP_PHASES))
    for step in sorted(results):
        line = f"{step:>8}"
        for s in GRASP_PHASES:
            r = results[step].get(s)
            line += f"{r['mean_err_mm']:>11.1f}{r['systematic']:>11.2f}" if r else f"{'-':>11}{'-':>11}"
        print(line)

    print()
    for s in GRASP_PHASES:
        pts = [(k, results[k][s]["mean_err_mm"]) for k in sorted(results) if s in results[k]]
        if not pts: continue
        best = min(pts, key=lambda kv: kv[1])
        last = pts[-1]
        verdict = ("STILL IMPROVING at the last checkpoint" if best[0] == last[0]
                   else f"TURNED -- best at {best[0]}, {100*(last[1]-best[1])/best[1]:+.1f}% worse by {last[0]}")
        print(f"  {SUB[s]:<16} ({WORKING_ARM[s].upper()})  best {best[1]:.1f} mm @ step {best[0]}   -> {verdict}")
    print(f"\nwrote {a.out}")
    print("Open-loop and teacher-forced: every chunk starts from the HUMAN's observation, so this")
    print("cannot show error accumulation. It answers 'is it still learning', not 'does it work'.")


if __name__ == "__main__":
    main()
