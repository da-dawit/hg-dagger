"""Where does GR00T actually look at the grasp instant -- and how much of that lands on padding?

Author: Dawit Chun

The action expert cross-attends to the VLM's image tokens. In AlternateVLDiT the blocks alternate:
  idx % 2 == 1                      -> self-attention
  idx % 2 == 0, idx % (2*n) == 0    -> cross-attend to NON-image (text) tokens
  idx % 2 == 0, idx % (2*n) != 0    -> cross-attend to IMAGE tokens      <- these are the ones
(gr00t/model/modules/dit.py:378-404, n = attend_text_every_n_blocks).

We hook those blocks, recompute softmax(QK^T/sqrt(d)) over the image-token slice, average over
heads and over the action queries, and split the result per camera using image_grid_thw.

The number this exists to produce: our fork applies LetterBoxPad() unconditionally, so ~44% of every
256x256 model input is black bars -- while the base checkpoint ships letter_box_transform=False.
`padding attention` below is the fraction of the policy's visual attention spent on those bars.

  python attn_map.py --ckpt <checkpoint> --episode 1 --frame 898 --out attn.png
"""
import argparse, json, sys, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

G = "/home/robotis/robot_omy/cyclo_intelligence/cyclo_brain/policy/groot/Isaac-GR00T"
sys.path.insert(0, G); sys.path.insert(0, "/home/robotis/robot_aiworker/aiworker_deploy")
SUB = ["Grab bolt", "Place in hole", "Grab driver", "Screw in", "Home"]


def letterbox(img, side):
    """Replicate what the model actually saw, so the heatmap is drawn on the right pixels.

    Mirrors image_augmentations.py: CropToSquare() when GR00T_SQUARE_CROP=1, else LetterBoxPad().
    Returns (square_image, content_mask); content_mask is all-True for the crop path since there
    are no black bars.
    """
    import cv2, os
    h, w = img.shape[:2]
    if os.environ.get("GR00T_SQUARE_CROP", "0") == "1":
        rj = os.environ.get("GR00T_ROI_JSON")
        r = None
        if rj and os.path.exists(rj):
            import json as _j
            for _, v in _j.load(open(rj)).items():
                if (int(v["H"]), int(v["W"])) == (h, w): r = v; break
        if r is not None:
            ch = int(np.clip(int(r["height"]), 1, h)); cw = int(np.clip(int(r["width"]), 1, w))
            top = int(np.clip(int(r["top"]) + int(r["height"]) // 2 - ch // 2, 0, h - ch))
            left = int(np.clip(int(r["left"]) + int(r["width"]) // 2 - cw // 2, 0, w - cw))
            sub = img[top:top + ch, left:left + cw]
            if ch != cw:                       #ROI is not square -> pad only the ROI to square
                m = max(ch, cw)
                canv = np.zeros((m, m, 3), sub.dtype)
                cont = np.zeros((m, m), bool)
                y0, x0 = (m - ch) // 2, (m - cw) // 2
                canv[y0:y0 + ch, x0:x0 + cw] = sub; cont[y0:y0 + ch, x0:x0 + cw] = True
                return (cv2.resize(canv, (side, side), interpolation=cv2.INTER_AREA),
                        cv2.resize(cont.astype(np.uint8), (side, side),
                                   interpolation=cv2.INTER_NEAREST).astype(bool))
            return (cv2.resize(sub, (side, side), interpolation=cv2.INTER_AREA),
                    np.ones((side, side), bool))
        n = min(h, w)
        top, left = (h - n, 0) if h > w else (0, (w - n) // 2)   #portrait -> bottom, landscape -> centre
        sq = img[top:top + n, left:left + n]
        return (cv2.resize(sq, (side, side), interpolation=cv2.INTER_AREA),
                np.ones((side, side), bool))
    m = max(h, w)
    canvas = np.zeros((m, m, 3), img.dtype)
    canvas[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img
    content = np.zeros((m, m), bool)
    content[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = True
    return (cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA),
            cv2.resize(content.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST).astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val")
    ap.add_argument("--v30", default="/home/robotis/robot_aiworker/datasets/screwing35_follower_val_v3.0")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--frame", type=int, default=None, help="default: 75%% through the driver grasp")
    ap.add_argument("--side", type=int, default=256, help="model input side (shortest_image_edge)")
    ap.add_argument("--only-cam", default=None,
                    help="draw only the camera whose name contains this. The model still receives "
                         "all three and the attention is still computed over all image tokens -- "
                         "this filters the DISPLAY only.")
    ap.add_argument("--tile", type=int, default=None, help="panel tile size (default = --side)")
    ap.add_argument("--out", default="attn_map.png")
    ap.add_argument("--video", action="store_true", help="sweep a frame range and encode an mp4")
    ap.add_argument("--subtask", type=int, default=2,
                    help="--video: which subtask to sweep; -1 = the WHOLE episode")
    ap.add_argument("--stride", type=int, default=4, help="--video: sample every Nth frame")
    ap.add_argument("--fps", type=int, default=10, help="--video: playback fps")
    ap.add_argument("--out-notext", default=None,
                    help="--video: also write this file WITHOUT the instruction overlay "
                         "(same inference pass, so it is free)")
    a = ap.parse_args()

    import torch, cv2
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.eval.open_loop_eval import parse_observation_gr00t
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.model.modules.dit import AlternateVLDiT, DiT
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

    policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=a.ckpt,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    mc = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc, video_backend="torchcodec")
    traj = loader[a.episode]

    v30 = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(str(Path(a.v30) / "data/**/*.parquet"), recursive=True))])
    sub = v30[v30.episode_index == a.episode].sort_values("frame_index").subtask_index.to_numpy()
    if a.frame is None:
        idx = np.where(sub == 2)[0]                      # "Grab the driver"
        a.frame = int(idx.min() + 0.75 * (idx.max() - idx.min()))
    phase = int(sub[a.frame])
    print(f"episode {a.episode} frame {a.frame} -- subtask {phase} '{SUB[phase]}'")

    # ---- hooks
    grab = {"thw": None, "image_mask": None, "attn": []}
    net = policy.model if hasattr(policy, "model") else policy
    for _, m in net.named_modules():
        if isinstance(m, Qwen3Backbone):
            m.register_forward_pre_hook(
                lambda mod, args, kw: grab.__setitem__("thw", (args[0] if args else kw["vl_input"])["image_grid_thw"]),
                with_kwargs=True)
    dit = next((m for _, m in net.named_modules() if isinstance(m, (AlternateVLDiT, DiT))), None)
    if dit is None: raise SystemExit("no DiT found")
    n_text = int(getattr(dit, "attend_text_every_n_blocks", 2))
    dit.register_forward_pre_hook(
        lambda mod, args, kw: grab.__setitem__("image_mask", kw.get("image_mask")), with_kwargs=True)
    for name, m in net.named_modules():
        if name.endswith(".attn1") and hasattr(m, "to_q") and hasattr(m, "to_k"):
            def mk(nm, mod):
                def pre(module, args, kwargs):
                    hs = args[0] if args else kwargs.get("hidden_states")
                    eh = kwargs.get("encoder_hidden_states")
                    if eh is not None and hs is not None:
                        grab["attn"].append((nm,
                                             hs.detach().float().cpu().numpy(),
                                             eh.detach().float().cpu().numpy(), mod))
                return pre
            m.register_forward_pre_hook(mk(name, m), with_kwargs=True)

    # ---- one frame -> (panel BGR, per-camera stats)
    def render(t_frame):
      grab["attn"] = []
      st = extract_step_data(traj, t_frame, mc, "new_embodiment")
      obs = {f"state.{k}": v for k, v in st.states.items()}
      raw_imgs = {k: np.array(v) for k, v in st.images.items()}
      for k, v in raw_imgs.items(): obs[f"video.{k}"] = v
      obs["annotation.human.primitive_instruction"] = st.text
      with torch.no_grad():
        policy.get_action(parse_observation_gr00t(obs, mc))

      im_mask = grab["image_mask"]
      if im_mask is None: raise SystemExit("image_mask not captured")
      im_idx = im_mask[0].bool().cpu().numpy()

      # ---- action -> image attention, IMAGE cross-attn blocks only
      heat, used = None, 0
      grab["per_block"] = per_block = []
      with torch.no_grad():
        for nm, hs_np, eh_np, mod in grab["attn"]:
            if eh_np.shape[1] != im_mask.shape[1]: continue
            mm = re.search(r"transformer_blocks\.(\d+)\.", nm)
            idx = int(mm.group(1)) if mm else -1
            if not (idx >= 0 and idx % 2 == 0 and idx % (2 * n_text) != 0): continue
            dev = next(mod.parameters()).device
            dt = next(mod.parameters()).dtype
            hs = torch.from_numpy(hs_np).to(dev, dt)
            eh = torch.from_numpy(eh_np[:, im_idx, :]).to(dev, dt)
            H = mod.heads
            q = mod.to_q(hs); k = mod.to_k(eh)
            d = q.shape[-1] // H
            q = q.view(q.shape[0], q.shape[1], H, d).permute(0, 2, 1, 3)
            k = k.view(k.shape[0], k.shape[1], H, d).permute(0, 2, 1, 3)
            p = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (d ** 0.5), dim=-1)
            h = p.mean(dim=(1, 2))[0]
            #per-BLOCK selectivity, before any averaging across blocks/steps. If individual blocks
            #are peaked and only the average is flat, the flatness is my aggregation, not the model.
            _pr = (h / h.sum().clamp_min(1e-12)).float().cpu().numpy()
            _e = float(-(_pr * np.log(_pr + 1e-12)).sum() / np.log(_pr.size))
            #also the single sharpest (head, query) pair in this block
            _pq = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (d ** 0.5), dim=-1)[0]
            _pq = _pq.reshape(-1, _pq.shape[-1])
            _eq = (-(_pq * (_pq + 1e-12).log()).sum(-1) / np.log(_pq.shape[-1])).min().item()
            per_block.append((idx, _e, _eq))
            heat = h if heat is None else heat + h
            used += 1
      if heat is None: raise SystemExit("no image cross-attention block captured")
      heat = (heat / used).float().cpu().numpy()

      #How INFORMATIVE is this attention? If it is near-uniform, "x% on padding" is just the
      #padding's share of the token grid and says nothing about what the policy uses.
      #  norm_entropy 1.0 = perfectly uniform (no selectivity), 0.0 = all mass on one token.
      #  top8_share   = fraction of mass in the 8 strongest of 64 tokens; uniform would give 12.5%.
      n_tok = heat.size
      pr = heat / max(heat.sum(), 1e-12)
      ent = float(-(pr * np.log(pr + 1e-12)).sum() / np.log(n_tok))
      top8 = float(np.sort(pr)[::-1][:8].sum())
      grab["stats_info"] = (ent, top8, n_tok, per_block)

      # ---- split per camera and render
      cams = list(raw_imgs.keys())
      thw = grab["thw"]
      tiles, stats = [], []
      off = 0
      lo, hi = (float(np.percentile(heat, 2)), float(np.percentile(heat, 98)))
      #shared across the three cameras so panels are comparable, but percentile-clipped: a plain
      #min/max lets one outlier token crush the other two panels to flat blue.
      for j, cam in enumerate(cams):
        t, gh, gw = [int(x) for x in thw[j]]
        hh, ww = gh // 2, gw // 2
        seg = heat[off: off + t * hh * ww][: hh * ww]; off += t * hh * ww
        hm = seg.reshape(hh, ww)

        img = raw_imgs[cam]
        if img.ndim == 4: img = img[0]                       #extract_step_data keeps a time axis
        if img.ndim == 3 and img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8: img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
        shown, content = letterbox(img, a.side)

        # attention mass landing on the black bars
        cm = cv2.resize(content.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_AREA).astype(bool)
        pad_frac = float(seg.reshape(hh, ww)[~cm].sum() / max(seg.sum(), 1e-9)) if (~cm).any() else 0.0
        stats.append((cam, hh, ww, pad_frac, float((~cm).mean()), float(seg.sum() / max(heat.sum(), 1e-9))))

        n = np.clip((hm - lo) / (hi - lo + 1e-9), 0, 1)        #shared, percentile-clipped
        up = cv2.resize(n, (a.side, a.side), interpolation=cv2.INTER_CUBIC)
        hc = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        tiles.append(cv2.addWeighted(cv2.cvtColor(shown, cv2.COLOR_RGB2BGR), 0.55, hc, 0.45, 0))

      if a.only_cam:
          keep = [i for i, st_ in enumerate(stats) if a.only_cam in st_[0]]
          if not keep: raise SystemExit(f"--only-cam {a.only_cam!r} matched none of "
                                        f"{[s_[0] for s_ in stats]}")
          tiles = [tiles[i] for i in keep]; stats = [stats[i] for i in keep]
      if a.tile and a.tile != a.side:
          tiles = [cv2.resize(t_, (a.tile, a.tile), interpolation=cv2.INTER_LINEAR) for t_ in tiles]
      TS = a.tile or a.side
      pad = 8
      W = sum(t.shape[1] for t in tiles) + pad * (len(tiles) + 1)
      panel = np.full((TS + 116, W, 3), 24, np.uint8)
      x = pad
      for t_, (cam, hh, ww, pf, pa, share) in zip(tiles, stats):
        panel[70:70 + TS, x:x + t_.shape[1]] = t_
        cv2.putText(panel, cam.replace("cam_", ""), (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{ww}x{hh} tokens   attention share {share:.0%}", (x, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 185), 1, cv2.LINE_AA)
        cv2.putText(panel, f"black bars = {pa:.0%} of frame", (x, TS + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 185), 1, cv2.LINE_AA)
        cv2.putText(panel, f"ATTENTION ON PADDING {pf:.1%}", (x, TS + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 140, 255), 1, cv2.LINE_AA)
        x += t_.shape[1] + pad
      ph = int(sub[t_frame]) if t_frame < len(sub) else 0
      cv2.putText(panel, f"ep{a.episode}  f{t_frame}  subtask {ph}/4  |  {used} image cross-attn blocks",
                  (pad, TS + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 155), 1, cv2.LINE_AA)

      #the instruction is drawn only on the second copy, so one inference pass yields both a
      #with-language and a without-language video
      panel_txt = panel.copy()
      cv2.putText(panel_txt, f'"{st.text}"', (pad, TS + 34),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 230, 255), 2, cv2.LINE_AA)
      return panel, panel_txt, stats, st.text, used

    if not a.video:
        panel, _pt, stats, text, used = render(a.frame)
        cv2.imwrite(a.out, panel)
        ent, top8, n_tok, per_block = grab["stats_info"]
        print(f"instruction: {text!r}\naggregated over {used} image cross-attn captures")
        print(f"  SELECTIVITY  normalised entropy {ent:.4f}  (1.000 = uniform, no information)")
        print(f"               top-8 of {n_tok} tokens hold {top8:.1%}  (uniform would be {8/n_tok:.1%})")
        if per_block:
            import collections
            byb = collections.defaultdict(list)
            for i, e, eq in per_block: byb[i].append((e, eq))
            print(f"  per-block entropy (averaged over the 4 denoising steps):")
            for i in sorted(byb):
                es = [x[0] for x in byb[i]]; eqs = [x[1] for x in byb[i]]
                print(f"     block {i:>2}   mean-over-heads {np.mean(es):.4f}   "
                      f"sharpest single (head,query) {np.min(eqs):.4f}")
        print(f"wrote {a.out}")
        for cam, hh, ww, pf, pa, share in stats:
            print(f"  {cam:<18} {ww}x{hh} tokens   share {share:5.1%}   black bars {pa:5.1%}   attention on padding {pf:6.2%}")
        return

    if a.subtask < 0:
        idx = np.arange(len(sub)); what = "WHOLE EPISODE (all 5 subtasks)"
    else:
        idx = np.where(sub == a.subtask)[0]; what = f"subtask {a.subtask} '{SUB[a.subtask]}'"
    frames = list(range(int(idx.min()), int(idx.max()) + 1, a.stride))
    print(f"video: {what}, frames {frames[0]}..{frames[-1]} stride {a.stride} -> {len(frames)} frames")
    writer, writer2, trend = None, None, []
    import time; t0 = time.time()
    for i, f in enumerate(frames):
        panel, panel_txt, stats, text, _ = render(f)
        trend.append([pf for _, _, _, pf, _, _ in stats])
        for pn in (panel, panel_txt):
            cv2.rectangle(pn, (0, pn.shape[0] - 4),
                          (int(pn.shape[1] * (i + 1) / len(frames)), pn.shape[0]), (120, 200, 255), -1)
        if writer is None:
            H, W = panel.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(a.out, fourcc, a.fps, (W, H))
            if a.out_notext:
                writer2 = cv2.VideoWriter(a.out_notext, fourcc, a.fps, (W, H))
        writer.write(panel_txt)
        if writer2 is not None: writer2.write(panel)
        if i % 10 == 0 or i == len(frames) - 1:
            print(f"  {i+1}/{len(frames)}  f{f}  padding-attn "
                  + " ".join(f"{x:.0%}" for x in trend[-1]) + f"   ({time.time()-t0:.0f}s)", flush=True)
    writer.release()
    if writer2 is not None: writer2.release()
    tr = np.array(trend)
    print(f"\nwrote {a.out}  ({len(frames)} frames @ {a.fps} fps)")
    cams = [c for c, _, _, _, _, _ in stats]
    print(f"\n  attention on padding, across the grasp:")
    for j, c in enumerate(cams):
        q = tr[:, j]
        print(f"    {c:<18} mean {q.mean():6.1%}   min {q.min():5.1%}   max {q.max():5.1%}   "
              f"first->last {q[0]:.0%} -> {q[-1]:.0%}")


if __name__ == "__main__":
    main()
