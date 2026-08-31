#!/usr/bin/env python3
"""Fit the intervention gate, turn it into a reward, and test the assumption it all rests on.

This is stages 2-4 of HIL_PIPELINE.md and the implementation of HIL_THEORY.md. It runs entirely
offline on collected DAgger rounds -- no robot, no simulator -- because the point is to kill the
idea cheaply if it is wrong.

WHAT IT COMPUTES
  p_hat(s,a)      a classifier on EVERY timestep's intervene/don't-intervene label. Not just the
                  interventions: the ~95% of steps where the operator did nothing are labels too,
                  and they are the reason this costs no extra human time.
  r' = -logit p   the reward. By HIL_THEORY.md (9) this is the task reward plus potential-based
                  shaping with Phi = V^H, so by Ng-Harada-Russell (1999) it has the SAME optimal
                  policy as the task, while being dense where the success bit is sparse.

THE THREE GATES, in the order they should kill the idea
  A1   do operators threshold on REGRET, or on habit and impatience? Regress realised return
       difference on logit p_hat. No positive slope -> the reduction is void and nothing below
       matters. This is the load-bearing assumption and it is about people, not mathematics.
  A2   is the classifier calibrated enough for its logit to be an affine readout of regret?
       Reliability curve + Brier decomposition. Mis-calibration costs scale, not direction, so
       this is the weakest of the three.
  EFF  at a threshold that queries the operator only ~8% of the time, what fraction of their real
       interventions does the gate catch? Recall at fixed labour. This is the efficiency claim
       stated as a number rather than a hope.

Usage:
    python3 gate_model.py --rounds dagger_round1 [dagger_round2 ...]
    python3 gate_model.py --rounds ... --query-rate 0.08
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


# ── data ──────────────────────────────────────────────────────────────────────────────────────────

def load_rounds(rounds):
    """Every timestep of every episode, with the gate label and both actions."""
    import pandas as pd
    frames = []
    for rd in rounds:
        files = sorted(glob.glob(str(Path(rd) / "data" / "**" / "*.parquet"), recursive=True))
        if not files:
            print(f"  [warn] no parquet under {rd}")
        for f in files:
            d = pd.read_parquet(f)
            if "intervention" not in d:
                print(f"  [warn] {f} has no intervention column, skipping")
                continue
            if "action_policy" not in d:
                #Collected before the schema fix. The rejected proposal is unrecoverable, so the
                #per-action features degrade to state-only. Say so rather than silently proceeding.
                print(f"  [warn] {f} predates the action_policy column -- "
                      f"regret features will be state-only for it")
                d = d.assign(action_policy=d["action"])
            d = d.assign(_round=Path(rd).name, _file=Path(f).name)
            frames.append(d)
    if not frames:
        raise SystemExit("no usable episodes found")
    return pd.concat(frames, ignore_index=True)


def featurise(df):
    """(state, policy action, and the disagreement between proposal and execution)."""
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    A = np.stack(df["action_policy"].to_numpy()).astype(np.float64)
    X = np.hstack([S, A, A - S[:, :A.shape[1]] if A.shape[1] <= S.shape[1] else A])
    return X


# ── model ─────────────────────────────────────────────────────────────────────────────────────────

def _irls(X, y, w_obs=None, l2=1.0, iters=200):
    """Logistic regression by Newton/IRLS, in numpy.

    Written out rather than imported: this machine has no scikit-learn, and more importantly a
    result that silently inherits a library's regularisation default is not reproducible. IRLS is
    the exact Newton step for the log-loss, so there is no learning rate that could hide a bad fit.
    `w_obs` carries per-sample weights, used here to balance the classes -- interventions are ~5% of
    timesteps and an unweighted fit would predict "never intervene" and score 95% accuracy.
    """
    A = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(A.shape[1])
    ow = np.ones(len(X)) if w_obs is None else w_obs
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9) * ow
        g = A.T @ (ow * (p - y)) + l2 * w
        H = (A * W[:, None]).T @ A + l2 * np.eye(A.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def _auc(y, s):
    """ROC AUC by rank statistic (Mann-Whitney U), ties averaged."""
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    us, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(us)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


class _Fitted:
    """Minimal stand-in for the sklearn object the rest of this file used."""
    def __init__(self, w, mu, sd):
        self.w, self.mu, self.sd = w, mu, sd

    def decision_function(self, X):
        Z = (X - self.mu) / self.sd
        return np.hstack([Z, np.ones((len(Z), 1))]) @ self.w

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.stack([1 - p, p], 1)


def fit_logistic(X, y, groups, l2=1.0, seed=0):
    """Logistic regression, held out BY EPISODE.

    Frames from one episode are 30 Hz samples of one trajectory; splitting them at random puts
    near-duplicates on both sides and reports a score memorisation alone can reach.
    """
    rng = np.random.default_rng(seed)
    eps = np.array(sorted(set(groups.tolist())))
    rng.shuffle(eps)
    te_eps = set(eps[: max(1, len(eps) // 4)].tolist())
    te = np.array([g in te_eps for g in groups])
    tr = ~te
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    #class balancing: without it the fit predicts "never intervene" and scores 95% accuracy
    ytr = y[tr]
    ow = np.where(ytr == 1, 0.5 / max(ytr.mean(), 1e-6), 0.5 / max(1 - ytr.mean(), 1e-6))
    w = _irls((X[tr] - mu) / sd, ytr, w_obs=ow, l2=l2)
    clf = _Fitted(w, mu, sd)
    return clf, None, tr, te


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ── the three gates ───────────────────────────────────────────────────────────────────────────────

def test_A1(df, lg, te):
    """Do operators threshold on regret? Realised return difference vs logit p_hat.

    Proxy for return difference: how much the executed action departed from the policy's proposal,
    summed over the intervention. Under A1 an operator only pays that cost when the regret is large,
    so the two should rise together. It is a proxy -- a true return difference needs the outcome
    label, which arrives with `success` in the episode metadata.
    """
    A_exec = np.stack(df["action"].to_numpy()).astype(np.float64)
    A_pol = np.stack(df["action_policy"].to_numpy()).astype(np.float64)
    dev = np.abs(A_exec - A_pol).mean(1)                 #how hard the operator pulled
    x, y = lg[te], dev[te]
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    slope = np.polyfit(x, y, 1)[0]
    return dict(spearman=float(rho), p_value=float(p), slope=float(slope), n=int(te.sum()))


def test_A2(y_true, p_hat, bins=10):
    """Calibration: is logit p_hat an affine readout of regret, or just an ordering?"""
    edges = np.quantile(p_hat, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for i in range(bins):
        m = (p_hat >= edges[i]) & (p_hat < edges[i + 1])
        if m.sum() < 5:
            continue
        rows.append((float(p_hat[m].mean()), float(y_true[m].mean()), int(m.sum())))
    brier = float(np.mean((p_hat - y_true) ** 2))
    base = float(np.mean((y_true.mean() - y_true) ** 2))
    return rows, brier, base


def test_EFF(y_true, p_hat, query_rate):
    """At a threshold that queries only `query_rate` of timesteps, what recall do we get?"""
    thr = np.quantile(p_hat, 1 - query_rate)
    fired = p_hat >= thr
    tp = int((fired & (y_true == 1)).sum())
    n_int = int((y_true == 1).sum())
    return dict(threshold=float(thr), query_rate=float(fired.mean()),
                recall=float(tp / max(n_int, 1)),
                precision=float(tp / max(fired.sum(), 1)), interventions=n_int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--query-rate", type=float, default=0.08,
                    help="labour budget: fraction of timesteps the operator is asked about")
    ap.add_argument("--out", default="/home/robotis/robot_aiworker/act_ab/hil_gate.npz")
    a = ap.parse_args()

    df = load_rounds(a.rounds)
    y = df["intervention"].to_numpy().astype(int)
    groups = df["episode_index"].to_numpy()
    print(f"\n{len(df)} timesteps, {len(set(groups.tolist()))} episodes, "
          f"{y.mean()*100:.1f}% intervened")
    if y.sum() < 20:
        raise SystemExit("fewer than 20 intervention frames -- collect more before judging anything")

    X = featurise(df)
    clf, _, tr, te = fit_logistic(X, y, groups)
    p_hat = clf.predict_proba(X)[:, 1]
    lg = logit(p_hat)

    auc = _auc(y[te], p_hat[te])
    print(f"held-out AUC (by episode)        {auc:.3f}   "
          f"{'-- better than chance' if auc > 0.5 else '-- NO SIGNAL'}")

    print("\n--- GATE A1: do operators threshold on regret? ---")
    r = test_A1(df, lg, te)
    if r is None:
        print("  not computable on this data")
    else:
        print(f"  spearman(logit p_hat, operator effort) = {r['spearman']:+.3f}  "
              f"(p={r['p_value']:.2g}, n={r['n']})")
        verdict = ("PASS -- consistent with A1" if r["spearman"] > 0.1 and r["p_value"] < 0.05
                   else "FAIL -- A1 is not supported; the reduction in HIL_THEORY.md is VOID")
        print(f"  {verdict}")

    print("\n--- GATE A2: calibration ---")
    rows, brier, base = test_A2(y[te], p_hat[te])
    print(f"  Brier {brier:.4f} vs base-rate {base:.4f}  "
          f"({'informative' if brier < base else 'NO BETTER THAN THE BASE RATE'})")
    for pm, ym, n in rows:
        print(f"    predicted {pm:.3f}   observed {ym:.3f}   n={n}")

    print(f"\n--- GATE EFF: recall at {100*a.query_rate:.0f}% labour ---")
    e = test_EFF(y[te], p_hat[te], a.query_rate)
    print(f"  threshold {e['threshold']:.3f}   queries {100*e['query_rate']:.1f}% of timesteps")
    print(f"  catches {100*e['recall']:.1f}% of the operator's real interventions "
          f"(precision {100*e['precision']:.1f}%)")
    print(f"  {'PASS' if e['recall'] > 0.5 else 'WEAK -- the gate cannot replace the operator yet'}")

    np.savez(a.out, p_hat=p_hat, logit=lg, y=y, groups=groups,
             reward=-lg,          #HIL_THEORY.md (9): r' = -logit p_hat, up to a positive affine map
             te=te)
    print(f"\nwrote {a.out}   (`reward` is the drop-in replacement for the success classifier)")








# ── self-test ─────────────────────────────────────────────────────────────────────────────────────

def self_test():
    """Exercise the whole path on synthetic DAgger-shaped data, where the truth is known.

    No real rounds exist yet, and code that has never run is not code. This generates episodes from
    a SIMULATED operator who obeys A1 exactly -- intervening when a known regret exceeds a threshold
    -- and checks the pipeline recovers it. A negative control with a coin-flipping operator must
    produce AUC ~0.5, otherwise the test would pass on anything.
    """
    import pandas as pd, tempfile, os
    rng = np.random.default_rng(0)
    D, H = 16, 16

    def make(rounds_dir, informative=True, n_ep=24, T=300, tau_drift=0.0):
        w = rng.normal(size=D)
        #GLOBAL threshold, matching A1 as stated: g = 1{R > tau} with tau FIXED. The first version
        #of this generator recomputed tau per episode, i.e. an operator whose standard drifts from
        #episode to episode -- and the held-out AUC collapsed to 0.587 even though the regret was
        #perfectly linear in the state. That is a real finding about the method, not a bug: a single
        #global gate cannot track a moving threshold, because 1{R > tau_e} is not a function of the
        #state alone. `tau_drift` reproduces it on demand and it is measured as a third case below.
        probe = np.cumsum(rng.normal(0, .05, (4000, D)), 0) @ w
        tau0 = float(np.quantile(probe, 0.92))            # ~8% intervention rate, as measured
        for e in range(n_ep):
            S = np.cumsum(rng.normal(0, .05, (T, D)), 0)
            Apol = S + rng.normal(0, .02, (T, D))
            R = S @ w                                     # the true regret, hidden from the fitter
            tau = tau0 + tau_drift * rng.normal() * np.std(probe)
            p = 1 / (1 + np.exp(-(R - tau) / 0.3))
            g = (rng.random(T) < (p if informative else 0.08)).astype(np.int8)
            Aex = Apol + g[:, None] * rng.normal(0, .3, (T, D))   # operator pulls when they act
            d = os.path.join(rounds_dir, "data", "chunk-000")
            os.makedirs(d, exist_ok=True)
            pd.DataFrame({
                "observation.state": list(S.astype(np.float32)),
                "action": list(Aex.astype(np.float32)),
                "action_policy": list(Apol.astype(np.float32)),
                "intervention": g,
                "pre_intervention": np.zeros(T, np.int8),
                "episode_index": np.full(T, e),
            }).to_parquet(os.path.join(d, f"file-{e:03d}.parquet"))

    cases = (("A1-obeying operator (fixed threshold)", True, 0.0),
             ("coin-flip operator (negative control)", False, 0.0),
             ("A1 operator with a DRIFTING threshold", True, 0.5))
    for label, informative, drift in cases:
        with tempfile.TemporaryDirectory() as td:
            make(td, informative, tau_drift=drift)
            df = load_rounds([td])
            y = df["intervention"].to_numpy().astype(int)
            groups = df["episode_index"].to_numpy()
            X = featurise(df)
            clf, _, tr, te = fit_logistic(X, y, groups)
            p_hat = clf.predict_proba(X)[:, 1]
            auc = _auc(y[te], p_hat[te])
            eff = test_EFF(y[te], p_hat[te], 0.08)
            a1 = test_A1(df, logit(p_hat), te)
            print(f"\n  {label}")
            print(f"    held-out AUC                {auc:.3f}")
            print(f"    A1 spearman                 {a1['spearman']:+.3f} (p={a1['p_value']:.1e})")
            print(f"    recall at 8% query budget   {100*eff['recall']:.1f}%")
            if not informative:
                ok, want = (0.35 < auc < 0.65), "no signal"
            elif drift == 0.0:
                ok, want = (auc > 0.75), "strong signal"
            else:
                #not a pass/fail -- this case exists to MEASURE how much operator drift costs
                ok, want = True, "degraded by drift (measurement, not a gate)"
            print(f"    {'PASS' if ok else 'FAIL'} -- expected {want}")


if __name__ == "__main__":
    #ONE entry point, placed after every definition. The first version put the --self-test guard
    #above self_test()'s own def, so it raised NameError -- the failure mode of scattering module
    #-level guards through a file instead of keeping a single one at the bottom.
    import sys
    if "--self-test" in sys.argv:
        print("=== gate_model self-test on synthetic data ===")
        self_test()
    else:
        main()
