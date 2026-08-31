#!/usr/bin/env python3
"""Theorem 4: the gate scale beta is identifiable from the learner's own critic.

Author: Dawit Chun

WHY THIS MATTERS. verify_theorem3.py showed that potential shaping with Phi from eq (10) leaves
the optimum invariant for ANY beta -- but that getting beta wrong by 20x destroys every bit of the
sample-efficiency gain. Correctness is free; speed is not. So beta has to come from somewhere, and
asking the human to report their own decision noise is not an option.

THE DERIVATION. V_H(s) does not depend on the action. From eq (10), for every action a in a fixed
state s,

    Q_pi(s,a) + beta * logit p_hat(s,a) = V_H(s) - tau = const(s)                  (12)

so WITHIN a state, Q_pi and logit p_hat lie on a line of slope exactly -beta. Pool the
within-state-centred pairs and the slope of the regression is -beta. No extra human labels, no
extra rollouts: the learner's critic already contains the scale.

tau never needs estimating. It shifts Phi by a constant, and a constant potential contributes a
state-independent term that the policy gradient discards.

WHERE IT FAILS, and this is the useful part. Eq (12) has content only where logit p_hat VARIES with
the action. In states the human would never question, p_hat saturates at 0 or 1, the logit is
extreme and noise-dominated, and the regression learns nothing. beta is identifiable exactly in the
states where the human is UNCERTAIN -- which is where the gate is informative in the first place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_theorem3 as V


def estimate_beta(Q_pi, logit, n_obs=None, p_hat=None, keep=None):
    """Within-state estimate of beta from eq (12). Returns (beta_hat, n_states_used).

    TWO THINGS MATTER AND BOTH ARE EASY TO GET WRONG.

    1. DIRECTION. logit p_hat is the NOISY variable (it is a finite-sample estimate of a
       probability); Q_pi is comparatively clean. Regressing the clean variable on the noisy one
       is classical errors-in-variables and attenuates the slope toward zero -- measured at -100%,
       i.e. beta_hat ~ 0 regardless of how many labels are collected. Regress logit ON Q instead
       and invert: slope -> -1/beta.

    2. WEIGHTING, NOT FILTERING. A saturated cell is uninformative because its logit is noisy, and
       the variance of a sample logit is 1/(n p (1-p)). Weighting by n*p_hat*(1-p_hat) is exactly
       inverse-variance weighting and downweights those cells smoothly. Hard-filtering them
       instead discards the cells that carry most of the spread in Q and makes the estimate worse.
    """
    w_all = (n_obs * p_hat * (1 - p_hat)) if (n_obs is not None and p_hat is not None) \
        else np.ones_like(logit)
    num = den = 0.0
    used = 0
    for s in range(Q_pi.shape[0]):
        x, y, w = Q_pi[s], logit[s], w_all[s]
        if w.sum() <= 0 or x.std() < 1e-12:
            continue
        xm = (w * x).sum() / w.sum()
        ym = (w * y).sum() / w.sum()
        xc, yc = x - xm, y - ym
        num += (w * xc * yc).sum()
        den += (w * xc * xc).sum()
        used += 1
    if den <= 0 or num == 0:
        return np.nan, used
    slope = num / den            #d logit / d Q  ->  -1/beta
    return float(-1.0 / slope), used


def estimate_beta_glm(Q_pi, k, n_obs, ridge=1e-6, iters=60):
    """Joint MLE of beta. This is what Theorem 2 actually licenses.

    The two-stage estimator -- form p_hat per cell, take logits, regress -- is biased twice over:
    errors-in-variables attenuates the slope, and the Laplace smoothing needed to keep logits
    finite shrinks p_hat toward 0.5, compressing the logit range and inflating beta. Measured:
    +182% at 20 labels/cell decaying to +34% at 2000, i.e. the bias is a smoothing artefact that
    only washes out with far more human labels than anyone will collect.

    Instead fit the model eq (12) implies, in one step, by maximum likelihood:

        P(g=1 | s,a) = sigmoid( c(s) - Q_pi(s,a)/beta )

    a binomial GLM with a free intercept per state and ONE shared coefficient on Q. No plug-in
    logits, no smoothing, and the log-loss is the proper scoring rule Theorem 2 relies on. The
    fitted coefficient on Q is -1/beta.
    """
    NSx, NAx = Q_pi.shape
    rows = NSx * NAx
    X = np.zeros((rows, NSx + 1))
    X[np.arange(rows), np.repeat(np.arange(NSx), NAx)] = 1.0
    X[:, -1] = Q_pi.ravel()
    kk = k.ravel().astype(float)
    nn = np.full(rows, float(n_obs))
    w = np.zeros(NSx + 1)
    for _ in range(iters):
        eta = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = nn * p * (1 - p) + 1e-9
        z = eta + (kk - nn * p) / W
        A = X.T @ (W[:, None] * X) + ridge * np.eye(NSx + 1)
        w_new = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    slope = w[-1]
    return (np.nan if abs(slope) < 1e-12 else float(-1.0 / slope))


def setup(n=16, gamma=0.97, query=0.08, human_eps=0.15):
    V.set_size(n)
    P, R = V.build_mdp()
    Q_opt = V.q_star(P, R, gamma)
    pol_H = V.eps_greedy(Q_opt, human_eps)
    _, V_H = V.q_pi(P, R, pol_H, gamma)
    Q_snap, _, _ = V.q_learn(P, R, gamma, np.random.default_rng(0), episodes=120,
                             P_true=P, R_true=R, eval_every=10 ** 9)
    pol_L = V.eps_greedy(Q_snap, 0.1)
    Q_L, _ = V.q_pi(P, R, pol_L, gamma)
    regret = V_H[:, None] - Q_L
    tau = np.quantile(regret, 1.0 - query)
    beta = 0.15 * regret.std()
    p_true = 1.0 / (1.0 + np.exp(-(regret - tau) / beta))
    return dict(P=P, R=R, gamma=gamma, Q_opt=Q_opt, Q_L=Q_L, V_H=V_H,
                regret=regret, tau=tau, beta=beta, p_true=p_true)


def main():
    d = setup()
    P, R, g, Q_L, beta_true = d["P"], d["R"], d["gamma"], d["Q_L"], d["beta"]
    V_opt = V.true_value(d["Q_opt"], P, R, g)
    print(f"true beta = {beta_true:.5f}   (query rate {d['p_true'].mean()*100:.1f}%)")
    print(f"V* = {V_opt:.4f}\n")

    print("PART 1  recovering beta, as human labels get scarcer")
    print(f"{'labels/cell':>13}{'two-stage':>13}{'err':>9}{'joint GLM':>13}{'err':>9}")
    print("-" * 58)
    rng = np.random.default_rng(3)
    b_glm_400 = None
    for n_obs in (20, 50, 100, 400, 2000):
        ts, gl = [], []
        for rep in range(20):
            k = rng.binomial(n_obs, d["p_true"])
            p_hat = (k + 0.5) / (n_obs + 1.0)
            lg = np.log(p_hat / (1 - p_hat))
            ts.append(estimate_beta(Q_L, lg, n_obs, p_hat)[0])
            gl.append(estimate_beta_glm(Q_L, k, n_obs))
        a_, b_ = float(np.median(ts)), float(np.median(gl))
        if n_obs == 400:
            b_glm_400 = b_
        print(f"{n_obs:>13}{a_:>13.5f}{100*(a_-beta_true)/beta_true:>8.0f}%"
              f"{b_:>13.5f}{100*(b_-beta_true)/beta_true:>8.0f}%")

    print("\nPART 2  does it survive a NOISY critic? (Q is an estimate, not exact)")
    print(f"{'critic noise':>14}{'beta_hat (GLM)':>18}{'err':>9}")
    print("-" * 42)
    k400 = rng.binomial(400, d["p_true"])
    for rel in (0.0, 0.02, 0.05, 0.10, 0.25):
        bs = [estimate_beta_glm(Q_L + rng.normal(0, rel * Q_L.std(), Q_L.shape), k400, 400)
              for _ in range(20)]
        b_ = float(np.median(bs))
        print(f"{rel:>13.0%}{b_:>18.5f}{100*(b_-beta_true)/beta_true:>8.0f}%")

    p_hat = (k400 + 0.5) / 401.0
    lg = np.log(p_hat / (1 - p_hat))
    b_ts = estimate_beta(Q_L, lg, 400, p_hat)[0]
    b_flt = b_glm_400
    print("\nPART 3  the two estimators side by side at 400 labels/cell")
    print(f"  two-stage plug-in   beta_hat {b_ts:.5f}   err {100*(b_ts-beta_true)/beta_true:+.0f}%")
    print(f"  joint GLM (Thm 4)   beta_hat {b_flt:.5f}   err {100*(b_flt-beta_true)/beta_true:+.0f}%")

    print("\nPART 4  learning with the ESTIMATED beta (the thing that actually matters)")
    tgt = 0.95 * V_opt
    def phi(b):
        p = (Q_L + d["tau"] + b * lg).mean(1)
        p[V.TERM] = 0.0
        return p
    rows = [("no shaping", None),
            ("beta 20x too small", phi(beta_true / 20)),
            ("beta ESTIMATED (Thm 4)", phi(b_flt)),
            ("beta TRUE (oracle)", phi(beta_true)),
            ("beta 20x too large", phi(beta_true * 20))]
    print(f"{'setting':<26}{'final % of V*':>16}{'episodes to 95%':>18}")
    print("-" * 62)
    for name, Phi in rows:
        vs, sols = [], []
        for sd in range(8):
            Q, _, k = V.q_learn(P, R, g, np.random.default_rng(400 + sd), 3000,
                                shaping=Phi, P_true=P, R_true=R, target=tgt)
            vs.append(V.true_value(Q, P, R, g))
            sols.append(np.nan if k is None else k)
        ns = "never" if np.all(np.isnan(sols)) else f"{np.nanmedian(sols):.0f}"
        print(f"{name:<26}{100*np.mean(vs)/V_opt:>15.1f}%{ns:>18}")

    ok = abs(b_flt - beta_true) / beta_true < 0.15
    print(f"\n[{'PASS' if ok else 'FAIL'}] beta recovered to "
          f"{100*abs(b_flt-beta_true)/beta_true:.1f}% with no additional human labels")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
