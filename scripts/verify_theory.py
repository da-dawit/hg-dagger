#!/usr/bin/env python3
"""Verify every step of HIL_THEORY.md numerically, on data where the truth is known.

Algebra hides sign errors and estimators fail in ways derivations do not. Each claim below is
re-derived as a computation with a KNOWN ground truth, and each has a negative control so a passing
result cannot be a tautology.

    T1   a state-dependent baseline contributes zero to the policy gradient
    T2   a positive affine map of the advantage preserves the ascent direction
    T3   fitting a classifier to thresholded-regret labels recovers regret up to a positive
         affine map -- the estimator, not just the algebra
    T4   -R used as a REWARD preserves the optimal policy (potential shaping, Phi = V^H)
    T5   the reduction still works when the human is NOISY and SUBOPTIMAL, which is the only
         regime that exists in practice

Run:  python3 verify_theory.py
Exit code 0 iff every claim passes.
"""
from __future__ import annotations
import numpy as np

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")


# ── T1 / T2 : the policy-gradient identities ──────────────────────────────────────────────────────

def t1_t2():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    S, A = 6, 4
    pol = nn.Sequential(nn.Linear(S, 16), nn.Tanh(), nn.Linear(16, A))
    s = torch.randn(S)
    p = torch.softmax(pol(s), -1).detach()

    def grad_of(coef):
        g = None
        for a in range(A):
            pol.zero_grad()
            torch.log_softmax(pol(s), -1)[a].backward(retain_graph=True)
            gv = torch.cat([q.grad.reshape(-1) for q in pol.parameters()])
            g = p[a] * coef(a) * gv if g is None else g + p[a] * coef(a) * gv
        return g

    Adv = np.random.default_rng(0).normal(size=A) * 0.7
    g_true = grad_of(lambda a: float(Adv[a]))
    g_base = grad_of(lambda a: float(Adv[a] - 3.14159))
    g_aff = grad_of(lambda a: float(2.5 * Adv[a] - 0.83))
    g_neg = grad_of(lambda a: float(-Adv[a]))

    rel = float(torch.norm(g_base - g_true) / torch.norm(g_true))
    cos = lambda x, y: float((x @ y) / (torch.norm(x) * torch.norm(y)))
    report("T1  baseline invariance", rel < 1e-5,
           f"||grad(A-b) - grad(A)|| / ||grad(A)|| = {rel:.2e}   (target 0)")
    report("T2  positive affine preserves direction", abs(cos(g_aff, g_true) - 1) < 1e-6,
           f"cos = {cos(g_aff, g_true):.10f}, norm ratio = "
           f"{float(torch.norm(g_aff)/torch.norm(g_true)):.4f} (should equal beta=2.5)")
    report("T2  negative control (sign flip must flip the gradient)",
           abs(cos(g_neg, g_true) + 1) < 1e-6, f"cos = {cos(g_neg, g_true):+.6f}   (target -1)")


# ── T3 : the ESTIMATOR recovers regret from binary labels ─────────────────────────────────────────

def _logreg(X, y, iters=400, l2=1e-4):
    """Logistic regression by Newton/IRLS, in numpy.

    Written out rather than imported so this verification depends on nothing but numpy: a proof
    script that silently inherits a library's regularisation default is not a proof. IRLS is the
    exact Newton step for the log-loss, so it converges in tens of iterations and there is no
    learning rate to tune away a bad result.
    """
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        g = X.T @ (p - y) + l2 * w
        H = (X * W[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w                       # decision_function(X) = [X,1] @ w


def t3(n=40_000, d=8, seed=0):
    """Ground truth R is known by construction. Fit only on g = 1{R > tau} + logistic noise."""
    from scipy.stats import pearsonr
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    R = X @ w                                          # the true regret, never shown to the fitter
    tau, beta = 0.7, 0.5
    p = 1 / (1 + np.exp(-(R - tau) / beta))            # A2's logistic link
    g = (rng.random(n) < p).astype(int)                # the ONLY thing the fitter sees

    w = _logreg(X, g)
    lg = np.hstack([X, np.ones((len(X), 1))]) @ w      # logit p_hat
    r_pear = pearsonr(lg, R)[0]
    # recovered affine map: R ~= c0 + c1 * logit
    c1, c0 = np.polyfit(lg, R, 1)
    report("T3  classifier logit recovers regret up to affine", r_pear > 0.99,
           f"pearson(logit p_hat, true R) = {r_pear:.4f}; recovered R ~ {c0:+.3f} + {c1:.3f}*logit "
           f"(true beta={beta} => slope should be ~{beta:.3f}), c1>0: {c1 > 0}")

    # NEGATIVE CONTROL, done as a PERMUTATION TEST rather than against a threshold I pick.
    #
    # The first version of this control compared |pearson| against a hard 0.1 and "failed" at 0.371.
    # That was a defect in the TEST, not in the estimator: a classifier fitted to random labels
    # produces a random direction in R^d, and the correlation between two random directions in d
    # dimensions concentrates near 1/sqrt(d) -- 0.354 for d=8. A fixed cutoff of 0.1 was therefore
    # unreachable by construction. Measured: E|cos(random,random)| = 0.29 at d=8, 0.20 at d=16.
    #
    # The statistically correct control generates the null distribution instead of assuming it, and
    # asks whether the observed statistic lies outside it.
    null = []
    for k in range(30):
        g_rand = (rng.random(n) < g.mean()).astype(int)      # same base rate, no signal
        w_r = _logreg(X, g_rand)
        null.append(abs(pearsonr(np.hstack([X, np.ones((len(X), 1))]) @ w_r, R)[0]))
    null = np.array(null)
    p99 = float(np.percentile(null, 99))
    report("T3  negative control (permutation test)", r_pear > p99,
           f"observed |r| = {r_pear:.4f} vs null distribution mean {null.mean():.3f}, "
           f"99th pct {p99:.3f}  ->  observed is {'OUTSIDE' if r_pear > p99 else 'INSIDE'} the null")


# ── T4 : -R as a REWARD preserves the optimal policy ──────────────────────────────────────────────

def t4(trials=200, S=12, A=4, gamma=0.95, seed=0):
    rng = np.random.default_rng(seed)

    def vi(P, R, iters=2000):
        V = np.zeros(P.shape[0])
        for _ in range(iters):
            Q = R + gamma * P @ V
            V = Q.max(1)
        return Q.argmax(1), Q, V

    bad = 0
    for _ in range(trials):
        P = rng.random((S, A, S)); P /= P.sum(-1, keepdims=True)
        R = rng.random((S, A)) * (rng.random((S, A)) < 0.15)      # sparse, like a success bit
        pi_star, _, _ = vi(P, R)
        pi_H = pi_star.copy()
        flip = rng.random(S) < 0.25                                # a NEAR-optimal human
        pi_H[flip] = rng.integers(0, A, flip.sum())
        V_H = np.linalg.solve(np.eye(S) - gamma * P[np.arange(S), pi_H],
                              R[np.arange(S), pi_H])
        R_shaped = R + gamma * (P @ V_H) - V_H[:, None]            # = -R(s,a), by HIL_THEORY (9)
        pi_sh, _, _ = vi(P, R_shaped)
        bad += int(not np.array_equal(pi_star, pi_sh))
    report("T4  -R as reward preserves the optimal policy", bad == 0,
           f"{bad}/{trials} random MDPs changed optimum under Phi=V_H shaping   (target 0)")

    bad2 = 0
    for _ in range(trials):
        P = rng.random((S, A, S)); P /= P.sum(-1, keepdims=True)
        R = rng.random((S, A)) * (rng.random((S, A)) < 0.15)
        pi_star, _, _ = vi(P, R)
        pi_b, _, _ = vi(P, R + rng.normal(0, 0.5, (S, A)))         # action-dependent: NOT a potential
        bad2 += int(not np.array_equal(pi_star, pi_b))
    report("T4  negative control (non-potential shaping must break it)", bad2 > trials // 2,
           f"{bad2}/{trials} changed optimum   (must be large, else the test is vacuous)")


# ── T5 : does it survive a noisy, suboptimal, drifting human? ─────────────────────────────────────

def t5(trials=120, S=12, A=4, gamma=0.95, seed=1):
    """The practical regime. Phi = V^H for a MEDIOCRE human is still a valid potential -- Ng's
    theorem asks nothing of Phi except that it be a function of state. So the optimum should be
    preserved no matter how bad the operator is; only the SHAPING QUALITY degrades."""
    rng = np.random.default_rng(seed)

    def vi(P, R, iters=2000):
        V = np.zeros(P.shape[0])
        for _ in range(iters):
            Q = R + gamma * P @ V
            V = Q.max(1)
        return Q.argmax(1), Q, V

    for frac, label in ((0.25, "near-optimal"), (0.60, "mediocre"), (1.00, "random")):
        bad = 0
        for _ in range(trials):
            P = rng.random((S, A, S)); P /= P.sum(-1, keepdims=True)
            R = rng.random((S, A)) * (rng.random((S, A)) < 0.15)
            pi_star, _, _ = vi(P, R)
            pi_H = pi_star.copy()
            m = rng.random(S) < frac
            pi_H[m] = rng.integers(0, A, m.sum())
            V_H = np.linalg.solve(np.eye(S) - gamma * P[np.arange(S), pi_H],
                                  R[np.arange(S), pi_H])
            pi_sh, _, _ = vi(P, R + gamma * (P @ V_H) - V_H[:, None])
            bad += int(not np.array_equal(pi_star, pi_sh))
        report(f"T5  optimum preserved with a {label} operator", bad == 0,
               f"{bad}/{trials} changed   -- the learner's ceiling is the TASK optimum, "
               f"not the operator's skill")


if __name__ == "__main__":
    print("=" * 78)
    print("Verifying HIL_THEORY.md, claim by claim, with negative controls")
    print("=" * 78)
    t1_t2(); print()
    t3();    print()
    t4();    print()
    t5()
    print("\n" + "=" * 78)
    ok = all(RESULTS)
    print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed -- "
          f"{'every claim verified' if ok else 'SOMETHING IS WRONG, do not build on this'}")
    raise SystemExit(0 if ok else 1)
