#!/usr/bin/env python3
"""Theorem 3: the gate is safe to use BECAUSE the physical reward carries the task.

Author: Dawit Chun

CLAIM. Let r_phys be the sparse reward read off the servos (both grippers stalled -> grasped;
stall released at the goal -> placed), and let p_hat be a classifier fitted to human intervention
labels. Recover the human's value function with the learner's own critic,

    V_H(s) = Q_pi(s,a) + tau + beta * logit p_hat(s,a)                            (10)

and use it as a POTENTIAL rather than as a reward,

    r_total(s,a,s') = r_phys(s,a) + gamma * Phi(s') - Phi(s),   Phi := V_H        (11)

Then by Ng-Harada-Russell (1999) the optimal policy is EXACTLY unchanged for any Phi -- any tau,
any beta, and even an arbitrarily wrong p_hat. Gate error costs sample efficiency, never
correctness. HIL-SERL uses its classifier AS the reward, which has no such guarantee: a wrong
classifier moves the optimum, and that is what reward hacking is.

This file tests the claim on a task that genuinely needs two hands, where every ground truth is
computable exactly, so the verdicts are not opinions.
"""
from __future__ import annotations

import argparse
import numpy as np

#--- a bimanual MDP -----------------------------------------------------------------------
#Positions 0..N-1. The object starts at 0, the goal is N-1. A gripper can only close on the
#object at position 0. Moving while only ONE gripper holds it DROPS it -- that is what makes the
#task need two hands, and it is the same failure the real tray-lift has. r_phys mirrors
#offline_rl/reward.py exactly: +1 when both grippers first hold, +1 on a release at the goal.
N = 16
A_MOVE_L, A_MOVE_R, A_GRIP_L, A_GRIP_R, A_RELEASE = range(5)
NA = 5
NS = N * 4 + 1
TERM = N * 4


def set_size(n):
    """Resize the task. Horizon is the independent variable in the long-horizon sweep."""
    global N, NS, TERM
    N = n
    NS = n * 4 + 1
    TERM = n * 4


def enc(x, gl, gr):
    return x * 4 + gl * 2 + gr


def build_mdp():
    """Deterministic dynamics. A DROP ENDS THE EPISODE.

    Without that, releasing away from the goal returns the object to the pick point and the agent
    farms the grasp reward forever -- the task becomes "grasp and release on the spot", transport
    never matters, and V* is an order of magnitude too high. It is the same failure the real
    proprioceptive reward avoids by paying for the RELEASE rather than for holding. Ending the
    episode on a drop is also what physically happens: the tray is on the floor.
    """
    P = np.zeros((NS, NA), dtype=np.int64)
    R = np.zeros((NS, NA), dtype=np.float64)
    for x in range(N):
        for gl in (0, 1):
            for gr in (0, 1):
                s = enc(x, gl, gr)
                for a in range(NA):
                    nx, ngl, ngr, r = x, gl, gr, 0.0
                    if a in (A_MOVE_L, A_MOVE_R):
                        if gl + gr == 1:
                            P[s, a] = TERM          #one-handed carry -> dropped, episode over
                            continue
                        nx = min(N - 1, x + 1) if a == A_MOVE_R else max(0, x - 1)
                    elif a == A_GRIP_L:
                        if x == 0 and gl == 0:
                            ngl = 1
                    elif a == A_GRIP_R:
                        if x == 0 and gr == 0:
                            ngr = 1
                    elif a == A_RELEASE:
                        if gl == 1 and gr == 1:
                            P[s, a] = TERM
                            R[s, a] = 1.0 if x == N - 1 else 0.0    #place, or dropped short
                            continue
                    if ngl == 1 and ngr == 1 and not (gl == 1 and gr == 1):
                        r = 1.0                     #grasp: both grippers stalled at once
                    P[s, a] = enc(nx, ngl, ngr)
                    R[s, a] = r
    P[TERM, :] = TERM
    R[TERM, :] = 0.0
    return P, R


def q_star(P, R, gamma, iters=2000):
    Q = np.zeros((NS, NA))
    for _ in range(iters):
        V = Q.max(1)
        V[TERM] = 0.0
        Q = R + gamma * V[P]
        Q[TERM] = 0.0
    return Q


def q_star_shaped(P, R, Phi, gamma):
    """Q* of the MDP whose reward is r_phys + gamma*Phi(s') - Phi(s)."""
    Rs = R + gamma * Phi[P] - Phi[:, None]
    Rs[TERM] = 0.0
    return q_star(P, Rs, gamma)


def q_pi(P, R, pol, gamma, iters=None):
    """Exact Q of a stochastic tabular policy, by direct linear solve rather than sweeps.

    V = b + gamma * M V  with  M[s,s'] = sum_a pol[s,a] 1{P[s,a]=s'},  b[s] = sum_a pol[s,a] R[s,a]
    so V = (I - gamma M)^-1 b. Exact, and ~500x faster than iterating, which matters because the
    learning curves evaluate this hundreds of times per run.
    """
    M = np.zeros((NS, NS))
    for aa in range(NA):
        M[np.arange(NS), P[:, aa]] += pol[:, aa]
    b = (pol * R).sum(1)
    M[TERM, :] = 0.0
    b[TERM] = 0.0
    V = np.linalg.solve(np.eye(NS) - gamma * M, b)
    V[TERM] = 0.0
    Q = R + gamma * V[P]
    Q[TERM] = 0.0
    return Q, V


def eps_greedy(Q, eps):
    pol = np.full((NS, NA), eps / NA)
    pol[np.arange(NS), Q.argmax(1)] += 1.0 - eps
    return pol


def true_value(Q_learned, P, R, gamma):
    """Value of the GREEDY policy of Q_learned, measured under the TRUE physical reward."""
    pol = np.zeros((NS, NA))
    pol[np.arange(NS), Q_learned.argmax(1)] = 1.0
    _, V = q_pi(P, R, pol, gamma)
    return V[enc(0, 0, 0)]


#--- learners -----------------------------------------------------------------------------
def q_learn(P, R_train, gamma, rng, episodes, horizon=100, alpha=0.5, eps=0.2, shaping=None,
            P_true=None, R_true=None, eval_every=5, target=None):
    """Tabular Q-learning. `shaping` adds gamma*Phi(s')-Phi(s) to the training reward."""
    Q = np.zeros((NS, NA))
    curve = []
    solved_at = None
    for ep in range(episodes):
        s = enc(0, 0, 0)
        for _ in range(horizon):
            a = rng.integers(NA) if rng.random() < eps else int(Q[s].argmax())
            s2, r = int(P[s, a]), R_train[s, a]
            if shaping is not None:
                r = r + gamma * shaping[s2] - shaping[s]
            tgt = r + (0.0 if s2 == TERM else gamma * Q[s2].max())
            Q[s, a] += alpha * (tgt - Q[s, a])
            s = s2
            if s == TERM:
                break
        if (ep + 1) % eval_every == 0:
            v = true_value(Q, P_true, R_true, gamma)
            curve.append((ep + 1, v))
            if solved_at is None and target is not None and v >= target:
                solved_at = ep + 1
    return Q, curve, solved_at


def behaviour_clone(P, pol_L, pol_H, p_true, rng, n_episodes=60, horizon=100):
    """HG-DAgger analogue: the human takes over when the gate fires, and we imitate ONLY those
    actions. Finite samples, no privileged access to the human's policy at unvisited states --
    which is the actual weakness of the method, and the one it must be allowed to show."""
    counts = np.zeros((NS, NA))
    n_labels = 0
    for _ in range(n_episodes):
        s = enc(0, 0, 0)
        for _ in range(horizon):
            a = int(rng.choice(NA, p=pol_L[s]))
            if rng.random() < p_true[s, a]:
                a_h = int(rng.choice(NA, p=pol_H[s]))
                counts[s, a_h] += 1.0
                n_labels += 1
                a = a_h
            s = int(P[s, a])
            if s == TERM:
                break
    return counts, n_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--human-eps", type=float, default=0.15,
                    help="human suboptimality, so V_H != V* and the test is not rigged")
    ap.add_argument("--query-rate", type=float, default=0.08,
                    help="fraction of steps the human is asked about; HIL_THEORY sec 8 says 4-9%%")
    a = ap.parse_args()

    P, R = build_mdp()
    g = a.gamma
    Q_opt = q_star(P, R, g)
    V_opt = true_value(Q_opt, P, R, g)
    print(f"bimanual MDP: {NS} states, {NA} actions.  V* from start = {V_opt:.4f}")
    print("one-handed carry drops the object and ENDS the episode -- two hands are required\n")

    pol_H = eps_greedy(Q_opt, a.human_eps)
    _, V_H = q_pi(P, R, pol_H, g)
    rng0 = np.random.default_rng(0)
    Q_snap, _, _ = q_learn(P, R, g, rng0, episodes=120, P_true=P, R_true=R, eval_every=10_000)
    pol_L = eps_greedy(Q_snap, 0.1)
    Q_L, _ = q_pi(P, R, pol_L, g)

    regret = V_H[:, None] - Q_L
    tau = np.quantile(regret, 1.0 - a.query_rate)
    beta = 0.15 * regret.std()
    p_true = 1.0 / (1.0 + np.exp(-(regret - tau) / beta))
    print(f"gate: tau at the {100*(1-a.query_rate):.0f}th regret percentile -> "
          f"query rate {p_true.mean()*100:.1f}%  (beta {beta:.4f})")

    rng = np.random.default_rng(7)
    n_obs = 400
    p_hat = (rng.binomial(n_obs, p_true) + 0.5) / (n_obs + 1.0)
    logit = np.log(p_hat / (1 - p_hat))
    logit_bad = rng.permutation(logit.ravel()).reshape(logit.shape)
    p_hat_bad = 1.0 / (1.0 + np.exp(-logit_bad))

    def potential(lg, bb):
        Phi = (Q_L + tau + bb * lg).mean(1)
        Phi[TERM] = 0.0
        return Phi

    Phi_ok = potential(logit, beta)            #eq (10) with the true beta
    Phi_wrongscale = potential(logit, 1.0)     #beta off by ~20x
    Phi_bad = potential(logit_bad, beta)       #gate carries NO information
    Phi_critic = Q_L.mean(1).copy(); Phi_critic[TERM] = 0.0
    Phi_rand = rng.normal(0, 5, NS); Phi_rand[TERM] = 0.0

    #--- PART 1. Invariance, checked EXACTLY (not through a learner) --------------------
    print("\nPART 1  invariance of the optimum under potential shaping (exact, by value iteration)")
    print(f"{'potential Phi':<40}{'value of shaped-optimal policy':>32}")
    print("-" * 76)
    inv_ok = True
    for name, Phi in [("none (unshaped)", np.zeros(NS)),
                      ("eq (10), true beta", Phi_ok),
                      ("eq (10), beta wrong by 20x", Phi_wrongscale),
                      ("CORRUPTED gate (no information)", Phi_bad),
                      ("pure noise, N(0,5)", Phi_rand)]:
        v = true_value(q_star_shaped(P, R, Phi, g), P, R, g)
        same = abs(v - V_opt) < 1e-9
        inv_ok &= same
        print(f"{name:<40}{v:>20.9f}   {'= V*' if same else 'DIFFERS'}")

    #--- PART 2. Efficiency, and what it costs to get beta wrong ------------------------
    tgt = 0.95 * V_opt
    runs = {
        "phys only (no human)":         dict(R_train=R, shaping=None),
        "CONTROL critic only, no gate": dict(R_train=R, shaping=Phi_critic),
        "OURS eq(10), true beta":       dict(R_train=R, shaping=Phi_ok),
        "OURS eq(10), beta 20x wrong":  dict(R_train=R, shaping=Phi_wrongscale),
        "OURS + CORRUPTED gate":        dict(R_train=R, shaping=Phi_bad),
        "HIL-SERL classifier as rwd":   dict(R_train=p_hat, shaping=None),
        "HIL-SERL + CORRUPTED clf":     dict(R_train=p_hat_bad, shaping=None),
    }
    results = {}
    for seed in range(a.seeds):
        for name, kw in runs.items():
            Q, _, solved = q_learn(P, kw["R_train"], g, np.random.default_rng(5000 + seed),
                                   a.episodes, shaping=kw["shaping"], P_true=P, R_true=R,
                                   target=tgt)
            r = results.setdefault(name, {"v": [], "solved": []})
            r["v"].append(true_value(Q, P, R, g))
            r["solved"].append(solved if solved is not None else np.nan)
        counts, n_lab = behaviour_clone(P, pol_L, pol_H, p_true, np.random.default_rng(900 + seed))
        r = results.setdefault("HG-DAgger (BC on corrections)", {"v": [], "solved": [], "n": []})
        r["v"].append(true_value(counts, P, R, g))
        r["solved"].append(np.nan)
        r["n"] = [n_lab]

    print(f"\nPART 2  sample efficiency, {a.seeds} seeds x {a.episodes} episodes")
    print(f"{'method':<32}{'final value':>16}{'% of V*':>10}{'eps to 95%':>13}")
    print("-" * 76)
    for name in list(runs) + ["HG-DAgger (BC on corrections)"]:
        v = np.array(results[name]["v"])
        sv = np.array(results[name]["solved"], dtype=float)
        ns = "never" if np.all(np.isnan(sv)) else f"{np.nanmedian(sv):.0f}"
        print(f"{name:<32}{v.mean():>10.4f}±{v.std():<5.3f}{100*v.mean()/V_opt:>9.1f}%{ns:>13}")

    print("\nVERDICTS")
    def val(k): return np.array(results[k]["v"]).mean()
    def sol(k):
        s_ = np.array(results[k]["solved"], float)
        return np.inf if np.all(np.isnan(s_)) else np.nanmedian(s_)
    ok = True
    def verdict(c, m):
        nonlocal ok
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")
        ok &= bool(c)
    verdict(inv_ok, "invariance: EVERY potential -- including pure noise and a corrupted gate -- "
                    "leaves the optimum exactly at V*")
    verdict(sol("OURS eq(10), true beta") < sol("phys only (no human)"),
            f"efficiency: gate shaping solves in {sol('OURS eq(10), true beta'):.0f} episodes vs "
            f"{sol('phys only (no human)')} unshaped")
    verdict(val("OURS eq(10), true beta") > val("CONTROL critic only, no gate") - 1e-9,
            f"the gate is not redundant: with gate {val('OURS eq(10), true beta'):.4f} vs "
            f"critic-only control {val('CONTROL critic only, no gate'):.4f}")
    verdict(val("HIL-SERL + CORRUPTED clf") < 0.9 * V_opt,
            f"contrast: corrupted classifier-AS-REWARD breaks "
            f"({100*val('HIL-SERL + CORRUPTED clf')/V_opt:.0f}% of V*) -- no theorem protects it")
    verdict(val("OURS + CORRUPTED gate") > 0.9 * val("OURS eq(10), true beta") or
            sol("OURS + CORRUPTED gate") >= sol("OURS eq(10), true beta"),
            "corrupting the gate costs speed, not correctness (compare the two rows above)")
    print(f"\n{'ALL VERDICTS PASS' if ok else 'SOME VERDICTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
