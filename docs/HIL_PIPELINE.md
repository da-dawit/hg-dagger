# Pipeline: learning from the operator's attention instead of their hands

Implements `HIL_THEORY.md`. Every stage below states a **claim**, the **evidence** for it, and the
**kill criterion** that stops the pipeline if the claim is false. Nothing proceeds on assertion.

Verification suite: `scripts/verify_theory.py` — **10/10 checks pass**, each with a negative
control. Run it before trusting anything downstream.

---

## The one-line change

HIL-SERL learns from the operator's **hands** (the ~5% of timesteps they intervene) and needs a
separate success classifier for reward. This learns from their **attention** (the intervene /
don't-intervene decision at 100% of timesteps) and proves that replaces the reward model.

| | HIL-SERL | here |
|---|---|---|
| reward | learned binary success classifier | **deleted** |
| new component | — | gate classifier `p̂(s,a)` |
| signal into RLPD | `r̂(s)`, sparse, hackable | `−logit p̂`, dense, every timestep |
| labels used per session | ~5% | **100%** |
| learner | RLPD | RLPD, unchanged |
| operator workload during collection | as before | **identical** |

---

## Stage 0 — hardware I/O  *(blocking, not yet written)*

`scripts/dagger_collect.py` is complete and human-gated on
`/leader/joystick_controller/tact_trigger`. It needs one ROS wrapper, `robot_io.py`, exposing
`observation()`, `send()`, `trigger_held()`, `leader_command()`, `home()`, `wait_next_tick()`.

**Schema change already made (2026-08-13):** the collector now records `action_policy` — the action
the policy proposed — alongside the executed action. Without it the rejected proposal is lost at
every intervened step, and an intervention stops being a *comparison*. That pair is what the regret
label is about; see `HIL_THEORY.md` §2.

---

## Stage 1 — collect

~30 episodes, operator intervening as they naturally would. Per timestep:
`state`, `action` (executed), `action_policy` (proposed), `intervention`, `pre_intervention`,
three camera frames at native resolution, plus an episode-level `success` flag.

**Cost:** one afternoon. **Everything below is offline**, so a wrong idea dies here, not after a
paper.

---

## Stage 2 — GATE A1: do operators threshold on regret?

**Claim.** `g(s) = 1{ R(s, a_π) > τ }`, with `R = V^H − Q^π`.

**Status: an assumption about people. Not proved, not provable from the armchair.** It is the single
load-bearing premise of the whole reduction.

**Test.** `scripts/gate_model.py` regresses realised return difference on `logit p̂` over
held-out episodes and reports Spearman ρ with a p-value.

**Kill criterion.** ρ ≤ 0.1 or p ≥ 0.05 → operators are intervening on habit, impatience or safety
reflex rather than on regret. `HIL_THEORY.md` is then void and **the pipeline stops here.** Do not
proceed to stage 3 "to see what happens".

---

## Stage 3 — fit the gate, derive the reward

**Claim (Theorem 2).** A classifier fitted to the gate labels recovers regret up to a positive
affine map: `R ≈ τ + β·logit p̂`.

**Verified,** not assumed. `verify_theory.py::t3` constructs data with a *known* `R`, shows the
fitter only `g = 1{R > τ}` with logistic noise, and recovers `pearson(logit p̂, R) = 0.9999` with
slope `0.502` against a true `β = 0.5`.

> The first negative control I wrote for this **failed at |r| = 0.371 against a 0.1 threshold — and
> the defect was in my test.** A classifier fitted to random labels is a random direction in `R^d`,
> and two random directions in `d` dimensions correlate at ≈ `1/√d` = 0.354 for d=8; a 0.1 cutoff was
> unreachable by construction. Replaced with a permutation test, which generates the null rather than
> assuming it: observed 0.9999 against a null mean of 0.304 and 99th percentile 0.803.

**Reward.** `r' := −logit p̂`, written to `hil_gate.npz` as a drop-in for the success classifier.

---

## Stage 4 — train with RLPD, unchanged

**Claim (the one that makes this usable off-policy).** `−R` is not merely an advantage — it is a
*reward*:

    −R(s,a) = r(s,a) + γ·E[V^H(s')] − V^H(s)

whose last two terms are exactly `γΦ(s') − Φ(s)` with `Φ = V^H`. By Ng, Harada & Russell (1999),
potential-based shaping preserves the optimal policy.

**Verified.** `verify_theory.py::t4` — 200 random MDPs, exact value iteration: **0/200** optimum
changes. Negative control with action-dependent (non-potential) shaping: **200/200** change, so the
test can detect a broken shaping.

**Why this matters.** It removes the objection that the reduction is on-policy while HIL-SERL's
efficiency is off-policy. `r'` goes straight into RLPD as the reward. No advantage weighting
anywhere — which keeps it clear of `RL_PREDICAMENT.md` §3.1, where advantage weighting lost to
uniform weighting six times out of six.

**Three consequences, each traceable to the equation:**
1. the task reward `r` is still present, so deleting the success classifier is not training on noise;
2. `r'` is dense where `r` fires once per episode — the efficiency claim, with an optimality
   guarantee attached rather than the usual shaping caveat;
3. both ambiguities are harmless: `β > 0` scales all values, a constant adds `c/(1−γ)` to all values,
   and neither moves an argmax.

---

## Stage 5 — hand the gate the trigger

Once `p̂` is accurate, it fires the query instead of the operator watching for it: the robot halts,
holds position, and **queues** a request. The operator answers a batch later. Continuous attention
becomes asynchronous attention — sound only because tabletop manipulation is quasi-static, so scope
the claim there and say so.

**Where the threshold has to sit, from data we already have.** Per-state error of the base policy on
held-out episodes (n=160): median 0.0352 rad, p90 0.0566, p99 0.0790. Gating at `k` demonstration
steps (1 step = 0.0071 rad) queries:

    k = 4  ->  68.8%    k = 6  ->  25.0%    k = 8  ->  8.8%    k = 10 ->  3.8%

So the operating point is **near the 90th percentile**, not the median — at the median it fires on
half the timesteps and saves nobody anything. `|Δ|` is a proxy for regret, so this is an
order-of-magnitude check, not a substitute for stage 2.

**Kill criterion.** At an 8% query budget, recall of the operator's real interventions ≤ 50% → the
gate cannot carry the operator's job yet; keep them in the loop and collect another round.

---

## Stage 6 — evaluate on the axis being optimised

Report **success rate against operator-seconds**, not environment steps.

This is not presentational. HIL-SERL's baselines are compared unfavourably to it precisely because
it receives extra human corrective data that they do not; measuring the resource actually being spent
makes the comparison fair and makes the claim defensible. Against a method already reporting 100%
success, "higher success" is a hard claim; "same success for a third of the attention" is a
measurable one.

---

## What this does **not** fix

The left/right generalisation failure. Held-out error is 3.9× the training-episode error
(0.0362 vs 0.0094 rad) and that is a data-coverage problem. It is orthogonal to everything above and
must not be claimed. The residual on Δ and broader placement coverage are the tools for that.

---

## Robustness result worth stating separately

`verify_theory.py::t5` sweeps operator quality — near-optimal, mediocre, and **fully random** — and
the optimum is preserved in **0/120** cases each time. Ng's theorem asks nothing of `Φ` except that
it be a function of state, so a bad operator degrades the *shaping quality* (how fast learning goes)
and never the *ceiling* (where it ends up).

That is the sharpest contrast with imitation. GAIL matches `ρ_E` and is capped at the demonstrator by
construction; HIL-SERL is capped by whatever satisfies its success classifier. Here the ceiling is
the task optimum, and the operator only accelerates the route to it. **The learner is free to
overtake the person teaching it.**
