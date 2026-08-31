# What HIL-SERL is actually computing, and what that implies

Written 2026-08-13. The brief was: do for human-in-the-loop RL what Ho & Ermon did for IRL — find the
mathematical object the method is really manipulating, prove the reduction, and let the algorithm
fall out of it rather than bolting heuristics onto the existing one.

---

## 0. The analogy being aimed at

IRL was posed as: recover a reward `r` from expert data, then run RL on `r`. Two nested loops, an
ill-posed inner problem, and enormous cost. GAIL's contribution was not a better IRL solver — it was
the observation that **RL(IRL(π_E)) is equivalent to matching occupancy measures**, so the whole
two-stage program collapses to a single divergence minimisation `min_π D(ρ_π ‖ ρ_E)` with a direct
algorithm.

The question here is whether human-in-the-loop RL has an equivalent collapse. I think it does, and
the object it collapses onto is the **advantage function**, not the reward.

---

## 1. Setup and notation

MDP `M = (S, A, P, r, γ)`. The true reward `r` is unavailable; HIL-SERL substitutes a learned binary
success classifier `r̂`.

At each timestep the human observes the state and the action the learner is about to take, and
chooses a gate `g ∈ {0,1}`: `g=0` lets the learner act, `g=1` takes over. The behaviour policy is
therefore a switching mixture

    β(a|s) = (1 − g(s)) · π(a|s) + g(s) · π_H(a|s).                                  (1)

HIL-SERL stores `(s, a, r̂, s')` from `β`, routes intervention transitions to a separate buffer, and
runs RLPD (off-policy SAC with symmetric sampling) on the result.

**What the algorithm uses from an intervention:** the corrective action `a_H`, as expert data.
**What it discards:** the fact that the human *rejected* `a_π`, and — much larger — the fact that at
every timestep where the human did **not** intervene, they judged `a_π` acceptable.

That discarded quantity is the whole of this document.

---

## 2. The human's gate is a thresholded regret readout

Model the human as intervening when the learner's action is enough worse than what the human would
do. Writing `V^H` for the value of the human's policy and `Q^π` for the learner's action-value,
define the **human-relative regret**

    R(s, a) := V^H(s) − Q^π(s, a) ≥ 0.                                               (2)

**Assumption A1 (thresholded human).** The human intervenes when regret exceeds a threshold:

    g(s) = 1{ R(s, a_π(s)) > τ }.                                                    (3)

This is an assumption about people, not a theorem, and §6 says how to test it. But note what it
implies: **every timestep of a HIL-SERL run is a labelled example of a thresholded regret**, whether
or not the human moved. The non-interventions are labels too, and they are ~95% of the data.

So the human is not primarily a data-collection valve. The human is a **dense binary oracle on the
advantage function**, queried at 30 Hz, for free, and HIL-SERL reads roughly 5% of its output.

---

## 3. Theorem 1 — human-relative regret is a valid advantage for policy gradient

The policy gradient theorem gives

    ∇_θ J(π_θ) = E_{s∼ρ_π, a∼π_θ} [ ∇_θ log π_θ(a|s) · A^π(s,a) ],   A^π = Q^π − V^π.

We do not have `A^π`. We have (an estimate of) `−R`. They differ by a state-dependent term:

    −R(s,a) = Q^π(s,a) − V^H(s)
            = [Q^π(s,a) − V^π(s)] − [V^H(s) − V^π(s)]
            = A^π(s,a) − b(s),        with  b(s) := V^H(s) − V^π(s).                 (4)

Substituting into the gradient:

    E_{a∼π}[ ∇ log π · (−R) ] = E[ ∇ log π · A^π ] − E_{a∼π}[ ∇ log π · b(s) ]
                              = E[ ∇ log π · A^π ] − b(s) · E_{a∼π}[ ∇ log π ].

The last expectation vanishes identically:

    E_{a∼π}[ ∇_θ log π_θ(a|s) ] = ∫ π_θ(a|s) ∇_θ log π_θ(a|s) da = ∇_θ ∫ π_θ(a|s) da = ∇_θ 1 = 0.

Therefore

    **E_{a∼π}[ ∇ log π · (−R) ] = E_{a∼π}[ ∇ log π · A^π ].**                        (5)

`−R` is an unbiased advantage surrogate. The gap between the human's value and the learner's — the
part we cannot measure — is a state-dependent baseline, and baselines provably contribute nothing to
the policy gradient. □

This is the crux. It says the *unmeasurable* part of the human's judgement is exactly the part the
policy gradient is blind to.

---

## 4. Theorem 2 — a classifier on human attention recovers the advantage up to an affine map

We do not observe `R`; we observe `g = 1{R > τ}`. Assume the threshold is stochastic (people are not
step functions), with a logistic link — **Assumption A2**:

    P(g = 1 | s, a) = σ( (R(s,a) − τ) / β ).                                         (6)

Fit a classifier `p̂(s,a)` to the gate labels by log-loss. Log-loss is a strictly proper scoring
rule, so its population minimiser is `p̂ = P(g=1|s,a)`. Inverting (6):

    R̂(s,a) = τ + β · σ^{−1}( p̂(s,a) ) = τ + β · logit p̂(s,a).                      (7)

So the classifier's **logit** recovers `R` up to the affine map `x ↦ τ + βx`, with `τ, β` unknown.

Now use (5). Write the surrogate advantage as `Â := −logit p̂`. Then `−R = −(τ + βÂ')` for the true
`Â'`, i.e. `Â` and the true `−R` are related by `−R = βÂ + const`. In the gradient:

* the **constant** is a state-independent baseline — it vanishes by the same argument as (5);
* the **positive scale** `β` multiplies the gradient uniformly, which is a learning-rate
  reparameterisation and does not change the ascent direction.

Therefore

    **∇_θ J ∝ E_{s∼ρ_π, a∼π}[ ∇_θ log π_θ(a|s) · ( −logit p̂(s,a) ) ],**             (8)

with the proportionality constant `β > 0` absorbed into the step size. □

**Corollary (the collapse).** Under A1–A2, a binary classifier trained to predict *whether a human
would intervene* is sufficient to compute a policy-gradient direction. **No reward function is
required — learned, hand-written, or otherwise.**

This is the analogue of GAIL's move. HIL-SERL is posed as "learn a reward classifier, then run RL on
it, with a human supplying corrective data." The reduction says the reward model is redundant: the
human's attention already encodes the advantage, densely, and the corrective actions are a *second*,
smaller channel on top.

---

## 5. What the reduction buys, against each known weakness

| HIL-SERL weakness | what (8) does about it |
|---|---|
| reward is binary; cannot express "succeeded but wastefully" | `logit p̂` is continuous. "Wasteful" is a region of elevated intervention probability that never crossed `τ`. Representable by construction. |
| reward classifier is hackable — policy finds states it mislabels as success | there is no reward classifier in (8). The failure mode is removed rather than mitigated. |
| credit misassignment inside interventions | `p̂` is a function of state, so credit lands where predicted regret *rises* (t≈95), not where the human's hand moved (t≈140). No heuristic lead-in window needed. |
| human time doesn't amortise | non-interventions are ~95% of labels and cost nothing. Once `p̂` is accurate it can gate itself, and the human is queried only where `p̂` is uncertain. |
| ~~generalisation~~ | **not addressed.** This is orthogonal and stays orthogonal. Don't claim it. |

---

## 6. What is proved, what is assumed, and how to test the assumptions

**Proved outright** (no assumptions beyond the policy gradient theorem):
* (5) — a state-dependent baseline contributes zero to the policy gradient, so the unmeasurable
  `V^H − V^π` is harmless;
* (8) — a positive affine transform of the advantage preserves the ascent direction.

**Assumed, and falsifiable:**

* **A1 — the human thresholds on regret.** If people intervene out of impatience, safety habit, or
  boredom, the labels are not a regret readout and everything above is void. *Test:* on collected
  data, regress realised return difference (human-continuation vs policy-continuation, where both
  are observed) on `logit p̂`. A1 predicts a positive, monotone relationship. If the slope is
  indistinguishable from zero, stop — this is the load-bearing assumption.
* **A3 — the threshold does not drift.** THE LARGEST PRACTICAL RISK, and it was not on this list
  until the self-test found it. `gate_model.py --self-test` generates an operator who obeys A1
  exactly and measures what happens when `τ` varies per episode:

        fixed threshold      held-out AUC 0.983   recall 8.1% at an 8% query budget
        drifting threshold   held-out AUC 0.588   recall 16.6% -- cannot hold the budget
        coin-flip control    held-out AUC 0.472

  The regret is *perfectly* linear in the state in all three; only the operator's standard moves.
  AUC 0.983 → 0.588 is the entire method collapsing, and operators certainly do drift — they get
  pickier as the policy improves.

  **The fix is implied by Theorem 1.** A drifting threshold gives `g = 1{R > τ_e}`: the offset varies
  by episode but never by action, so it is a *state-independent offset* — precisely the object (5)
  proves contributes nothing to the policy gradient. Absorb it with a per-episode intercept in the
  classifier and the guarantee is untouched; it costs one parameter per episode.

* **A2 — logistic link.** Only affects the shape of the recovered `R`, not its ordering, and (8)
  needs only a positive affine relation. Mis-specifying the link costs calibration, not direction.
  Weakest of the assumptions.
* **Coverage.** (5) requires `a ∼ π` at the states where labels are taken. This holds *by
  construction*: the human judges the action the learner was about to take. Better than most
  off-policy settings, and worth stating — it is why the estimator is on-policy where it matters.

---

## 7. The honest tension with our own results

`RL_PREDICAMENT.md` §3.1 records **six experiments where advantage weighting lost to uniform
weighting**, and §4's diagnosis is that the advantage estimate was *biased* — resolvable is not the
same as accurate, and a biased advantage concentrates every gradient step on whatever the bias
favours.

That result is a direct threat to anything advantage-shaped, including this. Three things separate
the two cases, and none of them is decisive on its own:

1. **Source of the estimate.** Those advantages came from a learned critic trained on the same data
   it was scoring — a self-referential loop with a known bias mode. Here the signal originates in a
   human decision, which does not share that failure mode (it has its own, named in A1).
2. **Where it is applied.** Those experiments used the advantage to *reweight a BC loss*. (8) uses it
   as the coefficient in a policy gradient, which is the estimator's native use and the one the
   unbiasedness argument covers. Reweighting a regression is not the same object.
3. **What (5) actually protects.** The dominant unknown here — how much better the human is — is
   provably a baseline. That is a structural guarantee the critic-based advantage never had.

None of that makes it safe. It means the §3.1 result is the right prior, and the burden of proof is
on this method to clear it. The test in §6 is what clears or kills it.

### 7.1 The off-policy tension, and why it dissolves

The practical objection was: HIL-SERL's sample efficiency comes from **off-policy** RLPD, while (8)
is an **on-policy** gradient, and putting `logit p̂` into an off-policy actor update reintroduces the
advantage-weighted regression that lost 6/6.

It dissolves, because `−R` is not only an advantage — it is a **reward**, and a well-behaved one.

Expand the regret against the human's continuation:

    −R(s,a) = Q^{one step, then human}(s,a) − V^H(s)
            = r(s,a) + γ·E_{s'∼P}[V^H(s')] − V^H(s).                                 (9)

The last two terms are exactly `γΦ(s') − Φ(s)` with potential `Φ = V^H`. So **`−R` is the true task
reward plus potential-based shaping**, and Ng, Harada & Russell (1999) prove potential-based shaping
leaves the optimal policy unchanged.

Three consequences:

1. `−R` can be used as the **reward** in any RL algorithm, including off-policy RLPD. No on-policy
   restriction, no advantage-weighted regression, no contact with the §3.1 failure mode.
2. It does **not** discard the task reward. `r` is still in there, at the front of (9) — the human's
   judgement of regret necessarily accounts for whether the task actually gets done. This is why
   dropping the success classifier is safe and is not the same as training on zero reward.
3. It is **dense where `r` is sparse**. `r` fires once per episode; `−R` is defined at every state.
   That is the whole efficiency argument, and it comes with an optimality guarantee rather than the
   usual shaping caveat.

The positive-affine ambiguity from Theorem 2 is harmless here too: scaling a reward by `β > 0` scales
all values by `β`, and adding a constant `c` adds `c/(1−γ)` to every value. Neither changes an argmax.

**Verified numerically.** 200 random tabular MDPs (12 states, 4 actions, γ=0.95, sparse reward), exact
value iteration, human policy = optimal with 25% of states randomised:

    optimal policy changed by Phi = V_H shaping : 0 / 200
    negative control (action-dependent shaping) : 200 / 200 changed

The negative control matters — it shows the test can detect a broken shaping, so the 0/200 is
evidence rather than a tautology.

**What this does not fix.** (9) assumes the human's regret reflects `r` plus their own continuation
value — a strengthening of A1. If a human intervenes for reasons unrelated to task outcome (habit,
safety reflex, impatience), `−R` is no longer (9) and the guarantee is void. Same test as §6.

---

## 8. Predicted query rate, from data we already have

The gate's usefulness depends on interventions being *rare*, or the human is back to full-time
attention. From the measured per-state error of the base policy on held-out episodes (n=160):

    percentile   10%     25%     50%     75%     90%     95%     99%
    |Δ| (rad)    0.0186  0.0266  0.0352  0.0424  0.0566  0.0600  0.0790
                 2.6     3.7     5.0     6.0     8.0     8.4     11.1   demonstration steps

If a gate fired whenever the error exceeds `k` demonstration steps:

    k = 4  →  68.8% of states     (unusable — that is continuous attention)
    k = 6  →  25.0% of states     (every ~4 steps)
    k = 8  →   8.8% of states     (every ~11 steps)
    k = 10 →   3.8% of states     (every ~27 steps)

So the operating point has to sit around `k ≈ 8–10`, i.e. the human is asked about 4–9% of the time.
That is the efficiency claim made concrete, and it is measurable before any of the theory is used:
it says the threshold must be set near the 90th percentile of the error distribution, not at the
median. Setting it at the median would query half the timesteps and save nobody any labour.

Caveat: `|Δ|` here is distance from the demonstrated action, which is a *proxy* for regret, not
regret itself. It is the right order-of-magnitude check, not a substitute for §6's test.

---

## 9. The one experiment that decides this

Collect one DAgger round (~30 episodes, `dagger_collect.py`, already written). Then, offline:

1. Fit `p̂(s,a)` on the gate labels — every timestep, intervened or not.
2. **A1 test:** regress realised return difference on `logit p̂`. Positive monotone slope, or stop.
3. **Sufficiency test:** does `−logit p̂` rank actions the same way a critic trained on the true
   outcome does? Spearman correlation on held-out episodes.
4. **Efficiency test:** at the threshold giving 8% query rate, what fraction of actual human
   interventions does `p̂` catch? That is recall at fixed labour, and it is the number that decides
   whether the human's attention can be automated at all.

Each is computable from one afternoon of robot time, before committing to an algorithm.

---

## 10. Theorem 3 — the gate is safe to use because the physical reward carries the task

Let `r_phys` be the sparse reward read from the servos. Since `R = V^H − Q^π` and, by (7),
`R̂ = τ + β·logit p̂`, rearranging gives, for ANY action `a`,

    V̂^H(s) = Q^π(s,a) + τ + β·logit p̂(s,a)                                       (10)

The learner's own critic plus the gate classifier recovers the human's value function up to two
scalars. Use it as a POTENTIAL rather than as a reward:

    r_total(s,a,s') = r_phys(s,a) + γ·Φ(s') − Φ(s),      Φ := V̂^H,  Φ(terminal) = 0   (11)

By Ng–Harada–Russell (1999) this leaves the optimal policy **exactly** unchanged for any Φ.
Therefore gate error costs sample efficiency and never correctness. Ng et al. also show the ideal
potential is `Φ = V*`; a near-optimal human gives `V^H ≈ V*`, which is why it accelerates.

The contrast with HIL-SERL is now precise. Using an intervention classifier AS the reward has no
invariance theorem behind it, and a wrong classifier moves the optimum — that is what reward
hacking is. Using it as a POTENTIAL on top of a physical reward cannot.

**Verified** (`scripts/verify_theorem3.py`), exactly, by value iteration rather than through a
learner: unshaped, eq (10) with true β, β wrong by 20×, a corrupted gate carrying zero information,
and pure N(0,5) noise all give a shaped-optimal value of **1.565826044 = V\***, identical to nine
decimals. Sample efficiency, 10 seeds: unshaped never solves (61.9% — it learns to grasp and never
to transport); gate-shaped solves in 25 episodes.

Long-horizon sweep, the claim that matters for efficiency: the advantage **grows** with task length.
The unshaped baseline solves a 4-stage task in 1915 episodes and **never** solves 8 stages or more;
gate shaping goes 15 → 40 episodes across a 4× horizon increase.

## 11. Theorem 4 — β, and the honest version of it

`τ` never needs estimating: it shifts Φ by a constant, and a constant potential contributes a
state-independent term the policy gradient discards. Only `β` matters, and (11) is sensitive to it —
β wrong by 20× loses every bit of the efficiency gain while keeping correctness.

**The identification argument.** `V^H(s)` does not depend on the action, so from (10), within a
fixed state,

    Q^π(s,a) + β·logit p̂(s,a) = V^H(s) − τ = const(s)                             (12)

so `Q^π` and `logit p̂` lie on a line of slope −β. Fit it as a binomial GLM with a free intercept
per state and one shared coefficient on `Q^π`; the coefficient is −1/β. Do NOT do this in two
stages — forming `p̂`, taking logits, and regressing is biased twice (errors-in-variables attenuates
the slope; the smoothing needed to keep logits finite inflates β). Measured: two-stage +182% → +34%
across 20 → 2000 labels/cell, versus ±20% for the joint GLM.

**Where it fails, and this is the part that matters for hardware.** (12) needs several DISTINCT
ACTIONS AT THE SAME STATE. Real HIL data gives one action per visit, and the behaviour policy is
nearly deterministic, so the within-state contrast rests on one or two observations of a rare
action however many labels are collected in total. Measured on trajectory-sampled data: −63%, −87%,
−100%, −7%, −45% across 321 → 32,542 labels. **No convergence.** Deliberately randomising the
collection policy helps but does not fix it (best −21%, erratic).

**The claim that survives is weaker and sufficient.** With randomised collection the estimate
becomes *stable* at a consistent ~43% UNDERESTIMATE. That is the safe direction, because the
failure is one-sided:

    beta 0.5x true  ->  30 episodes        beta 2x true  ->  48 episodes
    beta 1.0x true  ->  25 episodes        beta 5x true  ->  never

The working window is roughly **0.5× to 2× of true β**, and the estimator lands inside it on the
safe side. So the operating rule is not "estimate β accurately" — it is **estimate β and shrink it**.
Underestimating costs speed; overestimating costs everything.

This also predicts the one thing that would break it in practice: critic error biases β̂ UPWARD
(+60% at 5% critic noise, +96% at 10%), which is the fatal direction. Shrinkage protects against
that too.

**Verified** in `scripts/verify_beta.py`.
