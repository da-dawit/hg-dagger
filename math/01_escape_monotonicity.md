# The protected escape: what is proved, and what is not

Author: Dawit Chun · 2026-08-27

Concerns `_ESCAPE_*` in `igen_ppo/igen_agents/training/generational.py:165-173, 400-415`.
The code comment claims an escape *"can only strengthen the dynasty, never weaken it."*
That claim is **true for the estimated return and false for the true return**, and the gap is
quantifiable. This note proves the first, bounds the second, and gives the cheapest fix.

---

## Setup

Fix a generation. Let

- $\theta_0$ — the mature best child at the moment the escape triggers,
- $\theta_1$ — the child after `_ESCAPE_BURST = 15` high-entropy updates ($\varepsilon = 0.40$)
  followed by `_ESCAPE_SETTLE = 8` low-entropy updates,
- $J(\theta)$ — the true expected undiscounted return of the greedy policy $\pi_\theta$,
- $\widehat{J}_n(\theta)$ — its estimate from $n$ greedy evaluation episodes.

The escape rule carried forward is

$$
\theta_\star \;=\;
\begin{cases}
\theta_1, & \widehat{J}_n(\theta_1) > \widehat{J}_n(\theta_0)\\[2pt]
\theta_0, & \text{otherwise.}
\end{cases}
\tag{1}
$$

---

## Theorem 1 (Monotonicity in the estimate)

$\widehat{J}_n(\theta_\star) \ge \widehat{J}_n(\theta_0)$, deterministically.

**Proof.** Two cases, exhaustive by (1). If $\widehat{J}_n(\theta_1) > \widehat{J}_n(\theta_0)$ then
$\theta_\star=\theta_1$ and $\widehat{J}_n(\theta_\star) = \widehat{J}_n(\theta_1) > \widehat{J}_n(\theta_0)$.
Otherwise $\theta_\star=\theta_0$ and equality holds. $\blacksquare$

So the *recorded* dynasty curve is non-decreasing across escapes. This is what the code comment
asserts, and it is correct as stated.

---

## Theorem 2 (Budget neutrality)

An escape consumes $B+S = 23$ updates **from** the generation's allotment of
`updates_per_gen`, not in addition to it. Hence for a generation permitting at most
$K = $ `_ESCAPE_MAX_PER_GEN` $= 2$ escapes, the environment-step count is exactly

$$
N_{\text{steps}} \;=\; G \cdot U \cdot E \cdot T
$$

independent of how many escapes fire, where $G,U,E,T$ are generations, updates/gen, num_envs,
rollout_len. For `final_minatar_g6`: $6 \times 150 \times 512 \times 128 = 58{,}982{,}400$.

**Consequence.** The comparison against `vanilla`, `random_parent` etc. is budget-matched *by
construction*, not by calibration. A reviewer asking "does the escape buy extra samples?" is
answered by counting, not by trust. **This is the property that makes the escape admissible at
all**; an escape that added updates would invalidate every efficiency claim in the paper.

---

## Theorem 3 (The monotonicity does **not** transfer to $J$)

Let $\widehat{J}_n(\theta) = J(\theta) + \epsilon_\theta$ with $\mathbb{E}[\epsilon_\theta]=0$,
$\operatorname{Var}(\epsilon_\theta) = \sigma^2/n$, and $\epsilon_{\theta_0} \perp \epsilon_{\theta_1}$.
Suppose the escape is *useless*: $J(\theta_1) = J(\theta_0) = J$. Then

$$
\mathbb{E}\!\left[J(\theta_\star)\right] = J
\quad\text{but}\quad
\mathbb{E}\!\left[\widehat{J}_n(\theta_\star)\right]
= J + \frac{\sigma}{\sqrt{n}}\cdot\frac{1}{\sqrt{\pi}} \;>\; J .
\tag{2}
$$

**Proof.** The true return is unaffected since both candidates have return $J$, giving the first
equality. For the second, $\widehat{J}_n(\theta_\star) = J + \max(\epsilon_0,\epsilon_1)$. For
i.i.d. $\mathcal{N}(0,\sigma^2/n)$ variables,
$\mathbb{E}[\max(\epsilon_0,\epsilon_1)] = \frac{\sigma}{\sqrt{n}}\cdot\frac{1}{\sqrt{\pi}}$. $\blacksquare$

**This is a selection bias, and it is the honest caveat.** Accepting on
$\widehat{J}_n(\theta_1) > \widehat{J}_n(\theta_0)$ takes a max over two noisy estimates, so the
*reported* dynasty curve drifts upward by $\Theta(\sigma/\sqrt{n})$ per escape even when escapes
do nothing. With $K$ escapes per generation over $G$ generations the accumulated optimism is
$O(GK\sigma/\sqrt{n})$.

**Numerically.** Breakout inter-seed SD is $\approx 4$ (`results_g6_final`). Taking $\sigma \approx 4$
and $n=30$: bias per escape $\approx 4/\sqrt{30}/\sqrt{\pi} \approx 0.41$. Over $G=6$, $K=2$:
**up to $\approx 4.9$ points of optimism** — comparable to the entire 0–4 point margin the
method is trying to demonstrate. A reviewer who spots this sinks the result.

---

## Corollary 4 (The fix, and its cost)

Replace the accept rule (1) with a **margin rule**

$$
\theta_\star = \theta_1 \iff \widehat{J}_n(\theta_1) > \widehat{J}_n(\theta_0) + \delta,
\qquad \delta = z\,\sigma\sqrt{2/n}.
\tag{3}
$$

Under $J(\theta_1)=J(\theta_0)$ the acceptance probability falls from $1/2$ to $1-\Phi(z)$, so the
optimism in (2) is suppressed to $O((1-\Phi(z))\,\sigma/\sqrt n)$. Choosing $z=1$ cuts false
acceptances from 50% to 16% at the cost of rejecting genuine improvements smaller than $\delta$.

**This costs nothing at runtime** — it is a comparison, not a computation — and it converts the
claim of Theorem 1 from "monotone in the estimate" to "monotone in the estimate, with controlled
false-acceptance rate." Recommended before any submission.

---

## Proposition 5 (The escape cost induces the intended incentive)

The parent receives reward $r - c\,\mathbb{1}[\text{escape failed}]$ with
$c = $ `_ESCAPE_COST` $=0.5$. Let $p = \Pr[\text{escape succeeds}]$ and $g = \mathbb{E}[\text{gain}\mid\text{success}]$.
The REINFORCE gradient favours raising the escape probability iff the expected reward differential
is positive:

$$
p\,g - (1-p)\,c \;>\; 0
\quad\Longleftrightarrow\quad
p \;>\; \frac{c}{c+g}.
\tag{4}
$$

**Interpretation.** With $c=0.5$, the parent is pushed to escape only where its success rate
exceeds $\frac{0.5}{0.5+g}$ — e.g. $g=2$ requires $p>20\%$; $g=0.5$ requires $p>50\%$. The cost is
therefore not a penalty on exploration per se, but a **threshold on expected value**, which is the
stated design intent ("the parent must LEARN to be wise about it"). Note (4) also shows $c$ is not
free: setting $c$ too high forbids escapes whose payoff is real but modest.

---

## What this does and does not establish

**Established.** Escapes cannot reduce the recorded dynasty return (Thm 1); they consume no extra
environment steps, so budget-matching is structural (Thm 2); the cost term implements a
value threshold with an explicit break-even (Prop 5).

**Not established.** That escapes *help* — Theorems 1–2 are safety properties, not performance
claims. Whether $p\,g > 0$ in practice is precisely what the `results_rerun_contested` run
measures.

**Actively flagged.** The accept rule is optimistically biased by $\Theta(\sigma/\sqrt n)$ per
escape (Thm 3), plausibly ~5 points on Breakout — the same order as the effect being claimed.
Corollary 4 fixes it for free. **Until that is applied, any IGEN-vs-random margin on the order of
a few points should be treated as unproven.**
