# Context for framing discussion — DFL battery dispatch dissertation

This file summarizes an extended technical investigation (one long working session) into a
decision-focused-learning (DFL) battery dispatch project, for the purpose of a fresh
conversation focused on **research framing**: what the dissertation's actual contribution is,
and whether the currently-planned write-up structure demonstrates it. The author's own words
going into this: *"I need to consider what it actually is I am trying to show here — my
current plan doesn't do that."* That's the question this conversation should help answer.
Don't re-derive the technical findings below — take them as given and focus on framing.

## What the project is

A DFL pipeline for battery dispatch / arbitrage in the UK electricity market:
- A quantile-forecasting GRU predicts prosumption (net load).
- A frozen Gaussian-copula sampler turns the quantile forecast into correlated scenarios.
- A differentiable convex dispatch layer (via `cvxpylayers`) solves for battery charge/discharge
  given the forecast, and gradients flow *back through the optimization* to train the
  forecaster end-to-end against realised decision cost — not just forecast accuracy.
- Compared against a baseline forecaster trained conventionally (pinball/CRPS loss only).

**"Corners"** tested: price settlement model (single-price vs dual-price/asymmetric) × `k`
(0 = pure economic/profit-maximizing dispatch, 1 = pure "dispatchability"/tracking-focused
dispatch, no price awareness at all). Plus multiple dispatch *formulations*: a 1-stage
no-recourse baseline, a "point-robust" 3-point box approximation (used for training, cheap
to differentiate), and a full LP-dual robust counterpart (used for final testing, exact
worst-case feasibility guarantee).

## The central finding from this session

**Headline**: DFL appeared to improve regret (economic cost vs. a perfect-foresight oracle)
across every corner tested. Investigation showed this "improvement" is substantially — in some
corners almost entirely — an artifact of the objective/settlement structure, not genuine
decision-quality improvement. This was diagnosed rigorously, not assumed:

1. **First bug found and fixed**: `evaluate.py` was testing against a dispatch formulation
   (`dispatch_layer.py`) with robustness deliberately stripped out for training-speed reasons —
   meaning the recourse gain matrices (`D_ch`/`D_dis`) had *no feasibility constraint at all* at
   test time, only a weak Tikhonov penalty. This caused pathological saturation (raw decisions
   swinging far outside physical bounds, clipped back). Fixed by switching to the proper full
   robust formulation (`dispatch_layer_robust.py`). This changed the reported numbers
   materially but didn't change the core story below.

2. **The real finding**: even after that fix, DFL-trained forecasters show a systematic,
   measurable bias — they predict *more solar generation* (lower net prosumption) than
   genuinely justified, concentrated specifically in real solar-generation hours (not
   uniformly). Measured directly: 0.3–1.05 MW mean shift vs. baseline, strongest in the dual-price
   k=1 corner, and this bias magnitude *correlates with lower reported regret*
   (correlation up to +0.58) — i.e. the model is rewarded for the bias.

3. **Mechanism, derived precisely, not guessed**: the bid is pinned as
   `bid = pl_hat + p_ch_hat − p_dis_hat`. Since `C_da = pi_da · bid · dt` and `pi_da` is almost
   always positive, lowering `pl_hat` mechanically cheapens the *reported* day-ahead cost,
   independent of whether the lower forecast is true. Splitting total cost into `C_da`/`C_imb`
   confirmed this: in 5 of 6 corners, `C_imb` (the real, physically-settled cost) got **worse**
   under DFL, while `C_da` improved by more than enough to mask it. The "decision quality"
   improvement is largely a bid-accounting artifact.

4. **Deeper root cause (k=1 specifically)**: the k=1 objective (`sum_trace`) is supposed to
   represent `Var(ξ) + bias²` (derived from the user's own LaTeX formulation of the true
   objective), but only implements `Var(ξ)`. This is structural: `xi_samples` is defined as
   *deviation from the model's own mean*, which is exactly zero-mean by construction regardless
   of whether that mean is accurate — the term is mathematically blind to bias, not just
   empirically insensitive to it. Compounding this: at k=1 exactly, the dispatch layer has zero
   price-awareness (price parameters aren't even instantiated), so it has no competing economic
   lever — meaning *all* cost-reduction pressure from training funnels through the one channel
   left open, `pl_hat`. This is why k=1 showed the largest bias and the largest `C_imb`
   degradation of all corners.

5. **Ruled out as numerical artifacts, not just asserted**: Sobol-vs-plain-Monte-Carlo redraw
   noise (~0.008 MW), analytical-quadrature-vs-scenario-average mean (~0.006 MW), and
   autocorrelation-driven per-hour bias in the scenario generator — all measured directly and
   found negligible (40-100x smaller than the actual effect). The bias is a structural
   objective-misspecification problem, not a sampling or implementation quirk.

## Fixes discussed (not yet implemented)

Two complementary fixes, not mutually exclusive:
- **Anchor `pl_hat` to the frozen baseline forecaster's mean** instead of the currently-training
  model's own mean — severs the gradient path enabling the exploit. Same principle already used
  elsewhere in the codebase (the robust box is computed from the frozen baseline, for exactly
  this "shouldn't get to game its own benchmark" reason).
- **Make the k=1 training loss itself cost-agnostic** (e.g. train on `abs_dev_MWh` instead of
  economic regret for k=1) — removes the asymmetric price signal at its source rather than
  closing the channel it escapes through. More principled but a bigger implementation change.

Both require full retraining of affected corners and re-evaluation; not a quick patch.

## Other design questions worked through this session (mostly resolved)

- **Oracle design**: a perfect-foresight k=1 "tracking" oracle is provably degenerate — under a
  pinned bid, `p_imb ≡ 0` for *any* dispatch policy once the bid matches the known true value,
  regardless of what function of imbalance you try to minimize (squared cost, volume, anything).
  Conclusion: don't build one; the value is analytically zero, can be hardcoded rather than
  solved. The existing day-ahead-only economic oracle remains the correct, non-degenerate
  benchmark (meaningful because of physical battery constraints, not because of anything about
  imbalance).
- **Pinning vs. unpinning the bid**: pinning prevents pure day-ahead/imbalance price-spread
  arbitrage (empirically confirmed problematic when tried unpinned historically in this
  project). Unpinning was also argued to likely perform *worse* on average, not just
  differently — the price-uncertainty proxy fed into training is a single noisy sample (not a
  proper multi-scenario representation like load uncertainty gets), so an unpinned,
  price-responsive bid would effectively be placing all-in bets on noise. Conclusion: keep
  pinning, but note explicitly that pinning doesn't eliminate the gaming incentive, it just
  relocates it into `pl_hat` — which is exactly the mechanism found above.
- **The "k" notation is arguably misleading**: `k` is only ever evaluated at its two extremes
  (0, 1) in this study — never as a genuine continuous trade-off — but the `(1-k)·A + k·B`
  notation implies a tunable blend that doesn't exist here. The two settings are structurally
  different optimization problems (different variables/parameters exist or don't), not points
  on a shared spectrum. Recommendation under discussion: rename descriptively in prose
  ("economic corner" / "dispatchability corner"), and/or test at least one intermediate k value
  to make the "trade-off parameter" framing actually earned rather than assumed (this would also
  serve as strong dose-response evidence for the bias mechanism above).
- **Single-price corner's value**: shown to be a valuable *negative control* — it has zero
  measured bias/gaming, because it lacks the asymmetric settlement structure the exploit depends
  on (`C_imb ≡ 0` identically for single price at k=0). This comparison is what let the
  diagnosis isolate price *asymmetry* as the causal driver, rather than something generic to
  DFL or bid-pinning. Recommendation under discussion: keep it as a control/validation point in
  the write-up, not as a full equally-weighted corner requiring its own deep results section.

## Prior framing advice given (worth scrutinizing, not accepting uncritically)

- Advised picking one narrative spine rather than presenting the GRU architecture, surrogate
  model comparisons, and convex-vs-complex formulation comparisons as separate, equal-weight
  contributions — suggested treating those as supporting methodology for the bias-finding
  narrative instead.
- Advised against an early proposed framing — "DFL is validated by its positive effect,
  especially with asymmetric costs" — since the asymmetric/dual corners are exactly where the
  exploit was found to be worst; that framing risks being contradicted by the author's own
  `C_da`/`C_imb` evidence.
- Suggested (tentatively) that the honest narrative arc might be: found DFL improving reported
  regret → investigated why → diagnosed a structural objective/settlement mismatch → proposed
  and (pending) implemented a fix → report the corrected picture. This was offered as one
  option, not settled — worth pressure-testing from a researcher's-contribution perspective
  rather than taking as given.

## The actual question for this conversation

What is this dissertation's real contribution, stated as a researcher would state it — and
does the current plan (validate DFL works, especially under asymmetric costs → compare 1-stage
vs. N=3 surrogates → compare N=3 corners against each other) actually demonstrate that
contribution, given everything above? Two candidate framings worth explicitly weighing against
each other (not the only two, but a starting point):

1. *"DFL improves battery dispatch decisions in a UK-market setting"* — a positive-results
   claim, increasingly complicated by the evidence above; would require the fixes to be
   implemented and to actually hold up before this claim is safe to make as the headline.
2. *"A case study in how decision-focused-learning objectives can silently misalign with
   settlement mechanics, diagnosed rigorously in a realistic battery-dispatch testbed"* — a
   methodological/critical contribution, which the evidence already strongly and directly
   supports, independent of whether the fixes fully resolve the issue.

These aren't necessarily exclusive, but they lead to different emphases, different result
sections, and different claims about what "validates" the work — worth deciding deliberately
rather than letting the write-up structure default to whichever framing was assumed before this
investigation happened.
