# Relating Cameron et al. (2021), "The Perils of Learning Before Optimizing", to this project

Scope: how Cameron, Hartford, Lundy & Leyton-Brown's theoretical account of when
end-to-end (decision-focused) learning beats two-stage learning relates to the
single-price/dual-price gradient mechanisms already derived in this folder. Read in full
(`9_analysis/Cameron et al. - 2021 - The Perils of Learning Before Optimizing.pdf`).
Bullet-form working notes, grown out of a chat discussion.

## 1. The paper's core mechanism, precisely

- Setting: predict some stochastic target `Y|X`, then optimize a downstream decision `z`
  against that prediction. Two-stage: train a predictor via a standard loss (they focus on
  MSE, whose Bayes-optimal solution is `E[Y|X]`), then plug the point prediction into a
  deterministic optimization. End-to-end: differentiate the predictor's parameters
  directly against the downstream task loss.
- **Central result**: if the downstream objective is *linear* in the stochastic target,
  two-stage is provably optimal -- zero gap versus the true stochastic optimum (Theorem
  3.8 condition (i), and the general "linear in y" corollary in the appendix, both direct
  consequences of linearity of expectation).
- The gap only opens when the objective is a *nonlinear* function of the target, because
  then `E[f(Y)] != f(E[Y])` (Lemma 3.6 -- a Jensen's-inequality statement). A two-stage
  predictor plugging the mean into a nonlinear objective is optimizing the wrong quantity,
  regardless of how accurate that mean is -- this holds even in their idealised
  "error-free" setting (Bayes-optimal predictor, infinite data, no misspecification).
  End-to-end can adapt *which statistic* of the distribution it effectively targets, since
  it trains against the true downstream loss rather than a fixed proxy.
- Their headline technical contribution goes further: when the objective combines
  *multiple* prediction targets nonlinearly (they focus on products, `y1*y2`, e.g.
  demand x travel-time) and those targets are correlated, the two-stage/end-to-end gap
  can grow *unbounded* in problem dimension (Theorem 3.4), formalised via a connection to
  the *price of correlation* from stochastic optimization (Agrawal et al. 2012).

## 2. Direct mapping to dual-price -- the clean case

- `9_analysis/dual_price_gradient_mechanism.md` Section 5 already derived, independently
  of this paper, that dual-price's realised cost as a function of `pl_hat` is an
  asymmetric pinball loss targeting a **price-determined quantile**
  `tau = (pi_up - pi_da) / (pi_up - pi_down)`, not the mean -- verified numerically to
  floating-point precision.
- This is exactly an instance of Cameron et al.'s mechanism: `C_imb(realised)` is
  piecewise-linear (kinked at `realised = pl_hat`) -- genuinely nonlinear in the
  stochastic target -- so a mean-only prediction (what standard pinball/CRPS training
  targets) is provably not what the downstream objective needs, *even with a perfect
  forecaster*. Their paper supplies the formal vocabulary for a mechanism this project
  already found empirically: dual-price DFL isn't "improving accuracy" in the usual
  sense, it's correcting a target-statistic mismatch that would exist regardless of
  forecast quality.
- Consistent with Section 4 of `8_testing/balanced_single_vs_dual_findings.md`: the
  "DFL pushes mean toward its own median" theory was tested and found to hold only weakly
  for single-price and not at all for dual-price. Cameron et al. explains why the
  *median* was never the right thing to check -- the correct target statistic is
  `tau`-dependent and generally isn't 0.5.
- **Caveat on how far this explains the *trained* model, not just the local gradient**:
  `bias_spread_solar_mediation.md` (Section 5a) checked whether `tau`'s day-to-day
  variation actually shows up in how the trained dual-price model's bias pattern differs
  from single-price's -- it doesn't (partial R^2<0.001). What dominates the trained
  divergence instead is a shared, solar-conditioned shift that both price modes learn
  almost identically, with dual-price's version simply damped by a stable ~0.6x factor
  relative to single's (same document, Section 5) -- consistent with Cameron et al.'s
  mechanism *existing* (a genuine target-statistic gap that end-to-end training can close)
  but not with day-to-day `tau`-tracking being the *visible* mechanism by which the
  trained model closes it. The Jensen's-inequality argument correctly predicts a gap
  should exist for dual-price's nonlinear objective; it does not, on its own, predict --
  and the data does not support -- that the trained forecaster's behaviour is legible as
  live quantile-tracking.

## 3. Single-price as the boundary-condition check -- confirmed more cleanly than
## originally realised, once seed-averaged

- `9_analysis/single_price_gradient_mechanism.md` established that single-price's
  settlement, `C_imb = pi_imb*(realised - pl_hat)`, is *linear* in `realised`.
- Per Cameron et al.'s own theorem, a linear objective means two-stage (i.e. training the
  forecaster on standard pinball/CRPS loss alone, then plugging in the mean) should
  *already be optimal* -- zero genuine gap between two-stage and end-to-end, full stop.
- **This is now confirmed directly, not just argued around.** An early, single-seed
  reading of this project reported a modest single-price DFL benefit (-0.30% vs
  baseline) and went looking for a mechanism to explain it (a fixed, solar-conditioned
  shift, `bias_spread_solar_mediation.md`). Once properly seed-averaged at N=5
  (`why_dfl_helps.md` Part 3a), that "benefit" turned out not to reproduce: mean
  -0.011% +/- 0.132%, not distinguishable from zero, and the R^2 for the mechanism built
  to explain it swings from 0.001 to 0.955 across seeds -- a coin flip on whether the
  story appears at all. **This is exactly what Cameron et al.'s theorem predicts**: zero
  genuine two-stage/end-to-end gap for a linear objective. The single-seed "benefit" was
  noise the theorem correctly says shouldn't be there in expectation; the null result is
  the theorem's cleanest empirical confirmation in this project, not a failure to find
  what the theory promised.
- This is a genuinely sharper claim than "cite the paper because DFL beats two-stage
  here": the theory correctly predicts *zero* mechanism for single-price, and once tested
  at a proper evidentiary standard, that's exactly what's observed -- not approximately,
  not "a small residual benefit remains," but statistically indistinguishable from the
  theorem's exact prediction. Worth stating explicitly rather than citing the paper as
  uniform justification for either corner having "a benefit."
- **A second, independent confirmation**: if single-price genuinely has no
  accuracy-restoring mechanism (Cameron et al.'s theorem, exactly), a single-price-trained
  forecaster should show no reliable improvement -- but should also show nothing pulling
  it *back* toward calibration if training drifts it away, unlike dual-price. That
  prediction is what the cross-price transfer check (`why_dfl_helps.md` Part 3b, RQ2)
  independently confirms: a single-price-trained forecaster, deployed under dual-price's
  genuinely nonlinear settlement (Section 2 above), costs money reliably -- all 5 of 5
  seeds, p=0.013. The theorem doesn't just explain why single-price shows no within-mode
  benefit; it explains, in advance, why that same forecaster is the one that
  transfers badly once it meets a settlement structure that actually has a target
  statistic to miss.

## 4. What does *not* transfer

- Their headline technical contribution -- the price-of-correlation bounds, the
  unbounded-gap construction (Theorem 3.4) -- is specifically about *multiple*
  prediction targets combined nonlinearly with unaccounted correlation between them
  (their running example: demand and travel-time, both driven by a latent weather
  variable neither model conditions on). This project has **one** stochastic target
  (prosumption), not several correlated ones being combined into the objective's
  coefficients.
- So this project should not be presented as a demonstration of their main result --
  only of the simpler, single-target Jensen's-inequality principle that underlies it
  (their Lemma 3.6 / Theorem 3.7), which is the more basic half of the paper but is
  correctly and cleanly theirs to cite.

## 5. A deeper, more speculative connection worth a limitations sentence

- Their multi-target-correlation machinery *would* become directly relevant to the parts
  of this project's formulation that combine multiple correlated `xi` values nonlinearly
  across hours -- which is exactly what `full-robust`'s cross-hour recourse does:
  `R @ xi_samples.T` where `R = I + D_ch - D_dis` mixes the (copula-correlated) scenario
  deviations across all 24 leads through a genuine linear-decision-rule policy.
- But 1-stage training structurally cannot learn from this -- established early in this
  project's investigation: the epigraph is fully decoupled from the returned decision,
  `D_ch = D_dis = 0` throughout training (see `single_price_gradient_mechanism.md`
  Section 1 and `dual_price_gradient_mechanism.md` Section 1). That's not an oversight;
  it's precisely because differentiating through the box-robustified full recourse
  formulation at every training step is computationally intractable (the original
  motivation for the 1-stage/full-robust split, documented in the commit history: "the
  robust formulation was far too computationally intense to perform a backwards pass with
  CVXPYLayers").
- So there's a real, citable point here: the design choice that makes DFL training
  feasible in this project (1-stage) may be foreclosing exactly the kind of
  correlation-exploiting benefit Cameron et al.'s theory says could exist in the
  full-robust formulation this project only ever uses for *evaluation*, never for
  training. Worth a limitations-section sentence, not a full separate investigation --
  testing it properly would mean training against the full-robust formulation directly,
  which is the exact computational cost this project's whole 1-stage design was built to
  avoid.

## Recommendation for the write-up

Three distinct uses, not one blanket citation:
1. Cite it to formally explain the dual-price mechanism (Section 2 above) -- this is the
   strongest, cleanest connection.
2. Cite it *again*, contrastively, to explain why single-price shows no reliable
   within-mode benefit once seed-averaged, and why that same absence of an accuracy
   anchor is exactly what predicts its harmful cross-price transfer (Section 3) --
   demonstrates the theory's predictive precision on both counts, rather than using it as
   generic decision-focused-learning motivation.
3. One sentence in limitations (Section 5) noting the unexplored connection to
   full-robust's cross-hour recourse, framed as future work rather than a claim.
