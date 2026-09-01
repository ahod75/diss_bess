# Why does DFL provide any benefit over standard statistical learning?

Scope: capstone synthesis for this folder. Third and final structural revision. The first
version answered the title question with a small positive result; the second traced that
result to a training-dynamics confound and a real but modest calibration-correction
mechanism; a full N=5 seed-averaging campaign (`7_model_training/train_crps_only.py`/
`train_dfl_forecasts.py --seed`) then showed neither survives averaging. This version
states that plainly, explains it mechanistically, and treats the investigation's own
near-misses as the citable contribution, rather than softening or omitting the null
result -- see `9_analysis/beichter_et_al_relation.md` for why that's the more defensible
choice than reverting to the pre-campaign numbers. A later, RQ2-motivated extension
(`8_testing/cross_eval_seeded.py`, Part 3b) found one effect that *does* survive
seed-averaging -- not a within-mode benefit, but an asymmetric cross-price transfer harm.

**Research questions this document answers** (RQ1: does training move the forecaster
toward the behaviour each settlement rewards, and at what forecast-quality cost -- Parts
1 and 3a; RQ2: does that behaviour transfer to the other settlement structure -- Part 3b;
RQ3: does it survive deployment under the recourse-aware full-robust formulation --
answered implicitly by every number above, since all evaluation already runs through
`setup_full_robust` regardless of the `setup_1stage` training formulation, no separate
comparison built). See `9_analysis/rq_findings_summary.md` for the RQ-organised version
of this document's content.

## Part 1 -- what's true regardless of any specific training run

Two facts, proven algebraically and verified against the running code via autograd, not
measured on any particular checkpoint. Nothing below changes them.

### 1.1 Dual-price has a genuine, provable reason a benefit *could* exist

Standard/two-stage training (pinball/CRPS loss) optimizes for calibration. Per Cameron et
al. (2021), that coincides with the economically optimal forecast only if the downstream
objective is *linear* in the forecast target (`cameron_et_al_relation.md` Section 1) --
if nonlinear, `E[f(Y)] != f(E[Y])` (Jensen's inequality), so a mean-only forecast is
provably wrong for the decision even with a perfect forecaster. Dual-price's settlement,
`C_imb = pi_up*(imb)+ + pi_down*(-imb)+`, is genuinely nonlinear -- piecewise-linear,
kinked at `pl_hat = realised`. The exact local training gradient, derived and
autograd-verified independently of Cameron et al., is an asymmetric pinball loss
targeting a price-determined quantile:

```
Total(pl_hat) = dt * (pi_up - pi_down) * pinball_tau(realised, pl_hat) + const
tau = (pi_up - pi_da) / (pi_up - pi_down)
```

verified to ~1e-14 precision (`dual_price_gradient_mechanism.md` Section 5). `tau` is not
degenerate on real data -- mean 0.43 (train) / 0.50 (test), spans the full `[0,1]` range.

### 1.2 Single-price has no such channel, and no accuracy anchor at all

Single-price's settlement, `C_imb = pi_imb*(realised - pl_hat)`, is exactly linear.
Cameron et al.'s theorem says this should mean *zero* genuine two-stage/end-to-end gap.
Its local gradient, `d(f_dfl)/d(pl_hat_h) = dt*(pi_da_h - pi_imb_h)`
(`single_price_gradient_mechanism.md` Section 1, exact), has **no minimum** -- nothing
in the objective pulls it back toward accuracy. This is a structural, checkable-before-
training property of the formulation, not an empirical finding: a linear cost with no
minimum gives decision-focused training no floor. Whatever signal reaches the forecaster
during training -- genuine or incidental -- can degrade calibration without limit in
principle. Section 3 below is the empirical consequence of exactly this fact.

## Part 2 -- what a single-seed investigation suggested (context, not the finding)

Before seed-averaging, this folder built a substantial mechanistic account: a real,
solar-correlated calibration bias in baseline's own forecaster (confirmed intrinsic to
the forecaster, not the copula, `bias_spread_solar_mediation.md` Section 8); a causal
ablation showing DFL's shift and an economic-loss-free control's shift pointed the same
direction; hour-of-day R^2 figures of 0.93-0.96 linking that shift to solar irradiance.
That account is not fabricated -- every step was verified against real code and real
data, on the checkpoints that existed at the time. It just turned out to describe one
seed, not a property of the method. Kept here as context for why the investigation looked
the way it did, not as a standing claim.

## Part 3 -- what survives seed-averaging, and what doesn't

### 3a. Within-mode (native) economic effect and mechanism -- null

A full N=5 seed-averaging campaign (seeds 20240801, the original, plus 20240802-20240805;
`aggregate_results.ipynb` Q1, `forecast_characteristics.ipynb` Sections 7/10), matching
Beichter et al. (2025)'s own convention:

**Economic effect** (% cost change vs. baseline, mean +/- std across seeds):

| market | forecaster | pct_change | range | win rate |
|---|---|---|---|---|
| single-price | CRPS-only control | -0.109% +/- 0.090 | -0.207% to +0.059% | 55.2% +/- 3.4 |
| single-price | DFL | **-0.011% +/- 0.132** | -0.269% to +0.098% | 51.1% +/- 3.6 |
| dual-price | CRPS-only control | -0.020% +/- 0.165 | -0.247% to +0.254% | 45.6% +/- 11.0 |
| dual-price | DFL | **+0.060% +/- 0.134** | -0.114% to +0.223% | 42.6% +/- 7.4 |

Neither DFL mean is distinguishable from zero given its own standard deviation.
Dual-price's mean is on the *harmful* side of zero. The control fares no better than DFL
-- this is not "the control wins," it's that every point estimate in this project's
headline economic results sits within noise this large, DFL's and the control's alike.

**Mechanism** (hour-of-day solar-bias R^2, this project's most-cited mechanistic number,
previously reported as a single-seed 0.93-0.96):

| market | per-seed R^2 | mean +/- std |
|---|---|---|
| single-price | [0.955, 0.947, 0.001, 0.055, 0.448] | 0.481 +/- 0.414 |
| dual-price | [0.933, 0.873, 0.912, 0.799, 0.013] | 0.706 +/- 0.350 |

For single-price, whether the "solar mediates the bias" story appears at all is close to
a coin flip across seeds -- one seed shows literally no relationship. Dual-price is more
consistent (4 of 5 seeds strong) but still has one complete outlier. Both the economic
effect and the mechanism built to explain it fail to reproduce.

### 3b. Cross-price transfer (RQ2) -- asymmetric, and this one holds

Does a forecaster trained under one settlement structure perform well when its quantile
output is instead run through the *other* structure's dispatch/settlement pipeline
(`8_testing/cross_price_eval.ipynb`, deliberately crossing `eval_raw.py`'s price-mode
firewall)? Seed-averaged (N=5, same seeds and convention as 3a;
`8_testing/cross_eval_seeded.py`), against each direction's own native (within-mode) N=5
result for comparison:

| direction | forecaster | pct_change vs. baseline | range | win rate | t-test vs. 0 |
|---|---|---|---|---|---|
| on single-price pipeline | DFL single-price (native) | -0.011% +/- 0.147 | -0.269% to +0.098% | 51.1% | p=0.877 |
| on single-price pipeline | DFL dual-price (**cross-eval**) | -0.022% +/- 0.069 | -0.124% to +0.060% | 50.1% | p=0.512 |
| on dual-price pipeline | DFL dual-price (native) | +0.060% +/- 0.149 | -0.114% to +0.223% | 42.6% | p=0.418 |
| on dual-price pipeline | DFL single-price (**cross-eval**) | **+0.264% +/- 0.140** | **+0.025% to +0.391%** | 34.0% | **p=0.013** |

Unlike every other headline number in this document, the bottom row -- the single-price
forecaster cross-evaluated on the dual-price pipeline it wasn't trained under -- does not
straddle zero: **all 5 seeds are harmful, every one**, ranging +0.025% to +0.391%, and
win rate collapses from a coin flip to 34.0% (24.9%-41.8% across seeds). The 5/5-seeds-agree-in-sign fact is stronger evidence than the p-value alone
(N=5 is small for a parametric test; unanimous sign across every independently-trained
seed is a distribution-free result no single seed's noise can produce by chance as
easily). The other direction -- dual-price-trained forecaster evaluated on the
single-price pipeline -- shows no such effect: near zero, not significant, and actually
*tighter* (std 0.069 vs. native single-price's own 0.147).

This asymmetry is exactly what Part 1.2/4.1 predict, and independently confirms them: a
forecaster trained under single-price has no accuracy-restoring minimum pulling it back
toward calibration (Part 1.2) -- whatever state that unconstrained drift leaves it in,
deploying it under dual-price's genuinely convex, accuracy-seeking settlement (V-shaped,
minimised exactly at the realised value) has nowhere good for that drift to land, and it
consistently doesn't. Dual-price training, by contrast, has a real (if seed-unstable)
channel pulling toward the price-implied quantile (Part 1.1) -- deploying that under
single-price's settlement, which is *structurally indifferent* to the forecast's accuracy
(linear, no minimum), has no mechanism to punish or reward whatever calibration made it
across, and it doesn't. The direction that lacks an accuracy anchor at training time is
the one that reliably costs money when exposed to a settlement that has one; the
direction that has an anchor transfers close to neutrally into a settlement that doesn't
need it. **This is the one economic finding in the project that both survives
seed-averaging and has a mechanistic account that predicted it in advance, rather than
being fitted to it after the fact.**

## Part 4 -- why: three identified, checkable factors

Not "DFL didn't work" -- three specific properties of this setup, each independently
sufficient to predict high seed-variance, all three present at once.

**4.1 No accuracy-restoring minimum (single-price specifically).** Established
unconditionally in Section 1.2. A linear objective gives training no floor -- whatever
noise reaches the forecaster is not damped. Predicts single-price should be the *more*
unstable of the two modes; it is (R^2 std 0.414 vs. dual's 0.350; economic std 0.132 vs.
0.134 is closer, but single-price's control is comparatively tighter at 0.090 vs. dual's
0.165 -- see 4.2 for why the control's own instability needs a separate explanation). It
also independently predicts Part 3b's cross-price transfer asymmetry -- a mode with no
accuracy anchor should produce a forecast that costs money once exposed to a settlement
that has one, and does, consistently, across all 5 seeds -- the strongest empirical
support this factor has, since it was predicted before the transfer check was built, not
fitted after.

**4.2 Full fine-tuning of a small model on a small dataset, not parameter-efficient
fine-tuning.** This project fine-tunes 100% of a small GRU's weights on 365 training
days, batch_size=8, early-stopped on a 181-day validation signal. Beichter et al. (2025)
-- same problem class, dispatchable feeder, `beichter_et_al_relation.md` -- constrain
fine-tuning to under 0.7% of parameters (LoRA/DoRA on a 91M-parameter foundation model)
and report N=5-seed results with tight, non-overlapping standard deviations (DFF
12.75+/-0.04 vs. PFF-MSE 14.08+/-0.05). The contrast is a controlled comparison of
exactly this factor: same problem family, same seed-averaging convention, opposite
parameter-efficiency regime, opposite reproducibility outcome. This factor alone would
produce high variance even for dual-price, whose objective *does* have a minimum -- and
dual-price's economic result is indeed still unstable (std 0.134, mean wrong-signed)
despite Section 1.1's genuine theoretical channel.

**4.3 The missing control was never run until this campaign.** Every DFL evaluation
compared against a *frozen* baseline, never a matched-budget, task-loss-free fine-tune.
The proper control (`train_crps_only.py`, bypassing `self_balanced_loss` entirely --
feeding it a zeroed `f_dfl` degenerates the combined loss to identically zero rather than
gracefully reducing to pinball-only, a real trap caught before it produced a meaningless
run) shows comparable instability to DFL itself once seed-averaged. Without this control,
a single favourable seed for DFL against a single frozen baseline is indistinguishable
from a genuine effect -- which is exactly what happened here on the original seed.

## Part 5 -- a checklist, the reusable contribution

Five checks this project's own near-misses motivate, each with a quantified example of
the failure mode it catches:

1. **Settlement/formula sign audit.** A sign bug in the dual-price imbalance formula
   produced an apparent -5.75% benefit, 84.4% win rate (`balanced_single_vs_dual_
   findings.md` Section 0) -- a confident-looking, publishable-seeming number, purely a
   bug.
2. **Data anomaly audit.** A single anomalous test-year price hour (1975 GBP/MWh, ~68
   std devs from the mean) accounted for 174% of dual-price's entire second-half
   baseline-relative benefit (Section 0b, same document) -- one data point out of 8784.
3. **Matched-budget, task-loss-free control**, not just a frozen baseline (Part 4.3) --
   the single check that most directly distinguishes "DFL's mechanism helped" from "more
   training of any kind helped."
4. **N>=5 seed-averaging**, matching Beichter et al.'s own standard -- Part 3's entire
   contribution. An informal, non-early-stopped, single-run version of the Part 4.3
   control produced a diurnal shift *opposite in sign* to a properly early-stopped run
   differing only by which epoch's weights got used -- the seed/stopping-point axis is
   not a minor concern here, it flips conclusions outright.
5. **Structural check of the task objective**, before training: does it have a genuine
   minimum at the true value (Part 4.1)? Checkable from the formulation alone, no
   training run required, and it predicts which of two modes will be more fragile before
   either is ever trained.

A DFL evaluation that passes all five and still reports a clean effect (as Beichter et
al.'s does) has a real result. This project's early, pre-campaign numbers would have
failed checks 1, 2, 3, and 4 simultaneously, and did not fail visibly until each was run.

## Part 6 -- bottom line

No positive claim about DFL's economic or mechanistic *benefit* in this project survives
seed-averaging -- but one negative claim does: training under the settlement structure
with no accuracy anchor produces a forecast that reliably costs money once exposed to a
settlement structure that has one (Part 3b), unanimous across all 5 seeds and predicted
in advance by Part 1.2/4.1, not fitted to the result afterward. What else survives: the
exact, unconditional mathematics of Part 1 (Cameron et
al.'s theorem applies exactly to dual-price's settlement, the DPP-compliant epigraph
reformulation, the sign-bug fix, the pinball-quantile-`tau` identity, verified to
floating-point precision) -- none of it an empirical claim about a trained model, all of
it true regardless of what any specific run produces. What also survives is a
methodological result with teeth: this setting's null outcome is *explained*, not just
observed, by three identified and checkable factors (Part 4), sharpened by a direct,
same-problem-class contrast with a positive result (Beichter et al.), and it produced a
concrete, reusable checklist (Part 5) built from the project's own documented failure
modes. The honest framing is not "DFL beats CRPS here" and not "DFL doesn't work" as a
general claim -- it is that this project built and ran the checks a DFL benefit claim
actually needs to clear, most of them missing from the wider evaluation practice this
folder's own early drafts also fell into, and reported exactly what they showed.
