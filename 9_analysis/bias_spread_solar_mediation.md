# Forecast bias vs price spread: a confound, corrected, then resolved via solar

Scope: follows up directly on `single_price_gradient_mechanism.md` Section 4 (the
duck-curve/bang-bang investigation) and applies the same question to dual-price. That
section's numbers (`R^2(bias,solar)=0.9546`, `R^2(bias,spread)=0.2902`) already used
`bias = mean_DFL - mean_baseline` (a delta against the price-naive baseline, loosely
labelled "bias" there) computed on **test-period** data only. This document makes that
delta framing explicit, extends it to dual-price and the training period, and adds two
checks that materially change the mechanistic story: (1) a confound check using each
forecaster's *own* bias against realised, and (2) whether the effect scales with spread
*magnitude* or is fixed. Bullet/working-notes style, same standard as the other docs in
this folder -- every number below is freshly computed and code-verified, not recalled.

**Method**: fresh forward passes of `baseline_forecaster_best.pt`,
`dfl_1stage_single-price.pt`, `dfl_1stage_dual-price.pt` over every gate-aligned day in
both the **training period** (2018, 365 days) and the **sealed test period** (366 days),
producing `pl_hat` (copula mean anchor) per day/hour via the same frozen
`FrozenCopulaSampler.mean_and_errors` used in training/eval. `bias_model = mean_model -
realised` (Section 10d sign convention: positive = over-predicts). `spread_single = da -
imb`, `spread_up = da - imb_up`, `spread_down = da - imb_down` (true, settlement-time
prices). All correlations below are **hour-of-day** (pool all days in the period, average
to one point per hour, correlate across the 24 points) unless marked pointwise.

## 1. The confound: does baseline's own bias already track the spread?

Never checked before -- prior work only ever looked at DFL's bias (or the delta) against
spread, never asked whether the **price-naive baseline** (zero price input, ever) shows
the same pattern on its own.

| period | forecaster's own bias vs... | `da-imb` | `da-imb_up` | `da-imb_down` |
|---|---|---|---|---|
| train | baseline | R^2=0.112 | R^2=0.215 | R^2=0.019 |
| train | DFL-single | R^2=0.005 | R^2=0.638 | R^2=0.088 |
| train | DFL-dual | R^2=0.002 | R^2=0.343 | R^2=0.100 |
| test | baseline | **R^2=0.532** | R^2=0.138 | **R^2=0.519** |
| test | DFL-single | R^2=0.067 | R^2=0.005 | R^2=0.190 |
| test | DFL-dual | R^2=0.043 | R^2=0.087 | R^2=0.001 |

Baseline -- which has never seen a price column in its life -- shows a bias-vs-spread
relationship **as strong or stronger** than either DFL forecaster's own bias, most
starkly in the test year (baseline R^2=0.532 vs DFL-single R^2=0.067 against the exact
spread that drives single-price's own gradient). Taken naively, a raw
DFL-bias-vs-spread correlation would be a **confounded, misleading** piece of evidence
for "DFL learned to track price spread" -- both baseline's bias and the spread likely
share a common solar/seasonal driver, and a naive check can't separate DFL's genuine
contribution from that shared confound. Pointwise (day-to-day) correlations are all
R^2 < 0.005 everywhere, both periods, all forecasters/spreads -- only the hour-of-day
level carries any real structure, consistent with prior checks elsewhere in this
project.

## 2. The correction: delta against baseline is the right control, and it survives

`delta_model = bias_model - bias_baseline` subtracts out exactly the confound in
Section 1 (any driver shared identically by both bias series cancels by construction).
This is what `single_price_gradient_mechanism.md` Section 4 was already computing
(labelled "bias" there); made explicit and extended to dual-price and the training
period here:

| period | delta vs... | `da-imb` (single) | `da-imb_up` (dual short) | `da-imb_down` (dual long) |
|---|---|---|---|---|
| train | delta_single | R^2=0.032 (r=+0.18, wrong-sign-ish) | -- | -- |
| test | delta_single | **R^2=0.290** (r=-0.539, theory-consistent sign) | -- | -- |
| train | delta_dual | -- | **R^2=0.699** (r=+0.836) | R^2=0.013 |
| test | delta_dual | -- | R^2=0.005 | **R^2=0.437** (r=-0.661) |

The test-year R^2=0.290 for single-price reproduces exactly (this was never in
question -- Section 1's confound doesn't touch it, since subtracting baseline removes
the confound by construction). But the *train* year shows a weak, wrong-signed
relationship (R^2=0.032), and for dual-price, the strong relationship is with the
*opposite* side of the mechanism in each period (`spread_up` dominates in train,
`spread_down` in test). The delta-vs-spread relationship is real but **period-fragile**
-- present and theory-consistent in test, weak or on the wrong side in train, for both
price modes.

**Sign convention note for the theory check**: the closed-form single-price gradient is
`d(f_dfl)/d(pl_hat_h) = dt*(pi_da_h - pi_imb_h)` (see the gradient-mechanism doc);
descent moves `pl_hat` opposite the gradient, so `spread_single > 0` (imbalance cheaper)
should pull `pl_hat`, hence `delta_single`, **down** -- a negative correlation is the
theory-consistent sign, which is what test year shows (r=-0.539) and train year does not
(r=+0.179).

## 3. The resolution: delta vs solar is stable across both periods; delta vs spread is not

If DFL had genuinely learned to read the spread directly, period-fragility here would be
hard to explain away. But the forecaster's inputs (`hist_cols`/`exo_cols`: prosumption,
solar, panel/ambient temp, calendar cyclicals) **contain no price column at all, in
either mode** -- so a direct "read the spread" mechanism is architecturally impossible.
Whatever's learned must be encoded as a function of the *price-free* features that
happened to correlate with spread during training. Checked directly:

| period | delta vs **solar** | `spread_single` vs **solar** (for reference) |
|---|---|---|
| train | delta_single: **R^2=0.960** (r=-0.980) | R^2=0.087 (r=-0.296) |
| test | delta_single: **R^2=0.955** (r=-0.977) | R^2=0.269 (r=+0.519) |
| train | delta_dual: **R^2=0.940** (r=-0.970) | (same spread_single row) |
| test | delta_dual: **R^2=0.933** (r=-0.966) | (same spread_single row) |

Both DFL deltas correlate with solar irradiance at R^2~0.93-0.96, **same sign, both
periods, both price modes, both statistically and practically indistinguishable between
train and test**. What actually differs between periods is the *spread's own*
relationship to solar (r=-0.296 train vs r=+0.519 test -- a genuine sign flip, previously
documented in `2_eda/seasonal_price_patterns.ipynb`). The delta-vs-spread fragility in
Section 2 is not evidence the learned mechanism is unstable -- it's a direct consequence
of correlating a stable thing (DFL's solar response) against an unstable proxy for it
(spread, whose own link to solar moves between years). Pointwise R^2 for delta vs solar
is lower (0.44-0.77, computed in Section 4 below) than the hour-of-day figures above, as
expected -- hour-of-day averaging cancels day-to-day weather-realisation noise that
pointwise data still carries.

**Revised mechanism statement, superseding Section 4 of `single_price_gradient_mechanism.md`**:
DFL did not learn to track the day-ahead/imbalance spread. It learned a strong, stable,
solar-conditioned downward shift relative to baseline -- present in both price modes,
both periods, at near-identical strength. The spread-correlation findings throughout this
project were always a downstream shadow of that solar mechanism, not the mechanism
itself; the price-spread's *own* seasonal instability (documented separately) is what
made the shadow move, not any instability in what DFL actually learned.

## 4. Does the solar-response magnitude scale with spread size? -- no

Two live hypotheses for *why* DFL's shift is solar-conditioned: (a) genuine price
learning, routed indirectly through solar as the best available proxy at training time --
in which case the size of the shift on a given day should scale with how large that
day's spread happens to be; or (b) a fixed policy baked into the weights, driven by
solar/battery-physics interactions largely independent of the market mechanism's
specifics -- in which case spread size shouldn't matter once solar is accounted for.

Tested via **pointwise** (day-level, not hour-of-day-averaged -- needed to retain the
day-to-day spread variation this question is actually about) partial correlation of
delta against spread controlling for solar, plus incremental R^2 from adding spread to a
solar-only linear fit of delta:

| period | delta vs spread \| solar | partial R^2 | R^2(solar only) | R^2(solar+spread) | incremental R^2 |
|---|---|---|---|---|---|
| train | single vs `da-imb` | 0.0027 | 0.767 | 0.768 | +0.0006 |
| train | dual vs `da-imb_up` | 0.0127 | 0.445 | 0.452 | +0.0071 |
| train | dual vs `da-imb_down` | 0.0004 | 0.445 | 0.445 | +0.0002 |
| test | single vs `da-imb` | 0.0002 | 0.754 | 0.755 | +0.0000 |
| test | dual vs `da-imb_up` | 0.0000 | 0.523 | 0.523 | +0.0000 |
| test | dual vs `da-imb_down` | 0.0008 | 0.523 | 0.524 | +0.0004 |

(Using `|spread|` instead of signed spread, to test a magnitude-only/sign-agnostic
effect, gives the same near-zero picture in every case -- not tabulated separately.)

Spread adds at most 0.7 percentage points of R^2 beyond solar alone, in every
combination, both periods -- **no evidence the shift scales with spread size**. This
favours (b): a fixed, solar-conditioned policy, not a live price-magnitude-responsive
one.

This also has a structural explanation, not just an empirical one: since spread is never
a forecaster input, the network cannot know a new day's actual spread value at inference
time -- it can only condition on solar/hour/season/history. The training-time gradient
*was* exactly proportional to that day's spread (the verified closed form), but that
signal gets absorbed and averaged across all 365 training days into fixed weights;
once frozen, the resulting function has no channel left to re-modulate itself per-day
against a quantity it structurally cannot see. The near-zero incremental R^2 is close to
the only possible outcome given the architecture, not a surprising empirical result.

## 5. Single vs dual: shared direction, different magnitude

Section 3 found near-identical solar correlations for `delta_single` and `delta_dual`
(R^2~0.93-0.96, both periods) -- raising the question of whether the two price modes are
actually learning distinguishable, mode-specific behaviour at all, given they're
supposedly converging toward different "dominant bidding strategies" (single-price:
linear, no accuracy anchor, corner-seeking; dual-price: piecewise-convex, genuine
accuracy-seeking valley at `pl_hat=realised` -- see the two gradient-mechanism docs).
Checked directly by correlating the two deltas **against each other**, not just each
against solar:

| period | corr(delta_single, delta_dual) hour-of-day | corr(...) pointwise | slope (delta_dual ~ delta_single) | mean\|delta_single\| / mean\|delta_dual\| |
|---|---|---|---|---|
| train | R^2=0.964 (r=0.982) | R^2=0.538 (r=0.733) | 0.606 | 1.313 |
| test | R^2=0.967 (r=0.983) | R^2=0.579 (r=0.761) | 0.605 | 1.329 |

Two distinct findings, not a contradiction:

- **At the hour-of-day level, the two modes share almost the same shape**
  (R^2~0.96-0.97) -- both deltas are downstream of the same shared solar/duck-curve
  driver (Section 3), so of course they point the same direction with a similar relative
  profile. This is *why* Section 3's solar correlations looked so similar between modes --
  not two independent mechanisms coincidentally arriving at the same answer, but both
  tracking one shared cause.
- **They consistently differ in magnitude, in exactly the direction each mechanism
  predicts.** `delta_dual ~= 0.605 * delta_single` (slope, both periods, essentially
  unchanged: 0.6061 train vs 0.6053 test) -- single-price's shift is **~31-33% larger**
  than dual's, and this ratio is one of the few quantities in this entire investigation
  that is *not* period-fragile (1.313 train vs 1.329 test, compare to the spread
  correlations in Sections 1-2 which swing wildly or flip sign between periods). This
  matches the two mechanisms exactly: single-price has no minimum in its settlement cost
  (linear in `p_imb`, nothing pulls it back once it starts leaning into the shared
  duck-curve direction), while dual-price's settlement is piecewise-convex with a genuine
  minimum at `pl_hat=realised` -- an accuracy anchor that tempers the same underlying
  pull rather than letting it run as far.

The genuinely mode-specific mechanisms (single's spread-sign sensitivity, dual's
price-implied-quantile tracking -- see Section 5 of `dual_price_gradient_mechanism.md`
for the pinball-quantile derivation) are not
absent -- they show up at the **pointwise** level, where only R^2~0.54-0.58 of variance
is shared between the two deltas (i.e. 42-46% is mode-specific, day-to-day divergence).
They're just smoothed into a near-common shape once averaged to hour-of-day, which is why
Section 3's hour-of-day view alone made the two modes look almost indistinguishable.

**Answer to "don't they still seem to be moving toward the dominant strategy in each
market?"**: yes -- refined rather than contradicted. Both markets' price asymmetries
share a common solar-driven cause, so both models lean the same direction; *how hard*
they lean is mode-specific and matches theory (bang-bang leans further, accuracy-seeking
is reined in, by a stable ~1.3x ratio). This is compatible with Section 4's magnitude-vs-
spread-size null result: neither model's *day-to-day* shift scales with that day's actual
spread size (both are solar-locked, not spread-magnitude-responsive), but the two modes'
*overall willingness to commit* to the shared direction still differs systematically and
matches each mode's settlement shape.

### 5a. What actually drives the ~42-46% pointwise divergence -- mostly noise, with a small stable second-order solar curvature

Tested the obvious economically-motivated candidates against the divergence directly.
Two framings, since raw divergence (`delta_single - delta_dual`) is partly a mechanical
consequence of the two deltas' shared ~0.605 linear relationship not fully cancelling (it
inherits a diluted version of whatever drives `delta_single` itself, e.g. raw divergence
vs solar: R^2=0.224 train / 0.203 test -- not a new finding, just leakage from Section 5's
shared component). The clean target is the **residual**: `resid_dual = delta_dual -
(intercept + slope * delta_single)`, i.e. whatever dual does that isn't already explained
by the shared linear relationship with single.

Pointwise (n=8760/8784), `resid_dual` vs each candidate -- **all negligible**:

| period | vs solar | vs `spread_single` | vs `spread_up` | vs `spread_down` | vs `tau-0.5` | vs `bias_baseline` |
|---|---|---|---|---|---|---|
| train | R^2=0.0013 | R^2=0.0065 | R^2=0.0108 | R^2=0.0000 | R^2=0.0000 | R^2=0.0044 |
| test | R^2=0.0093 | R^2=0.0001 | R^2=0.0001 | R^2=0.0020 | R^2=0.0001 | R^2=0.0027 |

None of the theoretically obvious drivers -- spread sign/magnitude, dual's own
price-implied quantile `tau = (pi_up-pi_da)/(pi_up-pi_down)` (mean~0.43-0.50, spans the
full `[0,1]` range, confirmed not degenerate), or baseline's own forecast error (a proxy
for "how much accuracy-correction pull dual should feel that day") -- explain more than
~1% of the residual pointwise, in either period. Partial correlations (each controlling
for the other) confirm this isn't a suppression artefact between `tau_dev` and
`bias_baseline`.

Decomposed `resid_dual`'s own variance into hour-of-day-systematic vs pure noise:
**only 8.8% (train) / 8.0% (test)** of the residual's variance is explained by its
hour-of-day group means -- the other ~91-92% is unstructured, idiosyncratic day-to-day
variation with no candidate variable tested here able to account for it (plausibly the
optimizer-level entanglement already flagged in `single_price_gradient_mechanism.md`
Section 5: Adam, self-balanced-loss reweighting, gradient clipping, cross-hour/
cross-quantile weight sharing).

But the small systematic part that *does* exist is real and highly stable: the
hour-of-day `resid_dual` profile correlates between train and test at **R^2=0.990**
(near-perfect), and against solar (hour-of-day) at R^2=0.783 (train) / 0.734 (test) --
peaking at hour 12 (+0.053 train, +0.047 test) and troughing around hour 4 (-0.044,
-0.040). Since `resid_dual`'s sign convention is "more than the linear single-derived
prediction," a positive peak at midday means **dual pulls back even further than the
0.605x shared ratio alone would predict, specifically at peak-solar hours** -- i.e. the
single-vs-dual relationship isn't purely linear/proportional; it has a genuine, small,
stable non-linearity where dual gets extra-cautious (relative to single) exactly when
solar/prosumption volatility is highest, on top of the already-established ~0.605x
overall damping (Section 5). Directionally sensible for an accuracy-anchored mechanism
(more caution exactly when the forecasting problem is hardest), but this is a
second-order refinement (~8-9% of divergence variance) on top of Section 5's main
finding, not a new dominant mechanism -- the headline result of this subsection is the
**null**: none of the obvious price-mechanism candidates explain the mode-specific
divergence; it's mostly noise with a small, curvature-shaped solar footnote.

## 7. Causal ablation: fine-tuning with zero economic loss weight still produces a diurnal shift

**Superseded by a proper version of this same control -- see `why_dfl_helps.md` Part 3a/4.3.**
This section's ablation was an informal, fixed-5-epoch run; rebuilding it properly (same
training budget/warm-start as every DFL corner, early-stopped on validation pinball loss
rather than a fixed epoch count) reverses the sign found below -- the proper control's
shift moves the *same* direction as DFL's own, not opposite, and on the original seed
beat DFL outright on most headline metrics. That single-seed "beats DFL outright" reading
is itself now superseded again: the N=5 seed-averaging campaign (`why_dfl_helps.md`
Part 3a) found the control no more stable than DFL, not reliably better than it. The
mechanism-level conclusions below (a diurnal shift appears with zero economic signal;
period-stability doesn't prove price-specificity) still stand; the specific *direction*
and *magnitude* reported here do not, and the reversal itself is the more important
finding -- it demonstrates the underlying quantity is unstable to which epoch's weights
get used, not just which seed.

Sections 1-5a establish *what* the trained models' shift looks like but stop short of a
causal test of its origin -- every correlational candidate (spread, solar, `tau`, clock
time) is too mutually confounded in this one-year, seasonally-structured dataset to
distinguish "genuinely price-driven, mediated through solar" from "a generic,
price-incidental training-dynamics artifact that happens to be diurnally shaped." Ran the
one test that can discriminate them: fine-tuned a **fresh copy of the baseline weights**
(matching the real DFL warm-start, `train_dfl_forecasts.py` line ~100: "FRESH copy of the
baseline weights every corner") for 5 epochs on **pinball loss only** -- `f_dfl`'s weight
held at exactly zero, no dispatch solve, no `self_balanced_loss`, same
seed/lr(5e-4)/batch_size(8) shape as real DFL training otherwise.

**A strong, diurnally-shaped delta appears anyway:**

| period | delta_ablated vs solar (hour-of-day) | vs `hour_cos` | mean\|delta\| |
|---|---|---|---|
| train | R^2=0.904 | R^2=0.801 | 0.103 MW |
| test | R^2=0.908 | R^2=0.805 | 0.105 MW |

Nearly as strong as the real DFL models' own solar correlations (R^2=0.93-0.96, Section
3) and comparably period-stable (near-identical hourly profile train vs test). **So a
diurnal shift does not require any economic signal at all** -- most plausibly a
noisy-small-batch-SGD artifact (batch_size=8 here and in real DFL training, vs the
baseline's own original batch_size=64), systematic rather than random because
higher-solar hours are presumably harder-to-fit, higher-variance targets where noisy
gradient estimates drift consistently rather than cancel.

**But it is not the same artifact as DFL's -- the sign is opposite.** The ablation shifts
the forecast *up* at midday (+0.10 to +0.20 MW peak); real DFL training shifts it *down*
(Section 3). Two consequences follow, not one:

1. This rules out "DFL's shift is just this generic drift" as a complete explanation --
   the observed DFL delta is a **net** of a genuine, oppositely-signed economic pull
   fighting against a real generic drift of comparable size, not the drift alone. The
   true economically-attributable pull is plausibly *larger* than the ~0.10-0.13 MW
   measured in Sections 3/5 (roughly `net + drift-magnitude`, if the two are
   approximately additive -- not verified, a first-order reading), since it has to
   overcome this headwind to produce the observed net result.
2. **Period-stability, used throughout Sections 3-5 as evidence of "a real learned
   mechanism" (as opposed to noise), is not actually diagnostic of price-specific origin.**
   The generic drift is *just as* period-stable (R^2=0.904 train / 0.908 test) as the real
   deltas' solar correlation. Both are period-stable because both are downstream of the
   same underlying, period-stable diurnal/solar structure -- stability distinguishes
   systematic-vs-random, not price-driven-vs-generic-artifact. This is a retroactive
   caveat on how the evidence in Sections 3 and 5 should be weighed, not just a new data
   point: the R^2=0.93-0.96 solar correlation and the R^2=0.964-0.967 single-vs-dual shape
   correlation are real and unaffected as *measurements*, but were previously read as
   stronger evidence of "genuine learning" than they can actually support alone.
3. **Comparisons *between* single and dual price are more robust to this confound than
   claims about either delta's absolute magnitude.** Both modes share the same warm start
   and a similar training setup, so they likely carry a similar-sized dose of this same
   generic drift, which mostly cancels out of a *ratio* (Section 5's ~0.605 damping
   factor) even though it contaminates each one's raw magnitude.

## 8. Baseline's own calibration bias -- real, substantial, and not a copula artifact

A natural alternative hypothesis for single-price's puzzle (Section 2 of
`why_dfl_helps.md`: benefit with no theoretical target-statistic-mismatch channel
available): CRPS/pinball training targets calibrated *quantiles* (median included, via
its exact MAE-equivalent tau=0.5 pinball term), while `pl_hat` -- what the decision
actually uses -- is a copula-*derived* **mean**, computed downstream of those quantiles
by `FrozenCopulaSampler.mean_and_errors`, not trained directly. If that derived mean were
a poor estimate relative to a well-calibrated median, that mismatch alone (unrelated to
price) could explain part of the benefit.

**The gap is real but small.** `mean_b - median_b` (baseline's copula-derived mean minus
its own raw 0.5-quantile output) correlates with solar at R^2=0.207 (train)/0.125 (test),
negative at high-solar hours (e.g. hour 18: -0.069 MW train) -- consistent with a
left-skewed residual distribution there (net load = demand - solar; an occasional burst
of full solar output creates a low-load left tail that pulls the mean below the median).
Real, physically sensible, but small: ~0.037-0.039 MW, only 13-21% of the R^2 needed to
explain Section 3's delta magnitudes (~0.10-0.13 MW) -- too small to be the primary
driver.

**What it surfaced instead is bigger: baseline's mean *and* median are both
substantially, similarly biased relative to realised, not just relative to each other.**
`bias_mean` (mean-realised) correlates with solar at R^2=0.47 (train)/0.27 (test);
`bias_median` correlates *even more strongly*, R^2=0.68/0.31, comparable magnitude
(~0.29-0.36 MW), same sign (positive -- baseline **over-predicts** at high-solar hours,
at every quantile level checked, not just the derived mean).

**This directly rules out the copula as the source.** `bias_median` never passes through
`FrozenCopulaSampler` at all -- it's `q_b[:, median_idx]` straight from the raw baseline
forecaster, denormalised, nothing else. If the copula's PIT/interpolation machinery were
introducing the bias, the median should look meaningfully better-calibrated than the
mean; it doesn't -- it's comparable or slightly worse. Algebraically, `bias_mean -
bias_median = mean_b - median_b` -- exactly the small gap measured above -- so ~85-90% of
`bias_mean`'s total magnitude is already present in `bias_median`, inherited from the raw
forecaster, before the copula does anything. The bias is intrinsic to the baseline GRU
forecaster's own quantile outputs -- most plausibly genuine forecasting difficulty at
high-solar hours (weather-driven solar variability feeding the "known future" exogenous
input, harder-to-predict residual spread there) -- not an artifact of the mean-
reconstruction step.

**This discriminates between DFL and the Section 7 ablation, and not in the ablation's
favour.** Baseline over-predicts at high-solar hours; correcting that means moving the
forecast *down*. Real single-price DFL's delta is down there -- directionally consistent
with correcting this real, pre-existing, CRPS-blind bias. The Section 7 ablation's delta
is *up* at those same hours -- the wrong direction for a calibration fix, reinforcing that
it is generic SGD drift rather than a legitimate continued-optimization correction.

See `why_dfl_helps.md` Part 4 for how this and Section 7 above combine into the current
explanation of why single-price shows no reliable within-mode benefit once seed-averaged
(Part 3a), rather than evidence of one -- and Part 3b for why the same no-accuracy-anchor
property this section's ablation illustrates predicts (and gets) a real, seed-consistent
cost once that forecaster is deployed under a settlement structure that does reward
accuracy.

## 9. Net revision to the project's narrative

### 9a. Status of specific earlier claims

- **Unaffected**: the exact closed-form local gradients (both gradient-mechanism docs,
  Section 1/2 in each) -- these are algebraic facts about training-time dynamics,
  verified via autograd, untouched by any of the above. Everything in this document is
  about what the *trained* models do, never a challenge to the instantaneous mechanism
  itself.
- **Corrected**: any claim that DFL's *trained, deployed* forecast behaviour "tracks the
  price spread" is too strong and not well supported -- the honest claim is that it
  learned a fixed solar-conditioned shift, whose *origin* is plausibly (not proven) the
  training-period spread-solar link, but whose *operation* at inference time is
  solar-only and spread-magnitude-blind (Section 4). This specifically supersedes the
  "current honest position" closing `single_price_gradient_mechanism.md` Section 4, and
  qualifies (without invalidating) the dual-price target-statistic-correction framing in
  `cameron_et_al_relation.md` Section 2 -- see 6b.
- **Newly load-bearing**: the solar-conditioned shift's stability (R^2~0.93-0.96, both
  periods, both modes, Section 3) is a stronger, cleaner empirical anchor for the
  dissertation than the previous spread-correlation numbers ever were -- it should
  probably replace R^2=0.29 as the headline figure for "DFL learns a systematic,
  non-trivial, generalising bias adjustment," with the price-spread material reframed as
  a secondary discussion of *why* that adjustment points the direction it does, not
  *what* the adjustment is a response to.
- **Further corrected, by Sections 7-8**: that stability claim itself needed qualifying --
  a generic, price-free training artifact is *equally* period-stable (Section 7), so
  stability alone no longer counts as evidence the shift is price-specific. What does
  discriminate single-price's real shift from that artifact is *direction*: it points the
  way that corrects a real, substantial, pre-existing, solar-correlated calibration bias
  in baseline's own forecaster (Section 8), which the generic artifact does not. That
  bias is intrinsic to the baseline GRU forecaster, confirmed not a copula artifact.
- **Superseded again, by the N=5 seed-averaging campaign -- do not use R^2~0.93-0.96 as
  a headline figure.** The "newly load-bearing" recommendation two bullets up was itself
  a single-seed reading. Seed-averaged (`why_dfl_helps.md` Part 3a), single-price's R^2
  is [0.955, 0.947, 0.001, 0.055, 0.448] (mean 0.481 +/- 0.414) -- one seed shows
  literally no relationship, so "stable, ~0.93-0.96" describes one favourable draw, not a
  property of the method. Dual-price is comparatively steadier ([0.933, 0.873, 0.912,
  0.799, 0.013], mean 0.706 +/- 0.350) but still has one complete outlier. Neither
  belongs in the dissertation as a headline stability figure; the honest headline figure
  is the seed-averaged instability itself.

### 9b. What this implies for what each formulation learns

Pulling Sections 1-5a together into one statement, rather than leaving it as a sequence
of individually-corrected claims:

- **Both formulations learn nearly the same thing**: a static, solar-conditioned
  downward adjustment to the mean forecast, not a live, price-responsive policy. It
  cannot be anything else, structurally -- price is never a forecaster input (Section 3),
  so whatever training bakes into the weights has to be expressed purely as a function of
  solar/hour/season/history at inference time. The two modes' learned shifts share
  ~96-97% of their hour-of-day shape (Section 5) and both are governed almost entirely by
  solar, not by any given day's actual spread (Section 4).
- **The one thing that meaningfully distinguishes the two markets is a single scalar**:
  how hard the model commits to that shared direction. Single-price leans ~31-33% harder
  than dual-price, and this ratio is one of the most period-stable quantities in this
  entire investigation (Section 5). It follows directly from settlement shape: single's
  linear cost has no minimum to pull it back once it starts leaning into the shared
  duck-curve direction; dual's piecewise-convex cost has a genuine accuracy-seeking valley
  that tempers the same pull. Not a richer strategy -- less restraint.
- **The theoretically mode-specific mechanisms are real but don't survive into
  distinguishable trained behaviour.** The bang-bang/spread-sign story (single) and the
  price-implied-quantile-`tau` story (dual, `dual_price_gradient_mechanism.md` Section 5)
  are exact, verified facts about the *instantaneous local gradient* -- but they explain
  essentially none of how the two trained models actually diverge from each other
  (Section 5a: <1.3% of the residual divergence, either candidate, either period). What's
  left after removing the shared solar-driven, magnitude-scaled component is ~91-92%
  noise. So "dual-price learns to track a price-determined quantile" is true as a
  training-dynamics statement and not really true as a description of the deployed
  model's behaviour -- the signal gets smoothed away by Adam/self-balanced-loss/
  weight-sharing well before it becomes a visible, exploitable pattern.
- **This reframes the project's own headline economic result, rather than adding a
  footnote to it** -- or so it looked on the single seed this section was written from.
  Single-price showed the larger benefit over baseline (-0.30%, 57.7% win,
  `8_testing/balanced_single_vs_dual_findings.md`) than dual-price (-0.15%, 53.0% win) --
  previously a mildly puzzling asymmetry given dual-price's theoretically "cleaner"
  accuracy-seeking mechanism, explained here as single-price leaning harder into the same
  solar-shaped correction because nothing in its settlement anchors it back toward
  accuracy. **This entire paragraph is superseded by the N=5 seed-averaging campaign**
  (`why_dfl_helps.md` Part 3a): neither -0.30% nor -0.15% reproduces as a reliable
  mean effect (single-price N=5: -0.011% +/- 0.132%; dual-price: +0.060% +/- 0.134%,
  both indistinguishable from zero), and the "leaning harder into the shared correction"
  mechanism itself doesn't reproduce either (solar-bias R^2 for single-price ranges
  0.001-0.955 across seeds). The *qualitative* asymmetry argument -- single-price has
  nothing anchoring it back toward accuracy, dual-price does -- turned out to be exactly
  right, just not in the way this paragraph originally used it: it doesn't explain a real
  within-mode benefit gap (there isn't one, reliably), but it does correctly predict the
  cross-price transfer asymmetry found later (`why_dfl_helps.md` Part 3b) -- single-price
  training costs money once deployed under dual-price's settlement, consistently, across
  all 5 seeds.

**Net statement for the write-up -- superseded, kept for the record.** This section
originally concluded: "DFL training in this project is best described as learning a
modest, solar-conditioned, static bias correction... The single-vs-dual distinction
reduces to *how far* that shared correction is allowed to go, governed by settlement
convexity, not to *what* each mode learns." That was accurate for the single seed it was
computed from and is not the project's current position. **`why_dfl_helps.md` Part 6 is
the current net statement for the write-up**: no positive within-mode benefit survives
seed-averaging (Part 3a); the settlement-convexity asymmetry this section identified
does survive, but as an explanation for the cross-price transfer result (Part 3b, RQ2),
not for a within-mode benefit gap that turned out not to be real.

## Open threads

- **Substantially addressed by Sections 7-8, not fully closed.** The ablation (Section 7)
  rules out "the shift is entirely a generic artifact" (wrong sign) and rules out reading
  raw correlation/stability as sufficient evidence of price-specificity (the artifact is
  equally stable). Section 8 gives single-price's real shift a positive, non-price
  account of its *direction* (it corrects a real baseline calibration bias) that doesn't
  depend on resolving the solar/spread/clock-time confound at all. What's still not
  pinned down: whether the *magnitude and precise shape* of the genuine economic pull
  (net of the Section 7 artifact) is better explained as "correcting Section 8's
  calibration bias" or "tracking the price spread via a solar-mediated proxy" -- both
  point the same direction here and haven't been separated from each other, only jointly
  separated from the generic-artifact alternative.
- Partially addressed by Section 5 (delta_single vs delta_dual correlate at R^2~0.96-0.97
  hour-of-day, both periods, so whatever curve either mode traces against solar is at
  least highly consistent with the other mode's, in both periods) but not directly
  answered -- still worth an actual overlay plot of delta vs solar for both periods on
  shared axes, rather than inferring curve similarity from correlation strength alone.
- **Resolved (Section 5a)**: checked whether the ~42-46% pointwise divergence is driven
  by spread sign/magnitude, `tau`, or baseline's own error -- none of them explain more
  than ~1%. ~91-92% of the divergence is unstructured day-to-day noise (plausibly
  optimizer-level entanglement, not a clean economic signal); the small systematic part
  that remains (~8-9%) is a stable, solar-shaped second-order curvature (dual gets
  extra-cautious relative to single specifically at peak-solar hours), not a
  price-spread-driven effect. **Resolved by the N=5 seed-averaging campaign
  (`why_dfl_helps.md` Part 3a)**: the residual noise is not an artefact of this
  particular checkpoint or of insufficient averaging within one run -- the whole
  solar-bias R^2 signature this section (and Sections 3-8 generally) is built on swings
  from 0.001 to 0.955 across 5 independently-trained seeds for single-price (0.799-0.933
  for dual-price, tighter but still not stable). It is not a cleaner-training-setup
  problem; it is that the underlying quantity is this seed-unstable at this project's
  scale (full-parameter fine-tuning of a small model, 365 training days -- see
  `why_dfl_helps.md` Part 4.2 for the direct contrast with Beichter et al.'s
  parameter-efficient, seed-tight positive result on the same problem class).
- Cached per-(day,hour) forecast/price/solar data for both periods at
  `bias_spread_solar_{train,test}.parquet`, and the Section 7 ablation's checkpoint/data
  at `pinball_only_ablated.pt` / `ablation_{train,test}.parquet`, all in this session's job
  tmp dir (not committed to the repo) -- would need regenerating from `eval_raw.py`'s
  forecast/copula machinery and `pinball_only_ablation.py`'s training loop if this
  analysis is extended later.
