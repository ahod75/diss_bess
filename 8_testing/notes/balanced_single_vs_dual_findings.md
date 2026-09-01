# Balanced-battery single-price vs dual-price DFL: consolidated findings

Scope: `baseline` vs the two DFL 1stage checkpoints (`dfl_1stage_single-price`,
`dfl_1stage_dual-price`), balanced/"standard battery" archetype (C_ch=C_dis=2MW,
B_max=4MWh) -- now the ONLY archetype in the project (short_sharp/long_slow retired, see
`train_dfl_forecasts.py`'s docstring). Full sealed test year, 366 days
(2019-07-01..2020-06-30). Bullet-form working notes, not written up -- source material for
the eventual write-up.

**This version supersedes two earlier passes of these findings** -- a sign bug in the
dual-price imbalance settlement formula was found and fixed mid-project (see Section 0),
which changed the dual-price numbers substantially; separately, a genuine data anomaly was
found and fixed in the test-year `imb` price series (see Section 0b), which changed both
modes' headline economic numbers by a more modest amount. Every number below is post both
fixes.

## 0b. The March 2020 price anomaly -- what changed and why

- `imb` (test-year imbalance price) contained one extreme outlier: 1975.18 GBP/MWh at
  2020-03-04 18:00 -- ~68 standard deviations from the test-year mean (typical range
  roughly 0-90 GBP/MWh, 75th percentile ~47). A genuine data anomaly, not a real market
  event at that scale.
- Fixed by replacing 2020-03-04's `imb` column with 2020-03-03's (the previous day), then
  recomputing `imb_up = max(da, imb)` / `imb_down = min(da, imb)` consistently. Test-year
  max |imb| dropped from 1975.18 to 150.00; std from 28.96 to 20.19.
- This only touches **test-year settlement prices** -- training data (2018) is unaffected,
  so no checkpoint needed retraining; only evaluation (`eval_raw.py`'s full grid,
  `cross_price_eval.ipynb`, `aggregate_results.ipynb`) was re-run.
- **Impact was asymmetric between modes.** Checked directly before re-running everything:
  that single hour/day contributed 24.5% of single-price's entire second-half (Jan-Jun
  2020) benefit, but 174% of dual-price's -- meaning dual-price's second-half benefit was
  *entirely* attributable to this one anomalous day; excluding it, dual-price actually
  underperformed baseline over the rest of that half. Single-price's result was real and
  robust either way.
- **Net effect on full-year headline numbers**: single-price -0.30%->**-0.269%** (small
  reduction, ~10% relative); dual-price -0.15%->**-0.114%** (larger reduction, ~24%
  relative). Both remain genuine, directionally-unchanged benefits -- this was a magnitude
  correction, not a reversal, for both modes. See Section 1 for the full updated numbers.
- Metrics not affected by this fix, confirmed by direct comparison against the pre-fix
  values (unchanged, no updates needed below): CRPS, forecast bias/MAE by hour (Section 7 --
  pure forecast-vs-realised quantities, no price involved), net imbalance/volume
  (Section 5 -- p_imb is a physical quantity, `p_g - bid`, that doesn't depend on price),
  and the solar-mediation R^2 figures (Section 4 -- properties of the trained model
  weights, which never changed since training only ever used 2018 data).

## 0. The sign bug -- what changed and why (read this first)

- Both `dispatch_wrapper.realised_breakdown` (settlement, used for training AND eval) and
  `dispatch_objectives._build_dual_price_epigraph` (the decision-time LP, only non-inert
  at eval time via `setup_full_robust`) computed dual-price's imbalance cost as
  `C_imb = pi_up*(p_imb)+ + pi_down*(-p_imb)+` -- **both terms ADDED**, making `C_imb`
  always >=0: a party is charged for being short AND charged again for being long, never
  credited for surplus.
- Standard two-price (SBP/SSP-style) settlement should CREDIT the long/surplus side:
  `C_imb = pi_up*(p_imb)+ - pi_down*(-p_imb)+`. Verified this is what the fixed code now
  computes, is DPP-compliant, and correctly collapses to single-price's own
  `pi_imb * p_imb` formula when `pi_up == pi_down` (the buggy version collapsed to
  `pi_imb * |p_imb|` instead -- always >=0, not the same function at all -- a clean
  internal-consistency check the bug fails and the fix passes).
- Fixed in both files; the balanced dual-price checkpoint was fully retrained from the
  frozen baseline under the corrected formula (same seed, same hyperparameters); every
  dual-price evaluation number (baseline's included -- baseline is evaluated through the
  same `setup_full_robust` dual-price pipeline) was regenerated. Single-price is
  completely unaffected -- different code path, never touches this formula.
- **Net effect**: dual-price's apparent economic benefit collapsed from -5.75% (84.4% win
  rate) to -0.15% (53.0% win rate) -- see Section 1. Most of what looked like a large,
  genuine dual-price advantage was a settlement-accounting artifact, not the asymmetric-
  settlement mechanism itself. The mechanism is still real (Section 3) but far weaker in
  practice than first measured.

## 1. Headline economic results (Q1, `aggregate_results.ipynb`)

Post anomaly-fix (Section 0b); pre-anomaly-fix values in parens for reference.

- Single-price: baseline mean cost 1746.55 -> DFL 1741.85 (**-0.269%**, was -0.30%, win
  rate 57.65%, was 57.7%)
  - CRPS *worsens*, unaffected by the price fix: 0.268 -> 0.291
- Dual-price: baseline mean cost 1797.54 -> DFL 1795.49 (**-0.114%**, was -0.15%, win rate
  52.73%, was 53.0%)
  - CRPS stays flat / improves slightly, unaffected by the price fix: 0.268 -> 0.267
- Dual-price's benefit is now smaller than single-price's by a slightly larger margin than
  before (2.36x, was 2.0x) -- the anomalous day was propping up dual-price's number more
  than single-price's (Section 0b), so removing it widened rather than narrowed the gap.
  The part of the old headline that still survives: dual-price still doesn't trade accuracy
  away for its (now even more modest) economic gain, while single-price does.
- Dual-price's baseline cost (1797.54) remains close to single-price's baseline cost
  (1746.55, ~2.9% gap) -- essentially unchanged from the post-sign-fix, pre-anomaly-fix
  figures (was 1798.38/1747.25, ~3% gap); the anomaly fix barely moved the *baseline*
  numbers since it affects only one day's settlement price, not the forecasts baseline cost
  is otherwise driven by.

## 2. Single-price is structurally forecaster-invariant at the decision level

*(Unaffected by the bug/fix -- this section is unchanged.)*

- Proven: `p_ch_hat`, `p_dis_hat`, `p_da_rel` are IDENTICAL for baseline vs DFL at eval
  time (confirmed empirically -- `p_ch_r`/`p_dis_r` bit-for-bit equal despite very
  different `pl_hat`).
- Cause: single-price's decision objective is `C_da = pi_da @ p_da_rel * dt` only --
  `C_imb = 0.0` literally (not even a cvxpy expression) in the decision-time objective,
  since `E[pi_imb . p_imb] = 0` given the pinned bid. `D_ch = D_dis = 0` always
  (confirmed empirically across 3 solvers at gamma=0 -- no cost-relevant reason for the
  solver to pick anything else, though at gamma=0 this reflects solver behaviour on a
  genuinely degenerate objective, not a uniqueness guarantee the way gamma=1e-4 training
  gives).
  `pl_hat` never enters the decision LP as a Parameter at all.
- Consequence: entire single-price benefit reduces to ONE lever -- the forecast MEAN,
  weighted by local price spread. Exact identity (verified to floating-point precision):
  `baseline_cost - DFL_cost = sum_h (pi_da_h - pi_imb_h) * (pl_hat_base_h - pl_hat_dfl_h) * dt`

## 3. Gradient-level mechanism (training) -- revised post-fix

- Full differentiable chain unchanged: `theta -> q_norm -> q_phys -> copula
  interpolation (pl_hat, xi) -> cvxpylayers LP -> realised_breakdown -> self_balanced_loss`.
- `self_balanced_loss(L_base, f_dfl) = harmonic_mean(L_base, f_dfl)` exactly -- a
  scale-equalizer (whichever term is larger gets LESS weight on its own gradient that
  batch), not a "focus on the worse metric" mechanism. Unaffected by the fix.
- `xi_samples`/the epigraph has **zero** gradient effect on `f_dfl` during 1stage
  training in either mode (`setup_1stage` has no `D_ch`/`D_dis`, the epigraph is fully
  decoupled from the returned decision) -- established earlier, still true post-fix. Real
  hedging via `xi_samples` only exists at eval time (`setup_full_robust`).
- **The corrected settlement gradient, derived carefully this session**:
  - Single-price: `C_imb` linear in `pl_hat`, no minimum -- unchanged from before.
  - Dual-price, `C_imb` ALONE is now monotonically NON-INCREASING in `pl_hat` (no interior
    minimum on its own -- pushing the forecast up without bound keeps "earning" credit on
    the now-negative-capable term). This sounds alarming but isn't: **total realized cost**
    (`C_da + C_imb`, what `f_dfl` actually is) is still valley-shaped with its minimum
    exactly at `pl_hat = realised`, because `C_da`'s slope combines with `C_imb`'s to give
    slope `= pi_da - pi_up <= 0` for `pl_hat < realised` and `= pi_da - pi_down >= 0` for
    `pl_hat > realised` (since `pi_down <= pi_da <= pi_up` always, by construction of the
    `_fc` proxy columns).
  - The under-forecast-side restoring pull (`pi_up - pi_da`) is **identical** to before the
    fix -- the bug never touched that branch. The over-forecast-side restoring pull is
    **weaker** than before: `pi_da - pi_down` now, vs the old (buggy) `pi_da + pi_down` --
    strictly smaller since `pi_down >= 0`.
  - Practical read: dual-price training still has a genuine accuracy-seeking pull at
    `pl_hat = realised`, but is now asymmetric in STRENGTH -- weaker at correcting
    over-forecasting than under-forecasting. This directly explains Section 7's revised
    hour-of-day pattern (the midday overshoot softened, didn't disappear or reverse).
- **Diagnostic aside (not a reported result, a quick sanity check)**: trained both modes
  for 3 epochs on a 60-day subset using RAW `f_dfl` only (no pinball, bypassing
  `self_balanced_loss` entirely), to test whether either mode has a genuine accuracy
  anchor without pinball's help. Single-price's mean bias grew/drifted across epochs (no
  stable point, matches the "no accuracy anchor" prediction from the linear-gradient
  argument); dual-price's mean bias stayed flat/improved slightly (matches the valley
  argument above). Spread (q95-q05) was too noisy over just 3 epochs to say anything about
  whether it degrades without pinball's supervision (an open question, see below) -- `f_dfl`
  never depends on anything but `pl_hat`, so in principle nothing trains the other 18
  quantile levels' shape at all; a longer run would be needed to see if that matters in
  practice.

## 4. What actually drives the mean-shift (theories tested, in order)

- **Theory: DFL pushes mean to converge with its own median.** Weakly true at best
  (unaffected by the fix, not re-tested).
- **Theory: driven by day-ahead/imbalance price spread.** Modest at best (unaffected by
  the fix, not re-tested; hour-of-day R^2 was 0.29).
- **Solar irradiance timing -- still the dominant driver, though the dual-price fit is
  slightly less clean now**:
  - Single-price bias vs solar_irrad (hourly, vs baseline): **R^2=0.9546** (unchanged --
    single-price forecaster/training untouched by the fix).
  - Dual-price (balanced) bias vs solar_irrad (vs baseline): **R^2=0.9329** (was 0.9575
    pre-fix -- still very strong, slightly weaker fit, consistent with the gentler
    over-forecast-side correction from Section 3).
  - Mechanism unchanged: baseline systematically over-predicts prosumption at peak solar
    hours; DFL sharpens the dip. Direct accuracy check at hour 10 (averaged-then-absolute
    error, matching the original check's convention): baseline 0.226 -> DFL single-price
    0.013 (unchanged, ~17x better) -> DFL dual-price 0.094 (also much better than
    baseline, though less dramatically than single-price at this specific hour).
  - **Major revision**: comparing the two DFL forecasters DIRECTLY against each other
    (not each vs baseline), residual bias-vs-solar **R^2=0.9262** -- NOT the ~0.098 found
    pre-fix. The two forecasters no longer learned "essentially the same" solar
    correction; the retrained dual-price forecaster's correction is real, strongly
    solar-shaped, but **systematically gentler** than single-price's (matches Section 3's
    weaker-restoring-pull finding directly). This is the single biggest qualitative change
    from the earlier version of these findings -- the old "both forecasters converge to
    the same fix" story was itself downstream of the bug's over-strong correction pressure.

## 5. Net imbalance / volume (evaluated on the SINGLE-price pipeline)

- `p_imb = realised - pl_hat` exactly here (single-price always has `D_ch=D_dis=0`).

| forecaster | mean net imbalance (MWh/day) | mean abs imbalance volume (MWh/day) |
|---|---|---|
| baseline | -3.23 | 8.75 |
| DFL single-price (native) | -2.37 | 9.14 (worse than baseline) |
| DFL dual-price (cross-eval) | -2.08 | **8.51** (best of the three) |

- Revised from pre-fix: dual-price-cross's net imbalance is **no longer near-zero**
  (was +0.60; now -2.08, similar order of magnitude to single-price's -2.37, not
  qualitatively different/uniquely well-calibrated as previously reported).
- What survives: dual-price-cross still has the best (lowest) volume of the three, and
  still improves net bias somewhat over single-price-native -- a real but much more
  modest advantage than "near-zero net imbalance" implied before.

### 5a. Imbalance volume is a property of the forecaster, not the settlement pipeline

Checked both cross-evaluation directions (`cross_price_eval.ipynb`) with the same
mean-abs-daily-imbalance metric, now also totalled over the year:

| pipeline | forecaster | mean abs daily imbalance (MWh) | total/year (MWh) |
|---|---|---|---|
| single-price | baseline | 8.753 | 3203.68 |
| single-price | DFL single-price (native) | 9.137 (worse) | 3344.17 |
| single-price | DFL dual-price (cross-eval) | **8.507** (better) | **3113.62** |
| dual-price | baseline | 8.479 | 3103.26 |
| dual-price | DFL dual-price (native) | **8.236** (better) | **3014.37** |
| dual-price | DFL single-price (cross-eval) | 8.898 (worse) | 3256.56 |

The pattern is consistent across all four DFL rows: the **dual-price-trained** forecaster
reduces imbalance volume relative to baseline in both pipelines it's evaluated under
(native dual-price AND cross-evaluated single-price); the **single-price-trained**
forecaster increases it in both (native single-price AND cross-evaluated dual-price). So
this isn't a settlement-mechanism effect at evaluation time -- it's a property of which
training objective produced the forecaster. Dual-price's accuracy-anchored gradient
(Section 3) yields a genuinely better-calibrated forecaster that carries its
imbalance-reducing property wherever it's deployed; single-price's unanchored gradient
yields a forecaster that's worse-calibrated everywhere, regardless of market -- consistent
with, and a more portable version of, Section 4's finding that the two forecasters' solar
corrections are no longer near-identical post-fix.

## 6. Cross-evaluation (deliberately crossing the price-mode firewall)

- `cross_price_eval.ipynb`, recomputed post anomaly-fix (Section 0b) as well as post
  sign-fix; pre-anomaly-fix values in parens.
- **Direction A** (dual-price-trained forecaster -> single-price pipeline): still a
  near-tie, still slightly favouring native. Mean cost: baseline 1746.55, native
  (DFL single-price) 1741.85 (**-0.269%**, was -0.30%, win 57.65%, was 57.7%), cross
  (DFL dual-price) 1744.39 (**-0.124%**, was -0.16%, win vs baseline 51.37%, was 51.4%).
  Head-to-head: native wins 57.1% of days, cross wins 42.9% -- **unchanged** by the
  anomaly fix (both forecasters were affected by the same single anomalous day in this
  pipeline, so the day-by-day head-to-head ordering barely moved). CRPS unaffected: cross
  still 0.267 vs native's 0.291.
- **Direction B** (single-price-trained forecaster -> dual-price pipeline): mean cost:
  baseline 1797.54, native (DFL dual-price) 1795.49 (**-0.114%**, was -0.15%, win 52.73%,
  was 53.0%), cross (DFL single-price) 1797.99 (**+0.025%**, was -0.02% -- **sign flips**
  from a tiny net benefit to a tiny net harm, both close enough to zero that this is a
  negligible-magnitude crossing, not a substantive reversal). Head-to-head: native wins
  61.2%, cross wins 38.8% (was 60.9%/39.1%, essentially unchanged) -- a real but modest
  native advantage, consistent with the post-sign-fix picture.
- **Revised "why", updated**: the anomaly fix left the qualitative cross-evaluation story
  from the sign-fix (Section 0) fully intact -- both directions still show near-parity
  rather than a collapse, native still modestly beats cross in both directions -- it just
  trimmed the magnitudes, most visibly for dual-price's native number in Direction B and
  cross number in Direction A, both of which lost more (in relative terms) to the anomaly
  fix than single-price's numbers did (Section 0b).

## 7. Hour-of-day accuracy (MAE) and bias pattern -- revised post-fix

- The qualitative pattern survives: dual-price is relatively stronger at edge hours
  (night/morning/evening) and had a distinctive midday-hours pattern -- but the midday
  **overshoot softened substantially**, it didn't just persist.
- Own-bias per hour (`realised - mean`, each forecaster's own signed bias):
  - h10: baseline -0.226, DFL single-price +0.013, DFL dual-price **-0.094** (was -uncomputed
    directly pre-fix in this exact convention, but the pattern is now under-correcting
    relative to single-price rather than matching/overshooting it).
  - h12 (the overshoot hour): baseline -0.117, DFL single-price +0.169 (unchanged), DFL
    dual-price **+0.036** (was +0.282 pre-fix -- overshot HARDER than single-price before;
    now sits much closer to zero than single-price's own overshoot).
  - h22 (night): baseline -0.034, DFL single-price -0.178 (unchanged, still a large
    untargeted shift), DFL dual-price -0.068 (was -0.054 pre-fix -- essentially unchanged,
    still close to baseline).
- MAE per hour (`mean(|realised - mean|)`, Jensen-consistent):
  - h12: baseline 0.880, DFL single-price 0.845, DFL dual-price **0.851** -- dual-price
    now BEATS baseline at its own former weak point (was 0.894, marginally worse than
    baseline, pre-fix).
  - h22: baseline 0.096, DFL single-price 0.187, DFL dual-price 0.107 -- dual-price still
    far more conservative/accurate at night than single-price.
  - Overall (24h mean): baseline 0.362, DFL single-price 0.378 (worse overall), DFL
    dual-price **0.351** (best of the three) -- confirms dual-price's accuracy-preserving
    property holds in aggregate, not just at cherry-picked hours.
- Net read, revised: single-price still applies an untargeted, uniform correction (helps
  where baseline is worst, hurts where baseline was fine, matching Section 3's linear
  gradient). Dual-price applies a well-targeted, solar-shaped correction that no longer
  meaningfully overshoots at its hardest hour -- the asymmetric-restoring-pull finding
  from Section 3 (weaker over-forecast correction) shows up here as a gentler, better-
  calibrated midday adjustment rather than the more dramatic overcorrection reported
  pre-fix.

## 8. Core synthesis (revised, post anomaly-fix)

Dual-price's imbalance settlement, correctly implemented, is `pi_up*(p_imb)+ -
pi_down*(-p_imb)+` -- asymmetric, and its TOTAL realized cost (day-ahead + imbalance
together) is still valley-shaped in the forecast mean with a genuine minimum at the true
value, same as before. But the restoring pull is asymmetric in strength: identical to
single-price's on the under-forecast side, WEAKER than the (buggy) pre-fix version on the
over-forecast side. That asymmetry -- not a qualitative "V-shaped vs linear" distinction --
is the real mechanism, and it produces a real but modest set of advantages: dual-price
still preserves (even slightly improves) CRPS where single-price degrades it; still shows
a better-calibrated, lower-volume net imbalance; still transfers reasonably in
cross-evaluation; still applies a more surgical, solar-targeted correction that no longer
overshoots at its own hardest hour. What it no longer supports is the headline claim that
asymmetric settlement makes DFL training dramatically more valuable economically
(-5.75% vs -0.30%, the pre-sign-fix comparison) -- that gap had already essentially closed
post-sign-fix (-0.15% vs -0.30%), and the subsequent anomaly fix (Section 0b) widened it
further still in single-price's favour (-0.114% vs -0.269%), since one anomalous test-year
hour turned out to be propping up dual-price's number more than single-price's. The honest
framing, now doubly corrected: dual-price's structural advantage is real and
mechanistically explained, but small and now measurably smaller than single-price's raw
economic benefit; most of what originally looked like a large, decisive economic case for
asymmetric-settlement DFL training was a settlement-formula bug, and a further slice of the
apparent gap-narrowing between the two modes turned out to be a single anomalous day in the
test-year price data, not the underlying mechanism.

## Open threads / not yet done

- Ablation isolating how much of dual-price's (now much smaller) benefit is the shared
  mean-shift vs the residual asymmetric-calibration difference (Section 4's R^2=0.93
  residual) -- mentioned, not built.
- No seed-averaging anywhere in this analysis -- single training run per corner (and per
  the pure-DFL diagnostic), so point estimates only, no run-to-run variance quantified.
  Given how much the headline numbers moved from one bug fix, this matters more than it
  did before.
- Whether the forecast SPREAD (not just the mean) degrades under pure-DFL training with
  no pinball at all is untested -- the diagnostic in Section 3 ran too few epochs to see
  it either way.
- Q2/Q3 (archetype modulation/transferability) are no longer applicable -- the project has
  moved to balanced-only; those sections were removed from `aggregate_results.ipynb`
  rather than reconciled.
