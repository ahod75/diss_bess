# Balanced-battery single-price vs dual-price DFL: consolidated findings

Scope: `baseline` vs the two DFL 1stage checkpoints (`dfl_1stage_single-price`,
`dfl_1stage_dual-price`), balanced/"standard battery" archetype (C_ch=C_dis=2MW,
B_max=4MWh) only. Full sealed test year, 366 days (2019-07-01..2020-06-30). Bullet-form
working notes, not written up -- source material for the eventual write-up.

## 1. Headline economic results (Q1, `aggregate_results.ipynb`)

- Single-price: baseline mean cost 1747.25 -> DFL 1741.95 (-0.30%, win rate 57.7%)
  - CRPS *worsens*: 0.268 -> 0.291
- Dual-price: baseline mean cost 2089.96 -> DFL 1969.85 (-5.75%, win rate 84.4%)
  - CRPS stays flat: 0.268 -> 0.267
- Dual-price's benefit is ~19x larger than single-price's, while *improving* forecast
  accuracy rather than trading it away.

## 2. Single-price is structurally forecaster-invariant at the decision level

- Proven: `p_ch_hat`, `p_dis_hat`, `p_da_rel` are IDENTICAL for baseline vs DFL at eval
  time (confirmed empirically -- `p_ch_r`/`p_dis_r` bit-for-bit equal despite very
  different `pl_hat`).
- Cause: single-price's decision objective is `C_da = pi_da @ p_da_rel * dt` only --
  `C_imb = 0.0` literally (not even a cvxpy expression) in the decision-time objective,
  since `E[pi_imb . p_imb] = 0` given the pinned bid. `D_ch = D_dis = 0` always
  (confirmed: no cost-relevant reason for the solver to pick anything else at gamma=0).
  `pl_hat` never enters the decision LP as a Parameter at all.
- Consequence: entire single-price benefit reduces to ONE lever -- the forecast MEAN,
  weighted by local price spread. Exact identity (verified to floating-point precision):
  `baseline_cost - DFL_cost = sum_h (pi_da_h - pi_imb_h) * (pl_hat_base_h - pl_hat_dfl_h) * dt`

## 3. Gradient-level mechanism (training), and a correction made mid-session

- Full differentiable chain: `theta -> q_norm (NN, softplus-cumsum monotone
  construction) -> q_phys (affine denorm, order-preserving) -> copula interpolation
  (pl_hat, xi; frozen bracket/weight plan, differentiable in quantiles) ->
  cvxpylayers LP -> realised_breakdown (settlement) -> self_balanced_loss`.
- `self_balanced_loss(L_base, f_dfl) = harmonic_mean(L_base, f_dfl)` exactly (algebraic
  identity: `alpha*L_base == beta*f_dfl` always). Because harmonic mean is dominated by
  the smaller term, whichever of `L_base`/`f_dfl` is currently LARGER gets LESS weight on
  its own gradient contribution that batch -- a scale-equalizer, not a "focus on the
  worse metric" mechanism.
- **Correction (initially got this wrong)**: originally claimed dual-price training
  shapes the model's predicted *spread* via cvxpylayers differentiating through a real
  `D_ch`/`D_dis` hedging policy ("Route B"). FALSE for 1stage training: `setup_1stage`
  has no `D_ch`/`D_dis` variables at all (`R=None` in the epigraph), the epigraph
  (`p_plus`, `p_minus`, `xi_samples`) is fully DECOUPLED from the returned
  `(p_ch_hat, p_dis_hat, p_da_rel)` (no shared constraints), and `p_plus`/`p_minus`
  aren't even in the layer's `variables` list. So `xi_samples` has **zero** gradient
  effect on `f_dfl` during 1stage training, in both modes. That real hedging mechanism
  only exists at EVAL time (`setup_full_robust`, which does have real `D_ch`/`D_dis`).
- pl_hat is a genuine free variable in the LP too -- and since p_da_rel doesn't depend on
  price_mode or pl_hat, `dec` is **identical** between single/dual training for the same
  forecaster/day. The only real difference between modes is the SETTLEMENT gradient
  shape on `pl_hat`:
  - Single-price: `C_imb` LINEAR in `pl_hat` (`d(C_imb)/d(pl_hat) = -pi_imb`, constant
    sign, no minimum -- a directionless price-driven nudge, unrelated to accuracy).
  - Dual-price: `C_imb` **V-shaped** in `pl_hat` (asymmetric-pinball-loss-shaped,
    minimised exactly at `pl_hat = realised`; gradient always points TOWARD realised
    from either side). Structurally self-correcting/accuracy-seeking, on top of L_base.
  - `C_da`'s pl_hat-gradient is linear and mode-invariant in both cases -- the V-shape is
    entirely dual-price's `C_imb` contribution.
- This is the single mechanistic root cause behind almost every other finding below.

## 4. What actually drives the mean-shift (theories tested, in order)

- **Theory: DFL pushes mean to converge with its own median.** Weakly true at best.
  - Single-price: mean-median |gap| 0.0369 (baseline) -> 0.0323 (DFL), ~12% smaller.
  - Dual-price (all 3 archetypes): gap stays flat or *grows* (e.g. balanced 0.0369 ->
    0.0423, +15%). Not a general mechanism.
- **Theory: driven by day-ahead/imbalance price spread `(pi_da - pi_imb)`.**
  - Day-level (pointwise): R^2 ~ 0.002 -- essentially zero (forecaster has no price
    input at all, so it structurally can't react to a specific day's price).
  - Hour-of-day level (systematic, averaged over all 366 days): R^2 = 0.29 -- real but
    modest, sign matches theory.
- **Solar irradiance timing -- the dominant, well-supported driver.**
  - Single-price bias vs solar_irrad (hourly): r=-0.977, **R^2=0.9546**,
    slope=-0.000826 MW per W/m^2.
  - Dual-price (balanced) bias vs solar_irrad: r=-0.978, **R^2=0.9575**,
    slope=-0.000883 MW per W/m^2 (~7% steeper than single-price).
  - Dual-price specialists range R^2 0.78 (short_sharp) to 0.96 (balanced/long_slow).
  - Mechanism: baseline systematically OVER-predicts prosumption at peak solar hours
    (doesn't dip enough for the real generation-driven trough); DFL training sharpens
    that dip beyond what pure CRPS/pinball training converged to. Confirmed via direct
    accuracy check: baseline |error| at hour 10 = 0.226, DFL single-price = 0.013 (17x
    better specifically at that hour).
  - When the two DFL forecasters are compared DIRECTLY against each other (not each vs
    baseline), residual bias-vs-solar R^2 drops to **0.098** -- confirms both learned
    essentially the SAME solar correction; what's left over (~-0.12 MW, fairly constant
    across hours) is a small, mostly solar-independent residual.

## 5. Net imbalance / volume (all evaluated on the SINGLE-price pipeline)

- `p_imb = realised - pl_hat` exactly here (single-price always has D_ch=D_dis=0).

| forecaster | mean net imbalance (MWh/day) | mean abs imbalance volume (MWh/day) |
|---|---|---|
| baseline | -3.23 | 8.75 |
| DFL single-price (native) | -2.37 | **9.14** (worse than baseline) |
| DFL dual-price (cross-eval) | **+0.60** (~zero) | **8.43** (best of the three) |

- Single-price-native improves net bias somewhat but its hour-to-hour VOLUME gets worse
  than baseline's -- consistent with the linear, non-accuracy-seeking gradient (can shift
  net bias without improving genuine tracking).
- Dual-price-cross improves BOTH net bias and volume together, even though it was never
  trained on this pipeline -- consistent with the V-shaped gradient producing a mean
  that's genuinely better-calibrated, not just price-nudged.

## 6. Cross-evaluation (deliberately crossing the price-mode firewall)

- Firewall is enforced only by `FORECASTERS` dict shape in `eval_raw.py`, not a runtime
  check -- safe to cross directly via `evaluate_one_pair`. New notebook:
  `cross_price_eval.ipynb`.
- **Direction A** (dual-price-trained forecaster -> single-price pipeline): near-tie.
  Mean cost 1741.14 (cross) vs 1741.95 (native) -- cross slightly ahead on aggregate.
  Day-to-day paired: cross wins 48.9%, native wins 51.1% (right-skewed -- native wins
  more often, cross wins bigger when it wins). CRPS stays at 0.267 (cross) vs 0.291
  (native) -- cross achieves this WITHOUT sacrificing accuracy.
- **Direction B** (single-price-trained forecaster -> dual-price pipeline): collapse.
  Native dual-price: -5.75% vs baseline, 84.4% win rate. Cross (single-price forecaster
  on dual-price pipeline): only -1.23% vs baseline, 49.7% win rate (a coin flip).
  Head-to-head: cross loses to native dual-price forecaster on 92.1% of days, mean gap
  +94.5 (dual-price cost units).
- **Why the asymmetry**: both forecasters learned nearly the same solar correction
  (Section 4's residual R^2=0.098), so single-price's mean transfers reasonably to
  dual-price's settlement for that shared component (explains A's near-tie) -- but
  dual-price's mean is ADDITIONALLY calibrated specifically for asymmetric settlement
  (via the V-shaped gradient, section 3), which single-price's mean never saw any
  version of. Single-price's mean is simply the wrong shape for what asymmetric pricing
  punishes, hence B's collapse.

## 7. Hour-of-day accuracy (MAE) and bias pattern

- User's own observation, confirmed numerically: dual-price is MORE accurate than
  single-price (and often than baseline) at "edge" hours (night/morning/evening), LESS
  accurate specifically at midday peak-solar hours.
- Bias-per-hour breakdown explains why:
  - Single-price's bias shift vs baseline is roughly FLAT/uniform across most hours
    (e.g. -0.16 to -0.21 through hours 19-23) -- matches the linear, price-sign-only
    gradient (blind to whether that hour actually needs correcting). Pushes bias further
    negative at NIGHT where baseline was already fine -> hurts night accuracy
    (baseline |err| h22 = 0.096, single = 0.187).
  - Dual-price's bias shift is much more SOLAR-SHAPED (small at night, close to
    baseline's own bias there: e.g. h22 baseline bias -0.034 vs dual -0.054) but
    OVERSHOOTS specifically at midday (h12: baseline bias -0.117, single +0.169, dual
    +0.282) -- large enough that dual-price's MAE at h12 (0.894) is marginally worse
    than baseline's own (0.880), while single's smaller overshoot pairs with a genuine
    MAE improvement there (0.845).
- Net read: single-price applies an untargeted, uniform correction (helps where baseline
  is worst, hurts where baseline was fine); dual-price applies a well-targeted,
  solar-shaped correction that overshoots specifically at the hardest hours but leaves
  everywhere else alone or improved.

## 8. Core synthesis (one paragraph, for reference)

Dual-price's imbalance settlement (`pi_up*(p_imb)^+ + pi_down*(-p_imb)^+`, both prices
clamped nonneg at decision AND settlement time) is provably always >=0 and, as a function
of the forecast mean holding realised fixed, is V-shaped -- literally an
asymmetrically-reweighted pinball loss, minimised exactly at the true value. Single-price's
settlement (`pi_imb * p_imb`, unclamped, can go negative) is linear in the mean -- no
minimum, no accuracy-seeking property, just a directionless nudge set by price sign. That
one structural difference (asymmetric+nonneg vs signed+unclamped) is the root cause behind
essentially everything else observed: why dual-price preserves CRPS while single-price
degrades it: why dual-price's net imbalance and volume both improve together while
single-price's net improves at the cost of volume; why cross-evaluation is asymmetric
(shared solar correction transfers one way, dual-price's asymmetric-settlement-specific
mean calibration doesn't transfer the other way); and why dual-price's hour-of-day error
pattern is well-targeted-but-overshooting rather than single-price's untargeted-but-safer
shift.

## Open threads / not yet done

- Ablation isolating how much of dual-price's benefit is the shared mean-shift vs
  something else (mentioned once, not built).
- No seed-averaging anywhere in this analysis -- single training run per corner, so
  point estimates only, no run-to-run variance quantified.
- Q2/Q3 (archetype modulation/transferability, `aggregate_results.ipynb`) still assume
  the full 3-archetype grid -- not reconciled with the balanced-only narrowing decision.
