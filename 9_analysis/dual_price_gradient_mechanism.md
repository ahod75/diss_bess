# Dual-price DFL: closed-form gradient mechanism

Scope: the dual-price analogue of `single_price_gradient_mechanism.md` -- same
questions, same standard of verification (autograd-checked against the running code, not
just derived by hand), covering what makes dual-price's mechanism structurally different
from single-price's. Bullet-form working notes, grown out of a chat discussion.

Sections 1-4 are local/instantaneous training-gradient facts and remain current. Section
5's pinball-quantile reformulation is also exact and unaffected -- but see
`bias_spread_solar_mediation.md` Section 5a for the trained-model caveat: the
price-implied quantile `tau` derived there, despite being the exact local target, does
not detectably explain how the *trained* dual-price model's bias pattern differs from
single-price's (partial R^2<0.001 against the residual divergence, both periods). The
mechanism is real at the gradient level; it doesn't survive as a visible signature in
deployed behaviour.

## 1. The decision is identical to single-price's -- verified, not assumed

Since `D_ch=D_dis=0` in 1-stage regardless of mode, and the epigraph (`p_plus`,
`p_minus`, tied to `xi_samples`) shares no constraint with the returned decision, dual-
price's `setup_1stage` solves *literally the same optimisation problem* as single-price's:
`minimize pi_da*p_da_bat*dt + gamma*(||p_ch_hat||^2+||p_dis_hat||^2+||p_da_bat||^2)`
subject to identical physical constraints. Confirmed by solving both for the same
forecaster output/day and comparing:

```
p_ch_hat:  max|single - dual| = 2.83e-09
p_dis_hat: max|single - dual| = 4.20e-09
p_da_bat:  max|single - dual| = 4.16e-09
```

Differences are pure ECOS solver noise. Every difference between single-price and
dual-price training therefore lives entirely in the settlement formula -- there is no
decision-level channel at all.

## 2. The closed-form gradient

```
d(f_dfl)/d(pl_hat_h) = dt * (pi_da_h - pi_up_h)     if pl_hat_h < realised_h   (short)
                     = dt * (pi_da_h - pi_down_h)    if pl_hat_h > realised_h   (long)
                     = 0                             for the whole day, if C_da+C_imb < 0
                       before the clamp (torch.clamp(x, min=0) has zero gradient for x<0
                       -- ordinary autograd behaviour, verified directly, not specific to
                       this problem)
```

**All prices here (`pi_da`, `pi_up`, `pi_down`) are TRUE, settlement-time prices**, not
the `_fc` proxy prices fed to the decision LP (which are `pi_da_fc` and, for dual-price,
`xi_samples`/`pi_imb_up_fc`/`pi_imb_down_fc` -- see `single_price_gradient_mechanism.md`
Section 1 for why writing both without disambiguating them risks implying cvxpylayers
sees true prices at decision time; it doesn't). `f_dfl`'s only path to `pl_hat` is through
`realised_breakdown`, always called with true prices. `p_da_bat` is the only decision-LP
output that could carry proxy-price information forward, and Section 1 already
establishes the decision is identical regardless of which prices fed it (mode-invariant,
and by the same argument as the single-price doc, `pl_hat`-invariant) -- so no proxy price
appears anywhere in this gradient.

Verified via autograd against the real `setup_1stage`/`realised_breakdown` code path on
random test data -- exact match (zero numerical difference) in both regimes (short and
long), across multiple random seeds and both signs of `(da-imb)`.

## 3. Direction vs magnitude -- the key structural contrast with single-price

Easy to accidentally import single-price's intuition here, so worth stating precisely:

- **Single-price**: gradient sign is fixed entirely by the price comparison
  (`pi_da` vs `pi_imb`) and doesn't care where `pl_hat` sits relative to `realised` at
  all. No accuracy anchor -- see `single_price_gradient_mechanism.md`.
- **Dual-price**: gradient sign is fixed entirely by which side of `realised` `pl_hat`
  currently sits on, and *always* points back toward it -- short gives gradient `<=0`
  (since `pi_up >= pi_da` always, by construction of `pi_up=max(da,imb)`), so descent
  increases `pl_hat`; long gives gradient `>=0` (since `pi_down <= pi_da` always), so
  descent decreases `pl_hat`. Both branches point the *same* way, toward accuracy,
  regardless of the actual price numbers that hour. The prices only ever set how
  *strongly* it pulls -- never whether it pulls toward or away from `realised`.

This is the formal version of "dual-price has a genuine accuracy-seeking valley,
single-price doesn't" established earlier in the project, now derived to the level of an
exact per-hour closed form rather than a qualitative V-shape/linear-shape distinction.

## 4. How often does the clamp actually trigger? -- empirically checked, corrects an earlier guess

Computed directly via the real 1-stage/`D_ch=D_dis=0` path (not the full-robust eval
path, which genuinely differs since it has real recourse) over the **training period**
(2018, 365 days), both for the frozen baseline forecaster and the currently-trained
`dfl_1stage_dual-price` checkpoint:

```
baseline forecaster: 0/365 days with raw<0 (0.00%), mean raw=3199.59, min raw=899.92
trained checkpoint:  0/365 days with raw<0 (0.00%), mean raw=3192.07, min raw=921.24
```

Essentially never triggers, at the scale this problem actually operates at. This
corrects an earlier speculative claim made in chat (before this was checked) that the
corrected settlement formula would make the clamp trigger *more* often than the
historical ~0.55% baseline figure, since `C_imb` can now go negative. It doesn't, in
practice: `C_da`'s typical scale (~3200, driven by `pi_da*pl_hat*dt` summed over 24
hours) dwarfs `C_imb`'s realistic credit magnitude, which is bounded by how large
`p_imb = realised - pl_hat` can plausibly get for a reasonably-calibrated forecaster (a
few MW at most per hour) times price. Full-robust eval time shows a small nonzero rate
(0.27%, both baseline and DFL, from `aggregate_results.ipynb`'s Q1 data) -- plausibly
because genuine recourse (`D_ch`/`D_dis` != 0 there) can amplify `p_imb` beyond what a
no-recourse forecast deviation alone produces, something 1-stage structurally can't do.

## 5. Exact reformulation: dual-price cost is a scaled, price-determined-quantile pinball loss

Working from the two-branch closed form, the total realised cost as a function of
`pl_hat` (dropping the additive constant that doesn't depend on it) is:

```
Total(p_imb) - const = dt * [(pi_up - pi_da)*(p_imb)+ + (pi_da - pi_down)*(-p_imb)+]
```

where `p_imb = realised - pl_hat`. This has exactly the structural form of a pinball
loss, `tau*(y-q)+ + (1-tau)*(q-y)+`, scaled and re-parametrised:

```
Total(pl_hat) = dt * (pi_up - pi_down) * pinball_tau(realised, pl_hat) + const
tau = (pi_up - pi_da) / (pi_up - pi_down)
```

Verified numerically to floating-point precision (max diff ~1e-14 after fixing an
initial sign slip in the first check -- caught by comparing the *difference* between the
direct and reformulated expressions against the predicted constant, rather than trusting
the algebra blind). `tau` is guaranteed in `[0,1]` since `pi_down <= pi_da <= pi_up`
always; checked empirically it spans the full range on random test data.

**Consequence**: dual-price training doesn't pull `pl_hat` toward the mean or the
median in general -- it pulls it toward whichever quantile `tau` happens to be that
hour, and `tau=0.5` only when `pi_da` sits exactly halfway between `pi_down` and
`pi_up`. Since `pi_up`/`pi_down` are constructed asymmetrically around `pi_da`
(`max(da,imb)`/`min(da,imb)`, not a symmetric band), `tau` will typically drift away from
0.5.

This gives a mechanistic explanation for a previously-unresolved negative result in
`8_testing/balanced_single_vs_dual_findings.md` Section 4: the "DFL pushes mean to
converge with its own median" theory was tested and found to hold only weakly for
single-price and not at all for dual-price ("gap stays flat or grows"). This result shows
why -- dual-price was never being pulled toward the median in the first place; it's being
pulled toward a price-determined quantile that generally isn't 0.5.

## Open threads

- **Resolved**: `tau`'s distribution on real `da`/`imb_up`/`imb_down` data (computed in
  `bias_spread_solar_mediation.md` Section 5a) -- mean 0.43 (train) / 0.50 (test), std
  ~0.50, spans the full `[0,1]` range, not degenerate. But it does *not* correlate with
  the trained model's hour-of-day bias pattern (partial R^2<0.001 controlling for the
  shared single/dual relationship) -- so `tau` is a real, non-trivial quantity, just not
  one whose day-to-day variation shows up in what the trained model actually does.
- Same open thread as the single-price doc: this is the exact *local* gradient, not a
  closed form for the *trained* outcome -- `self_balanced_loss`'s dynamic reweighting,
  Adam, gradient clipping, and weight sharing across hours/quantile levels all still sit
  between this and the actually-observed post-training bias.
