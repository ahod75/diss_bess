# Single-price DFL bias: closed-form gradient mechanism

Scope: a precise, code-verified derivation of *why* single-price DFL training biases the
forecast mean the way it does -- what the training gradient actually is, why `p_da_bat`
drops out of it, whether that's a formulation fact or an implementation artifact, and how
far a "closed form" explanation can honestly be pushed. Grew out of a working discussion
in chat; not written up neat, bullet-form working notes like
`8_testing/balanced_single_vs_dual_findings.md`.

Sections 1-3 (the local gradient, why `p_da_bat` drops out, formulation-vs-implementation)
remain current and unaffected by later work. **Section 4's empirical conclusion about the
*trained* bias pattern is superseded by `bias_spread_solar_mediation.md`** (linked again
at the end of Section 4) -- read that document for the current understanding of what the
deployed model's bias pattern actually reflects.

## 1. The closed-form gradient

For single-price 1-stage training, the entire gradient of `f_dfl` with respect to the
forecaster flows through exactly one tensor, `pl_hat` (the copula-derived mean), and has
an exact closed form:

```
d(f_dfl)/d(pl_hat_h) = dt * (pi_da_h - pi_imb_h)
```

**Both prices here are TRUE, settlement-time prices, not the `_fc` proxy prices fed to
the decision LP** -- worth being explicit about this, since writing both as bare
`pi_da`/`pi_imb` risks looking like it implies cvxpylayers sees true prices at decision
time. It doesn't. `f_dfl`'s only path to `pl_hat` is through `realised_breakdown`, which
is always called with true prices (`_combine_loss(dec, realised, mean, true_prices_t,
...)` in `dfl_train_utils.py`) -- `pl_hat` never enters the decision LP as a Parameter at
all (Section 3), so there is no route by which the *decision* solve's proxy prices could
appear in this gradient. `p_da_bat` is the only decision-LP output that could carry
proxy-price information into `f_dfl`, and Section 2 establishes it has zero gradient
dependence on `pl_hat` regardless. So the closed form is correct exactly as written, but
the two `pi_da_h`/`pi_imb_h` symbols should be read as the true price series throughout,
not the `_fc` proxy used for `dec`.

Verified directly against the running code via autograd (not just derived by hand):
built the actual `setup_1stage` single-price bundle, solved it, ran `realised_breakdown`,
and compared `torch.autograd.grad(f_dfl, pl_hat)` against the predicted closed form on
random test data -- **exact match, zero numerical difference**, including on hours with
`da > imb` and `da < imb` (sign flips both ways, matched exactly either way).

Consequence: whether training pushes `pl_hat` up or down at a given hour, on a given
training day, is *exactly* determined by the sign of `(pi_da - pi_imb)` that hour/day --
not a tendency, not a correlation, the literal derivative.

## 2. Why `p_da_bat` contributes exactly zero -- two distinct reasons

`dec = (p_ch_hat, p_dis_hat, p_da_bat)` comes from a separate cvxpylayers solve whose
**only** Parameter, for single-price 1-stage, is `pi_da` (confirmed by reading
`bundle.params` directly -- `['pi_da']`, nothing else). `pl_hat` was never wired into
that LP's Parameter list at all -- `build_objective_1stage` drops the constant
`pi_da*pl_hat*dt` term from the objective before it ever reaches the solver, since a
term that doesn't depend on the decision variables can't affect the argmin regardless of
whether it's included.

Two separate mechanisms follow from this, one per cost term:

- **`C_da = pi_da*(pl_hat + p_da_bat)*dt`**: `p_da_bat` has zero *gradient* dependence on
  `pl_hat` (it's a function of `pi_da` alone). Differentiating the sum, `p_da_bat`'s
  contribution to the *slope* is zero -- though it still affects `C_da`'s raw *value*
  (it's a real additive term in `bid`), just not its derivative w.r.t. `pl_hat`.
- **`C_imb` (via `p_imb`)**: a different, stronger mechanism -- an exact *algebraic*
  cancellation, not a gradient argument. Since `p_g = realised + p_ch_hat - p_dis_hat`
  and `bid = pl_hat + p_da_bat = pl_hat + (p_ch_hat - p_dis_hat)` (pinned constraint),
  the `p_ch_hat - p_dis_hat` terms cancel identically in `p_imb = p_g - bid = realised -
  pl_hat`. This cancellation would hold even in a hypothetical world where `p_da_bat`
  *did* respond to `pl_hat` -- it's a value-level identity, not a gradient-level one.

## 3. Formulation-level fact, not an implementation artifact

Checked explicitly whether this is a consequence of splitting the computation into a
DPP-compliant cvxpylayers decision solve (proxy `_fc` prices) plus a separate plain-
PyTorch settlement pass (`realised_breakdown`, true prices) -- done for DPP-compliance
and no-leakage reasons -- or whether it's forced by the formulation itself.

It's the formulation. In a hypothetical fully-joint single-solve version (no split, true
prices fed directly into one LP), `p_ch_hat`/`p_dis_hat`/`p_da_bat` would still come out
independent of `pl_hat`, for two formulation-level reasons:

1. `pi_da*pl_hat*dt` is additively separable from the decision variables -- `pl_hat`
   isn't a decision variable, so a term depending only on it can't change which decision
   values minimize the objective, whether or not the term is included in the solve.
2. Single-price's `C_imb` structurally never touches the battery decision variables at
   all, with or without recourse -- it's `pi_imb*(realised - pl_hat)`, a function of
   `pl_hat` and data only, because 1-stage has no recourse mechanism (`D_ch=D_dis=0`)
   for it to act through.

Both are consequences of the pinned bid and single-price's no-recourse settlement
structure -- not of how the code happens to be split. The split is a correct, efficient
*exploitation* of a separability that already exists, not the cause of it.

## 4. The duck-curve / bang-bang hypothesis -- investigated, partially confirmed

Working theory (economically motivated): single-price's linear, no-minimum gradient
means the "market-optimal" position at any hour is a corner solution -- lean fully into
whichever of day-ahead/imbalance is cheaper. If imbalance is systematically cheaper than
day-ahead specifically at high-solar hours (a duck-curve effect), that would explain the
solar-correlated downward bias as *learned price arbitrage*, not just accuracy
correction.

**Constraint worth stating first**: the forecaster has zero price input (`hist_cols`/
`exo_cols` are prosumption, solar, temperature, calendar only). It cannot react to a
specific day's price spread at inference time. Any such effect can only be a static,
hour-of-day-indexed prior baked in by the *training-period* gradient (which does use true
prices for settlement, even though the model never sees them as input).

**First test (linear, misleading)**: raw correlation `R^2(bias, solar) = 0.9546`,
`R^2(bias, spread) = 0.2902`. But solar and spread are themselves collinear
(`R^2(solar, spread) = 0.2692` -- a real duck-curve relationship in the data). Partial
correlation of bias against spread *with solar's linear effect removed from both*:
`R^2 = 0.0304` -- looked at first like the price-spread theory was almost entirely a
confound of the solar relationship.

**Second test (sign-frequency, more revealing)**: a bang-bang gradient cares about the
*sign* of the local pull, not the magnitude the linear test emphasises. Checked
`P(imbalance cheaper than day-ahead)` binned by solar-irradiance decile (pooled across
hours, hour-agnostic):

| solar decile (mean W/m^2) | P(imb cheaper) |
|---|---|
| 0.10 | 0.523 |
| 26.9 | 0.510 |
| 101.0 | 0.506 |
| 214.6 | 0.555 |
| 399.6 | 0.578 |
| **687.1** | **0.608** |

A real, monotonic-ish rise from ~0.51 to ~0.61 across the solar range -- not overwhelming,
but a genuine majority tilt, not noise. Per-hour version (hour-of-day rather than solar
decile) peaks at h14 (0.604), slightly *lagging* the solar peak (h12).

**Why the linear partial-correlation test understated this**: it implicitly treated solar
and price-spread as *competing* covariates ("does spread explain anything beyond solar").
If the true structure is a *mediation* chain instead -- `solar -> P(imbalance cheaper) ->
bang-bang gradient -> bias` -- then controlling for solar removes exactly the channel
through which the price mechanism would operate, making a real, mediated effect look
artificially weak. `2*P(imb cheaper) - 1` (the expected sign of the local gradient) rises
from ~0.02-0.04 at low solar to ~0.22 at high solar -- a much bigger effective swing than
the raw linear test suggested.

**Current honest position at the time this section was written**: the bang-bang/price
mechanism has real, non-trivial, monotonic support and is very likely operating
*alongside* the accuracy-correction mechanism, reinforcing it in the same direction (both
push the forecast down at high-solar hours) -- not the sole explanation, and hard to
cleanly separate from accuracy-correction using observational data alone, since they're
confounded by construction (both duck-curve driven) and both point the same way.
"Over-optimistic" was too strong a dismissal of the original hypothesis; "a real
contributing mechanism, mediated through solar rather than independent of it" is the
accurate characterisation.

**Superseded -- see `bias_spread_solar_mediation.md` for the current position.** That
document runs the missing control (does *baseline*, which never sees price, show the same
bias-vs-spread pattern? -- yes, often more strongly, Section 1 there) and the missing
magnitude test (does the trained model's shift scale with how large a given day's spread
actually is, controlling for solar? -- no, Section 4 there, R^2<0.013 in every case). Net
result: this section's "real contributing mechanism, mediated through solar" framing
undersold how completely solar dominates -- the trained forecast's shift is essentially
*only* a function of solar (R^2~0.95-0.96, stable across both training and test years),
spread-magnitude-blind at inference time, for the structural reason already noted above
(no price input). The bang-bang story is not wrong about the *training-time* gradient
(Section 1's closed form is exact and unaffected), but it does not survive as a
description of what the *deployed* model's bias pattern actually is.

## 5. What's actually closed-form, and what isn't

Section 1's gradient is exact and verified -- but that's the *instantaneous, local*
mechanism (the pull at one training instance), not a closed form for the *final trained
bias* actually measured post-training. Four genuine gaps between the two:

1. **`self_balanced_loss` rescales it dynamically.** The gradient that actually reaches
   `pl_hat` each batch is `beta * dt*(pi_da - pi_imb)`, where `beta = L_base/(L_base +
   f_dfl + eps)` is detached (no gradient through it) but numerically varies batch-to-
   batch as the two loss components' relative scales shift over training. Not a simple
   time-average of the raw quantity.
2. **Adam, not plain SGD.** Per-parameter adaptive learning rates from running gradient-
   moment estimates make the map from "sequence of raw gradients" to "sequence of weight
   updates" nonlinear and history-dependent.
3. **Gradient clipping** (`grad_clip=3.0`) adds a further nonlinearity, especially on
   early/large-magnitude batches.
4. **Weight sharing across hours and quantile levels** (the GRU/dense architecture, plus
   the copula interpolation touching most of the 19 quantile levels for any given
   `pl_hat`). A pull at hour `h` doesn't produce an isolated, hour-local weight update --
   there's cross-hour/cross-quantile entanglement through shared parameters that hasn't
   been quantified.

So: the exact microscopic law (the force at every instant) is proven and verified: the
resulting trajectory is not solved in closed form, and realistically can't be without
linearising the whole multi-epoch, Adam-optimised, weight-shared training process --
which is why the bias pattern is characterised empirically (hour-of-day tables, R^2
figures) throughout this project rather than predicted from first principles.

## Open thread

The theoretically cleanest single-number proxy for "what the training gradient actually
saw, per hour" would be `E_d[dt*(pi_da_h,d - pi_imb_h,d)]` -- the mean signed spread by
hour, computed over the **training-period** (2018) data specifically, not the test period
used for the correlation checks above. Proposed in chat, not yet computed.
