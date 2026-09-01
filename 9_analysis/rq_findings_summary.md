# Findings organised by research question

Scope: this folder's other documents are organised by investigation history (see
`README.md`'s reading order) -- this one re-organises the same findings by the project's
Aim/Objective/RQ1-3 instead, for direct use when writing results/discussion around that
framing. It doesn't re-derive anything; every claim below points back to the document
that established it. Bullet-form working notes, same standard as the rest of this folder.

**Aim.** To characterise what decision-focused retraining can and cannot teach a
probabilistic prosumption forecaster under explicitly modelled imbalance settlement, and
to test that characterisation on GB market data.

**Objective.** Derive, from the single-stage training cost functions and the established
optimal-offer results for each settlement regime, what bidding behaviour decision-focused
retraining should be able to induce, and what it structurally cannot.

---

## RQ1. Does decision-focused retraining move the forecaster toward the behaviour each
settlement structure rewards, and what does it cost in forecast quality?

**What each settlement structure should reward, derived unconditionally
(`why_dfl_helps.md` Part 1, exact algebra, autograd-verified to ~1e-14):**

- Dual-price's settlement is genuinely nonlinear in the forecast (piecewise-linear,
  kinked at `pl_hat = realised`). Its exact local training gradient is an asymmetric
  pinball loss targeting a price-determined quantile `tau = (pi_up - pi_da) / (pi_up -
  pi_down)`, not degenerate on real data (mean 0.43 train / 0.50 test, spans `[0,1]`).
  Per Cameron et al. (2021), a genuine end-to-end/two-stage gap *can* exist here
  (`cameron_et_al_relation.md`).
- Single-price's settlement is exactly linear. Its local gradient,
  `d(f_dfl)/d(pl_hat_h) = dt*(pi_da_h - pi_imb_h)`, has **no minimum at all** -- a
  structural, checkable-before-training fact, not an empirical one. Per Cameron et al.'s
  own theorem, two-stage (ordinary CRPS training) should already be optimal here; nothing
  in the objective can pull decision-focused training toward accuracy or away from it in
  a principled way.

**Whether training actually gets there, tested properly (`why_dfl_helps.md` Part 3a,
N=5 seed-averaged, matching Beichter et al. 2025's own convention):**

| market | forecaster | pct_change vs. baseline (mean +/- std) | range | win rate |
|---|---|---|---|---|
| single-price | CRPS-only control | -0.109% +/- 0.090 | -0.207% to +0.059% | 55.2% +/- 3.4 |
| single-price | DFL | -0.011% +/- 0.132 | -0.269% to +0.098% | 51.1% +/- 3.6 |
| dual-price | CRPS-only control | -0.020% +/- 0.165 | -0.247% to +0.254% | 45.6% +/- 11.0 |
| dual-price | DFL | +0.060% +/- 0.134 | -0.114% to +0.223% | 42.6% +/- 7.4 |

Neither mean is distinguishable from zero given its own standard deviation; dual-price's
mean is on the *harmful* side of zero. The mechanistic signature built to explain an
earlier, single-seed positive reading -- hour-of-day solar-bias R^2 -- is equally
unstable once seed-averaged (single-price: [0.955, 0.947, 0.001, 0.055, 0.448], mean
0.481 +/- 0.414; dual-price: [0.933, 0.873, 0.912, 0.799, 0.013], mean 0.706 +/- 0.350).
Three checkable reasons this is close to inevitable given the setup, not evidence DFL
"doesn't work" in general, are in `why_dfl_helps.md` Part 4 -- no accuracy-restoring
minimum (single-price specifically), full-parameter fine-tuning of a small model at small
N (contrasted directly with Beichter et al.'s PEFT, tight-std-dev positive result on the
same problem class), and no matched-budget task-loss-free control existing until this
project built one.

**Cost in forecast quality:** single-price DFL degrades CRPS relative to baseline on the
original seed (`8_testing/balanced_single_vs_dual_findings.md`: 0.268 -> 0.291); the
CRPS-only control matches or beats DFL on accuracy more broadly across the campaign
(`why_dfl_helps.md` Part 3a's control rows). Optimising the wrong target is not neutral --
it costs forecast quality without a reliably compensating economic return.

**Answer.** Theoretically, yes -- dual-price has a real, derivable channel toward the
behaviour its settlement rewards; single-price structurally has none. Empirically, once
tested at the same evidentiary standard as the positive result this project is contrasted
against (N=5 seeds, a genuine control), neither mode's forecaster reliably moves toward
that behaviour in an economically meaningful way, and single-price's forecast quality
measurably degrades in the attempt. The theory correctly predicts *which* mode has a
mechanism available and *why* the other doesn't (see RQ2 below for where that structural
difference does show up cleanly) -- it just doesn't survive as a reliable trained-model
effect at this project's scale and configuration.

---

## RQ2. Does the resulting behaviour transfer to the settlement structure the forecaster
was not trained under?

**Method.** `8_testing/cross_price_eval.ipynb` deliberately crosses `eval_raw.py`'s
price-mode firewall: a forecaster's quantile output, trained under one settlement
structure, is run through the *other* structure's dispatch/settlement pipeline via
`evaluate_one_pair` directly. Originally single-seed; extended to N=5 seeds
(`8_testing/cross_eval_seeded.py`, same seeds/convention as RQ1's campaign) once RQ1's
result showed a one-seed transfer claim would be exactly as unreliable as the one-seed
economic claim it inherited from.

**Result (`why_dfl_helps.md` Part 3b):**

| direction | forecaster | pct_change vs. baseline | range | win rate | t-test vs. 0 |
|---|---|---|---|---|---|
| on single-price pipeline | DFL single-price (native) | -0.011% +/- 0.147 | -0.269% to +0.098% | 51.1% | p=0.877 |
| on single-price pipeline | DFL dual-price (**cross-eval**) | -0.022% +/- 0.069 | -0.124% to +0.060% | 50.1% | p=0.512 |
| on dual-price pipeline | DFL dual-price (native) | +0.060% +/- 0.149 | -0.114% to +0.223% | 42.6% | p=0.418 |
| on dual-price pipeline | DFL single-price (**cross-eval**) | **+0.264% +/- 0.140** | **+0.025% to +0.391%** | 34.0% | **p=0.013** |

This is **asymmetric, and one direction is the single cleanest result in the project**:

- **Dual-price-trained -> single-price pipeline**: transfers close to neutrally. Not
  significant, and actually *tighter* (std 0.069) than single-price's own native result
  (std 0.147).
- **Single-price-trained -> dual-price pipeline**: transfers harmfully, **in all 5 of 5
  independently-trained seeds**, every one on the harmful side of zero (+0.025% to
  +0.391%), win rate collapsing from a coin flip to 34.0% (24.9%-41.8% across seeds).
  Unanimous sign across 5 independent seeds is stronger evidence than the (still
  significant, p=0.013) t-test alone -- it is the only headline economic number in this
  project that does not straddle zero at N=5.

**Why, mechanistically (not fitted after the fact -- predicted in advance by RQ1's
structural half):** single-price training has no accuracy-restoring minimum (RQ1 above;
`why_dfl_helps.md` Part 1.2/4.1) -- whatever state that leaves the forecaster in,
deploying it under dual-price's genuinely convex, accuracy-seeking settlement (V-shaped,
minimised exactly at the realised value) has nowhere good for that unconstrained drift to
land, and it consistently doesn't. Dual-price training has a real (if seed-unstable, per
RQ1) channel toward the price-implied quantile -- deploying that under single-price's
settlement, which is *structurally indifferent* to forecast accuracy (linear, no
minimum), has no mechanism to punish or reward whatever calibration made it across, and
it doesn't.

**Answer.** No and yes, depending on direction, and the asymmetry itself is the finding:
behaviour learned under a settlement structure with a genuine accuracy channel (dual-price)
transfers close to neutrally into a settlement structure without one (single-price).
Behaviour learned under a settlement structure with *no* accuracy channel (single-price)
transfers *harmfully and consistently* into a settlement structure that has one
(dual-price). A forecaster is not penalised for lacking an accuracy-focused training
signal until it meets a market structure that actually prices accuracy -- at which point
the absence of that signal costs money reliably, not occasionally.

---

## RQ3. Does behaviour learned through the simplified single-stage training formulation
survive deployment in the recourse-aware dispatch formulation?

**No new comparison was built for this question** (by design/instruction) -- what
follows uses only evidence RQ1/RQ2 already produced, and is deliberately weaker as a
result.

**What the existing evidence can say.** Every number reported under RQ1 and RQ2 is
already a `setup_full_robust`-evaluated outcome of a `setup_1stage`-trained model --
`eval_raw.py` uses the recourse-aware, LP-dual-robustified full-robust formulation
exclusively for every reported evaluation, regardless of which formulation trained the
forecaster (`setup_1stage` has no `D_ch`/`D_dis` recourse at all; only `setup_full_robust`
does). In that weak, definitional sense, RQ3 is tested continuously throughout this
project: whatever 1-stage training does to the forecaster is always subsequently exposed
to, and measured under, deployment-time recourse.

**What it cannot say.** This is not an isolated test of RQ3, and shouldn't be read as
one. RQ2's transfer axis (price-mode: single-price vs. dual-price) is orthogonal to
RQ3's axis (training formulation vs. deployment formulation) -- both RQ1's and RQ2's
numbers are full-robust-deployment measurements throughout; neither includes a paired
number showing what the *same* trained model would have measured under its own 1-stage
training formulation on the same test days. Without that pairing, "the trained behaviour
X was/wasn't observed under full-robust deployment" cannot be separated from "X would
have looked the same/different/absent under the training formulation itself" -- i.e.
this project cannot currently distinguish *a behaviour failing to survive deployment*
from *a behaviour that was never reliably induced by training in the first place* (which
RQ1's null result shows is already the dominant explanation for most of what was tested).
A dedicated surrogate-vs-deployment paired comparison (same forecaster, same test days,
`setup_1stage`'s own decision variables/objective vs. `setup_full_robust`'s) would be
needed to isolate this cleanly, and none exists in this project.

**Answer.** Not established either way, and this document does not claim otherwise.
Every result in this project already reflects full-robust deployment (so nothing here
was measured "only in the training surrogate and never checked against deployment"), but
in the absence of a direct surrogate-vs-deployment pairing, the honest answer is that
this project cannot currently attribute its results to "training induced real behaviour
which then held up under deployment" as opposed to "training rarely induced reliable
behaviour to begin with" (RQ1) -- the two are confounded by construction here. Flagged as
an open methodological gap, not answered by extrapolation from RQ1/RQ2.

---

## Bottom line across all three

RQ1's structural half is unconditionally true and derived independently of any trained
model; its empirical half is a rigorously-tested null once evaluated at a proper
evidentiary standard. RQ2 is where the structural asymmetry RQ1 predicts actually shows
up as a real, seed-consistent, economically meaningful effect -- not a benefit of
decision-focused training, but a cost of training under a settlement structure that
doesn't discipline the forecaster toward accuracy, once that forecaster is asked to
operate somewhere accuracy is priced. RQ3 remains genuinely open: this project's design
cannot currently separate "did the 1-stage-trained behaviour survive deployment" from
"was there reliable 1-stage-trained behaviour to survive" -- a real limitation, stated
here rather than glossed over, consistent with `why_dfl_helps.md` Part 5's checklist
philosophy of naming what wasn't checked rather than implying it was.
