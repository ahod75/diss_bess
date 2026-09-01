# 9_analysis -- reading guide

Working notes on *why* DFL training affects the forecaster the way it does, grown out of
chat investigations into the closed-form training gradients and their empirical
consequences. All bullet-form, code-verified (autograd checks, direct correlation/
regression on real data), not polished prose -- written for feeding into the
dissertation's methodology/discussion chapters, not as chapter drafts themselves.

**Start with `why_dfl_helps.md`** -- the capstone synthesis. Its answer, as of the N=5
seed-averaging campaign, is a rigorously-explained **null result for within-mode
benefit, plus one result that isn't null**: no positive claim about DFL's economic or
mechanistic benefit within the mode it was trained under survives seed-averaging (its
Part 3a); what survives is the exact, unconditional local-gradient mathematics (its Part
1), a three-factor mechanistic explanation for why the within-mode null result was
near-inevitable here (its Part 4), a direct contrast with a same-problem-class positive
result (Beichter et al., doc 5), a reusable five-point checklist built from this
project's own near-miss false positives (its Part 5) -- and a genuinely seed-consistent
*negative* transfer result (its Part 3b, RQ2): a forecaster trained under single-price
(no accuracy anchor) reliably costs money when deployed under dual-price's settlement
(which has one), unanimous across all 5 seeds, and predicted in advance by Part 1.2/4.1
rather than discovered after the fact. `9_analysis/rq_findings_summary.md` organises all
of this explicitly by research question (RQ1-3) rather than by document. The other five
documents below are `why_dfl_helps.md`'s supporting derivations and verifications; read
them for the full derivation or numeric verification behind a specific claim, not to
reconstruct the answer yourself.

Documents build on each other; some conclusions in the earlier documents were later
revised by the later ones. Rather than re-deriving that history from each file
independently, read in this order and take the note at the top of each file as the
current status:

0. **`why_dfl_helps.md`** -- capstone synthesis. Read this first.
1. **`single_price_gradient_mechanism.md`** -- the exact local training gradient for
   single-price (Sections 1-3, still current), plus an early empirical investigation into
   the *trained* bias pattern (Section 4, **superseded** -- see its own top-of-file note).
2. **`dual_price_gradient_mechanism.md`** -- the dual-price analogue: local gradient,
   clamp-frequency check, and the exact pinball/quantile-`tau` reformulation (all still
   current at the gradient level; see its top-of-file note for a trained-behaviour
   caveat).
3. **`bias_spread_solar_mediation.md`** -- the single-seed mechanistic investigation
   (Part 2 of doc 0) that motivated the seed-averaging campaign: the causal ablation
   (Section 7, later superseded in direction by doc 0's proper control) and the baseline
   calibration-bias check (Section 8: baseline's own forecaster, not the copula, carries a
   real solar-correlated over-prediction bias -- this specific finding is single-checkpoint
   scoped, since baseline was deliberately excluded from seed-averaging, and stands on its
   own terms). Supersedes Section 4 of doc 1; qualifies (without invalidating) the
   `tau`-tracking framing in doc 2 and the dual-price discussion in doc 4.
4. **`cameron_et_al_relation.md`** -- relates docs 1-2's gradient mechanisms to Cameron
   et al. (2021)'s theory of when end-to-end learning beats two-stage learning. Sections 2
   and 3 carry caveats added after doc 3 existed, distinguishing the theory's correct
   predictions about the local gradient/target-statistic gap from what the trained models
   are actually observed to do. The primary source behind doc 0's Part 1.
5. **`beichter_et_al_relation.md`** -- relates this project's now-largely-negative
   within-mode DFL-vs-control result (doc 0 Part 3a) to Beichter et al. (2025)'s
   same-problem-class (dispatchable feeder), 5-seed-averaged, decisively *positive* DFL
   result. A foil, not a validator: locates the divergence in three specific differences
   (PEFT vs. full fine-tuning, cost-function convexity, baseline strength) and directly
   motivated the seed-averaging campaign (`7_model_training/train_crps_only.py`/
   `train_dfl_forecasts.py --seed`) doc 0 Part 3 now reports.
6. **`rq_findings_summary.md`** -- not a new investigation; re-organises everything
   above by the project's Aim/Objective/RQ1-3 instead of by document. RQ1 (within-mode
   behaviour + forecast-quality cost) draws on docs 0's Parts 1/3a/4. RQ2 (cross-price
   transfer) draws on doc 0's Part 3b, the one seed-consistent economic result in the
   project, built directly in response to this question
   (`8_testing/cross_price_eval.ipynb`/`cross_eval_seeded.py`). RQ3 (survival under
   recourse-aware deployment) is answered only in the weak sense the existing evidence
   supports, with the gap this leaves stated explicitly rather than papered over -- no
   new comparison was built for it.

The source PDFs (`Cameron et al. - 2021 - The Perils of Learning Before Optimizing.pdf`,
`Beichter et al. - 2025 - Decision-focused fine-tuning of time series foundation models
for dispatchable feeder optimization.pdf`) are referenced throughout docs 4 and 5
respectively.

## What stays true regardless of revision

The exact closed-form local gradients (doc 1 Sections 1-3, doc 2 Sections 1-2/4-5;
doc 0 Part 1) are algebraic facts about training-time dynamics, verified directly against
the running code via autograd -- none of it an empirical claim about a trained model, all
of it unconditional on what any specific run produced. Nothing in this folder's empirical
findings challenges them; the entire rest of the investigation is about the gap between
"what the instantaneous gradient does" and "what the fully-trained, deployed model's
behaviour looks like across seeds" -- a real and, it turned out, decisive gap.

## The null result, and why it's the finding rather than the failure

Doc 3's single-seed investigation, and doc 0's earlier syntheses of it, went through
several readings of what the trained models' diurnal shift *is*, each correcting the
last: price-tracking (confounded by a baseline-vs-baseline control), a stable
solar-conditioned policy (an informal single-run ablation produced a comparably strong
shift with zero economic signal), a calibration-bias correction distinguishable from that
ablation by direction (a *properly* early-stopped version of the same ablation reversed
that direction entirely). The load-bearing result is what a full N=5 seed-averaging
campaign found once built (doc 0 Part 3a): the within-mode economic effect (single-price
-0.011% +/-0.132, dual-price +0.060% +/-0.134 -- mean wrong-signed) and the mechanism used
to explain it (solar-bias R^2 swinging from 0.001 to 0.955 across seeds for single-price)
both fail to reproduce. Every earlier reading in this paragraph was real, on the
checkpoint it was computed from, and none of them describe a property of the method.

**This is presented as the project's finding, not a limitation to work around.** Doc 0
Part 4 gives three specific, checkable reasons it was near-inevitable (no accuracy-
restoring minimum in single-price's objective; full-parameter fine-tuning of a small model
on 365 training days, contrasted directly against Beichter et al.'s parameter-efficient,
seed-robust positive result on the same problem class; no matched-budget control until
this campaign built one). Doc 0 Part 5 turns the project's own near-miss false positives
-- a sign bug that manufactured an apparent -5.75% benefit, a single anomalous price hour
propping up 174% of a headline number, an ablation whose sign flipped from one epoch's
difference in stopping point -- into a five-point checklist for evaluating DFL claims
generally. That checklist, and the mechanistic explanation, are the reusable
contributions; the within-mode null economic result is what they were built to explain,
not independently the point.

**One result is not null, though, and it's the one Part 4.1's mechanism predicted before
it was checked.** Doc 0 Part 3b (RQ2, `8_testing/cross_eval_seeded.py`) cross-evaluated
every seed's forecaster under the settlement structure it *wasn't* trained on. A
dual-price-trained forecaster transfers to the single-price pipeline close to neutrally
(-0.022% +/-0.069, not significant). A single-price-trained forecaster transfers to the
dual-price pipeline harmfully in **all 5 of 5 seeds** (+0.264% +/-0.140, range +0.025% to
+0.391%, win rate collapsing to 34.0%, one-sample t-test p=0.013) -- the only headline
economic number in this folder that doesn't straddle zero. The asymmetry matches Part
1.2/4.1 exactly: the mode with no accuracy-restoring minimum at training time is the one
whose forecast reliably costs money once it meets a settlement structure that actually
penalises miscalibration.
