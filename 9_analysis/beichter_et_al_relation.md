# Relating Beichter et al. (2025), "Decision-focused fine-tuning of time series foundation models for dispatchable feeder optimization", to this project

Scope: how Beichter, Friederich, Pinter, Werling, Phipps, Beichter, Neumann, Mikut,
Hagenmeyer & Heidrich's decision-focused fine-tuning (DFF) results relate to this
project's own, now largely negative, DFL-vs-control finding
(`why_dfl_helps.md` Part 3a). Read in full (`9_analysis/Beichter et al. - 2025 -
Decision-focused fine-tuning of time series foundation models for dispatchable feeder
optimization.pdf`). Bullet-form working notes, grown out of a chat discussion, same
standard as the other relation docs in this folder.

## 1. Same problem class, different everything else

Beichter et al. solve the *same dispatchable feeder problem this project does* --
day-ahead dispatch schedule, real-time deviation cost, a battery managing a residential
prosumption series -- citing the same problem-formulation lineage this project's own
6_models/ folder descends from (their [8]/[15]/[47], the Werling et al. dispatchable-
feeder papers). Past that, the two setups diverge on nearly every methodological axis:

| | this project | Beichter et al. |
|---|---|---|
| forecaster | small GRU, trained from scratch, ~tens of thousands of params, **all** trainable in DFL | Moirai foundation model, 91.4M params, only 589,824-626,688 (<0.7%) trainable via LoRA/DoRA |
| DFL mechanism | exact differentiable optimization (`cvxpylayers`, backward through the LP's KKT conditions) | a learned surrogate network (ensemble of 5 small nets) approximating the cost function, backprop through the surrogate |
| settlement/cost shape | single-price: exactly linear, no minimum; dual-price: piecewise-linear, kinked minimum | `C_DS + alpha*C_Imb`, both terms quadratic-plus-linear in deviation -- strongly convex, genuine minimum everywhere, alpha=10 heavily weights imbalance |
| seeds | 1 (until the campaign this document motivated) | 5, mean +/- std reported for every configuration (their Table 5) |
| instances | 1 (single prosumption series) | 200 test buildings (Ausgrid dataset), plus a global-vs-local fine-tuning axis this project's problem has no analogue of |
| control tested | baseline vs matched-budget task-loss-free fine-tune (`crps_only_retrained`, this project's own addition) | PFF-MSE and PFF-MAE: matched-budget, alternative-loss fine-tunes, present in their design from the start |

## 2. Where they independently cross-validate this project's own findings

- **A wrong loss can be actively worse than not fine-tuning at all.** Their MAE-fine-tuned
  Moirai costs *more* than zero-shot Moirai and the naive-48 persistence baseline (Table 5:
  best local PFF-MAE EUR17.53, global PFF-MAE up to EUR20.37, vs zero-shot EUR15.99) --
  the same shape as this project's single-price DFL degrading CRPS relative to baseline
  (`8_testing/balanced_single_vs_dual_findings.md`: 0.268 -> 0.291, single-seed, not
  seed-averaged -- see `why_dfl_helps.md` Part 3a for the seed-averaged economic answer).
  Optimizing the wrong target isn't neutral in
  either project; it actively costs money.
- **Forecast quality and forecast value genuinely diverge**, demonstrated with the same
  logic this project's whole investigation is built on -- their synthetic toy example
  (Section 4.2, Fig. 2/Table 1): two forecasts with *identical* MAE=0.25kW/MSE=0.0625kW^2
  but Daily Total Costs of EUR9.85 (over-estimate) vs EUR23.71 (under-estimate), because an
  empty battery can't absorb the under-estimate's evening deviation. Same Cameron-et-al.-
  style point (`cameron_et_al_relation.md`), independently arrived at.
- **They flag their own DFL instability risk too** -- their surrogate "does not ensure
  convexity, which may result in unstable solutions across different buildings" (Section
  7, Future work). A different mechanism from this project's stopping-point sensitivity
  (`why_dfl_helps.md` Part 5, checklist item 4 -- the sign-reversal between two
  nominally-similar ablation runs), but the same genre of caution: DFL's gradient signal,
  however it's obtained, is not automatically well-behaved.

## 3. Where they diverge sharply -- and why that's the useful part

Beichter et al. report DFF beating matched-budget PFF-MSE by **9.45%** (best DFF, local
DoRA: EUR12.75 +/- 0.04; best PFF-MSE, local: EUR14.08 +/- 0.05) and zero-shot Moirai by
20.26%, averaged over 5 independent runs with tight, non-overlapping standard deviations.
This is exactly the positive-control result this project's own N=5 seed-averaging campaign
(`why_dfl_helps.md` Part 3a, launched directly off this comparison) checked for and did
not find within either mode's own price structure: neither DFL nor the matched-budget
control is reliably better than the other, or than zero, for either price mode -- a
materially different (and less clean) outcome than the single original seed's
"dual-price DFL loses to the control on cost, win rate, CRPS, and imbalance" reading that
motivated running the campaign in the first place. (A related but distinct question --
whether trained behaviour *transfers* to the other price structure -- does turn up a
clean, seed-consistent result; see `why_dfl_helps.md` Part 3b.) Three
concrete, checkable differences explain why this project landed here rather than at
Beichter et al.'s clean result, not just "DFL works there and not here":

1. **PEFT vs. full fine-tuning.** Beichter et al. can move under 0.7% of parameters, in a
   constrained low-rank subspace. This project's DFL fine-tunes 100% of a small GRU's
   weights. This project's largest single finding this session -- that ordinary further
   training, with *zero* economic signal, produces a diurnal shift comparable to or larger
   than DFL's own (`why_dfl_helps.md` Part 3a's mechanism table -- the `crps_only_retrained`
   control's solar-bias R^2 is as seed-unstable as DFL's) -- is a full-fine-tuning-scale
   phenomenon. Under a <0.7%-of-parameters PEFT budget,
   that channel is structurally far more constrained; there's much less room for "generic
   re-training drift" to compete with the decision-focused signal at all.
2. **Cost-function convexity.** Their `C_DS`/`C_Imb` are quadratic-plus-linear -- strongly
   convex, a clean minimum everywhere. This project's single-price mode is exactly linear
   with *no* minimum (proven algebraically, not assumed, `why_dfl_helps.md` Part 1.2);
   dual-price is only piecewise-linear, with a genuine but weak channel back to accuracy
   (Part 1.1). `why_dfl_helps.md` Part 4.1 predicts exactly this ordering -- a task loss
   with no accuracy-restoring minimum gives training no floor against whatever noise rides
   along with its gradient -- and Beichter et al.'s strongly-convex objective sits at the
   safe end of a spectrum this project's own two modes only sample the risky half of.
3. **Baseline strength.** This project's baseline is a bespoke forecaster from an
   80-configuration hyperparameter search, trained on nothing but this one series. Their
   baseline is a general-purpose foundation model, zero-shot or lightly PEFT-adapted to a
   specific building it was never trained on. There is plausibly far more "easy" value
   left on the table in their setup for *any* fine-tuning -- predictive or decision-
   focused -- to capture, consistent with their effect sizes (9-20%) being 30-70x larger in
   relative terms than this project's own pre-control headline numbers (0.1-0.3%).

## 4. What their methodology directly motivated in this project

Their Table 5's 5-seeds-with-tight-std-dev reporting convention is the direct template
for the seed-averaging campaign launched off this comparison (`why_dfl_helps.md` Part 3,
`7_model_training/train_crps_only.py --seed`/`train_dfl_forecasts.py --seed`) -- N=5,
matching theirs exactly, chosen specifically so the two projects' evidentiary standards
are comparable rather than this project asserting a negative result at a lower standard
of proof than the positive result it's being contrasted against.

## Recommendation for the write-up

Cite as a foil, not a validator. Two uses:
1. To state precisely what a robust, seed-averaged, positive DFL result looks like in
   this exact problem class (dispatchable feeder) -- Beichter et al.'s 9.45%, tight std
   dev, is the standard of evidence this project's own point-estimate numbers never met
   before the anomaly/control checks, and the standard the seed-averaging campaign is
   built to meet or fail honestly.
2. To locate *why* the two projects diverge in three specific, falsifiable places (PEFT
   vs. full fine-tuning, cost-function convexity, baseline strength) rather than leaving
   "one paper found DFL works, the other didn't" as an unresolved disagreement -- each of
   the three differences independently predicts the direction this project's result went,
   which is a stronger claim than either result in isolation.
3. This project does have one seed-averaged result built to the same N=5 standard that
   holds up as cleanly as Beichter et al.'s own -- just not the one originally being
   looked for. It isn't a within-mode economic benefit; it's the cross-price transfer
   result (`why_dfl_helps.md` Part 3b, RQ2): all 5 of 5 seeds agree a single-price-trained
   forecaster costs money once deployed under dual-price's settlement. Worth citing
   alongside Beichter et al. as evidence this project's negative within-mode result is a
   genuine finding about this setup, not a symptom of an inability to detect real effects
   at N=5 in the first place -- the same seed-averaging protocol that failed to find a
   within-mode benefit *did* find a real, unanimous effect elsewhere.
