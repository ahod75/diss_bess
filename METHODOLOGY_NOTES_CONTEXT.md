# Methodology notes — for the write-up conversation

Prepared for a separate chat instance drafting the dissertation's Methodology chapter,
alongside separately-provided notes on the Introduction/Literature Review. Comprehensive,
code-verified notes on how the data pipeline, forecaster, scenario generator, dispatch
optimisation formulations, DFL training procedure, and evaluation protocol actually work
— bullet form, source material for prose, not prose itself.

Scope: describes the **live/final pipeline** — the two architectures actually used
(1-stage for training, full-robust for evaluation), single-price and dual-price market
modes, and the three battery archetypes. Retired components (point-robust, the k∈{0,1}
weighting, dispatchability mode) are covered only briefly, as design-history context,
since they were abandoned before producing any reported results. (Note:
`WRITEUP_FRAMING_CONTEXT.md` in this same directory describes an *earlier* state of the
project — a pl_hat-gaming exploit and the old point-robust/k∈{0,1} design — since fixed
and retired respectively. These notes describe the current, post-fix pipeline.)

Everything below was verified directly against the source files during this session
(three parallel code-exploration passes + direct re-reads/greps/checkpoint loads), not
recalled from memory. Where something could not be fully confirmed, it's flagged
explicitly.

---

## 1. Data pipeline (`1_data/`, `2_eda/`, `3_data_processing/`)

- **Source**: WPD (Western Power Distribution) open-data competition dataset ("set4") —
  `demand_{train,test}_set4.csv`, `pv_{train,test}_set4.csv`, `weather_{train,test}_set4.csv`
  under `1_data/raw/wpd_comp_data_2021/`. Demand, solar PV generation, and weather
  features, resampled from source resolution to **hourly**, merged on a shared UTC
  datetime index (`3_data_processing/clean_transform.ipynb`).
- **Target variable**: `prosumption = demand_MW − pv_power_mw` (net load: positive =
  net consumption, can go negative during high solar output) — this is the quantity the
  forecaster predicts and the dispatch problem schedules against.
- **Anomaly correction** (documented + visually justified in the notebook, 3 fixes):
  1. 2018-04-23→05-06 demand replaced by the same-length preceding week (source anomaly:
     2018-05-07→05-20).
  2. 2018-10-28→11-02 demand replaced by the preceding week (anomaly: 2018-11-03→11-08).
  3. Demand clipped during 2020-03-16→04-22 (the COVID demand-collapse period — this
     matches, and explains, the "~0.1-1.4% of hours, clustered in Apr-Jul 2020" negative-price
     clipping note found independently in `dispatch_shared.py`'s price-column docstring —
     worth cross-referencing in the write-up as the same real-world event surfacing in two
     independent parts of the pipeline).
- **Feature selection**: correlation matrix of all candidate weather/PV features against
  `prosumption`; kept `solar_irrad` (from `solar_location3`), `panel_temp` (from
  `panel_temp_C`), `ambient_temp` (from `temp_location3`) — renamed from their raw
  station-specific names.
- **Final `hist_cols`** (fed as historical/lookback features to the forecaster):
  `prosumption, solar_irrad, panel_temp, ambient_temp` (4 features).
- **Final `exo_cols`** (known-in-advance / calendar features, fed for both history and
  future horizon): `hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos, is_weekend,
  solar_irrad, ambient_temp` (9 features) — cyclical encodings of hour/day-of-week/
  day-of-year, weekend flag, plus solar irradiance and ambient temperature treated as
  "known future" (i.e. a weather forecast proxy, not lagged).
- Imputation/gap-handling via `reindex_and_impute` (`4_forecasting/forecasting.py`),
  called with `warn_gap=6` in the training pipeline.

## 2. Price data and the decision-time forecast/proxy columns (`2_eda/power_price_eda.ipynb`)

- Real GB market prices: day-ahead (`da`) and imbalance (`imb`, single-price convention;
  `imb_up`/`imb_down`, dual-price convention) — Elexon-sourced settlement prices, merged
  onto the same hourly datetime index.
- **Why proxy columns exist at all**: the dissertation does not forecast `imb` (would need
  its own pre-training history, risking leakage into the DFL train/val/test split — "more
  work than already done", per the notebook's own framing). Instead, decision-time price
  *uncertainty* is represented via a **causally-valid stochastic proxy**, so the dispatch
  layer always has a legitimate (non-negative, non-leaking) price input to condition its
  decision on, while true prices are reserved for settlement.
- **Construction of `imb_fc`** (`2_eda/power_price_eda.ipynb`, cells 20–37):
  - `LEAD_HOURS=9` (decisions issued 9h before the delivery window starts), `WINDOW_HOURS=24`,
    `ROLL_HOURS=24` (trailing volatility window — chosen because the autocorrelation of
    `|da−imb|` carries real signal to ~24h, with a diurnal echo at the 24h lag, then
    flattens by ~48–72h).
  - `sigma_proxy`: 24h trailing rolling std of `|da−imb|`, shifted forward by
    `24−LEAD_HOURS=15h` and masked to the issue-hour (gate) value, then forward-filled for
    the next 23h — ensures the sigma used for any hour in a delivery window is one that
    was genuinely knowable at the 9am issue time, never using information from within the
    window itself (explicit no-leakage design).
  - Noise model: a **Student-t distribution** was fit to the historical `da−imb` spread
    (`scipy.stats.t.fit`) and preferred over Gaussian specifically because it fits the
    fat-tailed imbalance spread better (visually/statistically compared in the notebook).
  - Sample: `t_noise = standard_t(df=t_dof) * sqrt((df−2)/df) * sigma_proxy` (the
    `sqrt((df-2)/df)` factor rescales the unit-variance Student-t draw to match the
    empirical sigma), then `imb_fc = da + t_noise`.
  - `imb_up_fc = max(da, imb_fc)`, `imb_down_fc = min(da, imb_fc)` — **full settlement
    price levels for each direction, not a premium over `da`** (this convention is
    load-bearing: `6_models/dispatch_shared.py` documents that an earlier
    premium-only convention under-charged shortfalls by exactly `da` and was an
    exploitable, unbounded incentive for a policy to bias its own forecast — fixed by
    switching to full-price-level columns).
  - All four proxy columns (`da_fc, imb_fc, imb_up_fc, imb_down_fc`) finally clipped to
    `≥0` (real negative-price hours are rare/exogenous and not something a policy should
    be able to manufacture via its own forecast).
- **Decision-time vs settlement-time separation** (`dispatch_shared.py`,
  `select_fc_columns`/`get_prices`): every optimisation *decision* solve — training,
  evaluation, and the oracle's own decision step — is fed the `_fc` proxy columns only;
  the true `da`/`imb`/`imb_up`/`imb_down` columns are reserved exclusively for
  *settlement* (computing the actual realised cost after the fact, via
  `realised_breakdown`). This firewall is enforced by which columns are selected before
  calling `get_prices`, not by a runtime type check.

## 3. Baseline probabilistic forecaster (`4_forecasting/forecasting.py`)

- **Architecture** (`Baseline_Forecaster`, confirmed from the actual trained checkpoint
  `4_forecasting/baseline_forecaster_best.pt`'s stored `model_config`): GRU encoder over
  historical features → concatenated with future/exogenous features → dense head →
  per-quantile outputs, built to be **monotonic across quantile levels** via a
  softplus-cumsum construction (raw NN output passed through softplus then cumulatively
  summed, guaranteeing non-crossing quantiles by construction — this is one of the
  project's methodological contributions, called out explicitly in the commit history as
  producing "monotonic CDFs for use with CVXPYLayers").
  - `n_hist_features=4`, `n_exo_features=9`, `n_quantiles=19`, `gru_hidden_size=32`,
    `dense_hidden_width=128`, `n_gru_layers=2`, `dropout=0.1`, `horizon=24`, `eps=1e-4`.
  - **Note**: the module-level `DEFAULT_MODEL_CONFIG` in `forecasting.py` is stale
    relative to the actual trained/deployed checkpoint — use the checkpoint's own
    `model_config` (values above) as the source of truth, not the source file's default.
- **Quantile levels**: `QUANTILE_LEVELS = [0.05, 0.10, ..., 0.95]` step 0.05, **Q=19**
  levels (`forecasting.py:44-45`; confirmed identical in the checkpoint).
- **Training hyperparameters** (from checkpoint): `best_lr=0.0005`, `batch_size=64`,
  `weight_decay=1e-4`, early stopping `max_epochs=200, patience=10`, best epoch **24**,
  seed `20240801` (this exact seed is reused everywhere else in the pipeline — copula
  Sobol draws, TrainConfig defaults — for reproducibility/consistency).
- **Hyperparameter search**: 80-configuration grid over `gru_hidden_size ∈ {4,8,16,32}` ×
  `dense_hidden_width ∈ {8,16,32,64,128}` × `lr ∈ {0.0005,0.001,0.005,0.01}`, selected by
  validation pinball loss. Winner: `gru_hidden_size=32, dense_hidden_width=128, lr=0.0005`
  (val pinball 41.85, the global minimum of the grid).
- **Normalisation** (checkpoint `scaler_stats`): z-score on `hist_cols` (fitted mean/std
  per feature from the training split only) and on the scalar target `y` (`prosumption`);
  cyclical/flag `exo_cols` (`hour_sin`, etc.) left unscaled (mean 0, std 1 trivially),
  `solar_irrad`/`ambient_temp` (also appearing as exo features) scaled with the same
  stats as their `hist_cols` counterparts.
- **Loss**: pinball/quantile loss, computed per-day across all 19 quantile levels and 24
  leads (`pinball_per_day`, `7_model_training/dfl_train_utils.py:105-108`).
- **Data windows**: gate-aligned daily windows — one forecast issued per calendar day at
  `issue_hour=9`, predicting the next 24 hourly leads (`make_windows(...,
  gate_aligned_only=True, issue_hour=9)`), matching the price proxy's own 9am issue-time
  design (§2) so the whole pipeline shares one consistent decision-timing convention.
- **Splits** (from checkpoint `split_dates`, exact):
  - Train: 2018-01-01 00:00 → 2018-12-31 23:00 (`TRAIN_START`→`VAL_START−1h`)
  - Val: 2019-01-01 00:00 → 2019-06-30 23:00 (`VAL_START`→`TEST_START−1h`)
  - Test: 2019-07-01 00:00 → 2020-06-30 23:00 (`TEST_START`→`TEST_END`) — **366 days**,
    the "sealed test year" referenced throughout the evaluation notebooks.

## 4. Scenario generation — Gaussian copula (`5_scenario_gen/`)

- **Purpose**: convert the forecaster's per-lead marginal quantile forecasts into a
  *jointly* correlated ensemble of scenario trajectories across the 24-hour horizon
  (marginals alone say nothing about how errors at different leads co-move), for use as
  the SAA scenario set in the dual-price epigraph and, historically, as the recourse
  hedge input at eval time.
- **Correlation estimation** (`copula_testing_neat_final.ipynb`, the "build" notebook;
  functions later extracted into `copula_lib.py`):
  1. Run the **frozen baseline forecaster** over N=365 gate-aligned **training-split**
     days (2018), producing `(365, K=24, Q=19)` quantile forecasts.
  2. PIT + probit transform (`build_probit_matrix`): for each day/lead, build a marginal
     CDF from that day's 19 quantiles (with slope-extrapolated tails beyond the 0.05/0.95
     levels), map the realised value through it to get `u=F̂(y) ∈ (0,1)`, then
     `X = Φ⁻¹(u)` (inverse-normal / probit) → an approximately-standard-normal
     `(365, 24)` matrix.
  3. `year_corr_matrix = corrcoef(X)` — a plain **annual, year-pooled Pearson
     correlation** of the probit scores across the 24 lead hours. This *is* the Gaussian
     copula correlation matrix Σ (24×24).
  4. Seasonal/EWMA correlation alternatives were examined as diagnostics but **not used**
     (`covariance_choice` metadata literally says "annual (seasonal/EWMA examined, not
     used)"); a seasonal diagnostic function exists (`seasonal_correlation_matrices`), but
     **no EWMA implementation was found anywhere in the repo** — flag as an
     undocumented/unresolved alternative if this needs mentioning, not a claim to make
     as settled fact.
  5. `build_Z_corr`: 64 points drawn from a scrambled **Sobol low-discrepancy sequence**
     (`scipy.stats.qmc.Sobol`, `d=24`, `seed=20240801`) in `[0,1]^24` (chosen over plain
     pseudo-random for even, representative space-filling of the 64-scenario ensemble),
     mapped through `norm.ppf` to standard normal, then correlated via the Cholesky
     factor of Σ: `Z_corr = Z @ chol(Σ)ᵀ`, shape **(S=64, K=24)**.
  6. This whole bundle (`year_corr_matrix`, `Z_corr`, `quantile_levels`, `S=64`, `K=24`,
     `seed=20240801`) is pickled once to `5_scenario_gen/frozen_copula.pkl` — confirmed
     directly by loading the live file.
- **Why everything is frozen** (`copula_lib.py:266-282` docstring, near-verbatim):
  differentiable Gaussian-copula sampler with **frozen Σ and frozen draws**; a
  bracket-index/interpolation-weight "plan" is precomputed once from the frozen draws
  and the quantile *levels* only (never the quantile *values*), so it stays valid for
  every DFL training step — during training only the forecaster's **quantile values**
  flow through gradient (interpolated through the frozen plan); the dependence
  structure/rank draws are non-trainable buffers, matching the design intent that DFL
  only ever updates the forecaster, never the copula. Stated rationale: (a) speed — no
  re-searching the quantile grid every forward pass; (b) train/eval consistency — the
  same scenario rank structure is reused identically everywhere; (c) correct gradient
  design — dependence structure shouldn't be learnable from a single DFL run's data.
- **`FrozenCopulaSampler.prosumption`/`mean_and_errors`**: at call time, interpolates the
  *current* (differentiable) quantile values through the frozen bracket/weight plan
  (inverse-transform sampling) to produce `pl_hat` (mean anchor) and `xi` (per-scenario
  deviations from that mean), fully differentiable back to the forecaster's parameters.
- A batched variant (`prosumption_batched`) exists, is numerically verified against the
  per-day loop and gradient-checked, but **is not actually called anywhere in the live
  training loop** — `dfl_train_utils.py` loops per-day calling `mean_and_errors` instead.
  Worth noting as an implemented-but-unused optimisation, not a design decision to
  narrate as load-bearing.

## 5. Dispatch optimisation formulations (`6_models/`)

Two live formulations share one selector, `mode ∈ {"single-price", "dual-price"}`
(plus a vestigial `"dispatchability"` — see §7): **`setup_1stage`** (all DFL training)
and **`setup_full_robust`** (evaluation only, the one true "deployment" formulation). A
third, `setup_point_robust`, was fully retired (function no longer exists; see §7).

### 5a. Fixed physical parameters (`FixedParams`/`FixedParams1Stage`)

- `T=24` (hourly settlement periods), `dt=1.0h`, `eta_ch=eta_dis=0.95` (round-trip
  efficiency split symmetrically), `SOC0 = 0.5·B_max` (half-full start/end).
- **Battery archetypes** (`7_model_training/train_dfl_forecasts.py`, `(name, C_ch, C_dis,
  B_max)`, SOC0 derived): `short_sharp` (4MW/4MW/2MWh, 0.5h duration — fast, low
  capacity), `balanced` (2MW/2MW/4MWh, 2h duration), `long_slow` (1MW/1MW/8MWh, 8h
  duration — slow, high capacity). All three trained/evaluated per price mode.
- `gamma` (Tikhonov regularisation on the decision variables): **1e-4 at training time**
  (needed purely so the KKT system `cvxpylayers` implicitly differentiates through is
  invertible/well-posed — a strictly-convex-objective requirement for a unique,
  differentiable solution map), **0 at evaluation time** (plain forward Gurobi solves,
  no differentiation happening, so the regulariser has no remaining justification).

### 5b. `setup_1stage` — the training surrogate (no recourse)

- Decision variables: `p_ch_hat, p_dis_hat` (nominal charge/discharge), `p_da_bat`
  (day-ahead bid contribution) — **pinned**: `p_da_bat == p_ch_hat − p_dis_hat`.
- No `D_ch`/`D_dis` recourse variables exist at all in this formulation — chosen for
  training tractability (DFL needs to solve+differentiate through the layer every
  minibatch step, many times; the full robust box/LDR machinery is too expensive for
  that inner loop — this is exactly what the commit history refers to as "the robust
  formulation was far too computationally intense to perform a backwards pass with
  CVXPYLayers").
- Single-price objective: `C_da = dt·πda·p_da_bat`, `C_imb ≡ 0` (see §5d for why this is
  exact, not an approximation) — decision-time proxy prices only, per §2's firewall.
- Dual-price objective: `C_da` as above, plus an **epigraph-reformulated** SAA imbalance
  term over the frozen scenario ensemble (see §5e) — for 1-stage, the scenario deviations
  pass through with no recourse adjustment (`R=None` in the shared epigraph builder), so
  this term is a genuine part of the *objective's value* but mathematically **inert for
  the argmin** (it's additive-constant in the decision variables) — the real training
  signal for dual-price comes entirely from the settlement-side gradient (§6), not from
  this in-LP scenario term.
- Penalty: `gamma·(‖p_ch_hat‖² + ‖p_dis_hat‖² + ‖p_da_bat‖²)`.

### 5c. `setup_full_robust` — the evaluation formulation (real LDR recourse)

- Adds a genuine **linear decision rule (LDR)** recourse policy: `D_ch, D_dis ∈ ℝ^{T×T}`,
  constrained lower-triangular (`cp.upper_tri(D) == 0`) for **non-anticipativity** —
  recourse at settlement period `t` can only react to deviations realised by time `t`,
  never future information.
- `R = I + D_ch − D_dis` — the net map from a realised deviation vector `ξ` to realised
  net imbalance (identity term = one-for-one pass-through if no recourse acts).
- **Robustification via LP duality** (`robustify_vec`, `dispatch_setup.py`): every
  physical constraint (charge/discharge rate limits, non-negativity, SOC bounds, terminal
  SOC) must hold for *every* `ξ` in a box `{ξ : −h_minus ≤ ξ ≤ h_plus}`. This
  semi-infinite constraint is reformulated exactly via strong LP duality into a finite
  set of constraints with auxiliary dual variables `μ_p, μ_m ≥ 0`: given
  `A0 + (Aξ) ≤ B ∀ξ∈box` iff `∃μ_p,μ_m≥0: μ_p−μ_m=A, μ_p·h_plus+μ_m·h_minus ≤ B−A0`. 6
  constraint families → 12 extra `(T,T)` dual-variable blocks. This is what makes the
  robustness **exact** (not sampled/approximate) — the trade-off is that it's too
  expensive to differentiate through at every DFL training step, hence the 1-stage
  training surrogate.
- Terminal condition is robustified too: `G[T−1,:]==0` forces the recourse-induced SOC
  deviation to be exactly zero at the final period for *every* `ξ` in the box (not just
  nominally), i.e. the battery provably ends at `SOC0` under every realisation.
- **The box** (`compute_box`, `dispatch_shared.py`): `h_plus = q_{0.85} − mean`,
  `h_minus = mean − q_{0.15}`, both from a **frozen baseline (reference) forecaster's**
  quantiles, computed once per test day and detached (no gradient) — applied identically
  to every model being evaluated. `box_levels=(0.15, 0.85)` was **empirically locked in**
  via a full-year sweep over candidate levels using mean total realised cost per day as
  the selection metric (`6_models/param_sweeps/h_selection_sweep.ipynb`) — dominates the
  previously-used `(0.20, 0.80)` on both the robust model and the training surrogate.
- `Variables` dataclass deliberately has **no `imb_det` field**: under the pinned bid,
  the deterministic imbalance is provably always exactly 0 (algebraic consequence of
  `p_da_bat == p_ch_hat − p_dis_hat`), so it was removed as a field entirely rather than
  kept as an always-zero term.

### 5d. Single-price objective — why `C_imb` is exactly 0 at decision time

- `C_da = dt·πda·p_da_bat` (constant `πda·pl_hat·dt` term dropped from the cvxpy
  objective since it doesn't affect the argmin — restored downstream in settlement).
- `C_imb ≡ 0.0` (literally, not even a cvxpy expression): `E[πimb·p_imb] =
  πimb·(deterministic imbalance) = 0` — the recourse-driven `R·ξ` term vanishes in
  expectation (`E[ξ]=0`), and the deterministic imbalance is 0 by the pinned bid. `πimb`
  is never even declared as a Parameter in this mode (nothing left for it to multiply).
  For 1-stage this identity holds *pointwise* (no expectation needed, no recourse/box
  term exists at all); for full-robust it holds *in expectation over ξ*.
- **Consequence** (established earlier in this project's investigation, worth restating
  for methodology): single-price's *decision* is structurally invariant to the
  forecaster's mean — the entire economic lever available to single-price DFL training
  is the forecast mean's effect on the **settlement**-side linear imbalance cost, not on
  the decision LP itself.

### 5e. Dual-price objective — the epigraph reformulation

- Dual-price settlement is piecewise-linear/convex in the imbalance:
  `πup·(imb)⁺ + πdown·(−imb)⁺` — not directly DPP-compliant when `imb` itself is affine
  in Parameters times Variables (product of a Parameter with a `max`/`pos` of another
  Parameter-affine expression breaks Disciplined Parametrised Programming, which
  `cvxpylayers` requires for implicit differentiation).
- **Fix**: introduce free variables `p_plus, p_minus ≥ 0` with `p_plus − p_minus =
  imb_scen` (`imb_scen = R·ξ_samples` for full-robust, `= ξ_samples` directly for
  1-stage), and `C_imb = (dt/N)·Σ(πup·p_plus + πdown·p_minus)` — a pure LP, DPP-compliant
  since the price Parameters now multiply only free non-negative Variables. At the
  optimum (non-negative objective coefficients, minimised) this recovers the exact
  piecewise cost automatically. `N=64` scenarios (SAA over the frozen copula ensemble).
- By contrast, the **oracle** (never differentiated through — a plain one-off Gurobi
  solve) uses `cp.pos`/`cp.neg` directly, no epigraph needed — this contrast is the
  clean way to explain *why* the epigraph trick exists: it's purely a DPP-for-autodiff
  requirement, not a correctness requirement of the LP itself.
- `p_plus`/`p_minus` deliberately receive **no** gamma regularisation (measured directly:
  no effect on solver-accuracy flag rates, but 17–28% slower wall-clock — reverted).

### 5f. Settlement engine — `realised_breakdown` (`dispatch_wrapper.py`)

The single source of truth for realised quantities, used identically (same function) for
both DFL training loss and test-time evaluation.

- `xi_real = realised − pl_hat` (deviation from the SAME anchor the decision layer used),
  `bid = pl_hat + p_da_bat`.
- Raw (pre-saturation) recourse: `p_ch_raw = p_ch_hat + D_ch·xi_real`, `p_dis_raw =
  p_dis_hat + D_dis·xi_real` (recourse evaluated at the *actual* realised deviation, vs
  the nominal `ξ=0` path used inside the LP itself).
- **Per-timestep physical saturation loop** (`clip_recourse=True`, the default, used both
  in training and at test time): sequential SOC simulation clamps `p_ch`/`p_dis` each
  hour to `[0, min(rate limit, SOC-headroom-implied limit)]`, so the reported trajectory
  is physically realisable even if the LDR policy's raw recommendation would have
  over/undershot the battery's real capacity. A non-clipping branch also exists
  (vectorised, no saturation) for an internal correctness gate.
- `p_g = realised + p_ch_r − p_dis_r` (realised grid draw), `p_imb = p_g − bid`.
- `C_da = dt·πda·bid` (true settlement prices now, restoring the constant dropped from
  the decision LP). Single-price: `C_imb = dt·πimb·p_imb` (**signed**, can be negative —
  no minimum, no accuracy-seeking property, a directionless price-sign nudge). Dual-price:
  `C_imb = dt·(πup·(p_imb)⁺ + πdown·(−p_imb)⁺)` (**always ≥0**, asymmetric,
  pinball-loss-shaped in the forecast mean, minimised exactly at `pl_hat=realised`).
- This linear-vs-V-shaped settlement distinction is the single structural fact underlying
  essentially every empirical finding about *why* dual-price DFL training outperforms
  single-price (documented at length in `8_testing/balanced_single_vs_dual_findings.md`
  §3/§8) — worth including in the methodology as the formal mechanism, even though the
  *results themselves* belong in a later chapter.

### 5g. Oracle — perfect-foresight economic benchmark (`oracle.py`)

- Deliberately **one** oracle: the economic one (`min C_da+C_imb` under perfect
  foresight of the true `prosumption`) — a perfect-foresight *dispatch*-tracking oracle
  was deliberately not built, since it would collapse toward a forecast-accuracy gap
  that CRPS already measures.
- Bid is **pinned** to the true realised position (`bid = p_d + p_ch − p_dis`, not free)
  — a free bid would let the oracle arbitrage the known price spread using only price
  information the policy has equally available, corrupting "value of perfect load
  information" with "value of an arbitrage mechanism the policy is structurally barred
  from." Pinning makes `imb≡0` for the oracle by construction, so its cost reduces purely
  to `C_da` — pure price arbitrage of charge/discharge against the *true* load.
- **Decide-then-resettle procedure** (`oracle_realised_cost`): the oracle solves under
  the same `_fc` proxy prices the policy itself sees at decision time (fair comparison —
  not letting the oracle see true prices it wouldn't have had), then the *exact same*
  decision is resettled against true market prices via the identical
  `realised_breakdown` engine used for the policy's own reported cost.
- `regret = policy_realised_cost − economic_oracle_cost`, reported for every evaluated
  instance — interpreted as genuine economic decision regret.

## 6. DFL training procedure (`7_model_training/`)

- **Full differentiable chain**: forecaster weights θ → `q_norm` (raw NN output,
  softplus-cumsum monotone construction) → `q_phys` (affine de-normalisation,
  order-preserving) → Gaussian-copula interpolation (`pl_hat`, `xi`; frozen
  bracket/weight plan, differentiable in the quantile values) → `cvxpylayers` LP solve
  (implicit differentiation via `diffcp` through the KKT/cone-program conditions) →
  `realised_breakdown` settlement → loss.
- **Loss** (`_combine_loss`, `dfl_train_utils.py`): `L_base` = pinball loss on
  `q_norm` (standard probabilistic-forecast accuracy term); `f_dfl` =
  `clamp(C_da+C_imb, min=0)`, the realised economic cost from `realised_breakdown`
  (oracle-based regret was tried and removed — training now targets raw realised cost
  directly, avoiding an `_fc`/true price-basis mismatch that the oracle would have
  introduced into the loss).
- **`self_balanced_loss(L_base, f_dfl)` = harmonic mean of the two**, exactly (algebraic
  identity `alpha·L_base == beta·f_dfl` always holds) — a **scale-equalizer**, not a
  "focus on the worse metric" mechanism: whichever term is currently larger gets *less*
  weight on its own gradient contribution for that batch, since the harmonic mean is
  dominated by the smaller term.
- **Why the recourse mechanism can't be the training signal for 1-stage**: `setup_1stage`
  has no `D_ch`/`D_dis` at all, so `xi_samples`/the epigraph variables are fully
  decoupled from the returned decision — real hedging-through-recourse only exists at
  eval time (`setup_full_robust`). The entire mode-dependent training signal reduces to
  the settlement-side gradient shape difference (§5f/§5d/§5e).
- **Training modes**: `single-price`, `dual-price` (`dispatchability` structurally
  unsupported for 1-stage — raises explicitly, no recourse mechanism to build a tracking
  term from).
- Trained per archetype × price-mode (6 corners: `{single,dual}-price` ×
  `{balanced, long_slow, short_sharp}`), one training run each (no seed-averaging — a
  documented limitation, §9).
- `TrainConfig` defaults: `lr=5e-4`, `batch_size=8`, `max_epochs=20`, `patience=5`,
  `grad_clip=3.0`, `seed=20240801`; solver `ECOS` with `SCS` fallback.

## 7. Retired components — brief design-history context

- **`point-robust`** (a third, formerly-live constraint-masking approximation of the
  robust box, non-LP-dual): fully removed — no `setup_point_robust` function remains
  anywhere; only comment-residue across several files still references it. Per the
  commit history, the robust LDR formulation (what's now `full-robust`) was "far too
  computationally intense to perform a backwards pass with CVXPYLayers", which motivated
  moving to the 1-stage training surrogate + full-robust-at-eval-only split.
  Point-robust's own training path is what was replaced.
- **`dispatchability` mode / the `k∈{0,1}` weighting**: an earlier notation blended
  `(1−k)·A + k·B` between the economic objective and a price-agnostic
  imbalance-*tracking* term, but `k` was in practice always exactly 0 or exactly 1 —
  never actually blended — so the categorical `mode` selector replaced it as more honest
  notation. `dispatchability`'s tracking term, when it existed historically, lived purely
  in the *settlement*-side (`realised_breakdown`'s `p_imb`), price-agnostic by
  construction; the current `build_objective_2stage` has no dispatchability-specific term
  of its own (falls through to a bare Tikhonov-penalty problem for that mode) — consistent
  with the mode being out-of-scope/unexercised in every live training/eval script
  (`train_dfl_forecasts.py`, `eval_raw.py` both explicitly scope it out).
- `cholesky_of_second_moment` (`dispatch_shared.py`) — a differentiable
  covariance-of-scenario-deviations → Cholesky-factor helper, designed for a
  Sigma-based quadratic recourse-tracking penalty (`sum_squares(R·Σ_chol)`) that was
  real in the archived point-robust/full-robust dispatch layers — still present and
  called conditionally (`mode=="dispatchability"`) in the training code, but currently
  dead in practice since that mode is never trained.

## 8. Evaluation methodology (`8_testing/`)

- **`eval_raw.py`**: the collection-only evaluation backbone for every reported number.
  Uses `setup_full_robust` exclusively (the "deployment" formulation) — 1-stage and the
  retired point-robust are explicitly documented as "training-time surrogates only,
  never the real deployment decision-maker." `GAMMA=0`, `BOX_LEVELS=(0.15,0.85)`,
  `N_SCEN=64`, evaluated over the full 366-day sealed test year.
- **Price-mode firewall**: enforced structurally by the `FORECASTERS` dict's shape (which
  checkpoints are grouped under which price mode), not a runtime check — deliberately
  crossable for cross-evaluation experiments (below) via direct `evaluate_one_pair`
  calls.
- **Grid**: {baseline, DFL-trained forecaster} × {single-price, dual-price} × {3
  archetypes} — the full evaluation surface referenced by `aggregate_results.ipynb`'s
  Q1 (headline economic comparison table + scatter), Q2 (specialist-vs-baseline benefit
  by archetype), Q3 (cross-archetype transfer heatmap, benefit-over-baseline).
- **Metrics**: realised total cost (`C_da+C_imb`), CRPS (forecast accuracy, computed on
  `q_phys`, physical units), regret vs the perfect-foresight oracle, net (signed)
  imbalance vs gross imbalance volume, hour-of-day MAE and signed bias.
- **Cross-price-mode evaluation** (`8_testing/cross_price_eval.ipynb`): deliberately
  evaluating a dual-price-trained forecaster on the single-price settlement pipeline and
  vice versa, to isolate how much of each mode's benefit is a transferable mean-shift vs
  mode-specific calibration.
- Later analysis narrowed to the **balanced archetype only** for the single-vs-dual deep
  dive (a scoping decision, not reconciled with Q2/Q3's full 3-archetype grid — flag as
  an open thread, §9).

## 9. Known limitations / caveats worth stating explicitly

- **No seed-averaging**: one training run per corner; all reported numbers are point
  estimates with no run-to-run variance quantified.
- **Copula correlation fit in-sample**: `year_corr_matrix` is estimated from the
  baseline forecaster's residuals on its own **training-split** days (2018), not a
  held-out set — a methodological caveat worth naming plainly rather than glossing over.
- The "(pre-weather)" copula/correlation artifact files in `5_scenario_gen/` are plausibly
  from before weather features were added to the forecaster, but this was not confirmed
  by any code comment or commit message — treat as an inference, not a settled fact, if
  it comes up.
- Q2/Q3 (`aggregate_results.ipynb`) still assume the full 3-archetype grid; the later
  balanced-only narrowing for the single-vs-dual mechanism analysis was a separate,
  later scoping decision and the two haven't been reconciled into one consistent
  narrative scope.
- An ablation isolating how much of dual-price's benefit is the shared mean-shift vs the
  V-shaped-gradient-specific calibration was discussed but never built.

---

## How to use these notes

- Sections 1–5 describe the **system as built** (data → forecaster → scenario generator
  → optimisation formulations) — the "what" and "why" for a Methods chapter's design
  description.
- Section 6 describes the **training procedure** — the DFL loss construction and
  gradient mechanism.
- Section 8 describes the **evaluation protocol** — what a Methods chapter would call
  the experimental design (not the results themselves, which live in
  `8_testing/balanced_single_vs_dual_findings.md` and the three testing notebooks, out
  of scope for this handoff).
- Section 9's caveats are legitimate content for a Limitations subsection.
- All file:line references above are pointers back into the repo for anyone who wants to
  re-verify a specific claim before committing it to the dissertation text.
