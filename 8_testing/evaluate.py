"""Baseline-vs-DFL evaluation on the sealed TEST set (TEST_START..TEST_END).

For each (architecture, mode) corner, evaluates:
  - the BASELINE forecaster (unchanged CRPS-trained weights) through solve_plain
  - the corner-specific DFL checkpoint (7_model_training/dfl_{architecture}_{mode}.pt),
    if it exists

Both go through the identical dispatch problem (build_problem/solve_plain, plain Gurobi
QP -- no cvxpylayers/ECOS involved, this is a pure test-time forward solve) and the same
per-day metrics (per_day_metrics), so the two are directly comparable.

Regret is always reported against the single common ECONOMIC oracle (build_oracle's only
mode), for every mode including "dispatchability" -- see build_oracle's docstring: for
dispatchability this reads as "the economic price of dispatchability", not decision
regret. The oracle's own decision is solved under the SAME fc/proxy prices the policy's
decision solve gets (oracle.oracle_realised_cost), then settled at true prices exactly
like the policy's own reported cost -- an apples-to-oranges mismatch otherwise (see
oracle_realised_cost's docstring). Oracle-regret was removed from TRAINING entirely (see
dfl_train_utils.py's module docstring) but stays here as a reporting metric -- unaffected
either way, since the oracle was never in the training loop's gradient path.

Usage:
    python evaluate.py                 # full test set, all corners with an available checkpoint
    python evaluate.py --smoke 10      # first 10 test days only, all corners (for a fast check)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cvxpy as cp
import pickle
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
MODELS_ROBUST_DIR = MODEL_DIR / "models_robust"
DFL_TRAIN_DIR = ROOT_DIR / "7_model_training"
RESULTS_DIR = ROOT_DIR / "8_testing" / "results"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODELS_ROBUST_DIR))

from forecasting import (reindex_and_impute, build_features, make_windows,
                         normalise_hist, normalise_exo, denormalise_y, Baseline_Forecaster,
                         QUANTILE_LEVELS, TEST_START, TEST_END,
                         HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler
# oracle: the shared perfect-foresight economic oracle -- mode/architecture-independent,
# same benchmark for every corner. Originally kept from dispatch_layer.py (not the
# archived models_robust/dispatch_layer_robust.py, whose own build_oracle had a stale
# param_by_name -- old "imb_up"/"imb_down" keys, never updated to pi_imb_up/pi_imb_down
# -- that would KeyError against oracle_price_values' current output). dispatch_layer.py
# has since been stripped to just this oracle and renamed oracle.py.
from oracle import build_oracle, solve_oracle, oracle_realised_cost
from dispatch_wrapper import realised_breakdown, RealisedBreakdown
# BOTH training-time surrogates -- point_robust (cheap 3-point feasibility heuristic) AND
# 1stage (no recourse mechanism at all) -- are evaluated through the FULL LP-dual robust
# counterpart (dispatch_setup.setup_full_robust, the consolidated version of the former
# models_robust/dispatch_layer_robust.py) instead of their own training-speed-optimised
# formulations. Neither point_robust's D_ch/D_dis (no EXACT feasibility guarantee at test
# time, only the weak gamma Tikhonov penalty plus the 3-point check) nor 1stage's complete
# absence of a recourse policy is the real deployment decision-maker -- both exist purely
# to make DFL training tractable (point_robust: cheap forward solve; 1stage: no
# cvxpylayers epigraph at all). setup_1stage/build_objective_1stage are therefore never
# called from this file -- there is no evaluate_model_1stage; every checkpoint, regardless
# of which architecture trained it, is scored by plugging its forecaster weights into the
# SAME evaluate_model/setup_full_robust pipeline. Confirmed via a smoke test: the full
# robust solve_plain is ~0.18s/day (that slow-solve concern was about the differentiable
# cvxpylayers TRAINING path, not this plain Gurobi test-time solve), and cuts D_ch/D_dis
# norms roughly in half and sat_MWh by ~10x versus point_robust's own formulation.
from dispatch_setup import default_fixed_params, setup_full_robust as build_problem
from dispatch_shared import (solve_plain, compute_box, get_prices, oracle_price_values,
                              cholesky_of_second_moment, select_fc_columns,
                              price_model_for_settlement, build_layer_vals)

PRICE_COLS = ["da", "imb", "imb_up", "imb_down"]
PRICE_COLS_FC = ["da_fc", "imb_fc", "imb_up_fc", "imb_down_fc"]   # all 4 real, always->=0 columns
PRICE_COLS_ALL = PRICE_COLS + PRICE_COLS_FC
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cpu")   # plain Gurobi solve; forecast forward pass is cheap on CPU too

N_SCEN = 64
GAMMA = 1e-4
BOX_LEVELS = (0.15, 0.85)   # locked in from h_selection_sweep.ipynb -- same as training
CORNERS = ["single-price", "dual-price"]   # economic modes -- both point_robust- and
                                            # 1stage-trained checkpoints are evaluated here
# point_robust dispatchability: ONE trained checkpoint (its decision solve is
# price-independent -- no price Parameters declared, see dispatch_objectives.py), but
# evaluated/settled under BOTH conventions below, NOT folded into CORNERS -- "the
# economic price of dispatchability" is a genuinely different number depending which
# market the realised imbalance would have settled in, so both are real, reportable
# results from the one checkpoint, not a redundant rerun.
DISPATCHABILITY_SETTLEMENTS = ("single", "dual")


def load_test_windows():
    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS_ALL,
                  n_hist=N_HIST, horizon=HORIZON)
    return make_windows(frame, y_range=(TEST_START, TEST_END), **win_kw)


def forecast_quantiles(model, sc, x_hist_day, x_fut_day):
    model.eval()
    with torch.no_grad():
        xh = normalise_hist(np.asarray(x_hist_day), sc)
        xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
        xf = normalise_exo(np.asarray(x_fut_day), sc)
        xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
        q_norm = model(xh, xf)
        q_phys = denormalise_y(q_norm, sc)
    return q_phys.squeeze(0).to(torch.float64)


def crps_per_measurement(y_true, quantiles, levels=QUANTILE_LEVELS):
    """Quantile-based CRPS approximation, PHYSICAL units (MW), averaged PER HOURLY
    MEASUREMENT over the 24-hour delivery window -- NOT this script's per-day-TOTAL
    convention used elsewhere (throughput_MWh, imbalance_volume_MWh, etc. are still
    day-totals). CRPS is a per-measurement forecast-quality metric, not a physical
    quantity that accumulates over a day, so summing it across hours (the old
    convention) inflated it by a factor of K=24 for no meaningful reason -- averaging
    over hours instead gives a like-for-like per-measurement number.

    CRPS(F, y) ~= 2 * integral_0^1 pinball_tau(y, F^-1(tau)) dtau, approximated here as
    2 * mean over the Q quantile LEVELS of the pinball loss at that level, per hour, then
    averaged over the K=24 hours. This is the standard textbook quantile-CRPS estimator
    (properly normalised, with the 2x factor) -- NOT the same convention as this codebase's
    training-time "val_pinball" (which is an unnormalised sum over K*Q with no 2x factor,
    used only as a relative/internal training signal). If you need a number directly
    comparable to val_pinball, drop the "2.0 *" and the "/ Q" mean here (and the final
    per-hour mean).

    y_true    : (K,) realised values, physical units.
    quantiles : (K, Q) forecast quantiles, physical units (same units as y_true).
    levels    : (Q,) quantile levels in (0,1).
    Returns   : scalar, physical units, averaged over the K hourly measurements.
    """
    y_true = np.asarray(y_true, float)
    quantiles = np.asarray(quantiles, float)
    levels = np.asarray(levels, float)
    e = y_true[:, None] - quantiles                        # (K, Q)
    q = levels[None, :]                                     # (1, Q)
    pinball = np.maximum(q * e, (q - 1.0) * e)               # (K, Q)
    crps_per_hour = 2.0 * pinball.mean(axis=1)                # (K,)
    return float(crps_per_hour.mean())


# -------------------------------------------------------------------------------------
# THE SIX PER-DAY SCALARS (reductions over one breakdown). oracle_cost is precomputed
# per (day, price_model). Moved here from dispatch_wrapper.py -- this is its only real
# caller; the training scripts imported it but never called it (confirmed by grep
# before the move).
#
# Absolute saturation (sat_hours/sat_MWh, the clipping magnitude between p_ch_raw/
# p_dis_raw and their post-clip p_ch_r/p_dis_r) is deliberately NOT reported here --
# removed as an evaluation statistic on request. It remains the right signal for
# h_selection_sweep.py's own box-level elbow selection (a different, deliberate use),
# just not carried into this script's per-day/summary metrics.
# -------------------------------------------------------------------------------------
def per_day_metrics(fp, bd: RealisedBreakdown, oracle_cost: float) -> dict:
    dt = fp.dt
    total_cost = float(bd.C_da + bd.C_imb)
    return {
        "total_cost":     total_cost,                                  # 1
        # Regret against a ZERO-IMBALANCE, cost-minimising oracle -- perfect load
        # foresight, bid pinned to the true realised position, so it never transacts in
        # the imbalance market (C_imb == 0 by construction) and only ever arbitrages the
        # day-ahead price. NOT the true global cost optimum: single-price policies can
        # legitimately go negative here (a real, expected result, not a bug) by biasing
        # pl_hat to arbitrage the day-ahead/imbalance price spread -- a channel this
        # oracle is structurally barred from. Dual-price cannot (pi_imb_up/pi_imb_down
        # >= 0 means any imbalance only ever ADDS cost), so its regret stays >= 0.
        "regret_v_zero_imbalance_oracle": total_cost - float(oracle_cost),   # 2
        # SIGNED sum, not abs(p_imb) -- positive and negative hours within the day are
        # allowed to cancel, so this reads as the day's NET imbalance volume (net
        # long/short position), not its total unsigned magnitude/noise level.
        "imbalance_volume_MWh": float((bd.p_imb * dt).sum().item()),   # 3
        "C_da":           float(bd.C_da.item()),                       # 4a  (DA/IMB split...
        "C_imb":          float(bd.C_imb.item()),                      # 4b   ...sums to total_cost)
        "throughput_MWh": float(((bd.p_ch_r + bd.p_dis_r) * dt).sum().item()),  # 5 (grid-side)
    }


def precompute_test_oracle_costs(windows, mode, phys_fp, n_days=None):
    price_model = price_model_for_settlement(mode)
    ob = build_oracle(phys_fp, price_model, objective="economic")
    n = n_days if n_days is not None else len(windows.delivery_start)
    costs = np.empty(n)
    for d in range(n):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        fc_price_day = select_fc_columns(price_day, cols=PRICE_COLS_ALL, real_cols=PRICE_COLS_FC)
        fc_vals = oracle_price_values(fc_price_day, price_model, realised, cols=PRICE_COLS)
        true_prices = get_prices(price_day, price_model, cols=PRICE_COLS)
        costs[d] = oracle_realised_cost(
            phys_fp, ob, fc_vals, realised, price_model,
            true_pi_da=true_prices["pi_da"], true_pi_imb=true_prices.get("pi_imb"),
            true_pi_imb_up=true_prices.get("pi_imb_up"), true_pi_imb_down=true_prices.get("pi_imb_down"),
            solver=cp.GUROBI)
    return costs


def evaluate_model(model, sc, sampler, windows, oracle_costs, mode, price_model, n_days=None,
                    baseline_model=None, quantile_levels=None):
    """`mode` and `price_model` are DELIBERATELY separate arguments, not derived from
    one another via price_model_for_settlement -- for "single-price"/"dual-price" they
    always coincide (price_model_for_settlement(mode) at the call site), but for
    "dispatchability" the decision solve is price-independent (no price Parameters
    declared -- see dispatch_objectives.py), so the SAME trained decision can be, and
    is, reported under BOTH settlement conventions as two separate rows (see main()) --
    "the economic price of dispatchability" is a genuinely different number depending
    which market the realised imbalance would have settled in."""
    fp = default_fixed_params(num_scenarios=N_SCEN, gamma=GAMMA)
    bundle = build_problem(fp, mode)
    n = n_days if n_days is not None else len(windows.delivery_start)

    rows = []
    for d in range(n):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        q_phys = forecast_quantiles(model, sc, windows.x_hist[d], windows.x_fut[d])
        mean, xi = sampler.mean_and_errors(q_phys)

        # DECISION solve: proxy (_fc) prices only -- never the true columns here.
        fc_price_day = select_fc_columns(price_day, cols=PRICE_COLS_ALL, real_cols=PRICE_COLS_FC)
        fc_prices = get_prices(fc_price_day, price_model, cols=PRICE_COLS)

        # box is FROZEN -- always computed from the baseline forecaster, never from the
        # model currently being evaluated (matches training's convention exactly, see
        # dispatch_shared.compute_box's docstring).
        base_q = forecast_quantiles(baseline_model, sc, windows.x_hist[d], windows.x_fut[d])
        h_plus, h_minus = compute_box(base_q, sampler, quantile_levels, box_levels=BOX_LEVELS)

        xi_samples = xi.detach().cpu().numpy() if mode == "dual-price" else None
        Sigma_xi_chol = cholesky_of_second_moment(xi).detach().cpu().numpy() if mode == "dispatchability" else None
        fc_prices_np = {kk: np.asarray(vv, float) for kk, vv in fc_prices.items()}
        vals = build_layer_vals(fc_prices_np, h_plus=h_plus.detach().cpu().numpy(),
                                 h_minus=h_minus.detach().cpu().numpy(),
                                 xi_samples=xi_samples, Sigma_xi_chol=Sigma_xi_chol)
        vals["pl_hat"] = mean.detach().cpu().numpy()

        try:
            dec = solve_plain(bundle, vals, solver=cp.GUROBI)
        except RuntimeError as e:
            print(f"    day {d} solve_plain failed, skipping: {e}")
            continue

        # SETTLEMENT: TRUE prices always.
        true_prices = get_prices(price_day, price_model, cols=PRICE_COLS)
        bd = realised_breakdown(fp, dec["p_ch_hat"], dec["p_dis_hat"], dec["D_ch"], dec["D_dis"],
                                dec["p_da_rel"], realised=realised, pl_hat=vals["pl_hat"],
                                price_model=price_model, clip_recourse=True, **true_prices)
        m = per_day_metrics(fp, bd, oracle_costs[d])
        m["crps"] = crps_per_measurement(realised, q_phys.detach().cpu().numpy())
        m["day"] = d
        m["date"] = str(windows.delivery_start[d])
        rows.append(m)
    return pd.DataFrame(rows)


def fresh_baseline_model(ckpt):
    m = Baseline_Forecaster(**ckpt["model_config"])
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device)


def dfl_model_from(baseline_ckpt, dfl_ckpt):
    m = Baseline_Forecaster(**baseline_ckpt["model_config"])
    m.load_state_dict(dfl_ckpt["state_dict"])
    return m.to(device)


def summarise(df: pd.DataFrame, source: str, architecture: str, mode: str) -> dict:
    if df.empty:
        return {"source": source, "architecture": architecture, "mode": mode, "n_days": 0}
    return {
        "source": source, "architecture": architecture, "mode": mode, "n_days": len(df),
        "mean_total_cost":  df["total_cost"].mean(),
        "mean_regret_v_zero_imbalance_oracle": df["regret_v_zero_imbalance_oracle"].mean(),
        "mean_crps":        df["crps"].mean(),   # per-hourly-measurement, see crps_per_measurement
        "mean_imbalance_volume_MWh": df["imbalance_volume_MWh"].mean(),
        "mean_C_da":        df["C_da"].mean(),
        "mean_C_imb":       df["C_imb"].mean(),
        "mean_throughput_MWh": df["throughput_MWh"].mean(),
    }


def main(n_days=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("loading test windows...")
    windows = load_test_windows()
    n_total = len(windows.delivery_start)
    n = n_days if n_days is not None else n_total
    print(f"n_test={n_total}  (evaluating {n})")

    baseline_ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt",
                               weights_only=False, map_location="cpu")
    sc = baseline_ckpt["scaler_stats"]
    baseline_model = fresh_baseline_model(baseline_ckpt)

    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)

    phys_fp = default_fixed_params(num_scenarios=N_SCEN, gamma=GAMMA)
    # oracle cache keyed by the RESOLVED price_model_for_settlement(mode), not mode itself
    # -- "single-price" and "dispatchability" both resolve to "single" (see
    # price_model_for_settlement's docstring), so they share one oracle solve for free.
    oracle_cache = {}

    summary_rows = []
    for mode in CORNERS:
        print(f"\n=== corner: architecture=point_robust mode={mode} ===")
        price_model = price_model_for_settlement(mode)
        if price_model not in oracle_cache:
            print(f"  precomputing oracle costs ({price_model})...")
            oracle_cache[price_model] = precompute_test_oracle_costs(windows, mode, phys_fp, n)
        oracle_costs = oracle_cache[price_model]

        print("  evaluating baseline...")
        df_base = evaluate_model(baseline_model, sc, sampler, windows, oracle_costs, mode, price_model, n,
                                  baseline_model=baseline_model, quantile_levels=cop["quantile_levels"])
        df_base.to_csv(RESULTS_DIR / f"baseline_point_robust_{mode}.csv", index=False)
        summary_rows.append(summarise(df_base, "baseline", "point_robust", mode))

        dfl_path = DFL_TRAIN_DIR / f"dfl_point_robust_{mode}.pt"
        if dfl_path.exists():
            print(f"  evaluating DFL checkpoint ({dfl_path.name})...")
            dfl_ckpt = torch.load(dfl_path, weights_only=False, map_location="cpu")
            dfl_model = dfl_model_from(baseline_ckpt, dfl_ckpt)
            df_dfl = evaluate_model(dfl_model, sc, sampler, windows, oracle_costs, mode, price_model, n,
                                     baseline_model=baseline_model, quantile_levels=cop["quantile_levels"])
            df_dfl.to_csv(RESULTS_DIR / f"dfl_point_robust_{mode}.csv", index=False)
            summary_rows.append(summarise(df_dfl, "dfl", "point_robust", mode))
        else:
            print(f"  no DFL checkpoint yet at {dfl_path} -- skipping (baseline-only for now)")

        # 1stage is a training-time-only surrogate (no recourse mechanism at all) --
        # never evaluated through its own setup_1stage decision problem (see this
        # module's import-block comment). Its trained forecaster weights go through the
        # SAME evaluate_model/full_robust pipeline as point_robust above, reusing the
        # df_base/oracle_costs already computed this iteration -- there is no separate
        # "1stage baseline" to compute, since the untrained weights are identical and
        # would just repeat the point_robust baseline row through the same problem.
        dfl_1stage_path = DFL_TRAIN_DIR / f"dfl_1stage_{mode}.pt"
        if dfl_1stage_path.exists():
            print(f"  evaluating DFL checkpoint ({dfl_1stage_path.name}) via full_robust...")
            dfl_ckpt = torch.load(dfl_1stage_path, weights_only=False, map_location="cpu")
            dfl_model = dfl_model_from(baseline_ckpt, dfl_ckpt)
            df_dfl = evaluate_model(dfl_model, sc, sampler, windows, oracle_costs, mode, price_model, n,
                                     baseline_model=baseline_model, quantile_levels=cop["quantile_levels"])
            df_dfl.to_csv(RESULTS_DIR / f"dfl_1stage_{mode}.csv", index=False)
            summary_rows.append(summarise(df_dfl, "dfl", "1stage", mode))
        else:
            print(f"  no DFL checkpoint yet at {dfl_1stage_path} -- skipping")

    # dispatchability: ONE checkpoint, reported under BOTH settlement conventions.
    # Reuses the oracle costs already cached above (single-price/dual-price corners
    # already populated oracle_cache["single"]/["dual"] -- same oracle, no extra solve).
    for price_model in DISPATCHABILITY_SETTLEMENTS:
        label = f"dispatchability_settled-{price_model}"
        print(f"\n=== corner: architecture=point_robust mode=dispatchability settlement={price_model} ===")
        oracle_costs = oracle_cache[price_model]

        print("  evaluating baseline...")
        df_base = evaluate_model(baseline_model, sc, sampler, windows, oracle_costs,
                                  "dispatchability", price_model, n,
                                  baseline_model=baseline_model, quantile_levels=cop["quantile_levels"])
        df_base.to_csv(RESULTS_DIR / f"baseline_point_robust_{label}.csv", index=False)
        summary_rows.append(summarise(df_base, "baseline", "point_robust", label))

        dfl_path = DFL_TRAIN_DIR / "dfl_point_robust_dispatchability.pt"   # same checkpoint, both settlements
        if dfl_path.exists():
            print(f"  evaluating DFL checkpoint ({dfl_path.name})...")
            dfl_ckpt = torch.load(dfl_path, weights_only=False, map_location="cpu")
            dfl_model = dfl_model_from(baseline_ckpt, dfl_ckpt)
            df_dfl = evaluate_model(dfl_model, sc, sampler, windows, oracle_costs,
                                     "dispatchability", price_model, n,
                                     baseline_model=baseline_model, quantile_levels=cop["quantile_levels"])
            df_dfl.to_csv(RESULTS_DIR / f"dfl_point_robust_{label}.csv", index=False)
            summary_rows.append(summarise(df_dfl, "dfl", "point_robust", label))
        else:
            print(f"  no DFL checkpoint yet at {dfl_path} -- skipping (baseline-only for now)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\n" + "=" * 70)
    print(summary.to_string(index=False))
    print(f"\nPer-day CSVs and summary.csv written to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="evaluate only the first N test days (fast correctness check)")
    args = parser.parse_args()
    main(n_days=args.smoke)
