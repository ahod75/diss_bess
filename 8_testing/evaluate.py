"""Baseline-vs-DFL evaluation on the sealed TEST set (TEST_START..TEST_END).

For each of the four (price_model, k) corners, evaluates:
  - the BASELINE forecaster (unchanged CRPS-trained weights) through solve_plain
  - the corner-specific DFL checkpoint (7_model_training/dfl_{pm}_k{k}.pt), if it exists

Both go through the identical dispatch problem (build_problem/solve_plain, plain Gurobi
QP -- no cvxpylayers/ECOS involved, this is a pure test-time forward solve) and the same
per-day metrics (dispatch_wrapper.per_day_metrics), so the two are directly comparable.

Regret is always reported against the single common ECONOMIC oracle (build_oracle's only
mode), for both k=0 and k=1 -- see build_oracle's docstring: at k=1 this reads as "the
economic price of dispatchability", not decision regret.

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
DFL_TRAIN_DIR = ROOT_DIR / "7_model_training"
RESULTS_DIR = ROOT_DIR / "8_testing" / "results"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))

from forecasting import (reindex_and_impute, build_features, make_windows,
                         normalise_hist, denormalise_y, Baseline_Forecaster,
                         QUANTILE_LEVELS, TEST_START, TEST_END)
from copula_lib import FrozenCopulaSampler
from dispatch_layer import (default_fixed_params, build_problem, solve_plain,
                            build_oracle, solve_oracle)
from dispatch_wrapper import (get_prices, oracle_price_values, realised_breakdown, per_day_metrics,
                              cholesky_of_second_moment)

HIST_COLS  = ["prosumption", "solar_irrad", "panel_temp", "ambient_temp"]
FEAT_COLS  = ["solar_irrad", "panel_temp", "ambient_temp"]
EXO_COLS   = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
PRICE_COLS = ["da", "imb", "up_reg_cost", "down_reg_cost"]
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cpu")   # plain Gurobi solve; forecast forward pass is cheap on CPU too

N_SCEN = 64
GAMMA = 1e-4
CORNERS = [("single", 0.0), ("single", 1.0), ("dual", 0.0), ("dual", 1.0)]


def load_test_windows():
    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS,
                  n_hist=N_HIST, horizon=HORIZON)
    return make_windows(frame, y_range=(TEST_START, TEST_END), **win_kw)


def forecast_quantiles(model, sc, x_hist_day, x_fut_day):
    model.eval()
    with torch.no_grad():
        xh = normalise_hist(np.asarray(x_hist_day), sc)
        xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
        xf = torch.as_tensor(np.asarray(x_fut_day), dtype=torch.float32, device=device).unsqueeze(0)
        q_norm = model(xh, xf)
        q_phys = denormalise_y(q_norm, sc)
    return q_phys.squeeze(0).to(torch.float64)


def crps_per_day(y_true, quantiles, levels=QUANTILE_LEVELS):
    """Quantile-based CRPS approximation, PHYSICAL units (MW), summed over the 24-hour
    delivery window -- matches this script's per-day-TOTAL convention for every other
    metric (abs_dev_MWh, throughput_MWh, etc. are also day-totals, not hourly means).

    CRPS(F, y) ~= 2 * integral_0^1 pinball_tau(y, F^-1(tau)) dtau, approximated here as
    2 * mean over the Q quantile LEVELS of the pinball loss at that level, per hour, then
    summed over the K=24 hours. This is the standard textbook quantile-CRPS estimator
    (properly normalised, with the 2x factor) -- NOT the same convention as this codebase's
    training-time "val_pinball" (which is an unnormalised sum over K*Q with no 2x factor,
    used only as a relative/internal training signal). If you need a number directly
    comparable to val_pinball, drop the "2.0 *" and the "/ Q" mean here.

    y_true    : (K,) realised values, physical units.
    quantiles : (K, Q) forecast quantiles, physical units (same units as y_true).
    levels    : (Q,) quantile levels in (0,1).
    Returns   : scalar, physical units, summed over the K hours.
    """
    y_true = np.asarray(y_true, float)
    quantiles = np.asarray(quantiles, float)
    levels = np.asarray(levels, float)
    e = y_true[:, None] - quantiles                        # (K, Q)
    q = levels[None, :]                                     # (1, Q)
    pinball = np.maximum(q * e, (q - 1.0) * e)               # (K, Q)
    crps_per_hour = 2.0 * pinball.mean(axis=1)                # (K,)
    return float(crps_per_hour.sum())


def precompute_test_oracle_costs(windows, price_model, phys_fp, n_days=None):
    ob = build_oracle(phys_fp, price_model, objective="economic")
    n = n_days if n_days is not None else len(windows.delivery_start)
    costs = np.empty(n)
    for d in range(n):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        v = oracle_price_values(price_day, price_model, realised)
        costs[d] = solve_oracle(ob, v, solver=cp.GUROBI)
    return costs


def evaluate_model(model, sc, sampler, windows, oracle_costs, price_model, k, n_days=None):
    fp = default_fixed_params(k, num_scenarios=N_SCEN, gamma=GAMMA)
    bundle = build_problem(fp, price_model)
    n = n_days if n_days is not None else len(windows.delivery_start)

    rows = []
    for d in range(n):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        q_phys = forecast_quantiles(model, sc, windows.x_hist[d], windows.x_fut[d])
        mean, xi = sampler.mean_and_errors(q_phys)
        prices = get_prices(price_day, price_model)

        vals = {"pl_hat": mean.detach().cpu().numpy(), "xi_samples": xi.detach().cpu().numpy()}
        if k > 0.0:
            # k=1's tracking term needs Sigma_xi_chol, not xi_samples -- see
            # dispatch_layer.build_problem. No grad needed at test time (plain Gurobi solve).
            vals["Sigma_xi_chol"] = cholesky_of_second_moment(xi).detach().cpu().numpy()
        for kk, vv in prices.items():
            vals[kk] = np.asarray(vv, float)

        try:
            dec = solve_plain(bundle, vals, solver=cp.GUROBI)
        except RuntimeError as e:
            print(f"    day {d} solve_plain failed, skipping: {e}")
            continue

        bd = realised_breakdown(fp, dec["p_ch_hat"], dec["p_dis_hat"], dec["D_ch"], dec["D_dis"],
                                dec["p_da_rel"], realised=realised, pl_hat=vals["pl_hat"],
                                price_model=price_model, clip_recourse=True, **prices)
        m = per_day_metrics(fp, bd, oracle_costs[d])
        m["crps"] = crps_per_day(realised, q_phys.detach().cpu().numpy())
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


def summarise(df: pd.DataFrame, source: str, price_model: str, k: float) -> dict:
    if df.empty:
        return {"source": source, "price_model": price_model, "k": k, "n_days": 0}
    return {
        "source": source, "price_model": price_model, "k": k, "n_days": len(df),
        "mean_total_cost":  df["total_cost"].mean(),
        "mean_regret":      df["regret"].mean(),
        "mean_crps":        df["crps"].mean(),
        "mean_abs_dev_MWh": df["abs_dev_MWh"].mean(),
        "mean_C_da":        df["C_da"].mean(),
        "mean_C_imb":       df["C_imb"].mean(),
        "total_sat_hours":  df["sat_hours"].sum(),
        "total_sat_MWh":    df["sat_MWh"].sum(),
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

    phys_fp = default_fixed_params(0.0, num_scenarios=N_SCEN, gamma=GAMMA)
    oracle_cache = {}   # price_model -> costs array (oracle doesn't depend on k)

    summary_rows = []
    for price_model, k in CORNERS:
        print(f"\n=== corner: price_model={price_model} k={int(k)} ===")
        if price_model not in oracle_cache:
            print(f"  precomputing oracle costs ({price_model})...")
            oracle_cache[price_model] = precompute_test_oracle_costs(windows, price_model, phys_fp, n)
        oracle_costs = oracle_cache[price_model]

        print("  evaluating baseline...")
        df_base = evaluate_model(baseline_model, sc, sampler, windows, oracle_costs, price_model, k, n)
        df_base.to_csv(RESULTS_DIR / f"baseline_{price_model}_k{int(k)}.csv", index=False)
        summary_rows.append(summarise(df_base, "baseline", price_model, k))

        dfl_path = DFL_TRAIN_DIR / f"dfl_{price_model}_k{int(k)}.pt"
        if dfl_path.exists():
            print(f"  evaluating DFL checkpoint ({dfl_path.name})...")
            dfl_ckpt = torch.load(dfl_path, weights_only=False, map_location="cpu")
            dfl_model = dfl_model_from(baseline_ckpt, dfl_ckpt)
            df_dfl = evaluate_model(dfl_model, sc, sampler, windows, oracle_costs, price_model, k, n)
            df_dfl.to_csv(RESULTS_DIR / f"dfl_{price_model}_k{int(k)}.csv", index=False)
            summary_rows.append(summarise(df_dfl, "dfl", price_model, k))
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
