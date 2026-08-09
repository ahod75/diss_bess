"""Train the forecaster against the POINT-CONSTRAINED 2-stage dispatch model
(dispatch_layer_point_robust.py) -- feasibility grounded at 3 literal points (nominal,
+h_plus, -h_minus) instead of the full LP-duality robust reformulation. See
dispatch_layer_point_robust.py's module docstring for the full justification, and
h_selection_sweep.ipynb for the box calibration (locked in: (0.15, 0.85), which
dominated the previous (0.20,0.80) on both this model and the full robust one).

Four corners: {single, dual} x {k=0, k=1}. Unlike the 1-stage model, D_ch/D_dis are REAL
recourse matrices here, so saturation CAN genuinely occur -- tracked and logged.

Box: computed ONCE per day from the FROZEN baseline forecaster (never recomputed from
the model currently being trained -- see dispatch_wrapper_robust.compute_box's docstring
for why: a model that learns narrower quantiles shouldn't get to shrink its own robust
box for free).

Sigma_xi_chol (k>0): uses dispatch_wrapper.cholesky_of_second_moment (torch, differentiable
-- NOT dispatch_wrapper_robust's internal numpy path via make_dispatch_inputs, which
detaches xi and would silently zero the k=1 gradient back to the forecaster).

Usage: python train_corner_point_robust.py <price_model: single|dual> <k: 0|1>
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import pandas as pd
import torch
import cvxpy as cp
import pickle

import sys
import time
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
MODELS_ROBUST_DIR = MODEL_DIR / "models_robust"
DFL_TRAIN_DIR = ROOT_DIR / "7_model_training"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODELS_ROBUST_DIR))

from forecasting import (reindex_and_impute, build_features, make_windows,
                         normalise_hist, normalise_exo, denormalise_y, normalise_y,
                         pinball_loss, QUANTILE_LEVELS, Baseline_Forecaster,
                         TRAIN_START, VAL_START, TEST_START,
                         HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler
from dispatch_layer_point_robust import default_fixed_params, build_problem, make_layer
from dispatch_layer import build_oracle, solve_oracle   # oracle is k/D_ch-independent, reuse as-is
from dispatch_wrapper import (get_prices, oracle_price_values, realised_breakdown,
                              per_day_metrics, cholesky_of_second_moment)
from dispatch_wrapper_robust import compute_box   # box helper only -- NOT make_dispatch_inputs

PRICE_COLS = ["da", "imb", "up_reg_cost", "down_reg_cost"]
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 1e-4
BOX_LEVELS = (0.15, 0.85)   # locked in from h_selection_sweep.ipynb
TRAIN_SOLVER = "ECOS"
ORACLE_SOLVER = cp.GUROBI
QUANTILE_LEVELS_TENSOR = torch.as_tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
EPS_BALANCE = 1e-6
FALLBACK_SOLVER = "SCS"
SAT_TOL = 1e-6


@dataclass
class TrainConfig:
    price_model: str
    k: float
    lr: float = 5e-4
    batch_size: int = 16
    max_epochs: int = 10
    patience: int = 5
    min_delta: float = 0.02
    grad_clip: float = 3.0
    seed: int = 20240801


def forecast_train(model, sc, x_hist_day, x_fut_day, device, normalise_hist, denormalise_y,
                    normalise_exo):
    xh = normalise_hist(np.asarray(x_hist_day), sc)
    xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
    xf = normalise_exo(np.asarray(x_fut_day), sc)
    xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
    q_norm = model(xh, xf)
    q_phys = denormalise_y(q_norm, sc)
    return q_phys.squeeze(0).to(torch.float64), q_norm.squeeze(0)


def pinball_per_day(y_true_norm, q_norm, levels):
    e = y_true_norm.unsqueeze(-1) - q_norm
    q = levels.view(1, 1, -1)
    return torch.maximum(q * e, (q - 1.0) * e).sum(dim=(1, 2)).to(torch.float64)


def self_balanced_loss(L_base, f_dfl, eps=EPS_BALANCE):
    denom = L_base.detach() + f_dfl.detach() + eps
    alpha = f_dfl.detach() / denom
    beta = 1.0 - alpha
    return alpha * L_base + beta * f_dfl


def solve_with_retry(layer, args):
    try:
        return layer(*args, solver_args={"solve_method": TRAIN_SOLVER})
    except cvxpylayers_solver_error():
        return layer(*args, solver_args={"solve_method": FALLBACK_SOLVER})


def cvxpylayers_solver_error():
    from diffcp import SolverError
    return SolverError


def precompute_oracle_costs(windows, price_model, phys_fp):
    ob = build_oracle(phys_fp, price_model, objective="economic")
    costs = np.empty(len(windows.delivery_start))
    for d in range(len(windows.delivery_start)):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        v = oracle_price_values(price_day, price_model, realised)
        costs[d] = solve_oracle(ob, v, solver=ORACLE_SOLVER)
    return costs


def precompute_boxes(windows, baseline_model, sc, sampler, quantile_levels, device, fwd):
    """FROZEN baseline box (B1), computed ONCE per day from the frozen baseline
    forecaster -- never recomputed from the model currently being trained."""
    boxes = []
    baseline_model.eval()
    with torch.no_grad():
        for d in range(len(windows.delivery_start)):
            xh = fwd["normalise_hist"](np.asarray(windows.x_hist[d]), sc)
            xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
            xf = fwd["normalise_exo"](np.asarray(windows.x_fut[d]), sc)
            xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
            q_norm = baseline_model(xh, xf)
            q_phys = fwd["denormalise_y"](q_norm, sc).squeeze(0).to(torch.float64)
            h_plus, h_minus = compute_box(q_phys, sampler, quantile_levels, box_levels=BOX_LEVELS)
            boxes.append((h_plus.detach().cpu().numpy(), h_minus.detach().cpu().numpy()))
    return boxes


def build_layer_vals(prices, h_plus, h_minus, xi_samples=None, Sigma_xi_chol=None):
    vals = {**prices, "h_plus": h_plus, "h_minus": h_minus}
    if xi_samples is not None:
        vals["xi_samples"] = xi_samples
    if Sigma_xi_chol is not None:
        vals["Sigma_xi_chol"] = Sigma_xi_chol
    return vals


def dfl_loss_batch(batch_indices, *, model, fp, sampler, sc, windows, layer, keys, price_model, k,
                   device, fwd, oracle_costs, boxes):
    realised  = np.asarray(windows.y[batch_indices], float)
    price_day = np.asarray(windows.price[batch_indices], float)
    x_hist = windows.x_hist[batch_indices]
    x_fut = windows.x_fut[batch_indices]
    B = len(batch_indices)
    T = fp.T_total

    means_list, xi_list, q_norm_list = [], [], []
    for i in range(B):
        q_phys_i, q_norm_i = forecast_train(model, sc, x_hist[i], x_fut[i], device,
                                     fwd["normalise_hist"], fwd["denormalise_y"], fwd["normalise_exo"])
        mean_i, xi_i = sampler.mean_and_errors(q_phys_i)
        means_list.append(mean_i); xi_list.append(xi_i); q_norm_list.append(q_norm_i)
    mean = torch.stack(means_list, dim=0)
    xi = torch.stack(xi_list, dim=0)
    q_norm_batch = torch.stack(q_norm_list, dim=0)

    h_plus = torch.stack([torch.as_tensor(boxes[d][0], dtype=mean.dtype, device=device) for d in batch_indices])
    h_minus = torch.stack([torch.as_tensor(boxes[d][1], dtype=mean.dtype, device=device) for d in batch_indices])

    prices = get_prices(price_day, price_model)
    prices_t = {kk: torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
                for kk, vv in prices.items()}

    xi_samples = xi if (k < 1.0 and price_model == "dual") else None
    Sigma_xi_chol = cholesky_of_second_moment(xi) if k > 0.0 else None

    vals = build_layer_vals(prices_t, h_plus, h_minus, xi_samples=xi_samples, Sigma_xi_chol=Sigma_xi_chol)
    args = [vals[name] for name in keys]
    y_true_norm = torch.as_tensor(normalise_y(realised, sc), dtype=torch.float32, device=device)

    def _combine(dec, realised_s, mean_s, prices_s, q_norm_s, y_true_s, oracle_s):
        L_base = pinball_per_day(y_true_s, q_norm_s, QUANTILE_LEVELS_TENSOR)
        p_ch_hat, p_dis_hat, D_ch, D_dis = dec
        p_da_rel = p_ch_hat - p_dis_hat   # reconstructed -- not a layer output (evicted for speed)
        bd = realised_breakdown(fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
                                realised=realised_s, pl_hat=mean_s, price_model=price_model,
                                clip_recourse=True, **prices_s)
        raw_cost = bd.C_da + bd.C_imb
        oracle_t = torch.as_tensor(oracle_s, dtype=raw_cost.dtype, device=raw_cost.device)
        f_dfl = torch.clamp(raw_cost - oracle_t, min=0.0)
        clip_ch = torch.abs(bd.p_ch_raw - bd.p_ch_r)
        clip_dis = torch.abs(bd.p_dis_raw - bd.p_dis_r)
        sat_h = int(((clip_ch > SAT_TOL) | (clip_dis > SAT_TOL)).sum().item())
        return self_balanced_loss(L_base, f_dfl), L_base, f_dfl, sat_h

    try:
        dec = solve_with_retry(layer, args)
        oracle_arr = np.asarray([oracle_costs[d] for d in batch_indices], float)
        per_day, L_base, f_dfl, sat_h = _combine(dec, realised, mean, prices_t, q_norm_batch,
                                                y_true_norm, oracle_arr)
        return (per_day.sum(), B, sat_h, B * T,
                float(L_base.sum()), float(f_dfl.sum()))
    except cvxpylayers_solver_error():
        pass

    total = 0.0; n_survived = 0; L_base_total = 0.0; f_dfl_total = 0.0; sat_total = 0
    for i in range(B):
        args_i = [v[i:i + 1] for v in args]
        try:
            dec_i = solve_with_retry(layer, args_i)
        except cvxpylayers_solver_error():
            print(f"  SKIPPING day {batch_indices[i]}")
            continue
        oracle_i = np.asarray([oracle_costs[batch_indices[i]]], float)
        prices_i = {kk: v[i:i + 1] for kk, v in prices_t.items()}
        loss_i, L_base_i, f_dfl_i, sat_i = _combine(dec_i, realised[i:i + 1], mean[i:i + 1], prices_i,
                          q_norm_batch[i:i + 1], y_true_norm[i:i + 1], oracle_i)
        total = total + loss_i.sum()
        L_base_total += float(L_base_i.sum()); f_dfl_total += float(f_dfl_i.sum())
        sat_total += sat_i
        n_survived += 1
    if n_survived == 0:
        raise RuntimeError(f"all {B} days failed with both ECOS and SCS")
    return total, n_survived, sat_total, n_survived * T, L_base_total, f_dfl_total


def evaluate_regret(*, model, fp, sampler, sc, windows, oracle_costs, boxes, layer, keys,
                    price_model, k, device, fwd):
    model.eval()
    tot_combined = 0.0; tot_fsurr = 0.0; tot_base = 0.0; n_ok = 0; n_skipped = 0
    tot_sat_h = 0; tot_n_h = 0
    with torch.no_grad():
        for d in range(len(windows.delivery_start)):
            try:
                combined, n_survived, sat_h, n_h, L_base_sum, f_dfl_sum = dfl_loss_batch(
                    [d], model=model, fp=fp, sampler=sampler, sc=sc, windows=windows,
                    layer=layer, keys=keys, price_model=price_model, k=k, device=device,
                    fwd=fwd, oracle_costs=oracle_costs, boxes=boxes)
            except RuntimeError:
                n_skipped += 1
                continue
            tot_combined += float(combined); tot_fsurr += f_dfl_sum; tot_base += L_base_sum
            tot_sat_h += sat_h; tot_n_h += n_h
            n_ok += 1

    sat_frac = tot_sat_h / tot_n_h if tot_n_h > 0 else 0.0
    return tot_combined / n_ok, tot_fsurr / n_ok, tot_base / n_ok, n_skipped, sat_frac


def train_one_config(cfg: TrainConfig, *, model, fp, sampler, sc, train_windows, val_windows,
                     oracle_costs_train, oracle_costs_val, boxes_train, boxes_val, device, fwd):
    torch.manual_seed(cfg.seed)
    bundle = build_problem(fp, cfg.price_model)
    layer = make_layer(bundle)
    keys = [p.name() for p in bundle.params]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_train = len(train_windows.delivery_start)
    best_val = float("inf"); best_state = copy.deepcopy(model.state_dict()); patience = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        order = np.random.permutation(n_train)
        grad_norms = []
        sat_hours_epoch = 0; n_hours_epoch = 0
        for start in range(0, n_train, cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            opt.zero_grad()
            try:
                batch_loss, n_survived, sat_h, n_h, _, _ = dfl_loss_batch(batch.tolist(), model=model,
                                    fp=fp, sampler=sampler, sc=sc, windows=train_windows, layer=layer,
                                    keys=keys, price_model=cfg.price_model, k=cfg.k, device=device,
                                    fwd=fwd, oracle_costs=oracle_costs_train, boxes=boxes_train)
            except RuntimeError as e:
                print(f"  SKIPPING whole batch (start={start}): {e}")
                continue
            (batch_loss / n_survived).backward()
            if cfg.grad_clip is not None:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norms.append(float(pre_clip_norm))
            opt.step()
            sat_hours_epoch += sat_h; n_hours_epoch += n_h

        val_combined, val_fsurr, val_base, n_skipped, val_sat_frac = evaluate_regret(
            model=model, fp=fp, sampler=sampler, sc=sc, windows=val_windows,
            oracle_costs=oracle_costs_val, boxes=boxes_val, layer=layer, keys=keys,
            price_model=cfg.price_model, k=cfg.k, device=device, fwd=fwd)
        grad_norms_arr = np.asarray(grad_norms) if grad_norms else np.asarray([0.0])
        train_sat_frac = sat_hours_epoch / n_hours_epoch if n_hours_epoch > 0 else 0.0
        history.append({"val_combined": val_combined, "val_fsurr": val_fsurr,
                        "val_base": val_base, "n_skipped": n_skipped,
                        "grad_norm_mean": float(grad_norms_arr.mean()),
                        "grad_norm_max": float(grad_norms_arr.max()),
                        "train_sat_frac": train_sat_frac, "val_sat_frac": val_sat_frac})
        improved = val_fsurr < best_val * (1.0 - cfg.min_delta)
        print(f"[{cfg.price_model} k={cfg.k} point_robust] epoch {epoch:2d}  "
              f"val_fsurr={val_fsurr:.4f}  val_pinball={val_base:.4f}  "
              f"val_combined={val_combined:.4f}  skipped={n_skipped}  "
              f"grad_norm(mean/max)={grad_norms_arr.mean():.3f}/{grad_norms_arr.max():.3f}  "
              f"sat_frac(train/val)={train_sat_frac:.3f}/{val_sat_frac:.3f}  "
              f"{'*best' if improved else f'(patience {patience+1}/{cfg.patience})'}", flush=True)
        if improved:
            best_val = val_fsurr; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"  early stop at epoch {epoch} (best val_fsurr={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val, history


if __name__ == "__main__":
    pm_arg = sys.argv[1]
    k_arg = float(sys.argv[2])
    assert pm_arg in ("single", "dual")
    assert k_arg in (0.0, 1.0)

    t0 = time.time()
    print(f"CORNER_START price_model={pm_arg} k={k_arg} (point_robust)", flush=True)

    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS,
                  n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    val_windows   = make_windows(frame, y_range=(VAL_START, TEST_START - pd.Timedelta(hours=1)), **win_kw)
    print(f"n_train={len(train_windows.delivery_start)}  n_val={len(val_windows.delivery_start)}")

    ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    model = Baseline_Forecaster(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    sc = ckpt["scaler_stats"]

    # SEPARATE frozen copy, used ONLY for box computation -- never updated during training.
    baseline_model = Baseline_Forecaster(**ckpt["model_config"])
    baseline_model.load_state_dict(ckpt["state_dict"])
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)

    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)

    fp = default_fixed_params(k=k_arg, gamma=GAMMA)
    fwd = {"normalise_hist": normalise_hist, "denormalise_y": denormalise_y, "normalise_exo": normalise_exo}

    print("precomputing oracle costs (train)...")
    oracle_train = precompute_oracle_costs(train_windows, pm_arg, fp)
    print("precomputing oracle costs (val)...")
    oracle_val = precompute_oracle_costs(val_windows, pm_arg, fp)

    print("precomputing frozen baseline boxes (train)...")
    boxes_train = precompute_boxes(train_windows, baseline_model, sc, sampler, cop["quantile_levels"], device, fwd)
    print("precomputing frozen baseline boxes (val)...")
    boxes_val = precompute_boxes(val_windows, baseline_model, sc, sampler, cop["quantile_levels"], device, fwd)

    cfg = TrainConfig(price_model=pm_arg, k=k_arg)
    model, best_val, hist = train_one_config(cfg, model=model, fp=fp, sampler=sampler, sc=sc,
                                              train_windows=train_windows, val_windows=val_windows,
                                              oracle_costs_train=oracle_train, oracle_costs_val=oracle_val,
                                              boxes_train=boxes_train, boxes_val=boxes_val,
                                              device=device, fwd=fwd)

    elapsed = time.time() - t0
    k_tag = "k0" if k_arg == 0.0 else "k1"
    print(f"\nCORNER_DONE price_model={pm_arg} k={k_arg} best_val_fsurr={best_val:.4f} "
          f"epochs_run={len(hist)} elapsed_s={elapsed:.0f}", flush=True)
    out_path = DFL_TRAIN_DIR / f"dfl_point_robust_{pm_arg}_{k_tag}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                "best_val_fsurr": best_val, "val_history": hist}, out_path)
    print(f"CORNER_SAVED {out_path}")
