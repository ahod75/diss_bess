"""train_dfl_models.py -- trains EVERY DFL surrogate corner in one process, replacing
both the former train_corner.py (single-corner CLI, one architecture/mode per
invocation) and run_all_corners.py (subprocess-per-corner orchestrator). All helper
functions/constants live in dfl_train_utils.py; this file is just the corner list plus
the loop that runs them, so training data/baseline model/boxes -- identical across every
corner -- are loaded and computed exactly ONCE instead of once per corner.

All training here runs on CPU only -- no CUDA anywhere in this pipeline.

In-process instead of subprocess-per-corner: run_all_corners.py deliberately gave each
corner its own process (own memory context; one corner's crash/OOM couldn't take down
the rest). Going in-process trades that isolation for the shared-data speedup above and
simpler code, so two things replace it here:

  1. Per-corner try/except -- a corner that raises (including a memory error mid-
     backward(), which happens OUTSIDE dfl_loss_batch's own per-batch try/except and so
     would otherwise propagate uncaught) is logged as FAILED and the loop moves on,
     matching run_all_corners.py's "a failure in one corner does NOT stop the rest"
     guarantee as closely as achievable without real process isolation.
  2. gc.collect() after EVERY corner (success or failure) -- matters specifically
     because the autograd graph <-> cvxpylayers' CvxpyLayer/cvxpy Problem wiring can
     form reference cycles plain refcounting won't clear on its own, which would
     otherwise let each corner's memory footprint accumulate on top of the last one's
     instead of being released. It does NOT fix dual-price's own memory-pressure risk,
     though -- that's a within-corner batch-size/problem-size issue (the N=64-scenario
     epigraph), unrelated to what ran before it.

Usage: python train_dfl_models.py
"""
from __future__ import annotations
import gc
import time
import pickle

import pandas as pd
import torch

from dfl_train_utils import (
    ROOT_DIR, FORECASTING_DIR, DATA_DIR, COPULA_DIR, DFL_TRAIN_DIR,
    reindex_and_impute, build_features, make_windows,
    normalise_hist, normalise_exo, denormalise_y,
    Baseline_Forecaster, TRAIN_START, VAL_START, TEST_START,
    HIST_COLS, FEAT_COLS, EXO_COLS, FrozenCopulaSampler,
    PRICE_COLS_ALL, ISSUE_HOUR, HORIZON, N_HIST, device, GAMMA,
    TrainConfig, make_fp, precompute_boxes, train_one_config,
)

LOG_DIR = DFL_TRAIN_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# (architecture, mode). Cheap-first ordering (matches the former run_all_corners.py):
# 1stage first as a cheap guaranteed fallback, then point_robust ascending cost
# (single-price cheapest; dispatchability close behind, no epigraph; dual-price most
# expensive, driven by the N=64-scenario epigraph -- see 6_models/param_sweeps/
# gamma_sweep.py, which profiles exactly this corner).
SURROGATES = [
    ("1stage",       "single-price"),
    ("1stage",       "dual-price"),
    ("point_robust", "single-price"),
    ("point_robust", "dispatchability"),
    ("point_robust", "dual-price"),
]


def load_data():
    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS_ALL,
                  n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    val_windows   = make_windows(frame, y_range=(VAL_START, TEST_START - pd.Timedelta(hours=1)), **win_kw)
    return train_windows, val_windows


def main():
    print(f"device={device}  surrogates={SURROGATES}")

    print("loading training data (once, shared across every corner)...")
    train_windows, val_windows = load_data()
    print(f"n_train={len(train_windows.delivery_start)}  n_val={len(val_windows.delivery_start)}")

    ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    sc = ckpt["scaler_stats"]

    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)
    fwd = {"normalise_hist": normalise_hist, "denormalise_y": denormalise_y, "normalise_exo": normalise_exo}

    # FROZEN baseline copy + boxes: identical for every point_robust corner regardless
    # of mode (same baseline weights, same sampler) -- computed ONCE here instead of
    # once per corner, unlike the former train_corner.py (one process per corner, so no
    # opportunity to share this).
    baseline_model = Baseline_Forecaster(**ckpt["model_config"])
    baseline_model.load_state_dict(ckpt["state_dict"])
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)
    print("precomputing frozen baseline boxes (train)...")
    boxes_train = precompute_boxes(train_windows, baseline_model, sc, sampler, cop["quantile_levels"], device, fwd)
    print("precomputing frozen baseline boxes (val)...")
    boxes_val = precompute_boxes(val_windows, baseline_model, sc, sampler, cop["quantile_levels"], device, fwd)

    results = []
    for architecture, mode in SURROGATES:
        t0 = time.time()
        print(f"\nCORNER_START architecture={architecture} mode={mode}", flush=True)

        # Everything corner-specific -- including make_fp/TrainConfig construction, not
        # just train_one_config itself -- lives inside this try block: an invalid
        # architecture/mode combination or any other setup failure must be caught here
        # too, or it defeats the whole point of per-corner isolation (confirmed by a
        # deliberately-broken corner during testing -- make_fp raising outside the try
        # previously killed the entire run).
        model = fp = cfg = trained_model = None
        try:
            model = Baseline_Forecaster(**ckpt["model_config"])
            model.load_state_dict(ckpt["state_dict"])   # FRESH copy of the baseline weights every corner --
            model = model.to(device)                     # never chained from a previous corner's fine-tuning

            fp = make_fp(architecture, GAMMA)
            b_train = boxes_train if architecture == "point_robust" else None
            b_val = boxes_val if architecture == "point_robust" else None

            cfg = TrainConfig(architecture=architecture, mode=mode)
            print(f"batch_size={cfg.batch_size}")
            out_path = DFL_TRAIN_DIR / f"dfl_{architecture}_{mode}.pt"

            trained_model, best_val, hist = train_one_config(
                cfg, model=model, fp=fp, sampler=sampler, sc=sc,
                train_windows=train_windows, val_windows=val_windows,
                boxes_train=b_train, boxes_val=b_val, device=device, fwd=fwd, out_path=out_path)
            elapsed = time.time() - t0
            # Per-corner forward/backward totals, summed from each epoch's own timing
            # in `hist` (dfl_train_utils.train_one_config logs t_forward_train/
            # t_backward_train/t_forward_val per epoch) -- not measured separately here.
            # len(hist) == epochs actually run (one entry appended per epoch), so the
            # per-epoch average is just the cumulative total divided by that.
            n_epochs = len(hist)
            t_fwd_train = sum(h["t_forward_train"] for h in hist)
            t_bwd_train = sum(h["t_backward_train"] for h in hist)
            t_fwd_val = sum(h["t_forward_val"] for h in hist)
            print(f"\nCORNER_DONE architecture={architecture} mode={mode} best_val_fsurr={best_val:.4f} "
                  f"epochs_run={n_epochs} elapsed_s={elapsed:.0f} "
                  f"t_fwd_train_s={t_fwd_train:.1f} (avg {t_fwd_train/n_epochs:.2f}/epoch)  "
                  f"t_bwd_train_s={t_bwd_train:.1f} (avg {t_bwd_train/n_epochs:.2f}/epoch)  "
                  f"t_fwd_val_s={t_fwd_val:.1f} (avg {t_fwd_val/n_epochs:.2f}/epoch)",
                  flush=True)
            torch.save({"state_dict": trained_model.state_dict(), "cfg": cfg.__dict__,
                        "best_val_fsurr": best_val, "val_history": hist}, out_path)
            print(f"CORNER_SAVED {out_path}")
            results.append((architecture, mode, "OK", elapsed, t_fwd_train, t_bwd_train))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"CORNER_FAILED architecture={architecture} mode={mode} error={e!r} "
                  f"elapsed_s={elapsed:.0f} -- moving on to the next corner", flush=True)
            results.append((architecture, mode, f"FAILED({e!r})", elapsed, None, None))

        del model, trained_model, fp, cfg
        gc.collect()

    print("\n" + "=" * 70)
    print("PIPELINE_SUMMARY")
    for architecture, mode, status, elapsed, t_fwd_train, t_bwd_train in results:
        timing = f"t_fwd={t_fwd_train:.1f}s t_bwd={t_bwd_train:.1f}s" if t_fwd_train is not None else "t_fwd=n/a t_bwd=n/a"
        print(f"  {architecture:14s} {mode:16s} {status:30s} {elapsed:8.0f}s  {timing}")


if __name__ == "__main__":
    main()
