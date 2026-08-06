"""
Numpy-only validation of the load-bearing algebra in dispatch_layer.py.
Does NOT import cvxpy/torch (absent in this sandbox). Re-implements each formulation
identity in numpy and checks it holds. is_dcp / actual solves must be run in the user's env.
"""
import ast
import numpy as np
from pathlib import Path
from pyprojroot import here
import sys


rng = np.random.default_rng(0)
T, N = 6, 64                     # small T for exhaustive vertex checks
DT = 1.0
ok = lambda name, cond: print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------- 0. syntax
def check_parse():
    with open(DISPATCH_DIR) as f:
        src = f.read()
    ast.parse(src)                                # raises on syntax error
    print("  [PASS] dispatch_layer.py parses (ast)")


# ---------------------------------------------------------------- 1. robustification
def worst_case(A0, A, h_plus, h_minus):
    """max over xi in [-h_minus, h_plus]^T of  A0 + A @ xi, per row."""
    pos = np.maximum(A, 0.0) @ h_plus            # a_i>0 pushed to +h_plus
    neg = np.maximum(-A, 0.0) @ h_minus          # a_i<0 pushed to -h_minus
    return A0 + pos + neg

def check_robustify():
    A0 = rng.normal(size=T)
    A  = rng.normal(size=(T, T))
    h_plus  = np.abs(rng.normal(size=T)) + 0.1
    h_minus = np.abs(rng.normal(size=T)) + 0.1
    # reformulation LHS with tightest duals mu_p=pos(A), mu_m=neg(A):
    mu_p, mu_m = np.maximum(A, 0.0), np.maximum(-A, 0.0)
    refm = A0 + mu_p @ h_plus + mu_m @ h_minus
    wc   = worst_case(A0, A, h_plus, h_minus)
    ok("robust reformulation == analytic worst case", np.allclose(refm, wc, atol=1e-12))
    # vertex sampling: analytic worst >= value at every box vertex (check a random subset)
    verts_ok = True
    for _ in range(2000):
        xi = np.where(rng.random(T) < 0.5, -h_minus, h_plus)
        verts_ok &= np.all(A0 + A @ xi <= wc + 1e-9)
    ok("analytic worst dominates all sampled box vertices", verts_ok)


# ---------------------------------------------------------------- 2. cholesky identity
def check_cholesky():
    xi = rng.normal(size=(N, T)); xi -= xi.mean(0, keepdims=True)   # mean-centred
    M = xi.T @ xi / N
    M = 0.5 * (M + M.T) + 1e-9 * np.eye(T)
    Lc = np.linalg.cholesky(M)
    R = np.eye(T) + rng.normal(size=(T, T))
    lhs = np.sum((R @ Lc) ** 2)                                     # cp.sum_squares(R @ L)
    rhs = sum(R[t] @ M @ R[t] for t in range(T))                   # sum_t r_t^T M r_t
    ok("sum_squares(R @ L) == sum_t r_t^T M r_t", np.isclose(lhs, rhs, atol=1e-9))


# ---------------------------------------------------------------- 3. tracking exactness
def check_tracking_crossterm():
    """Full SAA of E[(Delta + R xi)^2] equals ||Delta||^2 + sum_t r_t^T M r_t EXACTLY,
    because cross term vanishes for mean-centred xi over the SAME N used for M."""
    xi = rng.normal(size=(N, T)); xi -= xi.mean(0, keepdims=True)
    M = xi.T @ xi / N
    R = np.eye(T) + rng.normal(size=(T, T)) * 0.3
    Delta = rng.normal(size=T)
    saa = np.mean([np.sum((Delta + R @ xi[s]) ** 2) for s in range(N)])
    reform = np.sum(Delta ** 2) + sum(R[t] @ M @ R[t] for t in range(T))
    ok("SAA tracking == ||Delta||^2 + sum_t r_t^T M r_t (cross term = 0)",
       np.isclose(saa, reform, atol=1e-10))


# ---------------------------------------------------------------- 4. epigraph reconstruction
def check_epigraph():
    xi = rng.normal(size=(N, T))
    R = np.eye(T) + rng.normal(size=(T, T)) * 0.3
    Delta = rng.normal(size=T)
    imb_scen = (Delta.reshape(T, 1) @ np.ones((1, N))) + R @ xi.T   # (T, N), as in the code
    direct   = np.stack([Delta + R @ xi[s] for s in range(N)], axis=1)
    ok("epigraph imb_scen == per-scenario Delta + R xi", np.allclose(imb_scen, direct))


# ---------------------------------------------------------------- 5. single analytic == SAA
def check_single_analytic():
    xi = rng.normal(size=(N, T)); xi -= xi.mean(0, keepdims=True)
    R = np.eye(T) + rng.normal(size=(T, T)) * 0.3
    Delta = rng.normal(size=T)
    pi_imb = rng.normal(size=T)                                     # SIGNED price
    saa = np.mean([pi_imb @ (Delta + R @ xi[s]) for s in range(N)]) * DT
    analytic = pi_imb @ Delta * DT
    ok("single E[pi_imb . p_imb] == pi_imb . Delta (R xi vanishes)",
       np.isclose(saa, analytic, atol=1e-12))


# ---------------------------------------------------------------- 6. free-bid identity
def check_free_bid():
    pl = rng.normal(size=T); pch = rng.normal(size=T); pdis = rng.normal(size=T)
    p_da_rel = rng.normal(size=T)
    g_hat = pl + pch - pdis
    p_da  = p_da_rel + pl
    Delta = pch - pdis - p_da_rel
    ok("g_hat - p_da == Delta (pl_hat cancels)", np.allclose(g_hat - p_da, Delta))
    # dropped DA constant is exactly pi_da . pl:
    pi_da = rng.normal(size=T)
    full  = pi_da @ p_da
    kept  = pi_da @ p_da_rel
    ok("dropped DA constant == pi_da . pl_hat", np.isclose(full - kept, pi_da @ pl))


# ---------------------------------------------------------------- 7. bid-bound reparam
def check_bid_bounds():
    pl = rng.normal(size=T); p_min, p_max = -5.0, 11.0
    lo, hi = p_min - pl, p_max - pl                                 # bounds on p_da_rel
    p_da_rel = np.clip(rng.normal(size=T), lo, hi)
    p_da = p_da_rel + pl
    ok("p_da_rel in [p_min-pl, p_max-pl] <=> p_da in [p_min, p_max]",
       np.all(p_da >= p_min - 1e-9) and np.all(p_da <= p_max + 1e-9))


if __name__ == "__main__":

    ROOT_DIR = here()
    MODELS_DIR = ROOT_DIR / "6_models"
    DISPATCH_DIR = MODELS_DIR / "dispatch_layer.py"

    print("Formulation validation (numpy re-implementation):")

    check_parse()
    check_robustify()
    check_cholesky()
    check_tracking_crossterm()
    check_epigraph()
    check_single_analytic()
    check_free_bid()
    check_bid_bounds()
    print("Done. (is_dcp(dpp=True) and real Gurobi/layer solves must run in your env.)")
