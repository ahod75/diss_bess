from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from dispatch_shared import _to_t

# =====================================================================================
# dispatch_wrapper.py  --  the realised-metric ENGINE, and nothing else now.
#
#   realised_breakdown(...)     -> the single source of truth for realised quantities
#   realised_cost(...)          -> thin C_da + C_imb wrapper on the breakdown (DFL loss,
#                                   economic-objective corners)
#   realised_imbalance(...)     -> thin dt**2 * sum_squares(p_imb) wrapper on the same
#                                   breakdown (DFL loss, dispatchability corners)
#
# ALL realised quantities are post-saturation and computed against the SAME pl_hat anchor
# the layer used (xi_real = realised - pl_hat). Free bid:  bid = pl_hat + p_da_rel.
#
# Everything that used to live here and wasn't this post-solve engine has moved out:
#   - _to_t, cholesky_of_second_moment, PRICE_COLS, get_prices, oracle_price_values,
#     compute_box -> dispatch_shared.py (pre-solve input shaping, formulation-agnostic,
#     used identically by training and testing). _to_t is now IMPORTED from there --
#     the dependency direction used to run the other way.
#   - per_day_metrics -> 8_testing/evaluate.py (its only real caller; training scripts
#     imported it but never called it, confirmed by grep before the move).
#   - assert_price_consistency, make_dispatch_inputs, money_plot_series, regret() ->
#     removed entirely (zero real callers anywhere in the live pipeline).
#
# `fp: FixedParams` used to be imported from dispatch_layer.py (now oracle.py, since
# FixedParams was dead code there too and got removed). realised_breakdown's `fp` is
# duck-typed across dispatch_setup.FixedParams and FixedParams1Stage -- there was never
# one correct static type for it, so no import/annotation replaces the old one.
# =====================================================================================


# =====================================================================================
# THE ENGINE.  One realised solve -> everything the metrics need.
# Runs in torch (serves the differentiable DFL loss AND the detached test path). At test,
# pass numpy decisions; _to_t lifts them (grad-free) and the reductions return floats.
# =====================================================================================
@dataclass
class RealisedBreakdown:
    p_ch_raw:  torch.Tensor   # (T,) raw recourse charge  = p_ch_hat  + D_ch  @ xi_real
    p_dis_raw: torch.Tensor   # (T,) raw recourse discharge
    p_ch_r:    torch.Tensor   # (T,) realised charge  (post-saturation)
    p_dis_r:   torch.Tensor   # (T,) realised discharge (post-saturation)
    soc:       torch.Tensor   # (T,) realised SOC trajectory (post-action)
    bid:       torch.Tensor   # (T,) committed DA bid = pl_hat + p_da_rel
    p_g:       torch.Tensor   # (T,) realised grid draw = realised + p_ch_r - p_dis_r
    p_imb:     torch.Tensor   # (T,) realised imbalance = p_g - bid
    C_da:      torch.Tensor   # scalar
    C_imb:     torch.Tensor   # scalar (signed single, >=0 dual)

### This function is used for both training AND testing.
# For training, variables need to be passed through as tensors so that Pytorch's autograd
# system can keep track of gradients.
# This is unecessary for testing, but it makes it simpler to keep it all as one function.
# Makes it slightly slower, but keeps it simpler and neater.
def realised_breakdown(
    fp,   # duck-typed: dispatch_setup.FixedParams or FixedParams1Stage, never checked by type
    p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,   # the 5 decisions from the layer / solve
    realised,                                      # (T,) realised prosumption for the day
    pl_hat,                                        # (T,) SAME anchor the layer used
    price_model,                                   # "single" | "dual"
    pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None,
    clip_recourse=True,                            # test: True; correctness gate uses False
) -> RealisedBreakdown:
    p_ch_hat = torch.as_tensor(p_ch_hat) if isinstance(p_ch_hat, torch.Tensor) \
           else torch.as_tensor(np.asarray(p_ch_hat, np.float64))
    p_dis_hat = _to_t(p_dis_hat, like=p_ch_hat)
    D_ch      = _to_t(D_ch, like=p_ch_hat)
    D_dis = _to_t(D_dis, like=p_ch_hat)
    p_da_rel  = _to_t(p_da_rel, like=p_ch_hat)
    realised  = _to_t(realised,  like=p_ch_hat)
    pl_hat    = _to_t(pl_hat,    like=p_ch_hat)
    pi_da     = _to_t(pi_da,     like=p_ch_hat)
    T, dt = fp.T_total, fp.dt

    xi_real = realised - pl_hat                                    # [B, 24]
    bid = pl_hat + p_da_rel                                        # [B, 24]

    # 1. Expand xi_real to [B, 24, 1] for batched matrix multiplication
    xi_real_col = xi_real.unsqueeze(-1)

    # 2. D_ch @ xi_real_col yields [B, 24, 1]. Squeeze returns it to [B, 24]
    p_ch_raw  = p_ch_hat  + (D_ch  @ xi_real_col).squeeze(-1)
    p_dis_raw = p_dis_hat + (D_dis @ xi_real_col).squeeze(-1)

    if clip_recourse:
        # Stratigakos-like state-dependent saturation. SOC headroom caps carry /dt (dt-correct).
        soc = _to_t(fp.SOC0, like=p_ch_hat)
        ch_list, dis_list, soc_list = [], [], []
        C_ch  = _to_t(fp.C_ch,  like=p_ch_hat)
        C_dis = _to_t(fp.C_dis, like=p_ch_hat)
        B_max = _to_t(fp.B_max, like=p_ch_hat)
        z = _to_t(0.0, like=p_ch_hat)

        ## This runs for every time-step of the day, to ensure accurate tracking of battery SOC.
        for t in range(T):
            max_ch  = torch.minimum(C_ch,  (B_max - soc) / (fp.eta_ch * dt))
            max_dis = torch.minimum(C_dis, (soc * fp.eta_dis) / dt)
            pc = torch.clamp(torch.minimum(torch.clamp(p_ch_raw[..., t],  min=z), torch.clamp(max_ch,  min=z)), min=z)
            pd = torch.clamp(torch.minimum(torch.clamp(p_dis_raw[..., t], min=z), torch.clamp(max_dis, min=z)), min=z)
            soc = soc + dt * (fp.eta_ch * pc - (1.0 / fp.eta_dis) * pd)
            ch_list.append(pc); dis_list.append(pd); soc_list.append(soc)
        p_ch_r  = torch.stack(ch_list, dim = -1)
        p_dis_r = torch.stack(dis_list, dim = -1)
        soc_traj = torch.stack(soc_list, dim = -1)
    else:
        # no-clip: realised = raw; SOC from raw actions (used only for the in-box gate).
        p_ch_r, p_dis_r = p_ch_raw, p_dis_raw
        net = fp.eta_ch * p_ch_r - (1.0 / fp.eta_dis) * p_dis_r
        soc_traj = fp.SOC0 + dt * torch.cumsum(net, dim=-1)

    ## Now accurate vectors have been produced, can aggregate them to realised values.
    p_g = realised + p_ch_r - p_dis_r                             # realised grid draw
    p_imb = p_g - bid                # = (deterministic imbalance, always 0 -- pinned bid) + R xi_real (in-box)

    C_da = (pi_da * bid).sum(dim=-1) * dt                              # includes pi_da . pl_hat (add-back)


    if price_model == "single":
        if pi_imb is None:
            raise ValueError("single needs pi_imb")
        C_imb = (_to_t(pi_imb, like=p_ch_hat) * p_imb).sum(dim=-1) * dt          # signed
    elif price_model == "dual":
        if pi_imb_up is None or pi_imb_down is None:
            raise ValueError("dual needs pi_imb_up, pi_imb_down")
        pi_up = _to_t(pi_imb_up, like=p_ch_hat); pi_down = _to_t(pi_imb_down, like=p_ch_hat)
        C_imb = (pi_up * torch.clamp(p_imb, min=0.0)
                 + pi_down * torch.clamp(-p_imb, min=0.0)).sum(dim=-1) * dt          # >= 0
    else:
        raise ValueError(price_model)

    return RealisedBreakdown(
        p_ch_raw=p_ch_raw, p_dis_raw=p_dis_raw, p_ch_r=p_ch_r, p_dis_r=p_dis_r,
        soc=soc_traj, bid=bid, p_g=p_g, p_imb=p_imb, C_da=C_da, C_imb=C_imb,
    )


def realised_cost(fp, *args, **kwargs) -> torch.Tensor:
    """Thin DFL-loss wrapper: total realised cost = C_da + C_imb on the SAME breakdown the
    test metrics use (so training loss and reported cost cannot drift)."""
    bd = realised_breakdown(fp, *args, **kwargs)
    return bd.C_da + bd.C_imb


def realised_imbalance(fp, *args, **kwargs) -> torch.Tensor:
    """Dispatchability-mode DFL-loss wrapper: realised tracking error on the SAME
    breakdown the test metrics use, in the SAME units as dispatch_objectives'
    dispatchability solve objective (dt**2 * sum_squares(imb)) -- so the post-solve
    training signal matches what the layer itself optimised for at decision time.

    Takes the identical call signature as realised_cost (including price_model/pi_da/
    pi_imb[_up/_down]) even though dispatchability is price-agnostic and only p_imb is
    used below -- there is no dummy/omitted-price special case here. The caller already
    has the TRUE prices on hand for the economic corners' realised_cost call at this same
    site, so passing them through costs nothing extra, and C_da/C_imb are simply computed
    and discarded rather than adding a second, price-optional code path to maintain."""
    bd = realised_breakdown(fp, *args, **kwargs)
    return fp.dt ** 2 * (bd.p_imb ** 2).sum(dim=-1)
