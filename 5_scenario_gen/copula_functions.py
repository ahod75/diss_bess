
import random
import numpy as np
from scipy.stats import qmc, norm
import torch
import torch.nn as nn


# tested core — same directory
from forecasting import (
    QUANTILE_LEVELS,
)

# ---- reproducibility (record SEED in the checkpoint) ----
SEED = 20240801
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

EPS = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| seed:", SEED)

def _t(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEVICE)

class MarginalDistCDF:
    """One lead time's predictive distribution from quantile forecasts, as a
    monotone CDF (F̂) / inverse-CDF (F̂⁻¹) by piecewise-linear interpolation.
    Used for PIT during estimation and for the non-differentiable eval path."""

    def __init__(
            self,
            levels,
            values,
            lower=None,
            upper=None
            ):
        levels = np.asarray(levels, float)

        # below not rlly necessary due to softmax + eps in forecaster but doubly ensures monotonicity.
        values = np.asarray(values, float)
        assert np.all(np.diff(values) > 0), "ERROR: marginal quantile values are not strictly monotonic!"

        ## Construct all points on the CDF from the forecaster's marginal distribution.
        
        ## Step 1: Ensure top and bottom of CDF are defined properly.
        # remember: probability is y-axis (levels), value is x-axis (values)

        # need to make sure we define a bottom value (x-axis) for every CDF.
        # if no lower bound defined before:
        # lower = bottom quantile + slope of bottom two quantiles * difference between 0 and lowest quantile prob
        if lower is None:
            slope = (values[1] - values[0]) / (levels[1] - levels[0])
            lower = values[0] + slope * ( 0 - levels[0])
        
        # for upper, same but reverse
        # upper = top quantile + slope of top two quantiles * difference between 1 and top quantile prob
        if upper is None:
            slope = (values[-1] - values[-2]) / (levels[-1] - levels[-2])
            upper = values[-1] + slope * (1.0 - levels[-1])

        l = np.concatenate(([0.0], levels, [1.0]))
        v = np.concatenate(([lower], values, [upper]))

        assert np.all(np.diff(l) > 0), "ERROR: appended 0.0 or 1.0 violates monotonicity of quantile LEVELS (check if levels already contain 0 or 1)"
        assert np.all(np.diff(v) > 0), "ERROR: extrapolated lower/upper bounds violated monotonicity of quantile VALUES"

        # define levels and values as instance attributes
        self._l = l
        self._v = v

    def cdf(self, x):
        """
        linearly interpollated CDF. Takes prosumption input, and returns
        corresponding CDF probability in [0,1] space.
        Does this by linearly interpollating CDF based on input discrete quantiles levels and values.
        """
        return np.interp(np.asarray(x, float), self._v, self._l)

    def inv_cdf(self, u):
        """
        Takes probability from [0,1] space, and returns exact prosumption relating to that CDF probability level
        Does this by linearly interpolating the cdf based on input discrete quantiles levels and values.
        """
    
        u = np.asarray(u, float)
    
        assert np.all((u >= 0.0) & (u <= 1.0)), (
        f"Probability input 'u' must be between 0.0 and 1.0 inclusive. "
        f"Got range [{u.min()}, {u.max()}].")

        # originally clipped values, change back if I do actually want to clip them, not generate error.
        #u = np.clip(np.asarray(u, float), 0.0, 1.0)

        return np.interp(u, self._l, self._v)


def build_probit_matrix(realisations, quantiles, levels=QUANTILE_LEVELS, eps = EPS):
    """
    Prob. integral transform + probit of the baseline forecaster's errors, for estimating copula covariance.

    realisations : (N_days, K) realised prosumption.
    quantiles    : (N_days, K, Q) the BASELINE forecaster's quantiles for each day.
    returns X    : (N_days, K) standard-normal transformed errors.
    """
    realisations = np.asarray(realisations, float)
    quantiles = np.asarray(quantiles, float)
    N, K = realisations.shape
    X = np.empty((N, K))
    for d in range(N):
        for k in range(K):
            # for each time-step:
            #create class that stores
            marginal_dist = MarginalDistCDF(levels, quantiles[d, k])

            # Ensure u stays at least 1e-6 away from 0 and 1
            # so norm.ppf (normal dist inv-CDF) yields finite values instead of +/-inf
            u = np.clip(marginal_dist.cdf(realisations[d, k]), eps, 1 - eps)
            X[d, k] = norm.ppf(u) # builds normalised error matrix, one value at a time.
    return X

def build_Z_corr(corr_matrix, K = 24,seed = SEED,n = 256, eps = EPS):
    """
    Uses Sobol sequences to build an evenly distributed, representative Z_corr that can be frozen.
    """
    
    # 1. Initiate Sobol sequence sampler
    sampler = qmc.Sobol(d=K, scramble=True, seed=seed)

    # 2. Draw evenly spread points in [0, 1]^K
    u_samples = sampler.random(n=n)  # Shape: (n, K)

    # 3. Prevent exact 0 and 1 boundaries for norm.ppf safety
    u_samples = np.clip(u_samples, eps, 1 - eps)

    # 4. Transform uniform QMC samples to Standard Normal N(0, I) using inverse standard normal CDF
    Z = norm.ppf(u_samples)  # Shape: (n, K)

    # 5. Apply Cholesky factor of the Target Covariance Matrix
    L_corr = np.linalg.cholesky(corr_matrix)
    Z_corr = Z @ L_corr.T  # Shape: (n, K)
    return Z_corr

def _precompute_plan(Z_corr, levels, eps = EPS):
    """
    This function pre-maps where the Z_corr variables fall within the array of probability levels of the CDF quantiles.
    As Z_corr is frozen, this only has to be computed once. Doing it now speeds up computation later on using pytorch,
    as pytorch doesn't have to search which quantiles a given point sits between each time it checks.
    """

    levels = np.asarray(levels, float)
    # Add 0.0 and 1.0 probability values to the grid. This allows pytorch to extrapolate tail values later.
    l_aug = np.concatenate(([0.0], levels, [1.0]))                    # (num_quantiles + 2,)

    assert np.all(np.diff(l_aug) > 0), "Probability levels must be strictly monotonic"

    # transform Z_corr to uniform space by passing it through normal CDF, 
    # so that we can find each Z_corr value in (S,K)'s respective index
    # within the respective marginal distribution's CDF.
    u = np.clip(norm.cdf(Z_corr), eps, 1.0 - eps)                     # clip by EPS for numerical stability around boundary values
    # u represents all scenario timesteps values in uniform space.
    num_pts = l_aug.shape[0]                                          # num_pts = num_quantiles + 2

    # finds array location for each uniformly transformed scenario timestep in marginal dist's quantiles in uniform space.
    idx = np.clip(np.searchsorted(l_aug, u, side="right") - 1, 0, num_pts - 2)  # finds index of (S,K) in [0,num_quantiles]
    
    # find how far between two surrounding quantile values each scenario timestep is. 
    # w is the fraction showing how much distance u has between the quantiles below and above it
    # in uniform probability space.
    # (distance from bottom) / (distance between top and bottom)
    w = (u - l_aug[idx]) / (l_aug[idx + 1] - l_aug[idx])              # (S,K)

    ## Also need median quantiles location in index for converting scenarios to errors for the optimiser.
    # if we don't know for certain that we have a quantile at 50% probability on CDF,
    # we can find the closest match by subtracting 0.5 from all quantile values,
    # taking the absolute value (so that each value now represents the distance of the 
    # quantil from the middle) and finding the smallest value. 
    med_idx = int(np.argmin(np.abs(l_aug - 0.5)))
    med_idx_q = int(np.argmin(np.abs(levels - 0.5)))   # index into UN-augmented quantiles
    return idx.astype(np.int64), w.astype(np.float32), med_idx, med_idx_q

class FrozenCopulaSampler(nn.Module):
    """
    Differentiable Gaussian-copula sampler with FROZEN Σ and FROZEN draws.

    Precomputes ONCE, from the frozen draws and the quantile LEVELS only:
        correlated normal draws -> uniforms -> (bracket-index, weight) plan
    against the augmented level grid [0, levels..., 1].
    The plan is independent of the forecast VALUES, so it stays valid for
    every DFL step. During training, `prosumption` / `errors` interpolate the
    CURRENT forecaster quantiles through that frozen plan — differentiable in
    the quantiles (marginals), constant in Sigma/draws.

    Args
    ----
    Z_corr : (S, K) frozen pre-drawn correlated standard-normal samples.
    levels : (num_quantiles,)   quantile levels the forecaster emits (must NOT include 0 or 1).
    """

    def __init__(self, Z_corr, levels=QUANTILE_LEVELS):
        super().__init__()
        levels = np.asarray(levels, dtype=float)

        self.num_quantiles = len(levels)
        self.S, self.K = Z_corr.shape

        # calculations all done with tail-extrapolated CDF.
        idx, w, med_idx, med_idx_q = _precompute_plan(Z_corr, levels)   # pre-computes levels to speed up scenario gen
        self.med_idx = med_idx
        self.med_idx_q = med_idx_q

        # idx indexes the augmented (num_quantiles+2) value grid, so it must live in [0, num_quantiles]
        assert idx.min() >= 0 and idx.max() <= self.num_quantiles, (
            f"idx range [{idx.min()},{idx.max()}] inconsistent with augmented grid "
            f"(expected [0,{self.num_quantiles}]); is _precompute_plan using the [0,levels,1] grid?"
        )

        # frozen buffers (move with .to(device), never trained)
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long))     # (S, K)
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))      # (S, K)
        self.register_buffer("levels_t", torch.as_tensor(levels, dtype=torch.float32))  # (num_quantiles,)

    # ------------------------------------------------------------------ #
    def _augmented_values(self, quantiles):
        """
        quantiles : (K, num_quantiles) monotone forecast quantiles (requires grad).
        returns   : (K, num_quantiles+2) grid [lower, quantiles..., upper] with slope-
                    extrapolated tail endpoints, differentiable in quantiles.
        """
        if quantiles.dim() != 2:
            raise ValueError("expects (K, num_quantiles) for one day; loop or batch externally")
        K, num_quantiles = quantiles.shape
        if num_quantiles != self.num_quantiles:
            raise ValueError(f"quantiles has num_quantiles={num_quantiles}, sampler built for num_quantiles={self.num_quantiles}")

        ## ensure that all calculations are done on same device and at same precision
        lev_t = self.levels_t.to(quantiles)                                      # (num_quantiles,)
    
        ## uses same logic as in piecewise linear CDF class
        # now, instead of finding the slope between each quantile per individual timestep, 
        # we want to do it for a matrix representing an entire window of K steps.
        # lower = q0 + (q1-q0)/(l1-l0) * (0 - l0)
        slope_lo = (quantiles[:, 1] - quantiles[:, 0]) / (lev_t[1] - lev_t[0])      # (K,)
        lower = quantiles[:, 0] + slope_lo * (0.0 - lev_t[0])                    # (K,)
        # upper = q_{-1} + (q_{-1}-q_{-2})/(l_{-1}-l_{-2}) * (1 - l_{-1})
        slope_hi = (quantiles[:, -1] - quantiles[:, -2]) / (lev_t[-1] - lev_t[-2])  # (K,)
        upper = quantiles[:, -1] + slope_hi * (1.0 - lev_t[-1])                  # (K,)

        return torch.cat([lower.unsqueeze(1), quantiles, upper.unsqueeze(1)], dim=1)  # (K, num_quantiles+2)


    def _interpolate(self, v_aug):
        """Gather the frozen bracket plan through the augmented values -> (S, K)."""
        K = v_aug.shape[0]
        vexp = v_aug.unsqueeze(0).expand(self.S, K, self.num_quantiles + 2)             
        # unsqueeze takes the matrix of augment values for each quantile, 
        # and adds new dimension at front of tensor to represent scenarios.
        # expand expands this dimension so that the tensor is duplicated
        # num_scenario times across the scenario dimension.
        # now, vexp represents the quantile values for each timestep, duplicated 
        # for every scenario.
        # It is now ready to interpolate the values for each timestep for each scenario
        # based on the precalculated idx values, weights and augment values (quantile values + 0 and 1 prob values)
        # final dimensions of vexp are (S, K, num_quantiles+2)

        # for each scenario's timestep, we want to find the values of the quantiles above and below
        # the scenario's value using the precomputed idx values.
        # Since we have precomputed the distance of each value between the quantiles above and below
        # it in uniform probability space already, we know that the realised value of that scenario's
        # timestep is the value of the below quantile, plus the distance of the point between the below and above
        # quantiles in probability space, multiplied by the distance between these quantiles in value space.  
        
        v_lo = torch.gather(vexp, 2, self.idx.unsqueeze(-1)).squeeze(-1)      # (S, K)
        v_hi = torch.gather(vexp, 2, (self.idx + 1).unsqueeze(-1)).squeeze(-1)  # (S, K)
        # To find the below and above quantile values:
        # for each value:
        # - unsqueeze the precomputed idx values so they have same number of dimensions as vexp
        # - use torch.gather to look along the third dimension of vexp (the quantile value dimension)
        # - torch.gather outputs a tensor that matches the shape of the index tensor provided.
        #   Therefore, the output of gather is (S, K, 1), as it only returns the index of a SINGLE quantile.
        # - Since we don't want that final dimension and there is only one of it, we can use 
        #   squeeze to remove it.
        # - this results in an (S,K) shaped tensor for each of v_lo and v_hi,
        #   which represents the above and below quantiles for each timestep of each scenario.
        #
        # - From here, all we have to do is return the final tensor of the interpolated final value of each
        #   scenario's timesteps, based on the precomputed values!
        return v_lo + self.w * (v_hi - v_lo)

    # ------------------------------------------------------------------ #
    def prosumption(self, quantiles):
        """
        quantiles : (K, num_quantiles) monotone quantile forecasts (torch, requires grad).
        returns   : (S, K) prosumption scenarios p^l, differentiable in quantiles.

        Tails are slope-extrapolated to prob 0/1 (no clamping), matching
        MarginalDistCDF, so scenarios can exceed the outer quantiles.
        """
        v_aug = self._augmented_values(quantiles)
        return self._interpolate(v_aug)
    
    def prosumption_mean(self, quantiles):
        """Mean prosumption forecast: average of all generated scenarios. (K,)"""
        p = self.prosumption(quantiles)              # (S, K)
        return p.mean(dim=0)                          # (K,)

    def errors_mean(self, quantiles):
        """Scenario deviations from the MEAN forecast: scenario - mean.
        Has EXACTLY zero sample mean by construction."""
        p = self.prosumption(quantiles)          # (S, K)
        mean = p.mean(dim=0, keepdim=True)        # (1, K)
        return p - mean                           # (S, K), zero column-mean

    def errors_median(self, quantiles):
        """Scenario deviations from the forecast median: scenario - q_median.
        quantiles : (K, Q) ; returns (S, K), differentiable in quantiles."""
        p = self.prosumption(quantiles)                       # (S, K)
        med = quantiles[:, self.med_idx_q]                    # (K,) un-augmented median
        return p - med.unsqueeze(0)

    def mean_and_errors(self, quantiles):
        """Mean forecast (K,) and mean-centred error scenarios (S, K) from ONE
        prosumption pass. p̂E and ξ share the same scenario mean by construction,
        so E[ξ]=0 exactly and the anchor/error are guaranteed consistent."""
        p = self.prosumption(quantiles)              # (S, K)
        mean = p.mean(dim=0)                          # (K,)
        return mean, p - mean.unsqueeze(0)            # (K,), (S, K)