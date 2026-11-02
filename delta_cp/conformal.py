"""Interval constructions on y = log life and per-split interval metrics.

All conformal scores are CQR scores S = max(lo - y, y - hi) of the quantile head.
* cell CV+: CV+ (Barber et al., 2021) over every calibration cell;
* campaign CV+: CV+ over one randomly chosen calibration cell per calibration unit (single subsampling for two-layer
  hierarchical data, Dunn et al., 2023), so that the units, not the cells, are treated as exchangeable;
* CQR-r: campaign CV+ with the score divided by the quantile-head width;
* Gaussian band from the out-of-fold residual SD; split CQR on a random 60/40 cell split; deep ensemble.
"""
import math

import numpy as np

from .model import QIDX

Z = {0.1: 1.6448536, 0.2: 1.2815516}


def cvplus_interval(lo_vals, hi_vals, alpha):
    """lo_vals, hi_vals: (n_cal, n_test) per-calibration-point bounds. Returns (L, U) per test point."""
    n = lo_vals.shape[0]
    ku = math.ceil((1 - alpha) * (n + 1))
    kl = math.floor(alpha * (n + 1))
    U = np.sort(hi_vals, axis=0)[ku - 1] if ku <= n else np.full(hi_vals.shape[1], np.inf)
    L = np.sort(lo_vals, axis=0)[kl - 1] if kl >= 1 else np.full(lo_vals.shape[1], -np.inf)
    return L, U


def campaign_cvplus(LOt, HIt, S, fcal, pick, alpha, width=None):
    """CV+ on the calibration cells `pick` (one per unit). LOt, HIt: [n_folds, n_test] bounds of every fold model;
    S: scores of all calibration cells; fcal: fold of each calibration cell; width [n_folds, n_test] rescales the score
    (CQR-r)."""
    f = fcal[pick]
    if width is None:
        return cvplus_interval(LOt[f] - S[pick][:, None], HIt[f] + S[pick][:, None], alpha)
    return cvplus_interval(LOt[f] - S[pick][:, None] * width[f], HIt[f] + S[pick][:, None] * width[f], alpha)


def split_cqr(lo_c, hi_c, y_c, lo_t, hi_t, alpha):
    s = np.maximum(lo_c - y_c, y_c - hi_c); n = len(s); kq = math.ceil((1 - alpha) * (n + 1))
    q = np.sort(s)[kq - 1] if kq <= n else np.inf
    return lo_t - q, hi_t + q


def deep_ensemble(Qm, alpha):
    """Members' quantile bounds averaged, widened by z x SD of the members' medians. Qm: [n_members, n_test, 5]."""
    il, ih = QIDX[alpha]
    spread = Qm[:, :, 2].std(0)
    return Qm[:, :, il].mean(0) - Z[alpha] * spread, Qm[:, :, ih].mean(0) + Z[alpha] * spread


def metrics(y, L, U, groups, alpha, point=None):
    cov = (y >= L) & (y <= U); w = U - L; fin = np.isfinite(w)
    wink = np.where(np.isfinite(w), w + (2 / alpha) * (np.maximum(L - y, 0) + np.maximum(y - U, 0)), np.inf)
    gcov = np.array([cov[groups == g].mean() for g in np.unique(groups)])
    out = dict(coverage=float(cov.mean()), frac_inf=float(1 - fin.mean()),
               width_factor_mean=float(np.exp(w[fin]).mean()) if fin.any() else np.inf,
               width_factor_median=float(np.exp(np.median(w))) if fin.mean() >= 0.5 else np.inf,
               winkler_mean=float(np.mean(wink)), n=int(len(y)), n_groups=int(len(gcov)),
               group_cov_p10=float(np.percentile(gcov, 10)), frac_groups_below=float(np.mean(gcov < 1 - alpha - 0.1)))
    if point is not None:
        out.update(logrmse=float(np.sqrt(np.mean((point - y) ** 2))),
                   mape=float(np.mean(np.abs(np.exp(point) - np.exp(y)) / np.exp(y))))
    return out
