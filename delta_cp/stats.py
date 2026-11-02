"""Pooled metrics and paired comparisons across outer splits.

Records are (outer split, test cell) pairs, each averaged over seeds. Uncertainty is a cluster bootstrap over campaigns
(95 % percentile intervals); paired Wilcoxon signed-rank tests run over outer splits, with Holm correction across
datasets.
"""
import collections

import numpy as np
from scipy.stats import wilcoxon

B = 10000            # bootstrap resamples
THR = 0.8            # a campaign "under-covered" below this coverage (alpha = 0.1)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def group_stats(rows, m):
    """Coverage of each campaign over all its (split, cell) records."""
    g = collections.defaultdict(list)
    for r in rows:
        g[r["group"]].append(r[m + "_cov"])
    return {k: float(np.mean(v)) for k, v in g.items()}


def gsum(rows, f):
    """Per-campaign (sum of f over records, number of records), campaigns in sorted order."""
    s, n = collections.defaultdict(float), collections.defaultdict(int)
    for r in rows:
        s[r["group"]] += f(r); n[r["group"]] += 1
    gs = sorted(s)
    return np.array([s[g] for g in gs]), np.array([n[g] for g in gs], float)


def boot_ratio(num, den, rng, n=B, num2=None, den2=None):
    """Cluster bootstrap over campaigns of sum(num) / sum(den) (or its ratio to sum(num2) / sum(den2))."""
    G = len(num); idx = rng.integers(0, G, size=(n, G))
    v = num[idx].sum(1) / den[idx].sum(1)
    if num2 is not None:
        v = v / (num2[idx].sum(1) / den2[idx].sum(1))
    return np.percentile(v, [2.5, 97.5]).tolist()


def summarise(rows, m):
    rr = list(rows.values())
    cov = np.array([r[m + "_cov"] for r in rr]); w = np.array([r[m + "_w"] for r in rr]); fin = np.isfinite(w)
    gs = group_stats(rr, m)
    return dict(coverage=float(cov.mean()), width_factor_mean=float(np.exp(w[fin]).mean()) if fin.any() else float("inf"),
                frac_inf=float(1 - fin.mean()), winkler=float(np.mean([r[m + "_wink"] for r in rr])),
                share_below=float(np.mean([v < THR for v in gs.values()])),
                campaign_cov_p10=float(np.percentile(list(gs.values()), 10)))


def paired(rows_a, rows_b):
    keys = sorted(set(rows_a) & set(rows_b))
    return [rows_a[k] for k in keys], [rows_b[k] for k in keys]


def split_logrmse(rows):
    s = collections.defaultdict(list)
    for r in rows:
        s[r["split"]].append(r["se"])
    return {k: float(np.sqrt(np.mean(v))) for k, v in s.items()}


def validity(ra, rref, rng, m, mref):
    """Valid = pooled coverage >= 0.88, its CI lower bound >= 0.85, and the share of under-covered campaigns no more than
    that of the reference + 0.05 (CI upper bound of the difference <= 0.05)."""
    num, den = gsum(ra, lambda r: r[m + "_cov"]); ci = boot_ratio(num, den, rng)
    nr, dr = gsum(rref, lambda r: r[mref + "_cov"])
    d = (num / den < THR).astype(float) - (nr / dr < THR).astype(float)
    ci_d = boot_ratio(d, np.ones_like(d), rng)
    cov = float(num.sum() / den.sum())
    return dict(coverage=cov, ci=ci, share_diff_ci=ci_d, valid=bool(cov >= 0.88 and ci[0] >= 0.85 and ci_d[1] <= 0.05))


def width_ratio(ra, rb, rng, ma, mb):
    """Mean width factor exp(U - L) of (ra, ma) over that of (rb, mb), with its cluster-bootstrap CI."""
    na, da = gsum(ra, lambda r: np.exp(r[ma + "_w"])); nb, db = gsum(rb, lambda r: np.exp(r[mb + "_w"]))
    return dict(ratio=float((na.sum() / da.sum()) / (nb.sum() / db.sum())), ci=boot_ratio(na, da, rng, num2=nb, den2=db))


def coverage_gap(rows, m, ref, rng, n=B):
    """Coverage of method m against method ref on the same records: pooled coverage and share of under-covered campaigns
    with their cluster-bootstrap CIs, the differences m - ref (same campaign resample for both), the number of outer
    splits in which m covers less than ref and a Wilcoxon test over splits."""
    rr = list(rows.values())
    nm, dm = gsum(rr, lambda r: r[m + "_cov"]); nr, dr = gsum(rr, lambda r: r[ref + "_cov"])
    idx = rng.integers(0, len(nm), size=(n, len(nm)))
    cov, cov_ref = nm[idx].sum(1) / dm[idx].sum(1), nr[idx].sum(1) / dr[idx].sum(1)
    low, low_ref = (nm / dm < THR).astype(float), (nr / dr < THR).astype(float)
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    out = dict(coverage=float(nm.sum() / dm.sum()), coverage_ci=ci(cov), share_below=float(low.mean()),
               share_below_ci=ci(low[idx].mean(1)))
    if m == ref:
        return out
    s = collections.defaultdict(list)
    for r in rr:
        s[r["split"]].append(r[m + "_cov"] - r[ref + "_cov"])
    d = np.round([np.mean(v) for _, v in sorted(s.items())], 12)   # seed-averaged coverages: treat float noise as a tie
    out.update(cov_diff=float(nm.sum() / dm.sum() - nr.sum() / dr.sum()), cov_diff_ci=ci(cov - cov_ref),
               share_diff=float(low.mean() - low_ref.mean()), share_diff_ci=ci((low - low_ref)[idx].mean(1)),
               splits_lower=int((d < 0).sum()), n_splits=len(d),
               wilcoxon_p=float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else None)
    return out


def accuracy(ra, rb, rng):
    """Point log-RMSE of ra vs rb: Wilcoxon over outer splits and cluster-bootstrap CI of the pooled difference."""
    sa, sb = split_logrmse(ra), split_logrmse(rb); ks = sorted(sa)
    d = np.array([sa[k] - sb[k] for k in ks])
    p = float(wilcoxon(d).pvalue) if len(d) >= 2 and np.any(d != 0) else 1.0
    ea, na = gsum(ra, lambda r: r["se"]); eb, _ = gsum(rb, lambda r: r["se"])
    idx = rng.integers(0, len(ea), size=(B, len(ea)))
    diff = np.sqrt(ea[idx].sum(1) / na[idx].sum(1)) - np.sqrt(eb[idx].sum(1) / na[idx].sum(1))
    return dict(logrmse_a=float(np.sqrt(ea.sum() / na.sum())), logrmse_b=float(np.sqrt(eb.sum() / na.sum())),
                mean_split_diff=float(d.mean()), splits_better=int((d < 0).sum()), n_splits=len(d), p=p,
                ci_diff=np.percentile(diff, [2.5, 97.5]).tolist())
