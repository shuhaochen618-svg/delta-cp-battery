"""Pool the runs, compare every interval method with campaign CV+, and compare CPT-Delta with CPT (alpha = 0.1, k = 100).

Usage:
    python scripts/aggregate.py [--runs runs] [--out results]
Per (dataset, variant): every (outer split, test cell) record is averaged over seeds. For each dataset: point log-RMSE
(Wilcoxon over outer splits, cluster-bootstrap CI of the difference), validity of campaign CV+ and the ratio of mean
interval width factors. On ISU_ILCC, CPT-Delta + campaign CV+ is also compared with the narrowest valid baseline among
all methods on CPT and CPMLP. Holm correction across the three accuracy tests. For every (dataset, variant), each
interval method is also compared with campaign CV+ on the same records (coverage and share of under-covered campaigns,
paired cluster bootstrap; one random stream per (dataset, variant), so the result does not depend on which runs are
present). Writes <out>/summary.{json,md}.
"""
import argparse, collections, glob, json, os, sys, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from delta_cp import stats as St

METHODS = ["gauss_oof", "split_cqr", "cell_cvplus", "campaign_cvplus", "deep_ensemble", "cqr_r", "qr_uncalibrated"]
BASELINES = METHODS[:-1]
OURS = "campaign_cvplus"
DATASETS = ["isu", "hust", "snl"]
VARIANTS = ["CPT", "CPT-Delta", "CPMLP"]
ALPHA, SEED, GAP_SEED = 0.1, 20261028, 20261030


def load(runs, ds, var, k=100):
    recs = collections.defaultdict(lambda: collections.defaultdict(list)); seeds = collections.defaultdict(set)
    for f in glob.glob(os.path.join(runs, ds, "*", f"{var}_s*", f"intervals_k{k}.npz")):
        split, seed = f.split(os.sep)[-3], f.split(os.sep)[-2].split("_s")[-1]
        z = np.load(f, allow_pickle=False); y = z["y"]; pt = z["point"]
        b0 = z["median_baseline"] if "median_baseline" in z.files else None
        for i, (cell, g) in enumerate(zip(z["files"].astype(str), z["groups"].astype(str))):
            r = recs[(split, cell)]; r["group"] = g
            r["se"].append(float((pt[i] - y[i]) ** 2))
            if b0 is not None:
                r["se_b0"].append(float((b0[i] - y[i]) ** 2))
            for m in METHODS:
                L, U = z[m + "_L"][i], z[m + "_U"][i]; w = U - L
                r[m + "_cov"].append(float(L <= y[i] <= U)); r[m + "_w"].append(float(w))
                r[m + "_wink"].append(float(w + (2 / ALPHA) * (max(L - y[i], 0) + max(y[i] - U, 0))))
        seeds[split].add(seed)
    rows = {key: dict(split=key[0], cell=key[1], group=r["group"], **{kk: float(np.mean(v)) for kk, v in r.items() if kk != "group"})
            for key, r in recs.items()}
    return rows, {s: len(v) for s, v in seeds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data, out = {}, dict(setting=dict(k=100, alpha=ALPHA, bootstrap=St.B), datasets={}, comparisons={}, coverage_gaps={})
    for ds in DATASETS:
        for var in VARIANTS:
            rows, seeds = load(a.runs, ds, var)
            if rows:
                data[(ds, var)] = rows
                E = dict(n_splits=len(seeds), seeds_per_split=sorted(set(seeds.values())),
                         n_campaigns=len({r["group"] for r in rows.values()}),
                         point_logrmse=float(np.sqrt(np.mean([r["se"] for r in rows.values()]))),
                         methods={m: St.summarise(rows, m) for m in METHODS})
                if all("se_b0" in r for r in rows.values()):
                    E["median_baseline_logrmse"] = float(np.sqrt(np.mean([r["se_b0"] for r in rows.values()])))
                out["datasets"][f"{ds}/{var}"] = E
    Cmp, tests = out["comparisons"], []
    for ds in DATASETS:
        if (ds, "CPT-Delta") not in data or (ds, "CPT") not in data:
            continue
        ra, rb = St.paired(data[(ds, "CPT-Delta")], data[(ds, "CPT")])
        acc = St.accuracy(ra, rb, rng); tests.append(f"{ds}_accuracy")
        Cmp[f"{ds}_accuracy"] = acc
        Cmp[f"{ds}_intervals"] = dict(validity=St.validity(ra, rb, rng, OURS, OURS), width_ratio=St.width_ratio(ra, rb, rng, OURS, OURS))
    if all(k in data for k in (("isu", "CPT-Delta"), ("isu", "CPT"), ("isu", "CPMLP"))):
        ref = data[("isu", "CPT")]; cands = []
        for var in ("CPT", "CPMLP"):
            for m in BASELINES:
                rv, rr = St.paired(data[("isu", var)], ref)
                if St.validity(rv, rr, rng, m, OURS)["valid"]:
                    cands.append((St.summarise({(r["split"], r["cell"]): r for r in rv}, m)["width_factor_mean"], var, m))
        if cands:
            _, var, m = min(cands)
            ra, rb = St.paired(data[("isu", "CPT-Delta")], data[("isu", var)])
            Cmp["isu_vs_narrowest_valid_baseline"] = dict(baseline=f"{var}/{m}",
                                                          valid_baselines=[f"{v_}/{m_} {w:.2f}" for w, v_, m_ in sorted(cands)],
                                                          width_ratio=St.width_ratio(ra, rb, rng, OURS, m))
    if tests:
        for name, pa in zip(tests, St.holm(np.array([Cmp[t]["p"] for t in tests]))):
            Cmp[name]["p_holm"] = float(pa)
    for (ds, var), rows in data.items():
        g_rng = np.random.default_rng([GAP_SEED, zlib.crc32(f"{ds}/{var}".encode())])
        out["coverage_gaps"][f"{ds}/{var}"] = {m: St.coverage_gap(rows, m, OURS, g_rng) for m in METHODS}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    L = ["# Summary (k = 100, alpha = 0.1)", ""]
    for key, E in out["datasets"].items():
        b0 = f"; median baseline {E['median_baseline_logrmse']:.3f}" if "median_baseline_logrmse" in E else ""
        L += [f"## {key}: {E['n_splits']} splits, seeds/split {E['seeds_per_split']}, {E['n_campaigns']} campaigns; "
              f"point log-RMSE {E['point_logrmse']:.3f}{b0}", "",
              "| Method | Coverage | Width factor | Winkler | Campaigns < 0.8 | Campaign cov p10 |", "|---|---|---|---|---|---|"]
        L += [f"| {m} | {s['coverage']:.3f} | {s['width_factor_mean']:.2f} | {s['winkler']:.3f} | {s['share_below']:.2f} | "
              f"{s['campaign_cov_p10']:.2f} |" for m, s in E["methods"].items()] + [""]
    f = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    for key, G in out["coverage_gaps"].items():
        L += [f"## {key}: each method vs {OURS}", "",
              "| Method | Coverage (95 % CI) | minus ours (95 % CI) | Campaigns < 0.8 (95 % CI) | minus ours (95 % CI) | "
              "Splits lower | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
        for m, r in G.items():
            if m == OURS:
                L.append(f"| {m} | {r['coverage']:.3f} {f(r['coverage_ci'])} | | {r['share_below']:.2f} {f(r['share_below_ci'])} | | | |")
            else:
                p = "" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.4f}"
                L.append(f"| {m} | {r['coverage']:.3f} {f(r['coverage_ci'])} | {r['cov_diff']:+.3f} {f(r['cov_diff_ci'])} | "
                         f"{r['share_below']:.2f} {f(r['share_below_ci'])} | {r['share_diff']:+.2f} {f(r['share_diff_ci'])} | "
                         f"{r['splits_lower']} / {r['n_splits']} | {p} |")
        L.append("")
    L += ["## CPT-Delta vs CPT", ""] + [f"* {k}: " + json.dumps(v, default=lambda x: round(x, 4)) for k, v in Cmp.items()]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
