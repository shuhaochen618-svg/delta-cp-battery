"""Train one backbone variant on one outer split and compute every interval method on its test campaigns.

Usage:
    python scripts/run_split.py --dataset isu --split r0f0 --variant CPT-Delta --seed 2021 --data-dir cache

Steps for one (split, variant, seed):
1. epoch budget e* by early stopping on the split's inner_val campaigns;
2. one CV+ fold model per campaign-disjoint fold (trained on train minus the fold, e* epochs);
3. split CQR model on a random 60/40 cell split of train; deep ensemble of 5 models on all of train;
4. for each history length k and alpha in {0.1, 0.2}: intervals of every method and their metrics.
Outputs: <out>/<dataset>/<split>/<variant>_s<seed>/{result.json, intervals_k<k>.npz}.
"""
import argparse, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from delta_cp import conformal as C
from delta_cp import model as M
from delta_cp.data import Data, calibration_unit_of, campaign_of, load_cache

SUBSET = {"isu": "ISU_ILCC", "hust": "HUST", "snl": "SNL"}
METHODS = ["gauss_oof", "split_cqr", "cell_cvplus", "campaign_cvplus", "deep_ensemble", "cqr_r", "qr_uncalibrated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SUBSET))
    ap.add_argument("--split", required=True)
    ap.add_argument("--variant", required=True, choices=list(M.VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", required=True, help="folder with the cache files written by extract_curves.py")
    ap.add_argument("--splits-dir", default=os.path.join(ROOT, "splits"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ks", type=int, nargs="+", default=[100], help="history lengths to evaluate")
    a = ap.parse_args()
    name, delta = M.VARIANTS[a.variant]
    t0 = time.time(); timing = {}
    device = torch.device(a.device)
    spath = os.path.join(a.splits_dir, f"{a.dataset}.json")
    split_sha = hashlib.sha256(open(spath, "rb").read()).hexdigest()
    sp = next(s for s in json.load(open(spath))["splits"] if s["id"] == a.split)
    D = Data(load_cache(a.data_dir, [SUBSET[a.dataset]], delta), device)
    tr, te, iv = D.ids(sp["train"]), D.ids(sp["test"]), D.ids(sp["inner_val"])
    folds = [D.ids(f) for f in sp["cv_folds"]]
    assert not set(tr) & set(te) and sorted(np.concatenate(folds).tolist()) == sorted(tr.tolist())
    fold_of = {c: f for f, cs in enumerate(folds) for c in cs}
    rng = np.random.default_rng(a.seed)
    train = lambda cells, seed, **kw: M.train_backbone(name, D, cells, seed, delta=delta, **kw)

    # 1. epoch budget
    _, _, e_star = train(np.setdiff1d(tr, iv), a.seed, val_cells=iv)
    timing["epoch_budget_s"] = time.time() - t0
    print(f"[{a.dataset} {a.split} {a.variant} s{a.seed}] e* = {e_star} ({timing['epoch_budget_s']:.0f}s)", flush=True)

    # 2. CV+ fold models
    cal = np.concatenate(folds); fcal = np.array([fold_of[c] for c in cal])
    P = {k: dict(Qcal=np.zeros((len(cal), 5)), Qte=np.zeros((len(folds), len(te), 5))) for k in a.ks}
    t1 = time.time()
    for f, fc in enumerate(folds):
        m, sc, _ = train(np.setdiff1d(tr, fc), a.seed * 100 + f, epochs=e_star)
        sel = fcal == f
        for k in a.ks:
            P[k]["Qcal"][sel] = M.predict(m, D, cal[sel], k, sc); P[k]["Qte"][f] = M.predict(m, D, te, k, sc)
        del m; torch.cuda.empty_cache()
    timing["cvplus_s"] = time.time() - t1

    # 3. split CQR: random 60/40 cell split inside train; deep ensemble of 5 on all training cells
    t2 = time.time()
    perm = rng.permutation(tr); a_, b_ = perm[:int(0.6 * len(tr))], perm[int(0.6 * len(tr)):]
    m1, sc1, _ = train(a_, a.seed * 100 + 99, epochs=e_star)
    Q1 = {k: (M.predict(m1, D, b_, k, sc1), M.predict(m1, D, te, k, sc1)) for k in a.ks}
    del m1; torch.cuda.empty_cache()
    Qm = {k: np.zeros((5, len(te), 5)) for k in a.ks}
    for e in range(5):
        me, sce, _ = train(tr, a.seed * 100 + 50 + e, epochs=e_star)
        for k in a.ks:
            Qm[k][e] = M.predict(me, D, te, k, sce)
        del me; torch.cuda.empty_cache()
    timing["split_cqr_ensemble_s"] = time.time() - t2

    # 4. intervals and metrics
    y_cal, y_te = D.y[cal], D.y[te]
    files_te = D.files[te]
    units = np.array([calibration_unit_of(f) for f in D.files[cal]]); uu = np.unique(units)
    groups_te = np.array([campaign_of(f) for f in files_te])
    pick = np.array([rng.choice(np.where(units == u)[0]) for u in uu])
    b0 = np.median(D.y[tr])
    res, diag = {}, dict(e_star=int(e_star), n_units=int(len(uu)), n_cal=int(len(cal)), n_test=int(len(te)))
    od = os.path.join(a.out, a.dataset, a.split, f"{a.variant}_s{a.seed}"); os.makedirs(od, exist_ok=True)
    for k in a.ks:
        Pk = P[k]; point = Pk["Qte"][:, :, 2].mean(0)
        save = {}
        for alpha, (il, ih) in M.QIDX.items():
            lo_c, hi_c = Pk["Qcal"][:, il], Pk["Qcal"][:, ih]
            LOt, HIt = Pk["Qte"][:, :, il], Pk["Qte"][:, :, ih]
            S = np.maximum(lo_c - y_cal, y_cal - hi_c)
            Wc, Wt = np.maximum(hi_c - lo_c, 1e-6), np.maximum(HIt - LOt, 1e-6)
            sd_oof = np.std(y_cal - Pk["Qcal"][:, 2], ddof=1)
            qb, qt = Q1[k]
            ints = {"gauss_oof": (point - C.Z[alpha] * sd_oof, point + C.Z[alpha] * sd_oof),
                    "split_cqr": C.split_cqr(qb[:, il], qb[:, ih], D.y[b_], qt[:, il], qt[:, ih], alpha),
                    "cell_cvplus": C.cvplus_interval(LOt[fcal] - S[:, None], HIt[fcal] + S[:, None], alpha),
                    "campaign_cvplus": C.campaign_cvplus(LOt, HIt, S, fcal, pick, alpha),
                    "deep_ensemble": C.deep_ensemble(Qm[k], alpha),
                    "cqr_r": C.campaign_cvplus(LOt, HIt, S / Wc, fcal, pick, alpha, width=Wt),
                    "qr_uncalibrated": (LOt.mean(0), HIt.mean(0))}
            for nm, (L, U) in ints.items():
                res[f"k{k}_a{alpha}_{nm}"] = C.metrics(y_te, L, U, groups_te, alpha, point=point)
            if alpha == 0.1:
                save.update({f"{nm}_L": L for nm, (L, U) in ints.items()}); save.update({f"{nm}_U": U for nm, (L, U) in ints.items()})
        res[f"k{k}_point"] = dict(logrmse=float(np.sqrt(np.mean((point - y_te) ** 2))),
                                 mape=float(np.mean(np.abs(np.exp(point) - np.exp(y_te)) / np.exp(y_te))))
        res[f"k{k}_median_baseline"] = dict(logrmse=float(np.sqrt(np.mean((b0 - y_te) ** 2))),
                                           mape=float(np.mean(np.abs(np.exp(b0) - np.exp(y_te)) / np.exp(y_te))))
        save.update(y=y_te, point=point, files=files_te, groups=groups_te, median_baseline=np.full(len(te), b0))
        np.savez_compressed(os.path.join(od, f"intervals_k{k}.npz"), **save)
    timing["total_s"] = time.time() - t0
    out = dict(dataset=a.dataset, split=a.split, variant=a.variant, backbone=name, delta=delta, seed=a.seed,
               split_file_sha256=split_sha, quantiles=M.QUANTILES, config=dict(M.COMMON, **M.CONFIGS[name]),
               diag=diag, timing=timing, results=res, torch=torch.__version__, device=str(device),
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    json.dump(out, open(os.path.join(od, "result.json"), "w"), indent=1)
    k0 = a.ks[0]
    for nm in METHODS:
        r = res[f"k{k0}_a0.1_{nm}"]
        print(f"  k{k0} a0.1 {nm:16s} cov {r['coverage']:.3f}  width x{r['width_factor_mean']:.2f}  "
              f"campaigns<0.8 {r['frac_groups_below']:.2f}", flush=True)
    print(f"  point log-RMSE {res[f'k{k0}_point']['logrmse']:.3f} (median baseline "
          f"{res[f'k{k0}_median_baseline']['logrmse']:.3f}); {timing['total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
