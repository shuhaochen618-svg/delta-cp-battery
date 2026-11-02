"""Generate the campaign-disjoint outer splits used in the paper (the shipped files in splits/ were made this way).

Usage:
    python scripts/make_splits.py --data-dir cache [--out splits]

* isu:  5-fold group K-fold over the 61 ISU_ILCC campaigns, repeated 4 times (20 outer splits); seed 20261027.
* hust: leave-two-batches-out over the 10 HUST batches (all 45 pairs); same random stream, drawn after isu.
* snl:  5-fold group K-fold over the 19 SNL conditions, repeated 4 times (20 outer splits); seed 20261028.
Every outer split lists, by cell file: `train`, `test`, `inner_val` (campaign-disjoint subset of train used only to
choose the epoch budget) and `cv_folds` (campaign-disjoint folds over train for CV+).
"""
import argparse, hashlib, itertools, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from delta_cp.data import campaign_of


def load_groups(data_dir, subset):
    z = np.load(os.path.join(data_dir, f"{subset}.npz"))
    groups = {}
    for f in sorted(str(x) for x in z["files"]):
        groups.setdefault(campaign_of(f), []).append(f)
    return {g: sorted(v) for g, v in sorted(groups.items())}


def cells(groups, names):
    return sorted(f for g in names for f in groups[g])


def gsplit(rng, names, n_folds):
    """Shuffle campaign names and deal them into n_folds near-equal folds."""
    names = list(names); rng.shuffle(names)
    return [sorted(names[i::n_folds]) for i in range(n_folds)]


def outer_entry(rng, groups, train_g, test_g, n_cv, inner_frac):
    tr = sorted(train_g)
    inner = sorted(rng.permutation(tr)[:max(1, int(round(inner_frac * len(tr))))].tolist())
    folds = gsplit(rng, tr, n_cv)
    return dict(train_groups=tr, test_groups=sorted(test_g), inner_val_groups=inner, cv_fold_groups=folds,
                train=cells(groups, tr), test=cells(groups, test_g), inner_val=cells(groups, inner),
                cv_folds=[cells(groups, f) for f in folds])


def kfold_repeats(rng, groups, dataset, n_rep=4, n_folds=5):
    out = []
    for rep in range(n_rep):
        for k, test_g in enumerate(gsplit(rng, groups, n_folds)):
            train_g = [g for g in groups if g not in test_g]
            out.append(dict(id=f"r{rep}f{k}", dataset=dataset, **outer_entry(rng, groups, train_g, test_g, 5, 0.2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "splits"))
    a = ap.parse_args()
    specs = {}

    rng = np.random.default_rng(20261027)
    isu, hust = load_groups(a.data_dir, "ISU_ILCC"), load_groups(a.data_dir, "HUST")
    specs["isu"] = dict(dataset="ISU_ILCC", seed=20261027, calibration_unit="campaign", splits=kfold_repeats(rng, isu, "ISU_ILCC"))
    hs = []
    for pair in itertools.combinations(list(hust), 2):
        train_g = [g for g in hust if g not in pair]
        hs.append(dict(id=f"{pair[0]}-{pair[1]}", dataset="HUST", **outer_entry(rng, hust, train_g, list(pair), 4, 0.25)))
    specs["hust"] = dict(dataset="HUST", seed=20261027, calibration_unit="cell (CV+ folds by batch)", splits=hs)

    rng = np.random.default_rng(20261028)
    snl = load_groups(a.data_dir, "SNL")
    specs["snl"] = dict(dataset="SNL", seed=20261028, calibration_unit="campaign", splits=kfold_repeats(rng, snl, "SNL"))

    os.makedirs(a.out, exist_ok=True)
    lines = []
    for name, spec in specs.items():
        for s in spec["splits"]:
            assert not set(s["train"]) & set(s["test"]) and not set(s["train_groups"]) & set(s["test_groups"])
            assert sorted(c for f in s["cv_folds"] for c in f) == s["train"] and set(s["inner_val"]) <= set(s["train"])
        p = os.path.join(a.out, f"{name}.json")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(spec, fh, indent=1, sort_keys=True); fh.write("\n")
        lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {name}.json")
        print(name, len(spec["splits"]), "outer splits")
    open(os.path.join(a.out, "SHA256SUMS"), "w", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
