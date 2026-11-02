"""Build the model-input cache of one BatteryLife subset from the processed `.pkl` files.

Usage:
    python scripts/extract_curves.py --subset ISU_ILCC --data-dir <BatteryLife>/ISU_ILCC \
        --labels "<BatteryLife>/Life labels/ISU_ILCC_labels.json" --out cache --workers 8

Uses the official BatteryLife resampling (third_party/batterylife/curves.py): first 100 cycles x (V / max V, C-rate,
Q / nominal capacity) x 300 points. Cell selection follows the official loader: a cell is skipped when its life label
is missing or <= 100 cycles, or when its early curves contain NaN. Writes <out>/<subset>.npz (uncompressed, fixed zip
metadata, so equal content gives an equal SHA-256) and <out>/manifest_<subset>.csv, and compares the SHA-256 of every
cell's curve tensor with splits/cells.csv.
"""
import argparse, glob, hashlib, io, json, os, pickle, sys, time, zipfile
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "batterylife"))
import numpy as np
import pandas as pd
from curves import BatteryLifeCurves

CDL, ECT = 300, 100
KEYS = ["curves", "files", "eol", "nominal_capacity", "valid_cycle_number", "n_real_cycles"]
_labels = {}


def _init(labels):
    global _labels
    _labels = labels


def process(path):
    name = os.path.basename(path)
    row = dict(file=name, status="ok", eol=None, nominal_capacity=None, valid_cycle_number=None, n_real_cycles=None, sha256=None)
    eol = _labels.get(name)
    row["eol"] = eol
    if eol is None:
        row["status"] = "no_label"; return row, None
    if eol <= ECT:
        row["status"] = f"eol<={ECT}"; return row, None
    try:
        data = pickle.load(open(path, "rb"))
        df, curves, eol, nom, vcn = BatteryLifeCurves(CDL, ECT).read_cell_df(data, eol, name)
    except Exception as e:
        row["status"] = f"error:{type(e).__name__}:{str(e)[:120]}"; return row, None
    if np.any(np.isnan(curves[:ECT])):
        row["status"] = "nan_in_early"; return row, None
    c = curves[:ECT].astype(np.float32)
    row.update(nominal_capacity=float(nom), valid_cycle_number=int(vcn), n_real_cycles=int(np.any(c != 0, axis=(1, 2)).sum()),
               sha256=hashlib.sha256(c.tobytes()).hexdigest())
    return row, c


def save_npz_deterministic(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for k in KEYS:
            buf = io.BytesIO(); np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.create_system, zi.external_attr = 3, 0o644 << 16
            zf.writestr(zi, buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=["HUST", "ISU_ILCC", "SNL"])
    ap.add_argument("--data-dir", required=True, help="folder with the subset's .pkl files")
    ap.add_argument("--labels", required=True, help="life-label JSON of the subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.data_dir, "*.pkl")))
    labels = json.load(open(a.labels))
    os.makedirs(a.out, exist_ok=True)
    print(f"{a.subset}: {len(paths)} files, workers={a.workers}", flush=True)
    t0, rows, arrs = time.time(), [], {}
    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(labels,)) as pool:
            for r, c in pool.imap(process, paths):
                rows.append(r); arrs[r["file"]] = c
    else:
        _init(labels)
        for p in paths:
            r, c = process(p)
            rows.append(r); arrs[r["file"]] = c

    man = pd.DataFrame(rows); man.insert(1, "subset", a.subset)
    man = man.sort_values("file").reset_index(drop=True)
    man.to_csv(os.path.join(a.out, f"manifest_{a.subset}.csv"), index=False, lineterminator="\n")
    ok = man[man.status == "ok"]
    save_npz_deterministic(os.path.join(a.out, f"{a.subset}.npz"), dict(
        curves=np.stack([arrs[f] for f in ok.file]) if len(ok) else np.zeros((0, ECT, 3, CDL), np.float32),
        files=ok.file.to_numpy(str), eol=ok.eol.to_numpy(np.int64), nominal_capacity=ok.nominal_capacity.to_numpy(np.float64),
        valid_cycle_number=ok.valid_cycle_number.to_numpy(np.int64), n_real_cycles=ok.n_real_cycles.to_numpy(np.int64)))
    print(f"{a.subset}: kept {len(ok)} of {len(man)}; {man.status.value_counts().to_dict()}; {time.time() - t0:.0f}s")

    ref = pd.read_csv(os.path.join(ROOT, "splits", "cells.csv"))
    ref = ref[(ref.subset == a.subset) & (ref.status == "ok")].set_index("file")["sha256"]
    got = ok.set_index("file")["sha256"]
    missing, extra = sorted(set(ref.index) - set(got.index)), sorted(set(got.index) - set(ref.index))
    differ = [f for f in ref.index.intersection(got.index) if ref[f] != got[f]]
    if missing or extra or differ:
        print(f"WARNING: differs from the cells used in the paper: {len(missing)} missing, {len(extra)} extra, "
              f"{len(differ)} with different curves (first: {(missing + extra + differ)[:3]})")
    else:
        print(f"all {len(ref)} cells match the cells used in the paper")


if __name__ == "__main__":
    main()
