"""Cached model inputs, campaign labels and the Delta-curve channels.

A cache file `<data_dir>/<SUBSET>.npz` (written by `scripts/extract_curves.py`) holds, for every kept cell, the first
100 cycles x 3 channels (voltage / max voltage, current in C-rate, capacity / nominal capacity) x 300 resampled points,
the cell's file name and its end-of-life cycle number.
"""
import os
import re

import numpy as np

ECT, CDL = 100, 300                     # early-cycle threshold, resampled points per cycle (charge + discharge)
SUBSETS = ("HUST", "ISU_ILCC", "SNL")
REF_CYCLE = 10                          # 1-based reference cycle of the Delta channels


def campaign_of(f):
    """Experimental campaign of a cell: ISU_ILCC group G<n>, HUST batch b<n>, SNL condition (file name without the
    replicate letter)."""
    m = re.match(r"ISU-ILCC_(G\d+)C\d+\.pkl$", f)
    if m:
        return m.group(1)
    if f.startswith("SNL_"):
        return re.sub(r"_[a-d]\.pkl$", "", f)
    m = re.match(r"HUST_(\d+)-\d+\.pkl$", f)
    if m:
        return "b" + str(int(m.group(1)))
    raise ValueError(f"unknown cell file name: {f}")


def calibration_unit_of(f):
    """Exchangeable unit in CV+: the campaign, except for HUST, where each cell is its own unit (a HUST training set has
    only 8 campaigns, too few units for a finite 90 % interval)."""
    return f if f.startswith("HUST") else campaign_of(f)


def delta_curves(curves):
    """[N, 100, 3, 300] -> [N, 100, 6, 300]: (V, I, Q) and (V, I, Q) minus a reference cycle: cycle 10 for cycles >= 10,
    cycle 1 for cycles 1-9 (1-based). A prefix of h cycles therefore only uses cycles it has seen."""
    c = np.asarray(curves)
    ref = np.where((np.arange(c.shape[1]) >= REF_CYCLE - 1)[None, :, None, None], c[:, REF_CYCLE - 1:REF_CYCLE], c[:, 0:1])
    return np.concatenate([c, c - ref], axis=2).astype(c.dtype)


def load_cache(data_dir, subsets, delta=False):
    """Concatenate the cache files of `subsets` (in the order given) into one array dict."""
    zs = [np.load(os.path.join(data_dir, f"{s}.npz")) for s in subsets]
    merged = {k: np.concatenate([z[k] for z in zs]) for k in ("curves", "files", "eol")}
    if delta:
        merged["curves"] = delta_curves(merged["curves"])
    return merged


class Data:
    """Curves on the device, y = log(end-of-life cycle), and cell-name lookup."""

    def __init__(self, z, device):
        import torch
        self.files = np.array([str(f) for f in z["files"]])
        self.idx = {f: i for i, f in enumerate(self.files)}
        self.y = np.log(z["eol"].astype(np.float64))
        self.curves = torch.as_tensor(z["curves"], device=device)
        self.device = device

    def ids(self, files):
        return np.array([self.idx[f] for f in files], dtype=np.int64)

    def batch(self, cells, hist):
        """Inputs of `cells` truncated to `hist` cycles (unseen cycles set to zero) and the cycle mask."""
        import torch
        c = torch.as_tensor(cells, device=self.device); L = torch.as_tensor(hist, device=self.device)
        mask = (torch.arange(ECT, device=self.device)[None, :] < L[:, None]).float()
        return self.curves[c] * mask[:, :, None, None], mask
