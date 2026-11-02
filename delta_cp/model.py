"""Official BatteryLife backbones (CPMLP, CPTransformer) with a quantile head on standardised log life.

* CPTransformer: `output_num = len(QUANTILES)` (the model's own argument).
* CPMLP: its single-output `head_output` is replaced after construction by a Linear with len(QUANTILES) outputs.
* Delta variant: the first layer (`intra_embed`, Linear(300 * 3 -> d_model)) is replaced by Linear(300 * 6 -> d_model) so
  that the backbone takes the six Delta-curve channels; nothing else changes.
Training follows the official recipe (one sample per history length 1..100, unseen cycles zeroed, Adam, constant learning
rate, official batch size x 2 GPUs) with the pinball loss instead of the MSE on raw life.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .data import CDL, ECT

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "batterylife")
if THIRD_PARTY not in sys.path:
    sys.path.insert(0, THIRD_PARTY)
try:
    import reformer_pytorch  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(THIRD_PARTY, "_stubs"))
from models import CPMLP, CPTransformer  # noqa: E402

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)   # alpha = 0.1 uses (0.05, 0.95), alpha = 0.2 uses (0.10, 0.90)
QIDX = {0.1: (0, 4), 0.2: (1, 3)}
CONFIGS = {  # official training scripts; batch = per-GPU batch x 2 GPUs
    "CPMLP": dict(d_model=128, d_ff=256, e_layers=4, d_layers=2, n_heads=8, dropout=0.0, lr=5e-5, batch_size=32),
    "CPTransformer": dict(d_model=128, d_ff=256, e_layers=6, d_layers=4, n_heads=4, dropout=0.0, lr=5e-5, batch_size=64),
}
COMMON = dict(charge_discharge_length=CDL, early_cycle_threshold=ECT, seq_len=1, factor=3, activation="relu",
              max_epochs=100, patience=5, least_epochs=5)
VARIANTS = {"CPT": ("CPTransformer", False), "CPT-Delta": ("CPTransformer", True),
            "CPMLP": ("CPMLP", False), "CPMLP-Delta": ("CPMLP", True)}


def build_model(name, device, delta=False):
    cfg = dict(COMMON, **CONFIGS[name], output_num=len(QUANTILES))
    if name == "CPTransformer":
        m = CPTransformer.Model(SimpleNamespace(**cfg))
    else:
        m = CPMLP.Model(SimpleNamespace(**cfg))
        m.head_output = nn.Linear(cfg["d_model"], len(QUANTILES))
    m = m.float().to(device)
    if delta:
        m.intra_embed = nn.Linear(cfg["charge_discharge_length"] * 6, cfg["d_model"]).float().to(device)
    return m, cfg


def pinball(pred, y):
    q = torch.as_tensor(QUANTILES, device=pred.device)[None, :]
    d = y - pred
    return torch.maximum(q * d, (q - 1) * d).mean()


def all_prefix_samples(cells):
    return np.array([(c, i) for c in cells for i in range(1, ECT + 1)], dtype=np.int64)


def train_backbone(name, D, train_cells, seed, delta=False, epochs=None, val_cells=None):
    """epochs=None: early stopping on val_cells (pinball over all prefixes; patience counted after least_epochs);
    epochs=int: train exactly that many epochs. Returns (model, (mu, sd) of the training log life, epochs used)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model, cfg = build_model(name, D.device, delta)
    mu, sd = D.y[train_cells].mean(), D.y[train_cells].std() + 1e-8
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    S = all_prefix_samples(train_cells)
    ys = torch.as_tensor((D.y[S[:, 0]] - mu) / sd, dtype=torch.float32, device=D.device)[:, None]
    gen = torch.Generator().manual_seed(seed)
    n_ep = epochs if epochs is not None else cfg["max_epochs"]
    best, best_ep, best_state, counter = np.inf, 0, None, 0
    for ep in range(1, n_ep + 1):
        model.train()
        perm = torch.randperm(len(S), generator=gen).numpy()
        for k in range(0, len(perm), cfg["batch_size"]):
            b = perm[k:k + cfg["batch_size"]]
            x, m = D.batch(S[b, 0], S[b, 1])
            opt.zero_grad(); loss = pinball(model(x, m), ys[b]); loss.backward(); opt.step()
        if epochs is None:
            v = val_loss(model, D, val_cells, mu, sd)
            if v < best:
                best, best_ep, counter = v, ep, 0
                best_state = {k_: t.detach().clone() for k_, t in model.state_dict().items()}
            else:
                if ep > cfg["least_epochs"]:
                    counter += 1
                if counter >= cfg["patience"]:
                    break
    if epochs is None:
        model.load_state_dict(best_state)
        return model, (mu, sd), best_ep
    return model, (mu, sd), n_ep


@torch.no_grad()
def val_loss(model, D, cells, mu, sd, bs=1024):
    model.eval(); S = all_prefix_samples(cells); tot = 0.0
    for k in range(0, len(S), bs):
        x, m = D.batch(S[k:k + bs, 0], S[k:k + bs, 1])
        y = torch.as_tensor((D.y[S[k:k + bs, 0]] - mu) / sd, dtype=torch.float32, device=D.device)[:, None]
        tot += pinball(model(x, m), y).item() * len(x)
    return tot / len(S)


@torch.no_grad()
def predict(model, D, cells, hist, scale, bs=512):
    """Sorted quantiles of log life [n, 5] at history length `hist`."""
    model.eval(); mu, sd = scale; Q = []
    for k in range(0, len(cells), bs):
        c = cells[k:k + bs]
        x, m = D.batch(c, np.full(len(c), hist))
        p = model(x, m)
        Q.append(torch.sort(p.float(), dim=1)[0].cpu().numpy() * sd + mu)
    return np.concatenate(Q)
