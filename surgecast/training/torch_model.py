"""Small MLP used only as a comparison against LightGBM.

The network is not the served model. Serving loads the LightGBM champion from
the MLflow registry. Numeric inputs are standardized with the training-split
mean and standard deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surgecast.features.compute import FEATURE_COLUMNS
from surgecast.training.metrics import regression_scores


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


class _Net:
    """Placeholder so type checkers are not required to import torch at module import."""


def train_mlp(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    horizon: str,
    epochs: int = 6,
    batch_size: int = 8192,
    seed: int = 7,
) -> dict:
    torch, nn = _torch()
    torch.manual_seed(seed)
    numeric_cols = [col for col in FEATURE_COLUMNS if col != "zone_id"]
    y_col = "y15" if horizon == "15" else "y30"

    mean = train[numeric_cols].to_numpy(dtype=np.float32).mean(axis=0)
    std = train[numeric_cols].to_numpy(dtype=np.float32).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    def pack(frame: pd.DataFrame):
        zone = torch.from_numpy(frame["zone_id"].to_numpy(dtype=np.int64))
        num = (frame[numeric_cols].to_numpy(dtype=np.float32) - mean) / std
        numeric = torch.from_numpy(np.ascontiguousarray(num))
        target = torch.from_numpy(frame[y_col].to_numpy(dtype=np.float32))
        return zone, numeric, target

    class DemandNet(nn.Module):
        def __init__(self, n_numeric: int, n_zones: int = 264, emb: int = 8) -> None:
            super().__init__()
            self.emb = nn.Embedding(n_zones, emb)
            self.net = nn.Sequential(
                nn.Linear(n_numeric + emb, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, zone, numeric):
            zone = zone.clamp(min=0, max=self.emb.num_embeddings - 1)
            hidden = torch.cat([self.emb(zone), numeric], dim=1)
            return self.net(hidden).squeeze(1)

    model = DemandNet(len(numeric_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    zone_t, num_t, y_t = pack(train)
    zone_v, num_v, y_v = pack(val) if len(val) else (None, None, None)
    n = zone_t.shape[0]
    best_val = None
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(zone_t[idx], num_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * idx.shape[0]
            seen += idx.shape[0]
        row = {"epoch": epoch + 1, "train_l1": total / max(seen, 1)}
        if zone_v is not None and zone_v.shape[0]:
            model.eval()
            with torch.no_grad():
                val_pred = model(zone_v, num_v)
                val_l1 = float(loss_fn(val_pred, y_v))
            row["val_l1"] = val_l1
            if best_val is None or val_l1 < best_val:
                best_val = val_l1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    zone_te, num_te, _ = pack(test)
    with torch.no_grad():
        pred = model(zone_te, num_te).numpy()
    scores = regression_scores(test[y_col].to_numpy(), pred)
    return {
        "horizon_min": int(horizon),
        "epochs_requested": epochs,
        "best_val_l1": best_val,
        "history": history,
        "scores": scores,
        "numeric_columns": numeric_cols,
    }
