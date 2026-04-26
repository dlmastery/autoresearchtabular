"""Simple MLP for tabular binary classification — Gorishniy 2021 baseline."""
from __future__ import annotations
import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from .registry import register_backbone


class _MLPNet(nn.Module):
    def __init__(self, in_dim: int = 28, hidden: List[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden = hidden or [256, 256, 256]
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@register_backbone("mlp")
class MLP:
    framework = "torch"

    def __init__(self, hidden=None, dropout: float = 0.1,
                 epochs: int = 100, patience: int = 16,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 batch_size: int = 1024, optimizer: str = "AdamW",
                 scheduler: str = "cosine", precision: str = "bf16",
                 in_dim: int = 28, **_):
        self.hidden = hidden or [256, 256, 256]
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.precision = precision
        self.in_dim = in_dim
        self.model: _MLPNet | None = None
        self._best_state = None

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        from sklearn.metrics import roc_auc_score
        device = self._device()
        self.model = _MLPNet(self.in_dim, self.hidden, self.dropout).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        crit = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
        Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)) if X_val is not None else None
        yv_np = np.asarray(y_val) if y_val is not None else None

        bs = self.batch_size
        n = len(Xt)
        best_val = -1e9
        no_improve = 0
        history = []
        t0 = time.time()
        rng = np.random.default_rng(0)
        use_amp = self.precision == "bf16" and device.type == "cuda"
        for ep in range(1, self.epochs + 1):
            self.model.train()
            order = rng.permutation(n)
            losses, n_seen = 0.0, 0
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = self.model(xb)
                        loss = crit(logits, yb)
                else:
                    logits = self.model(xb)
                    loss = crit(logits, yb)
                loss.backward()
                opt.step()
                losses += float(loss.item()) * len(idx)
                n_seen += len(idx)
            sched.step()
            tr_loss = losses / max(1, n_seen)
            # val
            val_auc = float("nan")
            if Xv is not None and yv_np is not None and len(np.unique(yv_np)) > 1:
                self.model.eval()
                ps = []
                with torch.no_grad():
                    for i in range(0, len(Xv), 4096):
                        xb = Xv[i:i + 4096].to(device, non_blocking=True)
                        ps.append(torch.sigmoid(self.model(xb)).float().cpu().numpy())
                p = np.concatenate(ps)
                val_auc = float(roc_auc_score(yv_np, p))
            history.append({"epoch": ep, "train_loss": tr_loss, "val_auc": val_auc})
            improved = val_auc > best_val + 1e-6
            if improved:
                best_val = val_auc
                self._best_state = {k: v.detach().cpu().clone()
                                     for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"  epoch {ep:>3}/{self.epochs} train_loss={tr_loss:.4f} "
                  f"val_auc={val_auc:.4f}{'  *best*' if improved else ''}", flush=True)
            if no_improve >= self.patience:
                print(f"  early stop at epoch {ep} (patience {self.patience})")
                break
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return {"history": history, "best_val_auc": best_val,
                "epochs_run": history[-1]["epoch"] if history else 0,
                "train_time_s": time.time() - t0}

    def predict_proba(self, X) -> np.ndarray:
        device = self._device()
        self.model.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        ps = []
        with torch.no_grad():
            for i in range(0, len(Xt), 4096):
                xb = Xt[i:i + 4096].to(device, non_blocking=True)
                ps.append(torch.sigmoid(self.model(xb)).float().cpu().numpy())
        return np.concatenate(ps)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                     "config": {"hidden": self.hidden, "dropout": self.dropout,
                                "in_dim": self.in_dim}}, path)
