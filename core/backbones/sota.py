"""SOTA tabular backbones (April 2026): TabM, FT-Transformer, MLP-PLR,
ExcelFormer-lite, TabPFN-v2 probe.

All wrappers train on full 10M Higgs rows by default (GPU, BF16, batch
4096+), with paper-cited hyperparameters as the registry default.
"""
from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .registry import register_backbone


# =====================================================================
# Common training utilities for big-data SOTA training
# =====================================================================

def _make_loaders(X_train, y_train, X_val, y_val, batch_size: int,
                   device: torch.device, pin_memory: bool = True):
    """Numpy → tensor → simple index-shuffle DataLoader (in-memory)."""
    Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
    yv = np.asarray(y_val)
    return Xt, yt, Xv, yv


def _train_torch_binary(
    *, model: nn.Module, X_train, y_train, X_val, y_val,
    epochs: int, patience: int, lr: float, weight_decay: float,
    batch_size: int, scheduler: str = "cosine",
    optimizer_name: str = "AdamW", precision: str = "bf16",
    grad_clip: float = 1.0, label: str = "?",
    eval_batch: int = 16384,
) -> Dict[str, Any]:
    """Generic GPU+BF16 training loop returning best state-dict + history.

    `model.forward(x)` must return a single logit per sample (squeeze any
    ensemble axis with `.mean(dim=...)`). For TabM (k-ensemble heads) the
    wrapper class subclasses nn.Module and averages the k heads in
    `forward()`.
    """
    from sklearn.metrics import roc_auc_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = (precision == "bf16") and (device.type == "cuda")

    Xt, yt, Xv, yv = _make_loaders(X_train, y_train, X_val, y_val,
                                    batch_size, device)

    if optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                            factor=0.5,
                                                            patience=3)
    else:
        sched = None
    crit = nn.BCEWithLogitsLoss()

    n = len(Xt)
    best_val = -1e9
    no_improve = 0
    best_state = None
    history: List[Dict[str, Any]] = []
    rng = np.random.default_rng(0)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses, n_seen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = Xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    # TabM-style k-head outputs (B, k): broadcast y per head
                    if logits.dim() == 2 and logits.shape[1] != yb.shape[0]:
                        yb_b = yb.unsqueeze(-1).expand(-1, logits.shape[1])
                        loss = crit(logits, yb_b)
                    else:
                        loss = crit(logits, yb)
            else:
                logits = model(xb)
                if logits.dim() == 2 and logits.shape[1] != yb.shape[0]:
                    yb_b = yb.unsqueeze(-1).expand(-1, logits.shape[1])
                    loss = crit(logits, yb_b)
                else:
                    loss = crit(logits, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses += float(loss.item()) * len(idx)
            n_seen += len(idx)
        if scheduler == "cosine":
            sched.step()
        tr_loss = losses / max(1, n_seen)
        # val
        val_auc = float("nan")
        if len(np.unique(yv)) > 1:
            model.eval()
            ps: List[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, len(Xv), eval_batch):
                    xb = Xv[i:i + eval_batch].to(device, non_blocking=True)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = model(xb)
                    else:
                        out = model(xb)
                    # Average k-head ensemble at inference per TabM paper
                    if out.dim() == 2 and out.shape[1] not in (1,):
                        # (B, k) — sigmoid each head then average probs
                        probs = torch.sigmoid(out).mean(dim=1)
                    else:
                        probs = torch.sigmoid(out)
                    ps.append(probs.float().cpu().numpy())
            p = np.concatenate(ps)
            val_auc = float(roc_auc_score(yv, p))
        history.append({"epoch": ep, "train_loss": tr_loss, "val_auc": val_auc})
        improved = val_auc > best_val + 1e-6
        if improved:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if scheduler == "plateau":
            sched.step(val_auc)
        print(f"  [{label}] ep {ep:>3}/{epochs}  loss={tr_loss:.4f}  "
              f"val_auc={val_auc:.4f}{'  *best*' if improved else ''}",
              flush=True)
        if no_improve >= patience:
            print(f"  [{label}] early stop at epoch {ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_auc": best_val,
            "epochs_run": history[-1]["epoch"] if history else 0,
            "train_time_s": time.time() - t0}


def _predict_torch_binary(model: nn.Module, X, *,
                           batch_size: int = 16384,
                           use_amp: bool = True) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    ps: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            xb = Xt[i:i + batch_size].to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(xb)
            else:
                out = model(xb)
            if out.dim() == 2 and out.shape[1] != 1:
                # (B, k) k-head ensemble — sigmoid each then average
                probs = torch.sigmoid(out).mean(dim=1)
            else:
                probs = torch.sigmoid(out.squeeze(-1) if out.dim() > 1 else out)
            ps.append(probs.float().cpu().numpy())
    return np.concatenate(ps)


# =====================================================================
# TabM (Gorishniy 2025 ICLR — arXiv:2410.24210)
# =====================================================================

class _TabMBinaryWrapper(nn.Module):
    """TabM with k ensemble heads + optional PLR numerical embeddings.

    During TRAINING the model returns shape (B, k) — each head's logits
    are independent so the BCE loss is applied to all k heads against
    the broadcast y target. During INFERENCE we average the k head
    probabilities (per the official example.ipynb) — implemented in
    `predict_proba` outside this module.
    """

    def __init__(self, n_num_features: int, k: int = 32,
                 hidden: Optional[List[int]] = None,
                 dropout: float = 0.1,
                 use_plr: bool = True,
                 plr_n_frequencies: int = 48,
                 plr_d_embedding: int = 64,
                 plr_lite: bool = False,
                 arch_type: str = "tabm"):
        super().__init__()
        from tabm import TabM
        from rtdl_num_embeddings import PeriodicEmbeddings
        self.k = k
        h = hidden or [512, 512, 512]
        num_embeddings = None
        if use_plr:
            num_embeddings = PeriodicEmbeddings(
                n_features=n_num_features,
                n_frequencies=plr_n_frequencies,
                d_embedding=plr_d_embedding,
                lite=plr_lite,
            )
        # Use TabM.make() factory which auto-resolves start_scaling_init
        # ('normal' if num_embeddings else 'random-signs').
        self.model = TabM.make(
            n_num_features=n_num_features,
            d_out=1,
            num_embeddings=num_embeddings,
            n_blocks=len(h),
            d_block=h[0],
            dropout=dropout,
            k=k,
            arch_type=arch_type,
        )

    def forward(self, x):
        # TabM expects x_num positional or kwarg; pass numeric via x_num
        out = self.model(x)  # (B, k, 1)
        return out.squeeze(-1)  # (B, k) — train loss broadcasts y


@register_backbone("tabm")
class TabMBackbone:
    framework = "torch"

    def __init__(self,
                 hidden: Optional[List[int]] = None,
                 dropout: float = 0.1, k: int = 32,
                 use_plr: bool = True,
                 plr_n_frequencies: int = 48,
                 plr_d_embedding: int = 64,
                 plr_lite: bool = False,
                 arch_type: str = "tabm",
                 epochs: int = 100, patience: int = 16,
                 lr: float = 2e-3, weight_decay: float = 1e-5,
                 batch_size: int = 4096, optimizer: str = "AdamW",
                 scheduler: str = "cosine", precision: str = "bf16",
                 in_dim: int = 28, **_):
        self.hidden = hidden or [512, 512, 512]
        self.dropout = float(dropout)
        self.k = int(k)
        self.use_plr = bool(use_plr)
        self.plr_n_frequencies = int(plr_n_frequencies)
        self.plr_d_embedding = int(plr_d_embedding)
        self.plr_lite = bool(plr_lite)
        self.arch_type = str(arch_type)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.precision = precision
        self.in_dim = int(in_dim)
        self.model: Optional[nn.Module] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        in_dim = X_train.shape[1]
        self.in_dim = in_dim
        self.model = _TabMBinaryWrapper(
            n_num_features=in_dim, k=self.k,
            hidden=self.hidden, dropout=self.dropout,
            use_plr=self.use_plr,
            plr_n_frequencies=self.plr_n_frequencies,
            plr_d_embedding=self.plr_d_embedding,
            plr_lite=self.plr_lite,
            arch_type=self.arch_type,
        )
        return _train_torch_binary(
            model=self.model, X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=self.epochs, patience=self.patience,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, scheduler=self.scheduler,
            optimizer_name=self.optimizer, precision=self.precision,
            label=f"tabm-k{self.k}",
        )

    def predict_proba(self, X) -> np.ndarray:
        return _predict_torch_binary(self.model, X, batch_size=16384,
                                      use_amp=(self.precision == "bf16"))

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                     "config": {"hidden": self.hidden,
                                 "dropout": self.dropout,
                                 "k": self.k,
                                 "in_dim": self.in_dim}}, path)


# =====================================================================
# FT-Transformer (Gorishniy 2021 NeurIPS — arXiv:2106.11189)
# =====================================================================

class _FTTransformerWrapper(nn.Module):
    def __init__(self, n_num_features: int, *,
                 n_blocks: int = 3, d_block: int = 192,
                 attention_n_heads: int = 8,
                 attention_dropout: float = 0.2,
                 ffn_d_hidden_multiplier: float = 4.0 / 3.0,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0):
        super().__init__()
        from rtdl_revisiting_models import FTTransformer
        self.model = FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(self, x):
        return self.model(x_cont=x, x_cat=None).squeeze(-1)


@register_backbone("ft_transformer")
class FTTransformerBackbone:
    framework = "torch"

    def __init__(self,
                 n_blocks: int = 3, d_block: int = 192,
                 attention_n_heads: int = 8,
                 attention_dropout: float = 0.2,
                 ffn_d_hidden_multiplier: float = 4.0 / 3.0,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0,
                 epochs: int = 100, patience: int = 16,
                 lr: float = 1e-4, weight_decay: float = 1e-5,
                 batch_size: int = 1024, optimizer: str = "AdamW",
                 scheduler: str = "cosine", precision: str = "bf16",
                 in_dim: int = 28, **_):
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.attention_n_heads = int(attention_n_heads)
        self.attention_dropout = float(attention_dropout)
        self.ffn_d_hidden_multiplier = float(ffn_d_hidden_multiplier)
        self.ffn_dropout = float(ffn_dropout)
        self.residual_dropout = float(residual_dropout)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.precision = precision
        self.in_dim = int(in_dim)
        self.model: Optional[nn.Module] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        in_dim = X_train.shape[1]
        self.in_dim = in_dim
        self.model = _FTTransformerWrapper(
            n_num_features=in_dim,
            n_blocks=self.n_blocks, d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )
        return _train_torch_binary(
            model=self.model, X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=self.epochs, patience=self.patience,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, scheduler=self.scheduler,
            optimizer_name=self.optimizer, precision=self.precision,
            label=f"ft-d{self.d_block}b{self.n_blocks}",
        )

    def predict_proba(self, X) -> np.ndarray:
        return _predict_torch_binary(self.model, X, batch_size=8192,
                                      use_amp=(self.precision == "bf16"))

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                     "config": {"n_blocks": self.n_blocks,
                                 "d_block": self.d_block,
                                 "in_dim": self.in_dim}}, path)


# =====================================================================
# MLP-PLR (Gorishniy 2022 ICLR — arXiv:2203.05556)
# =====================================================================

class _MLPPLRWrapper(nn.Module):
    def __init__(self, n_num_features: int, *,
                 plr_n_frequencies: int = 48,
                 plr_frequency_init_scale: float = 0.01,
                 plr_d_embedding: int = 64,
                 plr_lite: bool = False,
                 hidden: Optional[List[int]] = None,
                 dropout: float = 0.1):
        super().__init__()
        from rtdl_num_embeddings import PeriodicEmbeddings
        self.embeddings = PeriodicEmbeddings(
            n_features=n_num_features,
            n_frequencies=plr_n_frequencies,
            frequency_init_scale=plr_frequency_init_scale,
            d_embedding=plr_d_embedding,
            lite=plr_lite,
        )
        d_in = n_num_features * plr_d_embedding
        layers: List[nn.Module] = []
        prev = d_in
        for h in (hidden or [512, 512, 512]):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        e = self.embeddings(x)
        e = e.flatten(1)
        return self.head(e).squeeze(-1)


@register_backbone("mlp_plr")
class MLPPLRBackbone:
    framework = "torch"

    def __init__(self,
                 hidden: Optional[List[int]] = None, dropout: float = 0.1,
                 plr_n_frequencies: int = 48,
                 plr_frequency_init_scale: float = 0.01,
                 plr_d_embedding: int = 64, plr_lite: bool = False,
                 epochs: int = 100, patience: int = 16,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 batch_size: int = 4096, optimizer: str = "AdamW",
                 scheduler: str = "cosine", precision: str = "bf16",
                 in_dim: int = 28, **_):
        self.hidden = hidden or [512, 512, 512]
        self.dropout = float(dropout)
        self.plr_n_frequencies = int(plr_n_frequencies)
        self.plr_frequency_init_scale = float(plr_frequency_init_scale)
        self.plr_d_embedding = int(plr_d_embedding)
        self.plr_lite = bool(plr_lite)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.precision = precision
        self.in_dim = int(in_dim)
        self.model: Optional[nn.Module] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        in_dim = X_train.shape[1]
        self.in_dim = in_dim
        self.model = _MLPPLRWrapper(
            n_num_features=in_dim,
            plr_n_frequencies=self.plr_n_frequencies,
            plr_frequency_init_scale=self.plr_frequency_init_scale,
            plr_d_embedding=self.plr_d_embedding,
            plr_lite=self.plr_lite,
            hidden=self.hidden, dropout=self.dropout,
        )
        return _train_torch_binary(
            model=self.model, X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=self.epochs, patience=self.patience,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, scheduler=self.scheduler,
            optimizer_name=self.optimizer, precision=self.precision,
            label=f"plr-h{self.hidden[0]}",
        )

    def predict_proba(self, X) -> np.ndarray:
        return _predict_torch_binary(self.model, X, batch_size=16384,
                                      use_amp=(self.precision == "bf16"))

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                     "config": {"hidden": self.hidden,
                                 "in_dim": self.in_dim,
                                 "plr_d_embedding": self.plr_d_embedding,
                                 "plr_n_frequencies": self.plr_n_frequencies}},
                    path)


# =====================================================================
# ResNet-tabular (Gorishniy 2021 NeurIPS — arXiv:2106.11189)
# =====================================================================

class _ResNetWrapper(nn.Module):
    def __init__(self, n_num_features: int, *,
                 n_blocks: int = 2, d_block: int = 256,
                 d_hidden_multiplier: float = 2.0,
                 dropout1: float = 0.25, dropout2: float = 0.0):
        super().__init__()
        from rtdl_revisiting_models import ResNet
        self.model = ResNet(
            d_in=n_num_features, d_out=1, n_blocks=n_blocks,
            d_block=d_block,
            d_hidden=None,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1, dropout2=dropout2,
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


@register_backbone("resnet_tabular")
class ResNetTabularBackbone:
    framework = "torch"

    def __init__(self,
                 n_blocks: int = 2, d_block: int = 256,
                 d_hidden_multiplier: float = 2.0,
                 dropout1: float = 0.25, dropout2: float = 0.0,
                 epochs: int = 100, patience: int = 16,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 batch_size: int = 4096, optimizer: str = "AdamW",
                 scheduler: str = "cosine", precision: str = "bf16",
                 in_dim: int = 28, **_):
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.d_hidden_multiplier = float(d_hidden_multiplier)
        self.dropout1 = float(dropout1)
        self.dropout2 = float(dropout2)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.precision = precision
        self.in_dim = int(in_dim)
        self.model: Optional[nn.Module] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        in_dim = X_train.shape[1]
        self.in_dim = in_dim
        self.model = _ResNetWrapper(
            n_num_features=in_dim, n_blocks=self.n_blocks,
            d_block=self.d_block, d_hidden_multiplier=self.d_hidden_multiplier,
            dropout1=self.dropout1, dropout2=self.dropout2,
        )
        return _train_torch_binary(
            model=self.model, X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=self.epochs, patience=self.patience,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, scheduler=self.scheduler,
            optimizer_name=self.optimizer, precision=self.precision,
            label=f"resnet-d{self.d_block}b{self.n_blocks}",
        )

    def predict_proba(self, X) -> np.ndarray:
        return _predict_torch_binary(self.model, X, batch_size=16384,
                                      use_amp=(self.precision == "bf16"))

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                     "config": {"n_blocks": self.n_blocks,
                                 "d_block": self.d_block,
                                 "in_dim": self.in_dim}}, path)
