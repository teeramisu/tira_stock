"""CAMT Training — fast version without stock data (architecture-only test)."""

import sys, logging, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

try:
    import torch, torch.nn as nn
except ImportError:
    raise ImportError("PyTorch required")

from ml.model_v2 import CAMT, CAMTConfig
from ml.model import MultiTaskLoss, compute_class_weights
from ml.features import FeatureConfig

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def train_camt_fast(repo, market="cn", d_model=128, n_layers=3, n_heads=8, d_ff=256,
                    lookback=60, batch_size=48, lr=1e-3, max_epochs=30, patience=8, dropout=0.12):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"CAMT-fast on {device} | d={d_model} L={n_layers} H={n_heads} LB={lookback}")

    from ml.trainer import _load_market_data, _compute_regime_labels
    from ml.features import build_feature_matrix, create_sequences

    close_pivot, vol_pivot = _load_market_data(repo, market)
    regime_by_date = _compute_regime_labels(close_pivot)

    fc = FeatureConfig(forward_horizons=(5, 10, 20), min_stocks=30)
    fm = build_feature_matrix(close_pivot, vol_pivot, regime_by_date, config=fc)
    X, y_ret, y_reg, dates = create_sequences(fm, lookback=lookback, forward_horizon=5)

    n = len(X)
    val_size = max(n // 4, 50)
    train_end = n - val_size
    torch.manual_seed(42); np.random.seed(42)

    X_train, X_val = X[:train_end], X[train_end:]
    y_ret_train, y_ret_val = y_ret[:train_end], y_ret[train_end:]
    y_reg_train, y_reg_val = y_reg[:train_end], y_reg[train_end:]

    cfg = CAMTConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
                     dropout=dropout, max_seq_len=lookback+10, n_top_stocks=0)
    model = CAMT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params: {n_params:,}")

    cw = compute_class_weights(y_reg_train).to(device)
    criterion = MultiTaskLoss(regime_class_weights=cw, reward_mode=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_train))[:batch_size * 15]
        total_loss, nb = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            bx = torch.tensor(X_train[idx.cpu().numpy()], dtype=torch.float32, device=device)
            byr = torch.tensor(y_ret_train[idx.cpu().numpy()], dtype=torch.float32, device=device)
            byg = torch.tensor(y_reg_train[idx.cpu().numpy()], dtype=torch.long, device=device)
            preds = model(bx, None)  # no stock data
            losses = criterion(preds, byr, byg, primary_horizon="5")
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += losses["total"].item()
            nb += 1
        sch.step()

        # Validate
        model.eval()
        val_idx = np.arange(len(X_val))[:800]
        with torch.no_grad():
            bx = torch.tensor(X_val[val_idx], dtype=torch.float32, device=device)
            preds = model(bx, None)
            vl = criterion(preds,
                           torch.tensor(y_ret_val[val_idx], dtype=torch.float32, device=device),
                           torch.tensor(y_reg_val[val_idx], dtype=torch.long, device=device),
                           primary_horizon="5")
        vloss = vl["total"].item()
        if vloss < best_val_loss:
            best_val_loss = vloss; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience: break

    model.load_state_dict(best_state or model.state_dict())
    model.eval()
    vi = np.arange(len(X_val))
    bx = torch.tensor(X_val[vi], dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(bx, None)
        rp = preds["regime_logits"].argmax(dim=-1).cpu().numpy()
        rt = y_reg_val[vi]
        v = rt >= 0
        acc = (rp[v] == rt[v]).mean() if v.any() else 0.0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state or model.state_dict(),
        "metrics": {"regime_accuracy": float(acc), "val_loss": best_val_loss, "n_epochs": best_epoch+1},
        "config": {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "lookback": lookback},
    }, MODEL_DIR / "camt_latest.pt")

    return {"regime_accuracy": float(acc), "val_loss": best_val_loss, "n_epochs": best_epoch+1, "params": n_params}
