"""Walk-forward training pipeline for MarketTransformer.

Training strategy:
- Expanding window: train on [0, T], validate on [T, T+val_size]
- Early stopping on validation loss
- Save best model checkpoint

Usage:
    from ml.trainer import train_market_transformer
    result = train_market_transformer(repo, market="cn")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch required: pip install torch")

from .features import (
    FeatureConfig,
    FeatureMatrix,
    build_feature_matrix,
    create_sequences,
    normalize_features,
    FEATURE_NAMES,
)
from .model import (
    MarketTransformer,
    MultiTaskLoss,
    TransformerConfig,
    compute_class_weights,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    lookback: int = 60
    forward_horizon: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10  # early stopping patience
    val_ratio: float = 0.15  # validation split ratio
    n_splits: int = 3  # number of walk-forward splits
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    reward_mode: bool = False  # enable trading-aligned reward monitoring

    # Model config
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.15


@dataclass
class TrainResult:
    """Training result summary."""

    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    n_train_samples: int
    n_val_samples: int
    n_epochs: int
    model_path: str
    predictions: dict[str, Any] = field(default_factory=dict)
    regime_accuracy: float = 0.0
    return_mae: float = 0.0


def _get_device(cfg: TrainConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def _load_market_data(repo: Any, market: str) -> tuple[Any, Any | None]:
    """Load close and volume matrices from repository."""
    import pandas as pd

    rows = repo.conn.execute(
        """
        SELECT trade_date, symbol,
               COALESCE(adj_close, close) AS close,
               volume
        FROM daily_bars
        WHERE market = ?
        ORDER BY trade_date, symbol
        """,
        [market],
    ).fetchdf()

    if rows.empty:
        raise ValueError(f"No daily bars found for market={market}")

    rows["trade_date"] = pd.to_datetime(rows["trade_date"])
    close_matrix = rows.pivot_table(
        index="trade_date", columns="symbol", values="close", aggfunc="first"
    )
    volume_matrix = rows.pivot_table(
        index="trade_date", columns="symbol", values="volume", aggfunc="first"
    )
    # Convert index to date
    close_matrix.index = [d.date() if hasattr(d, "date") else d for d in close_matrix.index]
    volume_matrix.index = [d.date() if hasattr(d, "date") else d for d in volume_matrix.index]

    return close_matrix, volume_matrix


def _compute_regime_labels(close_matrix: Any) -> dict[date, str]:
    """Compute regime labels from close matrix using RegimeDetector."""
    from research.regime_detector import RegimeDetector, RegimeConfig

    detector = RegimeDetector(config=RegimeConfig())
    signals = detector.detect_all(close_matrix)
    return {s.trade_date: s.regime for s in signals}


def train_market_transformer(
    repo: Any,
    *,
    market: str = "cn",
    config: TrainConfig = TrainConfig(),
    on_progress: Any = None,
) -> TrainResult:
    """Train a MarketTransformer with walk-forward validation.

    Args:
        repo: MarketDataRepository instance.
        market: Market code ("cn" or "us").
        config: Training configuration.
        on_progress: Optional progress callback.

    Returns:
        TrainResult with metrics and model path.
    """
    device = _get_device(config)
    logger.info("Training MarketTransformer: market=%s, device=%s", market, device)

    def _emit(phase: str, pct: float, msg: str):
        if on_progress:
            on_progress({"phase": phase, "pct": pct, "message": msg})

    # Step 1: Load data
    _emit("load_data", 0.05, "加载行情数据…")
    close_matrix, volume_matrix = _load_market_data(repo, market)
    logger.info("Close matrix: %d dates x %d symbols", close_matrix.shape[0], close_matrix.shape[1])

    # Step 2: Compute regime labels
    _emit("regime", 0.10, "计算 Regime 标签…")
    regime_by_date = _compute_regime_labels(close_matrix)
    logger.info("Regime labels: %d dates", len(regime_by_date))

    # Step 3: Build feature matrix
    _emit("features", 0.20, "提取市场特征…")
    feat_config = FeatureConfig(forward_horizons=(5, 10, 20))
    fm = build_feature_matrix(
        close_matrix, volume_matrix, regime_by_date, config=feat_config
    )
    logger.info("Feature matrix: %d samples x %d features", fm.n_samples, fm.n_features)

    # Step 4: Create sequences
    X, y_ret, y_reg, seq_dates = create_sequences(
        fm, lookback=config.lookback, forward_horizon=config.forward_horizon
    )
    logger.info("Sequences: %d (lookback=%d)", len(X), config.lookback)

    # Step 5: Walk-forward splits (expanding window)
    n = len(X)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Generate walk-forward split indices
    n_splits = min(config.n_splits, n // 2)
    min_train_size = max(n // 3, config.lookback * 2)
    val_size = max(int(n * config.val_ratio), 50)

    best_overall_loss = float("inf")
    best_overall_state: dict | None = None
    best_overall_epoch = 0
    best_fold_config: dict = {}

    for split_idx in range(n_splits):
        # Expanding window: train on [0, train_end), validate on [train_end, train_end + val_size)
        train_end = min_train_size + split_idx * (n - min_train_size) // n_splits
        train_end = min(train_end, n - val_size - 1)
        val_end = min(train_end + val_size, n)

        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)

        X_train_raw, y_ret_train, y_reg_train = X[train_idx], y_ret[train_idx], y_reg[train_idx]
        X_val_raw, y_ret_val, y_reg_val = X[val_idx], y_ret[val_idx], y_reg[val_idx]

        # Per-fold normalization: compute stats on TRAINING fold only, apply to both
        X_train, fold_mean, fold_std = normalize_features(X_train_raw)
        X_val, _, _ = normalize_features(X_val_raw, mean=fold_mean, std=fold_std)

        logger.info(
            "Fold %d/%d: train=%d (date=%s), val=%d (date=%s)",
            split_idx + 1, n_splits,
            len(X_train), seq_dates[train_end - 1] if train_end > 1 else "?",
            len(X_val), seq_dates[val_end - 1] if val_end > 1 else "?",
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_ret_train),
            torch.tensor(y_reg_train),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_ret_val),
            torch.tensor(y_reg_val),
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Build model
        model_config = TransformerConfig(
            n_features=fm.n_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_len=config.lookback + 10,
        )
        model = MarketTransformer(model_config).to(device)

        class_weights = compute_class_weights(y_reg_train).to(device)
        criterion = MultiTaskLoss(
            return_weight=1.0,
            regime_weight=0.5,
            regime_class_weights=class_weights,
            reward_mode=config.reward_mode,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

        # Training loop for this fold
        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict | None = None

        for epoch in range(config.max_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_ret, batch_reg in train_loader:
                batch_X = batch_X.to(device)
                batch_ret = batch_ret.to(device)
                batch_reg = batch_reg.to(device)

                preds = model(batch_X)
                losses = criterion(preds, batch_ret, batch_reg, primary_horizon="5", dates=seq_dates)

                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += losses["total"].item()
                n_batches += 1

            scheduler.step()

            # Validate
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_X, batch_ret, batch_reg in val_loader:
                    batch_X = batch_X.to(device)
                    batch_ret = batch_ret.to(device)
                    batch_reg = batch_reg.to(device)

                    preds = model(batch_X)
                    losses = criterion(preds, batch_ret, batch_reg, primary_horizon="5")
                    val_loss += losses["total"].item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch - best_epoch >= config.patience:
                break

        # Track best across folds
        if best_val_loss < best_overall_loss:
            best_overall_loss = best_val_loss
            best_overall_state = best_state
            best_overall_epoch = best_epoch
            best_fold_config = {
                "fold_mean": fold_mean.squeeze().tolist(),
                "fold_std": fold_std.squeeze().tolist(),
                "n_train": len(X_train),
                "n_val": len(X_val),
            }

    # Step 6: Load best model across all folds
    model_config = TransformerConfig(
        n_features=fm.n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_seq_len=config.lookback + 10,
    )
    model = MarketTransformer(model_config).to(device)
    if best_overall_state:
        model.load_state_dict(best_overall_state)
    model.eval()

    # Evaluate on last fold's validation set
    # Use last fold's normalized data for final eval
    all_regime_preds = []
    all_regime_true = []
    all_ret_preds = []
    all_ret_true = []

    with torch.no_grad():
        for batch_X, batch_ret, batch_reg in val_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)

            regime_pred = preds["regime_logits"].argmax(dim=-1).cpu().numpy()
            all_regime_preds.extend(regime_pred.tolist())
            all_regime_true.extend(batch_reg.numpy().tolist())

            ret_pred = preds["return_5"].cpu().numpy()
            all_ret_preds.extend(ret_pred.tolist())
            all_ret_true.extend(batch_ret.numpy().tolist())

    valid_pairs = [(p, t) for p, t in zip(all_regime_preds, all_regime_true) if t >= 0]
    regime_acc = sum(1 for p, t in valid_pairs if p == t) / len(valid_pairs) if valid_pairs else 0.0
    ret_mae = float(np.mean(np.abs(np.array(all_ret_preds) - np.array(all_ret_true))))

    # Step 7: Save model with fold-specific normalization stats
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"market_transformer_{market}_latest.pt"
    save_dict = {
        "model_state": best_overall_state or model.state_dict(),
        "config": {
            "n_features": fm.n_features,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
            "max_seq_len": config.lookback + 10,
            "lookback": config.lookback,
            "feature_names": list(FEATURE_NAMES),
            "market": market,
        },
        "feat_mean": best_fold_config.get("fold_mean", [0.0] * fm.n_features),
        "feat_std": best_fold_config.get("fold_std", [1.0] * fm.n_features),
        "metrics": {
            "best_val_loss": best_overall_loss,
            "regime_accuracy": regime_acc,
            "return_mae": ret_mae,
            "n_epochs": best_overall_epoch + 1,
            "n_splits": n_splits,
            "n_train": best_fold_config.get("n_train", 0),
            "n_val": best_fold_config.get("n_val", 0),
        },
    }
    torch.save(save_dict, model_path)
    logger.info("Model saved to %s (best across %d folds)", model_path, n_splits)

    # Generate latest prediction
    _emit("predict", 0.85, "生成最新预测…")
    from .model import predict_with_model

    # Normalize latest features using best fold's statistics
    latest_raw = X[-1]  # (lookback, n_features) — raw features
    latest_mean = np.array(best_fold_config.get("fold_mean", [0.0] * fm.n_features))
    latest_std = np.array(best_fold_config.get("fold_std", [1.0] * fm.n_features))
    latest_features = (latest_raw - latest_mean) / np.where(latest_std < 1e-8, 1.0, latest_std)
    latest_features = np.nan_to_num(latest_features, nan=0.0)
    latest_pred = predict_with_model(model, latest_features, device=device)

    return TrainResult(
        best_val_loss=best_overall_loss,
        train_losses=[],
        val_losses=[],
        n_train_samples=best_fold_config.get("n_train", 0),
        n_val_samples=best_fold_config.get("n_val", 0),
        n_epochs=best_overall_epoch + 1,
        model_path=str(model_path),
        predictions=latest_pred,
        regime_accuracy=regime_acc,
        return_mae=ret_mae,
    )


def load_trained_model(
    model_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[MarketTransformer, dict]:
    """Load a trained model and its metadata.

    Returns:
        (model, metadata_dict)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model_config = TransformerConfig(
        n_features=config["n_features"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )
    model = MarketTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    metadata = {
        "config": config,
        "feat_mean": np.array(checkpoint["feat_mean"]),
        "feat_std": np.array(checkpoint["feat_std"]),
        "metrics": checkpoint.get("metrics", {}),
    }
    return model, metadata
