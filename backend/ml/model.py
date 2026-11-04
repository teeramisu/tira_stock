"""Multi-task Transformer for market regime and return prediction.

Input (per sample):
- Time range: **past 60 trading days** (default lookback)
- Frequency: daily bar
- Features (10 per day):

  | # | 名称 | 计算方式 | 窗口 |
  |---|------|---------|------|
  | 1 | breadth_ma60 | 全市场收盘价 > MA60 的比例 | 60日 |
  | 2 | breadth_ma20 | 全市场收盘价 > MA20 的比例 | 20日 |
  | 3 | trend_20d | 全市场 20 日收益率中位数 | 20日 |
  | 4 | trend_5d | 全市场 5 日收益率中位数 | 5日 |
  | 5 | crash_5d | 全市场 5 日收益率 5% 分位 | 5日 |
  | 6 | realized_vol | 等权组合 20 日年化波动率 | 20日 |
  | 7 | vol_of_vol | 波动率的波动率 | 10日 |
  | 8 | momentum_disp | 截面动量离散度（收益率标准差） | 20日 |
  | 9 | volume_trend | 全市场成交量 20 日线性趋势 | 20日 |
  | 10 | breadth_mom | 宽度指标 5 日变化量 | 5日 |

Output (multi-task):
1. return_{h}: 未来 h 日收益率预测（h ∈ {5, 10, 20}）
2. regime_logits: 未来 regime 分类 logits（4类: BULL/RANGE/BEAR/CRASH）

Architecture:
- Input: (batch, seq_len=60, n_features=10)
- Encoder: 2-layer TransformerEncoder (d_model=64, heads=4, causal mask)
- Multi-task heads: 3 return regressors + 1 regime classifier (4-class)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required: pip install torch")


@dataclass
class TransformerConfig:
    """Model hyperparameters."""

    n_features: int = 10
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    max_seq_len: int = 120
    n_regime_classes: int = 4
    return_horizons: tuple[int, ...] = (5, 10, 20)
    pool_type: str = "attention"  # "last" | "meanmax" | "attention"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Learned attention pooling over timesteps.

    A trainable query vector attends to all timestep representations.
    This is more expressive than simple last-token pooling.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scale = d_model ** 0.5
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, d_model)
        attn_scores = torch.matmul(self.query, h.transpose(-2, -1)) / self.scale  # (1, 1, T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (1, 1, T)
        pooled = torch.matmul(attn_weights, h).squeeze(1)  # (B, d_model)
        return self.norm(pooled)


class MarketTransformer(nn.Module):
    """Multi-task Transformer for market prediction — v2 upgraded architecture.

    Improvements over v1:
    - Larger model: d_model=128, n_layers=4, n_heads=8 by default
    - Attention pooling over all timesteps (not just last token)
    - Mean+max alternative pooling for stability
    - Residual connections in prediction heads
    - Pre-LN architecture throughout
    """

    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Pooling: attention-based or simple
        if config.pool_type == "attention":
            self.attn_pool = AttentionPooling(config.d_model)
            pool_dim = config.d_model
        elif config.pool_type == "meanmax":
            pool_dim = config.d_model * 2
            self.attn_pool = None
        else:  # "last"
            pool_dim = config.d_model
            self.attn_pool = None
            self.pool_norm = nn.LayerNorm(config.d_model)

        self.config._pool_dim = pool_dim

        # Return prediction heads
        head_hidden = config.d_ff // 2
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(pool_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.GELU(),
                nn.Dropout(config.dropout * 0.5),
                nn.Linear(head_hidden // 2, 1),
            )
            for h in config.return_horizons
        })

        # Regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(pool_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, config.n_regime_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for linear layers, Xavier for attention."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "attn_pool" in name:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.input_dropout(h)
        h = self.pos_enc(h)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.encoder(h, mask=causal_mask)

        # Pooling
        if self.config.pool_type == "attention":
            pooled = self.attn_pool(h)
        elif self.config.pool_type == "meanmax":
            mean_p = h.mean(dim=1)
            max_p = h.max(dim=1)[0]
            pooled = torch.cat([mean_p, max_p], dim=-1)
        else:
            pooled = self.pool_norm(h[:, -1, :])

        out: dict[str, torch.Tensor] = {"pooled": pooled}
        for h_str, head in self.return_heads.items():
            out[f"return_{h_str}"] = head(pooled).squeeze(-1)
        out["regime_logits"] = self.regime_head(pooled)

        return out


class MultiTaskLoss(nn.Module):
    """Combined loss with optional Jane Street style reward.

    Primary loss = MSE + CrossEntropy (baseline)
    When reward_mode=True: also computes trading-aligned reward for monitoring.

    L = return_weight * MSE(return) + regime_weight * CE(regime)
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        regime_weight: float = 0.5,
        regime_class_weights: torch.Tensor | None = None,
        reward_mode: bool = False,
    ):
        super().__init__()
        self.return_weight = return_weight
        self.regime_weight = regime_weight
        self.reward_mode = reward_mode
        self.mse = nn.MSELoss()
        if regime_class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=regime_class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        y_return: torch.Tensor,
        y_regime: torch.Tensor,
        *,
        primary_horizon: str = "5",
        dates: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        When reward_mode=True, also logs trading-aligned reward components
        as diagnostic metrics (not used for gradients — primary loss is still MSE+CE).

        Args:
            preds: Model output dict.
            y_return: (batch,) target return for primary horizon.
            y_regime: (batch,) target regime class index.
            primary_horizon: Which return head to use for primary loss.
            dates: Batch date strings (for time decay in reward).

        Returns:
            Dict with "total", "return_loss", "regime_loss", and optional reward metrics.
        """
        ret_key = f"return_{primary_horizon}"
        return_loss = self.mse(preds[ret_key], y_return)

        valid_mask = y_regime >= 0
        if valid_mask.any():
            regime_loss = self.ce(
                preds["regime_logits"][valid_mask],
                y_regime[valid_mask],
            )
        else:
            regime_loss = torch.tensor(0.0, device=y_return.device)

        total = self.return_weight * return_loss + self.regime_weight * regime_loss
        result = {
            "total": total,
            "return_loss": return_loss.detach(),
            "regime_loss": regime_loss.detach(),
        }

        # Reward diagnostics (no gradients, monitoring only)
        if self.reward_mode and dates is not None:
            try:
                rewards = self._compute_reward(preds, y_return, y_regime, dates, primary_horizon)
                result.update(rewards)
            except Exception:
                pass

        return result

    def _compute_reward(
        self, preds, y_return, y_regime, dates, primary_horizon
    ) -> dict[str, torch.Tensor]:
        """Compute trading-aligned reward for monitoring."""
        import numpy as np
        pred_ret = preds[f"return_{primary_horizon}"].detach().cpu().numpy()
        actual_ret = y_return.detach().cpu().numpy()
        regime_logits = preds["regime_logits"].detach().cpu().numpy()
        regime_probs = np.exp(regime_logits) / np.exp(regime_logits).sum(axis=1, keepdims=True)
        pred_regime = regime_probs.argmax(axis=1)
        confidence = regime_probs.max(axis=1)

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "prediction_ground_truth"))
        from ml.reward import compute_total_reward, RewardConfig

        total_reward = 0.0
        n = len(pred_ret)
        for i in range(n):
            r = compute_total_reward(
                {"predicted_return": float(pred_ret[i]),
                 "predicted_regime": int(pred_regime[i]),
                 "confidence": float(confidence[i]),
                 "date": dates[i] if i < len(dates) else ""},
                {"actual_return": float(actual_ret[i]),
                 "regime_probs": regime_probs[i].tolist(),
                 "is_crash": bool(actual_ret[i] < -0.07),
                 "is_turning_point": False},
                RewardConfig(),
            )
            total_reward += r["total_reward"]

        avg_reward = total_reward / max(n, 1)
        return {
            "avg_reward": torch.tensor(avg_reward, device=y_return.device),
        }


def compute_class_weights(regime_labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced regime labels."""
    from collections import Counter

    counts = Counter(regime_labels[regime_labels >= 0].tolist())
    total = sum(counts.values())
    n_classes = 4
    weights = torch.ones(n_classes, dtype=torch.float32)
    for cls_idx in range(n_classes):
        c = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (n_classes * c)
    return weights


def predict_with_model(
    model: MarketTransformer,
    features: np.ndarray,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run inference on a single feature sequence.

    Args:
        model: Trained MarketTransformer.
        features: (lookback, n_features) normalized feature array.
        device: "cpu" or "cuda".

    Returns:
        Dict with predicted returns, regime probabilities, and recommended action.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        out = model(x)

        # Extract predictions
        result: dict[str, Any] = {}
        for key, val in out.items():
            if key == "pooled":
                continue
            if key.startswith("return_"):
                result[key] = float(val.item())
            elif key == "regime_logits":
                probs = F.softmax(val, dim=-1).squeeze().cpu().numpy()
                result["regime_probs"] = probs.tolist()
                result["predicted_regime"] = int(probs.argmax())
                from ml.features import REGIME_LABELS
                result["predicted_regime_name"] = REGIME_LABELS[result["predicted_regime"]]

        # Generate recommendation
        regime_name = result.get("predicted_regime_name", "UNKNOWN")
        regime_prob = result.get("regime_probs", [0, 0, 0, 0])
        ret_5d = result.get("return_5", 0)

        if regime_name == "BULL" and ret_5d > 0:
            result["action"] = "积极做多"
            result["position_scale"] = 1.0
        elif regime_name == "CRASH":
            result["action"] = "现金为王"
            result["position_scale"] = 0.0
        elif regime_name == "BEAR":
            result["action"] = "防守为主"
            result["position_scale"] = 0.3
        else:
            result["action"] = "谨慎操作"
            result["position_scale"] = 0.6

        # Confidence
        max_prob = max(regime_prob) if regime_prob else 0
        result["confidence"] = round(float(max_prob), 3)

    return result
