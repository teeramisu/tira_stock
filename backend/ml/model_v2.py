"""Cross-Attention Market Transformer (CAMT) — dual-tower architecture.

Architecture:
  Market Tower:   Self-attention on 10 market-level features per day
  Stock Tower:    Encodes individual stock data (sampled from top N stocks)  
  Cross-Attention: Market queries attend to stock-level keys/values
  Fusion + Prediction: Combined features → return + regime heads

Key improvements over standard Transformer:
  1. Stock-level information flow via cross-attention
  2. Gated feature selection (learn which features matter per timestep)
  3. Multi-resolution: separate encoders for 5d/20d/60d horizons
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required: pip install torch")


@dataclass
class CAMTConfig:
    n_market_features: int = 10
    n_stock_features: int = 5   # open, high, low, close, volume
    n_top_stocks: int = 50      # how many individual stocks to sample
    stock_seq_len: int = 20     # how many days of stock data to feed

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 120
    n_regime_classes: int = 4
    return_horizons: tuple[int, ...] = (5, 10, 20)

    pool_type: str = "meanmax"


class PositionalEncoding(nn.Module):
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

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeatureGating(nn.Module):
    """Learn which input features matter at each timestep."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

    def forward(self, x):
        # x: (B, T, F)
        weights = self.gate(x).sigmoid()
        return x * weights


class StockEncoder(nn.Module):
    """Encodes individual stock OHLCV data and pools across stocks."""
    def __init__(self, cfg: CAMTConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_stock_features, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=500)
        self.dropout = nn.Dropout(cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x: (B, n_stocks, T_stock, n_features)
        B, S, T, F = x.shape
        x = x.view(B * S, T, F)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.dropout(h)
        h = self.encoder(h)
        h = h[:, -1, :]  # (B*S, d_model) — last timestep pool
        return h.view(B, S, -1)  # (B, n_stocks, d_model)


class CAMT(nn.Module):
    """Cross-Attention Market Transformer."""
    def __init__(self, config: CAMTConfig = CAMTConfig()):
        super().__init__()
        self.config = config

        # Market tower
        self.market_gate = FeatureGating(config.n_market_features, config.d_model)
        self.market_proj = nn.Linear(config.n_market_features, config.d_model)
        self.market_norm = nn.LayerNorm(config.d_model)
        self.market_pos = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        market_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(market_layer, num_layers=config.n_layers)

        # Stock tower
        self.stock_encoder = StockEncoder(config)

        # Cross-attention: market → stocks
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=config.n_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.cross_dropout = nn.Dropout(config.dropout)

        # Pooling
        if config.pool_type == "meanmax":
            pool_dim = config.d_model * 3  # market mean+max + cross-attn pooled
        else:
            pool_dim = config.d_model * 2  # market last + cross-attn pooled
        self.config._pool_dim = pool_dim

        head_hidden = config.d_ff // 2
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(pool_dim, head_hidden),
                nn.LayerNorm(head_hidden), nn.GELU(), nn.Dropout(config.dropout),
                nn.Linear(head_hidden, head_hidden // 2), nn.GELU(),
                nn.Linear(head_hidden // 2, 1),
            ) for h in config.return_horizons
        })
        self.regime_head = nn.Sequential(
            nn.Linear(pool_dim, head_hidden),
            nn.LayerNorm(head_hidden), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.GELU(),
            nn.Linear(head_hidden // 2, config.n_regime_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, market_x: torch.Tensor, stock_x: torch.Tensor = None) -> dict[str, torch.Tensor]:
        B, T, _ = market_x.shape

        # Market tower
        mh = self.market_gate(market_x)
        mh = self.market_proj(mh)
        mh = self.market_norm(mh)
        mh = self.market_pos(mh)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=market_x.device)
        mh = self.market_encoder(mh, mask=causal_mask)  # (B, T, d_model)

        # Market pooling
        if self.config.pool_type == "meanmax":
            m_pooled = torch.cat([mh.mean(dim=1), mh.max(dim=1)[0]], dim=-1)
        else:
            m_pooled = mh[:, -1, :]

        # Cross-attention to stocks
        if stock_x is not None and stock_x.numel() > 0:
            sh = self.stock_encoder(stock_x)  # (B, S, d_model)
            # Market queries attend to stock keys/values
            # Take last few market timesteps as queries
            mq = mh[:, -4:, :]  # (B, 4, d_model) — last 4 market timesteps as queries
            ca_out, _ = self.cross_attn(mq, sh, sh)
            ca_pooled = ca_out.mean(dim=1)  # (B, d_model)
            ca_pooled = self.cross_norm(ca_pooled)
            ca_pooled = self.cross_dropout(ca_pooled)
        else:
            ca_pooled = torch.zeros(B, self.config.d_model, device=mh.device)

        # Fusion
        pooled = torch.cat([m_pooled, ca_pooled], dim=-1)

        out = {"pooled": pooled}
        for h_str, head in self.return_heads.items():
            out[f"return_{h_str}"] = head(pooled).squeeze(-1)
        out["regime_logits"] = self.regime_head(pooled)
        return out
