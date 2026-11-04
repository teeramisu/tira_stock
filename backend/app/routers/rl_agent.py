"""
RL-MACS API routes: training, inference, backtest, agent status.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# 保证可导入 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rl", tags=["RL-MACS"])

# ---------- Lazy state ----------
_inference_engine = None
_training_status: dict[str, Any] = {"status": "idle", "progress": 0.0}
_training_lock = threading.Lock()


def _get_inference_engine():
    """Lazy-init the RL inference engine."""
    global _inference_engine  # noqa: PLW0603
    if _inference_engine is not None:
        return _inference_engine

    from rl.inference import RLInferenceEngine

    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "rl", "models")
    model_path = os.path.join(model_dir, "rl_macs_ppo.zip")

    if os.path.exists(model_path):
        _inference_engine = RLInferenceEngine(model_path=model_path)
    else:
        _inference_engine = RLInferenceEngine()  # No model loaded yet

    return _inference_engine


# ---------- Request/Response Models ----------


class TrainRequest(BaseModel):
    """Request to start RL training."""

    tickers: list[str] = Field(
        default=["SPY", "QQQ", "IWM"],
        description="Ticker symbols to train on",
    )
    lookback_years: int = Field(default=5, ge=1, le=20)
    warmup_steps: int = Field(default=50_000, ge=1000)
    main_steps: int = Field(default=500_000, ge=10_000)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2)
    n_envs: int = Field(default=4, ge=1, le=16)


class PredictRequest(BaseModel):
    """Request for single-step prediction."""

    tickers: list[str] = Field(default=["SPY", "QQQ", "IWM"])
    lookback_days: int = Field(default=120, ge=60, le=500)


class BacktestRequest(BaseModel):
    """Request for RL backtest."""

    tickers: list[str] = Field(default=["SPY", "QQQ", "IWM"])
    lookback_years: int = Field(default=3, ge=1, le=10)
    initial_cash: float = Field(default=1_000_000.0, ge=10_000)
    commission_rate: float = Field(default=0.001, ge=0, le=0.01)


# ---------- Endpoints ----------


@router.get("/status")
def get_rl_status():
    """Get RL-MACS system status."""
    engine = _get_inference_engine()
    engine_status = engine.get_status()

    return {
        "system": "RL-MACS",
        "version": "0.1.0",
        "training": _training_status,
        "inference": engine_status,
        "agents": {
            "factor": {"dim": 8, "status": "ready"},
            "macro": {"dim": 6, "status": "ready"},
            "sentiment": {"dim": 6, "status": "ready"},
            "technical": {"dim": 6, "status": "ready"},
        },
        "total_opinion_dim": 26,
    }


@router.post("/train")
def start_training(req: TrainRequest):
    """Start RL training (runs in background thread)."""
    global _training_status  # noqa: PLW0603

    with _training_lock:
        if _training_status.get("status") == "running":
            raise HTTPException(status_code=409, detail="Training already in progress")

    def _run_training():
        global _training_status  # noqa: PLW0603
        try:
            _training_status = {"status": "running", "progress": 0.0, "phase": "data_loading"}

            # Load market data
            from marketdata.data_portal import DataPortal

            portal = DataPortal()
            price_data = {}
            for ticker in req.tickers:
                try:
                    df = portal.get_ohlcv(ticker, period=f"{req.lookback_years}y")
                    if df is not None and len(df) > 100:
                        price_data[ticker] = df["Close"].values.astype(float)
                except Exception as e:
                    logger.warning(f"Failed to load {ticker}: {e}")

            if len(price_data) < 1:
                _training_status = {"status": "failed", "error": "No valid price data loaded"}
                return

            # Generate synthetic market features and regime labels for training
            n_steps = max(len(v) for v in price_data.values())
            import numpy as np

            market_features = np.random.randn(n_steps, 10).astype(np.float32) * 0.1
            regime_labels = np.random.randint(0, 4, size=n_steps).astype(np.int64)

            _training_status = {"status": "running", "progress": 0.1, "phase": "precomputing_opinions"}

            # Pre-compute agent opinions
            from rl.trainer import TrainingConfig, RLTrainer, precompute_agent_opinions

            opinions = precompute_agent_opinions(price_data, market_features, regime_labels)

            _training_status = {"status": "running", "progress": 0.2, "phase": "training"}

            # Train
            config = TrainingConfig(
                warmup_steps=req.warmup_steps,
                main_steps=req.main_steps,
                learning_rate=req.learning_rate,
                n_envs=req.n_envs,
            )
            trainer = RLTrainer(config=config)
            result = trainer.train(
                price_data=price_data,
                market_features=market_features,
                regime_labels=regime_labels,
                agent_opinions=opinions,
            )

            _training_status = {
                "status": "completed",
                "progress": 1.0,
                "model_path": result.model_path,
                "total_steps": result.total_steps,
                "training_time": result.training_time_seconds,
                "best_sharpe": result.best_sharpe,
            }
        except Exception as e:
            logger.exception("Training failed")
            _training_status = {"status": "failed", "error": str(e)}

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {"message": "Training started", "status": "running"}


@router.post("/predict")
def predict(req: PredictRequest):
    """Run single-step RL prediction."""
    engine = _get_inference_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run /api/rl/train first.",
        )

    try:
        import numpy as np
        from marketdata.data_portal import DataPortal

        portal = DataPortal()
        price_data = {}
        for ticker in req.tickers:
            df = portal.get_ohlcv(ticker, period="1y")
            if df is not None and len(df) > 60:
                price_data[ticker] = df["Close"].values.astype(float)

        if not price_data:
            raise HTTPException(status_code=400, detail="Could not load price data")

        n_steps = max(len(v) for v in price_data.values())
        market_features = np.zeros((n_steps, 10), dtype=np.float32)
        regime_labels = np.ones(n_steps, dtype=np.int64)  # Default RANGE

        result = engine.predict_with_context(
            price_data=price_data,
            market_features=market_features,
            regime_labels=regime_labels,
            step=n_steps - 1,
        )

        return {
            "positions": result.positions.tolist(),
            "regime": result.regime,
            "confidence": result.confidence,
            "agent_opinions": {
                k: v.tolist() for k, v in result.agent_opinions.items()
            },
            "risk_info": result.risk_info,
            "tickers": sorted(price_data.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
def run_backtest(req: BacktestRequest):
    """Run RL backtest over historical period."""
    engine = _get_inference_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run /api/rl/train first.",
        )

    try:
        import numpy as np
        from marketdata.data_portal import DataPortal

        portal = DataPortal()
        price_data = {}
        for ticker in req.tickers:
            df = portal.get_ohlcv(ticker, period=f"{req.lookback_years}y")
            if df is not None and len(df) > 100:
                price_data[ticker] = df["Close"].values.astype(float)

        if not price_data:
            raise HTTPException(status_code=400, detail="Could not load price data")

        n_steps = max(len(v) for v in price_data.values())
        market_features = np.zeros((n_steps, 10), dtype=np.float32)
        regime_labels = np.ones(n_steps, dtype=np.int64)

        result = engine.run_backtest(
            price_data=price_data,
            market_features=market_features,
            regime_labels=regime_labels,
            initial_cash=req.initial_cash,
            commission_rate=req.commission_rate,
        )

        return {
            "total_return": round(result.total_return, 6),
            "annual_return": round(result.annual_return, 6),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 6),
            "volatility": round(result.volatility, 6),
            "win_rate": round(result.win_rate, 4),
            "n_trades": result.n_trades,
            "total_costs": round(result.total_costs, 4),
            "regime_distribution": result.regime_distribution,
            "equity_curve_length": len(result.equity_curve),
            "tickers": sorted(price_data.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/status")
def get_agents_status():
    """Get detailed status of all RL information agents."""
    return {
        "agents": [
            {
                "name": "factor",
                "description": "Multi-factor signal agent",
                "opinion_dim": 8,
                "factors": [
                    "momentum", "value", "quality", "volatility",
                    "liquidity", "size", "reversal", "factor_momentum",
                ],
                "status": "ready",
            },
            {
                "name": "macro",
                "description": "Macro-economic environment agent",
                "opinion_dim": 6,
                "factors": [
                    "regime_strength", "trend_persistence", "vol_regime",
                    "breadth_momentum", "cross_market", "liquidity_env",
                ],
                "status": "ready",
            },
            {
                "name": "sentiment",
                "description": "Market sentiment agent",
                "opinion_dim": 6,
                "factors": [
                    "news_proxy", "fear_index", "participation",
                    "capital_sentiment", "rotation_heat", "sentiment_momentum",
                ],
                "status": "ready",
            },
            {
                "name": "technical",
                "description": "Technical analysis agent",
                "opinion_dim": 6,
                "factors": [
                    "trend_strength", "ma_alignment", "rsi",
                    "macd", "bollinger", "volume_confirmation",
                ],
                "status": "ready",
            },
        ],
        "decision_agent": {
            "architecture": "PPO + Safety-Critic",
            "hidden_dims": [256, 128, 64],
            "safety_threshold": 0.3,
            "status": "ready" if _get_inference_engine().is_ready else "no_model",
        },
        "total_opinion_dim": 26,
    }
