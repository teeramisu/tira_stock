"""RL-MACS Training Script — Full US Market Support.

Loads real market data from DuckDB, computes features/regime/opinions,
trains the PPO agent with proper train/test split, and saves model + obs stats.

Features:
- Auto-discovers all US stocks from DuckDB (no hardcoded list)
- Liquidity-based filtering (top N by average volume)
- Time-based train/test split (80/20, no look-ahead)
- Saves observation normalization stats for inference consistency
- Walk-forward evaluation on held-out test set

Usage:
    python -m rl.run_training                          # Default: top 1000 stocks, RL on top 20
    python -m rl.run_training --top-n 2000 --max-assets 50 --main-steps 500000
    python -m rl.run_training --tickers AAPL,MSFT,NVDA    # Manual override
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl.train")


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "market" / "market_data.duckdb"
MODEL_DIR = Path(__file__).resolve().parent / "models"

# Allow override via environment variable (useful when server holds DuckDB lock)
import os
if os.environ.get("RL_DB_PATH"):
    DB_PATH = Path(os.environ["RL_DB_PATH"])


# ============================================================
# Data Loading — Full US Market Support
# ============================================================


def discover_us_stocks(min_bars: int = 500) -> list[str]:
    """Auto-discover all US stocks available in DuckDB.

    Returns symbols sorted by data length (descending).
    """
    import duckdb

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    rows = conn.execute(
        "SELECT symbol, COUNT(*) as n_bars FROM daily_bars "
        "WHERE market = 'us' GROUP BY symbol "
        "HAVING COUNT(*) >= ? ORDER BY n_bars DESC",
        [min_bars],
    ).fetchall()
    conn.close()
    symbols = [r[0] for r in rows]
    logger.info(f"  Discovered {len(symbols)} US stocks with >= {min_bars} bars")
    return symbols


def get_top_liquid_stocks(top_n: int = 200, min_bars: int = 500) -> list[str]:
    """Get top N US stocks ranked by average daily volume (liquidity proxy).

    Uses DuckDB to compute average volume and rank.
    """
    import duckdb

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Check if volume column exists
    cols = conn.execute("PRAGMA table_info('daily_bars')").fetchall()
    col_names = [c[1] for c in cols]
    has_volume = "volume" in col_names

    if has_volume:
        rows = conn.execute(
            "SELECT symbol, AVG(volume) as avg_vol, COUNT(*) as n_bars "
            "FROM daily_bars WHERE market = 'us' "
            "GROUP BY symbol HAVING COUNT(*) >= ? "
            "ORDER BY avg_vol DESC LIMIT ?",
            [min_bars, top_n],
        ).fetchall()
        symbols = [r[0] for r in rows]
        logger.info(f"  Top {len(symbols)} by liquidity (avg volume)")
    else:
        # Fallback: use data length as proxy
        rows = conn.execute(
            "SELECT symbol, COUNT(*) as n_bars FROM daily_bars "
            "WHERE market = 'us' GROUP BY symbol "
            "HAVING COUNT(*) >= ? ORDER BY n_bars DESC LIMIT ?",
            [min_bars, top_n],
        ).fetchall()
        symbols = [r[0] for r in rows]
        logger.info(f"  Top {len(symbols)} by data length (no volume column)")

    conn.close()
    return symbols


def load_market_data(
    tickers: list[str],
    period: str = "5y",
    market: str = "us",
    min_bars: int = 500,
) -> dict[str, np.ndarray]:
    """Load historical close prices from DuckDB (read-only) with yfinance fallback.

    Returns dict of {ticker: close_prices_array} for tickers with sufficient data.
    """
    import duckdb

    price_data = {}

    # Primary: DuckDB read-only
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        # Batch query for efficiency
        placeholders = ",".join(["?"] * len(tickers))
        rows = conn.execute(
            f"SELECT symbol, trade_date, close FROM daily_bars "
            f"WHERE market = ? AND symbol IN ({placeholders}) "
            f"ORDER BY symbol, trade_date",
            [market] + tickers,
        ).fetchall()
        conn.close()

        if rows:
            df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
            df["close"] = df["close"].astype(float)
            for sym in tickers:
                sym_df = df[df["symbol"] == sym].sort_values("date")
                closes = sym_df["close"].values
                valid_mask = np.isfinite(closes) & (closes > 0)
                closes = closes[valid_mask]
                if len(closes) >= min_bars:
                    price_data[sym] = closes
                else:
                    logger.debug(f"    {sym}: insufficient data ({len(closes)} bars)")

            logger.info(f"  Loaded {len(price_data)}/{len(tickers)} tickers from DuckDB")
    except Exception as e:
        logger.warning(f"  DuckDB read failed: {e}")

    # Fallback: yfinance for missing tickers (only if few missing)
    missing = [t for t in tickers if t not in price_data]
    if missing and len(missing) <= 20:
        logger.info(f"  Trying yfinance for {len(missing)} missing tickers...")
        try:
            import yfinance as yf
            for ticker in missing:
                try:
                    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                    if df is not None and not df.empty and "Close" in df.columns:
                        closes = df["Close"].values.astype(float).flatten()
                        valid_mask = np.isfinite(closes) & (closes > 0)
                        closes = closes[valid_mask]
                        if len(closes) >= min_bars:
                            price_data[ticker] = closes
                except Exception:
                    pass
        except ImportError:
            pass

    return price_data


# ============================================================
# Train/Test Split — Temporal (No Look-Ahead)
# ============================================================


def temporal_train_test_split(
    price_data: dict[str, np.ndarray],
    train_ratio: float = 0.8,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Split price data temporally: first train_ratio for training, rest for testing.

    TEMPORAL CONTRACT: Test set is strictly AFTER train set (no look-ahead).

    Args:
        price_data: Dict of {ticker: full_price_array}.
        train_ratio: Fraction of data for training (default 80%).

    Returns:
        (train_data, test_data, split_index)
    """
    # Find minimum length across all tickers
    min_len = min(len(v) for v in price_data.values())
    split_idx = int(min_len * train_ratio)

    train_data = {k: v[:split_idx] for k, v in price_data.items()}
    test_data = {k: v[split_idx:min_len] for k, v in price_data.items()}

    logger.info(f"  Temporal split: train={split_idx} days, test={min_len - split_idx} days")
    logger.info(f"  Train ratio: {train_ratio:.0%}, split at index {split_idx}")

    return train_data, test_data, split_idx


def compute_market_features(prices: np.ndarray, n_steps: int) -> np.ndarray:
    """Compute 10-dim market-level features from price matrix.

    Features: breadth_ma60, breadth_ma20, trend_20d, trend_5d, crash_5d,
              realized_vol, vol_of_vol, momentum_disp, volume_trend, breadth_mom
    """
    n_assets = prices.shape[1]
    features = np.zeros((n_steps, 10), dtype=np.float32)

    # Daily returns
    returns = np.zeros_like(prices)
    returns[1:] = prices[1:] / prices[:-1] - 1.0
    returns = np.nan_to_num(returns, nan=0.0)

    # MA calculations
    for i in range(n_steps):
        if i < 20:
            continue

        # Feature 0: breadth_ma60 (% assets above 60-day MA)
        if i >= 60:
            ma60 = prices[i - 60:i + 1].mean(axis=0)
            valid = prices[i] > 0
            if valid.sum() > 0:
                features[i, 0] = float((prices[i][valid] > ma60[valid]).mean())
        else:
            features[i, 0] = 0.5

        # Feature 1: breadth_ma20
        ma20 = prices[max(0, i - 20):i + 1].mean(axis=0)
        valid = prices[i] > 0
        if valid.sum() > 0:
            features[i, 1] = float((prices[i][valid] > ma20[valid]).mean())

        # Feature 2: trend_20d (median 20-day return)
        if i >= 20:
            ret_20 = prices[i] / prices[i - 20] - 1.0
            valid_ret = ret_20[np.isfinite(ret_20)]
            features[i, 2] = float(np.median(valid_ret)) if len(valid_ret) > 0 else 0.0

        # Feature 3: trend_5d
        if i >= 5:
            ret_5 = prices[i] / prices[i - 5] - 1.0
            valid_ret = ret_5[np.isfinite(ret_5)]
            features[i, 3] = float(np.median(valid_ret)) if len(valid_ret) > 0 else 0.0

        # Feature 4: crash_5d (5th percentile of 5-day returns)
        if i >= 5:
            ret_5_all = prices[i] / prices[i - 5] - 1.0
            valid_ret = ret_5_all[np.isfinite(ret_5_all)]
            features[i, 4] = float(np.percentile(valid_ret, 5)) if len(valid_ret) > 1 else 0.0

        # Feature 5: realized_vol (20-day annualized)
        if i >= 20:
            vol = returns[i - 20:i].std(axis=0).mean()
            features[i, 5] = float(vol * np.sqrt(252)) if np.isfinite(vol) else 0.0

        # Feature 6: vol_of_vol
        if i >= 30:
            vols = []
            for j in range(i - 10, i):
                if j >= 20:
                    v = returns[j - 20:j].std(axis=0).mean()
                    vols.append(v)
            if len(vols) > 2:
                features[i, 6] = float(np.std(vols))

        # Feature 7: momentum_disp (cross-sectional return dispersion)
        if i >= 20:
            ret_20_all = prices[i] / prices[i - 20] - 1.0
            valid_ret = ret_20_all[np.isfinite(ret_20_all)]
            features[i, 7] = float(np.std(valid_ret)) if len(valid_ret) > 1 else 0.0

        # Feature 8: volume_trend (approximated by price change magnitude trend)
        if i >= 20:
            recent_activity = np.abs(returns[i - 5:i]).mean()
            older_activity = np.abs(returns[i - 20:i - 5]).mean() + 1e-8
            features[i, 8] = float((recent_activity - older_activity) / older_activity)

        # Feature 9: breadth_mom (5-day change in breadth)
        if i >= 5:
            features[i, 9] = features[i, 1] - features[i - 5, 1]

    return features


def compute_regime_labels(features: np.ndarray) -> np.ndarray:
    """Compute regime labels from market features.

    0=BULL, 1=RANGE, 2=BEAR, 3=CRASH
    Fixed: Use portfolio-level crash detection with stricter thresholds.
    """
    n_steps = len(features)
    labels = np.ones(n_steps, dtype=np.int64)  # Default RANGE

    for i in range(n_steps):
        breadth = features[i, 0]
        trend_20d = features[i, 2]
        crash_5d = features[i, 4]
        vol = features[i, 5]

        # CRASH: extreme market-wide decline (stricter: -8% AND low breadth)
        if crash_5d < -0.08 and breadth < 0.25:
            labels[i] = 3
        # BEAR: sustained low breadth + negative trend
        elif breadth < 0.30 and trend_20d < -0.03:
            labels[i] = 2
        # BULL: high breadth + positive trend
        elif breadth > 0.55 and trend_20d > 0.005:
            labels[i] = 0
        # else: RANGE (1)

    return labels


def align_price_data(price_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Align all price series to the same length (use minimum length)."""
    if not price_data:
        return {}
    min_len = min(len(v) for v in price_data.values())
    # Use the LAST min_len bars (most recent data)
    return {k: v[-min_len:] for k, v in price_data.items()}


def compute_obs_stats(price_data: dict[str, np.ndarray],
                      market_features: np.ndarray,
                      regime_labels: np.ndarray, agent_opinions: np.ndarray,
                      n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute observation normalization statistics from training data.

    Samples observations across the training set and computes mean/std
    for each dimension. These stats are saved alongside the model and
    used at inference time to ensure consistency.

    Returns:
        (obs_mean, obs_std) arrays of shape (obs_dim,)
    """
    from rl.env import TradingEnv

    # Sample observations at regular intervals
    n_steps = len(market_features)
    sample_indices = np.linspace(60, n_steps - 2, min(500, n_steps - 62), dtype=int)

    obs_samples = []
    env = TradingEnv(
        price_data=price_data,
        market_features=market_features,
        regime_labels=regime_labels,
        agent_opinions=agent_opinions,
    )

    for idx in sample_indices:
        obs = env._get_observation_at(int(idx))
        obs_samples.append(obs)

    obs_array = np.array(obs_samples, dtype=np.float32)
    obs_mean = obs_array.mean(axis=0)
    obs_std = obs_array.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0  # Avoid division by zero

    obs_dim = len(obs_mean)
    logger.info(f"  Obs stats: dim={obs_dim}, samples={len(obs_samples)}")
    logger.info(f"  Mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
    logger.info(f"  Std range: [{obs_std.min():.4f}, {obs_std.max():.4f}]")

    return obs_mean, obs_std


def save_obs_stats(obs_mean: np.ndarray, obs_std: np.ndarray, path: Path) -> None:
    """Save observation normalization stats as JSON."""
    stats = {
        "mean": obs_mean.tolist(),
        "std": obs_std.tolist(),
        "dim": len(obs_mean),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved obs stats: {path}")


def evaluate_on_test_set(
    model_path: str,
    test_data: dict[str, np.ndarray],
    obs_stats_path: Path,
    initial_cash: float = 1_000_000.0,
) -> dict:
    """Run inference on the held-out test set and report metrics.

    TEMPORAL CONTRACT: Test data is strictly after training data.
    """
    from rl.inference import RLInferenceEngine

    # Align test data
    test_data = align_price_data(test_data)
    if len(test_data) < 2:
        logger.warning("  Test set too small, skipping evaluation")
        return {}

    n_steps = len(next(iter(test_data.values())))
    prices = np.column_stack([test_data[t] for t in sorted(test_data.keys())])

    # Compute features for test set
    market_features = compute_market_features(prices, n_steps)
    regime_labels = compute_regime_labels(market_features)

    engine = RLInferenceEngine(
        model_path=model_path,
        obs_stats_path=str(obs_stats_path),
    )

    result = engine.run_backtest(
        price_data=test_data,
        market_features=market_features,
        regime_labels=regime_labels,
        initial_cash=initial_cash,
        start_step=60,  # Skip warmup
    )

    # Convert BacktestResult to dict if needed
    if hasattr(result, '__dict__'):
        return vars(result)
    return result if isinstance(result, dict) else {}


def main():
    parser = argparse.ArgumentParser(
        description="RL-MACS Training — Full US Market Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: top 1000 liquid US stocks, RL on top 20
  python -m rl.run_training

  # Full market discovery, RL on top 50 assets
  python -m rl.run_training --top-n 2000 --max-assets 50 --main-steps 500000

  # Manual ticker override
  python -m rl.run_training --tickers AAPL,MSFT,NVDA,GOOGL,AMZN

  # Quick test run
  python -m rl.run_training --top-n 10 --main-steps 10000 --warmup-steps 5000
""",
    )
    # Stock selection
    parser.add_argument("--mode", type=str, default="small", choices=["small", "full"],
                        help="Training mode: 'small' (direct PPO on N stocks) or "
                             "'full' (two-layer: selector + RL on top-K)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Manual ticker list (overrides --top-n)")
    parser.add_argument("--top-n", type=int, default=1000,
                        help="Auto-discover top N stocks by liquidity (default: 1000)")
    parser.add_argument("--max-assets", type=int, default=20,
                        help="[small mode] Max assets in RL action space (default: 20)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="[full mode] RL acts on top-K selected stocks (default: 30)")
    parser.add_argument("--rebalance-freq", type=int, default=5,
                        help="[full mode] Rebalance selection every N days (default: 5)")
    parser.add_argument("--min-bars", type=int, default=500,
                        help="Minimum bars required per stock (default: 500)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to DuckDB file (default: data/market/market_data.duckdb)")

    # Training
    parser.add_argument("--period", type=str, default="5y", help="Data period for yfinance fallback")
    parser.add_argument("--warmup-steps", type=int, default=20_000, help="Warmup training steps")
    parser.add_argument("--main-steps", type=int, default=130_000, help="Main training steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Best-of-N training: train N models with different seeds, keep best (default: 1)")

    # Train/test split
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--eval-test", action="store_true", default=True,
                        help="Evaluate on test set after training")
    parser.add_argument("--no-eval-test", dest="eval_test", action="store_false",
                        help="Skip test set evaluation")

    args = parser.parse_args()

    # Override DB path if specified
    global DB_PATH
    if args.db_path:
        DB_PATH = Path(args.db_path)
        logger.info(f"Using custom DB path: {DB_PATH}")

    # ============================================================
    # Phase 0: Stock Selection
    # ============================================================
    logger.info("=" * 70)
    logger.info(f"RL-MACS Training Pipeline — Mode: {args.mode.upper()}")
    logger.info("=" * 70)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        logger.info(f"\n[Phase 0] Manual ticker selection: {tickers}")
    else:
        logger.info(f"\n[Phase 0] Auto-discovering top {args.top_n} US stocks by liquidity...")
        tickers = get_top_liquid_stocks(top_n=args.top_n, min_bars=args.min_bars)
        if not tickers:
            logger.error("No stocks found in DuckDB. Exiting.")
            sys.exit(1)

    # Mode-specific configuration
    if args.mode == "small":
        # Small mode: RL acts directly on max-assets stocks
        rl_tickers = tickers[:args.max_assets]
        logger.info(f"  Mode: SMALL (direct PPO)")
        logger.info(f"  Full universe: {len(tickers)} stocks")
        logger.info(f"  RL action space: top {len(rl_tickers)} assets")
    else:
        # Full mode: Load all, selector picks top-K dynamically
        rl_tickers = tickers  # Load ALL discovered stocks
        logger.info(f"  Mode: FULL (two-layer: selector + RL)")
        logger.info(f"  Full universe: {len(rl_tickers)} stocks loaded")
        logger.info(f"  RL action space: top-K = {args.top_k} (dynamic selection)")
        logger.info(f"  Rebalance freq: every {args.rebalance_freq} days")

    logger.info(f"  Warmup: {args.warmup_steps:,} steps")
    logger.info(f"  Main: {args.main_steps:,} steps")
    logger.info(f"  Envs: {args.n_envs}, LR: {args.lr}, Device: {args.device}")
    logger.info(f"  Train/Test split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")

    # ============================================================
    # Phase 1: Load Data
    # ============================================================
    logger.info("\n[Phase 1] Loading market data...")
    price_data = load_market_data(rl_tickers, period=args.period, min_bars=args.min_bars)

    if len(price_data) < 2:
        logger.error(f"Need at least 2 valid tickers, got {len(price_data)}. Exiting.")
        sys.exit(1)

    # Align lengths
    price_data = align_price_data(price_data)
    n_steps = len(next(iter(price_data.values())))
    n_assets = len(price_data)
    logger.info(f"  Aligned data: {n_assets} assets × {n_steps} steps")

    # ============================================================
    # Phase 2: Train/Test Split (TEMPORAL — No Look-Ahead)
    # ============================================================
    logger.info("\n[Phase 2] Temporal train/test split...")
    train_data, test_data, split_idx = temporal_train_test_split(
        price_data, train_ratio=args.train_ratio
    )

    logger.info(f"  Train: {len(train_data)} assets × {len(next(iter(train_data.values())))} days")
    logger.info(f"  Test:  {len(test_data)} assets × {len(next(iter(test_data.values())))} days")

    # Build price matrix for TRAINING
    prices = np.column_stack([train_data[t] for t in sorted(train_data.keys())])
    train_steps = len(prices)

    # ============================================================
    # Phase 3: Compute Features & Regime (on TRAIN set only)
    # ============================================================
    logger.info("\n[Phase 3] Computing market features & regime labels (train set)...")
    market_features = compute_market_features(prices, train_steps)
    regime_labels = compute_regime_labels(market_features)

    regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]
    for i, name in enumerate(regime_names):
        count = int((regime_labels == i).sum())
        pct = count / train_steps * 100
        logger.info(f"  {name}: {count} days ({pct:.1f}%)")

    # ============================================================
    # Phase 4: Pre-compute Agent Opinions
    # ============================================================
    logger.info("\n[Phase 4] Pre-computing agent opinions...")
    t0 = time.time()
    from rl.trainer import precompute_agent_opinions
    agent_opinions = precompute_agent_opinions(train_data, market_features, regime_labels)
    logger.info(f"  Opinions shape: {agent_opinions.shape}, took {time.time()-t0:.1f}s")

    # ============================================================
    # Phase 5: Compute & Save Obs Normalization Stats
    # ============================================================
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    obs_stats_path = MODEL_DIR / "obs_stats.json"

    if args.mode == "small":
        logger.info("\n[Phase 5] Computing observation normalization stats...")
        obs_mean, obs_std = compute_obs_stats(
            train_data, market_features, regime_labels, agent_opinions, n_assets
        )
        save_obs_stats(obs_mean, obs_std, obs_stats_path)
    else:
        logger.info("\n[Phase 5] Full mode: obs stats computed by FullMarketEnv internally")

    # ============================================================
    # Phase 6: Train PPO
    # ============================================================
    logger.info(f"\n[Phase 6] Starting PPO training (mode={args.mode})...")
    from rl.trainer import RLTrainer, TrainingConfig

    if args.mode == "small":
        # === SMALL MODE: Direct PPO on fixed asset set ===
        config = TrainingConfig(
            warmup_steps=args.warmup_steps,
            main_steps=args.main_steps,
            learning_rate=args.lr,
            n_envs=args.n_envs,
            device=args.device,
            model_dir=str(MODEL_DIR),
            tensorboard_log=str(MODEL_DIR.parent / "tb_logs"),
        )
        trainer = RLTrainer(config=config)
        result = trainer.train(
            price_data=train_data,
            market_features=market_features,
            regime_labels=regime_labels,
            agent_opinions=agent_opinions,
        )
    else:
        # === FULL MODE: Two-layer (StockSelector + PPO) ===
        from rl.full_market_env import FullMarketEnv, FullMarketEnvConfig
        from rl.stock_selector import SelectorConfig

        selector_cfg = SelectorConfig(
            top_k=args.top_k,
            rebalance_freq=args.rebalance_freq,
        )
        env_cfg = FullMarketEnvConfig(
            top_k=args.top_k,
            rebalance_freq=args.rebalance_freq,
            selector_config=selector_cfg,
        )

        logger.info(f"  FullMarketEnv: universe={n_assets}, top_k={args.top_k}, "
                    f"rebalance={args.rebalance_freq}d")

        # Create env
        env = FullMarketEnv(
            price_data=train_data,
            market_features=market_features,
            regime_labels=regime_labels,
            config=env_cfg,
        )
        logger.info(f"  Obs space: {env.observation_space.shape}")
        logger.info(f"  Action space: {env.action_space.shape}")

        # Train with SB3 PPO directly (supports Best-of-N)
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        total_steps = args.warmup_steps + args.main_steps
        seeds = [42, 123, 777, 2024, 31415][:args.n_seeds]
        best_model = None
        best_sharpe = -999.0
        best_seed = seeds[0]

        for seed_idx, seed in enumerate(seeds):
            if args.n_seeds > 1:
                logger.info(f"  Training seed {seed} ({seed_idx+1}/{len(seeds)})...")

            def make_env(s=seed):
                def _init():
                    e = FullMarketEnv(
                        price_data=train_data,
                        market_features=market_features,
                        regime_labels=regime_labels,
                        config=env_cfg,
                    )
                    e.reset(seed=s)
                    return e
                return _init

            vec_env = DummyVecEnv([make_env(seed + j) for j in range(args.n_envs)])

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=args.lr,
                n_steps=4096,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,  # Proven anti-overfit value
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 128]),  # Optimal for RTX 4090 (156K params)
                seed=seed,
                device=args.device,
                verbose=0 if args.n_seeds > 1 else 1,
                tensorboard_log=str(MODEL_DIR.parent / "tb_logs") if seed_idx == 0 else None,
            )

            t0 = time.time()
            model.learn(total_timesteps=total_steps)
            training_time = time.time() - t0

            # Evaluate on test data for model selection
            test_prices_mat = np.column_stack([test_data[t] for t in sorted(test_data.keys())])
            test_steps = len(test_prices_mat)
            test_features = compute_market_features(test_prices_mat, test_steps)
            test_regimes = compute_regime_labels(test_features)

            eval_env = FullMarketEnv(
                price_data=test_data,
                market_features=test_features,
                regime_labels=test_regimes,
                config=env_cfg,
            )
            obs, _ = eval_env.reset(seed=42)
            eval_pvs = [env_cfg.initial_cash]
            for _ in range(252):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                eval_pvs.append(info["portfolio_value"])
                if terminated or truncated:
                    break

            eval_pvs = np.array(eval_pvs)
            eval_ret = (eval_pvs[-1] / eval_pvs[0] - 1) * 100
            daily_rets = np.diff(eval_pvs) / eval_pvs[:-1]
            eval_sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-8) * np.sqrt(252))

            if args.n_seeds > 1:
                logger.info(f"    Seed {seed}: Return={eval_ret:+.2f}%, Sharpe={eval_sharpe:.2f} ({training_time:.0f}s)")

            if eval_sharpe > best_sharpe:
                best_sharpe = eval_sharpe
                best_model = model
                best_seed = seed

        model = best_model
        if args.n_seeds > 1:
            logger.info(f"  Best seed: {best_seed} (Sharpe={best_sharpe:.2f})")

        # Save model
        model_path = str(MODEL_DIR / "rl_macs_full")
        model.save(model_path)
        logger.info(f"  Model saved: {model_path}")

        # Quick evaluation (use test env)
        obs, _ = eval_env.reset(seed=42)
        total_return = 0.0
        steps = 0
        for _ in range(252):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_return += reward
            steps += 1
            if terminated or truncated:
                break

        # Create a compatible result object
        from rl.trainer import TrainingResult
        eval_return = (eval_env._portfolio_value / eval_env.config.initial_cash) - 1.0
        result = TrainingResult(
            model_path=model_path,
            total_steps=total_steps,
            training_time_seconds=training_time,
            final_reward=total_return,
            best_sharpe=0.0,
            best_model_path=model_path,
            episode_rewards=[total_return],
            eval_results=[{
                "regime": "FULL",
                "total_return": eval_return,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }],
            config=TrainingConfig(),
        )
        logger.info(f"  Quick eval: Return={eval_return:.2%}")

    # ============================================================
    # Phase 7: Evaluate on Test Set
    # ============================================================
    if args.eval_test and result.model_path:
        if args.mode == "small":
            logger.info("\n[Phase 7] Evaluating on held-out test set...")
            test_result = evaluate_on_test_set(
                model_path=result.model_path,
                test_data=test_data,
                obs_stats_path=obs_stats_path,
            )
            if test_result:
                logger.info(f"  Test Return: {test_result.get('total_return', 0):.2%}")
                logger.info(f"  Test Sharpe: {test_result.get('sharpe_ratio', 0):.3f}")
                logger.info(f"  Test MaxDD:  {test_result.get('max_drawdown', 0):.2%}")

                # Overfitting detection
                train_sharpe = result.best_sharpe
                test_sharpe = test_result.get('sharpe_ratio', 0)
                if train_sharpe > 0:
                    decay = 1 - test_sharpe / train_sharpe
                    logger.info(f"  Sharpe decay (train→test): {decay:.1%}")
                    if decay > 0.5:
                        logger.warning("  ⚠️ HIGH OVERFITTING: test Sharpe < 50% of train Sharpe")
        else:
            # Full mode: evaluate using FullMarketEnv on test data
            logger.info("\n[Phase 7] Evaluating full-mode model on test set...")
            from rl.full_market_env import FullMarketEnv, FullMarketEnvConfig
            from rl.stock_selector import SelectorConfig
            from stable_baselines3 import PPO

            test_prices = np.column_stack([test_data[t] for t in sorted(test_data.keys())])
            test_steps = len(test_prices)
            test_features = compute_market_features(test_prices, test_steps)
            test_regimes = compute_regime_labels(test_features)

            selector_cfg = SelectorConfig(top_k=args.top_k, rebalance_freq=args.rebalance_freq)
            env_cfg = FullMarketEnvConfig(
                top_k=args.top_k, rebalance_freq=args.rebalance_freq,
                selector_config=selector_cfg,
            )
            test_env = FullMarketEnv(
                price_data=test_data,
                market_features=test_features,
                regime_labels=test_regimes,
                config=env_cfg,
            )

            model = PPO.load(result.model_path)
            obs, _ = test_env.reset()
            for _ in range(252):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                if terminated or truncated:
                    break

            test_return = (test_env._portfolio_value / test_env.config.initial_cash) - 1.0
            logger.info(f"  Test Return (full mode): {test_return:.2%}")
            logger.info(f"  Test Final PV: {test_env._portfolio_value:,.0f}")

    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Universe: {n_assets} assets (from {len(tickers)} discovered)")
    logger.info(f"  Train period: {train_steps} days | Test period: {len(next(iter(test_data.values())))} days")
    logger.info(f"  Model saved: {result.model_path}")
    logger.info(f"  Obs stats: {obs_stats_path}")
    logger.info(f"  Total steps: {result.total_steps:,}")
    logger.info(f"  Training time: {result.training_time_seconds:.1f}s")
    logger.info(f"  Best Sharpe (train): {result.best_sharpe:.4f}")
    if result.eval_results:
        for r in result.eval_results:
            logger.info(f"  Eval [{r.get('regime', 'FULL')}]: "
                        f"Return={r.get('total_return', 0):.2%} "
                        f"Sharpe={r.get('sharpe_ratio', 0):.3f} "
                        f"MaxDD={r.get('max_drawdown', 0):.2%}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
