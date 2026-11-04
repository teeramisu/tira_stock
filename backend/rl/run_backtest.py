"""Comprehensive RL-MACS Backtest Validation.

Features:
- Walk-forward split (80% train / 20% test)
- Benchmark comparison (Buy&Hold, Equal-Weight)
- Per-regime performance breakdown
- Detailed risk metrics (Sortino, Calmar, tail risk)
- Position analysis
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl.backtest")

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "market" / "market_data.duckdb"


# ═══════════════════════════════════════════════════════════
# Data Loading (reuse from run_training)
# ═══════════════════════════════════════════════════════════

def load_market_data(tickers: list[str], market: str = "us") -> dict[str, np.ndarray]:
    """Load from DuckDB read-only."""
    import duckdb

    price_data = {}
    conn = duckdb.connect(str(DB_PATH), read_only=True)
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
            valid = np.isfinite(closes) & (closes > 0)
            closes = closes[valid]
            if len(closes) > 200:
                price_data[sym] = closes
    return price_data


def align_price_data(price_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    min_len = min(len(v) for v in price_data.values())
    return {k: v[-min_len:] for k, v in price_data.items()}


def compute_market_features(prices: np.ndarray, n_steps: int) -> np.ndarray:
    """10-dim market features."""
    n_assets = prices.shape[1]
    features = np.zeros((n_steps, 10), dtype=np.float32)
    returns = np.zeros_like(prices)
    returns[1:] = prices[1:] / prices[:-1] - 1.0
    returns = np.nan_to_num(returns, nan=0.0)

    for i in range(n_steps):
        if i < 20:
            continue
        if i >= 60:
            ma60 = prices[i - 60:i + 1].mean(axis=0)
            valid = prices[i] > 0
            if valid.sum() > 0:
                features[i, 0] = float((prices[i][valid] > ma60[valid]).mean())
        else:
            features[i, 0] = 0.5

        ma20 = prices[max(0, i - 20):i + 1].mean(axis=0)
        valid = prices[i] > 0
        if valid.sum() > 0:
            features[i, 1] = float((prices[i][valid] > ma20[valid]).mean())

        if i >= 20:
            ret_20 = prices[i] / prices[i - 20] - 1.0
            valid_ret = ret_20[np.isfinite(ret_20)]
            features[i, 2] = float(np.median(valid_ret)) if len(valid_ret) > 0 else 0.0

        if i >= 5:
            ret_5 = prices[i] / prices[i - 5] - 1.0
            valid_ret = ret_5[np.isfinite(ret_5)]
            features[i, 3] = float(np.median(valid_ret)) if len(valid_ret) > 0 else 0.0
            features[i, 4] = float(np.percentile(valid_ret, 5)) if len(valid_ret) > 1 else 0.0

        if i >= 20:
            vol = returns[i - 20:i].std(axis=0).mean()
            features[i, 5] = float(vol * np.sqrt(252)) if np.isfinite(vol) else 0.0

        if i >= 30:
            vols = [returns[j - 20:j].std(axis=0).mean() for j in range(i - 10, i) if j >= 20]
            if len(vols) > 2:
                features[i, 6] = float(np.std(vols))

        if i >= 20:
            ret_20_all = prices[i] / prices[i - 20] - 1.0
            valid_ret = ret_20_all[np.isfinite(ret_20_all)]
            features[i, 7] = float(np.std(valid_ret)) if len(valid_ret) > 1 else 0.0

        if i >= 20:
            recent_activity = np.abs(returns[i - 5:i]).mean()
            older_activity = np.abs(returns[i - 20:i - 5]).mean() + 1e-8
            features[i, 8] = float((recent_activity - older_activity) / older_activity)

        if i >= 5:
            features[i, 9] = features[i, 1] - features[i - 5, 1]

    return features


def compute_regime_labels(features: np.ndarray) -> np.ndarray:
    """Regime labels: 0=BULL, 1=RANGE, 2=BEAR, 3=CRASH (stricter thresholds)."""
    n_steps = len(features)
    labels = np.ones(n_steps, dtype=np.int64)
    for i in range(n_steps):
        breadth = features[i, 0]
        trend_20d = features[i, 2]
        crash_5d = features[i, 4]
        if crash_5d < -0.08 and breadth < 0.25:
            labels[i] = 3
        elif breadth < 0.30 and trend_20d < -0.03:
            labels[i] = 2
        elif breadth > 0.55 and trend_20d > 0.005:
            labels[i] = 0
    return labels


# ═══════════════════════════════════════════════════════════
# Backtest Engine (Standalone, no train/test leakage)
# ═══════════════════════════════════════════════════════════

def run_rl_backtest(
    model_path: str,
    prices: np.ndarray,
    market_features: np.ndarray,
    regime_labels: np.ndarray,
    agent_opinions: np.ndarray,
    tickers: list[str],
    start_step: int = 60,
    end_step: int | None = None,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """Run RL policy backtest on given data slice."""
    from stable_baselines3 import PPO
    from rl.risk_overlay import RiskOverlay, RiskOverlayConfig

    model = PPO.load(model_path, device="cpu")
    overlay = RiskOverlay(config=RiskOverlayConfig())

    n_assets = prices.shape[1]
    n_steps = len(prices)
    if end_step is None:
        end_step = n_steps - 1

    daily_returns = np.zeros_like(prices)
    daily_returns[1:] = prices[1:] / prices[:-1] - 1.0

    regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]
    positions = np.zeros(n_assets)
    portfolio_value = 1.0
    equity_curve = [1.0]
    returns_list = []
    positions_history = []
    regime_history = []
    total_costs = 0.0
    turnover_list = []

    for step in range(start_step, end_step):
        regime = regime_names[int(regime_labels[step])]
        regime_history.append(regime)

        # Build observation (match env format)
        obs = _build_obs(step, prices, market_features, regime_labels, agent_opinions, positions)

        # Predict
        action, _ = model.predict(obs, deterministic=True)

        # Risk overlay
        risk_result = overlay.apply(action, positions, regime)
        new_positions = positions + risk_result.adjusted_action[:n_assets]
        new_positions = np.clip(new_positions, 0.0, 1.0)

        # Costs
        turnover = np.sum(np.abs(new_positions - positions))
        cost = turnover * (commission + slippage)
        total_costs += cost
        turnover_list.append(turnover)

        # Return (next day)
        if step + 1 < n_steps:
            step_ret = daily_returns[step + 1]
        else:
            step_ret = np.zeros(n_assets)
        port_return = float(np.sum(new_positions * step_ret)) - cost

        portfolio_value *= (1.0 + port_return)
        positions = new_positions.copy()

        equity_curve.append(portfolio_value)
        returns_list.append(port_return)
        positions_history.append(positions.copy())

    return {
        "equity_curve": np.array(equity_curve),
        "returns": np.array(returns_list),
        "positions": np.array(positions_history),
        "regime_history": regime_history,
        "total_costs": total_costs,
        "turnover": np.array(turnover_list),
        "tickers": tickers,
    }


def run_benchmark(prices: np.ndarray, start_step: int, end_step: int, mode: str = "equal_weight") -> dict:
    """Run benchmark strategy."""
    n_assets = prices.shape[1]
    daily_returns = np.zeros_like(prices)
    daily_returns[1:] = prices[1:] / prices[:-1] - 1.0

    if mode == "equal_weight":
        weights = np.ones(n_assets) / n_assets
    elif mode == "buy_hold_first":
        weights = np.zeros(n_assets)
        weights[0] = 1.0
    else:
        weights = np.ones(n_assets) / n_assets

    portfolio_value = 1.0
    equity_curve = [1.0]
    returns_list = []

    for step in range(start_step, end_step):
        if step + 1 < len(prices):
            port_ret = float(np.sum(weights * daily_returns[step + 1]))
        else:
            port_ret = 0.0
        portfolio_value *= (1.0 + port_ret)
        equity_curve.append(portfolio_value)
        returns_list.append(port_ret)

    return {
        "equity_curve": np.array(equity_curve),
        "returns": np.array(returns_list),
    }


def _build_obs(step, prices, market_features, regime_labels, agent_opinions, positions):
    """Build observation matching env format."""
    n_assets = prices.shape[1]
    mkt_feat = market_features[step].astype(np.float32)
    regime_onehot = np.zeros(4, dtype=np.float32)
    regime_onehot[int(regime_labels[step])] = 1.0

    lookback = min(5, step)
    if lookback > 0:
        asset_returns = (prices[step] / prices[step - lookback] - 1.0).astype(np.float32)
    else:
        asset_returns = np.zeros(n_assets, dtype=np.float32)

    asset_vol = np.zeros(n_assets, dtype=np.float32)
    if step >= 20:
        recent = prices[step - 20:step]
        rets = recent[1:] / recent[:-1] - 1.0
        asset_vol = rets.std(axis=0).astype(np.float32) * np.sqrt(252)

    opinions = agent_opinions[step].astype(np.float32) if step < len(agent_opinions) else np.zeros(26, dtype=np.float32)

    total_invested = float(np.sum(positions))
    cash_ratio = 1.0 - total_invested
    account_state = np.array([cash_ratio, total_invested, 0.0], dtype=np.float32)

    obs = np.concatenate([mkt_feat, regime_onehot, asset_returns, asset_vol, opinions, account_state])
    return obs.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def compute_metrics(equity_curve: np.ndarray, returns: np.ndarray, name: str = "") -> dict:
    """Compute comprehensive performance metrics."""
    n_days = len(returns)
    if n_days == 0:
        return {"name": name, "total_return": 0, "sharpe": 0}

    total_return = equity_curve[-1] / equity_curve[0] - 1.0
    annual_factor = 252.0 / n_days
    if total_return > -1:
        annual_return = (1 + total_return) ** annual_factor - 1
    else:
        annual_return = -1.0

    vol = float(returns.std() * np.sqrt(252))
    sharpe = annual_return / vol if vol > 1e-8 else 0.0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 1e-8
    sortino = annual_return / downside_vol if downside_vol > 1e-8 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / np.maximum(peak, 1e-8)
    max_dd = float(drawdowns.max())

    # Calmar ratio
    calmar = annual_return / max_dd if max_dd > 1e-8 else 0.0

    # Win rate
    win_rate = float((returns > 0).sum() / n_days)

    # Tail risk (5th percentile)
    tail_5 = float(np.percentile(returns, 5))

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else float('inf')

    return {
        "name": name,
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "tail_5pct": tail_5,
        "profit_factor": profit_factor,
        "n_days": n_days,
    }


def regime_breakdown(returns: np.ndarray, regime_history: list[str]) -> dict:
    """Per-regime performance."""
    results = {}
    for regime in ["BULL", "RANGE", "BEAR", "CRASH"]:
        mask = np.array([r == regime for r in regime_history])
        if mask.sum() == 0:
            continue
        regime_rets = returns[mask]
        n = len(regime_rets)
        cum_ret = float(np.prod(1 + regime_rets) - 1)
        vol = float(regime_rets.std() * np.sqrt(252))
        ann_ret = float(regime_rets.mean() * 252)
        sharpe = ann_ret / vol if vol > 1e-8 else 0.0
        results[regime] = {
            "n_days": n,
            "pct": n / len(returns) * 100,
            "cum_return": cum_ret,
            "ann_return": ann_ret,
            "volatility": vol,
            "sharpe": sharpe,
            "win_rate": float((regime_rets > 0).sum() / n),
        }
    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]
    model_path = "rl/models/rl_macs_ppo.zip"

    logger.info("=" * 70)
    logger.info("RL-MACS Comprehensive Backtest Validation")
    logger.info("=" * 70)

    # Load data
    logger.info("\n[1] Loading data...")
    price_data = load_market_data(tickers)
    price_data = align_price_data(price_data)
    tickers_loaded = sorted(price_data.keys())
    n_assets = len(tickers_loaded)
    prices = np.column_stack([price_data[t] for t in tickers_loaded])
    n_steps = len(prices)
    logger.info(f"  {n_assets} assets × {n_steps} steps")

    # Compute features
    logger.info("\n[2] Computing features & regime...")
    market_features = compute_market_features(prices, n_steps)
    regime_labels = compute_regime_labels(market_features)

    # Pre-compute opinions
    logger.info("\n[3] Pre-computing agent opinions...")
    from rl.trainer import precompute_agent_opinions
    agent_opinions = precompute_agent_opinions(price_data, market_features, regime_labels)

    # Walk-forward split: 80% train / 20% test
    split = int(n_steps * 0.8)
    logger.info(f"\n[4] Walk-forward split:")
    logger.info(f"  Train: step 0 ~ {split} ({split} days)")
    logger.info(f"  Test:  step {split} ~ {n_steps} ({n_steps - split} days, OUT-OF-SAMPLE)")

    # ─── Out-of-Sample Backtest ───
    logger.info("\n[5] Running OUT-OF-SAMPLE backtest (test set)...")
    rl_result = run_rl_backtest(
        model_path=model_path,
        prices=prices,
        market_features=market_features,
        regime_labels=regime_labels,
        agent_opinions=agent_opinions,
        tickers=tickers_loaded,
        start_step=split,
        end_step=n_steps - 1,
    )

    # Benchmarks on same test period
    ew_result = run_benchmark(prices, split, n_steps - 1, "equal_weight")
    bh_result = run_benchmark(prices, split, n_steps - 1, "buy_hold_first")

    # ─── In-Sample Backtest (for comparison) ───
    logger.info("\n[6] Running IN-SAMPLE backtest (train set, for reference)...")
    rl_is = run_rl_backtest(
        model_path=model_path,
        prices=prices,
        market_features=market_features,
        regime_labels=regime_labels,
        agent_opinions=agent_opinions,
        tickers=tickers_loaded,
        start_step=60,
        end_step=split,
    )
    ew_is = run_benchmark(prices, 60, split, "equal_weight")

    # ─── Metrics ───
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: OUT-OF-SAMPLE (Test Set)")
    logger.info("=" * 70)

    rl_metrics = compute_metrics(rl_result["equity_curve"], rl_result["returns"], "RL-MACS (OOS)")
    ew_metrics = compute_metrics(ew_result["equity_curve"], ew_result["returns"], "Equal-Weight (OOS)")
    bh_metrics = compute_metrics(bh_result["equity_curve"], bh_result["returns"], f"Buy&Hold {tickers_loaded[0]} (OOS)")

    rl_is_metrics = compute_metrics(rl_is["equity_curve"], rl_is["returns"], "RL-MACS (IS)")
    ew_is_metrics = compute_metrics(ew_is["equity_curve"], ew_is["returns"], "Equal-Weight (IS)")

    # Print comparison table
    print("\n" + "─" * 90)
    print(f"{'Metric':<20} {'RL-MACS (OOS)':>15} {'Equal-Weight (OOS)':>18} {'Buy&Hold (OOS)':>15} {'RL-MACS (IS)':>15}")
    print("─" * 90)

    def fmt_pct(v): return f"{v*100:.2f}%"
    def fmt_f(v): return f"{v:.3f}"

    rows = [
        ("Total Return", fmt_pct(rl_metrics["total_return"]), fmt_pct(ew_metrics["total_return"]),
         fmt_pct(bh_metrics["total_return"]), fmt_pct(rl_is_metrics["total_return"])),
        ("Annual Return", fmt_pct(rl_metrics["annual_return"]), fmt_pct(ew_metrics["annual_return"]),
         fmt_pct(bh_metrics["annual_return"]), fmt_pct(rl_is_metrics["annual_return"])),
        ("Volatility", fmt_pct(rl_metrics["volatility"]), fmt_pct(ew_metrics["volatility"]),
         fmt_pct(bh_metrics["volatility"]), fmt_pct(rl_is_metrics["volatility"])),
        ("Sharpe Ratio", fmt_f(rl_metrics["sharpe"]), fmt_f(ew_metrics["sharpe"]),
         fmt_f(bh_metrics["sharpe"]), fmt_f(rl_is_metrics["sharpe"])),
        ("Sortino Ratio", fmt_f(rl_metrics["sortino"]), fmt_f(ew_metrics["sortino"]),
         fmt_f(bh_metrics["sortino"]), fmt_f(rl_is_metrics["sortino"])),
        ("Max Drawdown", fmt_pct(rl_metrics["max_drawdown"]), fmt_pct(ew_metrics["max_drawdown"]),
         fmt_pct(bh_metrics["max_drawdown"]), fmt_pct(rl_is_metrics["max_drawdown"])),
        ("Calmar Ratio", fmt_f(rl_metrics["calmar"]), fmt_f(ew_metrics["calmar"]),
         fmt_f(bh_metrics["calmar"]), fmt_f(rl_is_metrics["calmar"])),
        ("Win Rate", fmt_pct(rl_metrics["win_rate"]), fmt_pct(ew_metrics["win_rate"]),
         fmt_pct(bh_metrics["win_rate"]), fmt_pct(rl_is_metrics["win_rate"])),
        ("Profit Factor", fmt_f(rl_metrics["profit_factor"]), fmt_f(ew_metrics["profit_factor"]),
         fmt_f(bh_metrics["profit_factor"]), fmt_f(rl_is_metrics["profit_factor"])),
        ("Tail Risk (5%)", fmt_pct(rl_metrics["tail_5pct"]), fmt_pct(ew_metrics["tail_5pct"]),
         fmt_pct(bh_metrics["tail_5pct"]), fmt_pct(rl_is_metrics["tail_5pct"])),
    ]
    for row in rows:
        print(f"{row[0]:<20} {row[1]:>15} {row[2]:>18} {row[3]:>15} {row[4]:>15}")
    print("─" * 90)

    # ─── Regime Breakdown (OOS) ───
    logger.info("\n[7] Per-Regime Performance (OOS):")
    regime_perf = regime_breakdown(rl_result["returns"], rl_result["regime_history"])
    print(f"\n{'Regime':<10} {'Days':>6} {'%':>6} {'CumRet':>10} {'AnnRet':>10} {'Vol':>8} {'Sharpe':>8} {'WinRate':>8}")
    print("─" * 70)
    for regime, m in regime_perf.items():
        print(f"{regime:<10} {m['n_days']:>6} {m['pct']:>5.1f}% {m['cum_return']*100:>9.2f}% "
              f"{m['ann_return']*100:>9.2f}% {m['volatility']*100:>7.2f}% {m['sharpe']:>7.3f} {m['win_rate']*100:>7.1f}%")

    # ─── Position Analysis ───
    logger.info("\n[8] Position Analysis (OOS):")
    pos = rl_result["positions"]
    avg_exposure = float(np.sum(np.abs(pos), axis=1).mean())
    avg_cash = 1.0 - avg_exposure
    avg_turnover = float(rl_result["turnover"].mean())
    total_costs = rl_result["total_costs"]

    print(f"  Average exposure: {avg_exposure*100:.1f}%")
    print(f"  Average cash: {avg_cash*100:.1f}%")
    print(f"  Average daily turnover: {avg_turnover*100:.2f}%")
    print(f"  Total transaction costs: {total_costs*100:.3f}%")
    print(f"  Per-asset avg weight:")
    for i, t in enumerate(tickers_loaded):
        print(f"    {t}: {pos[:, i].mean()*100:.2f}%")

    # ─── Overfitting Check ───
    logger.info("\n[9] Overfitting Check:")
    is_sharpe = rl_is_metrics["sharpe"]
    oos_sharpe = rl_metrics["sharpe"]
    degradation = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 1e-8) * 100
    print(f"  IS Sharpe:  {is_sharpe:.3f}")
    print(f"  OOS Sharpe: {oos_sharpe:.3f}")
    print(f"  Degradation: {degradation:.1f}%")
    if degradation > 50:
        print(f"  ⚠️  WARNING: Significant overfitting detected (>50% degradation)")
    elif degradation > 20:
        print(f"  ⚡ Moderate overfitting (20-50% degradation)")
    else:
        print(f"  ✓ Acceptable generalization")

    # ─── Framework Issues ───
    logger.info("\n[10] Framework Assessment:")
    regime_dist_full = {r: int((regime_labels == i).sum()) for i, r in enumerate(["BULL", "RANGE", "BEAR", "CRASH"])}
    crash_pct = regime_dist_full["CRASH"] / n_steps * 100
    print(f"\n  Regime Distribution (full data):")
    for r, cnt in regime_dist_full.items():
        print(f"    {r}: {cnt} days ({cnt/n_steps*100:.1f}%)")

    issues = []
    if crash_pct > 30:
        issues.append(f"CRASH regime占比过高({crash_pct:.1f}%)，阈值crash_5d<-4%过于敏感")
    if degradation > 30:
        issues.append(f"IS/OOS Sharpe退化{degradation:.0f}%，存在过拟合风险")
    if avg_cash > 0.5:
        issues.append(f"平均现金比例{avg_cash*100:.0f}%过高，策略过于保守")

    print(f"\n  Identified Issues:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")

    print("\n" + "=" * 70)
    print("BACKTEST VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
