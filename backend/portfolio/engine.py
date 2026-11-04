"""Minimal daily portfolio backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from time import perf_counter
from typing import Any, Callable

import json
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.data_portal import DataPortal
from marketdata.repository import MarketDataRepository
from marketdata.universe import UniverseConfig, effective_include_exchanges
from research.regime_detector import RegimeDetector, RegimeConfig, position_scale_for_regime

from .broker import BacktestBroker, Broker, CostModel, OrderValidator
from .models import Fill, PortfolioBacktestConfig, PortfolioBacktestResult, StrategyContext, TargetPosition
from .strategies import MomentumTopKStrategy, PortfolioStrategy

ProgressCallback = Callable[[dict[str, object]], None]


def _emit_progress(
    on_progress: ProgressCallback | None,
    *,
    phase: str,
    pct: float,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if on_progress is None:
        return
    payload: dict[str, object] = {
        "phase": phase,
        "pct": round(min(1.0, max(0.0, pct)), 4),
        "message": message,
    }
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    on_progress(payload)


@dataclass
class PortfolioEngine:
    """Run point-in-time daily portfolio backtests."""

    repository: MarketDataRepository
    strategy: PortfolioStrategy
    config: PortfolioBacktestConfig
    universe_config: UniverseConfig = field(default_factory=UniverseConfig)
    cost_model: CostModel = field(default_factory=CostModel)
    stop_loss_pct: float | None = 0.20  # 止损比例：持仓亏损超过此比例强制清仓，None=禁用
    regime_enable: bool = True  # 启用 Regime 仓位调节：根据市场状态自动缩放策略权重
    max_sector_pct: float = 0.30  # 单个行业最大权重，0=不限制
    sector_map_path: str | None = None  # 行业映射JSON路径，None=自动查找
    _cached_trade_dates: list | None = field(default=None, init=False, repr=False)
    _sector_map: dict[str, str] | None = field(default=None, init=False, repr=False)

    def run(self, on_progress: ProgressCallback | None = None) -> PortfolioBacktestResult:
        """Run the portfolio backtest."""
        self._validate_strategy_contract()
        _emit_progress(on_progress, phase="init", pct=0.02, message="初始化回测引擎…")
        started_at = perf_counter()
        time_trade_dates = perf_counter()
        _emit_progress(on_progress, phase="calendar", pct=0.05, message="加载交易日历…")
        trade_dates = self._trade_dates()
        trade_dates_seconds = perf_counter() - time_trade_dates
        time_load_bars = perf_counter()
        _emit_progress(on_progress, phase="load_bars", pct=0.10, message="加载行情数据…")
        bars_by_day = self._bars_by_day(trade_dates)
        load_bars_seconds = perf_counter() - time_load_bars
        time_load_close = perf_counter()
        _emit_progress(on_progress, phase="load_close", pct=0.15, message="构建收盘价矩阵…")
        close_matrix = self._close_matrix(trade_dates)
        load_close_seconds = perf_counter() - time_load_close
        time_load_score = perf_counter()
        _emit_progress(on_progress, phase="load_score", pct=0.20, message="预计算动量得分…")
        momentum_score_matrix = self._momentum_score_matrix(close_matrix)
        load_score_seconds = perf_counter() - time_load_score

        # ---- Regime 检测：根据全市场宽度/趋势自动分类市场状态 ----
        regime_by_date: dict[date, str] = {}  # trade_date -> regime label
        regime_signals: list = []
        if self.regime_enable and not close_matrix.empty:
            _emit_progress(on_progress, phase="regime", pct=0.22, message="检测市场状态…")
            detector = RegimeDetector(config=RegimeConfig())
            regime_signals = detector.detect_all(close_matrix, market=self.config.market)
            for sig in regime_signals:
                regime_by_date[sig.trade_date] = sig.regime

        time_prepare_universe = perf_counter()
        _emit_progress(on_progress, phase="universe", pct=0.25, message="预计算可交易股票池…")
        universe_by_day, universe_diag_by_day = self._precompute_universe_by_day(trade_dates)
        prepare_universe_seconds = perf_counter() - time_prepare_universe
        broker: Broker = BacktestBroker(
            cash=self.config.initial_cash,
            cost_model=self.cost_model,
            lot_size=self.config.lot_size,
        )
        equity_rows: list[dict[str, object]] = []
        position_rows: list[dict[str, object]] = []
        trade_rows: list[dict[str, object]] = []
        rebalance_rows: list[dict[str, object]] = []
        diagnostic_rows: list[dict[str, object]] = []
        universe_seconds = 0.0
        strategy_seconds = 0.0
        rebalance_seconds = 0.0
        bookkeeping_seconds = 0.0
        fast_strategy_seconds = 0.0
        can_fast_momentum = isinstance(self.strategy, MomentumTopKStrategy)
        total_days = len(trade_dates)
        progress_stride = max(1, total_days // 80)

        last_known_prices: dict[str, float] = {}  # 缓存最近收盘价，防止无bar时归零
        entry_costs: dict[str, tuple[float, int]] = {}  # symbol -> (total_cost, total_shares) 用于止损计算
        stop_loss_count = 0  # 止损触发次数
        for idx, trade_date in enumerate(trade_dates):
            if on_progress and (
                idx == 0 or idx == total_days - 1 or idx % progress_stride == 0
            ):
                day_pct = 0.25 + 0.60 * ((idx + 1) / max(total_days, 1))
                _emit_progress(
                    on_progress,
                    phase="simulate",
                    pct=day_pct,
                    message=f"模拟交易 {idx + 1}/{total_days} 日",
                    current=idx + 1,
                    total=total_days,
                )
            bars_by_symbol = bars_by_day.get(trade_date, {})
            if idx > self.strategy.warmup_days and self._should_rebalance(trade_date, idx):
                signal_date = trade_dates[idx - 1]
                started = perf_counter()
                universe_symbols = universe_by_day.get(signal_date, [])
                universe_diag = universe_diag_by_day.get(signal_date, {"final": len(universe_symbols)})
                universe_seconds += perf_counter() - started
                diagnostic_rows.append({"trade_date": signal_date, **universe_diag})
                started = perf_counter()
                if can_fast_momentum:
                    targets = self._momentum_targets_fast(
                        signal_date=signal_date,
                        universe=tuple(universe_symbols),
                        score_matrix=momentum_score_matrix,
                    )
                    fast_strategy_seconds += perf_counter() - started
                else:
                    portal = DataPortal(
                        self.repository,
                        dataset_version=self.config.dataset_version,
                        as_of=signal_date,
                    )
                    targets = self.strategy.generate_targets(
                        signal_date,
                        portal,
                        StrategyContext(
                            market=self.config.market,
                            universe=tuple(universe_symbols),
                            dataset_version=self.config.dataset_version,
                        ),
                    )
                strategy_seconds += perf_counter() - started

                # ---- 止损机制：持仓亏损超过 stop_loss_pct 强制清仓 ----
                if self.stop_loss_pct is not None and entry_costs:
                    target_symbols = {t.symbol for t in targets}
                    for sym, (total_cost, total_shares) in list(entry_costs.items()):
                        if sym in target_symbols or total_shares <= 0:
                            continue
                        if sym not in broker.positions or broker.positions[sym].shares <= 0:
                            continue
                        avg_price = total_cost / total_shares if total_shares > 0 else 0
                        bar = bars_by_symbol.get(sym)
                        if bar is not None:
                            cur_price = float(bar.get("Open", bar.get("Close", 0)))
                            if cur_price > 0 and avg_price > 0:
                                loss_pct = 1.0 - cur_price / avg_price
                                if loss_pct >= self.stop_loss_pct:
                                    # 触发止损：强制加入 targets (weight=0 = 全部卖出)
                                    targets.append(TargetPosition(
                                        symbol=sym, weight=0.0, score=0.0,
                                        reason=f"stop_loss({loss_pct:.0%}>{self.stop_loss_pct:.0%})",
                                    ))
                                    stop_loss_count += 1

                # ---- Regime 仓位调节：根据市场状态缩放目标权重 ----
                current_regime = regime_by_date.get(trade_date, "UNKNOWN")
                if current_regime != "UNKNOWN" and hasattr(self.strategy, 'id'):
                    scale = position_scale_for_regime(current_regime, self.strategy.id)
                    if scale < 1.0:
                        targets[:] = [
                            TargetPosition(
                                symbol=t.symbol,
                                weight=t.weight * scale,
                                score=t.score,
                                reason=f"{t.reason}|regime={current_regime}(x{scale:.0%})",
                            )
                            for t in targets
                        ]

                # ---- 行业分散约束：限制单行业最大权重 ----
                if self.max_sector_pct > 0 and len(targets) > 1:
                    sector_map = self._load_sector_map()
                    if sector_map:
                        sector_weights: dict[str, float] = {}
                        for t in targets:
                            sector = sector_map.get(t.symbol, "_unknown_")
                            sector_weights[sector] = sector_weights.get(sector, 0.0) + t.weight
                        # 若任一行业超限，按比例压缩并归一化
                        max_sector_w = max(sector_weights.values()) if sector_weights else 0.0
                        if max_sector_w > self.max_sector_pct:
                            targets[:] = [
                                TargetPosition(
                                    symbol=t.symbol,
                                    weight=t.weight * (self.max_sector_pct / max_sector_w),
                                    score=t.score,
                                    reason=f"{t.reason}|sector_cap(x{self.max_sector_pct/max_sector_w:.2f})",
                                )
                                for t in targets
                            ]

                OrderValidator.validate_targets(targets)
                started = perf_counter()
                fills = broker.rebalance(
                    trade_date=trade_date,
                    targets=targets,
                    bars_by_symbol=bars_by_symbol,
                )
                rebalance_seconds += perf_counter() - started
                trade_rows.extend(self._fill_rows(fills))

                # 更新持仓成本跟踪（用于止损计算）
                for fill in fills:
                    if fill.status != "filled" or fill.filled_shares <= 0:
                        continue
                    sym = fill.symbol
                    if fill.side == "buy":
                        prev_cost, prev_shares = entry_costs.get(sym, (0.0, 0))
                        entry_costs[sym] = (
                            prev_cost + fill.filled_shares * fill.fill_price,
                            prev_shares + fill.filled_shares,
                        )
                    elif fill.side == "sell":
                        if sym in entry_costs:
                            prev_cost, prev_shares = entry_costs[sym]
                            if prev_shares > 0:
                                avg = prev_cost / prev_shares
                                new_shares = prev_shares - fill.filled_shares
                                if new_shares <= 0:
                                    entry_costs.pop(sym, None)
                                else:
                                    entry_costs[sym] = (avg * new_shares, new_shares)

                rebalance_rows.append(
                    {
                        "signal_date": signal_date,
                        "trade_date": trade_date,
                        "dataset_version": self.config.dataset_version,
                        "universe_size": int(universe_diag.get("final", len(universe_symbols))),
                        "target_count": len(targets),
                        "filled_orders": sum(1 for fill in fills if fill.status == "filled"),
                        "rejected_orders": sum(1 for fill in fills if fill.status == "rejected"),
                        "regime": current_regime,
                        "targets": [
                            {
                                "symbol": t.symbol,
                                "weight": t.weight,
                                "score": t.score,
                                "reason": t.reason,
                            }
                            for t in targets
                        ],
                        "rejected_details": [
                            {
                                "symbol": fill.symbol,
                                "side": fill.side,
                                "reason": fill.reason,
                            }
                            for fill in fills
                            if fill.status == "rejected"
                        ],
                        "cost_snapshot": {
                            "buy_commission_rate": self.cost_model.buy_commission_rate,
                            "sell_commission_rate": self.cost_model.sell_commission_rate,
                            "stamp_tax_rate": self.cost_model.stamp_tax_rate,
                            "transfer_fee_rate": self.cost_model.transfer_fee_rate,
                            "min_commission": self.cost_model.min_commission,
                            "slippage_bps": self.cost_model.slippage_bps,
                            "market_impact_bps": self.cost_model.market_impact_bps,
                        },
                    }
                )

            started = perf_counter()
            # 更新最近已知价格缓存
            for sym, bar in bars_by_symbol.items():
                raw_px = float(bar.get("Close", bar.get("Open", 0.0)))
                if np.isfinite(raw_px) and raw_px > 0:
                    last_known_prices[sym] = raw_px
            equity = broker.market_value(bars_by_symbol)
            equity_rows.append(
                {
                    "trade_date": trade_date,
                    "equity": equity,
                    "cash": broker.cash,
                    "position_value": equity - broker.cash,
                    "regime": regime_by_date.get(trade_date, "UNKNOWN"),
                }
            )
            for symbol, position in broker.positions.items():
                bar = bars_by_symbol.get(symbol)
                if bar is None:
                    # 无当日bar时使用最近已知收盘价，避免粗暴归零
                    price = last_known_prices.get(symbol, 0.0)
                else:
                    raw_price = float(bar.get("Close", bar.get("Open", 0.0)))
                    price = raw_price if np.isfinite(raw_price) and raw_price > 0 else last_known_prices.get(symbol, 0.0)
                    if price > 0:
                        last_known_prices[symbol] = price
                market_value = position.shares * price
                position_rows.append(
                    {
                        "trade_date": trade_date,
                        "symbol": symbol,
                        "shares": position.shares,
                        "market_value": market_value,
                        "weight": float(market_value / equity) if equity > 0 else 0.0,
                    }
                )
            bookkeeping_seconds += perf_counter() - started

        _emit_progress(on_progress, phase="finalize", pct=0.88, message="汇总净值与成交记录…")
        equity_curve = pd.DataFrame(equity_rows)
        benchmark_curve = self._benchmark_curve(close_matrix, trade_dates)
        cost_summary = self._cost_summary(trades=pd.DataFrame(trade_rows))
        # Add regime summary to cost_summary for API output
        if regime_signals:
            from research.regime_detector import RegimeDetector as _RD
            cost_summary["regime_summary"] = _RD().regime_summary(regime_signals)  # type: ignore[assignment]
        trades = pd.DataFrame(trade_rows)
        elapsed_seconds = perf_counter() - started_at
        return PortfolioBacktestResult(
            equity_curve=equity_curve,
            positions=pd.DataFrame(position_rows),
            trades=trades,
            rebalance_logs=pd.DataFrame(rebalance_rows),
            filter_diagnostics=pd.DataFrame(diagnostic_rows),
            benchmark_curve=benchmark_curve,
            cost_summary=cost_summary,
            metrics=self._metrics(
                equity_curve,
                benchmark_curve,
                pd.DataFrame(position_rows),
                trades,
                elapsed_seconds,
                timing_breakdown={
                    "trade_dates_seconds": trade_dates_seconds,
                    "load_bars_seconds": load_bars_seconds,
                    "load_close_seconds": load_close_seconds,
                    "load_score_seconds": load_score_seconds,
                    "prepare_universe_seconds": prepare_universe_seconds,
                    "universe_seconds": universe_seconds,
                    "strategy_seconds": strategy_seconds,
                    "fast_strategy_seconds": fast_strategy_seconds,
                    "rebalance_seconds": rebalance_seconds,
                    "bookkeeping_seconds": bookkeeping_seconds,
                },
            ),
        )

    def _validate_strategy_contract(self) -> None:
        """Validate required strategy attributes before backtest starts."""
        required_attrs = ("data_frequency", "rebalance_frequency", "warmup_days")
        missing = [attr for attr in required_attrs if not hasattr(self.strategy, attr)]
        if missing:
            raise ValueError(f"strategy missing required attributes: {', '.join(missing)}")

    def _trade_dates(self) -> list[date]:
        if self._cached_trade_dates is not None:
            return self._cached_trade_dates
        rows = self.repository.conn.execute(
            """
            SELECT DISTINCT trade_date
            FROM daily_bars
            WHERE market = ?
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            [self.config.market, self.config.start, self.config.end],
        ).fetchall()
        result = [row[0] for row in rows]
        self._cached_trade_dates = result
        return result

    def _bars_by_symbol(self, trade_date: date) -> dict[str, dict[str, float | bool | None]]:
        df = self.repository.conn.execute(
            """
            SELECT
                symbol,
                COALESCE(adj_open, open) AS Open,
                COALESCE(adj_close, close) AS Close,
                paused, limit_up, limit_down
            FROM daily_bars
            WHERE market = ? AND trade_date = ?
            """,
            [self.config.market, trade_date],
        ).fetchdf()
        return {
            str(row["symbol"]): {
                "Open": row["Open"],
                "Close": row["Close"],
                "paused": row["paused"],
                "limit_up": row["limit_up"],
                "limit_down": row["limit_down"],
            }
            for _, row in df.iterrows()
        }

    def _bars_by_day(self, trade_dates: list[date]) -> dict[date, dict[str, dict[str, float | bool | None]]]:
        """Load all needed daily bars once to avoid per-day SQL queries."""
        if not trade_dates:
            return {}
        rows = self.repository.conn.execute(
            """
            SELECT
                trade_date, symbol,
                COALESCE(adj_open, open) AS Open,
                COALESCE(adj_close, close) AS Close,
                paused, limit_up, limit_down
            FROM daily_bars
            WHERE market = ?
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date, symbol
            """,
            [self.config.market, trade_dates[0], trade_dates[-1]],
        ).fetchall()
        out: dict[date, dict[str, dict[str, float | bool | None]]] = {}
        for trade_date, symbol, open_px, close_px, paused, limit_up, limit_down in rows:
            key = trade_date.date() if hasattr(trade_date, "date") else trade_date
            if key not in out:
                out[key] = {}
            out[key][str(symbol)] = {
                "Open": open_px,
                "Close": close_px,
                "paused": paused,
                "limit_up": limit_up,
                "limit_down": limit_down,
            }
        return out

    def _close_matrix(self, trade_dates: list[date]) -> pd.DataFrame:
        """Load close-price matrix indexed by trade_date for fast ranking."""
        if not trade_dates:
            return pd.DataFrame()
        df = self.repository.conn.execute(
            """
            SELECT trade_date, symbol, COALESCE(adj_close, close) AS close
            FROM daily_bars
            WHERE market = ?
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date, symbol
            """,
            [self.config.market, trade_dates[0], trade_dates[-1]],
        ).fetchdf()
        if df.empty:
            return pd.DataFrame()
        pivot = df.pivot(index="trade_date", columns="symbol", values="close")
        if hasattr(pivot.index, "date"):
            pivot.index = pd.Index([d.date() if hasattr(d, "date") else d for d in pivot.index])
        return pivot.sort_index()

    def _momentum_targets_fast(
        self,
        *,
        signal_date: date,
        universe: tuple[str, ...],
        score_matrix: pd.DataFrame,
    ) -> list[TargetPosition]:
        """Fast path for MomentumTopK on preloaded close matrix."""
        if not universe or score_matrix.empty or not isinstance(self.strategy, MomentumTopKStrategy):
            return []
        top_k = self.strategy.top_k
        if signal_date not in score_matrix.index:
            return []
        available = [symbol for symbol in universe if symbol in score_matrix.columns]
        if not available:
            return []
        row = score_matrix.loc[signal_date, available]
        if row.empty:
            return []
        values = row.to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            return []
        symbols = np.asarray(available, dtype=object)[valid]
        scores = values[valid]
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        if k < len(scores):
            # Partial sort is faster than full argsort for large universes.
            candidate_idx = np.argpartition(-scores, k - 1)[:k]
            keep = candidate_idx[np.argsort(-scores[candidate_idx])]
        else:
            keep = np.argsort(-scores)
        selected_symbols = symbols[keep]
        selected_scores = scores[keep]
        # Score-weighted sizing
        score_min = selected_scores.min()
        shifted = selected_scores - score_min + abs(score_min) * 0.01 + 0.001
        weights = shifted / shifted.sum()
        return [
            TargetPosition(
                symbol=str(symbol),
                weight=float(w),
                score=float(score),
                reason=f"{self.strategy.lookback_days}d_return_rank",
            )
            for symbol, score, w in zip(selected_symbols, selected_scores, weights, strict=False)
        ]

    def _momentum_score_matrix(self, close_matrix: pd.DataFrame) -> pd.DataFrame:
        """Precompute momentum scores for fast daily ranking."""
        if close_matrix.empty or not isinstance(self.strategy, MomentumTopKStrategy):
            return pd.DataFrame()
        lookback = self.strategy.lookback_days
        shifted = close_matrix.shift(lookback)
        valid = (close_matrix > 0) & (shifted > 0)
        scores = close_matrix / shifted - 1.0
        return scores.where(valid)

    def _precompute_universe_by_day(
        self, trade_dates: list[date]
    ) -> tuple[dict[date, list[str]], dict[date, dict[str, int]]]:
        """Precompute per-day eligible universe with step-by-step diagnostic counts."""
        if not trade_dates:
            return {}, {}
        static_diag = self._universe_static_diagnostics()
        exchanges = effective_include_exchanges(
            self.universe_config.market,
            self.universe_config.include_exchanges,
        )
        # Step 1: 基础查询（仅市场+日期+活跃）
        base_conditions = [
            "b.market = ?",
            "b.trade_date BETWEEN ? AND ?",
            "COALESCE(i.is_active, FALSE) = TRUE",
        ]
        params: list[Any] = [self.universe_config.market, trade_dates[0], trade_dates[-1]]
        if self.universe_config.symbols:
            symbols = sorted({symbol.upper() for symbol in self.universe_config.symbols})
            placeholders = ", ".join(["?"] * len(symbols))
            base_conditions.append(f"UPPER(i.symbol) IN ({placeholders})")
            params.extend(symbols)
        if exchanges:
            exchange_list = list(exchanges)
            placeholders = ", ".join(["?"] * len(exchange_list))
            base_conditions.append(f"i.exchange IN ({placeholders})")
            params.extend(exchange_list)
        if self.universe_config.exclude_neeq:
            base_conditions.append("COALESCE(i.exchange, '') <> 'NEEQ'")
        where_clause = " AND ".join(base_conditions)
        # 查询所有字段用于分步过滤
        rows = self.repository.conn.execute(
            f"""
            SELECT b.trade_date, b.symbol, i.list_date,
                   COALESCE(i.is_st, FALSE), COALESCE(b.paused, FALSE),
                   b.limit_up, b.limit_down, b.close
            FROM daily_bars AS b
            INNER JOIN instruments AS i
                ON i.market = b.market AND i.symbol = b.symbol
            WHERE {where_clause}
            ORDER BY b.trade_date, b.symbol
            """,
            params,
        ).fetchall()
        # 按日分组
        from datetime import date as _date
        raw_by_day: dict[date, list] = {}
        for row in rows:
            td = row[0].date() if hasattr(row[0], "date") else row[0]
            raw_by_day.setdefault(td, []).append(row)
        # 分步过滤 + 统计
        universe_by_day: dict[date, list[str]] = {}
        diagnostics_by_day: dict[date, dict[str, int]] = {}
        for trade_date in trade_dates:
            day_rows = raw_by_day.get(trade_date, [])
            diag = dict(static_diag)
            diag["missing_trade_bar"] = 0
            # Step 2: 上市天数过滤
            if self.universe_config.min_list_days > 0:
                before = len(day_rows)
                day_rows = [
                    r for r in day_rows
                    if r[2] is None or (trade_date - r[2]).days >= self.universe_config.min_list_days
                ]
                diag["after_min_list_days"] = len(day_rows)
            else:
                diag["after_min_list_days"] = len(day_rows)
            # Step 3: 排除 ST
            if self.universe_config.exclude_st:
                day_rows = [r for r in day_rows if not r[3]]
            diag["after_st"] = len(day_rows)
            # Step 4: 排除停牌
            if self.universe_config.exclude_suspended:
                day_rows = [r for r in day_rows if not r[4]]
            diag["after_suspended"] = len(day_rows)
            # Step 5: 排除涨停
            if self.universe_config.exclude_limit_up_buy:
                day_rows = [
                    r for r in day_rows
                    if not (r[5] is not None and r[7] is not None and r[7] >= r[5])
                ]
            diag["after_limit_up_buy"] = len(day_rows)
            # Step 6: 排除跌停
            if self.universe_config.exclude_limit_down_sell:
                day_rows = [
                    r for r in day_rows
                    if not (r[6] is not None and r[7] is not None and r[7] <= r[6])
                ]
            diag["after_limit_down_sell"] = len(day_rows)
            diag["final"] = len(day_rows)
            universe_by_day[trade_date] = [r[1] for r in day_rows]
            diagnostics_by_day[trade_date] = diag
        return universe_by_day, diagnostics_by_day

    def _universe_static_diagnostics(self) -> dict[str, int]:
        """Compute static diagnostics once from instrument master table."""
        rows = self.repository.conn.execute(
            """
            SELECT symbol, exchange, list_date, is_active, is_st
            FROM instruments
            WHERE market = ?
            """,
            [self.universe_config.market],
        ).fetchdf()
        diagnostics: dict[str, int] = {
            "initial": int(len(rows)),
            "missing_list_date": int(rows["list_date"].isna().sum()) if not rows.empty else 0,
        }
        if rows.empty:
            diagnostics.update(
                {
                    "after_custom_symbols": 0,
                    "after_exchange": 0,
                    "after_neeq": 0,
                    "after_active": 0,
                    "after_st": 0,
                }
            )
            return diagnostics
        current = rows.copy()
        if self.universe_config.symbols:
            allowed_symbols = {symbol.upper() for symbol in self.universe_config.symbols}
            current = current[current["symbol"].str.upper().isin(allowed_symbols)]
        diagnostics["after_custom_symbols"] = int(len(current))
        exchanges = effective_include_exchanges(
            self.universe_config.market,
            self.universe_config.include_exchanges,
        )
        if exchanges:
            current = current[current["exchange"].isin(exchanges)]
        diagnostics["after_exchange"] = int(len(current))
        if self.universe_config.exclude_neeq:
            current = current[current["exchange"] != "NEEQ"]
        diagnostics["after_neeq"] = int(len(current))
        current = current[current["is_active"] == True]  # noqa: E712
        diagnostics["after_active"] = int(len(current))
        if self.universe_config.exclude_st:
            current = current[current["is_st"] == False]  # noqa: E712
        diagnostics["after_st"] = int(len(current))
        return diagnostics

    def _load_sector_map(self) -> dict[str, str]:
        """Load symbol→sector mapping from board_constituents.json.

        Returns dict like {"000938.SZSE": "IT服务", ...}.
        Caches result in self._sector_map.
        """
        if self._sector_map is not None:
            return self._sector_map
        path = self.sector_map_path
        if path is None:
            path = str(Path(__file__).resolve().parents[2] / "data" / "board_constituents.json")
        try:
            with open(path) as f:
                data = json.load(f)
            raw = data.get("industry", {})
            result: dict[str, str] = {}
            for sector_name, sector_data in raw.items():
                stocks = sector_data.get("stocks", []) if isinstance(sector_data, dict) else []
                for item in stocks:
                    code = item["code"] if isinstance(item, dict) else str(item)
                    # 判断交易所后缀
                    if code.startswith("6") or code.startswith("9"):
                        full = f"{code}.SSE"
                    elif code.startswith("0") or code.startswith("3"):
                        full = f"{code}.SZSE"
                    elif code.startswith("4") or code.startswith("8"):
                        full = f"{code}.BSE"
                    else:
                        continue
                    if full not in result:
                        result[full] = sector_name
            self._sector_map = result
            return result
        except Exception:
            self._sector_map = {}
            return {}

    def _should_rebalance(self, trade_date: date, idx: int) -> bool:
        if self.config.rebalance_frequency == "daily":
            return True
        if self.config.rebalance_frequency == "weekly":
            return trade_date.weekday() == 0
        if self.config.rebalance_frequency == "monthly":
            return idx == 0 or trade_date.month != self._trade_dates()[idx - 1].month
        raise ValueError(f"Unsupported rebalance_frequency: {self.config.rebalance_frequency}")

    @staticmethod
    def _fill_rows(fills: list[Fill]) -> list[dict[str, object]]:
        return [
            {
                "trade_date": fill.trade_date,
                "symbol": fill.symbol,
                "side": fill.side,
                "requested_shares": fill.requested_shares,
                "filled_shares": fill.filled_shares,
                "raw_price": fill.raw_price,
                "fill_price": fill.fill_price,
                "status": fill.status,
                "reason": fill.reason,
                "commission": fill.costs.commission,
                "stamp_tax": fill.costs.stamp_tax,
                "transfer_fee": fill.costs.transfer_fee,
                "slippage_cost": fill.costs.slippage_cost,
                "total_cost": fill.costs.total_cost,
            }
            for fill in fills
        ]

    def _metrics(
        self,
        equity_curve: pd.DataFrame,
        benchmark_curve: pd.DataFrame,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
        elapsed_seconds: float,
        timing_breakdown: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0.0,
                "annualized_volatility": 0.0,
                "total_cost": 0.0,
                "turnover": 0.0,
                "cost_to_turnover_ratio": 0.0,
                "avg_holding_count": 0.0,
                "win_rate": 0.0,
                "benchmark_excess_return": 0.0,
                "elapsed_seconds": elapsed_seconds,
                **(timing_breakdown or {}),
            }
        total_return = float(equity_curve["equity"].iloc[-1] / self.config.initial_cash - 1.0)
        n_days = int(len(equity_curve))
        returns = equity_curve["equity"].pct_change().dropna()
        volatility = 0.0
        if len(returns) >= 2:
            vol = float(returns.std() * (252**0.5))
            volatility = vol if np.isfinite(vol) else 0.0
        annual_return = (
            float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
            if n_days > 0 and (1.0 + total_return) > 0
            else 0.0
        )
        sharpe_ratio = float(annual_return / volatility) if volatility > 0 else 0.0
        running_peak = equity_curve["equity"].cummax()
        drawdown = 1.0 - equity_curve["equity"] / running_peak
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
        drawdown_underwater = drawdown > 0
        drawdown_streak = drawdown_underwater.groupby((~drawdown_underwater).cumsum()).cumsum()
        max_drawdown_duration = int(drawdown_streak.max()) if not drawdown_streak.empty else 0
        total_cost = float(trades["total_cost"].sum()) if not trades.empty else 0.0
        turnover = float((trades["filled_shares"] * trades["fill_price"]).sum()) if not trades.empty else 0.0
        cost_to_turnover_ratio = float(total_cost / turnover) if turnover > 0 else 0.0
        avg_holding_count = 0.0
        if not positions.empty:
            positions_nonzero = positions[positions["shares"] > 0]
            if not positions_nonzero.empty:
                daily_counts = positions_nonzero.groupby("trade_date")["symbol"].nunique()
                avg_holding_count = float(daily_counts.mean()) if not daily_counts.empty else 0.0
        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
        benchmark_return = 0.0
        if not benchmark_curve.empty:
            benchmark_return = float(benchmark_curve["equity"].iloc[-1] / self.config.initial_cash - 1.0)
        benchmark_excess_return = float(total_return - benchmark_return)
        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": float(max_drawdown_duration),
            "annualized_volatility": volatility,
            "total_cost": total_cost,
            "turnover": turnover,
            "cost_to_turnover_ratio": cost_to_turnover_ratio,
            "avg_holding_count": avg_holding_count,
            "win_rate": win_rate,
            "benchmark_excess_return": benchmark_excess_return,
            "elapsed_seconds": elapsed_seconds,
        }
        if timing_breakdown:
            metrics.update(timing_breakdown)
        return metrics

    def _benchmark_curve(self, close_matrix: pd.DataFrame, trade_dates: list[date]) -> pd.DataFrame:
        """Build an equal-weight close-to-close benchmark curve."""
        if close_matrix.empty or not trade_dates:
            return pd.DataFrame(
                [{"trade_date": d, "equity": float(self.config.initial_cash)} for d in trade_dates]
            )
        aligned = close_matrix.reindex(trade_dates)
        daily_returns = aligned.pct_change().mean(axis=1, skipna=True).fillna(0.0)
        benchmark_equity = float(self.config.initial_cash) * (1.0 + daily_returns).cumprod()
        return pd.DataFrame({"trade_date": benchmark_equity.index, "equity": benchmark_equity.values})

    @staticmethod
    def _cost_summary(trades: pd.DataFrame) -> dict[str, float]:
        """Aggregate cost breakdown for response and artifacts."""
        if trades.empty:
            return {
                "commission": 0.0,
                "stamp_tax": 0.0,
                "transfer_fee": 0.0,
                "slippage_cost": 0.0,
                "total_cost": 0.0,
                "turnover": 0.0,
                "cost_to_turnover_ratio": 0.0,
            }
        commission = float(trades["commission"].sum())
        stamp_tax = float(trades["stamp_tax"].sum())
        transfer_fee = float(trades["transfer_fee"].sum())
        slippage_cost = float(trades["slippage_cost"].sum())
        total_cost = float(trades["total_cost"].sum())
        turnover = float((trades["filled_shares"] * trades["fill_price"]).sum())
        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "slippage_cost": slippage_cost,
            "total_cost": total_cost,
            "turnover": turnover,
            "cost_to_turnover_ratio": float(total_cost / turnover) if turnover > 0 else 0.0,
        }

