"""Per-symbol profit/drawdown attribution and trade-cycle artifacts."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from marketdata.repository import MarketDataRepository


def _to_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def build_trade_cycles(trades: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Group filled trades into round-trip cycles per symbol."""
    if trades.empty:
        return {}
    frame = trades.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"] == "filled"]
    if frame.empty:
        return {}
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame = frame.sort_values(["symbol", "trade_date", "side"], kind="stable")

    cycles_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        symbol_key = str(symbol)
        cycles: list[dict[str, Any]] = []
        position = 0
        cycle_id = 0
        current: dict[str, Any] | None = None
        fifo_lots: list[dict[str, float]] = []

        for row in group.to_dict(orient="records"):
            side = str(row["side"])
            shares = int(row.get("filled_shares") or 0)
            if shares <= 0:
                continue
            trade_date = _to_date(row["trade_date"])
            fill_price = float(row.get("fill_price") or 0.0)
            total_cost = float(row.get("total_cost") or 0.0)
            event = {
                "date": _serialize_date(trade_date),
                "side": side,
                "shares": shares,
                "price": fill_price,
                "cost": total_cost,
            }

            if side == "buy":
                if position == 0:
                    cycle_id += 1
                    current = {
                        "cycle_id": cycle_id,
                        "symbol": symbol_key,
                        "status": "open",
                        "entry_date": _serialize_date(trade_date),
                        "exit_date": None,
                        "holding_days": 0,
                        "events": [],
                        "realized_pnl": 0.0,
                        "return_pct": 0.0,
                    }
                    cycles.append(current)
                    fifo_lots = []
                cost_per_share = (shares * fill_price + total_cost) / shares
                fifo_lots.append({"shares": shares, "cost_per_share": cost_per_share})
                position += shares
                if current is not None:
                    current["events"].append(event)
            elif side == "sell" and current is not None:
                remaining = shares
                sell_pnl = -total_cost
                while remaining > 0 and fifo_lots:
                    lot = fifo_lots[0]
                    take = min(remaining, int(lot["shares"]))
                    sell_pnl += take * (fill_price - lot["cost_per_share"])
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] <= 0:
                        fifo_lots.pop(0)
                position = max(position - shares, 0)
                current["realized_pnl"] = float(current.get("realized_pnl", 0.0) + sell_pnl)
                current["events"].append(event)
                if position == 0:
                    current["status"] = "closed"
                    current["exit_date"] = _serialize_date(trade_date)
                    invested = sum(
                        event["shares"] * event["price"] + event["cost"]
                        for event in current["events"]
                        if event["side"] == "buy"
                    )
                    if invested > 0:
                        current["return_pct"] = float(current["realized_pnl"] / invested)
                    current = None
                    fifo_lots = []

        cycles_by_symbol[symbol_key] = cycles
    return cycles_by_symbol


def _holding_days_by_symbol(positions: pd.DataFrame) -> dict[str, int]:
    if positions.empty:
        return {}
    frame = positions.copy()
    frame = frame[frame["shares"] > 0]
    if frame.empty:
        return {}
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    return {
        str(symbol): int(group["trade_date"].nunique())
        for symbol, group in frame.groupby("symbol", sort=True)
    }


def _symbol_pnl_summary(
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    cycles_by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Aggregate realized/unrealized PnL per symbol."""
    summaries: dict[str, dict[str, float]] = {}
    symbols = set(cycles_by_symbol.keys())
    if not positions.empty:
        symbols.update(positions["symbol"].astype(str).tolist())
    if not trades.empty and "symbol" in trades.columns:
        symbols.update(trades["symbol"].astype(str).tolist())

    last_positions: dict[str, float] = {}
    if not positions.empty:
        frame = positions.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        last_day = frame["trade_date"].max()
        latest = frame[frame["trade_date"] == last_day]
        for row in latest.to_dict(orient="records"):
            if int(row.get("shares") or 0) > 0:
                last_positions[str(row["symbol"])] = float(row.get("market_value") or 0.0)

    for symbol in symbols:
        cycles = cycles_by_symbol.get(symbol, [])
        realized = float(sum(c.get("realized_pnl", 0.0) for c in cycles if c.get("status") == "closed"))
        open_cycles = [c for c in cycles if c.get("status") == "open"]
        unrealized = 0.0
        if open_cycles:
            cycle = open_cycles[-1]
            buy_notional = sum(
                event["shares"] * event["price"] + event["cost"]
                for event in cycle.get("events", [])
                if event["side"] == "buy"
            )
            sell_notional = sum(
                event["shares"] * event["price"] - event["cost"]
                for event in cycle.get("events", [])
                if event["side"] == "sell"
            )
            end_mv = last_positions.get(symbol, 0.0)
            unrealized = end_mv + sell_notional - buy_notional
            cycle["unrealized_pnl"] = unrealized
            if buy_notional > 0:
                cycle["return_pct"] = float((cycle.get("realized_pnl", 0.0) + unrealized) / buy_notional)

        total_pnl = realized + unrealized
        invested = 0.0
        if not trades.empty:
            sym_trades = trades[
                (trades["symbol"].astype(str) == symbol)
                & (trades["status"] == "filled" if "status" in trades.columns else True)
            ]
            if not sym_trades.empty:
                buys = sym_trades[sym_trades["side"] == "buy"]
                invested = float((buys["filled_shares"] * buys["fill_price"] + buys["total_cost"]).sum())
        total_return_pct = float(total_pnl / invested) if invested > 0 else 0.0
        summaries[symbol] = {
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
        }
    return summaries


def _portfolio_drawdown_window(equity_curve: pd.DataFrame) -> dict[str, Any]:
    if equity_curve.empty:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "drawdown_window_start": None,
            "drawdown_window_end": None,
        }
    frame = equity_curve.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    equity = frame["equity"].astype(float)
    peak = equity.cummax()
    drawdown = 1.0 - equity / peak.replace(0, np.nan)
    trough_idx = int(drawdown.idxmax())
    trough_date = frame.loc[trough_idx, "trade_date"].date()
    peak_idx = int(equity.iloc[: trough_idx + 1].idxmax())
    peak_date = frame.loc[peak_idx, "trade_date"].date()
    return {
        "max_drawdown": float(drawdown.max()) if not drawdown.empty else 0.0,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "drawdown_window_start": peak_date,
        "drawdown_window_end": trough_date,
    }


def _drawdown_contributions(
    positions: pd.DataFrame,
    drawdown_window: dict[str, Any],
) -> dict[str, dict[str, float]]:
    peak_date = drawdown_window.get("peak_date")
    trough_date = drawdown_window.get("trough_date")
    if peak_date is None or trough_date is None or positions.empty:
        return {}
    frame = positions.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    pivot = frame.pivot_table(index="trade_date", columns="symbol", values="market_value", aggfunc="last").fillna(0.0)
    if pivot.empty:
        return {}

    def _value_on(target: date) -> pd.Series:
        if target in pivot.index:
            return pivot.loc[target]
        eligible = pivot.index[pivot.index <= target]
        if len(eligible) == 0:
            return pd.Series(0.0, index=pivot.columns)
        return pivot.loc[eligible[-1]]

    peak_mv = _value_on(peak_date)
    trough_mv = _value_on(trough_date)
    out: dict[str, dict[str, float]] = {}
    for symbol in pivot.columns:
        peak_value = float(peak_mv.get(symbol, 0.0))
        trough_value = float(trough_mv.get(symbol, 0.0))
        if peak_value <= 0 and trough_value <= 0:
            continue
        contribution = trough_value - peak_value
        out[str(symbol)] = {
            "drawdown_contribution": contribution,
            "peak_market_value": peak_value,
            "trough_market_value": trough_value,
        }
    return out


def load_instrument_names(repository: MarketDataRepository, market: str, symbols: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return symbol->name map and symbols missing a usable name."""
    if not symbols:
        return {}, []
    placeholders = ", ".join(["?"] * len(symbols))
    rows = repository.conn.execute(
        f"""
        SELECT symbol, name
        FROM instruments
        WHERE market = ? AND symbol IN ({placeholders})
        """,
        [market, *symbols],
    ).fetchall()
    names = {str(symbol).upper(): str(name) for symbol, name in rows}
    misses = []
    for symbol in symbols:
        key = symbol.upper()
        name = names.get(key)
        if not name or name.upper() == key:
            misses.append(key)
            names[key] = key
    return names, misses


def _build_ranked_rows(
    symbols: list[str],
    *,
    cycles_by_symbol: dict[str, list[dict[str, Any]]],
    pnl_summary: dict[str, dict[str, float]],
    holding_days: dict[str, int],
    names: dict[str, str],
    trades: pd.DataFrame,
    drawdown_info: dict[str, dict[str, float]] | None = None,
    drawdown_window: dict[str, Any] | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        summary = pnl_summary.get(symbol, {})
        cycles = cycles_by_symbol.get(symbol, [])
        for cycle in cycles:
            cycle["name"] = names.get(symbol, symbol)
        trade_count = 0
        if not trades.empty:
            trade_count = int(
                len(
                    trades[
                        (trades["symbol"].astype(str) == symbol)
                        & (trades["status"] == "filled" if "status" in trades.columns else True)
                    ]
                )
            )
        row: dict[str, Any] = {
            "symbol": symbol,
            "name": names.get(symbol, symbol),
            "total_pnl": float(summary.get("total_pnl", 0.0)),
            "total_return_pct": float(summary.get("total_return_pct", 0.0)),
            "realized_pnl": float(summary.get("realized_pnl", 0.0)),
            "unrealized_pnl": float(summary.get("unrealized_pnl", 0.0)),
            "trade_count": trade_count,
            "holding_days": int(holding_days.get(symbol, 0)),
            "cycles": cycles,
        }
        if drawdown_info is not None:
            info = drawdown_info.get(symbol, {})
            row["drawdown_contribution"] = float(info.get("drawdown_contribution", 0.0))
            row["peak_market_value"] = float(info.get("peak_market_value", 0.0))
            row["trough_market_value"] = float(info.get("trough_market_value", 0.0))
            if drawdown_window:
                row["peak_date"] = _serialize_date(drawdown_window.get("peak_date"))
                row["trough_date"] = _serialize_date(drawdown_window.get("trough_date"))
                row["drawdown_window_start"] = _serialize_date(drawdown_window.get("drawdown_window_start"))
                row["drawdown_window_end"] = _serialize_date(drawdown_window.get("drawdown_window_end"))
        rows.append(row)
    for idx, row in enumerate(rows[:top_n], start=1):
        row["rank"] = idx
    return rows[:top_n]


def build_symbol_attribution_artifact(
    *,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    repository: MarketDataRepository,
    market: str,
    sample_start: str,
    sample_end: str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build symbol-level profit/drawdown attribution artifact."""
    cycles_by_symbol = build_trade_cycles(trades)
    pnl_summary = _symbol_pnl_summary(trades, positions, cycles_by_symbol)
    holding_days = _holding_days_by_symbol(positions)
    drawdown_window = _portfolio_drawdown_window(equity_curve)
    drawdown_info = _drawdown_contributions(positions, drawdown_window)

    # 计算组合总盈亏，用于计算每只股票的盈亏占比
    total_portfolio_pnl = sum(s["total_pnl"] for s in pnl_summary.values())

    profit_symbols = sorted(
        pnl_summary.keys(),
        key=lambda sym: (pnl_summary[sym]["total_pnl"], sym),
        reverse=True,
    )[:top_n]
    # 按实际盈亏排序（亏损最大的在前），而非回撤窗口市值快照
    loss_symbols = sorted(
        pnl_summary.keys(),
        key=lambda sym: (pnl_summary[sym]["total_pnl"], sym),
    )[:top_n]

    trade_symbols: set[str] = set()
    if not trades.empty and "symbol" in trades.columns:
        trade_symbols = {str(s) for s in trades["symbol"].astype(str).tolist()}
    all_traded_symbols = sorted(set(cycles_by_symbol.keys()) | trade_symbols)
    names, misses = load_instrument_names(repository, market, all_traded_symbols)

    top_profit = _build_ranked_rows(
        profit_symbols,
        cycles_by_symbol=cycles_by_symbol,
        pnl_summary=pnl_summary,
        holding_days=holding_days,
        names=names,
        trades=trades,
        top_n=top_n,
    )
    top_drawdown = _build_ranked_rows(
        loss_symbols,
        cycles_by_symbol=cycles_by_symbol,
        pnl_summary=pnl_summary,
        holding_days=holding_days,
        names=names,
        trades=trades,
        drawdown_info=drawdown_info,
        drawdown_window=drawdown_window,
        top_n=top_n,
    )

    # 为每只股票添加盈亏占比（该股盈亏 / 组合总盈亏）
    for row in top_profit + top_drawdown:
        if total_portfolio_pnl != 0:
            row["pnl_share_pct"] = float(row["total_pnl"] / total_portfolio_pnl * 100)
        else:
            row["pnl_share_pct"] = 0.0

    return {
        "artifact_type": "symbol_attribution_v1",
        "market": market,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "portfolio_drawdown": {
            "max_drawdown": drawdown_window["max_drawdown"],
            "peak_date": _serialize_date(drawdown_window.get("peak_date")),
            "trough_date": _serialize_date(drawdown_window.get("trough_date")),
            "drawdown_window_start": _serialize_date(drawdown_window.get("drawdown_window_start")),
            "drawdown_window_end": _serialize_date(drawdown_window.get("drawdown_window_end")),
        },
        "total_portfolio_pnl": float(total_portfolio_pnl),
        "top_profit_contributors": top_profit,
        "top_drawdown_contributors": top_drawdown,
        "symbol_names": names,
        "name_lookup_misses": misses,
    }
