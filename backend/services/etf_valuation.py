"""美股 ETF 估值筛选：基于 P/E、P/B、股息率等找出相对低估的 Top10。"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

# 常见美股股票型 ETF（数量控制以缩短总耗时，避免卡住）
US_EQUITY_ETFS = [
    "SPY", "QQQ", "VOO", "VTI", "IWM", "DIA", "VTV", "VUG", "SCHD", "VYM",
    "IVE", "RSP", "IJR", "IWD", "IWF", "VB", "DVY", "VBR",
]
FETCH_TIMEOUT = 5  # 单只 ETF 最多等 5 秒
MAX_WORKERS = 6    # 并发数，避免把 Yahoo 拉爆


def _get_info(ticker: str) -> Optional[dict[str, Any]]:
    """拉取单只 ETF 的 info，失败返回 None。"""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        if not info or not isinstance(info, dict):
            return None
        return info
    except Exception:
        return None


def _get_performance(ticker: str) -> dict[str, Any]:
    """拉取单只 ETF 近 1 年行情，计算 1月/6月/1年涨幅和 1 年胜率。"""
    result: dict[str, Any] = {
        "return_1m": None,
        "return_6m": None,
        "return_1y": None,
        "win_rate_1y": None,
    }
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period="1y", auto_adjust=True)
        if df is None or df.empty or len(df) < 20:
            return result
        close = df["Close"].dropna()
        if close.empty:
            return result

        latest = float(close.iloc[-1])

        # 涨幅计算
        def _ret(days_back: int) -> Optional[float]:
            target_date = close.index[-1] - timedelta(days=days_back)
            mask = close.index >= target_date
            if not mask.any():
                return None
            earliest = float(close.loc[mask].iloc[0])
            if earliest <= 0:
                return None
            return round((latest / earliest - 1) * 100, 2)

        result["return_1m"] = _ret(30)
        result["return_6m"] = _ret(182)
        result["return_1y"] = _ret(365)

        # 1 年胜率：过去 1 年中正收益交易日占比
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) > 0:
            wins = (daily_ret > 0).sum()
            result["win_rate_1y"] = round(wins / len(daily_ret) * 100, 1)

    except Exception:
        pass
    return result


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        f = float(v)
        return f if f == f else default  # reject nan
    except (TypeError, ValueError):
        return default


def _reasons(pe: Optional[float], pb: Optional[float], div: Optional[float], name: str) -> list[str]:
    """根据估值指标生成简短理由。"""
    reasons = []
    if pe is not None and pe > 0 and pe < 25:
        reasons.append(f"市盈率 {pe:.1f}，估值相对偏低")
    elif pe is not None and pe > 0 and pe < 18:
        reasons.append(f"市盈率 {pe:.1f}，处于较低水平")
    if pb is not None and pb > 0 and pb < 4:
        reasons.append(f"市净率 {pb:.2f}，价格相对净资产不高")
    if div is not None and div > 0:
        reasons.append(f"股息率 {div:.2%}，具备分红吸引力")
    if not reasons:
        reasons.append("综合估值与股息在候选 ETF 中相对占优")
    return reasons


def get_undervalued_etfs(top_n: int = 10) -> list[dict[str, Any]]:
    """
    从一批美股股票型 ETF 中，用 P/E、P/B、股息率 做简单打分，选出相对「低估」的 Top N。
    同时计算 1月/6月/1年涨幅和 1 年胜率。
    """
    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # 提交 info 和 performance 两类任务
        info_futures = {pool.submit(_get_info, symbol): symbol for symbol in US_EQUITY_ETFS}
        perf_futures = {pool.submit(_get_performance, symbol): symbol for symbol in US_EQUITY_ETFS}

        # 收集 performance 结果
        perf_results: dict[str, dict] = {}
        for fut in perf_futures:
            symbol = perf_futures[fut]
            try:
                perf_results[symbol] = fut.result(timeout=FETCH_TIMEOUT + 5)
            except Exception:
                perf_results[symbol] = {}

        for fut in info_futures:
            symbol = info_futures[fut]
            try:
                info = fut.result(timeout=FETCH_TIMEOUT)
            except (FuturesTimeoutError, Exception):
                continue
            if not info:
                continue
            short_name = (info.get("shortName") or info.get("longName") or symbol).strip()
            pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
            pb = _safe_float(info.get("priceToBook"))
            div_raw = _safe_float(info.get("dividendYield"))
            div = div_raw if div_raw is not None else _safe_float(info.get("yield"))
            # 部分 API 返回小数，部分返回百分数
            if div is not None and div < 0.01 and div > 0:
                div = div * 100.0

            # 至少要有 P/E 或 P/B 之一才参与排序
            if pe is None and pb is None:
                continue

            # 负 P/E、负 P/B 视为无效
            if pe is not None and pe <= 0:
                pe = None
            if pb is not None and pb <= 0:
                pb = None

            # 简单得分：P/E 越低越好，P/B 越低越好，股息率越高越好（归一化到 0-1 区间后加权）
            score = 0.0
            if pe is not None:
                score += max(0, 1.0 - (pe - 5) / 35.0) * 0.4
            if pb is not None:
                score += max(0, 1.0 - (pb - 0.5) / 4.0) * 0.4
            if div is not None and div > 0:
                score += min(1.0, div / 5.0) * 0.2

            reasons = _reasons(pe, pb, div, short_name)
            perf = perf_results.get(symbol, {})
            candidates.append({
                "symbol": symbol,
                "name": short_name,
                "pe": round(pe, 2) if pe is not None else None,
                "pb": round(pb, 2) if pb is not None else None,
                "dividend_yield_pct": round(div, 2) if div is not None else None,
                "score": round(score, 4),
                "reasons": reasons,
                "return_1m": perf.get("return_1m"),
                "return_6m": perf.get("return_6m"),
                "return_1y": perf.get("return_1y"),
                "win_rate_1y": perf.get("win_rate_1y"),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def get_etf_holdings(symbol: str, top_n: int = 20) -> list[dict[str, Any]]:
    """从 stockanalysis.com 获取 ETF 的 Top N 持仓（代码、名称、权重）。"""
    url = f"https://stockanalysis.com/etfs/{symbol.lower()}/holdings/"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception:
        # fallback URL format
        url = f"https://stockanalysis.com/etf/{symbol.lower()}/holdings/"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception:
            return []

    rows = re.findall(r"<tr[^>]*>.*?</tr>", r.text, re.DOTALL)
    holdings: list[dict[str, Any]] = []
    for row in rows:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(tds) < 4:
            continue
        rank_text = re.sub(r"<[^>]+>", "", tds[0]).strip()
        sym = re.sub(r"<[^>]+>", "", tds[1]).strip()
        name = re.sub(r"<[^>]+>", "", tds[2]).strip()
        weight_text = re.sub(r"<[^>]+>", "", tds[3]).strip()
        # Parse weight percentage
        weight_match = re.search(r"([\d.]+)%", weight_text)
        if not weight_match:
            continue
        try:
            rank = int(rank_text)
            weight = float(weight_match.group(1))
        except (ValueError, TypeError):
            continue
        holdings.append({
            "rank": rank,
            "symbol": sym,
            "name": name,
            "weight_pct": weight,
        })
        if len(holdings) >= top_n:
            break
    return holdings
