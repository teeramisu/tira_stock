"""美股 ETF 估值筛选：基于 P/E、P/B、股息率等找出相对低估的 Top10。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional

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
    理由基于当前市盈率、市净率、股息率文字描述。
    使用线程池并发拉取并带超时，避免单只卡住导致接口无响应。
    """
    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_get_info, symbol): symbol for symbol in US_EQUITY_ETFS}
        for fut in futures:
            symbol = futures[fut]
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
            candidates.append({
                "symbol": symbol,
                "name": short_name,
                "pe": round(pe, 2) if pe is not None else None,
                "pb": round(pb, 2) if pb is not None else None,
                "dividend_yield_pct": round(div, 2) if div is not None else None,
                "score": round(score, 4),
                "reasons": reasons,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]
