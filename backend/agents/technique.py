"""Technique Agent — technical analysis combining MA, RSI, MACD, KDJ signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from memory.models import SignalRecord
from memory.session import SessionMemory


class TechniqueAgent(BaseAgent):
    """Technical analysis agent using indicator-based scoring.

    Indicators used:
      - MA alignment (5/10/20/60)
      - RSI oversold/overbought
      - MACD histogram direction
      - KDJ golden/death cross
    """

    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

    def __init__(self, **kwargs: Any):
        super().__init__(name="technique", description="Technical indicator analysis", **kwargs)

    def analyze(self, ticker: str | None = None, session: SessionMemory | None = None, **_: Any) -> SignalRecord:
        session = session or self.session
        self._trace("start", f"TechniqueAgent: ticker={ticker}")

        bars = self._fetch_bars(ticker)
        if not bars or len(bars) < 30:
            return SignalRecord(
                agent_name=self.name, signal_type="score", score=50.0, confidence=0.3,
                reasoning="行情数据不足，无法进行技术分析", ticker=ticker,
                metadata={"data_sufficient": False, "calc_steps": ["① 行情数据不足(<30根K线) → 默认 score=50, confidence=0.3"]},
            )

        df = pd.DataFrame(bars)
        scores = []
        details = []

        # 1. MA alignment score
        ma_score, ma_detail = self._score_ma(df)
        scores.append(("ma", ma_score, 0.25))
        details.append(ma_detail)

        # 2. RSI score
        rsi_score, rsi_detail = self._score_rsi(df)
        scores.append(("rsi", rsi_score, 0.25))
        details.append(rsi_detail)

        # 3. MACD score
        macd_score, macd_detail = self._score_macd(df)
        scores.append(("macd", macd_score, 0.30))
        details.append(macd_detail)

        # 4. KDJ score
        kdj_score, kdj_detail = self._score_kdj(df)
        scores.append(("kdj", kdj_score, 0.20))
        details.append(kdj_detail)

        # Weighted composite
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 50.0
        composite = max(0, min(100, round(composite, 1)))

        # Confidence: based on signal agreement
        signal_values = [s for _, s, _ in scores]
        agreement = 1.0 - np.std(signal_values) / 50 if len(signal_values) > 1 else 0.5
        confidence = min(1.0, max(0.3, round(0.4 + 0.4 * agreement, 2)))

        reasoning = "技术面: " + "; ".join(details)

        # 构建详细计算步骤
        calc_steps = []
        calc_steps.append(f"① K线数据: {len(bars)} 根")
        weight_map = {"ma": "25%", "rsi": "25%", "macd": "30%", "kdj": "20%"}
        step_num = 2
        for name, s, w in scores:
            matched_detail = next((d for d in details if name in d.lower() or name.upper() in d), details[step_num - 2] if step_num - 2 < len(details) else "")
            calc_steps.append(f"{'②③④⑤'[step_num-2]} {name.upper()}: 得分 {s} (权重 {weight_map.get(name, '?')}) — {matched_detail}")
            step_num += 1
        weighted_parts = [f"{s}×{w}" for _, s, w in scores]
        calc_steps.append(f"{'②③④⑤⑥⑦'[step_num-2]} 加权合并: ({' + '.join(weighted_parts)}) / {total_weight:.2f} = {composite}")
        step_num += 1
        calc_steps.append(f"{'②③④⑤⑥⑦'[step_num-2]} 信号一致性: std={np.std(signal_values):.1f}, agreement={agreement:.2f}")
        step_num += 1
        calc_steps.append(f"{'②③④⑤⑥⑦'[step_num-2]} 置信度: 0.4 + 0.4×{agreement:.2f} = {confidence:.2f}")
        step_num += 1
        calc_steps.append(f"{'②③④⑤⑥⑦'[step_num-2]} 最终: score={composite}, confidence={confidence:.2f}")

        if self.llm is not None:
            llm_reason = self._llm_reason(
                f"作为技术分析师，综合评分 {composite}/100。"
                f"指标信号: {', '.join(details)}。"
                f"请用简洁中文给出技术面建议（50字以内）：",
            )
            if llm_reason:
                reasoning = llm_reason

        self._trace("complete", f"TechniqueAgent: score={composite}, confidence={confidence}")
        return SignalRecord(
            agent_name=self.name, signal_type="score", score=composite, confidence=confidence,
            reasoning=reasoning, ticker=ticker,
            metadata={
                "ma_score": ma_score, "rsi_score": rsi_score,
                "macd_score": macd_score, "kdj_score": kdj_score,
                "data_sufficient": True, "calc_steps": calc_steps,
            },
        )

    def _fetch_bars(self, ticker: str | None) -> list[dict[str, Any]]:
        if not ticker:
            return []
        tool = self.tools.get("market_data")
        if tool is None:
            return []
        try:
            result = tool.execute(ticker=ticker, market="cn")
            # Try to get full OHLCV from repository directly
            from marketdata.repository import MarketDataRepository
            from marketdata.db import connect as connect_market_db
            conn = connect_market_db()
            repo = MarketDataRepository(conn)
            bars = repo.read_daily_bars(market="cn", symbol=ticker)
            return [{"close": b.close, "open": b.open, "high": b.high, "low": b.low, "volume": b.volume}
                    for b in bars[-120:]]
        except Exception:
            return []

    def _score_ma(self, df: pd.DataFrame) -> tuple[float, str]:
        close = df["close"].astype(float)
        ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else None
        ma10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else None
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        if ma5 is None or ma10 is None or ma20 is None:
            return (50.0, "MA数据不足")
        latest = close.iloc[-1]
        if latest > ma5 > ma10 > ma20:
            return (75.0, "均线多头排列")
        if latest < ma5 < ma10 < ma20:
            return (25.0, "均线空头排列")
        return (50.0, "均线交织")

    def _score_rsi(self, df: pd.DataFrame) -> tuple[float, str]:
        close = df["close"].astype(float)
        if len(close) < 15:
            return (50.0, "RSI数据不足")
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        if loss == 0:
            rsi = 100.0
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        if rsi < self.RSI_OVERSOLD:
            return (70.0, f"RSI={rsi:.0f}超卖")
        if rsi > self.RSI_OVERBOUGHT:
            return (30.0, f"RSI={rsi:.0f}超买")
        return (50.0, f"RSI={rsi:.0f}中性")

    def _score_macd(self, df: pd.DataFrame) -> tuple[float, str]:
        close = df["close"].astype(float)
        if len(close) < 35:
            return (50.0, "MACD数据不足")
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        if len(hist) < 2:
            return (50.0, "MACD数据不足")
        curr = hist.iloc[-1]
        prev = hist.iloc[-2]
        if prev < 0 and curr > 0:
            return (75.0, "MACD金叉")
        if prev > 0 and curr < 0:
            return (25.0, "MACD死叉")
        if curr > 0:
            return (60.0, "MACD多头")
        return (40.0, "MACD空头")

    def _score_kdj(self, df: pd.DataFrame) -> tuple[float, str]:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        if len(close) < 10:
            return (50.0, "KDJ数据不足")
        low_9 = low.rolling(9).min()
        high_9 = high.rolling(9).max()
        rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        if len(k) < 2:
            return (50.0, "KDJ数据不足")
        if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]:
            return (70.0, f"KDJ金叉(J={j.iloc[-1]:.0f})")
        if k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]:
            return (30.0, f"KDJ死叉(J={j.iloc[-1]:.0f})")
        return (50.0, f"KDJ中性(J={j.iloc[-1]:.0f})")
