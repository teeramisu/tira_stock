"""
回测引擎单元测试：使用 MU (Micron) 真实行情数据，
验证「卖出后净值不应再变动」以及未平仓持仓的正确处理。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtest.engine import (
    run_backtest,
    run_sma_crossover,
    run_rsi,
    run_bollinger,
    STRATEGY_SMA,
    STRATEGY_RSI,
    STRATEGY_BOLLINGER,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_mu_df() -> pd.DataFrame:
    """加载 MU 2 年 OHLCV fixture 并构造 DataFrame。"""
    path = os.path.join(FIXTURE_DIR, "mu_ohlcv_2y.json")
    with open(path) as f:
        raw = json.load(f)
    records = raw["data"]
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 核心测试：卖出后净值不应再变动（除非有未平仓持仓）
# ---------------------------------------------------------------------------


class TestEquityAfterLastSell:
    """验证「平仓后净值不变」以及「未平仓持仓正确反映」。"""

    def test_mu_sma_equity_flat_after_last_sell_when_no_open(self):
        """SMA 策略：如果回测结束时没有未平仓持仓，
        最后一次平仓后的净值应该完全不变（全现金）。"""
        df = _load_mu_df()
        res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)

        completed = [t for t in res.trades if not t.is_open]
        open_trades = [t for t in res.trades if t.is_open]

        # 如果没有未平仓，则最后平仓日之后净值应恒定
        if not open_trades and completed:
            last_exit = completed[-1].exit_date
            equity_after_exit = []
            for d, v in res.equity_curve.items():
                if str(d.date()) >= last_exit:
                    equity_after_exit.append(v)

            if len(equity_after_exit) >= 2:
                first_val = equity_after_exit[0]
                for val in equity_after_exit[1:]:
                    assert abs(val - first_val) < 0.01, (
                        f"平仓日 {last_exit} 后净值不应变化: "
                        f"expected {first_val:.2f}, got {val:.2f}"
                    )

    def test_mu_sma_open_position_tracked(self):
        """SMA 策略：如果回测结束时仍有持仓，
        open_position 应正确记录入场信息和浮动盈亏。"""
        df = _load_mu_df()
        res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)

        open_trades = [t for t in res.trades if t.is_open]

        if open_trades:
            # 应有且仅有一笔未平仓
            assert len(open_trades) == 1
            ot = open_trades[0]
            assert ot.exit_date == ""
            assert res.open_position is not None
            assert res.open_position["entry_date"] == ot.entry_date
            assert res.open_position["shares"] == ot.shares

            # 浮动盈亏 = (现价 - 入场价) * 股数
            expected_pnl = (
                res.open_position["current_price"] - res.open_position["entry_price"]
            ) * res.open_position["shares"]
            assert abs(res.open_position["unrealized_pnl"] - expected_pnl) < 1.0

    def test_mu_sma_equity_rise_explained_by_open_position(self):
        """关键测试：如果最后平仓后净值上升，必须能由未平仓持仓解释。
        即：净值增量 ≈ 未平仓持仓的浮动盈亏。"""
        df = _load_mu_df()
        res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)

        completed = [t for t in res.trades if not t.is_open]
        open_trades = [t for t in res.trades if t.is_open]

        if not completed:
            return  # 无交易，跳过

        last_exit = completed[-1].exit_date
        # 找到平仓日的净值
        exit_equity = None
        for d, v in res.equity_curve.items():
            if str(d.date()) == last_exit:
                exit_equity = v
                break

        if exit_equity is None:
            return

        final_equity = res.equity_curve.iloc[-1]
        equity_diff = final_equity - exit_equity

        if abs(equity_diff) < 0.01:
            # 净值没变，没有问题
            return

        # 净值变了，必须有未平仓持仓来解释
        assert open_trades, (
            f"平仓日 {last_exit} 后净值变化 {equity_diff:+.2f}，"
            f"但没有未平仓持仓来解释！"
        )
        assert res.open_position is not None
        # 浮动盈亏应约等于净值变化
        assert abs(res.open_position["unrealized_pnl"] - equity_diff) < 1.0, (
            f"浮动盈亏 {res.open_position['unrealized_pnl']:.2f} "
            f"与净值变化 {equity_diff:.2f} 不一致"
        )

    def test_mu_rsi_open_position(self):
        """RSI 策略同样验证：未平仓持仓被正确追踪。"""
        df = _load_mu_df()
        res = run_rsi(df, initial_cash=100_000.0)

        open_trades = [t for t in res.trades if t.is_open]
        completed = [t for t in res.trades if not t.is_open]

        if open_trades:
            assert res.open_position is not None
            op = res.open_position
            assert op["shares"] > 0
            assert op["entry_price"] > 0
            assert op["current_price"] > 0

            # n_trades 只计已完成交易
            assert res.n_trades == len(completed)

    def test_mu_bollinger_open_position(self):
        """布林带策略同样验证。"""
        df = _load_mu_df()
        res = run_bollinger(df, initial_cash=100_000.0)

        open_trades = [t for t in res.trades if t.is_open]
        completed = [t for t in res.trades if not t.is_open]

        if open_trades:
            assert res.open_position is not None
            assert res.n_trades == len(completed)

    def test_mu_no_open_position_means_flat_equity(self):
        """通用验证：如果 open_position 为 None，
        则回测最后一段净值必须是平的（全现金）。"""
        df = _load_mu_df()

        for strategy, kwargs in [
            (STRATEGY_SMA, {"fast_period": 10, "slow_period": 30}),
            (STRATEGY_RSI, {}),
            (STRATEGY_BOLLINGER, {}),
        ]:
            res = run_backtest(df, strategy, initial_cash=100_000.0, **kwargs)

            if res.open_position is None:
                # 没有持仓 → 最后一段净值应恒定
                curve_vals = res.equity_curve.values
                if len(curve_vals) < 2:
                    continue
                # 从最后一个交易变化日开始检查
                # 找到最后一次净值变化的位置
                last_change_idx = len(curve_vals) - 1
                for i in range(len(curve_vals) - 1, 0, -1):
                    if abs(curve_vals[i] - curve_vals[i - 1]) > 0.01:
                        break
                    last_change_idx = i - 1

                # 从 last_change_idx 到最后应该全部相同
                tail = curve_vals[last_change_idx:]
                if len(tail) >= 2:
                    for v in tail[1:]:
                        assert abs(v - tail[0]) < 0.01, (
                            f"[{strategy}] open_position=None 但尾部净值不恒定: "
                            f"{tail[0]:.2f} vs {v:.2f}"
                        )


# ---------------------------------------------------------------------------
# 辅助测试：交易记录完整性
# ---------------------------------------------------------------------------


class TestTradeRecordIntegrity:
    """验证交易记录的完整性和一致性。"""

    def test_all_trades_have_consistent_state(self):
        """每笔交易的状态必须自洽：
        - 已平仓: exit_date 非空, pnl 有值, is_open=False
        - 未平仓: exit_date 为空, is_open=True
        """
        df = _load_mu_df()
        res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)

        for t in res.trades:
            if t.is_open:
                assert t.exit_date == "", f"未平仓交易应有空 exit_date: {t.exit_date}"
            else:
                assert t.exit_date != "", "已平仓交易应有非空 exit_date"
                assert not np.isnan(t.pnl), "已平仓交易应有非 NaN pnl"

    def test_n_trades_excludes_open(self):
        """n_trades 只统计已完成交易。"""
        df = _load_mu_df()
        res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)

        completed = [t for t in res.trades if not t.is_open]
        assert res.n_trades == len(completed)

    def test_final_value_equals_last_equity(self):
        """final_value 必须等于 equity_curve 最后一个值。"""
        df = _load_mu_df()
        res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)

        assert abs(res.final_value - res.equity_curve.iloc[-1]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
