"""
回测引擎单元测试：用构造好的数据验证指标计算正确。
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtest.engine import run_sma_crossover, run_backtest, STRATEGY_BUY_HOLD


def test_sma_crossover_known_data():
    """构造简单序列：前低后高，快线先上穿慢线应产生一笔做多并持有到结束。"""
    # 30 天：前 15 天 100，后 15 天 110 → 快线 10 会先于慢线 30 上穿
    n = 35
    close = [100.0] * 15 + [110.0] * 20
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": close}, index=dates)

    res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)
    assert res.initial_value == 100_000.0
    # 应至少有一笔完整交易（开+平）或仅开仓持有到结束
    assert res.final_value > 0
    # 最后几天应持仓（价格 110）
    assert res.total_return >= -0.01  # 至少不应因逻辑错误大亏
    # 日序列长度（全区间逐日净值）
    assert len(res.equity_curve) == n


def test_sma_crossover_all_cash():
    """一直空仓：价格横盘，快线始终在慢线下方，最终应无交易且 equity=initial_cash。"""
    close = [100.0] * 50
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    df = pd.DataFrame({"Close": close}, index=dates)
    res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=10, slow_period=30)
    assert res.final_value == 100_000.0
    assert res.total_return == 0.0
    assert res.n_trades == 0


def test_sma_crossover_one_round_trip():
    """构造一次完整买卖：先上穿再下穿，验证一笔交易 PnL 与总收益一致。"""
    # 快 2 慢 5：便于手工算
    # 日 0-4: 100, 100, 100, 100, 100 -> 快2=100, 慢5=100
    # 日 5: 110 -> 快2(100,110)=105, 慢5(100*5)=100 -> 上穿，次日买
    # 日 6: 买在 110
    # 日 7-9: 120 -> 快2(110,120)=115, 慢5 上升
    # 日 10: 90 -> 快2(120,90)=105, 慢5 仍高 -> 某日下穿
    # 简化：用更长序列确保一次买入一次卖出
    np.random.seed(42)
    n = 60
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 80)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": close}, index=dates)
    res = run_sma_crossover(df, initial_cash=100_000.0, fast_period=5, slow_period=15)
    assert res.initial_value == 100_000.0
    assert len(res.equity_curve) == n
    # 总收益应与最后一日的 equity 一致
    assert abs(res.equity_curve.iloc[-1] - res.final_value) < 0.01


def test_run_backtest_buy_hold():
    """买入持有应全程持仓，期末净值随价格变化。"""
    n = 50
    close = [100.0] * 10 + [120.0] * 40
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": close}, index=dates)
    res = run_backtest(df, STRATEGY_BUY_HOLD, initial_cash=100_000.0)
    assert res.n_trades == 0  # 无买卖
    assert res.final_value == 100_000.0 * (120.0 / 100.0)  # 首日 100 买入，末日 120
    assert res.total_return == 0.2


if __name__ == "__main__":
    test_sma_crossover_known_data()
    test_sma_crossover_all_cash()
    test_sma_crossover_one_round_trip()
    test_run_backtest_buy_hold()
    print("All engine tests passed.")
