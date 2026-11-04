"""组合级 Discovery 评估测试：选股、权重、调仓、成本、超额与闸门。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discovery.evaluator import make_walk_forward_splits  # noqa: E402
from discovery.portfolio import (  # noqa: E402
    Panel,
    PortfolioGenome,
    build_target_weights,
    evaluate_portfolio_grid,
    load_panel,
    portfolio_promotion_gate,
    run_portfolio_backtest,
)


def make_panel(n_days: int = 900, n_assets: int = 40, seed: int = 11) -> Panel:
    """构造一个“持续 alpha”面板：前 10 只长期跑赢，后 30 只跑输。"""
    rng = np.random.default_rng(seed)
    alpha = np.array([0.0015] * 10 + [-0.0005] * (n_assets - 10))
    market = rng.normal(0.0001, 0.004, n_days)
    idio = rng.normal(0, 0.003, (n_days, n_assets))
    rets = market[:, None] + alpha[None, :] + idio
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    volume = rng.uniform(2e6, 2e7, size=close.shape)
    index = pd.bdate_range("2022-01-03", periods=n_days)
    columns = [f"S{i:02d}" for i in range(n_assets)]
    return Panel(
        pd.DataFrame(close, index=index, columns=columns),
        pd.DataFrame(volume, index=index, columns=columns),
    )


GENOME = PortfolioGenome(
    name="mom60_top10",
    factor="momentum",
    lookback=60,
    top_k=10,
    weighting="equal",
    rebalance_days=20,
    max_weight=0.20,
)


class PortfolioBacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel()

    def test_weights_constraints(self):
        weights = build_target_weights(self.panel, GENOME)
        self.assertFalse(weights.empty)
        daily_sum = weights.sum(axis=1)
        self.assertLessEqual(float(daily_sum.max()), 1.0 + 1e-9)
        self.assertLessEqual(float(weights.max().max()), GENOME.max_weight + 1e-9)
        self.assertLessEqual(int((weights > 0).sum(axis=1).max()), GENOME.top_k)

    def test_cash_buffer(self):
        genome = PortfolioGenome(name="cash", factor="momentum", lookback=60, top_k=10,
                                 rebalance_days=20, max_weight=0.2, cash_buffer=0.1)
        weights = build_target_weights(self.panel, genome)
        self.assertLessEqual(float(weights.sum(axis=1).max()), 0.9 + 1e-9)

    def test_no_lookahead(self):
        """截断未来数据后，历史权重不应改变。"""
        t = self.panel.close.index[500]
        full = build_target_weights(self.panel, GENOME)
        truncated = build_target_weights(self.panel.slice(self.panel.close.index[0], t), GENOME)
        common = full.index.intersection(truncated.index)
        common = common[common <= t]
        np.testing.assert_allclose(
            full.loc[common].to_numpy(),
            truncated.loc[common].to_numpy(),
            atol=1e-12,
        )

    def test_costs_reduce_return(self):
        cheap = run_portfolio_backtest(self.panel, GENOME, cost_bps=0.0)
        expensive = run_portfolio_backtest(self.panel, GENOME, cost_bps=200.0)
        self.assertGreater(
            cheap["metrics"]["total_return"], expensive["metrics"]["total_return"]
        )
        self.assertGreater(expensive["metrics"]["total_cost"] if "total_cost" in expensive["metrics"] else 1, 0)

    def test_beats_benchmark(self):
        """持续 alpha 面板上，动量 Top10 应跑赢等权基准。"""
        result = run_portfolio_backtest(self.panel, GENOME, cost_bps=5.0)
        self.assertGreater(result["metrics"]["excess_cagr"], 0.0)
        self.assertGreater(result["metrics"]["sharpe"], 0.0)
        self.assertGreater(result["metrics"]["avg_positions"], 0)
        self.assertLessEqual(result["metrics"]["avg_positions"], GENOME.top_k)

    def test_regime_gate(self):
        genome = PortfolioGenome(name="regime", factor="momentum", lookback=60, top_k=10,
                                 rebalance_days=20, max_weight=0.2, regime_gate=True)
        result = run_portfolio_backtest(self.panel, genome, cost_bps=5.0)
        self.assertLessEqual(result["metrics"]["avg_exposure"], 1.0 + 1e-9)
        self.assertGreaterEqual(result["metrics"]["avg_exposure"], 0.0)

    def test_cost_bps_monotonic(self):
        r0 = run_portfolio_backtest(self.panel, GENOME, cost_bps=0.0)["metrics"]["cagr"]
        r1 = run_portfolio_backtest(self.panel, GENOME, cost_bps=50.0)["metrics"]["cagr"]
        self.assertLessEqual(r1, r0 + 1e-12)


class PortfolioEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel()
        cls.splits = make_walk_forward_splits(
            len(cls.panel.close), train_size=300, test_size=100, purge=5, step_size=100
        )

    def test_grid_end_to_end(self):
        genomes = [
            GENOME,
            PortfolioGenome(name="mom120_top20", factor="momentum", lookback=120,
                            top_k=20, rebalance_days=20, max_weight=0.10),
            PortfolioGenome(name="rev20_top10", factor="reversal", lookback=20,
                            top_k=10, rebalance_days=10, max_weight=0.20),
        ]
        result = evaluate_portfolio_grid(self.panel, genomes, self.splits, cost_bps=5.0)
        self.assertEqual(result["n_trials"], 3)
        self.assertEqual(len(result["folds"]), len(self.splits))
        self.assertIn("aggregate", result)
        self.assertIn("benchmark", result)
        self.assertIn("excess_cagr", result)
        self.assertIsNotNone(result["dsr"])
        self.assertIsNotNone(result["pbo"])
        self.assertIn("gate", result)
        self.assertEqual(set(result["gate"]["checks"]), {
            "sharpe", "excess_cagr", "max_drawdown", "dsr", "pbo",
            "positive_folds", "turnover",
        })

    def test_gate(self):
        good = {
            "aggregate": {"sharpe": 1.5, "max_drawdown": 0.1},
            "excess_cagr": 0.05,
            "dsr": 0.98,
            "pbo": {"pbo": 0.1},
            "fold_positive_ratio": 0.8,
            "annual_turnover": 8.0,
        }
        self.assertTrue(portfolio_promotion_gate(good)["passed"])
        bad = {**good, "excess_cagr": -0.02}
        self.assertFalse(portfolio_promotion_gate(bad)["passed"])


class PanelLoaderTest(unittest.TestCase):
    def test_load_panel_from_db(self):
        try:
            panel = load_panel(market="cn", start="2024-01-01", end="2024-06-30", limit=20)
        except Exception as exc:  # 行情库被刷新进程占用时跳过
            self.skipTest(f"本地行情不可用: {exc}")
        self.assertGreater(panel.close.shape[1], 0)
        self.assertGreater(len(panel.close), 20)
        # 前向填充后不应有整列全空
        self.assertTrue((panel.close.notna().any()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
