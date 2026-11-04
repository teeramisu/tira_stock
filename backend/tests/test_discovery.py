"""Discovery Loop P0/P1 测试：统计闸门、purged 评估、实验账本。

运行:
    python -m pytest backend/tests/test_discovery.py -q
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discovery.evaluator import (  # noqa: E402
    capacity_report,
    causality_test,
    cost_stress_test,
    delay_stress_test,
    evaluate_param_grid,
    evaluate_positions,
    expand_grid,
    make_cpcv_splits,
    make_walk_forward_splits,
    positions_to_returns,
    promotion_gate,
    shuffled_returns_significance,
)
from discovery.ledger import ExperimentLedger  # noqa: E402
from discovery.stats import (  # noqa: E402
    benjamini_hochberg,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    max_drawdown,
    performance_summary,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def _prices(n: int = 800, seed: int = 7, drift: float = 0.0004) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    index = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=index, name="close")


def sma_signal(prices: pd.Series, fast: int = 10, slow: int = 30) -> np.ndarray:
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    return (f > s).astype(float).fillna(0.0).to_numpy()


class StatsTest(unittest.TestCase):
    def test_basic_metrics(self):
        returns = [0.10, -0.50, 0.10]
        self.assertAlmostEqual(max_drawdown(returns), 0.5, places=6)
        summary = performance_summary(returns)
        self.assertEqual(summary["n_periods"], 3)
        self.assertGreater(summary["max_drawdown"], 0)
        self.assertTrue(math.isfinite(summary["sharpe"]))

    def test_sharpe_edge_cases(self):
        self.assertEqual(sharpe_ratio([]), 0.0)
        self.assertEqual(sharpe_ratio([0.01]), 0.0)
        self.assertEqual(sharpe_ratio([0.01] * 100), 0.0)  # 零波动

    def test_probabilistic_sharpe(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 500)
        psr = probabilistic_sharpe_ratio(r)
        self.assertIsNotNone(psr)
        self.assertGreater(psr, 0.8)

    def test_deflated_sharpe_penalizes_trials(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.001, 0.01, 500)
        d1 = deflated_sharpe_ratio(r, n_trials=1)
        d1000 = deflated_sharpe_ratio(r, n_trials=1000)
        self.assertIsNotNone(d1)
        self.assertIsNotNone(d1000)
        self.assertGreater(d1, d1000)

    def test_pbo_low_for_persistent_edge(self):
        rng = np.random.default_rng(3)
        t, n = 400, 20
        mat = rng.normal(0, 0.01, (t, n))
        mat[:, 0] = rng.normal(0.003, 0.008, t)  # 持续有效的策略
        result = probability_of_backtest_overfitting(mat)
        self.assertIsNotNone(result)
        self.assertLessEqual(result["pbo"], 0.2)
        self.assertEqual(result["n_candidates"], n)

    def test_pbo_high_for_pure_noise(self):
        rng = np.random.default_rng(4)
        mat = rng.normal(0, 0.01, (300, 10))
        result = probability_of_backtest_overfitting(mat)
        self.assertIsNotNone(result)
        self.assertTrue(0.0 <= result["pbo"] <= 1.0)
        self.assertGreater(result["pbo"], 0.2)

    def test_pbo_invalid(self):
        self.assertIsNone(probability_of_backtest_overfitting(np.zeros((10, 1))))

    def test_benjamini_hochberg(self):
        result = benjamini_hochberg([0.001, 0.01, 0.2, 0.5], alpha=0.1)
        self.assertEqual(result.n_rejected, 2)
        self.assertEqual(result.rejected, [True, True, False, False])
        self.assertEqual(result.n_tests, 4)

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.001, 0.01, 500)
        ci = block_bootstrap_ci(r, statistic="sharpe", n_boot=200, block=20, seed=1)
        self.assertIsNotNone(ci)
        self.assertLessEqual(ci["lower"], ci["upper"])

    def test_empty_inputs(self):
        self.assertEqual(performance_summary([])["n_periods"], 0)
        self.assertIsNone(probabilistic_sharpe_ratio([]))
        self.assertIsNone(deflated_sharpe_ratio([]))
        self.assertIsNone(block_bootstrap_ci([], block=20))


class SplitTest(unittest.TestCase):
    def test_walk_forward_purge(self):
        splits = make_walk_forward_splits(1000, train_size=250, test_size=50, purge=5, step_size=50)
        self.assertGreater(len(splits), 0)
        for s in splits:
            self.assertEqual(len(s.test), 50)
            self.assertLess(s.train[-1], s.test[0])
            self.assertGreaterEqual(s.test[0] - s.train[-1] - 1, 5)

    def test_cpcv_disjoint(self):
        splits = make_cpcv_splits(600, n_groups=6, n_test_groups=2, purge=3, embargo=3)
        self.assertEqual(len(splits), 15)
        for s in splits:
            self.assertEqual(np.intersect1d(s.train, s.test).size, 0)

    def test_invalid_splits(self):
        self.assertEqual(make_walk_forward_splits(10, train_size=100, test_size=10), [])
        self.assertEqual(make_cpcv_splits(10, n_groups=6), [])


class EvaluatorTest(unittest.TestCase):
    def test_positions_to_returns(self):
        prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=pd.bdate_range("2024-01-01", periods=4))
        rets = positions_to_returns(prices, [1, 1, 0, 0], cost_bps=0, lag=1)
        expected = [0.0, 0.1, -0.1, 0.0]
        np.testing.assert_allclose(rets.to_numpy(), expected, atol=1e-9)

    def test_positions_cost(self):
        prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=pd.bdate_range("2024-01-01", periods=4))
        rets = positions_to_returns(prices, [1, 1, 0, 0], cost_bps=100, lag=1)
        # 第 2 天和第 4 天各有一次 100% 换手，各扣 1%
        np.testing.assert_allclose(rets.to_numpy(), [0.0, 0.09, -0.1, -0.01], atol=1e-9)

    def test_evaluate_positions_buy_hold(self):
        prices = _prices(300)
        positions = np.ones(len(prices))
        splits = make_walk_forward_splits(len(prices), train_size=100, test_size=50, purge=3)
        result = evaluate_positions(prices, positions, splits)
        self.assertIn("aggregate", result)
        self.assertGreater(result["aggregate"]["n_periods"], 0)
        self.assertEqual(len(result["folds"]), len(splits))

    def test_causality_ok(self):
        prices = _prices(300)
        result = causality_test(prices, sma_signal, {"fast": 10, "slow": 30}, n_probes=5)
        self.assertTrue(result["passed"], result)

    def test_causality_catches_global_stat(self):
        def cheating(prices: pd.Series) -> np.ndarray:
            return (prices > prices.mean()).astype(float).to_numpy()

        result = causality_test(_prices(300), cheating, {}, n_probes=5)
        self.assertFalse(result["passed"])

    def test_causality_catches_future_shift(self):
        def cheating(prices: pd.Series) -> np.ndarray:
            return (prices.shift(-1) > prices).astype(float).fillna(0).to_numpy()

        result = causality_test(_prices(300), cheating, {}, n_probes=5)
        self.assertFalse(result["passed"])

    def test_param_grid_end_to_end(self):
        prices = _prices(900)
        splits = make_walk_forward_splits(len(prices), train_size=300, test_size=100, purge=5, step_size=100)
        grid = expand_grid({"fast": [5, 10, 20], "slow": [30, 60]})
        result = evaluate_param_grid(prices, sma_signal, grid, splits, cost_bps=5)
        self.assertEqual(result["n_trials"], 6)
        self.assertEqual(len(result["folds"]), len(splits))
        for fold in result["folds"]:
            self.assertIn(fold["best_params"]["fast"], [5, 10, 20])
        self.assertIsNotNone(result["dsr"])
        self.assertIsNotNone(result["pbo"])
        self.assertTrue(0 <= result["pbo"]["pbo"] <= 1)

    def test_delay_and_cost_stress(self):
        prices = _prices(600)
        splits = make_walk_forward_splits(len(prices), train_size=200, test_size=80, purge=3)
        params = {"fast": 10, "slow": 30}
        delay = delay_stress_test(prices, sma_signal, params, splits)
        self.assertIn("retention", delay)
        cost = cost_stress_test(prices, sma_signal, params, splits, cost_bps=10)
        self.assertEqual(len(cost["results"]), 3)
        self.assertLessEqual(
            cost["results"][-1]["cagr"], cost["results"][0]["cagr"] + 1e-12
        )

    def test_shuffle_significance(self):
        prices = _prices(400, drift=0.001)
        splits = make_walk_forward_splits(len(prices), train_size=150, test_size=50, purge=3)
        result = shuffled_returns_significance(
            prices, sma_signal, {"fast": 5, "slow": 20}, splits, n_permutations=10, seed=1
        )
        self.assertIn("p_value", result)
        self.assertTrue(0 <= result["p_value"] <= 1)

    def test_capacity(self):
        prices = _prices(10)
        volume = pd.Series(100_000.0, index=prices.index)
        result = capacity_report(np.ones(10), prices, volume, capital=1_000_000)
        self.assertFalse(result["passed"])  # 100% 仓位一天建仓，参与率 10% > 5%

    def test_promotion_gate(self):
        good = {
            "aggregate": {"sharpe": 1.5, "max_drawdown": 0.1},
            "dsr": 0.98,
            "pbo": {"pbo": 0.1},
            "fold_positive_ratio": 0.8,
            "fold_sharpe_mean": 0.5,
        }
        self.assertTrue(promotion_gate(good)["passed"])
        bad = {**good, "dsr": 0.5}
        self.assertFalse(promotion_gate(bad)["passed"])


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ExperimentLedger(Path(self.tmp.name) / "ledger.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidate_lifecycle(self):
        cid = self.ledger.record_candidate(
            {"alpha": "sma", "params": {"fast": 10, "slow": 30}},
            parent_ids=["p1", "p2"],
            archive_cell="bull|short",
        )
        self.assertEqual(len(cid), 64)
        row = self.ledger.get_candidate(cid)
        self.assertEqual(row["stage"], "generated")
        self.assertEqual(row["parent_ids"], ["p1", "p2"])
        self.assertTrue(self.ledger.update_candidate_status(cid, stage="validated", promotion_status="approved"))
        row = self.ledger.get_candidate(cid)
        self.assertEqual(row["stage"], "validated")
        self.assertEqual(row["promotion_status"], "approved")

    def test_experiment_and_leaderboard(self):
        e1 = self.ledger.record_experiment(
            genome_hash="g1", strategy="sma", params={"fast": 10},
            metrics={"sharpe": 1.2}, dsr=0.96, pbo=0.1, status="completed",
        )
        self.ledger.record_experiment(
            genome_hash="g2", strategy="rsi", params={"period": 14},
            metrics={"sharpe": 0.3}, dsr=0.5, pbo=0.5, status="rejected",
        )
        self.assertEqual(len(e1), 64)
        rows = self.ledger.list_experiments()
        self.assertEqual(len(rows), 2)
        board = self.ledger.leaderboard(metric="dsr", limit=1)
        self.assertEqual(board[0]["genome_hash"], "g1")
        rejected = self.ledger.list_experiments(status="rejected")
        self.assertEqual(len(rejected), 1)

    def test_lessons_and_summary(self):
        self.ledger.record_lesson(
            hypothesis="高换手动量在震荡市失效",
            failure_mode="regime_mismatch",
            evidence={"sharpe": -0.2},
        )
        self.assertEqual(len(self.ledger.list_lessons()), 1)
        summary = self.ledger.summary()
        self.assertEqual(summary["lessons"], 1)
        self.assertIn("path", summary)

    def test_idempotent_candidate(self):
        genome = {"alpha": "x"}
        a = self.ledger.record_candidate(genome)
        b = self.ledger.record_candidate(genome)
        self.assertEqual(a, b)
        self.assertEqual(len(self.ledger.list_candidates()), 1)


class DiscoveryApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.main import app

        cls.client = TestClient(app)

    def test_summary(self):
        res = self.client.get("/api/discovery/summary")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("experiments", body)
        self.assertIn("candidates", body)

    def test_experiments(self):
        res = self.client.get("/api/discovery/experiments?limit=5")
        self.assertEqual(res.status_code, 200)
        self.assertIn("items", res.json())
        self.assertLessEqual(len(res.json()["items"]), 5)

    def test_leaderboard_validation(self):
        self.assertEqual(self.client.get("/api/discovery/leaderboard?metric=xx").status_code, 400)
        self.assertEqual(self.client.get("/api/discovery/leaderboard?metric=dsr&limit=3").status_code, 200)

    def test_candidates_and_lessons(self):
        res = self.client.get("/api/discovery/candidates?limit=3")
        self.assertEqual(res.status_code, 200)
        self.assertIn("items", res.json())
        res = self.client.get("/api/discovery/lessons?limit=3")
        self.assertEqual(res.status_code, 200)
        self.assertIn("items", res.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
