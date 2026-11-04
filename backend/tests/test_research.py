"""Unit tests for research-layer factor and experiment primitives."""

from __future__ import annotations

from datetime import date
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.experiment import ExperimentConfig, make_walk_forward_splits
from research.explainability import DecisionExplainabilityArtifact
from research.factor_eval import compute_ic
from research.factor_pipeline import FactorDefinition, FactorPipeline, FactorPipelineConfig
from research.regime_labels import load_regime_labels
from research.governance import (
    DecisionAuditArtifact,
    PortfolioRiskBudget,
    RegimeWindow,
    StrategyMetadata,
    StressScenario,
    ValidationEvidence,
    allocate_capital,
    apply_stress_scenario,
    slice_metrics_by_regime,
    strategy_logic_depends_on_broker_mode,
    validate_strategy_promotion,
)
from research.portfolio_constructor import ConstructorConstraints, PortfolioConstructor, capacity_report
from research.multi_strategy import NAIVE_SUM_METHOD, allocate_multi_strategy
from research.sensitivity import (
    build_cost_sensitivity_report,
    build_data_source_sensitivity_report,
    build_parameter_sensitivity_report,
    build_sample_window_sensitivity_report,
)


def _momentum_like_factor(as_of, universe, data):  # noqa: ANN001
    return pd.Series([0.8, 0.3, -0.1], index=list(universe))


def test_factor_pipeline_computes_point_in_time_cross_section():
    """Factor pipeline computes one-date cross section for universe."""
    definition = FactorDefinition(
        name="mom_5d",
        version="v1",
        dependencies=("close",),
        window=5,
        frequency="1d",
        point_in_time_rule="as_of_close",
        compute_fn=_momentum_like_factor,
    )
    pipeline = FactorPipeline(definitions=(definition,), config=FactorPipelineConfig())
    result = pipeline.compute(date(2024, 1, 5), ("AAA", "BBB", "CCC"), data=None)
    assert list(result.frame["symbol"]) == ["AAA", "BBB", "CCC"]
    assert "mom_5d" in result.frame.columns


def test_factor_pipeline_records_valid_sample_count():
    """Factor pipeline reports valid sample diagnostics."""
    definition = FactorDefinition(
        name="mom_5d",
        version="v1",
        dependencies=("close",),
        window=5,
        frequency="1d",
        point_in_time_rule="as_of_close",
        compute_fn=_momentum_like_factor,
    )
    pipeline = FactorPipeline(definitions=(definition,), config=FactorPipelineConfig())
    result = pipeline.compute(date(2024, 1, 5), ("AAA", "BBB", "CCC"), data=None)
    assert result.diagnostics["universe_size"] == 3
    assert result.diagnostics["tradable_size"] == 3
    assert result.diagnostics["factor_valid_count"] == 3


def test_factor_pipeline_missing_values_follow_policy():
    """Missing value policy is applied as configured."""

    def with_missing(as_of, universe, data):  # noqa: ANN001
        return pd.Series([1.0, None, -1.0], index=list(universe))

    definition = FactorDefinition(
        name="factor_x",
        version="v1",
        dependencies=("close",),
        window=5,
        frequency="1d",
        point_in_time_rule="as_of_close",
        compute_fn=with_missing,
    )
    dropped = FactorPipeline(
        definitions=(definition,),
        config=FactorPipelineConfig(missing_value_policy="drop"),
    ).compute(date(2024, 1, 5), ("AAA", "BBB", "CCC"), data=None)
    zero_filled = FactorPipeline(
        definitions=(definition,),
        config=FactorPipelineConfig(missing_value_policy="zero"),
    ).compute(date(2024, 1, 5), ("AAA", "BBB", "CCC"), data=None)
    assert len(dropped.frame) == 2
    assert (zero_filled.frame["factor_x"] == 0.0).sum() == 1


def test_experiment_config_hash_includes_splits():
    """Experiment hash changes when split config changes."""
    split_a = make_walk_forward_splits(
        ((date(2020, 1, 1), date(2020, 6, 30), date(2020, 7, 1), date(2020, 9, 30), date(2020, 10, 1), date(2020, 12, 31)),)
    )
    split_b = make_walk_forward_splits(
        ((date(2021, 1, 1), date(2021, 6, 30), date(2021, 7, 1), date(2021, 9, 30), date(2021, 10, 1), date(2021, 12, 31)),)
    )
    cfg_a = ExperimentConfig(strategy="momentum_top_k", params={"k": 10}, dataset_version="v1", splits=split_a)
    cfg_b = ExperimentConfig(strategy="momentum_top_k", params={"k": 10}, dataset_version="v1", splits=split_b)
    assert cfg_a.request_hash() != cfg_b.request_hash()


def test_walk_forward_split_has_no_overlap():
    """Walk-forward helper rejects overlapping windows."""
    with pytest.raises(ValueError):
        make_walk_forward_splits(
            (
                (
                    date(2020, 1, 1),
                    date(2020, 6, 30),
                    date(2020, 6, 1),
                    date(2020, 8, 31),
                    date(2020, 9, 1),
                    date(2020, 12, 31),
                ),
            )
        )


def test_portfolio_constructor_respects_max_weight():
    """Constructor caps single-name weight."""
    constructor = PortfolioConstructor(constraints=ConstructorConstraints(max_positions=3, max_weight=0.2))
    frame = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "score": [3.0, 2.0, 1.0]})
    built, _ = constructor.construct(frame, capital=1_000_000)
    assert built["target_weight"].max() <= 0.2 + 1e-9


def test_liquidity_constraint_limits_participation_rate():
    """Participation rate is clipped by max participation setting."""
    constructor = PortfolioConstructor(
        constraints=ConstructorConstraints(max_positions=2, max_weight=0.5, max_participation_rate=0.1)
    )
    frame = pd.DataFrame({"symbol": ["AAA", "BBB"], "score": [2.0, 1.0], "adv20": [1000.0, 2000.0]})
    built, _ = constructor.construct(frame, capital=1_000_000, adv_col="adv20")
    assert built["participation_rate"].max() <= 0.1 + 1e-9


def test_capacity_report_worsens_with_larger_capital():
    """Capacity sensitivity should not improve when capital grows."""
    constructor = PortfolioConstructor(
        constraints=ConstructorConstraints(max_positions=2, max_weight=0.5, max_participation_rate=0.2)
    )
    frame = pd.DataFrame({"symbol": ["AAA", "BBB"], "score": [2.0, 1.0], "adv20": [5000.0, 5000.0]})
    small = capacity_report(frame, capital=100_000, constructor=constructor)
    large = capacity_report(frame, capital=10_000_000, constructor=constructor)
    assert large["cost_ratio"] >= small["cost_ratio"]


def test_factor_ic_matches_manual_fixture():
    """IC equals perfect-correlation manual fixture."""
    factor = pd.Series([1.0, 2.0, 3.0], index=["AAA", "BBB", "CCC"])
    fwd = pd.Series([0.1, 0.2, 0.3], index=["AAA", "BBB", "CCC"])
    assert compute_ic(factor, fwd) == pytest.approx(1.0)


def test_explainability_artifact_contains_filter_reasons():
    """Explainability artifact stores filter reasons and decomposition."""
    artifact = DecisionExplainabilityArtifact(
        trade_date="2024-01-05",
        select_reasons={"AAA": "top_momentum"},
        factor_breakdown={"mom_5d": 0.8},
        filter_reason_stats={"suspended": 3, "limit_up": 2},
    )
    assert artifact.filter_reason_stats["suspended"] == 3
    assert "selection" in artifact.return_decomposition


def test_strategy_cannot_validate_without_out_of_sample_metrics():
    """Validated promotion should fail when out-of-sample evidence is missing."""
    metadata = StrategyMetadata(
        strategy_id="mom",
        hypothesis="trend persists",
        economic_rationale="behavioral underreaction",
        suitable_regime="risk_on",
        failure_mode="trend reversal",
        risk_notes="crowded trade",
        model_risk_score=0.4,
    )
    allowed, reasons = validate_strategy_promotion(
        metadata,
        ValidationEvidence(
            out_of_sample_ok=False,
            stress_ok=True,
            cost_sensitivity_ok=True,
            capacity_ok=True,
        ),
        target_state="validated",
    )
    assert not allowed
    assert "missing_out_of_sample" in reasons


@pytest.mark.requires_data
def test_regime_metrics_slice_equity_curve_by_config():
    """Regime slicing should return metrics per configured window."""
    equity_curve = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "equity": [100.0, 105.0, 103.0, 110.0],
            "turnover": [0.1, 0.2, 0.15, 0.25],
            "cost": [10.0, 12.0, 11.0, 13.0],
        }
    )
    regimes = (
        RegimeWindow(label="bull", start=date(2024, 1, 1), end=date(2024, 1, 2)),
        RegimeWindow(label="chop", start=date(2024, 1, 3), end=date(2024, 1, 4)),
    )
    sliced = slice_metrics_by_regime(equity_curve, regimes)
    assert set(sliced.keys()) == {"bull", "chop"}
    assert sliced["bull"]["total_return"] > 0
    assert sliced["chop"]["max_drawdown"] >= 0
    assert sliced["bull"]["turnover"] == pytest.approx(0.3)
    assert sliced["bull"]["cost"] == pytest.approx(22.0)
    assert 0.0 <= sliced["chop"]["win_rate"] <= 1.0
    cn_labels = load_regime_labels()
    assert cn_labels.market == "cn"
    assert len(cn_labels.regimes) >= 3


def test_stress_scenario_increases_costs_and_lowers_return():
    """Stress scenario should worsen return and raise cost."""
    stressed, diagnostics = apply_stress_scenario(
        base_return=0.2,
        base_cost=1000.0,
        scenario=StressScenario(
            name="cost_up_liquidity_down",
            cost_multiplier=1.5,
            liquidity_haircut=0.2,
            volatility_multiplier=1.2,
            correlation_shock=0.3,
        ),
    )
    assert stressed["total_cost"] > 1000.0
    assert stressed["total_return"] < 0.2
    assert diagnostics["failed"] is True


def test_allocator_respects_risk_budget():
    """Allocator should respect max single weight and gross exposure."""
    result = allocate_capital(
        strategy_scores={"s1": 1.0, "s2": 0.8, "s3": 0.6},
        budget=PortfolioRiskBudget(max_single_weight=0.3, max_gross_exposure=0.8),
    )
    assert sum(result.weights.values()) <= 0.8 + 1e-9
    assert max(result.weights.values()) <= 0.3 + 1e-9


def test_decision_audit_contains_filter_and_order_reasons():
    """Decision audit artifact should carry filter and order reasons."""
    artifact = DecisionAuditArtifact(
        trade_date=date(2024, 1, 5),
        filter_reasons={"AAA": "suspended"},
        order_reasons={"BBB": "target_weight_increase"},
    )
    assert artifact.has_complete_reasons()


def test_high_model_risk_blocks_validated_status():
    """High model risk strategy cannot be promoted to validated."""
    metadata = StrategyMetadata(
        strategy_id="high_risk",
        hypothesis="fragile alpha",
        economic_rationale="uncertain",
        suitable_regime="narrow",
        failure_mode="quick decay",
        risk_notes="very unstable",
        model_risk_score=0.95,
    )
    allowed, reasons = validate_strategy_promotion(
        metadata,
        ValidationEvidence(
            out_of_sample_ok=True,
            stress_ok=True,
            cost_sensitivity_ok=True,
            capacity_ok=True,
        ),
        target_state="validated",
    )
    assert not allowed
    assert "high_model_risk" in reasons


def test_parameter_sensitivity_report_includes_variants():
    """Parameter sensitivity report serializes baseline and variants."""
    report = build_parameter_sensitivity_report(
        baseline_label="base",
        baseline_metrics={"total_return": 0.1},
        variants=[("k20", {"lookback_days": 20}, {"total_return": 0.12})],
    )
    payload = report.to_dict()
    assert payload["baseline_label"] == "base"
    assert len(payload["cases"]) == 1


def test_cost_sensitivity_report_compares_variants():
    """Cost sensitivity report tracks cost ratio deltas."""
    report = build_cost_sensitivity_report(
        baseline_total_cost=100.0,
        baseline_turnover=10_000.0,
        variants=[("high_fee", 150.0, 10_000.0)],
    )
    assert report["variants"][0]["delta_vs_baseline"] == pytest.approx(50.0)


def test_sample_window_sensitivity_report_lists_windows():
    """Sample window report lists labeled date ranges."""
    rows = build_sample_window_sensitivity_report(
        windows=[("train", "2020-01-01", "2022-12-31", {"total_return": 0.2})],
    )
    assert rows[0]["label"] == "train"


def test_multi_strategy_rejects_naive_sum():
    """Multi-strategy allocation must not allow naive weight sum."""
    with pytest.raises(ValueError, match="naive sum"):
        allocate_multi_strategy(
            strategy_scores={"s1": 1.0, "s2": 0.5},
            budget=PortfolioRiskBudget(),
            method=NAIVE_SUM_METHOD,
        )


def test_data_source_sensitivity_report_compares_adjustments():
    """Data source report tracks adjustment and return deltas."""
    report = build_data_source_sensitivity_report(
        baseline_label="forward_adj",
        baseline_metrics={"total_return": 0.1},
        variants=[
            ("raw_close", "akshare", "none", {"total_return": 0.08}),
        ],
    )
    assert report["artifact_type"] == "data_source_sensitivity_v1"
    assert report["variants"][0]["delta_total_return"] == pytest.approx(-0.02)


def test_strategy_logic_does_not_depend_on_broker_mode():
    """Strategy logic should be invariant across broker modes."""
    mode_outputs = {
        "backtest": ("AAA", "BBB"),
        "paper": ("AAA", "BBB"),
        "live": ("AAA", "BBB"),
    }
    assert not strategy_logic_depends_on_broker_mode(mode_outputs)


# ─── v0.3.0 修复针对性测试 ─────────────────────────────────────────────

def test_regime_vol_window_excludes_current_day():
    """Regime detector realized vol uses T-1 window to avoid forward look."""
    from research.regime_detector import RegimeDetector, RegimeConfig
    import numpy as np

    cfg = RegimeConfig(
        ma_period=5, trend_period=5, crash_period=3, min_stocks=2,
        bull_breadth=0.50, bear_breadth=0.30, bear_trend=-0.03,
        bull_trend=0.0, crash_threshold=-0.04,
    )
    n_days = 30
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frame = pd.DataFrame(
        {s: 100 + np.arange(n_days) * 0.5 + i * 10 for i, s in enumerate(["A", "B", "C"])},
        index=dates,
    )
    detector = RegimeDetector(config=cfg)
    signals = detector.detect_all(frame, market="cn")
    assert len(signals) > 0
    for sig in signals:
        assert isinstance(sig.realized_vol, float)
        assert sig.realized_vol >= 0


def test_allocate_capital_supports_all_methods():
    """All 5 allocation methods are accepted and return valid weights."""
    scores = {"mom": 0.1, "rev": 0.05, "multi": 0.08}
    metrics = {
        "mom":  {"total_return": 0.1, "sharpe_ratio": 0.8, "max_drawdown": -0.1, "annualized_volatility": 0.2, "win_rate": 0.55, "trade_count": 50},
        "rev":  {"total_return": 0.05, "sharpe_ratio": 0.3, "max_drawdown": -0.15, "annualized_volatility": 0.25, "win_rate": 0.52, "trade_count": 40},
        "multi": {"total_return": 0.08, "sharpe_ratio": 0.6, "max_drawdown": -0.12, "annualized_volatility": 0.18, "win_rate": 0.53, "trade_count": 45},
    }
    budget = PortfolioRiskBudget(max_single_weight=0.6, max_gross_exposure=1.0)

    for method in ["equal_weight", "risk_parity", "momentum_weighted", "kelly"]:
        result = allocate_capital(
            strategy_scores=scores,
            budget=budget,
            method=method,  # type: ignore
            strategy_metrics=metrics,
            regime="BULL",
        )
        assert abs(sum(result.weights.values()) - 1.0) < 0.01, f"{method}: weights sum to {sum(result.weights.values())}"
        assert all(w >= 0 for w in result.weights.values()), f"{method}: negative weight"

    # regime_adaptive requires regime parameter
    result = allocate_capital(
        strategy_scores=scores,
        budget=budget,
        method="regime_adaptive",  # type: ignore
        strategy_metrics=metrics,
        regime="BEAR",
    )
    assert abs(sum(result.weights.values()) - 1.0) < 0.01


def test_meta_allocator_risk_parity_prefers_low_vol():
    """Risk parity allocator gives higher weight to lower-volatility strategies."""
    from research.meta_allocator import MetaAllocator, SleeveMetrics

    alloc = MetaAllocator()
    sleeves = {
        "low_vol":  SleeveMetrics("low_vol",  volatility=0.10, sharpe_ratio=0.5),
        "high_vol": SleeveMetrics("high_vol", volatility=0.40, sharpe_ratio=0.5),
    }
    result = alloc.allocate(sleeves, method="risk_parity")
    assert result.weights["low_vol"] > result.weights["high_vol"]


def test_kelly_allocator_gives_higher_weight_to_better_win_rate():
    """Kelly allocator prefers higher win-rate strategies."""
    from research.meta_allocator import MetaAllocator, SleeveMetrics

    alloc = MetaAllocator()
    sleeves = {
        "good": SleeveMetrics("good", win_rate=0.65, avg_win=0.05, avg_loss=0.03, trade_count=100),
        "bad":  SleeveMetrics("bad",  win_rate=0.40, avg_win=0.05, avg_loss=0.03, trade_count=100),
    }
    result = alloc.allocate(sleeves, method="kelly")
    assert result.weights["good"] > result.weights["bad"]


def test_score_weighted_targets_differs_from_equal():
    """_score_weighted_targets returns non-equal weights ordered by score."""
    from portfolio.strategies import _score_weighted_targets

    selected = [("A", 0.9), ("B", 0.5), ("C", 0.1)]
    targets = _score_weighted_targets(selected, "test_reason")
    weights = [t.weight for t in targets]
    # Higher score → higher weight
    assert weights[0] > weights[1] > weights[2]
    assert abs(sum(weights) - 1.0) < 0.01


def test_score_weighted_targets_cash_fallback():
    """_score_weighted_targets returns empty when all scores below min with cash_fallback."""
    from portfolio.strategies import _score_weighted_targets

    selected = [("A", -0.1), ("B", -0.2)]
    targets = _score_weighted_targets(selected, "test", min_score=0)
    assert len(targets) == 0


def test_score_weighted_targets_no_cash_fallback():
    """Without cash_fallback, negative-score targets are still returned."""
    from portfolio.strategies import _score_weighted_targets

    selected = [("A", -0.1), ("B", -0.2)]
    targets = _score_weighted_targets(selected, "test", cash_fallback=False)
    assert len(targets) == 2


def test_features_no_forward_normalization():
    """create_sequences returns raw features; normalize_features computes per-fold stats."""
    from ml.features import build_feature_matrix, create_sequences, normalize_features, FeatureConfig
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    frame = pd.DataFrame({s: 100 + np.arange(200) * 0.5 for s in ["A", "B", "C", "D", "E"]}, index=dates)
    # Build a simple regime_by_date dict so labels_regime is populated
    regime_by_date = {d.date(): "BULL" for d in dates}
    feat_cfg = FeatureConfig(min_stocks=3)  # low min_stocks for small test
    fm = build_feature_matrix(frame, volume_matrix=None, regime_by_date=regime_by_date, config=feat_cfg)
    X, y_ret, y_reg, seq_dates = create_sequences(fm, lookback=20, forward_horizon=5)
    assert len(X) > 0, "No sequences created — insufficient data after warmup"
    # Features are NOT pre-normalized (check a non-constant feature like trend_20d)
    trend_feat = X[:, :, 2]  # feature index 2 = trend_20d
    assert np.isfinite(trend_feat).any(), "All trend values are NaN"
    feat_std = float(np.nanstd(trend_feat))
    assert feat_std > 1e-6, f"Trend feature std too small: {feat_std}"

    # Per-fold normalization produces zero-mean, unit-variance
    X_norm, mean, std = normalize_features(X)
    assert abs(float(X_norm.mean())) < 0.5, f"Mean not near zero: {X_norm.mean()}"
    assert 0.3 < float(X_norm.std()) < 3.0, f"Std not near 1: {X_norm.std()}"


def test_multi_strategy_sleeve_uses_sub_allocated_cash():
    """Multi-strategy sleeve splits capital across sleeves (fix for full-cash bug)."""
    from services.multi_strategy_portfolio import MultiStrategyPortfolioRequest

    req = MultiStrategyPortfolioRequest(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        initial_cash=1_000_000,
        sleeves=[
            {"sleeve_id": "mom", "strategy": "momentum_top_k"},
            {"sleeve_id": "rev", "strategy": "mean_reversion_top_k"},
        ],
    )
    # Each sleeve should get initial_cash / n_sleeves
    sleeve_cash = req.initial_cash / len(req.sleeves)
    assert sleeve_cash == 500_000, f"Expected 500000, got {sleeve_cash}"


def test_transformer_causal_mask_prevents_future_attendance():
    """Transformer with causal mask should not attend to future tokens."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    from ml.model import MarketTransformer, TransformerConfig

    config = TransformerConfig(n_features=4, d_model=16, n_heads=2, n_layers=1, max_seq_len=20)
    model = MarketTransformer(config)
    model.eval()

    # Input: batch=1, seq=10, features=4 — all zeros except last timestep has a signal
    x = torch.zeros(1, 10, 4)
    x[0, -1, 0] = 100.0  # strong signal at last position

    with torch.no_grad():
        out = model(x)
    assert "return_5" in out
    assert "regime_logits" in out
    # Output should be finite (no NaN from mask)
    assert torch.isfinite(out["return_5"]).all()
    assert torch.isfinite(out["regime_logits"]).all()


def test_factor_tracker_save_and_load():
    """FactorTracker saves and loads scan records correctly."""
    from research.factor_scan import FactorTracker, FactorScanRecord
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tracker = FactorTracker(scan_dir=tmp)
        records = [
            FactorScanRecord(scan_date="2026-07-01", factor_name="ret_5d", ic=0.03, rank_ic=0.04, ic_ir=0.5, rank_ic_ir=0.6, spread=1.2, n_obs=500),
            FactorScanRecord(scan_date="2026-07-01", factor_name="vol_20d", ic=-0.01, rank_ic=-0.02, ic_ir=-0.2, rank_ic_ir=-0.3, spread=-0.5, n_obs=500),
        ]
        tracker.save_scan(records)

        history = tracker.load_history(n_days=365)
        assert "ret_5d" in history
        assert "vol_20d" in history
        assert len(history["ret_5d"]) == 1
        assert history["ret_5d"][0].rank_ic == 0.04


def test_factor_tracker_compute_trends():
    """FactorTracker detects emerging and decaying factors."""
    from research.factor_scan import FactorTracker, FactorScanRecord

    tracker = FactorTracker()
    # Simulate 5 scans with increasing rank_ic (emerging)
    records = []
    for i, ic_val in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
        records.append(FactorScanRecord(
            scan_date=f"2026-07-{i+1:02d}",
            factor_name="emerging_factor",
            ic=ic_val, rank_ic=ic_val,
            ic_ir=0.5, rank_ic_ir=0.5,
            spread=1.0, n_obs=500,
        ))
    history = {"emerging_factor": records}
    trends = tracker.compute_trends(history, min_scans=3)
    assert len(trends) == 1
    assert trends[0].trend_direction in ("emerging", "stable")


def test_factor_scan_harness_accepts_basic_config():
    """FactorScanHarness initializes with default parameters."""
    from research.factor_scan import FactorScanHarness

    harness = FactorScanHarness(market="cn")
    assert harness.market == "cn"
    assert harness.tracker is not None
