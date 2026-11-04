"""Sensitivity report helpers for research and validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SensitivityCase:
    """One sensitivity variant with label and metric snapshot."""

    label: str
    params: dict[str, Any]
    metrics: dict[str, float]


@dataclass
class SensitivityReport:
    """Aggregated sensitivity output for API artifacts."""

    baseline_label: str
    baseline_metrics: dict[str, float]
    cases: list[SensitivityCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize report for JSON responses."""
        return {
            "baseline_label": self.baseline_label,
            "baseline_metrics": self.baseline_metrics,
            "cases": [
                {"label": c.label, "params": c.params, "metrics": c.metrics} for c in self.cases
            ],
        }


def build_parameter_sensitivity_report(
    *,
    baseline_label: str,
    baseline_metrics: dict[str, float],
    variants: list[tuple[str, dict[str, Any], dict[str, float]]],
) -> SensitivityReport:
    """Compare metrics across parameter variants."""
    cases = [
        SensitivityCase(label=label, params=params, metrics=metrics)
        for label, params, metrics in variants
    ]
    return SensitivityReport(
        baseline_label=baseline_label,
        baseline_metrics=baseline_metrics,
        cases=cases,
    )


def build_cost_sensitivity_report(
    *,
    baseline_total_cost: float,
    baseline_turnover: float,
    variants: list[tuple[str, float, float]],
) -> dict[str, object]:
    """Compare total cost and cost ratio under different cost assumptions."""
    baseline_ratio = baseline_total_cost / baseline_turnover if baseline_turnover > 0 else 0.0
    rows = [
        {
            "label": label,
            "total_cost": total_cost,
            "turnover": turnover,
            "cost_to_turnover_ratio": total_cost / turnover if turnover > 0 else 0.0,
            "delta_vs_baseline": total_cost - baseline_total_cost,
        }
        for label, total_cost, turnover in variants
    ]
    return {
        "baseline_total_cost": baseline_total_cost,
        "baseline_cost_to_turnover_ratio": baseline_ratio,
        "variants": rows,
    }


def build_sample_window_sensitivity_report(
    *,
    windows: list[tuple[str, str, str, dict[str, float]]],
) -> list[dict[str, object]]:
    """Report metrics across sample date windows (start, end labels)."""
    return [
        {
            "label": label,
            "start": start,
            "end": end,
            "metrics": metrics,
        }
        for label, start, end, metrics in windows
    ]


def build_data_source_sensitivity_report(
    *,
    baseline_label: str,
    baseline_metrics: dict[str, float],
    variants: list[tuple[str, str, str, dict[str, float]]],
) -> dict[str, object]:
    """Compare metrics across data source or adjustment conventions.

    Each variant is (label, data_source, adjustment, metrics), e.g.
    (\"akshare\", \"akshare\", \"forward_adjusted\", {...}).
    """
    rows = [
        {
            "label": label,
            "data_source": data_source,
            "adjustment": adjustment,
            "metrics": metrics,
            "delta_total_return": metrics.get("total_return", 0.0)
            - baseline_metrics.get("total_return", 0.0),
        }
        for label, data_source, adjustment, metrics in variants
    ]
    return {
        "artifact_type": "data_source_sensitivity_v1",
        "baseline_label": baseline_label,
        "baseline_metrics": baseline_metrics,
        "variants": rows,
    }
