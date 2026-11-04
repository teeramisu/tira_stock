"""SignalHub — aggregates signals and combines them with confidence weighting."""

from __future__ import annotations


from memory.models import SignalRecord


def combine_signals(
    signals: list[SignalRecord],
    *,
    method: str = "confidence_weighted",
    min_confidence: float = 0.3,
) -> tuple[float, float, str]:
    """Combine multiple signal scores into one.

    Args:
        signals: Signal records to combine.
        method: Aggregation method (confidence_weighted / equal / max).
        min_confidence: Filter out signals with lower confidence.

    Returns:
        (combined_score, combined_confidence, summary_reason)
    """
    valid = [s for s in signals if (s.confidence or 0) >= min_confidence and s.score is not None]
    if not valid:
        return (50.0, 0.0, "no valid signals with sufficient confidence")

    if method == "equal":
        scores = [s.score for s in valid if s.score is not None]
        confs = [s.confidence for s in valid]
        return (
            round(sum(scores) / len(scores), 1) if scores else 50.0,
            round(sum(confs) / len(confs), 2) if confs else 0.0,
            "equal-weight ensemble",
        )

    if method == "max":
        best = max(valid, key=lambda s: s.score or 0)
        return (round(best.score or 50, 1), best.confidence or 0.0, f"max signal from {best.agent_name}")

    # confidence_weighted
    total_conf = sum(s.confidence for s in valid if s.score is not None)
    if total_conf == 0:
        return (50.0, 0.0, "zero total confidence")
    weighted = sum((s.score or 50) * s.confidence for s in valid) / total_conf
    avg_conf = sum(s.confidence for s in valid) / len(valid)
    reasons = "; ".join(f"{s.agent_name}={s.score:.0f}" for s in valid)
    return (round(weighted, 1), round(avg_conf, 2), reasons)
