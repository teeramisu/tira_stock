"""Tests for signal combiner that aggregates agent signals."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.models import SignalRecord
from signals.combiner import combine_signals


def _signal(agent: str, score: float, confidence: float) -> SignalRecord:
    return SignalRecord(agent_name=agent, score=score, confidence=confidence, reasoning="")


class TestConfidenceWeighted:
    def test_equal_confidence_gives_average(self):
        sigs = [_signal("a", 80, 1.0), _signal("b", 60, 1.0)]
        score, conf, reason = combine_signals(sigs, method="confidence_weighted")
        assert score == 70.0
        assert conf == 1.0

    def test_weighted_average_with_different_confidences(self):
        sigs = [_signal("a", 100, 1.0), _signal("b", 50, 0.0)]
        score, conf, reason = combine_signals(sigs, method="confidence_weighted", min_confidence=0.0)
        assert score == 100.0
        assert conf == 0.5

    def test_single_signal_returns_itself(self):
        sigs = [_signal("a", 75, 0.8)]
        score, conf, reason = combine_signals(sigs)
        assert score == 75.0
        assert conf == 0.8

    def test_reason_contains_agent_names_and_scores(self):
        sigs = [_signal("bull", 85, 0.9), _signal("bear", 30, 0.7)]
        _, _, reason = combine_signals(sigs)
        assert "bull=85" in reason
        assert "bear=30" in reason

    def test_all_below_min_confidence_returns_fallback(self):
        sigs = [_signal("a", 80, 0.2), _signal("b", 60, 0.1)]
        score, conf, reason = combine_signals(sigs, min_confidence=0.3)
        assert score == 50.0
        assert conf == 0.0
        assert "no valid signals" in reason

    def test_zero_total_confidence_returns_fallback(self):
        sigs = [_signal("a", 80, 0.0), _signal("b", 60, 0.0)]
        score, conf, reason = combine_signals(sigs, min_confidence=0.0)
        assert score == 50.0
        assert conf == 0.0
        assert "zero total confidence" in reason

    def test_none_score_handled(self):
        sigs = [SignalRecord(agent_name="a", score=None, confidence=0.8, reasoning="")]
        score, conf, reason = combine_signals(sigs)
        assert score == 50.0
        assert conf == 0.0


class TestEqualMethod:
    def test_equal_weights(self):
        sigs = [_signal("a", 100, 0.5), _signal("b", 0, 0.9)]
        score, conf, reason = combine_signals(sigs, method="equal")
        assert score == 50.0
        assert conf == 0.7

    def test_gives_average_confidence(self):
        sigs = [_signal("a", 80, 0.6), _signal("b", 40, 0.8)]
        _, conf, _ = combine_signals(sigs, method="equal")
        assert conf == 0.7

    def test_empty_after_filter_falls_back(self):
        sigs = [_signal("a", 80, 0.1)]
        score, conf, reason = combine_signals(sigs, method="equal", min_confidence=0.5)
        assert score == 50.0
        assert conf == 0.0


class TestMaxMethod:
    def test_picks_highest_score(self):
        sigs = [_signal("a", 30, 0.5), _signal("b", 90, 0.8)]
        score, conf, reason = combine_signals(sigs, method="max")
        assert score == 90.0
        assert conf == 0.8
        assert "b" in reason

    def test_first_wins_on_tie(self):
        sigs = [_signal("a", 80, 0.5), _signal("b", 80, 0.9)]
        score, conf, reason = combine_signals(sigs, method="max")
        assert score == 80.0
        assert "a" in reason

    def test_empty_after_filter_falls_back(self):
        sigs = [_signal("a", 80, 0.1)]
        score, conf, reason = combine_signals(sigs, method="max", min_confidence=0.5)
        assert score == 50.0
        assert conf == 0.0


class TestEdgeCases:
    def test_empty_signals_list(self):
        score, conf, reason = combine_signals([])
        assert score == 50.0
        assert conf == 0.0

    def test_all_none_scores_filtered_out(self):
        sigs = [
            SignalRecord(agent_name="a", score=None, confidence=0.8, reasoning=""),
            SignalRecord(agent_name="b", score=None, confidence=0.7, reasoning=""),
        ]
        score, conf, _ = combine_signals(sigs)
        assert score == 50.0

    def test_large_number_of_signals(self):
        sigs = [_signal(f"a{i}", float(i * 10 % 100), 0.5) for i in range(100)]
        score, conf, _ = combine_signals(sigs)
        assert 0 <= score <= 100
        assert conf == 0.5

    def test_invalid_method_fallback_uses_confidence_weighted(self):
        sigs = [_signal("a", 80, 1.0)]
        score, conf, _ = combine_signals(sigs, method="unknown")
        assert score == 80.0
        assert conf == 1.0
