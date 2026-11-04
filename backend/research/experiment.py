"""Experiment and split configuration for research runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test date windows."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def has_overlap(self) -> bool:
        windows = [
            (self.train_start, self.train_end),
            (self.validation_start, self.validation_end),
            (self.test_start, self.test_end),
        ]
        for idx, (a_start, a_end) in enumerate(windows):
            for b_start, b_end in windows[idx + 1 :]:
                if max(a_start, b_start) <= min(a_end, b_end):
                    return True
        return False


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment identity with strategy params and split definitions."""

    strategy: str
    params: dict[str, object]
    dataset_version: str
    splits: tuple[SplitConfig, ...]

    def request_hash(self) -> str:
        payload = {
            "strategy": self.strategy,
            "params": self.params,
            "dataset_version": self.dataset_version,
            "splits": [
                {
                    "train": [s.train_start.isoformat(), s.train_end.isoformat()],
                    "validation": [s.validation_start.isoformat(), s.validation_end.isoformat()],
                    "test": [s.test_start.isoformat(), s.test_end.isoformat()],
                }
                for s in self.splits
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def make_walk_forward_splits(
    windows: tuple[tuple[date, date, date, date, date, date], ...],
) -> tuple[SplitConfig, ...]:
    """Build walk-forward splits and enforce non-overlap within each split."""
    splits = tuple(
        SplitConfig(
            train_start=w[0],
            train_end=w[1],
            validation_start=w[2],
            validation_end=w[3],
            test_start=w[4],
            test_end=w[5],
        )
        for w in windows
    )
    for split in splits:
        if split.has_overlap():
            raise ValueError("walk-forward split has overlap")
    return splits
