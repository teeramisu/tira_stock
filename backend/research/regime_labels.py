"""Regime label configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .governance import RegimeWindow

_DEFAULT_CN_LABELS = Path(__file__).resolve().parent / "data" / "regime_labels_cn.json"


@dataclass(frozen=True)
class RegimeLabelsConfig:
    """Manual regime windows for one market."""

    market: str
    regimes: tuple[RegimeWindow, ...]
    version: str = "v1"


def load_regime_labels(path: Path | str | None = None) -> RegimeLabelsConfig:
    """Load regime windows from JSON config."""
    config_path = Path(path) if path is not None else _DEFAULT_CN_LABELS
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    regimes = tuple(
        RegimeWindow(
            label=item["label"],
            start=date.fromisoformat(item["start"]),
            end=date.fromisoformat(item["end"]),
        )
        for item in payload["regimes"]
    )
    return RegimeLabelsConfig(
        market=str(payload["market"]),
        regimes=regimes,
        version=str(payload.get("version", "v1")),
    )
