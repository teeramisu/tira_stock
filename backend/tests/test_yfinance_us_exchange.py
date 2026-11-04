"""Tests for US exchange normalization helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marketdata.sources.yfinance_source import normalize_us_exchange


def test_normalize_us_exchange_maps_provider_codes():
    assert normalize_us_exchange("NYQ") == "NYSE"
    assert normalize_us_exchange("NMS") == "NASDAQ"
    assert normalize_us_exchange("ASE") == "AMEX"
    assert normalize_us_exchange("NasdaqGS") == "NASDAQ"
    assert normalize_us_exchange(None) == "UNKNOWN"
