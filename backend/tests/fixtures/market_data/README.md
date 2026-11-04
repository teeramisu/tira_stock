# Portfolio Engine Fixture Data

This directory stores deterministic fixture data for portfolio backtest tests.

- `daily_bars_sample.csv`: tiny 3-symbol x 5-day daily sample.
- `instruments_sample.csv`: tiny universe metadata sample.

Tests may either seed DuckDB directly from Python dataclasses or read these files
for golden/manual checks.

