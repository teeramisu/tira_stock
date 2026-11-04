"""Factor Backtest Engine — computes factors, evaluates IC, rank IC, quantile returns."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.catalog import FACTOR_CATALOG, FactorSpec, _ema, _roc, _sma, _std


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "market" / "market_data.duckdb"


class FactorBacktestEngine:
    """Factor evaluation engine using DuckDB data."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        market: str = "cn",
        sample_size: int = 500,
        min_bars: int = 250,
        recent_days: int = 500,
        existing_conn=None,
    ):
        self.db_path = str(db_path or DB_PATH)
        self.market = market
        self.sample_size = sample_size
        self.min_bars = min_bars
        self.recent_days = recent_days
        self._existing_conn = existing_conn
        self.data: dict[str, pd.DataFrame] = {}
        self.factor_values: dict[str, pd.DataFrame] = {}
        self.results: list[dict] = []

    # ---- Load ----

    def load_data(self) -> pd.DataFrame:
        """Load daily OHLCV data from DuckDB into a wide-format dict of DataFrames."""
        if self._existing_conn is not None:
            conn = self._existing_conn
        else:
            conn = duckdb.connect(self.db_path, read_only=True)

        # Get top symbols by bar count in recent period
        symbols = conn.execute(f"""
            SELECT symbol FROM daily_bars
            WHERE market = '{self.market}'
            GROUP BY symbol
            HAVING count(*) >= {self.min_bars}
            ORDER BY count(*) DESC
            LIMIT {self.sample_size}
        """).fetchall()
        symbols = [s[0] for s in symbols]
        print(f"  Loaded {len(symbols)} symbols for {self.market}")

        # Load data for all symbols
        placeholders = ",".join(["?"] * len(symbols))
        rows = conn.execute(
            f"SELECT symbol, trade_date, open, high, low, close, volume "
            f"FROM daily_bars WHERE market = '{self.market}' AND symbol IN ({placeholders}) "
            f"ORDER BY trade_date",
            symbols,
        ).fetchall()
        conn.close()

        df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Filter to recent period
        if self.recent_days:
            max_date = df["date"].max()
            min_date = max_date - pd.Timedelta(days=self.recent_days * 2)
            df = df[df["date"] >= min_date]

        print(f"  Total rows: {len(df)}, date range: {df['date'].min()} ~ {df['date'].max()}")
        self.data["raw"] = df
        return df

    # ---- Compute factors ----

    def compute_all_factors(self) -> dict[str, pd.DataFrame]:
        """Compute all catalog factors for the loaded data."""
        df = self.data.get("raw")
        if df is None:
            raise RuntimeError("Call load_data() first")

        results: dict[str, pd.DataFrame] = {}
        total = len(FACTOR_CATALOG)
        print(f"\n  Computing {total} factors...")

        for i, spec in enumerate(FACTOR_CATALOG):
            if i % 10 == 0:
                print(f"    [{i}/{total}] ...")
            try:
                factor_df = self._compute_factor(df, spec)
                if factor_df is not None and len(factor_df) > 0:
                    results[spec.name] = factor_df
            except Exception as e:
                print(f"    Factor {spec.name} failed: {e}")

        self.factor_values = results
        print(f"    Computed {len(results)}/{total} factors successfully")
        return results

    def _compute_factor(self, df: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame | None:
        """Compute a single factor value per symbol per date."""
        n = spec.param_lookback
        extra = spec.param_extra

        result = df[["symbol", "date", "close"]].copy()

        if spec.name in ("ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_30d",
                          "ret_60d", "ret_120d", "ret_250d"):
            result["value"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(n))

        elif spec.name.startswith("rev_1d"):
            result["value"] = df.groupby("symbol")["close"].transform(lambda x: -x.pct_change(1))

        elif spec.name == "rev_intraday":
            result["value"] = df.groupby("symbol").apply(
                lambda g: -(g["close"] - g["open"]) / g["open"].replace(0, np.nan)
            ).reset_index(level=0, drop=True)

        elif spec.name == "gap":
            result["prev_close"] = df.groupby("symbol")["close"].shift(1)
            result["value"] = (df["open"] - result["prev_close"]) / result["prev_close"].replace(0, np.nan)

        elif spec.name.startswith("vol_") and "ratio" not in spec.name:
            ln = int(spec.name.split("_")[1].replace("d", ""))
            result["ret"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(1))
            result["value"] = result.groupby("symbol")["ret"].transform(lambda x: x.rolling(ln).std())

        elif spec.name in ("vol_ratio_5d", "vol_ratio_20d"):
            slow = extra.get("denom_lookback", 20)
            fast = n
            result["vol_fast"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(fast).mean())
            result["vol_slow"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(slow).mean())
            result["value"] = result["vol_fast"] / result["vol_slow"].replace(0, np.nan)

        elif spec.name == "turnover_5d":
            result["vol_5"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(5).mean())
            result["vol_20"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20).mean())
            result["value"] = result["vol_5"] / result["vol_20"].replace(0, np.nan)

        elif spec.name.startswith("ma_cross"):
            slow = extra.get("slow", 20)
            result["ma_fast"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(n).mean())
            result["ma_slow"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(slow).mean())
            result["value"] = result["ma_fast"] / result["ma_slow"].replace(0, np.nan) - 1

        elif spec.name == "mr_20d":
            result["ma"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(n).mean())
            result["value"] = -(df["close"] - result["ma"]) / result["ma"].replace(0, np.nan)

        elif spec.name in ("macd", "macd_div"):
            slow, sig = extra.get("slow", 26), extra.get("signal", 9)
            result["ema_fast"] = df.groupby("symbol")["close"].transform(lambda x: _ema(x, n))
            result["ema_slow"] = df.groupby("symbol")["close"].transform(lambda x: _ema(x, slow))
            result["macd_line"] = result["ema_fast"] - result["ema_slow"]
            result["macd_signal"] = result.groupby("symbol")["macd_line"].transform(lambda x: _ema(x, sig))
            if spec.name == "macd":
                result["value"] = result["macd_line"]
            else:
                result["value"] = result["macd_line"] - result["macd_signal"]

        elif spec.name.startswith("rsi"):
            result["delta"] = df.groupby("symbol")["close"].transform(lambda x: x.diff())
            result["gain"] = result.groupby("symbol")["delta"].transform(
                lambda x: x.clip(lower=0).rolling(n).mean())
            result["loss"] = result.groupby("symbol")["delta"].transform(
                lambda x: (-x.clip(upper=0)).rolling(n).mean())
            result["rs"] = result["gain"] / result["loss"].replace(0, np.nan)
            result["value"] = 100 - (100 / (1 + result["rs"]))

        elif spec.name == "bb_position":
            std_n = extra.get("num_std", 2)
            result["ma"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(n).mean())
            result["std"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(n).std())
            result["upper"] = result["ma"] + std_n * result["std"]
            result["lower"] = result["ma"] - std_n * result["std"]
            denom = result["upper"] - result["lower"]
            result["value"] = (df["close"] - result["lower"]) / denom.replace(0, np.nan)

        elif spec.name in ("vol_ratio_20_60",):
            fast_n, slow_n = n, extra.get("denom_lookback", 60)
            result["ret"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(1))
            result["vol_fast"] = result.groupby("symbol")["ret"].transform(lambda x: x.rolling(fast_n).std())
            result["vol_slow"] = result.groupby("symbol")["ret"].transform(lambda x: x.rolling(slow_n).std())
            result["value"] = result["vol_fast"] / result["vol_slow"].replace(0, np.nan)

        elif spec.name == "ret_stability":
            result["ret"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(1))
            result["ret_sum"] = result.groupby("symbol")["ret"].transform(lambda x: x.rolling(n).sum())
            result["ret_std"] = result.groupby("symbol")["ret"].transform(lambda x: x.rolling(n).std())
            result["value"] = result["ret_sum"] / result["ret_std"].replace(0, np.nan)

        elif spec.name == "ret_consistency":
            result["ret"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(1))
            result["value"] = result.groupby("symbol")["ret"].transform(
                lambda x: (x > 0).rolling(n).mean())

        elif spec.name.startswith("max_dd"):
            result["peak"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(n).max())
            result["value"] = -(df["close"] - result["peak"]) / result["peak"].replace(0, np.nan)

        elif spec.name == "vp_corr_20d":
            result["ret"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(1))
            result["vol_chg"] = df.groupby("symbol")["volume"].transform(lambda x: x.pct_change(1))
            result["value"] = result.groupby("symbol").apply(
                lambda g: g["ret"].rolling(n).corr(g["vol_chg"])
            ).reset_index(level=0, drop=True)

        elif spec.name == "skew_20d":
            result["ret"] = df.groupby("symbol")["close"].transform(lambda x: x.pct_change(1))
            result["value"] = result.groupby("symbol")["ret"].transform(
                lambda x: x.rolling(n).skew())

        elif spec.name == "cn_limit_up":
            result["value"] = df.groupby("symbol")["close"].transform(
                lambda x: (x.pct_change(1) > 0.095).astype(float))

        elif spec.name == "cn_st_filter":
            result["value"] = df["close"].apply(lambda x: 0.0)

        else:
            return None

        result["value"] = result["value"].replace([np.inf, -np.inf], np.nan)
        return result[["symbol", "date", "value"]].dropna(subset=["value"])

    # ---- Factor correlation analysis ----

    def factor_correlation_matrix(
        self, factor_names: list[str] | None = None, top_n: int = 20
    ) -> dict:
        """Compute Spearman correlation matrix between factors.

        Args:
            factor_names: Specific factors to correlate (None = use top_n by |rank_ic|).
            top_n: Number of top factors to use if factor_names is None.

        Returns:
            Dict with correlation matrix, high-correlation warnings, clusters.
        """
        df = self.data.get("raw")
        if df is None:
            raise RuntimeError("Call load_data() first")

        if not self.factor_values:
            raise RuntimeError("No computed factors. Call compute_all_factors() first.")

        # Determine which factors to use
        if factor_names is None:
            if self.results:
                sorted_factors = sorted(self.results, key=lambda x: -abs(x.get("rank_ic", 0)))
                factor_names = [r["name"] for r in sorted_factors[:top_n]]
            else:
                factor_names = list(self.factor_values.keys())[:top_n]

        # Build a date × factor DataFrame of cross-sectional mean factor values
        import pandas as pd
        all_dates = set()
        for fv in self.factor_values.values():
            all_dates.update(fv["date"].unique())
        dates = sorted(all_dates)

        factor_ts: dict[str, list[float]] = {name: [] for name in factor_names}
        valid_dates = []

        for d in dates:
            day_values = {}
            has_any = False
            for name in factor_names:
                fv = self.factor_values.get(name)
                if fv is None:
                    continue
                day = fv[fv["date"] == d]
                if day.empty:
                    continue
                vals = day["value"].dropna()
                if len(vals) > 10:
                    day_values[name] = float(vals.mean())
                    has_any = True
            if has_any:
                valid_dates.append(d)
                for name in factor_names:
                    factor_ts[name].append(day_values.get(name, float("nan")))

        if len(valid_dates) < 10:
            return {"error": "Too few dates for correlation analysis"}

        corr_df = pd.DataFrame(factor_ts, index=valid_dates).fillna(method="ffill").fillna(method="bfill")
        # Drop factors with too few valid observations
        corr_df = corr_df.dropna(axis=1, how="all")
        if corr_df.shape[1] < 2:
            # Fallback: try with fewer factors or different top_n
            return {"error": "Need at least 2 factors with valid data"}
        if corr_df.shape[1] < 2:
            return {"error": "Need at least 2 factors with valid data"}

        spearman = corr_df.corr(method="spearman")
        pearson = corr_df.corr(method="pearson")

        # High correlation warnings
        high_pairs = []
        for i in range(len(spearman.columns)):
            for j in range(i + 1, len(spearman.columns)):
                col_i, col_j = spearman.columns[i], spearman.columns[j]
                r = spearman.iloc[i, j]
                if abs(r) > 0.7:
                    high_pairs.append({
                        "factor_a": col_i, "factor_b": col_j,
                        "spearman_r": round(r, 4),
                        "warning": "⚠️ 高相关 (>0.7)，建议去重" if abs(r) > 0.8 else "📌 中等相关 (>0.7)",
                    })

        # Factor clusters (greedy: group highly correlated factors)
        clustered = []
        assigned = set()
        for name in spearman.columns:
            if name in assigned:
                continue
            cluster = [name]
            assigned.add(name)
            for other in spearman.columns:
                if other in assigned:
                    continue
                if abs(spearman.loc[name, other]) > 0.7:
                    cluster.append(other)
                    assigned.add(other)
            if len(cluster) >= 2:
                clustered.append(cluster)

        return {
            "n_factors": len(spearman.columns),
            "n_dates": len(valid_dates),
            "spearman_matrix": {
                "columns": list(spearman.columns),
                "values": spearman.values.tolist(),
            },
            "high_correlation_pairs": high_pairs,
            "clusters": clustered,
            "single_factors": [c for c in spearman.columns if c not in assigned],
        }

    # ---- Single factor evaluation (for AI Agent integration) ----

    def compute_factor(self, spec: FactorSpec) -> pd.DataFrame | None:
        """Compute a single factor by spec. Returns (symbol, date, value) DataFrame."""
        df = self.data.get("raw")
        if df is None:
            raise RuntimeError("Call load_data() first")
        return self._compute_factor(df, spec)

    def evaluate_factor(
        self, factor_name: str, forward_period: int = 5
    ) -> dict:
        """Evaluate a single factor by name using loaded factor_values."""
        if factor_name not in self.factor_values:
            raise ValueError(f"Factor '{factor_name}' not computed. Call compute_factor() first.")
        df = self.data.get("raw")
        if df is None:
            raise RuntimeError("Call load_data() first")
        factor_df = self.factor_values[factor_name]
        result = self._evaluate_factor(df, factor_df, forward_period)
        result["name"] = factor_name
        return result

    # ---- Evaluate ----

    def evaluate_all(self, forward_period: int = 5) -> list[dict]:
        """Evaluate all computed factors using IC analysis."""
        df = self.data["raw"]
        if df is None or not self.factor_values:
            raise RuntimeError("Load data and compute factors first")

        print(f"\n  Evaluating {len(self.factor_values)} factors (forward={forward_period}d)...")
        results = []

        for name, factor_df in self.factor_values.items():
            try:
                metrics = self._evaluate_factor(df, factor_df, forward_period)
                metrics["name"] = name
                results.append(metrics)
            except Exception as e:
                print(f"    Eval {name}: {e}")

        results.sort(key=lambda x: abs(x.get("rank_ic", 0)), reverse=True)
        self.results = results
        return results

    def _evaluate_factor(self, df: pd.DataFrame, factor_df: pd.DataFrame, forward: int) -> dict:
        """Compute IC, Rank IC, quantile returns for a factor."""
        price_df = df[["symbol", "date", "close"]].copy()
        price_df["fwd_close"] = price_df.groupby("symbol")["close"].shift(-forward)
        price_df["fwd_ret"] = (price_df["fwd_close"] - price_df["close"]) / price_df["close"]

        merged = factor_df.merge(price_df[["symbol", "date", "fwd_ret"]], on=["symbol", "date"], how="inner")
        merged = merged.dropna(subset=["value", "fwd_ret"])
        if len(merged) < 100:
            return {"ic": 0, "rank_ic": 0, "top_ret": 0, "bottom_ret": 0, "spread": 0, "n_obs": 0}

        # Cross-sectional evaluation per date
        dates = merged["date"].unique()
        ics, rank_ics = [], []
        top_rets, bottom_rets = [], []

        for d in dates[-60:]:  # last 60 dates for stable estimates
            cross = merged[merged["date"] == d]
            if len(cross) < 20:
                continue
            # IC
            ic = cross["value"].corr(cross["fwd_ret"])
            if not np.isnan(ic):
                ics.append(ic)
            # Rank IC
            rank_ic = cross["value"].rank().corr(cross["fwd_ret"].rank())
            if not np.isnan(rank_ic):
                rank_ics.append(rank_ic)
            # Quantile returns
            cross["q"] = pd.qcut(cross["value"].rank(method="first"), 5, labels=False, duplicates="drop")
            if cross["q"].nunique() >= 5:
                top = cross[cross["q"] == 4]
                bottom = cross[cross["q"] == 0]
                top_rets.append(top["fwd_ret"].mean())
                bottom_rets.append(bottom["fwd_ret"].mean())

        # Long-term IC stability
        all_ic, all_rank_ic = [], []
        for d in dates:
            cross = merged[merged["date"] == d]
            if len(cross) < 20:
                continue
            ic = cross["value"].corr(cross["fwd_ret"])
            if not np.isnan(ic):
                all_ic.append(ic)
            ric = cross["value"].rank().corr(cross["fwd_ret"].rank())
            if not np.isnan(ric):
                all_rank_ic.append(ric)

        return {
            "ic": round(np.mean(ics), 4) if ics else 0,
            "ic_std": round(np.std(ics), 4) if ics else 0,
            "ic_ir": round(np.mean(ics) / np.std(ics), 2) if ics and np.std(ics) > 0 else 0,
            "rank_ic": round(np.mean(rank_ics), 4) if rank_ics else 0,
            "rank_ic_std": round(np.std(rank_ics), 4) if rank_ics else 0,
            "rank_ic_ir": round(np.mean(rank_ics) / np.std(rank_ics), 2) if rank_ics and np.std(rank_ics) > 0 else 0,
            "top_ret": round(np.mean(top_rets) * 100, 3) if top_rets else 0,
            "bottom_ret": round(np.mean(bottom_rets) * 100, 3) if bottom_rets else 0,
            "spread": round((np.mean(top_rets) - np.mean(bottom_rets)) * 100, 3) if top_rets and bottom_rets else 0,
            "long_ic": round(np.mean(all_ic), 4) if all_ic else 0,
            "long_rank_ic": round(np.mean(all_rank_ic), 4) if all_rank_ic else 0,
            "n_obs": len(merged),
        }

    # ---- Report ----

    def report(self, top_n: int = 30) -> str:
        """Generate a formatted report of top factors."""
        results = self.results or self.evaluate_all()

        spec_map = {s.name: s for s in FACTOR_CATALOG}
        lines = []
        lines.append("=" * 100)
        lines.append(f"  Factor Performance Report — {self.market.upper()} Market (forward 5-day return)")
        lines.append("=" * 100)
        lines.append(f"  {'Rank':<5} {'Factor':<22} {'Rank IC':>8} {'IC IR':>8} {'Spread%':>8} {'Time':<12} {'Type':<12}")
        lines.append("-" * 100)

        for i, r in enumerate(results[:top_n]):
            spec = spec_map.get(r["name"])
            time_str = spec.time.value if spec else "?"
            type_str = spec.ftype.value if spec else "?"
            emoji = "🔥" if abs(r["rank_ic"]) > 0.03 else "⭐" if abs(r["rank_ic"]) > 0.02 else "  "
            lines.append(
                f"  {emoji} {i+1:<3} {r['name']:<22} "
                f"{r['rank_ic']:>+8.4f} {r['ic_ir']:>8.2f} "
                f"{r['spread']:>8.3f} {time_str:<12} {type_str:<12}"
            )

        lines.append("-" * 100)
        lines.append(f"  Total factors evaluated: {len(results)}")
        lines.append(f"  Market: {self.market.upper()}, Sample: {self.sample_size} stocks")
        return "\n".join(lines)
