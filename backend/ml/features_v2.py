"""Expanded feature matrix — adds 20+ cross-domain features to the original 10."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "stock_backtest" / "backend"))

from ml.features import FeatureConfig, build_feature_matrix, FEATURE_NAMES as BASE_FEATURE_NAMES
import akshare as ak


EXTRA_FEATURE_NAMES = [
    "north_net_flow", "north_buy_vol", "north_sell_vol",
    "bond_cn_2y", "bond_cn_5y", "bond_cn_10y", "bond_us_10y",
    "commodity_index", "commodity_change", "cpi", "gold_close",
]

ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(EXTRA_FEATURE_NAMES)


def build_expanded_feature_matrix(
    close_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame | None = None,
    regime_by_date: dict | None = None,
    *,
    config: FeatureConfig = FeatureConfig(),
):
    """Build feature matrix with extra cross-domain features appended."""

    # 1. Base features
    fm = build_feature_matrix(close_matrix, volume_matrix, regime_by_date, config=config)
    n_dates = fm.features.shape[0]
    dates = fm.dates

    # 2. Extra features
    extra = np.zeros((n_dates, len(EXTRA_FEATURE_NAMES)), dtype=np.float32)

    # Fetch all external time series
    external = {}

    # Northbound flow
    try:
        nb = ak.stock_hsgt_hist_em()
        nb = nb.rename(columns={nb.columns[0]: 'date'})
        nb['date'] = pd.to_datetime(nb['date'])
        nb = nb.set_index('date').sort_index()
        cols = nb.columns[:3]
        for i, col in enumerate(cols):
            external[EXTRA_FEATURE_NAMES[i]] = nb[col].astype(float)
    except Exception:
        pass

    # Bond yields
    try:
        bonds = ak.bond_zh_us_rate()
        bonds = bonds.rename(columns={bonds.columns[0]: 'date'})
        bonds['date'] = pd.to_datetime(bonds['date'])
        bonds = bonds.set_index('date').sort_index()
        for src_col, tgt_name in [
            ('中国国债收益率2年', 'bond_cn_2y'), ('中国国债收益率5年', 'bond_cn_5y'),
            ('中国国债收益率10年', 'bond_cn_10y'), ('美国国债收益率10年', 'bond_us_10y'),
        ]:
            if src_col in bonds.columns:
                external[tgt_name] = bonds[src_col].astype(float)
    except Exception:
        pass

    # Commodity
    try:
        com = ak.macro_china_commodity_price_index()
        com = com.rename(columns={'日期': 'date'})
        com['date'] = pd.to_datetime(com['date'])
        com = com.set_index('date').sort_index()
        for src_col, tgt_name in [('最新值', 'commodity_index'), ('涨跌幅', 'commodity_change')]:
            if src_col in com.columns:
                external[tgt_name] = com[src_col].astype(float)
    except Exception:
        pass

    # CPI
    try:
        cpi_df = ak.macro_china_cpi_monthly()
        for col in cpi_df.columns:
            if '日期' in str(col) or 'date' in col.lower():
                cpi_df = cpi_df.rename(columns={col: 'date'})
                break
        cpi_df['date'] = pd.to_datetime(cpi_df['date'])
        cpi_df = cpi_df.set_index('date').sort_index()
        for col in cpi_df.columns:
            if '今值' in str(col) or '同比' in str(col):
                external['cpi'] = cpi_df[col].astype(float).resample('D').ffill()
                break
    except Exception:
        pass

    # Gold futures
    try:
        gold = ak.futures_zh_daily_sina(symbol='AU0')
        gold = gold.rename(columns={gold.columns[0]: 'date'})
        gold['date'] = pd.to_datetime(gold['date'])
        gold = gold.set_index('date').sort_index()
        external['gold_close'] = gold['close'].astype(float)
    except Exception:
        pass

    # Map external features to the feature matrix dates
    for i, d in enumerate(dates):
        d_ts = pd.Timestamp(d)
        for j, fname in enumerate(EXTRA_FEATURE_NAMES):
            if fname in external:
                series = external[fname]
                # Find nearest date
                if d_ts in series.index:
                    val = series.loc[d_ts]
                else:
                    before = series[series.index < d_ts]
                    if len(before) > 0:
                        val = before.iloc[-1]
                    else:
                        continue
                if not np.isnan(float(val)):
                    extra[i, j] = float(val)

    # Combine
    combined = np.concatenate([fm.features, extra], axis=1)
    combined = np.nan_to_num(combined, nan=0.0)

    fm.features = combined
    fm.feature_names = ALL_FEATURE_NAMES
    return fm
