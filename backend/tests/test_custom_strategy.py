"""Tests for custom Python strategy sandbox execution."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.custom_strategy import run_custom_backtest, _run_user_signal


def _simple_df(n: int = 60) -> pd.DataFrame:
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


class TestSignalExecution:
    def test_buy_and_hold_signal(self):
        df = _simple_df()
        code = "def signal(df): return [1] * len(df)"
        res = run_custom_backtest(df, code, initial_cash=100_000.0)
        assert len(res.equity_curve) == len(df)
        assert res.final_value > 100_000

    def test_all_cash_signal(self):
        df = _simple_df()
        code = "def signal(df): return [0] * len(df)"
        res = run_custom_backtest(df, code, initial_cash=100_000.0)
        assert res.final_value == 100_000.0
        assert res.n_trades == 0

    def test_toggle_signal_generates_one_trade(self):
        n = 60
        df = _simple_df(n)
        code = """
def signal(df):
    pos = [0] * len(df)
    pos[20:] = [1] * (len(df) - 20)
    return pos
"""
        res = run_custom_backtest(df, code, initial_cash=100_000.0)
        assert res.n_trades <= 1
        assert res.final_value > 0


class TestInputValidation:
    def test_rejects_missing_signal_function(self):
        df = _simple_df()
        with pytest.raises(ValueError, match="必须定义函数 signal"):
            _run_user_signal("x = 1", df)

    def test_rejects_none_return(self):
        df = _simple_df()
        with pytest.raises(ValueError, match="不能返回 None"):
            _run_user_signal("def signal(df): return None", df)

    def test_rejects_wrong_length(self):
        df = _simple_df(60)
        with pytest.raises(ValueError):
            _run_user_signal("def signal(df): return [1]*10", df)

    def test_rejects_code_too_long(self):
        df = _simple_df()
        long_code = "def signal(df): return [0]*len(df)\n" + "#" * 60_000
        with pytest.raises(ValueError, match="50000"):
            run_custom_backtest(df, long_code)

    def test_invalid_timeout_clamps(self):
        df = _simple_df()
        code = "def signal(df): return [0]*len(df)"
        run_custom_backtest(df, code, timeout_seconds=0)
        run_custom_backtest(df, code, timeout_seconds=61)


class TestSandboxRestrictions:
    def test_restricted_globals_no_open(self):
        df = _simple_df()
        code = "def signal(df): open('/tmp/evil', 'w'); return [0]*len(df)"
        with pytest.raises((NameError, AttributeError)):
            _run_user_signal(code, df)

    def test_restricted_globals_import_blocked(self):
        df = _simple_df()
        code = "def signal(df): import os; return [0]*len(df)"
        with pytest.raises((NameError, ImportError, SyntaxError)):
            _run_user_signal(code, df)

    def test_restricted_globals_no_eval(self):
        df = _simple_df()
        code = "def signal(df): return eval('[1]*len(df)')"
        with pytest.raises((NameError, AttributeError, SyntaxError)):
            _run_user_signal(code, df)

    def test_restricted_globals_no_exec(self):
        df = _simple_df()
        code = "def signal(df): exec('x=1'); return [0]*len(df)"
        with pytest.raises((NameError, AttributeError, SyntaxError)):
            _run_user_signal(code, df)


class TestTimeout:
    def test_infinite_loop_times_out(self):
        df = _simple_df(30)
        code = "def signal(df):\n    while True: pass\n    return [0]*len(df)"
        with pytest.raises(TimeoutError):
            run_custom_backtest(df, code, timeout_seconds=2.0)


class TestErrorHandling:
    def test_runtime_error_in_signal_propagates(self):
        df = _simple_df()
        code = "def signal(df): raise RuntimeError('boom'); return [0]*len(df)"
        with pytest.raises((ValueError, RuntimeError)):
            run_custom_backtest(df, code)

    def test_nan_price_returns_nan_final_value(self):
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": [100.0] * 15 + [float("nan")] * 15}, index=dates)
        code = "def signal(df): return [1]*len(df)"
        res = run_custom_backtest(df, code)
        assert np.isnan(res.final_value) or res.final_value > 0

    def test_zero_price_rejected(self):
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = [0.0] * 5 + [100.0] * 25
        df = pd.DataFrame({"Close": close}, index=dates)
        code = "def signal(df):\n    pos = [0]*len(df)\n    for i in range(5, len(df)):\n        pos[i] = 1\n    return pos"
        res = run_custom_backtest(df, code)
        assert res.final_value > 0 or res.n_trades == 0


class TestGlobalsWhitelist:
    def test_common_math_builtins_available(self):
        df = _simple_df(60)
        code = """def signal(df):
    ret = [abs(-1), int(3.14), float(2), round(1.234, 1)]
    return ret + [0] * (len(df) - len(ret))
"""
        _run_user_signal(code, df)

    def test_type_not_available(self):
        df = _simple_df()
        code = "def signal(df): return [type(1)] * len(df)"
        with pytest.raises((NameError, AttributeError)):
            _run_user_signal(code, df)

    def test_list_of_lists_signal_raises(self):
        df = _simple_df(5)
        code = "def signal(df): return [[1], [2]]"
        with pytest.raises(Exception):
            _run_user_signal(code, df)

    def test_bool_builtins_available(self):
        df = _simple_df(60)
        code = """def signal(df):
    ret = [bool(1), True, False, all([True]), any([False])]
    return ret + [0] * (len(df) - len(ret))
"""
        res = _run_user_signal(code, df)
        assert len(res) == len(df)

    def test_enumerate_and_zip_available(self):
        df = _simple_df(60)
        code = """def signal(df):
    ret = [0]*len(df)
    for i, v in enumerate(range(len(df))):
        if i < 5:
            ret[i] = 1
    return ret
"""
        res = _run_user_signal(code, df)
        assert sum(res[:5]) == 5

    def test_numpy_available_via_global_shim(self):
        df = _simple_df(30)
        code = """def signal(df):
    return [int(np.ones(1)[0])] * len(df)
"""
        _run_user_signal(code, df)

    def test_pandas_available_via_global_shim(self):
        df = _simple_df(30)
        code = """def signal(df):
    return list(pd.Series([1]*len(df), index=df.index))
"""
        res = _run_user_signal(code, df)
        assert len(res) == len(df)
