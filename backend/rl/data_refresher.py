"""Data Refresher — 市场数据增量刷新.

复用现有 marketdata 模块进行增量数据更新：
- 美股每日收盘后增量刷新
- 支持全量/增量模式
- 返回刷新结果（新增天数、失败数）
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger("rl.data_refresher")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "market" / "market_data_train.duckdb"


@dataclass
class RefreshResult:
    """数据刷新结果."""
    success: bool
    new_bars: int = 0
    symbols_refreshed: int = 0
    symbols_failed: int = 0
    duration_seconds: float = 0.0
    error: str = ""
    db_path: str = ""


class DataRefresher:
    """市场数据增量刷新器.
    
    调用 scripts/refresh_market_data.py 进行增量更新。
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        market: str = "us",
        provider: str = "auto",
    ):
        self.db_path = db_path
        self.market = market
        self.provider = provider

    def refresh(
        self,
        mode: str = "incremental",
        days_back: int = 5,
        full_fetch: bool = False,
    ) -> RefreshResult:
        """执行数据刷新.

        Args:
            mode: "incremental" (增量) 或 "init" (全量初始化)
            days_back: 增量模式回溯天数
            full_fetch: 是否刷新全市场

        Returns:
            RefreshResult
        """
        t0 = time.time()
        logger.info(f"Starting data refresh: market={self.market}, mode={mode}")

        # Build command
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "refresh_market_data.py"),
            "--market", self.market,
            "--mode", mode,
            "--start", start_date.isoformat(),
            "--end", end_date.isoformat(),
            "--db-path", str(self.db_path),
            "--us-provider", self.provider,
        ]

        if full_fetch:
            cmd.append("--full-fetch")

        logger.info(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
                cwd=str(PROJECT_ROOT),
            )

            duration = time.time() - t0

            if result.returncode == 0:
                # Parse output for stats
                output = result.stdout
                new_bars = self._parse_bars_written(output)
                symbols_refreshed = self._parse_symbols_count(output)

                logger.info(
                    f"  Refresh completed: {new_bars} new bars, "
                    f"{symbols_refreshed} symbols ({duration:.0f}s)"
                )
                return RefreshResult(
                    success=True,
                    new_bars=new_bars,
                    symbols_refreshed=symbols_refreshed,
                    duration_seconds=duration,
                    db_path=str(self.db_path),
                )
            else:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                logger.error(f"  Refresh failed: {error_msg}")
                return RefreshResult(
                    success=False,
                    duration_seconds=duration,
                    error=error_msg,
                    db_path=str(self.db_path),
                )

        except subprocess.TimeoutExpired:
            logger.error("  Refresh timed out (600s)")
            return RefreshResult(
                success=False,
                duration_seconds=time.time() - t0,
                error="Timeout after 600s",
                db_path=str(self.db_path),
            )
        except Exception as e:
            logger.error(f"  Refresh exception: {e}")
            return RefreshResult(
                success=False,
                duration_seconds=time.time() - t0,
                error=str(e),
                db_path=str(self.db_path),
            )

    def check_data_freshness(self) -> dict:
        """检查数据新鲜度.

        Returns:
            dict with last_date, days_since_update, n_symbols
        """
        try:
            import duckdb
            conn = duckdb.connect(str(self.db_path), read_only=True)
            row = conn.execute(
                "SELECT MAX(trade_date), COUNT(DISTINCT symbol) FROM daily_bars WHERE market = ?",
                [self.market],
            ).fetchone()
            conn.close()

            if row and row[0]:
                last_date = str(row[0])
                n_symbols = row[1]
                # Calculate days since last update
                try:
                    from datetime import datetime
                    last_dt = datetime.strptime(last_date[:10], "%Y-%m-%d")
                    days_since = (datetime.now() - last_dt).days
                except ValueError:
                    days_since = -1

                return {
                    "last_date": last_date,
                    "days_since_update": days_since,
                    "n_symbols": n_symbols,
                    "is_stale": days_since > 3,  # >3 days = stale (weekends)
                }
            return {"last_date": None, "days_since_update": -1, "n_symbols": 0, "is_stale": True}

        except Exception as e:
            logger.warning(f"  Failed to check data freshness: {e}")
            return {"last_date": None, "days_since_update": -1, "n_symbols": 0, "is_stale": True}

    def get_data_stats(self) -> dict:
        """获取数据统计信息."""
        try:
            import duckdb
            conn = duckdb.connect(str(self.db_path), read_only=True)
            stats = conn.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as n_symbols,
                    COUNT(*) as total_bars,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    AVG(volume) as avg_volume
                FROM daily_bars WHERE market = ?
            """, [self.market]).fetchone()
            conn.close()

            return {
                "n_symbols": stats[0],
                "total_bars": stats[1],
                "min_date": str(stats[2]) if stats[2] else None,
                "max_date": str(stats[3]) if stats[3] else None,
                "avg_volume": float(stats[4]) if stats[4] else 0,
            }
        except Exception as e:
            logger.warning(f"  Failed to get data stats: {e}")
            return {}

    def _parse_bars_written(self, output: str) -> int:
        """Parse bars written from refresh output."""
        import re
        # Look for patterns like "wrote N bars" or "upserted rows=N"
        patterns = [
            r"wrote\s+(\d+)\s+bars",
            r"upserted\s+rows=(\d+)",
            r"(\d+)\s+new bars",
        ]
        total = 0
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            total += sum(int(m) for m in matches)
        return total

    def _parse_symbols_count(self, output: str) -> int:
        """Parse symbols count from refresh output."""
        import re
        patterns = [
            r"(\d+)\s+symbols?\s+(?:refreshed|updated|processed)",
            r"refreshing\s+(\d+)\s+symbols",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0
