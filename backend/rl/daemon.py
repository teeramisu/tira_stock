"""RL Training Daemon — 7x24自主训练守护进程.

主服务入口，负责：
- APScheduler调度：每日16:30 ET触发训练
- 每小时健康检查
- 优雅关闭（SIGTERM/SIGINT）
- 启动时检查是否有遗漏训练
- CLI入口：python -m backend.rl.daemon

Usage:
    python -m backend.rl.daemon                    # 启动守护进程
    python -m backend.rl.daemon --run-now          # 立即执行一次训练
    python -m backend.rl.daemon --status           # 查看状态
    python -m backend.rl.daemon --health           # 健康检查
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rl.daemon")


class TrainingDaemon:
    """RL训练守护进程."""

    def __init__(
        self,
        db_path: str = None,
        schedule_hour: int = 16,
        schedule_minute: int = 30,
        timezone: str = "US/Eastern",
        health_check_interval_hours: int = 1,
    ):
        self.db_path = Path(db_path) if db_path else None
        self.schedule_hour = schedule_hour
        self.schedule_minute = schedule_minute
        self.timezone = timezone
        self.health_check_interval_hours = health_check_interval_hours

        self._running = False
        self._shutdown_event = threading.Event()
        self._scheduler = None

        # Initialize components lazily
        self._pipeline = None
        self._monitor = None

    @property
    def pipeline(self):
        """Lazy init pipeline."""
        if self._pipeline is None:
            from rl.pipeline import TrainingPipeline, PipelineConfig
            config = PipelineConfig()
            if self.db_path:
                config.db_path = self.db_path
            self._pipeline = TrainingPipeline(config)
        return self._pipeline

    @property
    def monitor(self):
        """Lazy init monitor."""
        if self._monitor is None:
            from rl.monitor import HealthMonitor
            self._monitor = HealthMonitor(
                registry=self.pipeline.registry,
                metrics_store=self.pipeline.metrics,
            )
        return self._monitor

    def start(self):
        """启动守护进程."""
        logger.info("=" * 70)
        logger.info("RL Training Daemon Starting")
        logger.info(f"  Schedule: Daily at {self.schedule_hour:02d}:{self.schedule_minute:02d} {self.timezone}")
        logger.info(f"  Health check: Every {self.health_check_interval_hours}h")
        logger.info(f"  DB Path: {self.db_path or 'default'}")
        logger.info("=" * 70)

        self._running = True

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Check for missed training on startup
        self._check_missed_training()

        # Start scheduler
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            self._scheduler = BackgroundScheduler()

            # Daily training job
            self._scheduler.add_job(
                self._daily_training_job,
                CronTrigger(
                    hour=self.schedule_hour,
                    minute=self.schedule_minute,
                    timezone=self.timezone,
                ),
                id="daily_training",
                name="Daily Training Cycle",
                replace_existing=True,
            )

            # Health check job
            self._scheduler.add_job(
                self._health_check_job,
                IntervalTrigger(hours=self.health_check_interval_hours),
                id="health_check",
                name="Health Check",
                replace_existing=True,
            )

            self._scheduler.start()
            logger.info("Scheduler started. Waiting for jobs...")

            # Print next run times
            for job in self._scheduler.get_jobs():
                logger.info(f"  Next {job.name}: {job.next_run_time}")

        except ImportError:
            logger.warning("APScheduler not installed, running in simple loop mode")
            self._run_simple_loop()
            return

        # Main loop - wait for shutdown
        try:
            while self._running:
                self._shutdown_event.wait(timeout=60)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """停止守护进程."""
        if not self._running:
            return  # Already stopped
        logger.info("Shutting down daemon...")
        self._running = False
        self._shutdown_event.set()

        if self._scheduler:
            try:
                self._scheduler.shutdown(wait=False)
                logger.info("Scheduler stopped")
            except Exception:
                pass  # Already stopped

        logger.info("Daemon stopped")

    def run_once(self, skip_data_refresh: bool = False) -> dict:
        """立即执行一次训练周期.

        Args:
            skip_data_refresh: 跳过数据刷新

        Returns:
            PipelineResult as dict
        """
        logger.info("Running one-time training cycle...")
        result = self.pipeline.run_daily_cycle(skip_data_refresh=skip_data_refresh)
        return {
            "success": result.success,
            "run_id": result.run_id,
            "best_sharpe": result.best_sharpe,
            "best_return": result.best_return,
            "promoted": result.promoted,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }

    def get_status(self) -> dict:
        """获取守护进程状态."""
        status = {
            "running": self._running,
            "scheduler_jobs": [],
        }

        if self._scheduler:
            for job in self._scheduler.get_jobs():
                status["scheduler_jobs"].append({
                    "name": job.name,
                    "next_run": str(job.next_run_time),
                })

        # Pipeline status
        try:
            status["pipeline"] = self.pipeline.get_status()
        except Exception as e:
            status["pipeline_error"] = str(e)

        return status

    def get_health(self) -> dict:
        """获取健康状态."""
        return self.monitor.check().to_dict()

    def _daily_training_job(self):
        """每日训练任务（由调度器调用）."""
        logger.info("\n" + "=" * 70)
        logger.info("DAILY TRAINING TRIGGERED")
        logger.info("=" * 70)

        try:
            result = self.pipeline.run_daily_cycle()

            if result.success:
                logger.info(f"Daily training SUCCESS: Sharpe={result.best_sharpe:.2f}")
                if result.promoted:
                    self.monitor.alert(
                        f"New model promoted: {result.version_id} (Sharpe={result.best_sharpe:.2f})",
                        level="info",
                    )
            else:
                logger.error(f"Daily training FAILED: {result.error}")
                self.monitor.alert(
                    f"Training failed: {result.error}",
                    level="error",
                )

        except Exception as e:
            logger.exception(f"Daily training exception: {e}")
            self.monitor.alert(f"Training exception: {e}", level="critical")

    def _health_check_job(self):
        """健康检查任务（由调度器调用）."""
        status = self.monitor.check()

        if not status.healthy:
            for alert in status.alerts:
                self.monitor.alert(alert, level="warning")

        # Log periodic status
        logger.debug(
            f"Health: {'OK' if status.healthy else 'ALERT'} | "
            f"Disk: {status.disk_free_gb:.0f}GB | "
            f"Mem: {status.memory_usage_percent:.0f}%"
        )

    def _check_missed_training(self):
        """检查是否有遗漏的训练（服务重启后）."""
        try:
            last_run = self.pipeline.metrics.get_last_successful_run()
            if last_run and last_run.get("finished_at"):
                last_dt = datetime.fromisoformat(last_run["finished_at"])
                hours_since = (datetime.now() - last_dt).total_seconds() / 3600

                if hours_since > 36:
                    logger.warning(
                        f"Last training was {hours_since:.0f}h ago. "
                        "Consider running --run-now to catch up."
                    )
            else:
                logger.info("No previous training runs found. First run pending.")

        except Exception as e:
            logger.warning(f"Failed to check missed training: {e}")

    def _run_simple_loop(self):
        """简单循环模式（无APScheduler时的fallback）."""
        logger.info("Running in simple loop mode (no APScheduler)")

        while self._running:
            # Check if it's time to run (simple hour check)
            now = datetime.now()
            if now.hour == self.schedule_hour and now.minute == self.schedule_minute:
                self._daily_training_job()
                # Sleep to avoid re-triggering in the same minute
                time.sleep(120)

            # Health check every N hours (simplified)
            self._health_check_job()

            # Sleep 60 seconds
            self._shutdown_event.wait(timeout=60)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()


def main():
    """CLI入口."""
    parser = argparse.ArgumentParser(
        description="RL-MACS Training Daemon (7x24 Autonomous Training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start daemon (runs forever, trains daily at 16:30 ET)
  python -m backend.rl.daemon

  # Run one training cycle immediately
  python -m backend.rl.daemon --run-now

  # Check daemon status
  python -m backend.rl.daemon --status

  # Health check
  python -m backend.rl.daemon --health

  # Custom schedule (train at 18:00 UTC)
  python -m backend.rl.daemon --hour 18 --minute 0 --timezone UTC
""",
    )

    # Actions
    parser.add_argument("--run-now", action="store_true",
                        help="Run one training cycle immediately and exit")
    parser.add_argument("--status", action="store_true",
                        help="Print daemon status and exit")
    parser.add_argument("--health", action="store_true",
                        help="Run health check and exit")
    parser.add_argument("--report", action="store_true",
                        help="Print full health report and exit")

    # Configuration
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to market data DuckDB")
    parser.add_argument("--hour", type=int, default=16,
                        help="Training schedule hour (default: 16)")
    parser.add_argument("--minute", type=int, default=30,
                        help="Training schedule minute (default: 30)")
    parser.add_argument("--timezone", type=str, default="US/Eastern",
                        help="Schedule timezone (default: US/Eastern)")
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Skip data refresh (for --run-now)")

    args = parser.parse_args()

    # Override DB path from environment
    db_path = args.db_path or os.environ.get("RL_DB_PATH")

    daemon = TrainingDaemon(
        db_path=db_path,
        schedule_hour=args.hour,
        schedule_minute=args.minute,
        timezone=args.timezone,
    )

    # Handle actions
    if args.run_now:
        result = daemon.run_once(skip_data_refresh=args.skip_refresh)
        print(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Sharpe: {result['best_sharpe']:.2f}")
        print(f"  Return: {result['best_return']:.2f}%")
        print(f"  Promoted: {result['promoted']}")
        print(f"  Duration: {result['duration_seconds']:.0f}s")
        if result['error']:
            print(f"  Error: {result['error']}")
        sys.exit(0 if result['success'] else 1)

    elif args.status:
        import json
        status = daemon.get_status()
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0)

    elif args.health:
        import json
        health = daemon.get_health()
        print(json.dumps(health, indent=2, default=str))
        sys.exit(0 if health['healthy'] else 1)

    elif args.report:
        print(daemon.monitor.get_report())
        sys.exit(0)

    else:
        # Start daemon
        daemon.start()


if __name__ == "__main__":
    main()
