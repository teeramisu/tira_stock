"""Health Monitor — 服务健康监控.

监控RL训练守护进程的健康状态：
- 服务存活检查
- 最近训练成功/失败状态
- 生产模型年龄（>7天告警）
- 磁盘/内存使用
- 日志告警
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rl.monitor")


@dataclass
class HealthStatus:
    """健康检查结果."""
    healthy: bool
    checked_at: str
    uptime_seconds: float = 0.0

    # Components
    service_alive: bool = True
    last_training_success: bool = True
    last_training_at: Optional[str] = None
    production_model_age_days: float = 0.0
    production_model_stale: bool = False

    # Resources
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0

    # Alerts
    alerts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "checked_at": self.checked_at,
            "uptime_seconds": self.uptime_seconds,
            "service_alive": self.service_alive,
            "last_training_success": self.last_training_success,
            "last_training_at": self.last_training_at,
            "production_model_age_days": self.production_model_age_days,
            "production_model_stale": self.production_model_stale,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_available_gb": self.memory_available_gb,
            "alerts": self.alerts,
            "warnings": self.warnings,
        }


class HealthMonitor:
    """健康监控器."""

    def __init__(
        self,
        registry=None,
        metrics_store=None,
        max_model_age_days: int = 7,
        min_disk_free_gb: float = 5.0,
        max_memory_percent: float = 90.0,
    ):
        self.registry = registry
        self.metrics_store = metrics_store
        self.max_model_age_days = max_model_age_days
        self.min_disk_free_gb = min_disk_free_gb
        self.max_memory_percent = max_memory_percent
        self._start_time = time.time()

    def check(self) -> HealthStatus:
        """执行健康检查.

        Returns:
            HealthStatus with all check results
        """
        status = HealthStatus(
            healthy=True,
            checked_at=datetime.now().isoformat(),
            uptime_seconds=time.time() - self._start_time,
        )

        # Check 1: Last training status
        self._check_training_status(status)

        # Check 2: Production model age
        self._check_model_age(status)

        # Check 3: Disk space
        self._check_disk_space(status)

        # Check 4: Memory usage
        self._check_memory(status)

        # Determine overall health
        status.healthy = len(status.alerts) == 0

        # Log results
        if status.healthy:
            logger.debug(f"Health check: OK (uptime: {status.uptime_seconds/3600:.1f}h)")
        else:
            logger.warning(f"Health check: ALERTS={status.alerts}")

        if status.warnings:
            logger.info(f"Health check warnings: {status.warnings}")

        return status

    def _check_training_status(self, status: HealthStatus):
        """Check recent training success/failure."""
        if not self.metrics_store:
            return

        last_run = self.metrics_store.get_last_successful_run()
        if last_run:
            status.last_training_success = True
            status.last_training_at = last_run.get("finished_at")

            # Check if training is overdue (>36 hours)
            if status.last_training_at:
                try:
                    last_dt = datetime.fromisoformat(status.last_training_at)
                    hours_since = (datetime.now() - last_dt).total_seconds() / 3600
                    if hours_since > 36:
                        status.warnings.append(
                            f"No successful training in {hours_since:.0f} hours"
                        )
                except ValueError:
                    pass
        else:
            status.last_training_success = False
            status.warnings.append("No successful training runs found")

        # Check recent failures
        recent_runs = self.metrics_store.get_recent_runs(limit=3)
        consecutive_failures = sum(1 for r in recent_runs if r.get("status") == "failed")
        if consecutive_failures >= 3:
            status.alerts.append(f"{consecutive_failures} consecutive training failures")

    def _check_model_age(self, status: HealthStatus):
        """Check production model age."""
        if not self.registry:
            return

        production = self.registry.get_production()
        if production:
            try:
                created = datetime.fromisoformat(production.created_at)
                age_days = (datetime.now() - created).total_seconds() / 86400
                status.production_model_age_days = age_days

                if age_days > self.max_model_age_days:
                    status.production_model_stale = True
                    status.warnings.append(
                        f"Production model is {age_days:.0f} days old (max: {self.max_model_age_days})"
                    )
            except ValueError:
                pass
        else:
            status.warnings.append("No production model registered")

    def _check_disk_space(self, status: HealthStatus):
        """Check available disk space."""
        try:
            total, used, free = shutil.disk_usage("/")
            status.disk_usage_percent = (used / total) * 100
            status.disk_free_gb = free / (1024 ** 3)

            if status.disk_free_gb < self.min_disk_free_gb:
                status.alerts.append(
                    f"Low disk space: {status.disk_free_gb:.1f}GB free (min: {self.min_disk_free_gb}GB)"
                )
        except Exception as e:
            status.warnings.append(f"Failed to check disk space: {e}")

    def _check_memory(self, status: HealthStatus):
        """Check memory usage."""
        try:
            # Read from /proc/meminfo (Linux)
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]  # Get numeric value
                        meminfo[key] = int(value)

            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", 0)

            if total_kb > 0:
                status.memory_usage_percent = ((total_kb - available_kb) / total_kb) * 100
                status.memory_available_gb = available_kb / (1024 ** 2)

                if status.memory_usage_percent > self.max_memory_percent:
                    status.alerts.append(
                        f"High memory usage: {status.memory_usage_percent:.0f}% (max: {self.max_memory_percent}%)"
                    )
        except Exception as e:
            status.warnings.append(f"Failed to check memory: {e}")

    def alert(self, message: str, level: str = "warning"):
        """发送告警.

        Args:
            message: 告警消息
            level: "info" | "warning" | "error" | "critical"
        """
        log_func = getattr(logger, level, logger.warning)
        log_func(f"ALERT: {message}")

        # Could extend with webhook/email notifications here
        # For now, just log

    def get_report(self) -> str:
        """生成健康报告（人类可读）."""
        status = self.check()
        lines = [
            "=" * 50,
            "RL Training Daemon Health Report",
            "=" * 50,
            f"Time: {status.checked_at}",
            f"Status: {'HEALTHY' if status.healthy else 'UNHEALTHY'}",
            f"Uptime: {status.uptime_seconds/3600:.1f} hours",
            "",
            "Training:",
            f"  Last success: {status.last_training_at or 'Never'}",
            f"  Production model age: {status.production_model_age_days:.1f} days",
            "",
            "Resources:",
            f"  Disk: {status.disk_usage_percent:.0f}% used ({status.disk_free_gb:.1f}GB free)",
            f"  Memory: {status.memory_usage_percent:.0f}% used ({status.memory_available_gb:.1f}GB available)",
        ]

        if status.alerts:
            lines.extend(["", "ALERTS:"])
            lines.extend(f"  - {a}" for a in status.alerts)

        if status.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"  - {w}" for w in status.warnings)

        lines.append("=" * 50)
        return "\n".join(lines)
