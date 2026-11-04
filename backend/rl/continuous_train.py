"""Continuous Training Loop — 持续训练直到指定时间.

反复执行完整训练周期（Best-of-5 + Walk-Forward + 晋升），
直到达到截止时间。用于夜间批量训练迭代。

Usage:
    python -m backend.rl.continuous_train --until 07:00
    python -m backend.rl.continuous_train --until 07:00 --n-seeds 5 --steps 150000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl.continuous")


def parse_until(until_str: str) -> datetime:
    """Parse target time (today or tomorrow if passed)."""
    now = datetime.now()
    hour, minute = map(int, until_str.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def main():
    parser = argparse.ArgumentParser(description="Continuous RL Training Loop")
    parser.add_argument("--until", type=str, default="07:00",
                        help="Stop time (HH:MM, default: 07:00)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Best-of-N seeds per cycle (default: 5)")
    parser.add_argument("--steps", type=int, default=150_000,
                        help="Training steps per seed (default: 150000)")
    parser.add_argument("--top-n", type=int, default=1000,
                        help="Stock universe size (default: 1000)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="RL top-K selection (default: 30)")
    parser.add_argument("--db-path", type=str,
                        default="data/market/market_data_train.duckdb",
                        help="DuckDB path")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    target_time = parse_until(args.until)
    hours_remaining = (target_time - datetime.now()).total_seconds() / 3600

    logger.info("=" * 70)
    logger.info("CONTINUOUS TRAINING LOOP")
    logger.info(f"  Run until: {target_time.strftime('%Y-%m-%d %H:%M')} ({hours_remaining:.1f}h)")
    logger.info(f"  Config: seeds={args.n_seeds}, steps={args.steps:,}, top_n={args.top_n}")
    logger.info("=" * 70)

    from rl.pipeline import TrainingPipeline, PipelineConfig

    config = PipelineConfig(
        db_path=Path(args.db_path),
        top_n=args.top_n,
        top_k=args.top_k,
        n_seeds=args.n_seeds,
        total_steps=args.steps,
        device=args.device,
    )

    pipeline = TrainingPipeline(config)

    cycle = 0
    best_sharpe_ever = 0.0
    promotions = 0

    while datetime.now() < target_time:
        cycle += 1
        remaining = (target_time - datetime.now()).total_seconds() / 60
        logger.info(f"\n{'#' * 70}")
        logger.info(f"CYCLE {cycle} | Remaining: {remaining:.0f} min | Best ever: {best_sharpe_ever:.2f}")
        logger.info(f"{'#' * 70}")

        # Check if enough time for another cycle (~10 min needed)
        if remaining < 12:
            logger.info("Not enough time for another cycle, stopping.")
            break

        try:
            result = pipeline.run_daily_cycle(skip_data_refresh=True)

            if result.success:
                if result.best_sharpe > best_sharpe_ever:
                    best_sharpe_ever = result.best_sharpe
                if result.promoted:
                    promotions += 1
                logger.info(
                    f"Cycle {cycle}: Sharpe={result.best_sharpe:.2f}, "
                    f"Return={result.best_return:+.2f}%, "
                    f"Promoted={'YES' if result.promoted else 'no'}"
                )
            else:
                logger.warning(f"Cycle {cycle} FAILED: {result.error}")

        except Exception as e:
            logger.exception(f"Cycle {cycle} exception: {e}")

        # Brief pause between cycles
        time.sleep(5)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("CONTINUOUS TRAINING COMPLETE")
    logger.info(f"  Cycles completed: {cycle}")
    logger.info(f"  Best Sharpe ever: {best_sharpe_ever:.2f}")
    logger.info(f"  Models promoted: {promotions}")
    logger.info(f"  Finished at: {datetime.now().strftime('%H:%M:%S')}")

    # Print registry status
    status = pipeline.get_status()
    logger.info(f"  Production model: {status.get('production_model')}")
    logger.info(f"  Production Sharpe: {status.get('production_sharpe', 0):.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
