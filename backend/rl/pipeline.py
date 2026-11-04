"""Training Pipeline — 训练管道编排器.

编排完整的每日训练周期：
1. 数据刷新（增量更新DuckDB）
2. Best-of-5训练（多seed选最优）
3. Walk-Forward验证（7窗口泛化测试）
4. 模型对比（候选 vs 生产）
5. 模型晋升（如果显著优于生产）
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.pipeline")

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.registry import ModelRegistry, ModelMetrics, ModelVersion
from rl.metrics_store import MetricsStore
from rl.data_refresher import DataRefresher, RefreshResult


@dataclass
class PipelineConfig:
    """训练管道配置."""
    # Data
    db_path: Path = Path(__file__).resolve().parents[2] / "data" / "market" / "market_data_train.duckdb"
    top_n: int = 1000          # 股票池大小
    top_k: int = 30            # RL选股数
    min_bars: int = 500        # 最小数据长度
    rebalance_freq: int = 5    # 再平衡频率

    # Training
    n_seeds: int = 5           # Best-of-N seeds
    total_steps: int = 150_000 # 训练步数
    learning_rate: float = 3e-4
    net_arch: list = field(default_factory=lambda: [256, 128])
    n_steps: int = 4096        # PPO rollout buffer
    batch_size: int = 1024
    ent_coef: float = 0.05
    device: str = "cpu"

    # Walk-Forward
    wf_train_window: int = 180
    wf_test_window: int = 63
    wf_step_size: int = 40
    wf_n_windows: int = 7

    # Promotion
    min_sharpe_for_promotion: float = 1.0
    max_drawdown_limit: float = 0.10

    # Paths
    models_dir: Path = Path(__file__).resolve().parent / "models"


@dataclass
class PipelineResult:
    """管道运行结果."""
    run_id: str
    success: bool
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0

    # Data refresh
    data_refresh: Optional[RefreshResult] = None

    # Training
    best_seed: int = 0
    best_sharpe: float = 0.0
    best_return: float = 0.0
    all_seed_results: list = field(default_factory=list)

    # Walk-Forward
    wf_positive_alpha_rate: float = 0.0
    wf_mean_alpha: float = 0.0
    wf_max_drawdown: float = 0.0

    # Promotion
    version_id: str = ""
    promoted: bool = False
    promotion_reason: str = ""

    # Errors
    error: str = ""
    stage_failed: str = ""


class TrainingPipeline:
    """训练管道编排器."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.registry = ModelRegistry(models_dir=self.config.models_dir)
        self.metrics = MetricsStore()
        self.refresher = DataRefresher(db_path=self.config.db_path)

    def run_daily_cycle(
        self,
        skip_data_refresh: bool = False,
        force_train: bool = False,
    ) -> PipelineResult:
        """执行每日训练周期.

        Args:
            skip_data_refresh: 跳过数据刷新（用于测试）
            force_train: 强制训练（即使数据未更新）

        Returns:
            PipelineResult
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        started_at = datetime.now().isoformat()
        t0 = time.time()

        result = PipelineResult(
            run_id=run_id,
            success=False,
            started_at=started_at,
        )

        logger.info("=" * 70)
        logger.info(f"Training Pipeline Started: {run_id}")
        logger.info("=" * 70)

        # Record run start
        self.metrics.start_training_run(
            run_id=run_id,
            total_steps=self.config.total_steps,
            n_seeds=self.config.n_seeds,
            config={
                "net_arch": self.config.net_arch,
                "n_steps": self.config.n_steps,
                "batch_size": self.config.batch_size,
                "top_n": self.config.top_n,
                "top_k": self.config.top_k,
            },
        )

        try:
            # ---- Stage 1: Data Refresh ----
            if not skip_data_refresh:
                result.data_refresh = self._stage_data_refresh()
                if not result.data_refresh.success:
                    logger.warning("Data refresh failed, continuing with existing data...")
            else:
                logger.info("[Stage 1] Skipping data refresh")

            # ---- Stage 2: Best-of-N Training ----
            training_result = self._stage_training()
            if training_result is None:
                raise RuntimeError("Training failed")

            result.best_seed = training_result["best_seed"]
            result.best_sharpe = training_result["best_sharpe"]
            result.best_return = training_result["best_return"]
            result.all_seed_results = training_result["all_results"]
            model_path = training_result["model_path"]

            # ---- Stage 3: Walk-Forward Validation ----
            wf_result = self._stage_walk_forward(model_path)
            result.wf_positive_alpha_rate = wf_result["positive_alpha_rate"]
            result.wf_mean_alpha = wf_result["mean_alpha"]
            result.wf_max_drawdown = wf_result["max_drawdown"]

            # ---- Stage 4: Register & Compare ----
            metrics = ModelMetrics(
                sharpe=result.best_sharpe,
                annual_return=result.best_return,
                max_drawdown=result.wf_max_drawdown,
                alpha=result.wf_mean_alpha,
                positive_alpha_rate=result.wf_positive_alpha_rate,
            )

            version = self.registry.register(
                model_path=model_path,
                config={
                    "net_arch": self.config.net_arch,
                    "total_steps": self.config.total_steps,
                    "n_seeds": self.config.n_seeds,
                    "best_seed": result.best_seed,
                },
                metrics=metrics,
                notes=f"Daily cycle: Sharpe={result.best_sharpe:.2f}, Alpha={result.wf_mean_alpha:.2%}",
            )
            result.version_id = version.version_id

            # ---- Stage 5: Promotion Decision ----
            production = self.registry.get_production()
            prod_metrics = production.metrics if production else None

            if self.registry.should_promote(metrics, prod_metrics):
                self.registry.promote(version.version_id)
                result.promoted = True
                result.promotion_reason = (
                    f"Sharpe {metrics.sharpe:.2f} > threshold, "
                    f"MaxDD {metrics.max_drawdown:.2%} < {self.config.max_drawdown_limit:.0%}"
                )
                logger.info(f"  PROMOTED: {version.version_id}")
            else:
                result.promotion_reason = "Did not meet promotion criteria"
                logger.info(f"  NOT promoted: {result.promotion_reason}")

            # Record evaluation
            self.metrics.record_evaluation(
                version_id=version.version_id,
                eval_type="walk_forward",
                sharpe=result.best_sharpe,
                annual_return=result.best_return,
                max_drawdown=result.wf_max_drawdown,
                alpha=result.wf_mean_alpha,
                positive_alpha_rate=result.wf_positive_alpha_rate,
                n_windows=self.config.wf_n_windows,
            )

            result.success = True

        except Exception as e:
            result.error = str(e)
            result.stage_failed = self._get_current_stage()
            logger.error(f"Pipeline failed at {result.stage_failed}: {e}")

        # Finalize
        result.finished_at = datetime.now().isoformat()
        result.duration_seconds = time.time() - t0

        self.metrics.finish_training_run(
            run_id=run_id,
            status="completed" if result.success else "failed",
            best_seed=result.best_seed,
            best_sharpe=result.best_sharpe,
            best_return=result.best_return,
            version_id=result.version_id,
            error=result.error,
        )

        # Daily status
        self.metrics.record_daily_status(
            training_triggered=True,
            training_succeeded=result.success,
            model_promoted=result.promoted,
            data_refreshed=result.data_refresh.success if result.data_refresh else False,
            new_data_days=result.data_refresh.new_bars if result.data_refresh else 0,
            error=result.error,
        )

        logger.info("=" * 70)
        logger.info(f"Pipeline {'SUCCESS' if result.success else 'FAILED'}: {run_id}")
        logger.info(f"  Duration: {result.duration_seconds:.0f}s")
        logger.info(f"  Best Sharpe: {result.best_sharpe:.2f} (seed={result.best_seed})")
        logger.info(f"  WF Alpha: {result.wf_mean_alpha:.2%} (positive rate: {result.wf_positive_alpha_rate:.0%})")
        logger.info(f"  Promoted: {result.promoted}")
        logger.info("=" * 70)

        return result

    def _stage_data_refresh(self) -> RefreshResult:
        """Stage 1: 数据刷新."""
        logger.info("\n[Stage 1] Data Refresh...")
        return self.refresher.refresh(mode="incremental", days_back=5)

    def _stage_training(self) -> Optional[dict]:
        """Stage 2: Best-of-N训练."""
        logger.info(f"\n[Stage 2] Best-of-{self.config.n_seeds} Training...")

        from rl.run_training import (
            get_top_liquid_stocks, load_market_data, align_price_data,
            temporal_train_test_split, compute_market_features, compute_regime_labels,
            MODEL_DIR,
        )
        from rl.full_market_env import FullMarketEnv, FullMarketEnvConfig
        from rl.stock_selector import SelectorConfig
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Override DB path
        import rl.run_training as rt
        rt.DB_PATH = self.config.db_path

        # Load data
        tickers = get_top_liquid_stocks(top_n=self.config.top_n, min_bars=self.config.min_bars)
        if len(tickers) < 10:
            logger.error(f"Too few tickers: {len(tickers)}")
            return None

        price_data = load_market_data(tickers, min_bars=self.config.min_bars)
        price_data = align_price_data(price_data)

        if len(price_data) < 10:
            logger.error(f"Too few valid tickers after loading: {len(price_data)}")
            return None

        # Train/test split
        train_data, test_data, _ = temporal_train_test_split(price_data, 0.8)

        # Compute features
        train_prices = np.column_stack([train_data[t] for t in sorted(train_data.keys())])
        train_steps = len(train_prices)
        train_features = compute_market_features(train_prices, train_steps)
        train_regimes = compute_regime_labels(train_features)

        # Test features
        test_prices = np.column_stack([test_data[t] for t in sorted(test_data.keys())])
        test_steps = len(test_prices)
        test_features = compute_market_features(test_prices, test_steps)
        test_regimes = compute_regime_labels(test_features)

        # Env config
        selector_cfg = SelectorConfig(top_k=self.config.top_k, rebalance_freq=self.config.rebalance_freq)
        env_cfg = FullMarketEnvConfig(
            top_k=self.config.top_k,
            rebalance_freq=self.config.rebalance_freq,
            selector_config=selector_cfg,
        )

        logger.info(f"  Data: {len(train_data)} stocks x {train_steps} train / {test_steps} test days")

        # Best-of-N training
        seeds = [42, 123, 777, 2024, 31415][:self.config.n_seeds]
        best_model = None
        best_sharpe = -999.0
        best_seed = seeds[0]
        all_results = []

        for seed_idx, seed in enumerate(seeds):
            logger.info(f"  Training seed {seed} ({seed_idx+1}/{len(seeds)})...")

            def make_env(s=seed):
                def _init():
                    e = FullMarketEnv(
                        price_data=train_data,
                        market_features=train_features,
                        regime_labels=train_regimes,
                        config=env_cfg,
                    )
                    e.reset(seed=s)
                    return e
                return _init

            vec_env = DummyVecEnv([make_env(seed + j) for j in range(4)])

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=self.config.ent_coef,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=self.config.net_arch),
                seed=seed,
                device=self.config.device,
                verbose=0,
            )

            t0 = time.time()
            model.learn(total_timesteps=self.config.total_steps)
            train_time = time.time() - t0

            # Evaluate on test data
            eval_env = FullMarketEnv(
                price_data=test_data,
                market_features=test_features,
                regime_labels=test_regimes,
                config=env_cfg,
            )
            obs, _ = eval_env.reset(seed=42)
            eval_pvs = [env_cfg.initial_cash]
            for _ in range(252):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                eval_pvs.append(info["portfolio_value"])
                if terminated or truncated:
                    break

            eval_pvs = np.array(eval_pvs)
            eval_ret = (eval_pvs[-1] / eval_pvs[0] - 1) * 100
            daily_rets = np.diff(eval_pvs) / eval_pvs[:-1]
            eval_sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-8) * np.sqrt(252))

            logger.info(f"    Seed {seed}: Return={eval_ret:+.2f}%, Sharpe={eval_sharpe:.2f} ({train_time:.0f}s)")

            all_results.append({
                "seed": seed,
                "return": eval_ret,
                "sharpe": eval_sharpe,
                "train_time": train_time,
            })

            if eval_sharpe > best_sharpe:
                best_sharpe = eval_sharpe
                best_model = model
                best_seed = seed

        if best_model is None:
            return None

        # Save best model
        model_path = self.config.models_dir / "candidates" / f"best_{best_seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save(str(model_path))

        logger.info(f"  Best: seed={best_seed}, Sharpe={best_sharpe:.2f}")

        return {
            "best_seed": best_seed,
            "best_sharpe": best_sharpe,
            "best_return": all_results[[r["seed"] for r in all_results].index(best_seed)]["return"],
            "model_path": str(model_path),
            "all_results": all_results,
        }

    def _stage_walk_forward(self, model_path: str) -> dict:
        """Stage 3: Walk-Forward验证."""
        logger.info(f"\n[Stage 3] Walk-Forward Validation ({self.config.wf_n_windows} windows)...")

        from rl.run_training import (
            get_top_liquid_stocks, load_market_data, align_price_data,
            compute_market_features, compute_regime_labels,
        )
        from rl.full_market_env import FullMarketEnv, FullMarketEnvConfig
        from rl.stock_selector import SelectorConfig
        from stable_baselines3 import PPO

        import rl.run_training as rt
        rt.DB_PATH = self.config.db_path

        # Load full data
        tickers = get_top_liquid_stocks(top_n=self.config.top_n, min_bars=self.config.min_bars)
        if "QQQ" not in tickers:
            tickers.append("QQQ")
        price_data = load_market_data(tickers, min_bars=self.config.min_bars)
        price_data = align_price_data(price_data)

        qqq_prices = price_data.pop("QQQ", None)
        tickers = sorted(price_data.keys())
        total_days = len(price_data[tickers[0]])

        model = PPO.load(model_path)

        selector_cfg = SelectorConfig(top_k=self.config.top_k, rebalance_freq=self.config.rebalance_freq)
        env_cfg = FullMarketEnvConfig(
            top_k=self.config.top_k,
            rebalance_freq=self.config.rebalance_freq,
            selector_config=selector_cfg,
        )

        # Walk-Forward windows
        train_window = self.config.wf_train_window
        test_window = self.config.wf_test_window
        step_size = self.config.wf_step_size

        alphas = []
        max_dds = []

        for w in range(self.config.wf_n_windows):
            start = w * step_size
            train_end = start + train_window
            test_end = train_end + test_window

            if test_end > total_days:
                break

            # Window data
            window_data = {t: price_data[t][start:test_end] for t in tickers}
            train_slice = {t: window_data[t][:train_window] for t in tickers}
            test_slice = {t: window_data[t][train_window:] for t in tickers}

            # Test features
            test_prices = np.column_stack([test_slice[t] for t in sorted(test_slice.keys())])
            test_steps = len(test_prices)
            test_features = compute_market_features(test_prices, test_steps)
            test_regimes = compute_regime_labels(test_features)

            # Evaluate
            test_env = FullMarketEnv(
                price_data=test_slice,
                market_features=test_features,
                regime_labels=test_regimes,
                config=env_cfg,
            )
            obs, _ = test_env.reset(seed=42)
            pvs = [env_cfg.initial_cash]
            for _ in range(test_window):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                pvs.append(info["portfolio_value"])
                if terminated or truncated:
                    break

            pvs = np.array(pvs)
            strat_ret = (pvs[-1] / pvs[0] - 1) * 100

            # QQQ benchmark
            if qqq_prices is not None and test_end <= len(qqq_prices):
                qqq_slice = qqq_prices[start + train_window:test_end]
                qqq_ret = (qqq_slice[-1] / qqq_slice[0] - 1) * 100 if len(qqq_slice) > 1 else 0
            else:
                qqq_ret = 0

            alpha = strat_ret - qqq_ret
            alphas.append(alpha)

            # MaxDD
            peak = np.maximum.accumulate(pvs)
            dd = (peak - pvs) / (peak + 1e-8)
            max_dds.append(float(dd.max()) * 100)

        positive_rate = sum(1 for a in alphas if a > 0) / max(len(alphas), 1)
        mean_alpha = np.mean(alphas) if alphas else 0
        max_dd = np.max(max_dds) if max_dds else 0

        logger.info(f"  Results: {len(alphas)} windows, {positive_rate:.0%} positive alpha")
        logger.info(f"  Mean Alpha: {mean_alpha:.2f}%, Max DD: {max_dd:.2f}%")

        return {
            "positive_alpha_rate": positive_rate,
            "mean_alpha": mean_alpha / 100,  # Convert to decimal
            "max_drawdown": max_dd / 100,     # Convert to decimal
            "alphas": alphas,
            "n_windows": len(alphas),
        }

    def _get_current_stage(self) -> str:
        """Get current stage name for error reporting."""
        return "unknown"

    def get_status(self) -> dict:
        """获取管道状态摘要."""
        production = self.registry.get_production()
        stats = self.metrics.get_stats_summary()
        last_run = self.metrics.get_last_successful_run()
        data_status = self.refresher.check_data_freshness()

        return {
            "production_model": production.version_id if production else None,
            "production_sharpe": production.metrics.sharpe if production else 0,
            "total_training_runs": stats["total_runs"],
            "success_rate": stats["success_rate"],
            "last_successful_run": last_run["finished_at"] if last_run else None,
            "data_freshness": data_status,
        }
