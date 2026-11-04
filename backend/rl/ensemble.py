"""
Ensemble Training for RL-MACS

核心思想：通过多seed训练 + 预测平均来降低方差，而非增加约束。

策略：
- 训练N个模型（不同random seed）
- 评估时取action均值
- 不改变reward/环境，只改训练随机性

优势：
- 降方差但不引入偏置（vs Phase 4的约束导致欠拟合）
- 每个模型独立学到略有不同的策略
- 平均后消除个别模型的过拟合噪声
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.full_market_env import FullMarketEnv, FullMarketEnvConfig
from rl.stock_selector import SelectorConfig
from rl.run_training import (
    get_top_liquid_stocks, load_market_data, align_price_data,
    compute_market_features, compute_regime_labels, MODEL_DIR
)

logger = logging.getLogger("rl.ensemble")


@dataclass
class EnsembleConfig:
    """Ensemble配置"""
    n_models: int = 3           # 模型数量
    seeds: list[int] = None     # 各模型seed
    n_steps: int = 120000       # 每个模型训练步数
    top_n: int = 1000           # 股票池
    top_k: int = 30             # RL选股数
    rebalance_freq: int = 5     # 再平衡频率
    min_bars: int = 500         # 最小数据长度
    train_ratio: float = 0.8    # 训练/测试分割

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 777, 2024, 31415][:self.n_models]


class EnsembleTrainer:
    """Ensemble训练器"""

    def __init__(self, config: EnsembleConfig, db_path: Path | None = None):
        self.config = config
        self.db_path = db_path
        self.models: list[PPO] = []
        self.model_paths: list[str] = []

    def train(self, train_data: dict[str, np.ndarray],
              market_features: np.ndarray,
              regime_labels: np.ndarray) -> list[str]:
        """训练N个模型"""
        selector_cfg = SelectorConfig(
            top_k=self.config.top_k,
            rebalance_freq=self.config.rebalance_freq,
        )
        env_cfg = FullMarketEnvConfig(
            top_k=self.config.top_k,
            rebalance_freq=self.config.rebalance_freq,
            selector_config=selector_cfg,
            enable_augmentation=False,  # Phase 5: 不用数据增强
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for i, seed in enumerate(self.config.seeds):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Model {i+1}/{self.config.n_models} (seed={seed})")
            logger.info(f"{'='*60}")

            def make_env(s=seed):
                def _init():
                    env = FullMarketEnv(
                        price_data=train_data,
                        market_features=market_features,
                        regime_labels=regime_labels,
                        config=env_cfg,
                    )
                    env.reset(seed=s)
                    return env
                return _init

            vec_env = DummyVecEnv([make_env(seed + j) for j in range(4)])

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[128, 64]),
                seed=seed,
                device="auto",
                verbose=0,
            )

            t0 = time.time()
            model.learn(total_timesteps=self.config.n_steps)
            elapsed = time.time() - t0

            model_path = str(MODEL_DIR / f"rl_macs_ensemble_{i}")
            model.save(model_path)
            self.models.append(model)
            self.model_paths.append(model_path)

            logger.info(f"  Model {i+1} trained in {elapsed:.1f}s → {model_path}")

        return self.model_paths

    def predict_ensemble(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Ensemble预测：取所有模型action的均值"""
        actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)

        # Average actions
        avg_action = np.mean(actions, axis=0)
        # Re-normalize to valid range
        avg_action = np.clip(avg_action, 0.0, 1.0)
        total = avg_action.sum()
        if total > 1.0:
            avg_action = avg_action / total

        return avg_action.astype(np.float32)

    def evaluate(self, test_data: dict[str, np.ndarray],
                 qqq_prices: np.ndarray | None = None) -> dict:
        """Ensemble评估"""
        tickers = sorted(test_data.keys())
        prices = np.column_stack([test_data[t] for t in tickers])
        n_steps = len(prices)

        features = compute_market_features(prices, n_steps)
        regimes = compute_regime_labels(features)

        selector_cfg = SelectorConfig(
            top_k=self.config.top_k,
            rebalance_freq=self.config.rebalance_freq,
        )
        env_cfg = FullMarketEnvConfig(
            top_k=self.config.top_k,
            rebalance_freq=self.config.rebalance_freq,
            selector_config=selector_cfg,
            enable_augmentation=False,
        )

        env = FullMarketEnv(
            price_data=test_data,
            market_features=features,
            regime_labels=regimes,
            config=env_cfg,
        )

        obs, _ = env.reset()
        pvs = [env.config.initial_cash]

        for _ in range(252):
            action = self.predict_ensemble(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            pvs.append(info['portfolio_value'])
            if terminated or truncated:
                break

        pvs = np.array(pvs)
        total_return = (pvs[-1] / pvs[0] - 1) * 100

        # Sharpe
        daily_ret = np.diff(pvs) / pvs[:-1]
        sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)

        # MaxDD
        peak = np.maximum.accumulate(pvs)
        dd = (peak - pvs) / peak
        maxdd = np.max(dd) * 100

        result = {
            'total_return': total_return,
            'sharpe': sharpe,
            'maxdd': maxdd,
            'n_days': len(pvs) - 1,
        }

        # QQQ comparison
        if qqq_prices is not None and len(qqq_prices) > 1:
            qqq_aligned = qqq_prices[:len(pvs)]
            qqq_return = (qqq_aligned[-1] / qqq_aligned[0] - 1) * 100
            result['qqq_return'] = qqq_return
            result['alpha'] = total_return - qqq_return

        return result


def run_ensemble_walkforward(
    price_data: dict[str, np.ndarray],
    qqq_prices: np.ndarray,
    config: EnsembleConfig,
    train_window: int = 252,
    test_window: int = 63,
    step_size: int = 63,
) -> list[dict]:
    """Ensemble + Walk-Forward组合验证"""
    tickers = sorted(price_data.keys())
    total_days = len(price_data[tickers[0]])

    logger.info(f"Ensemble Walk-Forward: {len(tickers)} stocks × {total_days} days")
    logger.info(f"Ensemble: {config.n_models} models, seeds={config.seeds}")
    logger.info(f"Windows: train={train_window}d, test={test_window}d, step={step_size}d")

    # Calculate windows
    windows = []
    start = 0
    while start + train_window + test_window <= total_days:
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = min(train_end + test_window, total_days)
        windows.append((train_start, train_end, test_start, test_end))
        start += step_size

    logger.info(f"Total windows: {len(windows)}")
    results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        logger.info(f"\n{'='*60}")
        logger.info(f"Window {i+1}/{len(windows)}: train[{train_start}:{train_end}] test[{test_start}:{test_end}]")

        # Split data
        train_data = {t: price_data[t][train_start:train_end] for t in tickers}
        test_data = {t: price_data[t][test_start:test_end] for t in tickers}

        # Compute features for training
        train_prices = np.column_stack([train_data[t] for t in tickers])
        train_features = compute_market_features(train_prices, len(train_prices))
        train_regimes = compute_regime_labels(train_features)

        # Train ensemble
        trainer = EnsembleTrainer(config)
        trainer.train(train_data, train_features, train_regimes)

        # Evaluate ensemble
        qqq_test = qqq_prices[test_start:test_end]
        eval_result = trainer.evaluate(test_data, qqq_test)
        eval_result['window_id'] = i
        eval_result['train_range'] = f"[{train_start}:{train_end}]"
        eval_result['test_range'] = f"[{test_start}:{test_end}]"
        results.append(eval_result)

        logger.info(f"  Result: Return={eval_result['total_return']:+.2f}% "
                   f"QQQ={eval_result.get('qqq_return', 0):+.2f}% "
                   f"Alpha={eval_result.get('alpha', 0):+.2f}% "
                   f"Sharpe={eval_result['sharpe']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ensemble Training for RL-MACS")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--n-models", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=80000)
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--step-size", type=int, default=63)
    parser.add_argument("--mode", choices=["single", "walkforward"], default="walkforward")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Load data
    import rl.run_training as rt
    if args.db_path:
        rt.DB_PATH = args.db_path

    tickers = get_top_liquid_stocks(top_n=args.top_n, min_bars=500)
    if 'QQQ' not in tickers:
        tickers.append('QQQ')

    price_data = load_market_data(tickers, min_bars=500)
    price_data = align_price_data(price_data)

    qqq_prices = price_data.pop('QQQ')

    config = EnsembleConfig(
        n_models=args.n_models,
        n_steps=args.n_steps,
        top_n=args.top_n,
    )

    if args.mode == "walkforward":
        results = run_ensemble_walkforward(
            price_data, qqq_prices, config,
            train_window=args.train_window,
            test_window=args.test_window,
            step_size=args.step_size,
        )

        # Summary
        alphas = [r.get('alpha', 0) for r in results]
        sharpes = [r['sharpe'] for r in results]
        returns = [r['total_return'] for r in results]
        maxdds = [r['maxdd'] for r in results]

        print("\n" + "=" * 60)
        print("ENSEMBLE WALK-FORWARD SUMMARY")
        print("=" * 60)
        print(f"  Models per window: {config.n_models}")
        print(f"  Seeds: {config.seeds}")
        print(f"  Windows: {len(results)}")
        print(f"  Positive Alpha: {sum(1 for a in alphas if a > 0)}/{len(alphas)} "
              f"({sum(1 for a in alphas if a > 0)/len(alphas)*100:.0f}%)")
        print(f"  Mean Alpha: {np.mean(alphas):+.2f}%")
        print(f"  Mean Return: {np.mean(returns):+.2f}%")
        print(f"  Mean Sharpe: {np.mean(sharpes):.2f}")
        print(f"  Mean MaxDD: {np.mean(maxdds):.2f}%")
        print("=" * 60)

        # Per-window detail
        print("\nPer-window results:")
        for r in results:
            print(f"  W{r['window_id']+1} {r['test_range']}: "
                  f"RL={r['total_return']:+.2f}% QQQ={r.get('qqq_return',0):+.2f}% "
                  f"α={r.get('alpha',0):+.2f}% Sharpe={r['sharpe']:.2f}")
    else:
        # Single train/test split
        from rl.run_training import temporal_train_test_split
        train_data, test_data, _ = temporal_train_test_split(price_data, 0.8)

        train_prices = np.column_stack([train_data[t] for t in sorted(train_data.keys())])
        train_features = compute_market_features(train_prices, len(train_prices))
        train_regimes = compute_regime_labels(train_features)

        trainer = EnsembleTrainer(config)
        trainer.train(train_data, train_features, train_regimes)

        qqq_test = qqq_prices[int(len(qqq_prices) * 0.8):]
        result = trainer.evaluate(test_data, qqq_test)

        print("\n" + "=" * 60)
        print("ENSEMBLE SINGLE SPLIT RESULT")
        print("=" * 60)
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        print("=" * 60)


if __name__ == "__main__":
    main()
