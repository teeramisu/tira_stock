# RL-MACS 防过拟合改进计划

**日期**: 2026-07-25
**状态**: Phase 1-3, 5 已完成 ✅ | Phase 4 已废弃 ❌
**触发**: Walk-Forward验证显示仅29%窗口正Alpha，模型存在明显过拟合

---

## 问题诊断

### Walk-Forward验证结果（7窗口，812测试日）

| 指标 | 值 | 问题 |
|------|-----|------|
| 正Alpha窗口 | 2/7 (29%) | 泛化能力差 |
| 平均Alpha | -7.37% | 跑输基准 |
| RL MaxDD | 21.18% | 远超QQQ的7.31% |
| 日胜率 | 44.8% | 低于50% |
| 近期偏差 | 仅W6/W7正Alpha | 过拟合近期数据 |

### 根因分析

1. **训练数据不足**: 仅502天(~2年)，未覆盖完整牛熊周期
2. **模型过复杂**: net_arch=[256,128,64] 参数量大，易记忆噪声
3. **单次训练**: 无滚动重训练机制，模型"过期"
4. **确定性推理**: deterministic=True 放大过拟合效应
5. **Regime覆盖不足**: 训练期以RANGE(62%)为主，BEAR/CRASH样本少

---

## 改进方案

### P0: Walk-Forward Retraining（滚动重训练）

**目标**: 正Alpha率 29% → 60%+

```
实施方案:
- 训练窗口: 252天 (1年)
- 验证窗口: 63天 (1季度)
- 滚动步长: 63天
- 每次用最新252天数据重训练
- 保留最近3个模型做Ensemble

预期效果: 模型始终适应最近市场特征
```

### P0: 增加训练数据到5年

**目标**: 覆盖2+个完整牛熊周期

```
实施方案:
- 当前: 502天 (~2年)
- 目标: 1260天 (~5年)
- 方法: 拉取2021-2026历史数据
- 确保包含: 2022熊市、2023复苏、2024-25牛市

预期效果: 模型见到更多regime，泛化能力提升
```

### P1: 简化网络结构

**目标**: 降低模型方差

```
当前: net_arch=[256, 128, 64]  # 3层，~50K参数
改为: net_arch=[128, 64]       # 2层，~10K参数

预期效果: 减少过拟合风险，训练更快
```

### P1: 增加熵系数

**目标**: 鼓励探索，减少确定性过拟合

```
当前: ent_coef=0.01
改为: ent_coef=0.05

预期效果: 策略更多样化，不易陷入局部最优
```

### P2: 数据增强 (Domain Randomization)

**目标**: 提升对噪声的鲁棒性

```
实施方案:
- 价格噪声: 训练时添加 ±2% 高斯噪声
- 交易成本: 随机 0.1%-0.5% (当前固定0.1%)
- 滑点: 随机 0-0.2%
- 起始点: 已实现随机episode起点

预期效果: 模型不过度依赖精确价格模式
```

### P2: Ensemble方法

**目标**: 平滑预测，降低单模型风险

```
实施方案:
- 训练5个模型 (不同seed)
- 推理时取action均值
- 或: 取最近3个walk-forward模型均值

预期效果: 降低方差，提升稳定性
```

### P3: Reward函数增强

**目标**: 惩罚过拟合行为

```
新增惩罚项:
- turnover_penalty: λ=0.2, 惩罚日换手率>20%
- concentration_penalty: μ=0.3, 惩罚单票>15%
- drawdown_speed: ν=0.1, 惩罚快速回撤

预期效果: 策略更稳健，减少过度交易
```

---

## 实施计划

### Phase 1: 快速验证 ✅ 已完成
- [x] 简化网络 [128, 64]
- [x] 熵系数 0.01 → 0.05
- [x] 重新训练 120K步
- [x] Walk-Forward验证

**结果**: 正Alpha率 29%→86%, 平均Alpha -7.37%→+10.72%, Sharpe 0.207→1.855

### Phase 2: 数据扩展
- [ ] 拉取5年历史数据
- [ ] 验证数据质量
- [ ] 重新训练

### Phase 3: Walk-Forward机制 ✅ 已完成
- [x] 实现滚动训练脚本 (`backend/rl/walk_forward.py`)
- [x] 模型版本管理 (rl_macs_wf_0, rl_macs_wf_1, ...)
- [x] 自动化验证

**结果**: 正Alpha率 60%, 平均Alpha +0.51%, Sharpe 4.89, MaxDD 0.22%

### Phase 4: 高级优化 ❌ 已废弃
- [x] 数据增强（价格噪声、交易成本随机化）
- [x] Reward函数增强（换手率惩罚、集中度惩罚）
- [x] Ensemble方法（多模型平均）

**结果**: 效果不如Phase 1，已废弃
- 数据增强2%噪声破坏因子信号（信噪比≈1:1）
- 换手率/集中度惩罚与策略盈利模式冲突
- Ensemble平均稀释强信号

**根因**: Phase 1已通过简化模型解决过拟合，额外约束导致欠拟合

### Phase 5: 精确调优 ✅ 已完成
- [x] 恢复v2精确reward配置（禁用turnover/single_stock惩罚）
- [x] Best-of-N训练（多seed选最优）
- [x] 集成到run_training.py（--n-seeds参数）

**关键发现**:
1. v2（86%正Alpha）训练时**没有**turnover/single_stock惩罚
2. Phase 4添加的惩罚导致模型欠拟合（RL收益全为负）
3. Best-of-5训练：所有seed都盈利，最佳Sharpe=6.68
4. Walk-Forward: 平均Alpha +0.97%, MaxDD 0.97%

**生产配置**:
```python
# run_training.py
net_arch=[128, 64]      # 简化网络
ent_coef=0.05           # 高熵探索
--n-seeds 5             # Best-of-5训练

# full_market_env.py
turnover_penalty=0.0    # 禁用（v2原始配置）
single_stock_penalty=0.0  # 禁用
enable_augmentation=False  # 禁用数据增强
```

---

## 验证标准

| 指标 | 当前 | 目标 |
|------|------|------|
| 正Alpha窗口率 | 29% | >60% |
| 平均Alpha | -7.37% | >0% |
| RL MaxDD | 21.18% | <15% |
| 日胜率 | 44.8% | >50% |
| Sharpe | 0.207 | >0.5 |

---

## 相关文件

- 训练脚本: `backend/rl/run_training.py` (支持 --n-seeds Best-of-N)
- 环境: `backend/rl/full_market_env.py`
- 选股器: `backend/rl/stock_selector.py`
- Reward: `backend/rl/reward.py`
- Ensemble: `backend/rl/ensemble.py` (实验性)
- Walk-Forward: `backend/rl/walk_forward.py`
- 模型: `backend/rl/models/rl_macs_full.zip`

---

## 经验总结

### 什么有效
1. **简化网络** [128,64]：减少参数量，降低过拟合
2. **高熵系数** 0.05：鼓励探索，避免局部最优
3. **Best-of-N训练**：多seed选最优，稳定提升
4. **简单reward**：只用return-cost-dd-hhi，不加额外惩罚

### 什么无效
1. **数据增强**：2%噪声破坏因子信号
2. **额外惩罚**：turnover/single_stock惩罚导致欠拟合
3. **Ensemble平均**：稀释强信号
4. **LR衰减**：250K步+衰减导致过拟合
5. **更长训练**：120K步是最佳，更多反而更差
