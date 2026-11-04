# RL-MACS 改进路线图

> 基于2025-2026年最新10篇量化RL论文调研，结合当前系统回测诊断结果制定。
> 创建时间: 2026-07-25

## 当前系统状态

- **模型**: PPO (Stable-Baselines3), 500K步训练
- **样本内评估**: Return=39.65%, Sharpe=2.953, MaxDD=5.41%
- **已修复**: Regime检测阈值(CRASH从49%→2.2%), Risk Overlay参数
- **待验证**: 样本外(OOS)回测、过拟合检测

---

## P0 — 必须立即实现

### 1. CPPO升级：加入CVaR尾部风险约束
- **来源**: FinRL-DeepSeek (arXiv 2025.02)
- **原理**: 在PPO目标上加入Conditional Value-at-Risk惩罚项
  - 公式: `L_CPPO = L_PPO - λ * E[(η - D(π))+] / α`
  - η为回报阈值，α为置信水平(如0.05=最坏5%)，λ为拉格朗日乘子
- **关键发现**: 标准PPO+LLM风险信号=噪音；CPPO+LLM风险信号=有效（熊市大幅跑赢）
- **实现方案**:
  - [ ] 修改`backend/rl/trainer.py`，将PPO替换为自定义CPPO
  - [ ] 在rollout buffer中计算轨迹回报的CVaR
  - [ ] 加入拉格朗日乘子自适应调整（对偶梯度上升）
  - [ ] 回测对比PPO vs CPPO在熊市期间的表现
- **预期效果**: 熊市保护能力质变，MaxDD降低30-50%

### 2. 训练步数提升到200万步
- **来源**: FinRL-DeepSeek论文明确证明
  - 40-50万步: LLM注入效果不明显
  - 200万步: PPO和CPPO均跑赢纳斯达克100
- **实现方案**:
  - [ ] 修改训练配置: `main_steps=1_950_000`
  - [ ] 加入学习率衰减: 前100万步3e-4 → 后100万步1e-4
  - [ ] 加入early stopping: 连续5次eval Sharpe下降则停止
  - [ ] 预计训练时间: ~7分钟(FPS 4500, 8 envs)
- **预期效果**: 从"初步收敛"到"充分收敛"

### 3. Walk-Forward滚动训练
- **来源**: RL in Financial Decision Making (arXiv 2025.12) 综述强调
- **原理**: 避免单次train/test分割的偶然性
  - 窗口1: 训练2012-2018, 测试2019-2020
  - 窗口2: 训练2013-2019, 测试2020-2021
  - 窗口3: 训练2014-2020, 测试2021-2022
  - ...
- **实现方案**:
  - [ ] 在`backend/rl/run_backtest.py`中实现滚动窗口
  - [ ] 每个窗口独立训练+评估
  - [ ] 汇总所有OOS窗口的Sharpe/Return/MaxDD
  - [ ] 计算OOS/IS Sharpe比率（>0.5为可接受）
- **预期效果**: 可靠评估策略泛化能力，消除过拟合假象

---

## P1 — 重要改进

### 4. 异质Agent集合 + MAC动态权重
- **来源**: MARS (ICLR 2025 / AAAI 2026, Microsoft Research)
- **原理**:
  - 10个Agent各有不同风险偏好（保守/中性/激进）
  - 每个Agent有Actor/Critic/Safety-Critic三网络
  - MAC（元自适应控制器）根据市场状态动态分配权重
  - 聚合动作 = Σ(weight_i × action_i)
- **MARS关键结果**:
  - 熊市: -0.86% (基线-36.37%)
  - 牛市: 收益和Sharpe均超越所有DRL基线
  - 消融: 静态权重版本表现明显差；同质Agent也不如异质
- **实现方案**:
  - [ ] 扩展`backend/rl/decision_agent.py`为多Agent版本
  - [ ] 实现3-5个不同风险偏好的Agent（保守/中性/激进/动量/均值回归）
  - [ ] 实现MAC网络（输入:市场状态 → 输出:各Agent权重）
  - [ ] 修改`backend/rl/env.py`支持多Agent动作聚合
- **预期效果**: 不同市场状态自适应切换策略风格

### 5. LLM风险信号"强耦合"注入
- **来源**: FinRL-DeepSeek + GIFT (arXiv 2026.06)
- **原理**: 不是把LLM信号当特征输入（弱耦合），而是直接修改奖励和动作（强耦合）
  - 风险分 → 调整轨迹回报 D' = R_f × D（影响CVaR罚项）
  - 推荐分 → 调整动作强度 a' = S_f × a（策略偏置）
- **GIFT方法**: LLM不直接交易，而是设计学习接口
  - Factor-guided State Enhancement (FSE): LLM生成额外状态特征
  - Risk-rule-guided Reward Shaping (RRS): LLM构建辅助奖励
  - Diagnostic-guided Refinement (DGR): PPO rollout诊断→LLM修订
- **实现方案**:
  - [ ] 在SentimentAgent中接入LLM（可用本地小模型）
  - [ ] 实现风险分→奖励缩放机制
  - [ ] 实现推荐分→动作缩放机制
  - [ ] 仅在RiskOverlay层注入，不改变PPO核心逻辑
- **预期效果**: 风险感知能力提升，黑天鹅事件响应更快

### 6. 观测归一化一致性修复
- **来源**: 回测诊断发现的问题
- **问题**: 训练时TradingEnv使用`_obs_mean/_obs_std`归一化，推理时inference.py未做归一化
- **实现方案**:
  - [ ] 训练结束后保存obs_mean/obs_std到模型目录
  - [ ] 推理时加载并应用相同归一化
  - [ ] 或使用SB3的VecNormalize包装器统一管理
- **预期效果**: 消除训练/推理不一致导致的性能损失

---

## P2 — 中期优化

### 7. 预训练→微调范式
- **来源**: MADDQN (Expert Systems with Applications 2024)
- **原理**: 在多资产混合数据上预训练通用策略，再微调到具体标的
- **实现方案**:
  - [ ] 收集20+美股ETF/大盘股数据
  - [ ] Phase 1: 在混合数据上训练500K步（通用市场理解）
  - [ ] Phase 2: 在目标标的上微调100K步（特定资产适应）
  - [ ] 对比直接训练 vs 预训练+微调的OOS表现
- **预期效果**: 跨资产泛化能力提升，减少过拟合

### 8. 交易成本优化：死区过滤
- **来源**: 回测诊断（累计交易成本16.8%）
- **问题**: PPO连续动作导致每天微调，产生大量无意义交易
- **实现方案**:
  - [ ] 在RiskOverlay中加入死区: |action| < 0.05 时不执行
  - [ ] 增加换手率惩罚权重（reward中beta从0.1提到0.3）
  - [ ] 考虑离散化动作空间（持有/加仓/减仓/清仓）
- **预期效果**: 交易成本降低50%+

### 9. 辩论机制 + 记忆系统
- **来源**: TradingAgents (arXiv 2024.12, 93K Stars)
- **原理**:
  - 多头Agent vs 空头Agent辩论，决策者裁决
  - 短期记忆(近期交易效果) + 长期记忆(历史策略表现)
- **实现方案**:
  - [ ] FactorAgent拆分为BullFactorAgent + BearFactorAgent
  - [ ] 决策网络输入加入"辩论摘要"（多空观点差异）
  - [ ] 实现EpisodeMemory: 记录最近N次调仓的效果
  - [ ] 将记忆作为额外obs维度输入决策网络
- **预期效果**: 决策质量提升，避免单边偏见

---

## P3 — 长期探索

### 10. LLM自动设计奖励/特征
- **来源**: GIFT (arXiv 2026.06)
- **原理**: LLM作为"接口设计师"，自动生成:
  - 状态增强函数（从OHLCV生成技术指标）
  - 奖励塑形逻辑（风险规则→奖励公式）
  - 通过PPO rollout诊断迭代优化
- **实现方案**:
  - [ ] 设计奖励函数DSL（领域特定语言）
  - [ ] LLM生成候选奖励函数代码
  - [ ] 自动回测评估 → 选择最优 → 迭代
- **预期效果**: 减少手工调参，自动发现有效奖励结构

### 11. 方向+仓位分离决策
- **来源**: FINRS (arXiv 2025.11)
- **原理**: 将交易决策分为三层
  - 方向Agent: 决定买/卖/持有
  - 数量Agent: 基于Kelly+CVaR决定仓位大小
  - 风险Agent: 最终约束检查
- **实现方案**:
  - [ ] 修改动作空间为: 方向(离散) + 幅度(连续)
  - [ ] 或保持连续动作但加入分层解码
- **预期效果**: 更精细的仓位管理

### 12. 图注意力多Agent (Graph Attention MARL)
- **来源**: Agentic Trading综述 (arXiv 2026.05) 提及
- **原理**: 用图结构建模资产间关系，注意力机制动态调整Agent间通信
- **实现方案**: 待进一步调研具体实现
- **预期效果**: 捕捉资产间联动关系

---

## 论文索引

| # | 论文 | 来源 | 核心贡献 |
|---|------|------|---------|
| 1 | MARS | ICLR 2025 / AAAI 2026 | 异质Agent+MAC+Safety-Critic |
| 2 | GIFT | arXiv 2026.06 | LLM设计状态-奖励接口 |
| 3 | FinRL-DeepSeek | arXiv 2025.02 | CVaR-PPO + LLM双通道注入 |
| 4 | Trading-R1 | arXiv 2025.09 | LLM推理+GRPO+课程学习 |
| 5 | TradingAgents | arXiv 2024.12 | 多角色辩论+记忆系统 |
| 6 | RL in Quant Finance Survey | ACM Computing Surveys 2025 | 167篇系统综述 |
| 7 | MADDQN | Expert Systems 2024 | TimesNet+预训练微调 |
| 8 | FINRS | arXiv 2025.11 | 方向+数量+风险三层Agent |
| 9 | Agentic Trading Survey | arXiv 2026.05 | LLM Agent分类框架 |
| 10 | RL Financial Decision Survey | arXiv 2025.12 | 实施质量>算法复杂性 |

---

## 实施顺序建议

```
Phase 1 (本周): P0-1 CPPO + P0-2 200万步 + P0-3 Walk-Forward
Phase 2 (下周): P1-4 异质Agent + P1-6 归一化修复 + P2-8 死区过滤
Phase 3 (后续): P1-5 LLM注入 + P2-7 预训练 + P2-9 辩论机制
Phase 4 (探索): P3-10~12
```
