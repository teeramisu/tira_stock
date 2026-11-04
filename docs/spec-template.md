# Spec 派生测试模板

> 问题：agent 写的测试容易「断言实现」—— 把当前行为抄进断言，改一行代码测试就红，
> 但业务上其实没坏；反过来，业务真的错了，测试却是绿的。
>
> 解法：**先写验收标准（spec），再写测试，最后写实现**。测试断言的是标准，
> 不是代码现在返回什么。

## 一、什么时候需要写 spec

| 场景 | 需要 spec |
|---|---|
| 新增评分/排序/阈值类逻辑 | ✅ 是（有明确的业务口径） |
| 改判定条件（比如「偏强」的门槛） | ✅ 是（口径变了要同步文档） |
| 纯重构、行为不变 | ❌ 不需要（用现有测试兜底） |
| 修 bug | ⚠️ 需要一条「复现条件」作为最小 spec |

## 二、模板

```markdown
# <模块> 验收标准

## 背景
一句话说明这个模块解决什么问题。

## 标准

| ID | 标准 | 判定方式 |
|---|---|---|
| SC-1 | 六个维度分数都在 0–100 | 边界值（-1e9 / 1e9）不越界 |
| SC-2 | tone 阈值：≥60 偏正、≤40 偏负、其余中性 | 构造分数刚好在 60 / 40 |
| SC-3 | 数据缺失时不崩溃，且该维度标「数据不足」 | 空 series |
| SC-4 | 增长强 + 流动性宽 + 估值低 → 判断为偏强 | 构造三条件同时成立 |
| SC-5 | ... | ... |

## 非目标（明确不做什么）
- 不预测点位、不给买卖建议
```

## 三、示例：宏观六维评分

见 `backend/tests/test_macro_spec.py` 与本文附录。

| ID | 标准 | 对应测试 |
|---|---|---|
| SC-1 | 六个信号分数都在 0–100，极端输入不越界 | `test_scores_are_clamped_to_0_100` |
| SC-2 | tone 阈值固定为 ≥60 / ≤40 | `test_tone_thresholds` |
| SC-3 | 数据缺失不崩溃，缺数据的维度标注「数据不足」 | `test_missing_data_degrades_gracefully` |
| SC-4 | 增长 ≥55 且流动性 ≥60 且估值 ≥55 → 「震荡偏强」 | `test_strong_conditions_give_positive_stance` |
| SC-5 | 增长 <40 且地产 <40 → 「震荡偏弱」 | `test_weak_growth_and_property_give_negative_stance` |
| SC-6 | catalysts / risks / watch 各不超过 4 条 | `test_outlook_lists_are_capped` |
| SC-7 | 必须带免责声明（合规） | `test_disclaimer_present` |
| SC-8 | `as_of` 取所有序列里最新的日期 | `test_as_of_is_latest_period` |

## 四、写法约定

1. **测试名带标准编号**：`test_scores_are_clamped_to_0_100`，docstring 写 `SC-1`；
2. **断言业务语言**，不复制实现里的公式（避免改系数就红）；
3. 边界值优先：0、负数、极大值、空数据 —— 这些是 agent 最容易漏的；
4. 一个标准一条测试，失败信息要能直接看出违反了哪条标准；
5. 标准变了就**同时改文档和测试**，提交信息里说明「口径调整」。

## 五、和 CI 的关系

- 新测试文件按 `test_<模块>*.py` 命名，自动进入对应分片（见 `docs/ci.md` 第二节）；
- 纯新增测试不触发受保护路径守卫（`deletions == 0`）；
- 覆盖率 ratchet 会因此上升，基线需要 `check_coverage.py --write` 刷新（人工确认）。

## 附录：宏观六维口径（当前实现）

| 维度 | 主要输入 | 说明 |
|---|---|---|
| growth | PMI、工业增加值、用电量、固投当月 | 50 为中性，PMI 每偏离 1 点 ±4 分 |
| inflation | CPI、PPI | CPI <0 记通缩压力；PPI 高于 CPI+3 记上游涨价 |
| liquidity | M1-M2 剪刀差、SHIBOR、LPR、M2、央行总资产同比 | 剪刀差 >0 加分 |
| property | 70 城新房/二手房价同比 | 新房同比 < -3 为深度磨底 |
| valuation | 全 A PE/PB 十年分位 | 分位越低分越高 |
| sentiment | 两融余额 20 日变化 | ±2% 为杠杆方向切换 |
