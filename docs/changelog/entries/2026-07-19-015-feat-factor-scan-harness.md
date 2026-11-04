# [feat] 因子扫描 Harness + 月度全因子扫描

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：feat

## 背景

现有系统缺一个系统化的因子扫描评估框架：FactorBacktestEngine 能评测所有因子但无人调度，结果不追踪，衰减不告警。

## 变更

- `backend/research/factor_scan.py`（新文件）：FactorScanHarness
  - `FactorScanner`：调度 FactorBacktestEngine 全因子扫描，结果持久化到 `data/factor_scan/`
  - `FactorTracker`：加载历史扫描记录，计算每个因子 IC 趋势、z-score、检测新兴/衰减因子
  - `FactorScanHarness.run_monthly_scan()`：全因子扫描 + AI 因子研究的一站式入口
- `backend/research/evolution_loop.py`：
  - `EvolutionRunner.run_monthly_cycle()`：月度自进化循环——日评估 + 因子扫描 + AI 研究
  - `CycleResult` 新增 `factor_scan` 字段，月度报告中包含因子级洞察

## 相关文件

- `backend/research/factor_scan.py`
- `backend/research/evolution_loop.py`

## 验证

- 新增 3 个测试：保存/加载、趋势计算、Harness 初始化
