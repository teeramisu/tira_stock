# Playbook · ci-change（改 CI / 脚本 / 依赖 / 仓库配置）

> 适用：`.github/**`、`scripts/ci/**`、`scripts/agent/**`、`requirements*`、
> `pytest.ini`、`ruff.toml`、`.coveragerc`、`ci/**`。
>
> 铁律：**这类改动等于改判分标准**，必须能被人工识别与放行。

## 步骤

1. **说清「防的是什么问题」**
   - 这条检查要挡住哪种具体错误？举一个真实发生过的例子
   - 判据：能写出「如果没有它，会出现 X」

2. **先确认现有检查没覆盖**
   ```bash
   grep -rn "<关键词>" scripts/ci .github/workflows
   ```
   - 避免重复造一条已有规则

3. **实现检查（失败要可执行）**
   - 输出必须包含：**哪条规则、哪个文件、怎么修**
   - 非 0 退出码表示失败
   - 自测正反例：
     ```bash
     python scripts/ci/check_xxx.py --files <合规样例>
     python scripts/ci/check_xxx.py --files <违规样例>
     ```

4. **接到 CI 与本地镜像**
   - `.github/workflows/ci.yml` 加 job 或 step
   - `scripts/ci/verify.sh` 加同一行（**本地必须能复现**）
   - 更新 `docs/ci.md` 的分层表

5. **评估误报与成本**
   - 会不会因为上游数据变化而随机变红？（如漏洞库、网络）
   - 观察期内可先 `continue-on-error: true`，稳定后转阻断

6. **开 PR 并请求放行**
   - 标题 `ci(<scope>): <描述>`
   - 描述必须写：**防的是什么、正反例自测输出、误报风险**
   - 守卫会红 —— 这是设计，等人工加 `allow-guard-change`
   - **不要自己贴标签**（agent token 不应有 `Issues: write`）

7. **合并后验证真实效果**
   - 下一个自然 PR 上确认该检查按预期工作
   - 若是"放宽"类改动（如允许纯新增），确认没有放宽到危险路径

## 完成判据

- [ ] 检查有正反例自测记录
- [ ] 已接入 CI **且** 已接入 `verify.sh`
- [ ] `docs/ci.md` 更新
- [ ] 描述里写了误报风险与缓解方式
- [ ] 人工标签已加（或明确说明为何不需要）

## 禁止事项

| 禁止 | 原因 |
|---|---|
| 为了让某个 PR 变绿而放宽规则 | 这是「改考卷」，必须单独 PR + 理由 |
| 在业务 PR 里顺带改 CI | `pr-scope` 直接拒绝（规则 R2） |
| 只改 CI 不改 `verify.sh` | 本地与 CI 不一致，问题会以「只有 CI 红」的形式复发 |
| 用 `|| true` 吞掉失败 | 检查变成装饰 |
