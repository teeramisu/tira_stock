# Playbook · feature（新增行为）

> 适用：新接口、新指标、新前端区块、新的判定逻辑。
> 步骤**原样抄进待办**，跳过要写原因。

## 步骤

1. **说清验收标准**（写下来，不是心里想）
   - 用户视角：什么输入 → 什么可见结果
   - 若涉及评分/排序/阈值，写成 `docs/spec-template.md` 里的 SC-xx 表
   - 判据：每条标准都能被一个测试或命令证伪

2. **找现有实现的位置**
   - `grep` 相关符号，确认是新增文件还是改现有函数
   - 记录要改的文件清单（预计 ≤ 5 个，超了就拆）

3. **先写测试**（spec 派生）
   - 位置：`backend/tests/test_<模块>_<主题>.py`（glob 会自动归入分片）
   - 断言业务口径，不复制实现公式
   - 判据：此时运行应**失败**（红），且失败信息说的是「功能不存在」

4. **最小实现**
   - 只写让测试通过所需的代码，不做「顺便优化」
   - 新增公开函数写 Google 风格 docstring（Args / Returns）

5. **本地自检**
   ```bash
   scripts/ci/verify.sh --full
   ```

6. **补文档**（若对外行为变化）
   - `README.md` 的 API / 功能表，或 `docs/` 下对应篇章
   - 新增接口必须写：路径、参数、返回字段含义

7. **前端改动加视觉证据**（若涉及 `frontend/`）
   ```bash
   python -m uvicorn main:app --app-dir backend/app --port 8000 &
   python scripts/ci/frontend_screenshots.py --out artifacts/screenshots
   ```
   - 桌面 + 移动端各看一眼，确认没塌

8. **开 PR**
   - 标题 `feat(<scope>): <描述>`
   - 描述四段：背景 / 改动 / 验证 / 回退
   - 回退节写清：revert 是否安全、有无数据或配置副作用

9. **等 CI 并合并**
   ```bash
   python scripts/agent/auto_merge.py --repo teeramisu/tira_stock --pr <N> --method squash --wait
   ```

## 完成判据

- [ ] 至少 1 个新测试，且「先红后绿」有记录
- [ ] `verify.sh --full` 全绿
- [ ] CI `CI 汇总` success
- [ ] PR 描述四段齐全
- [ ] 汇报里有「证据 / 未覆盖」

## 常见失误

| 失误 | 后果 |
|---|---|
| 先写实现再补测试 | 测试会「断言实现」，无法发现错误 |
| 新文件没按 `test_<模块>*.py` 命名 | glob 匹配不到任何分片 → CI 直接失败 |
| 改了 `backend/tests/` 里已有断言 | 触发守卫，需要人工放行（合理，但要有理由） |
| 一次加多个接口 | 体量/范围超限 |
