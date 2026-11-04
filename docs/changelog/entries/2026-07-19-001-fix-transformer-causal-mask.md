# [fix] Transformer 因果掩码修复

- **日期**：2026-07-19
- **版本**：v0.3.0
- **类型**：fix

## 背景

Transformer Encoder 未传递 causal mask，每个 position 可 attend 到所有 future positions，造成数据泄露。虽然不是逐 token 预测（只用 last token），但 pooled representation 仍可通过 attention 看到未来信息。

## 变更

- `backend/ml/model.py:148`：给 `self.encoder()` 传递 `nn.Transformer.generate_square_subsequent_mask()` 生成的因果掩码

## 相关文件

- `backend/ml/model.py`

## 验证

- 训练时 model 不再能 attend 到未来 token
