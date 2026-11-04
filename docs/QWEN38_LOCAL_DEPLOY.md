# Qwen3.8-27B 本地部署 + opencode 接入指南

RTX 4090 (24GB) + 125GB 内存，通过 llama.cpp 跑 Qwen3.8-27B（Q6_K 量化，全 GPU）。

## 1. 概览

| 项目 | 值 |
|---|---|
| 模型 | Qwen3.8-27B（Apache-2.0，27B 稠密，混合线性注意力，原生视觉语言） |
| 量化 | Q6_K（unsloth Dynamic V3.0），~20.46 GiB，接近无损 |
| 推理引擎 | llama.cpp（源码：`~/llama.cpp`，二进制：`~/llama.cpp/build/bin/llama-server`） |
| 模型文件 | `~/models/Qwen3.8-27B-UD-Q6_K.gguf` |
| API | OpenAI 兼容，`http://127.0.0.1:8080/v1` |
| 实测性能 | prefill ~1895 tok/s，decode ~24 tok/s（Q4_K_M 版 ~31 tok/s） |
| 显存 | 全 GPU（-ngl 64），占用 ~23.8GB；KV 溢出时自动落 125GB 内存 |

另有 Q4_K_M 档（`~/models/Qwen3.8-27B-UD-Q4_K_M.gguf`，15.3GB，更快），长上下文/极限速度场景可切换。

## 2. 启动 llama-server

### 方法 A：一键脚本（推荐，自动检测显存并计算层 offload）

```bash
# 部署脚本位于本仓库 scripts/deploy_qwen38.py
python3 scripts/deploy_qwen38.py \
  --model ~/models/Qwen3.8-27B-UD-Q6_K.gguf \
  --server ~/llama.cpp/build/bin/llama-server \
  -c 32768          # 上下文长度
```

脚本会打印检测到的显存与 `-ngl` 分配，然后 `exec` 启动 server（前台占用终端）。
想在后台常驻，用方法 B。

### 方法 B：手动后台常驻

```bash
setsid nohup ~/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen3.8-27B-UD-Q6_K.gguf \
  -ngl 64 \          # 64 层全放 GPU
  -c 32768 \         # 上下文 token 数
  --host 127.0.0.1 \
  --port 8080 \
  > /tmp/llama_q6.log 2>&1 &
```

查看日志：`tail -f /tmp/llama_q6.log`
停止：`pkill -f llama-server`

### 验证

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"thinking":{"type":"disabled"}}'
```

### 常用参数说明

| 参数 | 作用 |
|---|---|
| `-ngl N` | N 层放 GPU，其余自动放内存（CPU 计算）。`-ngl 64` = 全 GPU |
| `-c N` | 上下文长度。Q6_K 全 GPU 时显存很满，256K 超长上下文会让 KV 落内存（有 125GB 兜底，略慢） |
| `-t N` | CPU 线程数（默认自动） |
| `--mmproj file` | 加视觉投影器可读图（当前未装，需要时从 unsloth 仓库下 `mmproj-F16.gguf`） |

## 3. 在 opencode 中调用

### 3.1 注册自定义 provider

编辑配置（推荐**全局**配置，任何项目都能用；也可放单个项目根目录的 `opencode.json`）：

- 全局：`~/.config/opencode/opencode.json`
- 项目：`<项目根>/opencode.json`

在 `provider` 节点下新增一个本地 provider（保留你已有的其他 provider/字段，只追加）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    // ...你原有的 provider 保持不变...

    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama.cpp",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"              // llama-server 不校验 key，任意占位即可
      },
      "models": {
        "qwen3.8-q6k": {
          "name": "Qwen3.8-27B Q6_K",
          "reasoning": true,                    // 支持 thinking 模式
          "tool_call": true,                    // 支持工具调用
          "interleaved": { "field": "reasoning_content" }  // 思考内容在 reasoning_content 字段
        }
      }
    }
  },
  // 可选：设为默认模型（想继续用其它默认模型就不要加这一行，用 /models 切换）
  "model": "local/qwen3.8-q6k"
}
```

### 3.2 重启 opencode

opencode 只在启动时加载配置，改完必须**退出并重启 opencode** 才生效。

### 3.3 选择模型

重启后：
- 直接设为默认模型，或在会话里输入 `/models`，找到 `Local llama.cpp` → `Qwen3.8-27B Q6_K` 选中即可。

### 3.4 冒烟测试

让 opencode 干一件本地小任务（比如写个函数），观察是否走本地模型。确认方法：
- server 日志有请求记录；
- 回复首 token 慢、随后连续输出（本地 decode ~24 tok/s，比云端慢，属正常）。

## 4. 档位切换（Q6_K ↔ Q4_K_M）

| 场景 | 建议 |
|---|---|
| 日常写代码/工具调用 | Q6_K（当前，接近无损） |
| 极限速度、超长上下文(≥128K) | Q4_K_M（15.3GB，KV 余量大，decode ~31 tok/s） |

```bash
# 换成 Q4_K_M：停掉旧 server 再启动
pkill -f llama-server
setsid nohup ~/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 64 -c 32768 --host 127.0.0.1 --port 8080 \
  > /tmp/llama_q4.log 2>&1 &
```

两种量化下 opencode 的 provider 配置无需改动（同一端口），只有模型精度/速度不同。

## 5. 常见问题与调参

### 思考模式（Thinking）
- 模型**默认开启思考**，编码/复杂任务时是优点。
- llama.cpp 每个请求可控制：`"thinking": {"type": "disabled"}` 关闭，或
  `{"type":"enabled","budget_tokens":2000}` 限制思考预算。
- server 也可全局开关：启动加 `--reasoning off`。

### 请求超时
本地大模型首 token 较慢，且默认开启思考，长任务 token 数多。若 opencode 报超时，可在 provider 的 `options` 加大：
```jsonc
"options": {
  "baseURL": "http://127.0.0.1:8080/v1",
  "apiKey": "local",
  "timeout": false,           // 禁用请求超时（需要时）
  "headerTimeout": 600000     // 首 token 等待上限（毫秒）
}
```

### 显存不足 / CUDA OOM
Q6_K 全 GPU 已接近 24GB 上限。若同时跑其它占显存程序，减少层数让尾部层落内存：
```bash
-ngl 58    # 留出 ~6 层到内存，释放约 2GB 显存
```

### 换机器 / 换端口
改两处：启动命令的 `--host/--port` 与 opencode.json 的 `options.baseURL`，保持一致即可。

## 6. 相关文件位置

| 文件 | 路径 |
|---|---|
| llama.cpp 源码 | `~/llama.cpp` |
| llama-server 二进制 | `~/llama.cpp/build/bin/llama-server` |
| Q6_K 模型 | `~/models/Qwen3.8-27B-UD-Q6_K.gguf` |
| Q4_K_M 模型 | `~/models/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| 部署脚本（自动算 -ngl） | `scripts/deploy_qwen38.py`（本仓库） |
| server 运行日志 | `/tmp/llama_q6.log` |
