#!/usr/bin/env python3
"""Deploy Qwen3.8-27B on a single NVIDIA GPU with automatic layer-offload tuning.

Computes the optimal -ngl (layers kept on GPU) for a given quant profile based on
actual free VRAM, reserves KV-cache space for the requested context length, and
launches llama-server. Pure-text by default; pass --vision to load the mmproj.
"""

import argparse
import os
import shlex
import subprocess
import sys

N_LAYERS = 64
FULL_ATTN_LAYERS = 16
KV_HEADS = 4
HEAD_DIM = 256
BYTES_PER_ELEM = 2

BASE_URL = "https://hf-mirror.com/unsloth/Qwen3.8-27B-GGUF/resolve/main"

QUANT_PROFILES = {
    "Q4_K_M": {"file": "Qwen3.8-27B-UD-Q4_K_M.gguf", "size_gb": 16.5},
    "Q4_K_XL": {"file": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "size_gb": 17.5},
    "Q5_K_M": {"file": "Qwen3.8-27B-UD-Q5_K_M.gguf", "size_gb": 19.5},
    "Q6_K": {"file": "Qwen3.8-27B-UD-Q6_K.gguf", "size_gb": 22.5},
    "Q8_0": {"file": "Qwen3.8-27B-UD-Q8_K_L.gguf", "size_gb": 28.0},
}

MMPROJ = "mmproj-F16.gguf"


def get_vram_bytes():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f"[error] 无法读取 GPU 信息: {e}")

    total, used = [float(x) for x in out.strip().splitlines()[0].split(", ")]
    total_mb = total * 1024 * 1024
    used_mb = used * 1024 * 1024
    free_mb = total_mb - used_mb
    return total_mb, used_mb, free_mb


def kv_cache_bytes(context):
    per_token = FULL_ATTN_LAYERS * 2 * KV_HEADS * HEAD_DIM * BYTES_PER_ELEM
    return per_token * context


def estimate_ngl(file_bytes, free_vram, reserve):
    per_layer = file_bytes / N_LAYERS
    usable = free_vram - reserve
    ngl = int(usable / per_layer) if per_layer > 0 else 0
    return max(0, min(N_LAYERS, ngl))


def resolve_file_size(quant, path):
    if path and os.path.exists(path):
        return os.path.getsize(path)
    return int(QUANT_PROFILES[quant]["size_gb"] * 1024**3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-q", "--quant", default="Q4_K_M", choices=list(QUANT_PROFILES),
        help="量化档位",
    )
    p.add_argument("-m", "--model", help="本地 GGUF 路径（跳过下载）")
    p.add_argument("-c", "--context", type=int, default=32768, help="上下文长度")
    p.add_argument(
        "--reserve", type=float, default=None,
        help="额外预留显存 GB（默认按上下文自动算 KV cache）",
    )
    p.add_argument("--vision", action="store_true", help="加载视觉投影器 mmproj")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--server", default="llama-server", help="llama-server 可执行路径")
    p.add_argument("--outdir", default="models", help="下载目录")
    p.add_argument("--download", action="store_true", help="下载缺失的 GGUF 文件")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    total, used, free = get_vram_bytes()
    print(
        f"[gpu] 总显存 {total/1024**3:.1f}GB / 已用 {used/1024**3:.1f}GB "
        f"/ 可用 {free/1024**3:.1f}GB"
    )

    gguf = args.model
    if not gguf:
        gguf = os.path.join(args.outdir, QUANT_PROFILES[args.quant]["file"])
    if not os.path.exists(gguf):
        if args.download:
            os.makedirs(os.path.dirname(gguf) or ".", exist_ok=True)
            url = f"{BASE_URL}/{QUANT_PROFILES[args.quant]['file']}"
            print(f"[dl] {url}")
            if not args.dry_run:
                subprocess.run(["wget", "-c", "-O", gguf, url], check=True)
        else:
            sys.exit(
                f"[error] 模型不存在: {gguf}\n"
                f"        用 --download 下载，或 --model 指定本地路径"
            )

    file_bytes = resolve_file_size(args.quant, gguf)

    reserve = args.reserve * 1024**3 if args.reserve else kv_cache_bytes(args.context)
    if args.vision:
        reserve += int(2 * 1024**3)

    ngl = estimate_ngl(file_bytes, free, reserve)
    on_gpu = file_bytes * ngl / N_LAYERS

    print(f"[model] {gguf}")
    print(f"[model] 量化大小 {file_bytes/1024**3:.1f}GB, 层数 {N_LAYERS}")
    print(f"[mem] KV/预留 {reserve/1024**3:.1f}GB (context={args.context})")
    print(
        f"[offload] -ngl {ngl} -> GPU 权重 {on_gpu/1024**3:.1f}GB, "
        f"内存 {(file_bytes-on_gpu)/1024**3:.1f}GB"
    )

    cmd = [
        args.server,
        "-m", gguf,
        "-ngl", str(ngl),
        "-c", str(args.context),
        "--host", args.host,
        "--port", str(args.port),
        "--temp", str(args.temp),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
    ]
    if args.vision:
        mmproj = os.path.join(os.path.dirname(gguf) or ".", MMPROJ)
        if not os.path.exists(mmproj) and args.download:
            url = f"{BASE_URL}/{MMPROJ}"
            print(f"[dl] {url}")
            if not args.dry_run:
                subprocess.run(["wget", "-c", "-O", mmproj, url], check=True)
        if os.path.exists(mmproj):
            cmd += ["--mmproj", mmproj]
        else:
            sys.exit(f"[error] 视觉投影器不存在: {mmproj}")

    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return
    os.execvp(args.server, cmd)


if __name__ == "__main__":
    main()
