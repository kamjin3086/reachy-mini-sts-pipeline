# 运行路径与离线启动

测试环境：AMD Strix Halo / Radeon 8060S，ROCm 7.13，TheRock PyTorch。  
当前只保留两条启动路径：稳定基础路径和高性能 Qwen3-TTS 路径。

## 路径选择

| 路径 | 命令 | 适合场景 | 代价 |
|---|---|---|---|
| 基础路径 | `./scripts/sts_start.sh` | 环境最简单、便于排障 | Qwen3-TTS 可能低于实时，句中偶发停顿 |
| 高性能路径 | `./scripts/sts_start_qwen3_openai_fastapi_flash.sh` | Qwen3-TTS 使用 flash-attn 服务路径 | 需要隔离 flash-attn venv；torch.compile 默认关闭，极限吞吐需显式打开 |

不再保留“FastAPI 但不启用 flash-attn”的启动入口。当前高性能路径默认启用 `flash_attention_2`；`torch.compile` 因冷启动成本较高，改为显式可选。

## 离线启动

第一次联网启动成功后，后续应能离线启动。启动脚本现在会优先使用本地缓存：

| 组件 | 本地缓存 |
|---|---|
| Paraformer STT | `/home/kamjin/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| Qwen3-TTS 0.6B | `/home/kamjin/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5` |
| Qwen3-TTS 1.7B | `/home/kamjin/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe` |
| NLTK data | `/home/kamjin/nltk_data` |
| STS 持久缓存根目录 | `/home/kamjin/apps/sts-cache` |
| Torch compile cache | `/home/kamjin/apps/sts-cache/torchinductor_${USER}` |
| Qwen3-TTS FastAPI repo | `/home/kamjin/apps/Qwen3-TTS-Openai-Fastapi` |
| Qwen3-TTS FastAPI home/config | `/home/kamjin/apps/sts-cache/qwen3_openai_fastapi_flash_home` |
| flash-attn 源码 checkout | `/home/kamjin/apps/sts-cache/src/flash-attention` |

本次修复的离线问题：

- `funasr.AutoModel(model="paraformer-zh")` 会访问 ModelScope API，即使模型已缓存。脚本现在传本地模型目录。
- `ParaformerSTTHandler` 原本会把带 `/` 的模型名截断成本名，导致本地路径失效。`scripts/patch_paraformer_live_transcription.py` 已修复。
- FunASR 启动更新检查会联网。Paraformer patch 现在传 `disable_update=True`。
- `speech-to-speech` 检查 NLTK tagger 的目录写错，导致每次尝试下载。`scripts/patch_sts_offline_startup.py` 已修复为本地 tagger 路径。
- 高性能脚本原本默认依赖 `/tmp/Qwen3-TTS-Openai-Fastapi`，`/tmp` 清理后会重新 clone。默认目录已改为 `/home/kamjin/apps/Qwen3-TTS-Openai-Fastapi`。
- FastAPI home、torch.compile cache、flash-attn checkout 和 benchmark 输出默认放到 `/home/kamjin/apps/sts-cache`，避免 `/tmp` 清理后重复预热或丢失记录。

## 高性能环境

高性能路径使用隔离环境 `/home/kamjin/apps/.venv-qwen3-fa`，不会修改基础 venv。

已验证组合：

| 组件 | 版本 |
|---|---|
| PyTorch | `2.10.0+rocm7.13.0a20260513` |
| HIP | `7.13.26183` |
| Triton | `3.6.0+rocm7.13.0a20260513` |
| flash-attn | `2.8.4` |
| amd-aiter | `0.0.0` |

检查环境：

```bash
./scripts/install_qwen3_flash_attn_env.sh
```

目标 venv 已存在时，该脚本默认只验证，不会重装。需要重建时显式执行：

```bash
REINSTALL=1 ./scripts/install_qwen3_flash_attn_env.sh
```

## 性能结论

当前设备上，原 `faster-qwen3-tts` 路径吞吐不稳定，短句也可能低于实时。高性能路径默认使用 `Qwen3-TTS-Openai-Fastapi + flash_attention_2`。`torch.compile` 默认关闭，因为当前 ROCm/gfx1151 上 `max-autotune` 首次预热可能持续数分钟，不适合作为日常启动默认值。

需要极限吞吐时可显式打开：

```bash
QWEN3_FASTAPI_USE_COMPILE=true \
QWEN3_FASTAPI_COMPILE_MODE=reduce-overhead \
./scripts/sts_start_qwen3_openai_fastapi_flash.sh
```

历史实测 `flash_attention_2 + torch.compile + CUDA graphs` 的 Qwen3-TTS bridge 约 `RTF=1.81-1.89`，`realtime debt=0.00s`，但冷启动体验较差。

首次冷启动慢是预期行为。若启用 `torch.compile` 或 CUDA graphs，更好的做法是保留 `TORCHINDUCTOR_CACHE_DIR` 缓存并让服务常驻。
