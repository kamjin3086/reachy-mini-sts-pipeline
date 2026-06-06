# 脚本索引与使用约定

> 返回：[README.zh.md](../README.zh.md) · 运行路径：[docs/06](06-runtime-paths-and-offline.zh.md)

这个目录只放“可执行工具”。临时日志、模型缓存、虚拟环境和编译缓存不要放进 `scripts/`。

## 最常用入口

| 目标 | 命令 | 说明 |
|---|---|---|
| 查看常用命令 | `make help` | 轻量入口，不替代脚本文档 |
| 启动稳定路径 | `make start` 或 `./scripts/sts_start.sh` | 默认 WebSocket：`ws://0.0.0.0:8765/v1/realtime` |
| 启动高性能路径 | `make start-fast` 或 `./scripts/sts_start_qwen3_openai_fastapi_flash.sh` | 使用隔离 venv + Qwen3-TTS FastAPI + flash-attn |
| 跑测试 | `make test` | 只验证仓库内 patch 脚本行为，不会启动完整 STS |
| 清理本地生成物 | `make clean-local` | 删除 `scripts/__pycache__`、`tests/__pycache__` 和 `scripts/log_*.txt` |

## 启动脚本

| 脚本 | 用途 | 适合场景 |
|---|---|---|
| `scripts/sts_start.sh` | 稳定 STS 路径：Paraformer + 本地 OpenAI 兼容 LLM + Qwen3-TTS | 日常运行、排障、给别人复现 |
| `scripts/sts_start_qwen3_openai_fastapi_flash.sh` | 高性能 TTS 路径：额外启动 Qwen3-TTS OpenAI FastAPI bridge | TTS 吞吐优先，能接受预热和环境复杂度 |

两条启动路径都支持这些常用环境变量：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `STS_LLM_BASE_URL` | `http://127.0.0.1:8101/v1` | 本地 LLM OpenAI 兼容端点 |
| `STS_LLM_MODEL` | `Gemma-4-E4B-instruct` | LLM server 中注册的模型名 |
| `STS_WS_HOST` | `0.0.0.0` | Realtime WebSocket 监听地址 |
| `STS_WS_PORT` | `8765` | Realtime WebSocket 监听端口 |
| `STS_KILL_PORT` | `0` | 设为 `1` 时，启动前杀掉占用端口的进程 |
| `QWEN3_TTS_MODEL_NAME` | 优先本地缓存，否则 HF 模型名 | Qwen3-TTS 模型路径或模型名 |
| `PARAFORMER_STT_MODEL_NAME` | 优先本地缓存，否则 `paraformer-zh` | Paraformer 模型路径或模型名 |
| `INIT_CHAT_PROMPT` | Reachy Mini 中文短回复 prompt | 覆盖系统提示词 |
| `STS_CACHE_DIR` | `/home/kamjin/apps/sts-cache` | 高性能路径的持久缓存根目录 |
| `TORCHINDUCTOR_CACHE_DIR` | `$STS_CACHE_DIR/torchinductor_${USER}` | torch.compile 缓存目录 |
| `QWEN3_OPENAI_FASTAPI_HOME` | `$STS_CACHE_DIR/qwen3_openai_fastapi_flash_home` | FastAPI 配置与运行 home |
| `QWEN3_FASTAPI_USE_COMPILE` | `false` | 是否启用 torch.compile；默认关闭以避免数分钟冷启动 |
| `QWEN3_FASTAPI_COMPILE_MODE` | `reduce-overhead` | 开启 compile 时的模式 |
| `QWEN3_FASTAPI_USE_CUDA_GRAPHS` | `false` | 是否额外启用手动 CUDA graphs |
| `QWEN3_FASTAPI_COMPILE_CODEBOOK` | `false` | 是否 compile codebook predictor |
| `STS_PYTHON` | `/home/kamjin/apps/.venv-qwen3-fa/bin/python3` | 高性能路径和 `make verify-flash` 使用的 Python |

示例：

```bash
STS_LLM_MODEL=Qwen3.6-35B-A3B \
STS_WS_PORT=8766 \
./scripts/sts_start.sh
```

## 安装与环境检查

| 脚本 | 用途 | 备注 |
|---|---|---|
| `scripts/install_rocm_pytorch.sh` | 在指定 venv 安装 TheRock gfx1151 PyTorch | 默认 `VENV_DIR=/home/kamjin/apps/.venv` |
| `scripts/install_qwen3_flash_attn_env.sh` | 复制基础 venv，构建隔离 flash-attn 环境 | 默认目标 `/home/kamjin/apps/.venv-qwen3-fa`；源码 checkout 在 `$STS_CACHE_DIR/src/flash-attention` |
| `scripts/verify_qwen3_flash_attn_env.py` | 验证高性能路径依赖 | 可加 `--kernel-smoke` 做 kernel smoke test |

## Patch 脚本

这些脚本 patch “当前 Python 环境中已安装的依赖”，不是修改本仓库源码。启动脚本会自动检查并应用必要 patch。

| 脚本 | 修复内容 |
|---|---|
| `scripts/patch_sts_offline_startup.py` | 修复 NLTK 本地路径检查，减少离线启动联网 |
| `scripts/patch_paraformer_live_transcription.py` | 避免 Paraformer partial 字幕被当成多轮用户输入 |
| `scripts/patch_qwen3_tts_inline_instruct.py` | 把文本开头的括号语气提示转为 Qwen3-TTS instruct |
| `scripts/patch_qwen3_tts_realtime_stability.py` | 调整 Qwen3-TTS realtime 稳定性参数 |
| `scripts/patch_qwen3_tts_openai_fastapi_bridge.py` | 给 speech-to-speech Qwen3 handler 增加可选 FastAPI bridge |
| `scripts/patch_qwen3_openai_fastapi_compat.py` | 修复 Qwen3-TTS-Openai-Fastapi 与本地 ROCm/qwen_tts 的兼容性 |

## Benchmark 与调试

| 脚本 | 用途 |
|---|---|
| `scripts/bench_sts_pipeline.py` | 端到端 STS 分段性能测试 |
| `scripts/bench_llm_models.py` | 多个 LLM 的 TTFT 对比 |
| `scripts/bench_qwen3_tts_matrix.py` | Qwen3-TTS 中文语气/括号指令矩阵测试 |
| `scripts/bench_qwen3_tts_realtime_perf.py` | 当前 speech-to-speech 安装中的 Qwen3-TTS realtime 吞吐测试 |
| `scripts/bench_qwen3_openai_fastapi_tts.py` | Qwen3-TTS OpenAI FastAPI 服务吞吐测试 |
| `scripts/test_realtime_text_client.py` | 用文本事件测试 Realtime WebSocket 事件流 |
| `scripts/sts_test/gen_test_audio.py` | 生成 STS 测试音频 |
| `scripts/sts_test/test_s2s_ws.py` | WebSocket 端到端 smoke test |

## 新增脚本规则

- 脚本名使用动词前缀：`install_`、`patch_`、`bench_`、`test_`、`verify_`、`sts_start`。
- Python 脚本顶部保留简短 docstring，说明用途、修改对象和输出位置。
- 能用环境变量覆盖的路径不要写死为唯一选择；默认值可以保留本机已验证路径。
- 会修改 venv、外部 repo 或远程服务状态的脚本，必须先打印目标路径。
- 不把 `__pycache__`、`log_*.txt`、模型缓存、venv、下载目录提交进仓库。
- 可复用的模型、源码 checkout、torch.compile 缓存和 benchmark 结果不要默认写 `/tmp`；统一放到 `/home/kamjin/apps/sts-cache` 或通过 `STS_CACHE_DIR` 覆盖。
