# Speech-to-Speech 语音 Agent 管道

> [English](02-speech-to-speech-install.md) · [← 返回 README](../README.zh.md)

## 硬件与环境

| 项目 | 值 |
|---|---|
| 系统 | Fedora 44 |
| GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, Strix Halo) |
| 内存 | 共享系统内存 (UMA) |
| Python | 3.12.13 |
| venv | /home/kamjin/apps/.venv |
| PyTorch | 2.10.0+rocm7.13.0a20260513 (TheRock) |
| ROCm | 7.13 |

完整 ROCm 安装说明见 [rocm-gfx1151-pytorch-install.md](./rocm-gfx1151-pytorch-install.md)

## 管道架构

```
麦克风 → STT (Paraformer-zh) → LLM (Gemma-4-E4B, responses-api) → TTS (Qwen3-TTS) → 扬声器
```

| 组件 | 选择 | 理由 |
|---|---|---|
| **STT** | FunASR Paraformer-zh | 中文 CER 1.95%，内置 VAD + 标点，120x 实时 |
| **LLM** | Gemma-4-E4B (responses-api) | 已有本地服务 127.0.0.1:8101 |
| **TTS** | Qwen3-TTS (faster-qwen3-tts) | ROCm gfx1151 已实测可用 |

### STT 选型对比

| 方案 | 中文 CER | GPU 速度 | ROCm 兼容性 | 安装 |
|---|---|---|---|---|
| **Paraformer-zh** | 1.95% | 120x | PyTorch 直用 | `pip install funasr` |
| SenseVoice-Small | 2.96% | 170x | PyTorch 直用 | `pip install funasr` |
| faster-whisper | 5.14% | 9x | CTranslate2 4.7.1+ | `pip install faster-whisper` |

Paraformer-zh 中文准确率最高，且内置 VAD 和标点恢复，是中文场景首选。

### TTS 选型对比

| 方案 | 音质 | GPU 速度 | ROCm 兼容性 |
|---|---|---|---|
| **Qwen3-TTS** | 高 | ~1x (gfx1151) | 已实测可用 |
| Kokoro | 中 | 快 | 可用 |

Qwen3-TTS 音质更好，且已在 gfx1151 上验证可用。

## 安装

### 1. PyTorch ROCm 基础

```bash
# 创建 Python 3.12 虚拟环境
uv venv /home/kamjin/apps/.venv --python 3.12

# 从 TheRock gfx1151 索引安装
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

详见 [rocm-gfx1151-pytorch-install.md](./rocm-gfx1151-pytorch-install.md)

### 2. Speech-to-Speech 管道

```bash
source /home/kamjin/apps/.venv/bin/activate

# 安装 speech-to-speech + paraformer STT + funasr + qwen-tts
uv pip install "speech-to-speech[paraformer]" funasr
```

已安装的组件：

| 包 | 版本 | 用途 |
|---|---|---|
| speech-to-speech | 0.2.9 | 管道框架 |
| funasr | 1.3.9 | Paraformer STT 后端 |
| qwen-tts | 0.1.1 | Qwen3-TTS 后端 |
| faster-qwen3-tts | 0.2.6 | TTS 推理引擎 |
| hf-transfer | 0.1.9 | HuggingFace 高速下载 |

可选依赖：

| 包 | 用途 | 安装 | 状态 |
|---|---|---|---|
| sox | 音频格式转换（可选） | `sudo dnf install sox` | ⚠️ 未装 |
| DeepFilterNet | 音频降噪增强 | `pip install deepfilternet` | ✅ 已装（需小 patch） |
| flash-attn | 注意力加速 | **不推荐 gfx1151** | ❌ 跳过 |

## 环境变量

启动脚本 `sts_start.sh` 中设置的 ROCm 环境变量：

| 变量 | 值 | 作用 |
|---|---|---|
| `GPU_MAX_ALLOC_PERCENT` | 100 | 允许 GPU VRAM 全量分配（UMA 共享内存，不限制） |
| `GPU_MAX_HEAP_SIZE` | 100 | 限制 HIP 最大堆大小（避免 OOM） |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | 1 | 启用 AOTriton 实验性编译 |

> **注意**：`HSA_OVERRIDE_GFX_VERSION` 已从启动脚本中移除。ROCm 7.13 (TheRock) 已原生修复 gfx1151 VGPR bug，override 反而会导致 `hipErrorInvalidImage` 内核不匹配错误。

## 运行命令

```bash
#!/bin/bash
# Speech-to-Speech pipeline: paraformer (STT) + responses-api (LLM) + qwen3 (TTS)
# GPU: AMD Radeon 8060S (gfx1151) via ROCm 7.13
# STT: FunASR Paraformer-zh (中文优化, CER~1.95%, 120x 实时)
# TTS: faster-qwen3-tts (ROCm 已实测 gfx1151 可用)
# LLM: responses-api (本地 127.0.0.1:8101)

# 允许 GPU VRAM 全量分配（UMA 共享内存，不限制）
export GPU_MAX_ALLOC_PERCENT=100
# 限制 HIP 最大堆大小（避免 OOM）
export GPU_MAX_HEAP_SIZE=100
# 启用 AOTriton 实验性编译（ROCm 7.11+ PyTorch 性能优化）
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

unset HF_ENDPOINT

speech-to-speech \
    --responses_api_base_url "http://127.0.0.1:8101/v1" \
    --responses_api_api_key "" \
    --mode realtime \
    --model_name Gemma-4-E4B-instruct \
    --llm_backend responses-api \
    --responses_api_stream \
    --tts qwen3 \
    --ws_host 0.0.0.0 \
    --ws_port 8765 \
    --stt paraformer \
    --language auto \
    --enable_live_transcription
```

脚本位置：`/home/kamjin/sts_start.sh`

### 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--stt` | `paraformer` | 使用 FunASR Paraformer STT |
| `--tts` | `qwen3` | 使用 Qwen3-TTS |
| `--llm_backend` | `responses-api` | 通过 OpenAI 兼容 API 调用 LLM |
| `--mode` | `realtime` | 实时语音交互模式 |
| `--ws_host` | `0.0.0.0` | WebSocket 监听地址 |
| `--ws_port` | `8765` | WebSocket 端口 |
| `--language` | `auto` | 自动语言检测 |
| `--enable_live_transcription` | | 启用实时字幕 |

## 验证

```bash
source /home/kamjin/apps/.venv/bin/activate

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import speech_to_speech
print(f'speech-to-speech: {speech_to_speech.__version__}')

import funasr
print(f'funasr: {funasr.__version__}')

import qwen_tts
print(f'qwen-tts: installed')

# GPU 功能测试
x = torch.randn(100, 100, device='cuda')
print('GPU tensor: OK')
"
```

## 故障排查

### HIP `device kernel image is invalid`

```
HIP error: device kernel image is invalid
```

- **原因**：`HSA_OVERRIDE_GFX_VERSION=11.0.0` 导致内核架构不匹配
- **解决**：移除 `HSA_OVERRIDE_GFX_VERSION` 环境变量
- **根因**：ROCm 7.13 (TheRock) 已原生修复 gfx1151 VGPR bug，override 反而使 ROCm 误识别为 gfx1100

### GPU 张量分配失败

```
SIGSEGV or CUDA error
```

- 确认未设置 `HSA_OVERRIDE_GFX_VERSION`（ROCm 7.13 不需要）
- 确认使用的是 TheRock wheels（非 PyTorch.org）
- 参考 [rocm-gfx1151-pytorch-install.md](./rocm-gfx1151-pytorch-install.md)

### Paraformer 模型下载慢

```bash
# 使用 ModelScope 镜像
export MODELSCOPE_CACHE=~/.cache/modelscope
# 或使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Qwen3-TTS 推理慢

- flash-attn **不建议**安装（详见 [性能调优 - flash-attn](./speech-to-speech-status.md#4-启用-flash-attention)）
- 当前使用 PyTorch SDPA 默认 backend（math 路径），速度可接受
- TTS 首次 12-14s 含 CUDA graph 编译，后续降至个位数

### DeepFilterNet 重新安装后报 `torchaudio.backend.common` 找不到

- **原因**：deepfilternet 0.5.6 引用了 torchaudio 2.9 之前的 `torchaudio.backend.common.AudioMetaData`，而 TheRock 2.10 wheel 移除了该子包
- **解决**：重装后重新应用 patch，详见 [运行状态文档 - DeepFilterNet Patch](./speech-to-speech-status.md#deepfilternet-torchaudio-兼容-patch)

### SoX 缺失警告

```
WARNING: SoX could not be found!
```

- 不影响核心功能，可选安装：`sudo dnf install sox`

### MIOpen workspace 警告

```
MIOpen(HIP): Warning [IsEnoughWorkspace] Solver <GemmFwdRest>, workspace required: ...
```

- **影响**：无功能影响，MIOpen 会自动调整 workspace
- **原因**：ROCm MIOpen 的 GEMM solver workspace 估计偏差
- **处理**：忽略即可
