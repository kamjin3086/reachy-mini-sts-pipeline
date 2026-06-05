# Speech-to-Speech Voice Agent Pipeline

> [中文版](02-speech-to-speech-install.zh.md) · [← Back to README](../README.md)

## Hardware and environment

| Item | Value |
|---|---|
| System | Fedora 44 |
| GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, Strix Halo) |
| Memory | Shared system memory (UMA) |
| Python | 3.12.13 |
| venv | `/home/kamjin/apps/.venv` |
| PyTorch | 2.10.0+rocm7.13.0a20260513 (TheRock) |
| ROCm | 7.13 |

Full ROCm installation: [rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md)

## Pipeline architecture

```
microphone → STT (Paraformer-zh) → LLM (Gemma-4-E4B, responses-api) → TTS (Qwen3-TTS) → speaker
```

| Component | Choice | Why |
|---|---|---|
| **STT** | FunASR Paraformer-zh | Chinese CER 1.95%, built-in VAD + punctuation, 120× real-time |
| **LLM** | Gemma-4-E4B (responses-api) | Already running locally at 127.0.0.1:8101 |
| **TTS** | Qwen3-TTS (faster-qwen3-tts) | Verified working on ROCm gfx1151 |

### STT selection

| Option | Chinese CER | GPU speed | ROCm compat | Install |
|---|---|---|---|---|
| **Paraformer-zh** | 1.95% | 120× | PyTorch native | `pip install funasr` |
| SenseVoice-Small | 2.96% | 170× | PyTorch native | `pip install funasr` |
| faster-whisper | 5.14% | 9× | CTranslate2 4.7.1+ | `pip install faster-whisper` |

Paraformer-zh wins on Chinese accuracy, with built-in VAD and punctuation restoration. First choice for Chinese scenarios.

### TTS selection

| Option | Quality | GPU speed | ROCm compat |
|---|---|---|---|
| **Qwen3-TTS** | High | ~1× (gfx1151) | Verified |
| Kokoro | Medium | Fast | Works |

Qwen3-TTS has better audio quality and is verified working on gfx1151.

## Installation

### 1. PyTorch ROCm base

```bash
# Create Python 3.12 virtual environment
uv venv /home/kamjin/apps/.venv --python 3.12

# Install from TheRock gfx1151 index
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

See [01-rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md) for details.

### 2. Speech-to-Speech pipeline

```bash
source /home/kamjin/apps/.venv/bin/activate

# Install speech-to-speech + paraformer STT + funasr + qwen-tts
uv pip install "speech-to-speech[paraformer]" funasr
```

Installed components:

| Package | Version | Purpose |
|---|---|---|
| speech-to-speech | 0.2.9 | Pipeline framework |
| funasr | 1.3.9 | Paraformer STT backend |
| qwen-tts | 0.1.1 | Qwen3-TTS backend |
| faster-qwen3-tts | 0.2.6 | TTS inference engine |
| hf-transfer | 0.1.9 | HuggingFace fast download |

Optional dependencies:

| Package | Purpose | Install | Status |
|---|---|---|---|
| sox | Audio format conversion (optional) | `sudo dnf install sox` | ⚠️ Not installed |
| DeepFilterNet | Audio denoising | `pip install deepfilternet` | ✅ Installed (small patch needed) |
| flash-attn | Attention acceleration | **Not recommended for gfx1151** | ❌ Skipped |

## Environment variables

ROCm environment variables set in `sts_start.sh`:

| Variable | Value | Purpose |
|---|---|---|
| `GPU_MAX_ALLOC_PERCENT` | 100 | Allow full GPU VRAM allocation (UMA shared memory, don't restrict) |
| `GPU_MAX_HEAP_SIZE` | 100 | Cap HIP max heap (avoid OOM) |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | 1 | Enable AOTriton experimental compilation |

> **Note**: `HSA_OVERRIDE_GFX_VERSION` has been removed from the start script. ROCm 7.13 (TheRock) fixes the gfx1151 VGPR bug natively — the override actually triggers `hipErrorInvalidImage` from a kernel-architecture mismatch.

## Run command

```bash
#!/bin/bash
# Speech-to-Speech pipeline: paraformer (STT) + responses-api (LLM) + qwen3 (TTS)
# GPU: AMD Radeon 8060S (gfx1151) via ROCm 7.13
# STT: FunASR Paraformer-zh (Chinese-optimized, CER ~1.95%, 120× real-time)
# TTS: faster-qwen3-tts (verified on ROCm gfx1151)
# LLM: responses-api (local 127.0.0.1:8101)

# Allow full GPU VRAM allocation (UMA shared memory, don't restrict)
export GPU_MAX_ALLOC_PERCENT=100
# Cap HIP max heap (avoid OOM)
export GPU_MAX_HEAP_SIZE=100
# Enable AOTriton experimental compilation (ROCm 7.11+ PyTorch perf)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

unset HF_ENDPOINT

INIT_CHAT_PROMPT=${INIT_CHAT_PROMPT:-"你是 Reachy Mini 的中文语音助手。默认用中文口语化回答，每次只说 1 到 2 句，不使用 markdown。不要朗读动作标记、JSON、代码或工具调用内容；如果用户要求动作、表情或跳舞，回复一句自然短句，同时必须通过可用工具调用执行动作。"}

speech-to-speech \
    --responses_api_base_url "http://127.0.0.1:8101/v1" \
    --responses_api_api_key "" \
    --mode realtime \
    --model_name Gemma-4-E4B-instruct \
    --llm_backend responses-api \
    --responses_api_stream \
    --responses_api_disable_thinking \
    --stream_batch_sentences 1 \
    --init_chat_prompt "$INIT_CHAT_PROMPT" \
    --tts qwen3 \
    --qwen3_tts_language chinese \
    --qwen3_tts_speaker Serena \
    --qwen3_tts_instruct "用自然、亲切、清晰的中文口语语气说话。" \
    --qwen3_tts_streaming_chunk_size 12 \
    --qwen3_tts_blocksize 512 \
    --qwen3_tts_non_streaming_mode \
    --ws_host 0.0.0.0 \
    --ws_port 8765 \
    --stt paraformer \
    --language zh \
    --live_transcription_update_interval 0.5 \
    --no_enable_live_transcription
```

Script location: `/home/kamjin/sts_start.sh`

### Parameter reference

| Parameter | Value | Description |
|---|---|---|
| `--stt` | `paraformer` | Use FunASR Paraformer STT |
| `--tts` | `qwen3` | Use Qwen3-TTS |
| `--llm_backend` | `responses-api` | Call LLM via OpenAI-compatible API |
| `--mode` | `realtime` | Real-time voice interaction mode |
| `--ws_host` | `0.0.0.0` | WebSocket listen address |
| `--ws_port` | `8765` | WebSocket port |
| `--language` | `zh` | Force Chinese conversation mode |
| `--no_enable_live_transcription` | | Disable live subtitles until the Paraformer partial/final patch is applied |
| `--qwen3_tts_language` | `chinese` | Force Chinese TTS (`zh` is rejected by the Qwen3-TTS backend) |

To enable live subtitles, first run `python3 scripts/patch_paraformer_live_transcription.py`, then use `--enable_live_transcription --live_transcription_update_interval 0.5`.

## Verification

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
print('qwen-tts: installed')

# GPU functional test
x = torch.randn(100, 100, device='cuda')
print('GPU tensor: OK')
"
```

## Troubleshooting

### HIP `device kernel image is invalid`

```
HIP error: device kernel image is invalid
```

- **Cause**: `HSA_OVERRIDE_GFX_VERSION=11.0.0` causes a kernel-architecture mismatch
- **Fix**: Remove the `HSA_OVERRIDE_GFX_VERSION` environment variable
- **Root cause**: ROCm 7.13 (TheRock) already fixes the gfx1151 VGPR bug natively; the override misidentifies the GPU as gfx1100

### GPU tensor allocation fails

```
SIGSEGV or CUDA error
```

- Make sure `HSA_OVERRIDE_GFX_VERSION` is not set (not needed for ROCm 7.13)
- Make sure you are using TheRock wheels (not PyTorch.org)
- See [01-rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md)

### Paraformer model download is slow

```bash
# Use ModelScope mirror
export MODELSCOPE_CACHE=~/.cache/modelscope
# Or use HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Qwen3-TTS inference is slow

- **Do not install flash-attn** on gfx1151 (see [03 §2.a](03-speech-to-speech-status.md))
- Currently using PyTorch SDPA default backend (math path); speed is acceptable
- First TTS call takes 12–14 s (CUDA graph compilation); subsequent calls drop to single-digit seconds

### DeepFilterNet: `torchaudio.backend.common` not found after reinstall

- **Cause**: deepfilternet 0.5.6 references `torchaudio.backend.common.AudioMetaData`, which TheRock 2.10 has removed
- **Fix**: Re-apply the patch after reinstall — see [03 §7](03-speech-to-speech-status.md#7-deepfilternet--torchaudio-210-compat-patch)

### SoX missing warning

```
WARNING: SoX could not be found!
```

- Not blocking; install with `sudo dnf install sox` if needed

### MIOpen workspace warning

```
MIOpen(HIP): Warning [IsEnoughWorkspace] Solver <GemmFwdRest>, workspace required: ...
```

- **Impact**: None — MIOpen auto-adjusts
- **Cause**: MIOpen's GEMM solver workspace size estimate is off
- **Action**: Ignore
