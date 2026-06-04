# Speech-to-Speech 运行状态与调优指南

> 上次更新：2026-06-04（v2 — 修正 LLM TTFT 实测数据 + LLM 模型选型对比）
> 适用版本：speech-to-speech 0.2.9、funasr 1.3.9、qwen-tts 0.1.1、faster-qwen3-tts 0.2.6、deepfilternet 0.5.6
> 相关文档：[speech-to-speech-install.md](./02-speech-to-speech-install.md)（安装）、[rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md)（ROCm 基础）
>
> [English](03-speech-to-speech-status.md) · [← 返回 README](../README.zh.md)

## 目录

1. [当前运行状态](#当前运行状态)
2. [性能基线](#性能基线)
3. [性能测试方法](#性能测试方法)
4. [调优方向](#调优方向)
5. [已知问题与 workaround](#已知问题与-workaround)
6. [可替换组件选型](#可替换组件选型)
7. [日常运维](#日常运维)
8. [未来工作](#未来工作)

---

## 当前运行状态

### 硬件与软件栈

| 组件 | 当前配置 |
|---|---|
| 系统 | Fedora 44 |
| GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, Strix Halo) |
| 共享内存 | UMA, 124 GB |
| venv | `/home/kamjin/apps/.venv`（uv 创建，无 pip） |
| Python | 3.12.13 |
| PyTorch | 2.10.0+rocm7.13.0a20260513 (TheRock gfx1151 wheels) |
| ROCm/HIP | 7.13.26183 |
| LLM 后端 | 本地 llama-swap `http://127.0.0.1:8101/v1` |
| 当前 LLM 模型 | `Gemma-4-E4B-instruct` |
| numpy | 1.26.4（被 deepfilternet 强制降级，torch 仍正常） |
| DeepFilterNet | 0.5.6（已 patch torchaudio 兼容） |
| flash-attn | 未安装（gfx1151 上游不支持） |

### 启动脚本

`/home/kamjin/sts_start.sh`：

```bash
export GPU_MAX_ALLOC_PERCENT=100
export GPU_MAX_HEAP_SIZE=100
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

启动后所有组件正常 warmup，WebSocket 服务监听 `ws://0.0.0.0:8765/v1/realtime`。

### 模型与缓存路径

| 内容 | 路径 |
|---|---|
| Paraformer STT | `~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/` |
| Qwen3-TTS | HF Hub 缓存（按需下载） |
| Qwen3-TTS 默认声音 | `CustomVoice` 预设（含 Aiden / Vivian 等） |
| nltk_data | `~/nltk_data` |

---

## 性能基线

测试环境：Fedora 44、ROCm 7.13 (TheRock)、音频为合成的低幅度随机噪声（仅用于延迟测量，非真实语料）。

> **冷启动 vs 稳态**：以下数据均**区分**这两个状态。冷启动 = 服务刚启动后**第一次**请求（模型加载到 VRAM、CUDA graph capture、KV cache 预热）；稳态 = 第二次起的请求。

### 单组件延迟

| 组件 | 测试条件 | 延迟 | 备注 |
|---|---|---|---|
| **STT** | 1s 音频 | 0.054s | RTF 18.6x |
| **STT** | 2s 音频 | 0.062s | RTF 32.4x |
| **LLM 冷启动 TTFT** | 首次请求 | **16.97s** | llama-swap 把模型加载到 VRAM |
| **LLM 稳态 TTFT** | 2+ 次请求 | **0.05s** | KV cache 已暖，**基本无感** |
| **LLM 稳态吞吐** | 60 tokens | ~37 tok/s | 实时对话可接受 |
| **TTS TTFA** | 稳态 | ~0.84s | 首次含 CUDA graph 编译（**12-14s**） |
| **TTS Total 首次** | 冷启动 | 70.5s | 含模型加载 + CUDA graph capture |
| **TTS Total 稳态** | 2+ 次请求 | 个位数 | CUDA graph 缓存命中 |

### 端到端用户感知延迟

**感知延迟 = STT + LLM_TTFT + TTS_TTFA**（"用户停止说话 → 听到第一个合成音"的时间）

| 状态 | STT | LLM_TTFT | TTS_TTFA | **E2E 感知** | TTS_Total |
|---|---|---|---|---|---|
| **冷启动** (第 1 次) | 0.085s | 16.97s | 12-14s | **~29s** | 70.5s (含 graph compile) |
| **稳态** (2+ 次) | 0.085s | 0.05s | 0.84s | **~0.97s** | 个位数 |

> **关键观察（实测修正 2026-06-04）**：
> - **之前文档中"LLM_TTFT 3.26s"是冷启动 + 首请求混合值，不是稳态**。稳态 TTFT 实际仅 **~50ms**，对话几乎无延迟
> - 端到端稳态感知延迟 **~1s**，已经接近"自然对话"水平
> - 冷启动主要耗时在 **TTS 的 CUDA graph 编译（12-14s）** 和 **LLM 首次加载（~17s）**
> - **生产部署建议**：服务启动后做一次"预热请求"，把冷启动成本从用户感知路径中移除

### 性能瓶颈分布（稳态）

```
E2E 稳态感知延迟 (0.97s) 分解:
├── STT              0.085s  (  9%)  ← 非常快，已接近最优
├── LLM TTFT         0.050s  (  5%)  ← 稳态基本无感
└── TTS TTFA         0.840s  ( 86%)  ← 当前主要瓶颈
```

---

## 性能测试方法

### 快速测试

```bash
source /home/kamjin/apps/.venv/bin/activate
cd /home/kamjin/scripts

# 全部组件（每个 1-2 个测试点，约 1 分钟）
python3 bench_sts_pipeline.py --quick

# 只测单个组件
python3 bench_sts_pipeline.py --only stt
python3 bench_sts_pipeline.py --only llm
python3 bench_sts_pipeline.py --only tts
python3 bench_sts_pipeline.py --only e2e
```

### 完整测试

```bash
# STT 1/2/5/10s × 2 轮，LLM/TTS 5 文本 × 2 轮，E2E 3 轮
# 约 5-10 分钟（首次含模型加载）
python3 bench_sts_pipeline.py
```

### 脚本功能

`/home/kamjin/scripts/bench_sts_pipeline.py` 输出：

- **GPU 信息** — PyTorch / ROCm / 设备 / VRAM / 架构
- **STT** — Duration / Latency / RTF 表
- **LLM** — TTFT / Total / 估算 Token / tok/s 表
- **TTS** — TTFA / Total / Audio 长度 / RTF 表，保存 WAV 到 `/tmp/sts_bench/`
- **E2E** — STT / LLM_TTFT / LLM_Total / TTS_TTFA / TTS_Total / E2E 感知延迟表

### 验证语音质量

TTS 测试自动保存生成的音频到 `/tmp/sts_bench/tts_N.wav`，可直接播放听感。

要测 STT 真实效果，可用 `arecord` 录一段中文，再用脚本喂入：

```bash
arecord -f S16_LE -r 16000 -d 3 test.wav
# 改造脚本把 gen_silent_wav() 替换为加载 test.wav
```

---

## 调优方向

按收益从高到低排序：

### 1. 降低 LLM TTFT（边际收益已很小）

**实测发现**：当前 `Gemma-4-E4B-instruct` 稳态 TTFT 仅 **50ms**——基本无感，不需要换模型。

**LLM 选型实测对比**（2026-06-04，llama-swap 8101 端口，3 prompt 平均）：

| 模型 | 冷启动 | 稳态 TTFT | tok/s | 备注 |
|---|---|---|---|---|
| **Gemma-4-E4B-instruct (当前)** | 16.97s | **0.05s** | 37.1 | TTFT 最快，性价比最优 |
| GPT-OSS-20B | 0.46s | 0.39s | 64.8 | 吞吐 2x，但 TTFT 慢 8x |
| Qwen3.6-35B-A3B-instruct (MoE) | 28.9s | 0.18s | 45.3 | 综合接近，长答案更优 |
| Qwen3.5-4b-FLM-instruct (NPU) | 8.36s | 1.20s | 12.7 | **NPU 反而更慢** |
| Step-3.5-Flash-normal | 81.9s | 3.5s+ | 0 | 返回 0 tokens，坏掉 |
| Gemma-4-E2B (instruct/普通/thinking) | — | — | — | **启动失败** `upstream command exited prematurely` |

**结论**：当前 `Gemma-4-E4B-instruct` 在稳态 TTFT 上已经是最佳。**不要换**。

**如果真的想优化**，以下方向都比换模型更有效：

**a) 启用 prompt prefix cache**（推荐）

llama-swap / vLLM 支持相同 system prompt 复用 KV cache。speech-to-speech 每次调用 system prompt 相同，**~30% TTFT 收益**（从 50ms → 35ms，意义不大但聊胜于无）。需要在 llama-swap 端配置。

**b) 启动时预热 LLM**（推荐）

把 llama-swap 17s 冷启动成本从用户感知路径中移除。完整脚本见 [§2.c](#c-预热-tts--llm--stt消除冷启动)。

**c) 切换到 vLLM / SGLang**（如果 llama-swap 出现瓶颈）

吞吐和并发能力可能更强，但 vLLM 在 gfx1151 上的兼容性需要评估。**当前不需要**。

**d) 关闭 thinking / reasoning**（已默认开启）

当前已使用 `*instruct` 变体，无需调整。**不要**切换到 `*-thinking` 变体——会增加 ~5x 生成时间。

### 2. TTS 加速

**a) 启用 flash-attention — ❌ 不建议 gfx1151 安装**

经过 [ROCm/TheRock#1364](https://github.com/ROCm/TheRock/issues/1364) 和 [TesslateAI/FlashAttentionDist](https://github.com/TesslateAI/FlashAttentionDist) 的调研：

| 渠道 | gfx1151 支持 | 备注 |
|---|---|---|
| 官方 PyPI `flash-attn` | ❌ 无 wheel，仅 `.tar.gz` | 需 30-60 min 源码编译 |
| TesslateAI 预编译 | ❌ 仅 gfx90a/gfx942/gfx950 | 数据中心 GPU |
| TheRock PyTorch SDPA flash | ❌ 运行时禁用 | `hip/sdp_utils.cpp:961` 主动禁用 |
| 自建 AOTriton 0.11beta | ⚠️ 实验性 | "性能比 nightly 更慢"，需重编 PyTorch |

**实测**：在 TheRock 2.10.0+rocm7.13 wheel 上：

```python
import torch
print(hasattr(torch, 'ops') and hasattr(torch.ops, 'aotriton'))  # True (C++ 注册)
import importlib.util
print(importlib.util.find_spec('pyaotriton'))  # None (Python 运行时未打包)
```

`pyaotriton` Python 包在 wheel 中**不存在**（只有 C++ ops 注册），所以 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 启用了 ops 但无 backend。SDPA 默认走 math path，**0.20ms @ [2,8,64,64] fp16** 已经很够用。

**结论**：跳过 flash-attn 安装。强行编译 30+ min 大概率失败，或编译出无 gfx1151 HIP kernel 的废包。

**b) 切换到更轻量的 TTS**

Qwen3-TTS 1.7B 对中文支持有限（默认 CustomVoice 偏英文）。可选：

- `kokoro` — 速度更快，中文需额外 voice model
- `CosyVoice` — 中文更优，但 ROCm 兼容性未验证

修改 `sts_start.sh`：

```bash
--tts kokoro  # 需先 pip install "speech-to-speech[kokoro]"
```

**c) 预热 TTS + LLM + STT（消除冷启动）**

启动时调用一次所有组件，把冷启动成本从用户感知路径中移除：

```bash
# 加到 sts_start.sh 末尾
(
  sleep 5  # 等服务起来
  # STT 预热
  python3 -c "
import os; os.environ.pop('OPENAI_API_KEY', None)
import torch
if hasattr(torch, 'mps') and not torch.backends.mps.is_available():
    torch.mps.empty_cache = torch.cuda.empty_cache
from speech_to_speech import ParaformerASR
asr = ParaformerASR()
print('STT warmup OK')
" 2>&1 | tail -3
  # LLM 预热
  curl -s -X POST http://127.0.0.1:8101/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Gemma-4-E4B-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    > /dev/null
  # TTS 预热（首次会触发 CUDA graph 编译，~12-14s）
  python3 -c "
import os; os.environ.pop('OPENAI_API_KEY', None)
from speech_to_speech import Qwen3TTS
tts = Qwen3TTS()
print('TTS warmup OK')
" 2>&1 | tail -3
) &
```

**实测冷启动成本**：STT ~3s + LLM ~17s + TTS ~14s ≈ **34s**。预热后用户感知延迟直接进入稳态。

### 3. STT 优化

STT 已 18-32x 实时，延迟极低，几乎无优化空间。如果要更准的中文：

- `SenseVoice-Small`（2.96% CER，更小更快）— 修改 `sts_start.sh` `--stt sensevoice`
- 多语种混合场景：`paraformer` + `whisper` 双模型 fallback

### 4. 流式优化（高级）

当前 `TTS` 使用 `non_streaming_mode=True`（一次合成完整文本）。改为流式（边收 LLM chunk 边合成）可将感知延迟再降 30-50%：

```python
# bench_sts_pipeline.py 中
setup_kwargs={"non_streaming_mode": False, "streaming_chunk_size": 8}
```

启动脚本需要自定义封装（speech-to-speech CLI 当前不直接暴露此参数）。需要改造 `s2s_pipeline.py`。

### 5. 系统级优化

| 项目 | 状态 | 说明 |
|---|---|---|
| `GPU_MAX_ALLOC_PERCENT=100` | ✅ 已设 | 允许 VRAM 全量分配 |
| `GPU_MAX_HEAP_SIZE=100` | ✅ 已设 | 限制 HIP 堆 |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | ✅ 已设 | 启用 AOTriton 编译（实际无 backend） |
| SoX | ⚠️ 未装 | 可选，音频格式转换 |
| DeepFilterNet | ✅ 已装 | 含 torchaudio 兼容 patch，见下文 |

安装可选依赖：

```bash
sudo dnf install sox        # 音频格式工具
# DeepFilterNet 见下方专门章节
```

---

## 已知问题与 workaround

### 1. `hipErrorInvalidImage` (已修复)

`HSA_OVERRIDE_GFX_VERSION=11.0.0` 反而导致 ROCm 7.13 内核不匹配。

**解决**：从 `sts_start.sh` 移除该变量。详见 [01-rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md)。

### 2. `hf_transfer` 缺失 (已修复)

TTS 模型下载时 `hf_transfer` 未安装导致 `ValueError`。

**解决**：

```bash
uv pip install hf_transfer
```

### 3. Paraformer 库内 MPS bug (已在测试脚本 workaround)

`speech_to_speech/paraformer_handler.py:56` 无条件调用 `torch.mps.empty_cache()`，在 ROCm/CUDA 上崩溃。

**当前 workaround**：测试脚本头部 monkey-patch（见 `bench_sts_pipeline.py:35-38`）。生产环境需要 patch 源文件或升级 speech-to-speech。

**根治**：向 speech-to-speech 上游提 issue，或本地 patch：

```bash
sed -i 's/torch.mps.empty_cache()/torch.cuda.empty_cache()/' \
  /home/kamjin/apps/.venv/lib64/python3.12/site-packages/speech_to_speech/STT/paraformer_handler.py
```

### 4. MIOpen workspace 警告（无害）

```
MIOpen(HIP): Warning [IsEnoughWorkspace] Solver <GemmFwdRest>, workspace required: 41287680, ...
```

MIOpen GEMM solver 的 workspace 估计偏差，无功能影响。可通过设置 `MIOPEN_LOG_LEVEL=3` 抑制。

### 5. 中文 TTS 音色有限

Qwen3-TTS `CustomVoice` 默认以英文/国际化音色为主（`Aiden`, `Vivian` 等），中文音色音质一般。

**建议**：测试 `instruct` 参数调整中文表现，或评估 CosyVoice 等更优的中文 TTS。

### 6. WebSocket 客户端未实现

`ws://0.0.0.0:8765/v1/realtime` 需要 OpenAI Realtime API 兼容的客户端才能对接。常用：

- Web 端：LiveKit、Pipecat
- 桌面端：参考 speech-to-speech 仓库 examples

### 7. DeepFilterNet + torchaudio 2.10 兼容 patch

**问题**：deepfilternet 0.5.6 在 `df/io.py:9` 引用了 `torchaudio.backend.common.AudioMetaData`，但 TheRock 编译的 `torchaudio 2.10.0+rocm7.13` **移除了 `torchaudio.backend` 子包**，导致导入失败：

```python
>>> from df.enhance import init_df
ModuleNotFoundError: No module named 'torchaudio.backend'
```

**当前 workaround**：已在 `df/io.py` 加上 `try/except` fallback，导入时回退到 `Any`：

```python
try:
    from torchaudio.backend.common import AudioMetaData  # type: ignore[attr-defined]
except ImportError:
    AudioMetaData = Any  # type: ignore[assignment,misc]
```

**已验证**：1s 音频降噪在 ROCm 上 GPU 端到端跑通：

```
input:  shape=[1, 48000], peak=1.766
output: shape=[1, 48000], peak=0.318  (成功降噪)
RT:     2.33s（首次，含 cuDNN 编译；后续 < 0.5s）
```

**重新安装后重新应用 patch**（升级 deepfilternet / 重装 venv 时）：

```bash
# 用 sed 一行修复
sed -i 's/^from torchaudio.backend.common import AudioMetaData$/try:\n    from torchaudio.backend.common import AudioMetaData  # type: ignore[attr-defined]\nexcept ImportError:\n    AudioMetaData = Any  # type: ignore[assignment,misc]/' \
  /home/kamjin/apps/.venv/lib64/python3.12/site-packages/df/io.py
```

或手动编辑 `df/io.py`，将第 9 行替换为上面的 `try/except` 块。

**根本解决**：deepfilternet 上游 issue/PR，或等待兼容 torchaudio 2.10+ 的新版本。

### 8. flash-attn 未安装（决策记录）

详见 [调优方向 §2.a](#2-tts-加速)。决策依据：gfx1151 上游无 HIP flash kernel 优化，强行编译 30+ min 大概率产出无用的包，且会破坏 venv 一致性。

---

## 可替换组件选型

### STT 备选

| 模型 | 中文 CER | 速度 | ROCm | 备注 |
|---|---|---|---|---|
| **Paraformer-zh** (当前) | 1.95% | 18-32x | ✅ | 中文最优，库兼容良好 |
| SenseVoice-Small | 2.96% | ~170x | ✅ | 更小更快，多语种 |
| faster-whisper large-v3 | 5.14% | 9x | ⚠️ | 多语种通用，中文较弱 |

### TTS 备选

| 模型 | 中文音质 | 速度 | ROCm | 备注 |
|---|---|---|---|---|
| **Qwen3-TTS** (当前) | 中 | 慢 | ✅ | 库默认，英文向 |
| Kokoro | 中 | 快 | ✅ | 中需额外 voice |
| CosyVoice 2 | 高 | 中 | ⚠️ 未测 | 中文原生，需评估 ROCm |
| ChatTTS | 中 | 中 | ⚠️ | 需 `pip install speech-to-speech[chattts]` |

### LLM 备选

`llama-swap` 管理的全部模型都可用，编辑 `sts_start.sh` 的 `--model_name`（**改完需重启 pipeline**）。

**实测数据**（2026-06-04，3 prompt 平均，稳态）：

| 模型 | 稳态 TTFT | tok/s | 推荐场景 |
|---|---|---|---|
| **Gemma-4-E4B-instruct (当前)** | **0.05s** | 37.1 | 默认首选，对话 TTFT 之王 |
| GPT-OSS-20B | 0.39s | 64.8 | 适合长答案（吞吐快），TTFT 略差 |
| Qwen3.6-35B-A3B-instruct | 0.18s | 45.3 | 综合接近，长答案更优 |
| Qwen3.6-27B-instruct | — | — | 慢（27B），未测稳态 |
| Gemma-4-E2B-instruct | — | — | ❌ **启动失败** |
| Gemma-4-E2B-thinking | — | — | ❌ **启动失败** |
| Qwen3.5-4b-FLM-instruct | 1.20s | 12.7 | ❌ NPU 反而更慢 |
| Qwen3.5-9b-FLM-instruct | — | — | ⚠️ NPU 一致偏慢，未详测 |
| Step-3.5-Flash-normal | 3.5s+ | 0 | ❌ 返回 0 tokens，坏掉 |
| OmniCoder-9B | — | — | 编程专用 |
| MiroThinker-1.7-mini | — | — | Agentic 工具调用 |

**修改方法**：

```bash
# 编辑 sts_start.sh
sed -i 's/--model_name .*/--model_name GPT-OSS-20B/' /home/kamjin/sts_start.sh

# 重启 pipeline（llama-swap 不用重启）
pkill -f speech-to-speech
sleep 2
./sts_start.sh
```

---

## 日常运维

### 启动 / 停止

```bash
# 启动
./sts_start.sh

# 停止
pkill -f speech-to-speech

# 查看状态
curl -s http://127.0.0.1:8765/v1/realtime  # WebSocket (需 ws 客户端)
ps aux | grep speech-to-speech
```

### 日志位置

speech-to-speech 本身输出到 stdout/stderr。建议重定向：

```bash
nohup ./sts_start.sh > /tmp/sts.log 2>&1 &
```

### llama-swap 健康检查

```bash
curl -s http://127.0.0.1:8101/v1/models | jq '.data[].id'
```

### 重启 pipeline

```bash
pkill -f speech-to-speech
sleep 2
./sts_start.sh
```

### 重置模型缓存（如果出错）

```bash
# 保留缓存
ls ~/.cache/modelscope/hub/
ls ~/.cache/DeepFilterNet/   # DeepFilterNet 3 权重
ls ~/.cache/huggingface/hub/

# 清空（下次启动会重新下载）
rm -rf ~/.cache/modelscope/hub/iic/
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS*
rm -rf ~/.cache/DeepFilterNet/  # DeepFilterNet 3 权重
```

### 包依赖注记（重要）

`deepfilternet 0.5.6` 安装时强制降级了两个间接依赖，**torch 仍能正常工作**：

| 包 | 当前版本 | 原版本 | 原因 |
|---|---|---|---|
| numpy | 1.26.4 | 2.4.x | deepfilternet pin numpy<2 |
| packaging | 23.2 | 25+ | deepfilternet 间接依赖 |

**已验证**：torch 2.10.0+rocm7.13 GPU 计算、Paraformer、Qwen3-TTS 全部正常。

**如需恢复 numpy 2.x**（如升级 torch 时）：

```bash
source /home/kamjin/apps/.venv/bin/activate
uv pip install "numpy>=2.0" --no-deps
# 验证
python3 -c "import torch; a=torch.randn(10,device='cuda'); print(a.sum().item())"
# 验证 deepfilternet
python3 -c "from df.enhance import init_df; init_df()"
```

如果 deepfilternet 失效但 torch 必须用 numpy 2.x，二选一。

---

## 未来工作

### 短期（1-2 周）

- [x] ~~安装 SoX、DeepFilterNet（可选依赖）~~ — DeepFilterNet 已装（含 patch），SoX 未装
- [x] ~~尝试 `Gemma-4-E2B-instruct` 对比 TTFT 改善~~ — **失败**，3 个变体（instruct / 普通 / thinking）全部 `upstream command exited prematurely`，llama-swap 服务端问题，待配置方修复
- [x] ~~评估 llama-swap 各模型 TTFT~~ — 7 个模型实测，**当前 E4B-instruct 是稳态 TTFT 最优**，不要换
- [ ] 加 **LLM 预热** 到 `sts_start.sh`（消除 17s 冷启动）
- [ ] 评估 `CosyVoice 2` 中文 TTS 质量（需 ROCm 兼容性测试）
- [ ] 改造 `TTS` 为流式（`non_streaming_mode=False`）
- [ ] 写一个 OpenAI Realtime API 兼容的 Web 客户端（HTML+JS）
- [ ] 向 deepfilternet 上游提 issue（torchaudio.backend 兼容性）
- [ ] 向 llama-swap 维护方提 issue（Gemma-4-E2B 启动失败）

### 中期（1 个月）

- [ ] 部署 Web 客户端，支持浏览器语音对话
- [ ] 添加可观测性：prometheus metrics / OpenTelemetry
- [ ] 多用户并发测试：当前 pipeline 是单 session
- [ ] 调优 llama-swap 推理参数（temperature、repetition_penalty 等）

### 长期

- [ ] 评估升级到 vLLM / SGLang 替代 llama-swap
- [ ] ~~探索 ROCm Flash Attention 2/3（性能增益）~~ — **不可行**，gfx1151 无上游支持
- [ ] ~~评估 aotriton 内核~~ — **无 Python backend**，C++ ops 已注册但运行时缺失
- [ ] 等待 speech-to-speech 上游修复 MPS bug，移除 monkey-patch
- [ ] 升级到 Qwen3-TTS VoiceDesign / Base 模型支持自定义声音
- [ ] 等待 TheRock wheel 修复 `pyaotriton` Python 打包问题

---

## 故障排查速查

| 症状 | 原因 | 解决 |
|---|---|---|
| 启动报 `hipErrorInvalidImage` | HSA override 残留 | 移除 `HSA_OVERRIDE_GFX_VERSION` |
| `HF_HUB_ENABLE_HF_TRANSFER` 报错 | hf_transfer 未装 | `uv pip install hf_transfer` |
| OpenAI client 报 API key 缺失 | env 冲突 | 确认 `OPENAI_API_KEY` 未设置 |
| Paraformer 报 MPS 错误 | 库 bug | 用 patch 脚本或 monkey-patch |
| DeepFilterNet 报 `torchaudio.backend` 找不到 | 0.5.6 与 torchaudio 2.10 不兼容 | 应用 [§7 patch](#7-deepfilternet--torchaudio-210-兼容-patch) |
| MIOpen workspace 警告 | 估计偏差 | 忽略，或 `MIOPEN_LOG_LEVEL=3` |
| LLM 401 invalid_api_key | 客户端 env 干扰 | 确认 `OPENAI_API_KEY=""` 或未设 |
| TTS 首次 12s+ | CUDA graph 编译 | 预热即可（不可消除） |
| SoX 警告 | sox 未装 | `sudo dnf install sox` |
| `pip` 指向 Python 3.14 不是 venv | `~/.local/bin/pip` 在 PATH 中抢先 | 用 `uv pip` 或 `/home/kamjin/apps/.venv/bin/python -m pip` |

---

## 参考链接

- 仓库：[speech-to-speech](https://github.com/facebookresearch/speech-to-speech)
- 安装文档：[02-speech-to-speech-install.md](./02-speech-to-speech-install.md)
- ROCm 文档：[01-rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md)
- 测试脚本：`/home/kamjin/scripts/bench_sts_pipeline.py`（端到端）、`/home/kamjin/scripts/bench_llm_models.py`（LLM TTFT 对比）
- 启动脚本：`/home/kamjin/sts_start.sh`
- llama-swap: `http://127.0.0.1:8101/v1`
