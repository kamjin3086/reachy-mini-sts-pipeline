# Reachy Mini + 本地语音对话完整部署

> 在 [Reachy Mini Lite](https://www.pollen-robotics.com/reachy-mini/) 上跑通**纯本地**中英文语音对话的完整记录 — 从 CPU 失败到 ROCm GPU 加速的部署全过程。

[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ROCm](https://img.shields.io/badge/ROCm-7.13-ED1C24)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-gfx1151%20(Strix%20Halo)-76B900)](https://www.amd.com/en/products/processors/ryzen-ai)
[![English](https://img.shields.io/badge/English-README.md-blue)](README.md)

---

## 这个项目是什么

**2026 年 6 月初**，我组装了一台 Reachy Mini Lite —— 硬件用了大约 **3 小时**，之后又花了 **2 天的空闲时间** 调试 pipeline、配置环境、修改官方 Reachy Mini 对话 app。目标很简单：跑一个完全本地、支持中英文的实时语音对话，**全程零云 API**。

这个仓库记录了完整过程 —— 安装、踩坑、性能基线、最终能跑的方案 —— 给有类似需求（本地 STS、AMD GPU + Linux、Reachy Mini 集成）的人参考，省去我那些试错时间。

### 关键结果

| 指标 | 数值 |
|---|---|
| 端到端稳态感知延迟 | **~1.0 秒**（用户停嘴 → 听到第一声合成音）|
| ASR 速度 | **35x** 实时（中文 Paraformer-zh）|
| LLM 首 token 延迟（稳态） | **50 ms**（Gemma-4-E4B-instruct）|
| TTS 合成（稳态） | 个位数秒级 |
| 中文支持 | ✅ ASR 优秀；TTS 偏英文（生产建议换 CosyVoice 2）|
| 完全离线 | ✅ 无任何云 API 调用 |

冷启动首请求 ~29s（模型加载 + TTS CUDA graph 编译）。CLI 启动时内部会做预热，但第一个真实用户请求仍需承担完整冷启动成本，详见 [docs/03 性能基线](docs/03-speech-to-speech-status.md)。

### 故事线

我参考了 Hugging Face 的博客 [Local Reachy Mini Conversation](https://huggingface.co/blog/local-reachy-mini-conversation) 和 r/LocalLLaMA 上启发的帖子 [Reachy Mini Goes Fully Local](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/) 作为起点。最初在纯 CPU 上跑：Qwen3-TTS 不支持 AMD GPU，换成 Kokoro。管道跑起来了，但 Reachy Mini App 连上后"什么都不发生"——没日志、没 VAD、没声音。我把症状记录下来（草稿保留在 [docs/04](docs/04-reachy-mini-debug-journey.md) 作为调试参考）。

后来入手 **AMD Strix Halo 128G**（Ryzen AI Max+ 395，Radeon 8060S 集显，gfx1151）工作站，重新调研 ROCm 兼容性，解决了一系列 AMD 相关问题，把整套栈跑在 GPU 上。**这第二遍就是这个仓库记录的内容**。

过程中我还：

- Fork 并修改了官方 Reachy Mini 对话 app：**[kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app)** — 用 `pip install -e .` 安装，源码改动实时生效（省掉反复重装）
- 把调试工作流打包成可复用的 agent skill：**[kamjin3086/reachymini-debug-skill](https://github.com/kamjin3086/reachymini-debug-skill)**

## 硬件 / 软件要求

- **GPU**：AMD Strix Halo / Strix Point (gfx1151 / gfx1150) 或其他 ROCm 7.13+ 支持的 GPU。NVIDIA GPU 跑管道本身也行，但下面的安装步骤是 ROCm 专用
- **系统**：Fedora 44（其他主流 Linux 发行版小改也能用）
- **内存**：建议 32 GB+（LLM 加载需要 ~8 GB VRAM）
- **Python**：3.12+（用 [uv](https://github.com/astral-sh/uv) 管理 venv）
- **机器人**（可选，仅 Reachy Mini 集成需要）：Reachy Mini Lite，通过 Reachy Mini Control 连接

## 快速开始

整个管道分 **5 步**搭建，每步都锁了版本号（避免我那 2 天调试踩的坑）。**[uv](https://github.com/astral-sh/uv) 是推荐的 venv + 包管理器** —— 解析能力远好于 `venv` + `pip`，下面的步骤都默认用 uv。

### 装什么（版本锁）

| 组件 | 版本 | 为什么是这个版本 |
|---|---|---|
| Python | 3.12.13 | TheRock gfx1151 wheels 必需 |
| PyTorch（TheRock gfx1151 build）| `2.10.0+rocm7.13.0a20260513` | PyTorch.org 的 `rocm7.1` wheels 在 Strix Halo 上 SIGSEGV（Issue #2991）。TheRock gfx1151 带了 VGPR 修复 |
| ROCm / HIP | 7.13 | 第一个原生支持 gfx1151 的版本。**此版本不要设 `HSA_OVERRIDE_GFX_VERSION`** |
| numpy | 1.26.4 | **强制降级**，否则与 deepfilternet + torch 2.10 不兼容 |
| packaging | 23.2 | **强制降级**，deepfilternet 0.5.6 要求 |
| speech-to-speech | 0.2.9 | 管道框架 |
| funasr | 1.3.9 | Paraformer-zh STT —— 中文 CER **1.95%**，**120× 实时**（GPU）|
| qwen-tts | 0.1.1 | Qwen3-TTS 后端 |
| faster-qwen3-tts | 0.2.6 | TTS 推理引擎（ROCm gfx1151 已验证）|
| deepfilternet | 0.5.6 | 降噪 —— **+ 1 行 patch**（`df/io.py:9`，适配 TheRock torchaudio 2.10；每次重装后要重做）|
| hf-transfer | 0.1.9 | HuggingFace 高速下载 —— **TTS 模型下载必需** |

基础安装跳过 **flash-attn**。高性能 Qwen3-TTS 路径见 [docs/06](docs/06-runtime-paths-and-offline.zh.md)。

### 第 1 步 —— ROCm + PyTorch（TheRock gfx1151）

```bash
# 用 uv 创建 venv
uv venv ~/.venvs/sts --python 3.12
source ~/.venvs/sts/bin/activate

# 从 TheRock gfx1151 索引装 PyTorch —— 不是 PyTorch.org
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

验证：

```bash
python3 -c "
import torch
assert torch.cuda.is_available(), 'ROCm 没工作'
assert 'gfx1151' in torch.cuda.get_arch_list(), '用错 wheel 索引了'
x = torch.randn(100, 100, device='cuda')
print('ROCm + PyTorch OK:', torch.cuda.get_device_name(0))
"
```

完整 ROCm 排错：[docs/01 — ROCm gfx1151 PyTorch 安装](docs/01-rocm-gfx1151-pytorch-install.md)。

### 第 2 步 —— STT + TTS + 降噪

```bash
# 主管道 + STT/TTS 后端（降噪单独装，见下）
uv pip install "speech-to-speech[paraformer]" funasr qwen-tts \
  faster-qwen3-tts hf-transfer

# 降噪 —— 需要两个强制降级才能与 torch 2.10 共存
uv pip install "deepfilternet==0.5.6"
# 然后打 torchaudio 2.10 兼容 patch（一次性，每次重装后要重做）
# sed 命令见 docs/03 §7
```

**为什么选这些**（完整对比表见 [docs/02](docs/02-speech-to-speech-install.md)）：

- **STT: Paraformer-zh** —— 中文 CER 1.95%（对比 SenseVoice 2.96%、faster-whisper 5.14%），**120× 实时**（GPU），自带 VAD + 标点恢复
- **TTS: Qwen3-TTS** —— 唯一在 ROCm gfx1151 上验证可用的 TTS。Kokoro 能用但是 CPU only + 偏英文
- **降噪: DeepFilterNet 0.5.6** —— 质量最高；RNNoise 更快但质量差

### 第 3 步 —— LLM 后端

管道连的是任何 **OpenAI 兼容的 LLM 端点** —— 指向任何本地 LLM server（`llama-server`（llama.cpp 自带）、vLLM、SGLang，或任何其他）都行。`sts_start.sh` 默认 URL 是 `http://127.0.0.1:8101/v1` —— 按需调整 `--responses_api_base_url`。

本项目实际部署的是 **llama-swap**（OpenAI 兼容 HTTP 代理）后接 **lemonade**（[lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade)，AMD 优化推理后端，对 Strix Halo iGPU / Ryzen AI NPU 有专门加速）—— 下面的性能数据是基于这套跑的。换其他栈 LLM 性能会不一样，但管道本身不挑。

实测 **7 个模型** 后，`Gemma-4-E4B-instruct` 是稳态 TTFT 之最（50 ms）。完整数据见 [docs/03 §调优 §1](docs/03-speech-to-speech-status.md)。

### 第 4 步 —— 启动

```bash
git clone https://github.com/kamjin3086/reachy-mini-sts-pipeline.git
cd reachy-mini-sts-pipeline
$EDITOR scripts/sts_start.sh   # 把 --model_name 改成你 LLM server 提供的模型
./scripts/sts_start.sh
# → WebSocket: ws://0.0.0.0:8765/v1/realtime
```

当前只保留两条启动路径：

| 路径 | 命令 | 适合场景 |
|---|---|---|
| 稳定路径 | `./scripts/sts_start.sh` | 需要最简单、最容易排障的可用方案 |
| 高性能 Qwen3-TTS | `./scripts/sts_start_qwen3_openai_fastapi_flash.sh` | 需要更好的 Qwen3-TTS 运行吞吐，能接受较慢预热 |

离线启动和高性能环境说明见 [docs/06](docs/06-runtime-paths-and-offline.zh.md)。

生产默认关闭 Paraformer live transcription，避免中文 partial 字幕被当成多轮用户输入重复送进 LLM。若要启用实时字幕，先 patch 已安装的 handler：

```bash
python3 scripts/patch_paraformer_live_transcription.py
# 然后把 sts_start.sh 中的 --no_enable_live_transcription 改为 --enable_live_transcription
```

`sts_start.sh` 设了这些 ROCm 环境变量（**不要**设 `HSA_OVERRIDE_GFX_VERSION`，见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)）：

```bash
export GPU_MAX_ALLOC_PERCENT=100                    # 放行 UMA 全量分配
export GPU_MAX_HEAP_SIZE=100                        # 限制 HIP heap
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1    # AOTriton 性能
```

稳态感知延迟：**~1.0 秒**（用户停嘴 → 听到第一声合成音）。要接 Reachy Mini 见 fork 的对话 app：[kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app)。要边改源码边调试（`pip install -e .` 可编辑模式）见 [docs/04](docs/04-reachy-mini-debug-journey.zh.md#迭代修改-fork-的-app可编辑模式安装)。

## 文档索引

| 文档 | 用途 | 何时读 |
|---|---|---|
| [docs/01-rocm-gfx1151-pytorch-install.md](docs/01-rocm-gfx1151-pytorch-install.md) | ROCm 7.13 + TheRock gfx1151 wheels + PyTorch 安装 | **第一次**部署必读 |
| [docs/02-speech-to-speech-install.md](docs/02-speech-to-speech-install.md) | speech-to-speech 管道安装 + 选型对比（STT/TTS/LLM） | 看完 01 后 |
| [docs/03-speech-to-speech-status.md](docs/03-speech-to-speech-status.md) | 详细状态、调优记录和已知问题 | 排查内部问题时 |
| [docs/04-reachy-mini-debug-journey.md](docs/04-reachy-mini-debug-journey.md) | Reachy Mini 调试之旅（最初的"什么都不发生"问题记录）| 遇到 Reachy Mini 连接问题时 |
| [docs/05-qwen3tts-realtime-test-report.zh.md](docs/05-qwen3tts-realtime-test-report.zh.md) | 中文 Qwen3-TTS、括号语气指令、Realtime 工具调用实测 | 调中文对话效果时 |
| [docs/06-runtime-paths-and-offline.zh.md](docs/06-runtime-paths-and-offline.zh.md) | 两条启动路径、离线缓存行为和 flash-attn 环境检查 | 启动或分享项目前 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 关键 bug 速查表 | 故障时 |
| [README.md](README.md) | 英文版 | English readers |

## 性能测试

```bash
# 端到端 benchmark
python3 scripts/bench_sts_pipeline.py --quick

# 仅 LLM TTFT 对比多个模型
python3 scripts/bench_llm_models.py
```

实测 7 个 LLM 模型后，**`Gemma-4-E4B-instruct` 是稳态 TTFT 最优（50ms）** NPU 加速模型反而更慢。详见 [docs/03 调优方向 §1](docs/03-speech-to-speech-status.md)。

## 核心组件

| 组件 | 选型 | 备选 | 理由 |
|---|---|---|---|
| ASR | Paraformer-zh (FunASR) | SenseVoice, faster-whisper | 中文 CER 1.95% 最优 |
| LLM | Gemma-4-E4B-instruct | GPT-OSS-20B, Qwen3.6-35B-A3B | 稳态 TTFT 之王 |
| TTS | Qwen3-TTS (CustomVoice) | Kokoro, CosyVoice 2 | ROCm 兼容已验证 |
| LLM 后端 | 任何 OpenAI 兼容 server（如 `llama-server`、vLLM、SGLang） | — | 管道只需要一个 OpenAI 兼容的 HTTP 端点。实际部署的是 llama-swap（代理）+ lemonade（推理），见 [§第 3 步](#第-3-步--llm-后端) |
| 降噪 | DeepFilterNet 0.5.6 | RNNoise | 高质量、已 patch 兼容 |

## 已知问题与 workaround

- **MPS bug**：`speech_to_speech/paraformer_handler.py:56` 无条件调用 `torch.mps.empty_cache()` 在 ROCm/CUDA 上崩溃。已提供 `sed` 一行修复。
- **HSA override 副作用**：`HSA_OVERRIDE_GFX_VERSION=11.0.0` 在 ROCm 7.13 (TheRock) 上反而触发 `hipErrorInvalidImage`——**直接移除即可**。
- **DeepFilterNet + torchaudio 2.10**：`df/io.py` 的 `torchaudio.backend.common` 引用在 TheRock 2.10 中不存在。已提供 `try/except` fallback patch。
- **flash-attn**：不要装进基础 venv；已验证的可选路径见 [docs/06](docs/06-runtime-paths-and-offline.zh.md)。

完整列表见 [docs/03 §已知问题](docs/03-speech-to-speech-status.md) 和 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

## 相关仓库

- [kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app) — Fork 并修改的 Reachy Mini 对话 app
- [kamjin3086/reachymini-debug-skill](https://github.com/kamjin3086/reachymini-debug-skill) — 调试工作流打包的 agent skill

## 致谢

- [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) — 核心 STS 管道
- [FunASR/Paraformer](https://github.com/modelscope/FunASR) — 中文 ASR
- [Qwen3-TTS](https://huggingface.co/Qwen) — TTS
- [llama-swap](https://github.com/mostlygeek/llama-swap) —— OpenAI 兼容的 LLM 代理 / 模型切换
- [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade) —— AMD 优化的本地 AI server（Strix Halo iGPU + Ryzen AI NPU）
- [AMD TheRock](https://github.com/ROCm/TheRock) — gfx1151 PyTorch wheels
- [Pollen Robotics](https://www.pollen-robotics.com/reachy-mini/) — Reachy Mini 硬件
- [Hugging Face 博客：Local Reachy Mini Conversation](https://huggingface.co/blog/local-reachy-mini-conversation) — 起点灵感
- [Reddit r/LocalLLaMA：Reachy Mini Goes Fully Local](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/) — 社区参考

## 许可

本文档与脚本以 [CC-BY-4.0](LICENSE) 许可发布。你可以自由复制、修改、商用，**只需保留原作者署名**。如果你基于这些内容做出了改进，欢迎提 PR 或 issue 链接回来。

本文涉及到的第三方软件保留各自的许可，详见 [LICENSE](LICENSE) 文件的拆分说明。
