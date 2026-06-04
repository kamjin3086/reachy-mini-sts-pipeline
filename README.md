# Reachy Mini + 本地语音对话完整部署

> 在 [Reachy Mini Lite](https://www.pollen-robotics.com/reachy-mini/) 上跑通**纯本地**中英文语音对话的完整记录 — 从 CPU 失败到 ROCm GPU 加速的部署全过程。

[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ROCm](https://img.shields.io/badge/ROCm-7.13-ED1C24)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-gfx1151%20(Strix%20Halo)-76B900)](https://www.amd.com/en/products/processors/ryzen-ai)

---

## 这个项目是什么

我在 2026 年 5-6 月把 Reachy Mini Lite 机器人和**完全本地**的语音对话 pipeline 拼起来的过程记录。目标是：

- 🎤 **语音输入** — 实时 ASR（支持中文 + 英文）
- 🧠 **本地 LLM** — 推理不依赖任何云服务
- 🔊 **语音输出** — 流畅自然的 TTS
- 🤖 **机器人交互** — 接 Reachy Mini 实现头部/表情/动作联动

整个项目跑在一台 **AMD Strix Halo 128G**（Ryzen AI Max+ 395，Radeon 8060S 集显，gfx1151 架构）的工作站上。

## 关键结果

| 指标 | 数值 |
|---|---|
| 端到端稳态感知延迟 | **~1.0 秒**（用户停嘴 → 听到第一声合成音）|
| ASR 速度 | **35x** 实时（中文 Paraformer）|
| LLM 首 token 延迟（稳态） | **50 ms**（Gemma-4-E4B-instruct）|
| TTS 合成速度 | **个位数秒级**（稳态）|
| 中文支持 | ✅ ASR 优秀 / TTS 英文向（建议生产换 CosyVoice 2）|
| 完全离线运行 | ✅ 无任何云 API 调用 |

冷启动首请求 ~29s（模型加载 + TTS CUDA graph 编译）。**生产建议**加预热请求把冷启动成本从用户路径中移除，详见 [docs/03-speech-to-speech-status.md](docs/03-speech-to-speech-status.md)。

## 故事线

> 5 月初我组装好 Reachy Mini Lite，参考了 Hugging Face 的 [Local Reachy Mini Conversation](https://huggingface.co/blog/local-reachy-mini-conversation) 博客开始搭建。第一阶段纯 CPU，Qwen3-TTS 不支持 AMD GPU 才发现，改 Kokoro 跑通了——但 Reachy Mini App 连上后"什么都不发生"。我把问题写下来准备发 Reddit 求助。

> 后来入手了 Strix Halo 128G GPU 工作站，重新调研、解决了所有 ROCm + AMD 兼容性问题，最终跑通完整 GPU 加速管道。**那个最初"什么都不发生"的 debug 旅程完整记录在 [docs/04](docs/04-reachy-mini-debug-journey.md)，对其他用 Reachy Mini + AMD GPU 的人会有参考价值**。

## 硬件 / 软件要求

- **GPU**：AMD Strix Halo / Strix Point (gfx1151 / gfx1150) 或其他 ROCm 7.13+ 支持的 GPU
- **系统**：Fedora 44（也适用其他主流 Linux 发行版）
- **内存**：建议 32 GB+（LLM 加载需要 ~8 GB VRAM）
- **Python**：3.12+（用 [uv](https://github.com/astral-sh/uv) 管理 venv）
- **机器人**：Reachy Mini（Lite 或标准版），通过 Reachy2 Control 连接

> 理论上非 AMD GPU + CUDA 也能跑（speech-to-speech 默认 CUDA），但本文档的安装步骤和踩坑都是基于 ROCm/gfx1151。

## 快速开始

```bash
# 1. 装 ROCm + PyTorch（详见 docs/01）
# 2. 准备 llama-swap 或 OpenAI 兼容 LLM 后端（默认 http://127.0.0.1:8101/v1）
# 3. 克隆本仓库
git clone https://github.com/kamjin3086/reachy-mini-sts-pipeline.git
cd reachy-mini-sts-pipeline

# 4. 装 speech-to-speech + 可选依赖
pip install "speech-to-speech[paraformer]" funasr qwen-tts faster-qwen3-tts hf_transfer

# 5. 编辑启动脚本里的 --model_name 为你 llama-swap 中可用的模型
$EDITOR scripts/sts_start.sh

# 6. 启动
./scripts/sts_start.sh
# → WebSocket 服务监听 ws://0.0.0.0:8765/v1/realtime

# 7. 在 Reachy2 Control 桌面 App 里填入 IP:端口，连接，开始对话
```

## 文档索引

| 文档 | 用途 | 何时读 |
|---|---|---|
| [docs/01-rocm-gfx1151-pytorch-install.md](docs/01-rocm-gfx1151-pytorch-install.md) | ROCm 7.13 + TheRock gfx1151 wheels + PyTorch 安装 | **第一次**部署必读 |
| [docs/02-speech-to-speech-install.md](docs/02-speech-to-speech-install.md) | speech-to-speech 管道安装 + 选型对比（STT/TTS/LLM） | 看完 01 后 |
| [docs/03-speech-to-speech-status.md](docs/03-speech-to-speech-status.md) | 当前运行状态 + 性能基线 + 调优方向 + 已知问题 | 部署完成后、想调优时 |
| [docs/04-reachy-mini-debug-journey.md](docs/04-reachy-mini-debug-journey.md) | Reachy Mini 调试之旅（最初的"什么都不发生"问题记录）| 遇到 Reachy Mini 连接问题时 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 关键 bug 速查表 | 故障时 |

## 性能测试

```bash
# 端到端 benchmark
python3 scripts/bench_sts_pipeline.py --quick

# 仅 LLM TTFT 对比多个模型
python3 scripts/bench_llm_models.py
```

实测 7 个 LLM 模型后，**`Gemma-4-E4B-instruct` 是稳态 TTFT 最优（50ms）**——不要被 llama-swap 描述里的 "Very Fast" 标签骗了，NPU 加速模型反而更慢。详见 [docs/03 调优方向 §1](docs/03-speech-to-speech-status.md)。

## 核心组件

| 组件 | 选型 | 备选 | 理由 |
|---|---|---|---|
| ASR | Paraformer-zh (FunASR) | SenseVoice, faster-whisper | 中文 CER 1.95% 最优 |
| LLM | Gemma-4-E4B-instruct | GPT-OSS-20B, Qwen3.6-35B-A3B | 稳态 TTFT 之王 |
| TTS | Qwen3-TTS (CustomVoice) | Kokoro, CosyVoice 2 | ROCm 兼容已验证 |
| LLM 网关 | llama-swap | vLLM, SGLang | 轻量、模型切换快 |
| 降噪 | DeepFilterNet 0.5.6 | RNNoise | 高质量、已 patch 兼容 |

## 已知问题与 workaround

- **MPS bug**：`speech_to_speech/paraformer_handler.py:56` 无条件调用 `torch.mps.empty_cache()` 在 ROCm/CUDA 上崩溃。已提供 `sed` 一行修复。
- **HSA override 副作用**：`HSA_OVERRIDE_GFX_VERSION=11.0.0` 在 ROCm 7.13 (TheRock) 上反而触发 `hipErrorInvalidImage`——**直接移除即可**。
- **DeepFilterNet + torchaudio 2.10**：`df/io.py` 的 `torchaudio.backend.common` 引用在 TheRock 2.10 中不存在。已提供 `try/except` fallback patch。
- **flash-attn**：gfx1151 上游无 HIP kernel 优化，**不装**。详见 [docs/03 调优方向 §2.a](docs/03-speech-to-speech-status.md)。

完整列表见 [docs/03 §已知问题](docs/03-speech-to-speech-status.md) 和 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

## 致谢

- [facebookresearch/speech-to-speech](https://github.com/facebookresearch/speech-to-speech) — 核心 STS 管道
- [FunASR/Paraformer](https://github.com/modelscope/FunASR) — 中文 ASR
- [Qwen3-TTS](https://huggingface.co/Qwen) — TTS
- [llama-swap](https://github.com/mostlygeek/llama-swap) — 轻量 LLM 网关
- [AMD TheRock](https://github.com/ROCm/TheRock) — gfx1151 PyTorch wheels
- [Pollen Robotics](https://www.pollen-robotics.com/reachy-mini/) — Reachy Mini
- [Hugging Face 博客：Local Reachy Mini Conversation](https://huggingface.co/blog/local-reachy-mini-conversation) — 启发了这个项目

## 许可

本文档与脚本以 [CC-BY-4.0](LICENSE) 许可发布。你可以自由复制、修改、商用，**只需保留原作者署名**。如果你基于这些内容做出了改进，欢迎提 PR 或 issue 链接回来。
