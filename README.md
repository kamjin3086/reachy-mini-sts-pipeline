# Reachy Mini + Local Voice Conversation: Full Deployment

> A complete record of getting a **fully local** Chinese-English voice conversation pipeline running on a [Reachy Mini Lite](https://www.pollen-robotics.com/reachy-mini/) robot — from initial CPU attempt to a ROCm GPU-accelerated deployment.

[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ROCm](https://img.shields.io/badge/ROCm-7.13-ED1C24)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-gfx1151%20(Strix%20Halo)-76B900)](https://www.amd.com/en/products/processors/ryzen-ai)
[![中文](https://img.shields.io/badge/中文-README.zh.md-blue)](README.zh.md)

---

## What this is

In **early June 2026**, I assembled a Reachy Mini Lite robot — hardware took about **3 hours** — then spent **2 days of free time** debugging the pipeline, configuring the environment, and modifying the official Reachy Mini conversation app. The goal was simple: a fully local, real-time voice conversation in Chinese and English, with zero cloud API calls in the loop.

This repository documents the entire journey — installation, pitfalls, performance baselines, and the eventual working setup — so others with similar needs (local STS, AMD GPU + Linux, Reachy Mini integration) can skip the parts I had to figure out the hard way.

### Key results

| Metric | Value |
|---|---|
| End-to-end perceived latency (steady state) | **~1.0 s** (user stops speaking → first synthesized audio) |
| ASR speed | **35×** real-time (Chinese, Paraformer-zh) |
| LLM first-token latency (steady state) | **50 ms** (Gemma-4-E4B-instruct) |
| TTS synthesis (steady state) | Single-digit seconds |
| Chinese support | ✅ Excellent ASR; TTS is English-leaning (use CosyVoice 2 for production) |
| Fully offline | ✅ Zero cloud calls |

Cold-start first request takes ~29 s (model load + TTS CUDA graph capture). For production, add a warmup request to push the cold-start cost off the user's critical path — see [docs/03 §Performance Baseline](docs/03-speech-to-speech-status.md).

### The story

I followed the excellent [Hugging Face blog "Local Reachy Mini Conversation"](https://huggingface.co/blog/local-reachy-mini-conversation) and the inspiring [r/LocalLLaMA thread "Reachy Mini Goes Fully Local"](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/) as a starting point. My first attempt was CPU-only: Qwen3-TTS does not support AMD GPUs, so I switched to Kokoro. The pipeline ran, but the Reachy Mini app connected and then did absolutely nothing — no logs, no VAD, no audio. I wrote up the symptoms (the draft is preserved in [docs/04](docs/04-reachy-mini-debug-journey.md) as a debugging reference).

Once I got my hands on an **AMD Strix Halo 128G** (Ryzen AI Max+ 395, Radeon 8060S iGPU, gfx1151) workstation, I re-researched ROCm compatibility, solved a series of AMD-specific issues, and ran the whole stack on GPU. That second pass is what this repo documents.

Along the way I also:

- Forked and modified the official Reachy Mini conversation app: **[kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app)** — install it with `pip install -e .` so source edits take effect live (no reinstall loop)
- Packaged my debugging workflow as a reusable agent skill: **[kamjin3086/reachymini-debug-skill](https://github.com/kamjin3086/reachymini-debug-skill)**

## Requirements

- **GPU**: AMD Strix Halo / Strix Point (gfx1151 / gfx1150), or any ROCm 7.13+ supported GPU. NVIDIA GPUs work for the pipeline itself, but the installation steps below are ROCm-specific.
- **OS**: Fedora 44 (other modern Linux distros should work with minor adjustments)
- **RAM**: 32 GB+ recommended (LLM loading needs ~8 GB VRAM)
- **Python**: 3.12+ (managed with [uv](https://github.com/astral-sh/uv))
- **Robot** (optional, only for Reachy Mini integration): Reachy Mini Lite, controlled via Reachy Mini Control

## Quick start

```bash
# 1. Install ROCm + PyTorch — see docs/01
# 2. Stand up an OpenAI-compatible LLM backend (default: llama-swap at http://127.0.0.1:8101/v1)
# 3. Clone this repo
git clone https://github.com/kamjin3086/reachy-mini-sts-pipeline.git
cd reachy-mini-sts-pipeline

# 4. Install speech-to-speech and its dependencies
pip install "speech-to-speech[paraformer]" funasr qwen-tts faster-qwen3-tts hf_transfer

# 5. Adjust --model_name in the start script to a model your llama-swap knows about
$EDITOR scripts/sts_start.sh

# 6. Launch
./scripts/sts_start.sh
# → WebSocket server listens on ws://0.0.0.0:8765/v1/realtime
```

For Reachy Mini integration, also see the forked app: [kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app).

## Documentation

| Document | Purpose | When to read |
|---|---|---|
| [docs/01 — ROCm gfx1151 PyTorch Install](docs/01-rocm-gfx1151-pytorch-install.md) | ROCm 7.13 + TheRock gfx1151 wheels + PyTorch | **First-time** deployment |
| [docs/02 — STS Pipeline Install](docs/02-speech-to-speech-install.md) | speech-to-speech installation + STT/TTS/LLM selection rationale | After doc 01 |
| [docs/03 — Runtime Status & Tuning](docs/03-speech-to-speech-status.md) | Current state, performance baselines, tuning, known issues | After deployment, when optimizing |
| [docs/04 — Reachy Mini Debug Journey](docs/04-reachy-mini-debug-journey.md) | The original "nothing happens" debugging record | When Reachy Mini connection fails |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Quick symptom → fix lookup | When something breaks |
| [README.zh.md](README.zh.md) | This document in Chinese | 中文读者 |

## Benchmarking

```bash
# End-to-end pipeline benchmark
python3 scripts/bench_sts_pipeline.py --quick

# LLM TTFT comparison across models
python3 scripts/bench_llm_models.py
```

After benchmarking 7 LLM models, **Gemma-4-E4B-instruct** is the steady-state TTFT champion (50 ms). NPU-accelerated models were actually slower. Full data in [docs/03 §Tuning §1](docs/03-speech-to-speech-status.md).

## Component selection

| Component | Choice | Alternatives considered | Why |
|---|---|---|---|
| ASR | Paraformer-zh (FunASR) | SenseVoice, faster-whisper | Best Chinese CER (1.95%) |
| LLM | Gemma-4-E4B-instruct | GPT-OSS-20B, Qwen3.6-35B-A3B | Steady-state TTFT king |
| TTS | Qwen3-TTS (CustomVoice) | Kokoro, CosyVoice 2 | ROCm compatibility verified |
| LLM gateway | llama-swap | vLLM, SGLang | Lightweight, fast model switching |
| Denoising | DeepFilterNet 0.5.6 | RNNoise | High quality, patched for torchaudio 2.10 |

## Known issues & workarounds

- **MPS bug**: `speech_to_speech/paraformer_handler.py:56` calls `torch.mps.empty_cache()` unconditionally and crashes on ROCm/CUDA. One-line `sed` fix provided.
- **HSA override side effect**: `HSA_OVERRIDE_GFX_VERSION=11.0.0` actually triggers `hipErrorInvalidImage` on ROCm 7.13 (TheRock). **Just remove it.**
- **DeepFilterNet + torchaudio 2.10**: `df/io.py` references `torchaudio.backend.common` which TheRock 2.10 doesn't ship. `try/except` fallback patch provided.
- **flash-attn on gfx1151**: No upstream HIP kernel support. **Don't install.** Details in [docs/03 §2.a](docs/03-speech-to-speech-status.md).

Full list: [docs/03 §Known Issues](docs/03-speech-to-speech-status.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Related repositories

- [kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app) — Forked Reachy Mini conversation app, modified for this pipeline
- [kamjin3086/reachymini-debug-skill](https://github.com/kamjin3086/reachymini-debug-skill) — Reusable debugging workflow packaged as an agent skill

## Acknowledgments

- [facebookresearch/speech-to-speech](https://github.com/facebookresearch/speech-to-speech) — Core STS pipeline
- [FunASR/Paraformer](https://github.com/modelscope/FunASR) — Chinese ASR
- [Qwen3-TTS](https://huggingface.co/Qwen) — TTS
- [llama-swap](https://github.com/mostlygeek/llama-swap) — Lightweight LLM gateway
- [AMD TheRock](https://github.com/ROCm/TheRock) — gfx1151 PyTorch wheels
- [Pollen Robotics](https://www.pollen-robotics.com/reachy-mini/) — Reachy Mini hardware
- [Hugging Face — Local Reachy Mini Conversation](https://huggingface.co/blog/local-reachy-mini-conversation) — Original inspiration
- [Reddit r/LocalLLaMA — Reachy Mini Goes Fully Local](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/) — Community reference

## License

Documentation and scripts in this repository are released under [CC-BY-4.0](LICENSE) — free to copy, modify, and use commercially, as long as attribution is preserved. If you build something better on top of this, a PR or an issue link back is appreciated.

Third-party software mentioned in this repo retains its own license — see the [LICENSE](LICENSE) file for the breakdown.
