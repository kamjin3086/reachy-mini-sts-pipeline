# Speech-to-Speech Runtime Status & Tuning Guide

> Last updated: 2026-06-04 (v2 — corrected LLM TTFT measurements + LLM model selection)
> Applicable versions: speech-to-speech 0.2.9, funasr 1.3.9, qwen-tts 0.1.1, faster-qwen3-tts 0.2.6, deepfilternet 0.5.6
> Related: [02 — STS Pipeline Install](02-speech-to-speech-install.md), [01 — ROCm gfx1151 PyTorch Install](01-rocm-gfx1151-pytorch-install.md)
>
> [中文版](03-speech-to-speech-status.zh.md) · [← Back to README](../README.md)

## Table of contents

1. [Current runtime state](#current-runtime-state)
2. [Performance baseline](#performance-baseline)
3. [Benchmarking](#benchmarking)
4. [Tuning directions](#tuning-directions)
5. [Known issues & workarounds](#known-issues--workarounds)
6. [Component alternatives](#component-alternatives)
7. [Day-to-day operations](#day-to-day-operations)
8. [Future work](#future-work)

---

## Current runtime state

### Hardware and software stack

| Component | Current configuration |
|---|---|
| System | Fedora 44 |
| GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, Strix Halo) |
| Shared memory | UMA, 124 GB |
| venv | `/home/kamjin/apps/.venv` (uv-managed, no pip) |
| Python | 3.12.13 |
| PyTorch | 2.10.0+rocm7.13.0a20260513 (TheRock gfx1151 wheels) |
| ROCm/HIP | 7.13.26183 |
| LLM backend (proxy) | **llama-swap** at `http://127.0.0.1:8101/v1` (OpenAI-compatible, fast model switching) |
| LLM backend (inference) | **lemonade** ([lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade)) — AMD-optimized, has gfx1151 ROCm + Vulkan paths for Strix Halo iGPU and Ryzen AI NPU |
| Active LLM model | `Gemma-4-E4B-instruct` |
| numpy | 1.26.4 (downgraded by deepfilternet, torch still works) |
| DeepFilterNet | 0.5.6 (patched for torchaudio compat) |
| flash-attn | Not installed in the base venv; optional Qwen3-TTS FastAPI path verified in [docs/06](06-runtime-paths-and-offline.zh.md) |

### Start script

`scripts/sts_start.sh`:

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

After launch, all components warm up and the WebSocket server listens on `ws://0.0.0.0:8765/v1/realtime`.

### Model and cache paths

| Content | Path |
|---|---|
| Paraformer STT | `~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/` |
| Qwen3-TTS | HF Hub cache (downloaded on demand) |
| Qwen3-TTS default voices | `CustomVoice` presets (Aiden / Vivian / etc.) |
| nltk_data | `~/nltk_data` |

---

## Performance baseline

Test environment: Fedora 44, ROCm 7.13 (TheRock). Audio is synthesized low-amplitude random noise (latency measurement only, not real speech).

> **Cold start vs steady state**: All data below **distinguishes** these two states. Cold start = the **first** request after server boot (models loaded into VRAM, CUDA graph capture, KV cache warmup). Steady state = 2nd request onwards.

### Per-component latency

| Component | Test condition | Latency | Notes |
|---|---|---|---|
| **STT** | 1 s audio | 0.054 s | RTF 18.6× |
| **STT** | 2 s audio | 0.062 s | RTF 32.4× |
| **LLM cold-start TTFT** | First request | **16.97 s** | lemonade loads the model into VRAM (via llama-swap) |
| **LLM steady-state TTFT** | 2nd+ request | **0.05 s** | KV cache warm, **essentially free** |
| **LLM steady-state throughput** | 60 tokens | ~37 tok/s | Acceptable for real-time conversation |
| **TTS TTFA** | Steady state | ~0.84 s | First call includes CUDA graph compile (**12–14 s**) |
| **TTS Total, first** | Cold start | 70.5 s | Model load + CUDA graph capture |
| **TTS Total, steady state** | 2nd+ request | Single-digit seconds | CUDA graph cache hit |

### End-to-end perceived latency

**Perceived latency = STT + LLM_TTFT + TTS_TTFA** (user stops speaking → first synthesized audio plays)

| State | STT | LLM_TTFT | TTS_TTFA | **E2E perceived** | TTS_Total |
|---|---|---|---|---|---|
| **Cold start** (1st request) | 0.085 s | 16.97 s | 12–14 s | **~29 s** | 70.5 s (incl. graph compile) |
| **Steady state** (2nd+) | 0.085 s | 0.05 s | 0.84 s | **~0.97 s** | Single-digit seconds |

> **Key observations (corrected by measurement 2026-06-04)**:
> - The "LLM_TTFT 3.26 s" figure in earlier docs was a cold-start + first-request hybrid, **not** steady state. Steady-state TTFT is in fact only **~50 ms** — barely noticeable in conversation.
> - Steady-state E2E perceived latency is **~1 s**, already close to "natural conversation" territory.
> - Cold-start costs are dominated by **TTS CUDA graph compile (12–14 s)** and **LLM first load (~17 s)**.
> - **Production recommendation**: issue a warmup request right after server start to push cold-start cost off the user's critical path.

### Bottleneck breakdown (steady state)

```
E2E steady-state perceived latency (0.97 s) breakdown:
├── STT              0.085 s  (  9%)  ← very fast, near optimal
├── LLM TTFT         0.050 s  (  5%)  ← steady state, essentially free
└── TTS TTFA         0.840 s  ( 86%)  ← current main bottleneck
```

---

## Benchmarking

### Quick test

```bash
source /home/kamjin/apps/.venv/bin/activate
cd /home/kamjin/projects/reachy-mini-sts-pipeline

# All components (1-2 test points each, ~1 min)
python3 scripts/bench_sts_pipeline.py --quick

# Single component only
python3 scripts/bench_sts_pipeline.py --only stt
python3 scripts/bench_sts_pipeline.py --only llm
python3 scripts/bench_sts_pipeline.py --only tts
python3 scripts/bench_sts_pipeline.py --only e2e
```

### Full test

```bash
# STT 1/2/5/10s × 2 rounds, LLM/TTS 5 texts × 2 rounds, E2E 3 rounds
# ~5-10 min (first run includes model loading)
python3 scripts/bench_sts_pipeline.py
```

### Script output

`scripts/bench_sts_pipeline.py` outputs:

- **GPU info** — PyTorch / ROCm / device / VRAM / arch
- **STT** — Duration / Latency / RTF table
- **LLM** — TTFT / Total / estimated tokens / tok/s table
- **TTS** — TTFA / Total / Audio length / RTF table; saves WAVs to `~/apps/sts-cache/bench/sts_bench/`
- **E2E** — STT / LLM_TTFT / LLM_Total / TTS_TTFA / TTS_Total / E2E perceived latency table

### Validating audio quality

TTS tests auto-save generated audio to `~/apps/sts-cache/bench/sts_bench/tts_N.wav` — play them to evaluate quality.

For real STT quality, record Chinese with `arecord` and feed the WAV into the script:

```bash
arecord -f S16_LE -r 16000 -d 3 test.wav
# Modify the script to load test.wav instead of gen_silent_wav()
```

### LLM model TTFT comparison

`scripts/bench_llm_models.py` measures TTFT across multiple LLM models. Use this when evaluating alternatives to `Gemma-4-E4B-instruct`.

---

## Tuning directions

Ordered by expected benefit.

### 1. Reduce LLM TTFT (marginal benefit remaining)

**Measured finding**: `Gemma-4-E4B-instruct` steady-state TTFT is only **50 ms** — essentially free. Switching models is **not** worthwhile.

**LLM model comparison** (2026-06-04, llama-swap :8101 → lemonade, 3-prompt average):

| Model | Cold start | Steady-state TTFT | tok/s | Notes |
|---|---|---|---|---|
| **Gemma-4-E4B-instruct (current)** | 16.97 s | **0.05 s** | 37.1 | TTFT champion, best price/perf |
| GPT-OSS-20B | 0.46 s | 0.39 s | 64.8 | 2× throughput, 8× slower TTFT |
| Qwen3.6-35B-A3B-instruct (MoE) | 28.9 s | 0.18 s | 45.3 | Balanced; better for long answers |
| Qwen3.5-4b-FLM-instruct (NPU) | 8.36 s | 1.20 s | 12.7 | **NPU is actually slower** |
| Step-3.5-Flash-normal | 81.9 s | 3.5 s+ | 0 | Returns 0 tokens, broken |
| Gemma-4-E2B (instruct/base/thinking) | — | — | — | **Fails to start**: `upstream command exited prematurely` |

**Conclusion**: `Gemma-4-E4B-instruct` is already optimal for steady-state TTFT. **Don't switch.**

If you really want to optimize, these are more effective than switching models:

**a) Enable prompt prefix cache** (recommended)

lemonade (and other inference backends like vLLM) supports reusing the KV cache for identical system prompts. speech-to-speech sends the same system prompt every call, so this gives **~30% TTFT gain** (50 ms → 35 ms — small in absolute terms, but easy). Configure on the lemonade / llama-swap side.

**b) Warm up the LLM at startup** (recommended)

Push lemonade's 17 s cold-start cost (via llama-swap) off the user's critical path. See the full warmup script in [§2.c](#c-warm-up-tts--llm--stt-remove-cold-start).

**c) Switch to vLLM / SGLang** (only if llama-swap + lemonade becomes a bottleneck)

Higher throughput and concurrency, but vLLM's gfx1151 compatibility needs evaluation. **Not needed currently**.

**d) Disable thinking / reasoning** (already correct)

Already using `*instruct` variants — no action needed. **Do not** switch to `*-thinking` variants — adds ~5× generation time.

### 2. TTS acceleration

**a) flash-attention — base venv skipped; optional isolated path now verified**

Update on 2026-06-06: the base venv still deliberately skips flash-attn, but a Qwen3-TTS FastAPI path using flash-attn's AMD Triton backend has been verified. See [docs/06](06-runtime-paths-and-offline.zh.md). The older notes below explain why it should not be a base dependency.

Based on [ROCm/TheRock#1364](https://github.com/ROCm/TheRock/issues/1364) and [TesslateAI/FlashAttentionDist](https://github.com/TesslateAI/FlashAttentionDist):

| Source | gfx1151 support | Notes |
|---|---|---|
| Official PyPI `flash-attn` | ❌ No wheels, only `.tar.gz` | Requires 30-60 min source compile |
| TesslateAI prebuilt | ❌ gfx90a / gfx942 / gfx950 only | Datacenter GPUs |
| TheRock PyTorch SDPA flash | ❌ Disabled at runtime | `hip/sdp_utils.cpp:961` actively disables |
| Self-built AOTriton 0.11beta | ⚠️ Experimental | "Slower than nightly", requires recompiling PyTorch |

**Measured** on TheRock 2.10.0+rocm7.13 wheels:

```python
import torch
print(hasattr(torch, 'ops') and hasattr(torch.ops, 'aotriton'))  # True (C++ registered)
import importlib.util
print(importlib.util.find_spec('pyaotriton'))  # None (Python runtime not packaged)
```

The `pyaotriton` Python package is **absent** from the wheel (only C++ ops registered), so `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables the ops but provides no backend. SDPA falls back to the math path, which is **0.20 ms @ [2,8,64,64] fp16** — already fast enough.

**Base-path conclusion**: skip flash-attn in the main venv. Use the pinned path in [docs/06](06-runtime-paths-and-offline.zh.md) when optimizing Qwen3-TTS runtime throughput.

**b) Switch to a lighter TTS**

Qwen3-TTS 1.7B has limited Chinese support (the default `CustomVoice` leans English). Alternatives:

- `kokoro` — faster, but Chinese needs an extra voice model
- `CosyVoice` — better Chinese, ROCm compatibility unverified

Edit `sts_start.sh`:

```bash
--tts kokoro  # requires pip install "speech-to-speech[kokoro]"
```

**c) Warm up TTS + LLM + STT (eliminate cold start)**

Call each component once at startup, removing cold-start from the user's critical path:

```bash
# Append to sts_start.sh
(
  sleep 5  # wait for server to come up
  # STT warmup
  python3 -c "
import os; os.environ.pop('OPENAI_API_KEY', None)
import torch
if hasattr(torch, 'mps') and not torch.backends.mps.is_available():
    torch.mps.empty_cache = torch.cuda.empty_cache
from speech_to_speech import ParaformerASR
asr = ParaformerASR()
print('STT warmup OK')
" 2>&1 | tail -3
  # LLM warmup
  curl -s -X POST http://127.0.0.1:8101/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Gemma-4-E4B-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    > /dev/null
  # TTS warmup (first call triggers CUDA graph compile, ~12-14 s)
  python3 -c "
import os; os.environ.pop('OPENAI_API_KEY', None)
from speech_to_speech import Qwen3TTS
tts = Qwen3TTS()
print('TTS warmup OK')
" 2>&1 | tail -3
) &
```

**Measured cold-start cost**: STT ~3 s + LLM ~17 s + TTS ~14 s ≈ **34 s**. After warmup, user-perceived latency enters steady state immediately.

### 3. STT optimization

STT is already 18–32× real-time with negligible latency; little room to optimize. For better Chinese accuracy, consider:

- `SenseVoice-Small` (2.96% CER, smaller, faster) — change `sts_start.sh` to `--stt sensevoice`
- Mixed-language scenarios: `paraformer` + `whisper` dual-model fallback

### 4. Streaming optimization (advanced)

Current TTS uses `non_streaming_mode=True` (synthesizes the full text at once). Switching to streaming (synthesize as LLM chunks arrive) cuts perceived latency another 30–50%:

```python
# In bench_sts_pipeline.py
setup_kwargs={"non_streaming_mode": False, "streaming_chunk_size": 8}
```

The start script needs a custom wrapper (speech-to-speech CLI doesn't expose this directly). Requires modifying `s2s_pipeline.py`.

### 5. System-level optimizations

| Item | Status | Notes |
|---|---|---|
| `GPU_MAX_ALLOC_PERCENT=100` | ✅ Set | Allow full VRAM allocation |
| `GPU_MAX_HEAP_SIZE=100` | ✅ Set | Cap HIP heap |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | ✅ Set | Enable AOTriton compile (no backend in practice) |
| SoX | ⚠️ Not installed | Optional, audio format conversion |
| DeepFilterNet | ✅ Installed | With torchaudio compat patch, see below |

Install optional dependencies:

```bash
sudo dnf install sox        # audio format tools
# DeepFilterNet — see dedicated section below
```

---

## Known issues & workarounds

### 1. `hipErrorInvalidImage` (fixed)

`HSA_OVERRIDE_GFX_VERSION=11.0.0` actually causes ROCm 7.13 kernel mismatches.

**Fix**: Remove the variable from `sts_start.sh`. See [01-rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md).

### 2. `hf_transfer` missing (fixed)

TTS model downloads fail without `hf_transfer`.

**Fix**:

```bash
uv pip install hf_transfer
```

### 3. Paraformer in-tree MPS bug (workaround in test script)

`speech_to_speech/paraformer_handler.py:56` calls `torch.mps.empty_cache()` unconditionally and crashes on ROCm/CUDA.

**Current workaround**: monkey-patch in the test script header (see `bench_sts_pipeline.py:35-38`). Production needs the source file patched or speech-to-speech upgraded.

**Permanent fix**: file an issue upstream, or apply locally:

```bash
sed -i 's/torch.mps.empty_cache()/torch.cuda.empty_cache()/' \
  /home/kamjin/apps/.venv/lib64/python3.12/site-packages/speech_to_speech/STT/paraformer_handler.py
```

### 4. MIOpen workspace warning (harmless)

```
MIOpen(HIP): Warning [IsEnoughWorkspace] Solver <GemmFwdRest>, workspace required: 41287680, ...
```

MIOpen's GEMM solver workspace estimate is off; no functional impact. Suppress with `MIOPEN_LOG_LEVEL=3`.

### 5. Limited Chinese TTS voices

Qwen3-TTS `CustomVoice` defaults to English / international voices (`Aiden`, `Vivian`, etc.); Chinese voices sound mediocre.

**Suggestion**: tune the `instruct` parameter for Chinese, or evaluate CosyVoice for native Chinese quality.

### 6. No WebSocket client implemented

`ws://0.0.0.0:8765/v1/realtime` needs an OpenAI Realtime API-compatible client. Common options:

- Web: LiveKit, Pipecat
- Desktop: see the speech-to-speech repo examples
- **Custom**: the [kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app) fork in this project

### 7. DeepFilterNet + torchaudio 2.10 compat patch

**Problem**: deepfilternet 0.5.6 references `torchaudio.backend.common.AudioMetaData` in `df/io.py:9`, but the TheRock-built `torchaudio 2.10.0+rocm7.13` **removed the `torchaudio.backend` subpackage**, so import fails:

```python
>>> from df.enhance import init_df
ModuleNotFoundError: No module named 'torchaudio.backend'
```

**Current workaround**: `try/except` fallback added in `df/io.py` so the import degrades to `Any`:

```python
try:
    from torchaudio.backend.common import AudioMetaData  # type: ignore[attr-defined]
except ImportError:
    AudioMetaData = Any  # type: ignore[assignment,misc]
```

**Verified**: 1 s audio denoised end-to-end on ROCm:

```
input:  shape=[1, 48000], peak=1.766
output: shape=[1, 48000], peak=0.318  (denoising succeeded)
RT:     2.33 s (first call, includes cuDNN compile; subsequent < 0.5 s)
```

**Re-apply after reinstall** (when upgrading deepfilternet or recreating the venv):

```bash
# One-liner sed fix
sed -i 's/^from torchaudio.backend.common import AudioMetaData$/try:\n    from torchaudio.backend.common import AudioMetaData  # type: ignore[attr-defined]\nexcept ImportError:\n    AudioMetaData = Any  # type: ignore[assignment,misc]/' \
  /home/kamjin/apps/.venv/lib64/python3.12/site-packages/df/io.py
```

Or manually edit `df/io.py`, replacing line 9 with the `try/except` block above.

**Permanent fix**: file an issue/PR upstream, or wait for a deepfilternet release that supports torchaudio 2.10+.

### 8. flash-attn not installed (decision record)

See [Tuning §2.a](#a-flash-attention--not-recommended-for-gfx1151). Rationale: gfx1151 has no upstream HIP flash kernel support; forcing a 30+ min compile would likely produce a useless package and risk breaking the venv.

---

## Component alternatives

### STT alternatives

| Model | Chinese CER | Speed | ROCm | Notes |
|---|---|---|---|---|
| **Paraformer-zh** (current) | 1.95% | 18-32× | ✅ | Best Chinese, good library compat |
| SenseVoice-Small | 2.96% | ~170× | ✅ | Smaller, faster, multilingual |
| faster-whisper large-v3 | 5.14% | 9× | ⚠️ | Multilingual general, weaker Chinese |

### TTS alternatives

| Model | Chinese quality | Speed | ROCm | Notes |
|---|---|---|---|---|
| **Qwen3-TTS** (current) | Medium | Slow | ✅ | Library default, English-leaning |
| Kokoro | Medium | Fast | ✅ | Needs extra voice for Chinese |
| CosyVoice 2 | High | Medium | ⚠️ Untested | Native Chinese, needs ROCm eval |
| ChatTTS | Medium | Medium | ⚠️ | Requires `pip install speech-to-speech[chattts]` |

### LLM alternatives

All models registered in llama-swap (which delegates to lemonade) are available; edit `--model_name` in `sts_start.sh` (a **restart of the pipeline is required** after the change).

**Measured data** (2026-06-04, 3-prompt average, steady state):

| Model | Steady-state TTFT | tok/s | Recommended for |
|---|---|---|---|
| **Gemma-4-E4B-instruct (current)** | **0.05 s** | 37.1 | Default — TTFT king for conversation |
| GPT-OSS-20B | 0.39 s | 64.8 | Long answers (faster throughput), TTFT slightly worse |
| Qwen3.6-35B-A3B-instruct | 0.18 s | 45.3 | Balanced, better for long answers |
| Qwen3.6-27B-instruct | — | — | Slow (27B), steady state not measured |
| Gemma-4-E2B-instruct | — | — | ❌ **Fails to start** |
| Gemma-4-E2B-thinking | — | — | ❌ **Fails to start** |
| Qwen3.5-4b-FLM-instruct | 1.20 s | 12.7 | ❌ NPU is actually slower |
| Qwen3.5-9b-FLM-instruct | — | — | ⚠️ NPU consistently slow, not measured in detail |
| Step-3.5-Flash-normal | 3.5 s+ | 0 | ❌ Returns 0 tokens, broken |
| OmniCoder-9B | — | — | Code-specialized |
| MiroThinker-1.7-mini | — | — | Agentic, tool-calling |

**To switch**:

```bash
# Stop the old pipeline first. llama-swap does not need to restart.
pkill -f speech-to-speech
sleep 2

# Prefer an environment override instead of editing the script.
STS_LLM_MODEL=GPT-OSS-20B ./scripts/sts_start.sh
```

---

## Day-to-day operations

### Start / stop

```bash
# Start
./sts_start.sh

# Stop
pkill -f speech-to-speech

# Check status
curl -s http://127.0.0.1:8765/v1/realtime  # WebSocket (needs ws client)
ps aux | grep speech-to-speech
```

### Log location

speech-to-speech outputs to stdout/stderr. Recommended to redirect:

```bash
mkdir -p /home/kamjin/apps/sts-cache/logs
nohup ./scripts/sts_start.sh > /home/kamjin/apps/sts-cache/logs/sts.log 2>&1 &
```

### LLM stack health check (llama-swap + lemonade)

```bash
curl -s http://127.0.0.1:8101/v1/models | jq '.data[].id'
```

### Restart pipeline

```bash
pkill -f speech-to-speech
sleep 2
./sts_start.sh
```

### Reset model cache (on error)

```bash
# Inspect cache
ls ~/.cache/modelscope/hub/
ls ~/.cache/DeepFilterNet/   # DeepFilterNet 3 weights
ls ~/.cache/huggingface/hub/

# Clear (will re-download next launch)
rm -rf ~/.cache/modelscope/hub/iic/
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS*
rm -rf ~/.cache/DeepFilterNet/  # DeepFilterNet 3 weights
```

### Package dependency notes (important)

`deepfilternet 0.5.6` forced two downgrades on install, **torch still works**:

| Package | Current | Previous | Reason |
|---|---|---|---|
| numpy | 1.26.4 | 2.4.x | deepfilternet pins `numpy<2` |
| packaging | 23.2 | 25+ | deepfilternet indirect dependency |

**Verified**: torch 2.10.0+rocm7.13 GPU compute, Paraformer, Qwen3-TTS all work.

**To restore numpy 2.x** (e.g. when upgrading torch):

```bash
source /home/kamjin/apps/.venv/bin/activate
uv pip install "numpy>=2.0" --no-deps
# Verify
python3 -c "import torch; a=torch.randn(10,device='cuda'); print(a.sum().item())"
# Verify deepfilternet
python3 -c "from df.enhance import init_df; init_df()"
```

If deepfilternet breaks but you must use numpy 2.x with torch, pick one.

---

## Future work

### Short term (1–2 weeks)

- [x] ~~Install SoX, DeepFilterNet (optional deps)~~ — DeepFilterNet installed (with patch), SoX not
- [x] ~~Try `Gemma-4-E2B-instruct` for TTFT improvement~~ — **Failed**: all 3 variants (`instruct` / `base` / `thinking`) error with `upstream command exited prematurely`; lemonade backend issue, awaiting fix from config owner
- [x] ~~Benchmark LLM models for TTFT~~ — 7 models tested, **Gemma-4-E4B-instruct is steady-state TTFT optimal**, do not switch
- [ ] File issue on lemonade maintainer (Gemma-4-E2B startup failure)
- [ ] Tune lemonade / llama-swap inference parameters (temperature, repetition_penalty, etc.)
- [ ] Evaluate vLLM / SGLang as lemonade replacement
- [x] ROCm flash-attn optional path — base venv skipped; Qwen3-TTS FastAPI path verified in [docs/06](06-runtime-paths-and-offline.zh.md)
- [ ] ~~AOTriton kernels~~ — **No Python backend**: C++ ops registered but Python runtime missing
- [ ] Wait for speech-to-speech upstream to fix MPS bug, remove monkey-patch
- [ ] Upgrade to Qwen3-TTS VoiceDesign / Base models for custom voices
- [ ] Wait for TheRock wheels to fix `pyaotriton` Python packaging

---

## Quick troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Startup error `hipErrorInvalidImage` | HSA override lingering | Remove `HSA_OVERRIDE_GFX_VERSION` |
| `HF_HUB_ENABLE_HF_TRANSFER` error | hf_transfer not installed | `uv pip install hf_transfer` |
| OpenAI client reports missing API key | env conflict | Make sure `OPENAI_API_KEY` is unset |
| Paraformer reports MPS error | library bug | Use the patch script or monkey-patch |
| DeepFilterNet reports `torchaudio.backend` not found | 0.5.6 incompatible with torchaudio 2.10 | Apply [§7 patch](#7-deepfilternet--torchaudio-210-compat-patch) |
| MIOpen workspace warning | estimate off | Ignore, or `MIOPEN_LOG_LEVEL=3` |
| LLM 401 invalid_api_key | client env interference | Confirm `OPENAI_API_KEY=""` or unset |
| First TTS call 12 s+ | CUDA graph compile | Warm up (cannot be eliminated) |
| SoX warning | sox not installed | `sudo dnf install sox` |
| `pip` points to Python 3.14, not venv | `~/.local/bin/pip` in PATH takes precedence | Use `uv pip` or `/home/kamjin/apps/.venv/bin/python -m pip` |

---

## References

- Repository: [speech-to-speech](https://github.com/facebookresearch/speech-to-speech)
- Install doc: [02-speech-to-speech-install.md](./02-speech-to-speech-install.md)
- ROCm doc: [01-rocm-gfx1151-pytorch-install.md](./01-rocm-gfx1151-pytorch-install.md)
- Benchmark scripts: `scripts/bench_sts_pipeline.py` (end-to-end), `scripts/bench_llm_models.py` (LLM TTFT comparison)
- Start script: `scripts/sts_start.sh`
- llama-swap (proxy): `http://127.0.0.1:8101/v1`
- lemonade (inference backend): AMD TheRock + Vulkan paths
