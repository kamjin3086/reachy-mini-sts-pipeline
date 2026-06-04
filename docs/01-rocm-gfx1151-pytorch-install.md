# Strix Halo (Radeon 8060S) PyTorch ROCm Installation Notes

> [中文版](01-rocm-gfx1151-pytorch-install.zh.md) · [← Back to README](../README.md)

## Hardware

- **System**: Fedora 44
- **GPU**: AMD Ryzen AI Max+ 395 w/ Radeon 8060S Graphics
- **Architecture**: gfx1151 (Strix Halo)
- **Memory**: Shared system memory (UMA), no discrete VRAM
- **PCI ID**: 1002:1586

## The core issue: ROCm support for Strix Halo is unusual

Strix Halo (gfx1151) is an AMD APU (integrated GPU). Its ROCm support has gone through several iterations, and **compatibility varies dramatically across ROCm versions**. Picking the wrong one will segfault (SIGSEGV) immediately.

### Version line overview

ROCm currently has **two parallel release lines**:

| Version line | Positioning | gfx1151 support | Notes |
|---|---|---|---|
| **ROCm 7.0–7.2** (production) | Production-grade, stable | ❌ gfx1151 **not** in compatibility matrix | Traditional monolithic build; stable but APU support lags |
| **ROCm 7.11+** (TheRock) | Technical preview | ✅ **Native** gfx1151 support | New TheRock build system; APU priority |

### Why PyTorch.org's ROCm wheels don't work

```
PyTorch 2.12.0+rocm7.1  ← installed from PyTorch.org
```

- Built on **ROCm 7.1**, which has a **VGPR count bug** for gfx1151 (Issue #2991)
- Symptom: `torch.cuda.is_available()` returns `True`, but **any** GPU tensor allocation SIGSEGVs
- Independent of Python version (3.12 / 3.13 / 3.14 all affected)

### The fix

**You must use TheRock-built wheels**, which ship gfx1151-specific fixes:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
```

## Current install configuration

```
Python:    3.12.13
PyTorch:   2.10.0+rocm7.13.0a20260513
ROCm/HIP:  7.13.26183
Arch:      gfx1151
```

### Install commands

```bash
# 1. Create Python 3.12 virtual environment
uv venv /home/kamjin/apps/.venv --python 3.12

# 2. Install from TheRock gfx1151 index
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

### Verify the install

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('ROCm/HIP:', torch.version.hip)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('Arch:', torch.cuda.get_arch_list())

# Functional test
x = torch.randn(100, 100, device='cuda')
y = torch.randn(100, 100, device='cuda')
z = x @ y
print('Matrix mul: OK')
print('ALL TESTS PASSED')
"
```

## Issues fixed in ROCm 7.13

- **VGPR count bug (Issue #2991)**: ROCm 7.13 (TheRock) fixes gfx1151 VGPR handling natively — no more need for `HSA_OVERRIDE_GFX_VERSION`
- **Side effect of using the override**: `HSA_OVERRIDE_GFX_VERSION=11.0.0` triggers `hipErrorInvalidImage` (kernels compiled for gfx1100 don't match gfx1151 hardware)
- **Test result**:
  - With override → `HIP error: device kernel image is invalid`
  - Without override → all GPU operations work
- **Current config**: `HSA_OVERRIDE_GFX_VERSION` removed; keep `GPU_MAX_ALLOC_PERCENT=100`, `GPU_MAX_HEAP_SIZE=100`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

## Known issues and workarounds

### 1. MIOpen solver database file unreadable

```
MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable:
  ".../gfx1151_20.HIP.fdb.txt"
```

- **Impact**: None — just a solver cache file
- **Cause**: fdb file format in TheRock package may not match the current MIOpen version
- **Action**: Ignore; MIOpen rebuilds automatically

### 2. XNACK warning

```
warning: xnack 'Off' was requested for a processor that does not support it!
```

- **Impact**: None
- **Cause**: gfx1151 doesn't support XNACK, but ROCm requests it by default
- **Action**: Ignore

### 3. VRAM allocation strategy (large-UMA scenario)

With large UMA memory (e.g. 96 GB), PyTorch may skip the discrete-VRAM-style allocator and go straight to shared memory, causing apparent OOM.

- **Workaround**: Set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` or limit per-allocation size

### 4. Kernel version compatibility

- **Linux 6.18.4+**: Requires TheRock wheels or kernel patches for the VGPR fix
- **Linux 6.18.3 and earlier**: ROCm 7.1 also works (the VGPR bug wasn't introduced yet)

## Inference framework compatibility

| Framework | ROCm requirement | Recommended version | Notes |
|---|---|---|---|
| whisper.cpp | GGML HIP | ROCm 7.2+ | Bypasses rocWMMA, 7.2 is enough |
| whisper.pytorch | PyTorch ROCm | TheRock 7.11+ | Needs PyTorch GPU |
| Coqui TTS / XTTS | PyTorch ROCm | TheRock 7.11+ | Has known kernel issues |
| SpeechBrain | PyTorch ROCm | TheRock 7.11+ | — |
| llama.cpp | GGML HIP | ROCm 7.2+ | Verified working |

## Quick reference

### Install

```bash
# Create environment
uv venv /path/to/venv --python 3.12

# Install PyTorch ROCm (gfx1151)
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

### Verify

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "Radeon 8060S Graphics"
assert "gfx1151" in torch.cuda.get_arch_list()
x = torch.randn(10, 10, device="cuda")
assert x.device.type == "cuda"
```

### Common errors

| Error | Cause | Fix |
|---|---|---|
| SIGSEGV on tensor allocation | Used PyTorch.org's rocm7.1 wheels | Switch to TheRock gfx1151 wheels |
| `torch.cuda.is_available()` True but compute crashes | Same as above | Same as above |
| Flash Attention unavailable | gfx1151's FA requires extra aotriton build | Most inference doesn't need FA |
| Some torchaudio APIs missing | TheRock's torchaudio is stripped down | Doesn't affect core functionality |

## References

- [ROCm/ROCm#5853](https://github.com/ROCm/ROCm/issues/5853) — Strix Halo segfault on VRAM access
- [ROCm/TheRock#2991](https://github.com/ROCm/TheRock/issues/2991) — gfx1151 VGPR count crash
- [ROCm/TheRock#3081](https://github.com/ROCm/TheRock/issues/3081) — PyTorch.org wheels crash, TheRock works
- [ROCm/TheRock#3032](https://github.com/ROCm/TheRock/issues/3032) — VRAM allocation strategy
- [PyTorch#173367](https://github.com/pytorch/pytorch/issues/173367) — Strix Halo segfault on ROCm 7.1
- [AMD ROCm 7.11 docs](https://rocm.docs.amd.com/en/7.11.0-preview/)
- [AMD ROCm 7.2 compatibility matrix](https://rocmdocs.amd.com/en/develop/compatibility/compatibility-matrix.html)
