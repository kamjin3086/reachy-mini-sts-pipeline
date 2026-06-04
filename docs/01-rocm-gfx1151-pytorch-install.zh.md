# Strix Halo (Radeon 8060S) PyTorch ROCm 安装备忘

> [English](01-rocm-gfx1151-pytorch-install.md) · [← 返回 README](../README.zh.md)

## 硬件信息

- **系统**: Fedora 44

- **GPU**: AMD Ryzen AI Max+ 395 w/ Radeon 8060S Graphics
- **架构**: gfx1151 (Strix Halo)
- **内存**: 共享系统内存 (UMA)，非独立显存
- **PCI ID**: 1002:1586

## 核心问题：Strix Halo 的 ROCm 支持很特殊

Strix Halo (gfx1151) 是 AMD 的 APU（集成显卡），其 ROCm 支持经历了多次迭代，**不同 ROCm 版本的兼容性差异极大**。选错版本会直接段错误 (SIGSEGV)。

### 版本线说明

ROCm 目前有**两条并行发布线**：

| 版本线 | 定位 | gfx1151 支持 | 说明 |
|---|---|---|---|
| **ROCm 7.0 ~ 7.2** (生产流) | 生产级稳定 | ❌ 兼容性矩阵中**没有** gfx1151 | 传统 monolithic 构建，稳定但 APU 支持滞后 |
| **ROCm 7.11+** (TheRock) | 技术预览 | ✅ **原生支持** gfx1151 | 新构建系统 TheRock，APU 支持优先 |

### 为什么 PyTorch.org 的 ROCm 不行

```
PyTorch 2.12.0+rocm7.1  ← 从 PyTorch.org 安装
```

- 基于 **ROCm 7.1**，该版本对 gfx1151 有 **VGPR 计数 bug**（Issue #2991）
- 表现：`torch.cuda.is_available()` 返回 True，但任何 GPU 张量分配都会 **SIGSEGV**
- 与 Python 版本无关（3.12/3.13/3.14 都有此问题）

### 修复方案

**必须使用 TheRock 构建的 wheels**，它们包含 gfx1151 专属修复：

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision
```

## 当前安装配置

```
Python:    3.12.13
PyTorch:   2.10.0+rocm7.13.0a20260513
ROCm/HIP:  7.13.26183
架构:      gfx1151
```

### 安装命令

```bash
# 1. 创建 Python 3.12 虚拟环境
uv venv /home/kamjin/apps/.venv --python 3.12

# 2. 从 TheRock gfx1151 索引安装
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

### 验证安装

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('ROCm/HIP:', torch.version.hip)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('Arch:', torch.cuda.get_arch_list())

# 功能测试
x = torch.randn(100, 100, device='cuda')
y = torch.randn(100, 100, device='cuda')
z = x @ y
print('Matrix mul: OK')
print('ALL TESTS PASSED')
"
```

## ROCm 7.13 已修复的问题

- **VGPR 计数 bug (Issue #2991)**：ROCm 7.13 (TheRock) 已原生修复 gfx1151 VGPR 问题，不再需要 `HSA_OVERRIDE_GFX_VERSION` 覆盖
- **使用 override 的副作用**：`HSA_OVERRIDE_GFX_VERSION=11.0.0` 会导致 `hipErrorInvalidImage` 内核架构不匹配错误（内核编译为 gfx1100，但硬件是 gfx1151）
- **测试验证**：
  - 有 override → `HIP error: device kernel image is invalid`
  - 无 override → 所有 GPU 操作正常
- **当前配置**：移除 `HSA_OVERRIDE_GFX_VERSION`，保留 `GPU_MAX_ALLOC_PERCENT=100`、`GPU_MAX_HEAP_SIZE=100`、`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

## 已知问题和 workaround

### 1. MIOpen solver database 文件不可读

```
MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable:
  ".../gfx1151_20.HIP.fdb.txt"
```

- **影响**: 无功能影响，只是 solver 缓存文件
- **原因**: TheRock 包中打包的 fdb 文件格式可能不匹配当前 MIOpen 版本
- **处理**: 忽略即可，MIOpen 会自动重建

### 2. XNACK 警告

```
warning: xnack 'Off' was requested for a processor that does not support it!
```

- **影响**: 无功能影响
- **原因**: gfx1151 不支持 XNACK，但 ROCm 默认请求
- **处理**: 忽略

### 3. VRAM 分配策略（大内存场景）

当系统设置大 UMA 内存（如 96GB）时，PyTorch 可能跳过 VRAM 直接使用共享内存，导致 OOM。

- **Workaround**：设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 或减少分配的 VRAM 大小

### 4. 内核版本兼容性

- **Linux 6.18.4+**: 需要 TheRock wheels 或内核补丁修复 VGPR 问题
- **Linux 6.18.3 及以下**: ROCm 7.1 也能工作（VGPR bug 未引入）

## 常用推理框架兼容性

| 框架 | ROCm 要求 | 推荐版本 | 备注 |
|---|---|---|---|
| whisper.cpp | GGML HIP | ROCm 7.2+ | 绕过 rocWMMA，7.2 即可 |
| whisper.pytorch | PyTorch ROCm | TheRock 7.11+ | 需要 PyTorch GPU |
| Coqui TTS / XTTS | PyTorch ROCm | TheRock 7.11+ | 有已知 kernel 问题 |
| SpeechBrain | PyTorch ROCm | TheRock 7.11+ | — |
| llama.cpp | GGML HIP | ROCm 7.2+ | 已验证可用 |

## 快速参考

### 安装

```bash
# 创建环境
uv venv /path/to/venv --python 3.12

# 安装 PyTorch ROCm (gfx1151)
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision
```

### 验证

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "Radeon 8060S Graphics"
assert "gfx1151" in torch.cuda.get_arch_list()
x = torch.randn(10, 10, device="cuda")
assert x.device.type == "cuda"
```

### 常见错误

| 错误 | 原因 | 解决 |
|---|---|---|
| SIGSEGV on tensor alloc | 用了 PyTorch.org 的 rocm7.1 wheels | 换 TheRock gfx1151 wheels |
| `torch.cuda.is_available()` True 但计算崩 | 同上 | 同上 |
| Flash Attention 不可用 | gfx1151 的 FA 需要额外编译 aotriton | 大多数推理不需要 FA |
| torchaudio 部分 API 缺失 | TheRock 包的 torchaudio 精简 | 不影响核心功能 |

## 参考来源

- [ROCm/ROCm#5853](https://github.com/ROCm/ROCm/issues/5853) — Strix Halo segfault on VRAM access
- [ROCm/TheRock#2991](https://github.com/ROCm/TheRock/issues/2991) — gfx1151 VGPR count crash
- [ROCm/TheRock#3081](https://github.com/ROCm/TheRock/issues/3081) — PyTorch.org wheels crash, TheRock works
- [ROCm/TheRock#3032](https://github.com/ROCm/TheRock/issues/3032) — VRAM allocation strategy
- [PyTorch#173367](https://github.com/pytorch/pytorch/issues/173367) — Strix Halo segfault on ROCm 7.1
- [AMD ROCm 7.11 文档](https://rocm.docs.amd.com/en/7.11.0-preview/)
- [AMD ROCm 7.2 兼容性矩阵](https://rocmdocs.amd.com/en/develop/compatibility/compatibility-matrix.html)
