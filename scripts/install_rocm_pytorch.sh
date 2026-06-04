#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ROCm PyTorch 安装脚本 - Strix Halo (gfx1151) / Fedora 44
# 用法: chmod +x install_rocm_pytorch.sh && ./install_rocm_pytorch.sh
# ============================================================
VENV_DIR="/home/kamjin/apps/.venv"

echo "========================================"
echo " Step 0: 激活虚拟环境"
echo "========================================"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 虚拟环境 $VENV_DIR 不存在！"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo "✅ Python: $(which python3) — $(python3 --version)"

echo ""
echo "========================================"
echo " Step 1: 安装 ROCm 计算库 (系统级别)"
echo "========================================"
sudo dnf install -y \
    rocblas \
    hipblas \
    hipblaslt \
    miopen \
    rccl \
    rocfft \
    rocrand \
    rocsolver \
    rocsparse

echo ""
echo "========================================"
echo " Step 2: 卸载现有的 CPU 版 PyTorch"
echo "========================================"
pip uninstall -y torch torchaudio 2>/dev/null || true
echo "✅ 已卸载 CPU 版 torch/torchaudio"

echo ""
echo "========================================"
echo " Step 3: 安装 ROCm 版 PyTorch"
echo "========================================"
pip install \
    torch==2.12.0+rocm7.1 \
    torchaudio==2.11.0+rocm7.1 \
    --index-url https://download.pytorch.org/whl/rocm7.1 \
    --extra-index-url https://pypi.org/simple

echo ""
echo "========================================"
echo " Step 4: 验证 GPU 检测"
echo "========================================"
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
    # 跑一个小张量计算验证
    x = torch.randn(3, 3).cuda()
    y = torch.mm(x, x)
    print(f'张量计算验证: ✅ ({x.shape} @ {x.shape} = {y.shape})')
else:
    print('❌ GPU 不可用，请检查 ROCm 安装')
"

echo ""
echo "========================================"
echo " Step 5: ROCm 系统信息"
echo "========================================"
rocm-smi --showhw 2>/dev/null || echo "(rocm-smi 不可用)"

echo ""
echo "========================================"
echo " 安装完成！"
echo "========================================"
echo ""
echo "下一步重启 speech-to-speech 后端:"
echo "  --stt whisper --stt_model_name large-v3 --stt_device cuda --language zh"
echo ""
echo "如需更轻量的中文 STT 也可用:"
echo "  --stt paraformer"
echo "  --stt faster-whisper --faster_whisper_stt_device cuda"
