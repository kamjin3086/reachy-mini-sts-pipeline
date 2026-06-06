#!/usr/bin/env bash
# Install the verified TheRock ROCm PyTorch stack for gfx1151.

set -euo pipefail

VENV_DIR=${VENV_DIR:-/home/kamjin/apps/.venv}
THEROCK_INDEX_URL=${THEROCK_INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx1151/}

if [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "[error] Missing venv: $VENV_DIR" >&2
    echo "Create it first, for example: uv venv $VENV_DIR --python 3.12" >&2
    exit 1
fi

export PIP_CONFIG_FILE=/dev/null
unset HSA_OVERRIDE_GFX_VERSION

echo "[install] venv: $VENV_DIR"
echo "[install] TheRock index: $THEROCK_INDEX_URL"

"$VENV_DIR/bin/python3" -m ensurepip --upgrade
"$VENV_DIR/bin/python3" -m pip install --upgrade pip
"$VENV_DIR/bin/python3" -m pip install --index-url "$THEROCK_INDEX_URL" \
    torch torchaudio torchvision

"$VENV_DIR/bin/python3" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", getattr(torch.version, "hip", None))
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ROCm is not available")
print("Device:", torch.cuda.get_device_name(0))
print("Arch:", torch.cuda.get_arch_list())
if "rocm" not in torch.__version__:
    raise SystemExit("Installed torch is not a ROCm build")
if not any("gfx1151" in arch for arch in torch.cuda.get_arch_list()):
    raise SystemExit("Installed torch does not include gfx1151")
x = torch.randn(100, 100, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Matrix mul: OK", tuple(y.shape))
PY
