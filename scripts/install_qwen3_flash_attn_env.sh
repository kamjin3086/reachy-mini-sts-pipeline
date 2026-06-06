#!/usr/bin/env bash
# Build a reproducible, isolated ROCm flash-attn environment for Qwen3-TTS.
#
# This script is intentionally conservative:
# - it copies an existing working STS venv instead of modifying it;
# - it pins the external repositories to known-good commits;
# - it restores TheRock ROCm Triton after flash-attn installation, because PyPI
#   triton can otherwise replace the gfx1151-compatible build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"

SOURCE_VENV=${SOURCE_VENV:-/home/kamjin/apps/.venv}
TARGET_VENV=${TARGET_VENV:-/home/kamjin/apps/.venv-qwen3-fa}
FLASH_ATTN_DIR=${FLASH_ATTN_DIR:-/tmp/flash-attention}
FLASH_ATTN_REF=${FLASH_ATTN_REF:-bc58abc67bdd6470d6500414e08441b95708453f}
FLASH_AITER_REF=${FLASH_AITER_REF:-9bab8388c35936814a659b4ebd245c491e1b940a}
QWEN3_OPENAI_FASTAPI_DIR=${QWEN3_OPENAI_FASTAPI_DIR:-/home/kamjin/apps/Qwen3-TTS-Openai-Fastapi}
QWEN3_OPENAI_FASTAPI_REF=${QWEN3_OPENAI_FASTAPI_REF:-eb14f6e6a50445cf442979abb9203ff0d5042c43}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.org/simple}
MAX_JOBS=${MAX_JOBS:-4}
REINSTALL=${REINSTALL:-0}
RUN_KERNEL_SMOKE=${RUN_KERNEL_SMOKE:-0}
FORCE_INSTALL_EXISTING=${FORCE_INSTALL_EXISTING:-0}
TARGET_VENV_PREEXISTED=0

export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
unset HSA_OVERRIDE_GFX_VERSION

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_file() {
    if [ ! -e "$1" ]; then
        echo "[error] Missing required path: $1" >&2
        exit 1
    fi
}

clone_or_update_repo() {
    local url=$1
    local dir=$2
    local ref=$3

    if [ -d "$dir/.git" ]; then
        log "Updating $dir"
        git -C "$dir" fetch --tags origin
    else
        log "Cloning $url -> $dir"
        mkdir -p "$(dirname "$dir")"
        git clone "$url" "$dir"
        git -C "$dir" fetch --tags origin
    fi
    git -C "$dir" checkout -f "$ref"
}

copy_or_reuse_venv() {
    require_file "$SOURCE_VENV/bin/python3"

    if [ -e "$TARGET_VENV" ]; then
        if [ "$REINSTALL" = "1" ]; then
            local backup="${TARGET_VENV}.bak.$(date '+%Y%m%d-%H%M%S')"
            log "Moving existing target venv to $backup"
            mv "$TARGET_VENV" "$backup"
        else
            log "Target venv already exists: $TARGET_VENV"
            log "Defaulting to verify-only. Set FORCE_INSTALL_EXISTING=1 to reinstall into it, or REINSTALL=1 to move it aside and rebuild."
            TARGET_VENV_PREEXISTED=1
            return 0
        fi
    fi

    log "Copying $SOURCE_VENV -> $TARGET_VENV"
    cp -a "$SOURCE_VENV" "$TARGET_VENV"
}

python_bin() {
    printf '%s/bin/python3' "$TARGET_VENV"
}

pip_install() {
    "$(python_bin)" -m pip install "$@"
}

ensure_pip_and_build_tools() {
    log "Ensuring pip/build tools in target venv"
    "$(python_bin)" -m ensurepip --upgrade
    pip_install --upgrade pip wheel setuptools ninja psutil packaging
}

restore_therock_triton() {
    local src_site
    local dst_site
    src_site="$("$SOURCE_VENV/bin/python3" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
    dst_site="$("$(python_bin)" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

    require_file "$src_site/triton"
    log "Restoring TheRock ROCm Triton from source venv"
    rm -rf "$dst_site/triton" "$dst_site"/triton-*.dist-info
    cp -a "$src_site/triton" "$dst_site/triton"
    cp -a "$src_site"/triton-*.dist-info "$dst_site/"
}

install_flash_attn() {
    clone_or_update_repo https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_DIR" "$FLASH_ATTN_REF"

    log "Initializing flash-attn submodules"
    git -C "$FLASH_ATTN_DIR" submodule update --init --recursive
    git -C "$FLASH_ATTN_DIR/third_party/aiter" checkout "$FLASH_AITER_REF"
    git -C "$FLASH_ATTN_DIR/third_party/aiter" submodule update --init --recursive

    log "Installing flash-attn ROCm Triton backend"
    MAX_JOBS="$MAX_JOBS" FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        pip_install --no-build-isolation --no-cache-dir "$FLASH_ATTN_DIR"

    restore_therock_triton
}

prepare_fastapi_repo() {
    clone_or_update_repo \
        https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi.git \
        "$QWEN3_OPENAI_FASTAPI_DIR" \
        "$QWEN3_OPENAI_FASTAPI_REF"

    log "Applying local FastAPI compatibility patch"
    "$(python_bin)" "$REPO_DIR/scripts/patch_qwen3_openai_fastapi_compat.py" \
        --repo-dir "$QWEN3_OPENAI_FASTAPI_DIR"
}

write_snapshot() {
    local out="$TARGET_VENV/qwen3_flash_attn_env_snapshot.txt"
    log "Writing environment snapshot: $out"
    {
        echo "SOURCE_VENV=$SOURCE_VENV"
        echo "TARGET_VENV=$TARGET_VENV"
        echo "FLASH_ATTN_DIR=$FLASH_ATTN_DIR"
        echo "FLASH_ATTN_REF=$FLASH_ATTN_REF"
        echo "FLASH_AITER_REF=$FLASH_AITER_REF"
        echo "QWEN3_OPENAI_FASTAPI_DIR=$QWEN3_OPENAI_FASTAPI_DIR"
        echo "QWEN3_OPENAI_FASTAPI_REF=$QWEN3_OPENAI_FASTAPI_REF"
        echo
        "$(python_bin)" -m pip freeze
    } > "$out"
}

verify_env() {
    log "Verifying target environment"
    if [ "$RUN_KERNEL_SMOKE" = "1" ]; then
        "$(python_bin)" "$REPO_DIR/scripts/verify_qwen3_flash_attn_env.py" --kernel-smoke
    else
        "$(python_bin)" "$REPO_DIR/scripts/verify_qwen3_flash_attn_env.py"
    fi
}

copy_or_reuse_venv
if [ "$TARGET_VENV_PREEXISTED" = "1" ] && [ "$FORCE_INSTALL_EXISTING" != "1" ]; then
    verify_env
    log "Existing target venv was not modified."
    exit 0
fi
ensure_pip_and_build_tools
install_flash_attn
prepare_fastapi_repo
write_snapshot
verify_env

log "Done. Start the optimized path with:"
echo "  STS_PYTHON=$TARGET_VENV/bin/python3 ./scripts/sts_start_qwen3_openai_fastapi_flash.sh"
