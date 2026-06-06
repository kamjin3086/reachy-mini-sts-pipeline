#!/bin/bash
# Speech-to-Speech pipeline with Qwen3-TTS OpenAI FastAPI flash-attn optimization.
# This launcher uses an isolated venv and does not modify scripts/sts_start.sh.
#
# Common overrides:
#   STS_LLM_BASE_URL=http://127.0.0.1:8101/v1
#   STS_LLM_MODEL=Gemma-4-E4B-instruct
#   STS_WS_HOST=0.0.0.0
#   STS_WS_PORT=8765
#   STS_KILL_PORT=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"

STS_PYTHON=${STS_PYTHON:-/home/kamjin/apps/.venv-qwen3-fa/bin/python3}
FASTAPI_REPO_DIR=${QWEN3_OPENAI_FASTAPI_DIR:-/home/kamjin/apps/Qwen3-TTS-Openai-Fastapi}
STS_CACHE_DIR=${STS_CACHE_DIR:-/home/kamjin/apps/sts-cache}
FASTAPI_HOME=${QWEN3_OPENAI_FASTAPI_HOME:-$STS_CACHE_DIR/qwen3_openai_fastapi_flash_home}
FASTAPI_HOST=${QWEN3_OPENAI_FASTAPI_HOST:-127.0.0.1}
FASTAPI_PORT=${QWEN3_OPENAI_FASTAPI_PORT:-8881}
FASTAPI_URL="http://${FASTAPI_HOST}:${FASTAPI_PORT}"
QWEN3_FASTAPI_USE_COMPILE=${QWEN3_FASTAPI_USE_COMPILE:-false}
QWEN3_FASTAPI_COMPILE_MODE=${QWEN3_FASTAPI_COMPILE_MODE:-reduce-overhead}
QWEN3_FASTAPI_USE_CUDA_GRAPHS=${QWEN3_FASTAPI_USE_CUDA_GRAPHS:-false}
QWEN3_FASTAPI_COMPILE_CODEBOOK=${QWEN3_FASTAPI_COMPILE_CODEBOOK:-false}

QWEN3_TTS_06B_MODEL=${QWEN3_TTS_06B_MODEL:-/home/kamjin/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5}
QWEN3_TTS_17B_MODEL=${QWEN3_TTS_17B_MODEL:-/home/kamjin/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe}

export GPU_MAX_ALLOC_PERCENT=100
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_HW_QUEUES=1
export MIOPEN_FIND_MODE=FAST
export MIOPEN_LOG_LEVEL=3
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$STS_CACHE_DIR/torchinductor_${USER:-kamjin}}
unset HSA_OVERRIDE_GFX_VERSION
unset HF_ENDPOINT
export NLTK_DATA=${NLTK_DATA:-/home/kamjin/nltk_data}

DEFAULT_PARAFORMER_STT_MODEL="paraformer-zh"
LOCAL_PARAFORMER_STT_MODEL="/home/kamjin/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
if [ -z "${PARAFORMER_STT_MODEL_NAME:-}" ] && [ -d "$LOCAL_PARAFORMER_STT_MODEL" ]; then
    PARAFORMER_STT_MODEL_NAME="$LOCAL_PARAFORMER_STT_MODEL"
else
    PARAFORMER_STT_MODEL_NAME="${PARAFORMER_STT_MODEL_NAME:-$DEFAULT_PARAFORMER_STT_MODEL}"
fi

check_and_free_port() {
    local port=$1
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        if [ "${STS_KILL_PORT:-0}" = "1" ]; then
            echo "[port-check] Port $port is in use (PID: $pid), killing because STS_KILL_PORT=1..."
            kill -9 $pid 2>/dev/null || true
            sleep 1
            pid=$(lsof -ti :"$port" 2>/dev/null || true)
            if [ -n "$pid" ]; then
                echo "[port-check] Port $port is still occupied (PID: $pid)." >&2
                return 1
            fi
            echo "[port-check] Port $port is now free."
        else
            echo "[port-check] Port $port is in use (PID: $pid)." >&2
            echo "[port-check] Stop that process, choose another port, or set STS_KILL_PORT=1." >&2
            return 1
        fi
    else
        echo "[port-check] Port $port is free."
    fi
}

ensure_fastapi_repo() {
    if [ -f "$FASTAPI_REPO_DIR/api/main.py" ]; then
        return 0
    fi
    echo "[fastapi] Missing repo at $FASTAPI_REPO_DIR; cloning..."
    mkdir -p "$(dirname "$FASTAPI_REPO_DIR")"
    git clone https://github.com/dingausmwald/Qwen3-TTS-Openai-Fastapi.git "$FASTAPI_REPO_DIR"
}

verify_flash_attn() {
    "$STS_PYTHON" - <<'PY'
import flash_attn
import torch
print(f"[flash] flash_attn {flash_attn.__version__}")
print(f"[flash] torch {torch.__version__} hip={getattr(torch.version, 'hip', None)}")
PY
}

write_fastapi_config() {
    mkdir -p "$FASTAPI_HOME/qwen3-tts/voice_library" "$TORCHINDUCTOR_CACHE_DIR"
    cat > "$FASTAPI_HOME/qwen3-tts/config.yaml" <<EOF
default_model: 0.6B-CustomVoice
models:
  0.6B-CustomVoice:
    hf_id: $QWEN3_TTS_06B_MODEL
    type: customvoice
  1.7B-CustomVoice:
    hf_id: $QWEN3_TTS_17B_MODEL
    type: customvoice
optimization:
  attention: flash_attention_2
  compile_mode: $QWEN3_FASTAPI_COMPILE_MODE
  use_compile: $QWEN3_FASTAPI_USE_COMPILE
  use_cuda_graphs: $QWEN3_FASTAPI_USE_CUDA_GRAPHS
  use_fast_codebook: true
  streaming:
    decode_window_frames: 72
    emit_every_frames: 24
  compile_codebook_predictor: $QWEN3_FASTAPI_COMPILE_CODEBOOK
server:
  host: 0.0.0.0
  port: $FASTAPI_PORT
voices:
- language: Chinese
  name: Vivian
- language: Chinese
  name: Serena
- language: Chinese
  name: Uncle_Fu
- language: Chinese
  name: Dylan
- language: Chinese
  name: Eric
- language: English
  name: Ryan
EOF
}

wait_for_fastapi() {
    local i
    for i in $(seq 1 600); do
        if curl -fsS "$FASTAPI_URL/health" >/dev/null 2>&1; then
            echo "[fastapi] Ready: $FASTAPI_URL"
            return 0
        fi
        sleep 1
    done
    echo "[fastapi] Timed out waiting for $FASTAPI_URL/health" >&2
    return 1
}

apply_startup_patches() {
    echo "[patch] Checking installed speech-to-speech patches with $STS_PYTHON"
    "$STS_PYTHON" "$REPO_DIR/scripts/patch_sts_offline_startup.py"
    "$STS_PYTHON" "$REPO_DIR/scripts/patch_paraformer_live_transcription.py"
    "$STS_PYTHON" "$REPO_DIR/scripts/patch_qwen3_tts_inline_instruct.py"
    "$STS_PYTHON" "$REPO_DIR/scripts/patch_qwen3_tts_realtime_stability.py"
    "$STS_PYTHON" "$REPO_DIR/scripts/patch_qwen3_tts_openai_fastapi_bridge.py"
    "$STS_PYTHON" "$REPO_DIR/scripts/patch_qwen3_openai_fastapi_compat.py" --repo-dir "$FASTAPI_REPO_DIR"
}

start_fastapi() {
    check_and_free_port "$FASTAPI_PORT"
    echo "[fastapi] Starting flash-attn Qwen3-TTS OpenAI FastAPI on $FASTAPI_URL"
    echo "[fastapi] First startup can take several minutes while models load or optional compile warms up."
    HOME="$FASTAPI_HOME" \
    PYTHONPATH="$FASTAPI_REPO_DIR:$FASTAPI_REPO_DIR/patches/dffdeeq:${PYTHONPATH:-}" \
    TTS_BACKEND=optimized \
    HOST="$FASTAPI_HOST" \
    PORT="$FASTAPI_PORT" \
    WORKERS=1 \
    TTS_WARMUP_ON_START=false \
    GPU_KEEPALIVE_INTERVAL=0 \
    "$STS_PYTHON" -m api.main &
    FASTAPI_PID=$!
    trap 'kill "$FASTAPI_PID" 2>/dev/null || true' EXIT
    wait_for_fastapi
}

ensure_fastapi_repo
verify_flash_attn
write_fastapi_config
apply_startup_patches
start_fastapi

export STS_QWEN3_OPENAI_FASTAPI_URL="$FASTAPI_URL"
export STS_QWEN3_OPENAI_FASTAPI_MODEL=${STS_QWEN3_OPENAI_FASTAPI_MODEL:-qwen3-tts}
export STS_QWEN3_OPENAI_FASTAPI_VOICE=${STS_QWEN3_OPENAI_FASTAPI_VOICE:-Serena}
export STS_QWEN3_OPENAI_FASTAPI_LANGUAGE=${STS_QWEN3_OPENAI_FASTAPI_LANGUAGE:-Chinese}
export STS_QWEN3_OPENAI_FASTAPI_STREAM=${STS_QWEN3_OPENAI_FASTAPI_STREAM:-1}

INIT_CHAT_PROMPT=${INIT_CHAT_PROMPT:-"你是 Reachy Mini 的中文语音助手。默认用中文口语化回答，每次只说 1 到 2 句，不使用 markdown。不要朗读动作标记、JSON、代码、工具名或工具参数。如果用户要求点头、动作、表情或跳舞，必须根据当前可用工具 schema 调用工具执行；自然语言回复只说给用户听的短句，不要描述工具调用过程。"}

STS_LLM_BASE_URL=${STS_LLM_BASE_URL:-http://127.0.0.1:8101/v1}
STS_LLM_MODEL=${STS_LLM_MODEL:-Gemma-4-E4B-instruct}
STS_WS_HOST=${STS_WS_HOST:-0.0.0.0}
STS_WS_PORT=${STS_WS_PORT:-8765}

check_and_free_port "$STS_WS_PORT" || exit 1

"$STS_PYTHON" -m speech_to_speech.s2s_pipeline \
    --responses_api_base_url "$STS_LLM_BASE_URL" \
    --responses_api_api_key "" \
    --mode realtime \
    --model_name "$STS_LLM_MODEL" \
    --llm_backend responses-api \
    --responses_api_stream \
    --responses_api_disable_thinking \
    --stream_batch_sentences 3 \
    --init_chat_prompt "$INIT_CHAT_PROMPT" \
    --tts qwen3 \
    --qwen3_tts_model_name "$QWEN3_TTS_06B_MODEL" \
    --qwen3_tts_attn_implementation sdpa \
    --qwen3_tts_language chinese \
    --qwen3_tts_speaker Serena \
    --qwen3_tts_instruct "用自然、亲切、清晰的中文口语语气说话。" \
    --qwen3_tts_non_streaming_mode true \
    --qwen3_tts_streaming_chunk_size 12 \
    --qwen3_tts_max_new_tokens 128 \
    --qwen3_tts_blocksize 512 \
    --ws_host "$STS_WS_HOST" \
    --ws_port "$STS_WS_PORT" \
    --stt paraformer \
    --paraformer_stt_model_name "$PARAFORMER_STT_MODEL_NAME" \
    --language zh \
    --live_transcription_update_interval 0.5 \
    --no_enable_live_transcription
