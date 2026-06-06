#!/bin/bash
# Stable Speech-to-Speech pipeline:
# paraformer (STT) + responses-api (LLM) + qwen3 (TTS).
#
# Common overrides:
#   STS_LLM_BASE_URL=http://127.0.0.1:8101/v1
#   STS_LLM_MODEL=Gemma-4-E4B-instruct
#   STS_WS_HOST=0.0.0.0
#   STS_WS_PORT=8765
#   STS_KILL_PORT=1   # opt in to killing an existing process on STS_WS_PORT
# GPU: AMD Radeon 8060S (gfx1151) via ROCm 7.13
# STT: FunASR Paraformer-zh (中文优化, CER~1.95%, 120x 实时)
# TTS: faster-qwen3-tts (ROCm 已实测 gfx1151 可用)
# LLM: responses-api (本地 127.0.0.1:8101)

# 允许 GPU VRAM 全量分配（UMA 共享内存，不限制）
export GPU_MAX_ALLOC_PERCENT=100
# 限制 HIP 最大堆大小（避免 OOM）
export GPU_MAX_HEAP_SIZE=100
# 启用 AOTriton 实验性编译（ROCm 7.11+ PyTorch 性能优化）
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
# 抑制 ROCm/MIOpen workspace 估算警告；该警告会刷屏，但不是 TTS 卡顿主因
export MIOPEN_LOG_LEVEL=3
# 当前 ROCm/gfx1151 上 Qwen3-TTS decode 偶发低于实时播放速度。
# 需要完全避免句中停顿时可手动设为 1；默认关闭以免首句等待过长。
export STS_QWEN3_TTS_BUFFER_FULL_UTTERANCE=${STS_QWEN3_TTS_BUFFER_FULL_UTTERANCE:-0}
# 降低随机采样导致的 EOS 长尾波动；由本仓库 startup patch 转发给 faster-qwen3-tts。
export STS_QWEN3_TTS_DO_SAMPLE=${STS_QWEN3_TTS_DO_SAMPLE:-0}

# ROCm 7.13/TheRock 已原生支持 gfx1151；override 会导致 kernel 架构不匹配或慢路径。
unset HSA_OVERRIDE_GFX_VERSION
unset HF_ENDPOINT
export NLTK_DATA=${NLTK_DATA:-/home/kamjin/nltk_data}

DEFAULT_QWEN3_TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_QWEN3_TTS_MODEL="/home/kamjin/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe"
if [ -z "${QWEN3_TTS_MODEL_NAME:-}" ] && [ -d "$LOCAL_QWEN3_TTS_MODEL" ]; then
    QWEN3_TTS_MODEL_NAME="$LOCAL_QWEN3_TTS_MODEL"
else
    QWEN3_TTS_MODEL_NAME="${QWEN3_TTS_MODEL_NAME:-$DEFAULT_QWEN3_TTS_MODEL}"
fi

DEFAULT_PARAFORMER_STT_MODEL="paraformer-zh"
LOCAL_PARAFORMER_STT_MODEL="/home/kamjin/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
if [ -z "${PARAFORMER_STT_MODEL_NAME:-}" ] && [ -d "$LOCAL_PARAFORMER_STT_MODEL" ]; then
    PARAFORMER_STT_MODEL_NAME="$LOCAL_PARAFORMER_STT_MODEL"
else
    PARAFORMER_STT_MODEL_NAME="${PARAFORMER_STT_MODEL_NAME:-$DEFAULT_PARAFORMER_STT_MODEL}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"
if [ ! -f "$REPO_DIR/scripts/patch_qwen3_tts_inline_instruct.py" ]; then
    REPO_DIR="/home/kamjin/projects/reachy-mini-sts-pipeline"
fi

resolve_sts_python() {
    local sts_bin shebang
    sts_bin=$(command -v speech-to-speech 2>/dev/null || true)
    if [ -n "$sts_bin" ] && [ -r "$sts_bin" ]; then
        shebang=$(head -n 1 "$sts_bin" 2>/dev/null || true)
        if [[ "$shebang" == "#!"*python* ]]; then
            echo "${shebang#\#!}"
            return 0
        fi
    fi
    command -v python3 2>/dev/null || command -v python 2>/dev/null
}

apply_startup_patches() {
    local patch_python
    patch_python=$(resolve_sts_python)
    if [ -z "$patch_python" ]; then
        echo "[patch] Could not find python for startup patches." >&2
        return 1
    fi

    echo "[patch] Checking installed speech-to-speech patches with $patch_python"
    "$patch_python" "$REPO_DIR/scripts/patch_sts_offline_startup.py" || return 1
    "$patch_python" "$REPO_DIR/scripts/patch_paraformer_live_transcription.py" || return 1
    "$patch_python" "$REPO_DIR/scripts/patch_qwen3_tts_inline_instruct.py" || return 1
    "$patch_python" "$REPO_DIR/scripts/patch_qwen3_tts_realtime_stability.py" || return 1
}

apply_startup_patches || exit 1

# Keep assistant speech short and clean for TTS. Reachy actions must be emitted
# as realtime tool calls by the client/session, not spoken as text.
INIT_CHAT_PROMPT=${INIT_CHAT_PROMPT:-"你是 Reachy Mini 的中文语音助手。默认用中文口语化回答，每次只说 1 到 2 句，不使用 markdown。不要朗读动作标记、JSON、代码、工具名或工具参数。如果用户要求点头、动作、表情或跳舞，必须根据当前可用工具 schema 调用工具执行；自然语言回复只说给用户听的短句，不要描述工具调用过程。"}

# ── 端口检查与清理 ──
check_and_free_port() {
    local port=$1
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null)
    if [ -n "$pid" ]; then
        if [ "${STS_KILL_PORT:-0}" = "1" ]; then
            echo "[port-check] Port $port is in use (PID: $pid), killing because STS_KILL_PORT=1..."
            kill -9 $pid 2>/dev/null
            sleep 1
            pid=$(lsof -ti :"$port" 2>/dev/null)
            if [ -n "$pid" ]; then
                echo "[port-check] Port $port is still occupied (PID: $pid)." >&2
                return 1
            fi
            echo "[port-check] Port $port is now free."
        else
            echo "[port-check] Port $port is in use (PID: $pid)." >&2
            echo "[port-check] Stop that process, choose STS_WS_PORT, or set STS_KILL_PORT=1." >&2
            return 1
        fi
    else
        echo "[port-check] Port $port is free."
    fi
}

STS_LLM_BASE_URL=${STS_LLM_BASE_URL:-http://127.0.0.1:8101/v1}
STS_LLM_MODEL=${STS_LLM_MODEL:-Gemma-4-E4B-instruct}
STS_WS_HOST=${STS_WS_HOST:-0.0.0.0}
STS_WS_PORT=${STS_WS_PORT:-8765}

check_and_free_port "$STS_WS_PORT" || exit 1

# 各组件的 warmup（STT 模型加载、LLM 首请求、TTS CUDA graph 编译）
# 由 speech-to-speech CLI 内部自动完成，无需外部预热
speech-to-speech \
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
    --qwen3_tts_model_name "$QWEN3_TTS_MODEL_NAME" \
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
