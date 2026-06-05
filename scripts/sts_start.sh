#!/bin/bash
# Speech-to-Speech pipeline: paraformer (STT) + responses-api (LLM) + qwen3 (TTS)
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

unset HF_ENDPOINT

# Keep assistant speech short and clean for TTS. Reachy actions must be emitted
# as realtime tool calls by the client/session, not spoken as text.
INIT_CHAT_PROMPT=${INIT_CHAT_PROMPT:-"你是 Reachy Mini 的中文语音助手。默认用中文口语化回答，每次只说 1 到 2 句，不使用 markdown。不要朗读动作标记、JSON、代码、工具名或工具参数。如果用户要求点头、动作、表情或跳舞，必须根据当前可用工具 schema 调用工具执行；自然语言回复只说给用户听的短句，不要描述工具调用过程。"}

# ── 端口检查与清理 ──
check_and_free_port() {
    local port=$1
    local pid
    pid=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "[port-check] Port $port is in use (PID: $pid), killing..."
        kill -9 $pid 2>/dev/null
        sleep 1
        # 二次确认
        pid=$(lsof -ti :$port 2>/dev/null)
        if [ -n "$pid" ]; then
            echo "[port-check] Port $port still occupied (PID: $pid), force killing..."
            kill -9 $pid 2>/dev/null
            sleep 1
        fi
        echo "[port-check] Port $port is now free."
    else
        echo "[port-check] Port $port is free."
    fi
}

check_and_free_port 8765

# 各组件的 warmup（STT 模型加载、LLM 首请求、TTS CUDA graph 编译）
# 由 speech-to-speech CLI 内部自动完成，无需外部预热
speech-to-speech \
    --responses_api_base_url "http://127.0.0.1:8101/v1" \
    --responses_api_api_key "" \
    --mode realtime \
    --model_name Gemma-4-E4B-instruct \
    --llm_backend responses-api \
    --responses_api_stream \
    --responses_api_disable_thinking \
    --stream_batch_sentences 1 \
    --init_chat_prompt "$INIT_CHAT_PROMPT" \
    --tts qwen3 \
    --qwen3_tts_language chinese \
    --qwen3_tts_speaker Serena \
    --qwen3_tts_instruct "用自然、亲切、清晰的中文口语语气说话。" \
    --qwen3_tts_streaming_chunk_size 12 \
    --qwen3_tts_blocksize 512 \
    --qwen3_tts_non_streaming_mode \
    --ws_host 0.0.0.0 \
    --ws_port 8765 \
    --stt paraformer \
    --language zh \
    --live_transcription_update_interval 0.5 \
    --no_enable_live_transcription
