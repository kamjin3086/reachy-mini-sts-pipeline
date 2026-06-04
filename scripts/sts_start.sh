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
