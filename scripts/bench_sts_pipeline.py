#!/usr/bin/env python3
"""
Speech-to-Speech Pipeline 性能测试脚本

测试项目：
1. STT (Paraformer) - 中文 ASR 延迟
2. LLM (responses-api) - 流式推理延迟、TTFT、token/s
3. TTS (Qwen3) - 合成延迟、RTF
4. 端到端 - 语音 → 语音总延迟

使用方法：
    source /home/kamjin/apps/.venv/bin/activate
    python3 /home/kamjin/scripts/bench_sts_pipeline.py [--quick]
"""

import argparse
import os
import time
import wave
from pathlib import Path
from typing import List

# ROCm env vars (must be set before importing torch)
os.environ.setdefault("GPU_MAX_ALLOC_PERCENT", "100")
os.environ.setdefault("GPU_MAX_HEAP_SIZE", "100")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
# OpenAI client requires a non-empty key OR a valid OPENAI_API_KEY env var.
# For local llama-swap that accepts empty Bearer, we must NOT have the env var set.
os.environ.pop("OPENAI_API_KEY", None)

import numpy as np
import torch

# Workaround for upstream bug: speech_to_speech/paraformer_handler.py:56
# invokes torch.mps.empty_cache() unconditionally, which raises on ROCm/CUDA.
if hasattr(torch, "mps") and not torch.backends.mps.is_available():
    torch.mps.empty_cache = torch.cuda.empty_cache  # type: ignore[attr-defined]

import logging
for noisy in ["funasr", "modelscope", "httpx", "httpcore", "speech_to_speech"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

RESPONSES_API_BASE = "http://127.0.0.1:8101/v1"
LLM_MODEL = "Gemma-4-E4B-instruct"

TEST_SENTENCES_ZH = [
    "你好，请介绍一下你自己。",
    "今天天气怎么样？",
    "请帮我解释一下人工智能的基本概念。",
    "我正在开发一个语音助手项目，你能给我一些建议吗？",
    "ROCm 7.13 在 Strix Halo 上的性能如何？",
]

LLM_TEST_PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "用中文写一首关于秋天的四行诗。",
    "解释什么是 ROCm 和 HIP。",
    "Python 中如何异步处理多个任务？",
    "请推荐三本学习机器学习的入门书籍。",
]


def gen_silent_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate low-amplitude noise as fake 'voice' for benchmarking.
    Real speech should be substituted for actual quality testing."""
    n = int(duration_s * sample_rate)
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(n).astype(np.float32) * 0.1


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save a float32 mono array as 16-bit PCM WAV."""
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def banner(s: str):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70)


def fmt(s: float) -> str:
    return f"{s:>8.3f}s"


# ============================================================================
# GPU 信息
# ============================================================================
def print_gpu_info():

    banner("GPU 信息")
    print(f"PyTorch:    {torch.__version__}")
    print(f"ROCm/HIP:   {torch.version.hip}")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Arch:       {torch.cuda.get_arch_list()}")


# ============================================================================
# 1. STT 性能测试
# ============================================================================
def bench_stt(durations: List[float], n_runs: int = 3):
    banner("STT (Paraformer-zh) 性能测试")

    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler
    from speech_to_speech.pipeline.handler_types import STTIn

    import multiprocessing
    stop = multiprocessing.Event()
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()

    handler = ParaformerSTTHandler(
        stop,
        q_in,
        q_out,
        setup_kwargs={"model_name": "paraformer-zh", "device": "cuda", "gen_kwargs": {}},
    )
    print("Model loaded.\n")

    print(f"{'Duration':>10} {'Run':>4} {'Latency':>10} {'RTF':>8}")
    print("-" * 40)

    results = []
    for dur in durations:
        audio = gen_silent_wav(dur, 16000)
        run_latencies = []
        for run in range(n_runs):
            t0 = time.perf_counter()
            list(handler.process(STTIn(audio=audio, mode="final")))
            dt = time.perf_counter() - t0
            run_latencies.append(dt)
            rtf = dur / dt
            print(f"{dur:>8.1f}s {run+1:>4} {fmt(dt)} {rtf:>7.2f}x")
        avg = sum(run_latencies) / len(run_latencies)
        rtf_avg = dur / avg
        print(f"  → avg: {fmt(avg)}, RTF {rtf_avg:.2f}x\n")
        results.append((dur, avg, rtf_avg))

    handler.cleanup()
    return results


# ============================================================================
# 2. LLM 性能测试
# ============================================================================
def bench_llm(prompts: List[str], n_runs: int = 2):
    banner("LLM (responses-api) 性能测试")

    from speech_to_speech.LLM.responses_api_language_model import (
        ResponsesApiModelHandler,
    )
    from speech_to_speech.pipeline.handler_types import LLMIn, RuntimeConfig
    from speech_to_speech.LLM.chat import Chat, make_user_message
    from speech_to_speech.pipeline.messages import LLMResponseChunk

    import multiprocessing
    stop = multiprocessing.Event()
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()

    handler = ResponsesApiModelHandler(
        stop,
        q_in,
        q_out,
        setup_kwargs={
            "model_name": LLM_MODEL,
            "device": "cuda",
            "gen_kwargs": {},
            "base_url": RESPONSES_API_BASE,
            "api_key": "",
            "stream": True,
            "disable_thinking": True,
        },
    )
    print("Warmed up.\n")

    print(f"{'Prompt#':>8} {'Run':>4} {'TTFT':>10} {'Total':>10} {'Tokens':>8} {'tok/s':>8}")
    print("-" * 60)

    for i, prompt in enumerate(prompts):
        for run in range(n_runs):
            chat = Chat(size=30)
            chat.add_item(make_user_message(prompt))
            runtime_config = RuntimeConfig(chat=chat)

            t_start = time.perf_counter()
            t_first = None
            tts_text = ""

            for chunk in handler.process(LLMIn(runtime_config=runtime_config)):
                if not isinstance(chunk, LLMResponseChunk):
                    continue
                if t_first is None and chunk.text:
                    t_first = time.perf_counter()
                if chunk.text:
                    tts_text += chunk.text

            t_end = time.perf_counter()

            ttft = (t_first - t_start) if t_first else 0
            total = t_end - t_start
            # 估算 token 数（中文约 1.5 字符/token）
            n_tokens = max(1, int(len(tts_text) * 0.7))
            tok_s = n_tokens / total if total > 0 else 0

            print(
                f"{i+1:>8} {run+1:>4} {fmt(ttft)} {fmt(total)} {n_tokens:>8} {tok_s:>7.1f}"
            )
        print()

    handler.cleanup()


# ============================================================================
# 3. TTS 性能测试
# ============================================================================
def bench_tts(texts: List[str], n_runs: int = 2, output_dir: str = "/tmp/sts_bench"):
    banner("TTS (Qwen3) 性能测试")

    from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
    from speech_to_speech.pipeline.handler_types import TTSInput

    import multiprocessing
    stop = multiprocessing.Event()
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    should_listen = multiprocessing.Event()

    handler = Qwen3TTSHandler(
        stop,
        q_in,
        q_out,
        setup_args=(should_listen,),
        setup_kwargs={
            "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "device": "cuda",
            "non_streaming_mode": True,
        },
    )
    print("Model loaded.\n")

    print(f"{'Text#':>6} {'Chars':>6} {'Run':>4} {'TTFA':>10} {'Total':>10} {'Audio':>8} {'RTF':>8}")
    print("-" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total_ttfa = []
    total_rtf = []

    for i, text in enumerate(texts):
        run_results = []
        for run in range(n_runs):
            t_start = time.perf_counter()
            t_first_audio = None
            audio_chunks = []
            sample_rate = 24000

            for chunk in handler.process(TTSInput(text=text, language_code="zh")):
                if not isinstance(chunk, np.ndarray):
                    continue
                if t_first_audio is None and chunk.size > 0:
                    t_first_audio = time.perf_counter()
                audio_chunks.append(chunk)

            t_end = time.perf_counter()

            full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
            audio_dur = len(full_audio) / sample_rate if sample_rate > 0 else 0
            ttfa = (t_first_audio - t_start) if t_first_audio else 0
            total = t_end - t_start
            rtf = audio_dur / total if total > 0 else 0
            run_results.append((ttfa, total, audio_dur, rtf))
            total_ttfa.append(ttfa)
            total_rtf.append(rtf)
            print(
                f"{i+1:>6} {len(text):>6} {run+1:>4} {fmt(ttfa)} {fmt(total)} "
                f"{audio_dur:>7.2f}s {rtf:>7.2f}x"
            )

        if run_results and full_audio.size > 0:
            save_path = f"{output_dir}/tts_{i+1}.wav"
            save_wav(full_audio.astype(np.float32) / 32767, save_path, sample_rate)
            print(f"  → saved to {save_path}")
        print()

    if total_ttfa:
        avg_ttfa = sum(total_ttfa) / len(total_ttfa)
        avg_rtf = sum(total_rtf) / len(total_rtf)
        print(f"AVG TTFA: {fmt(avg_ttfa)}  |  AVG RTF: {avg_rtf:.2f}x")

    handler.cleanup()


# ============================================================================
# 4. 端到端 Pipeline 测试
# ============================================================================
def bench_e2e(audio_dur: float = 2.0, n_runs: int = 3):
    banner("端到端 Pipeline 性能测试")

    from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler
    from speech_to_speech.LLM.responses_api_language_model import (
        ResponsesApiModelHandler,
    )
    from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
    from speech_to_speech.pipeline.handler_types import (
        STTIn,
        LLMIn,
        TTSInput,
        RuntimeConfig,
    )
    from speech_to_speech.LLM.chat import Chat, make_user_message
    from speech_to_speech.pipeline.messages import LLMResponseChunk

    import multiprocessing
    stop = multiprocessing.Event()
    should_listen = multiprocessing.Event()

    print("Loading STT...")
    stt_in_q = multiprocessing.Queue()
    stt_out_q = multiprocessing.Queue()
    stt = ParaformerSTTHandler(
        stop,
        stt_in_q,
        stt_out_q,
        setup_kwargs={"model_name": "paraformer-zh", "device": "cuda"},
    )

    print("Loading LLM...")
    llm_in_q = multiprocessing.Queue()
    llm_out_q = multiprocessing.Queue()
    llm = ResponsesApiModelHandler(
        stop,
        llm_in_q,
        llm_out_q,
        setup_kwargs={
            "model_name": LLM_MODEL,
            "device": "cuda",
            "base_url": RESPONSES_API_BASE,
            "api_key": "",
            "stream": True,
            "disable_thinking": True,
        },
    )

    print("Loading TTS...")
    tts_in_q = multiprocessing.Queue()
    tts_out_q = multiprocessing.Queue()
    tts = Qwen3TTSHandler(
        stop,
        tts_in_q,
        tts_out_q,
        setup_args=(should_listen,),
        setup_kwargs={
            "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "device": "cuda",
            "non_streaming_mode": True,
        },
    )
    print()

    audio = gen_silent_wav(audio_dur, 16000)
    print(f"Audio: {audio_dur}s | runs: {n_runs}")
    print(f"{'Run':>4} {'STT':>8} {'LLM_TTFT':>10} {'LLM_Total':>10} {'TTS_TTFA':>10} {'TTS_Total':>10} {'E2E':>10}")
    print("-" * 80)

    e2e_times = []
    for run in range(n_runs):
        t0 = time.perf_counter()
        stt_text = ""
        for out in stt.process(STTIn(audio=audio, mode="final")):
            stt_text += out.text
        t_stt = time.perf_counter() - t0

        chat = Chat(size=30)
        chat.add_item(make_user_message(stt_text or "你好"))
        rc = RuntimeConfig(chat=chat)

        t0 = time.perf_counter()
        t_llm_first = None
        llm_text = ""
        for chunk in llm.process(LLMIn(runtime_config=rc)):
            if not isinstance(chunk, LLMResponseChunk):
                continue
            if t_llm_first is None and chunk.text:
                t_llm_first = time.perf_counter()
            if chunk.text:
                llm_text += chunk.text
        t_llm_end = time.perf_counter()
        t_llm_first = t_llm_first - t0
        t_llm_total = t_llm_end - t0

        t0 = time.perf_counter()
        t_tts_first = None
        tts_audio = []
        for chunk in tts.process(TTSInput(text=llm_text, language_code="zh")):
            if not isinstance(chunk, np.ndarray):
                continue
            if t_tts_first is None and chunk.size > 0:
                t_tts_first = time.perf_counter()
            tts_audio.append(chunk)
        t_tts_end = time.perf_counter()
        t_tts_first = t_tts_first - t0
        t_tts_total = t_tts_end - t0

        e2e = t_stt + t_llm_first + t_tts_first  # 用户感知延迟

        e2e_times.append(e2e)
        print(
            f"{run+1:>4} {fmt(t_stt)} {fmt(t_llm_first)} {fmt(t_llm_total)} "
            f"{fmt(t_tts_first)} {fmt(t_tts_total)} {fmt(e2e)}"
        )

    if e2e_times:
        avg = sum(e2e_times) / len(e2e_times)
        print(f"\n平均用户感知延迟 (STT + LLM_TTFT + TTS_TTFA): {fmt(avg)}")

    stt.cleanup()
    llm.cleanup()
    tts.cleanup()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Speech-to-Speech Pipeline 性能测试")
    parser.add_argument(
        "--quick", action="store_true", help="快速模式（更少测试点）"
    )
    parser.add_argument(
        "--only",
        choices=["stt", "llm", "tts", "e2e"],
        help="只跑指定测试",
    )
    args = parser.parse_args()

    print_gpu_info()

    if args.quick:
        stt_durations = [1.0, 2.0]
        llm_prompts = LLM_TEST_PROMPTS[:2]
        tts_texts = TEST_SENTENCES_ZH[:2]
        n_runs = 1
    else:
        stt_durations = [1.0, 2.0, 5.0, 10.0]
        llm_prompts = LLM_TEST_PROMPTS
        tts_texts = TEST_SENTENCES_ZH
        n_runs = 3

    if args.only in (None, "stt"):
        bench_stt(stt_durations, n_runs=max(1, n_runs - 1))
    if args.only in (None, "llm"):
        bench_llm(llm_prompts, n_runs=max(1, n_runs - 1))
    if args.only in (None, "tts"):
        bench_tts(tts_texts, n_runs=max(1, n_runs - 1))
    if args.only in (None, "e2e"):
        bench_e2e(audio_dur=2.0, n_runs=n_runs)

    print("\n[OK] All benchmarks complete.")


if __name__ == "__main__":
    main()
