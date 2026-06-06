#!/usr/bin/env python3
"""
Benchmark Qwen3-TTS realtime throughput on the active speech-to-speech install.

The main metric is whether synthesis produces audio faster than playback:

    rtf = generated_audio_seconds / wall_clock_generation_seconds

RTF below 1.0 means the audio player must eventually underrun. The script also
records chunk inter-arrival gaps from the speech-to-speech handler to separate
model throughput from downstream batching jitter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("GPU_MAX_ALLOC_PERCENT", "100")
os.environ.setdefault("GPU_MAX_HEAP_SIZE", "100")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

import multiprocessing

import numpy as np


PIPELINE_SAMPLE_RATE = 16000
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_TEXTS = [
    "你好，我是 Reachy Mini。我已经准备好了，可以开始和你聊天。",
    "我会尽量用简短自然的方式回答你。如果你让我做动作，我也会配合执行。",
    "这是一个稍微长一点的测试句子，用来检查语音合成速度是否能够跟上真实播放速度，以及中间会不会出现明显的停顿。",
]


@dataclass(frozen=True)
class ConfigCase:
    speaker: str
    streaming_chunk_size: int
    blocksize: int
    non_streaming_mode: bool
    parity_mode: bool
    dtype: str
    attn_implementation: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_k: int
    instruct: str | None

    @property
    def name(self) -> str:
        nsm = "nsm" if self.non_streaming_mode else "streamtext"
        parity = "parity" if self.parity_mode else "graph"
        instruct = "instruct" if self.instruct else "plain"
        return (
            f"{self.speaker}_chunk{self.streaming_chunk_size}_block{self.blocksize}_"
            f"{nsm}_{parity}_{self.dtype}_{self.attn_implementation}_max{self.max_new_tokens}_"
            f"sample{int(self.do_sample)}_temp{self.temperature:g}_topk{self.top_k}_{instruct}"
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    try:
        import torch

        summary.update(
            {
                "torch": torch.__version__,
                "hip": getattr(torch.version, "hip", None),
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            }
        )
    except Exception as exc:
        summary["torch_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import importlib.metadata as metadata

        for pkg in ("speech-to-speech", "qwen-tts", "faster-qwen3-tts", "transformers", "numpy"):
            summary[pkg] = metadata.version(pkg)
    except Exception as exc:
        summary["package_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def load_handler(args: argparse.Namespace, cfg: ConfigCase):
    from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

    stop = multiprocessing.Event()
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    should_listen = multiprocessing.Event()

    return Qwen3TTSHandler(
        stop,
        q_in,
        q_out,
        setup_args=(should_listen,),
        setup_kwargs={
            "model_name": args.model_name,
            "device": args.device,
            "dtype": cfg.dtype,
            "attn_implementation": cfg.attn_implementation,
            "max_new_tokens": cfg.max_new_tokens,
            "gen_kwargs": {
                "do_sample": cfg.do_sample,
                "temperature": cfg.temperature,
                "top_k": cfg.top_k,
            },
            "language": args.tts_language,
            "speaker": cfg.speaker,
            "streaming_chunk_size": cfg.streaming_chunk_size,
            "blocksize": cfg.blocksize,
            "non_streaming_mode": cfg.non_streaming_mode,
            "parity_mode": cfg.parity_mode,
            "instruct": cfg.instruct,
        },
    )


def run_one(handler: Any, args: argparse.Namespace, cfg: ConfigCase, text: str, case_index: int) -> dict[str, Any]:
    from speech_to_speech.pipeline.handler_types import TTSInput

    started = time.perf_counter()
    first_audio_at: float | None = None
    last_chunk_at: float | None = None
    inter_chunk_gaps: list[float] = []
    chunks: list[np.ndarray] = []
    error = ""

    try:
        for item in handler.process(TTSInput(text=text, language_code=args.tts_language)):
            if not isinstance(item, np.ndarray) or item.size == 0:
                continue
            now = time.perf_counter()
            if first_audio_at is None:
                first_audio_at = now
            if last_chunk_at is not None:
                inter_chunk_gaps.append(now - last_chunk_at)
            last_chunk_at = now
            chunks.append(item)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    audio_samples = int(sum(chunk.size for chunk in chunks))
    audio_s = audio_samples / float(args.sample_rate)
    total_s = ended - started
    ttfa_s = (first_audio_at - started) if first_audio_at else 0.0
    rtf = (audio_s / total_s) if total_s else 0.0
    realtime_debt_s = max(0.0, total_s - audio_s)
    max_gap_s = max(inter_chunk_gaps) if inter_chunk_gaps else 0.0
    p95_gap_s = percentile(inter_chunk_gaps, 95)
    avg_gap_s = statistics.fmean(inter_chunk_gaps) if inter_chunk_gaps else 0.0

    return {
        "case_index": case_index,
        "config": cfg.name,
        "speaker": cfg.speaker,
        "streaming_chunk_size": cfg.streaming_chunk_size,
        "blocksize": cfg.blocksize,
        "non_streaming_mode": cfg.non_streaming_mode,
        "parity_mode": cfg.parity_mode,
        "dtype": cfg.dtype,
        "attn_implementation": cfg.attn_implementation,
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "has_instruct": bool(cfg.instruct),
        "text_chars": len(text),
        "text": text,
        "chunk_count": len(chunks),
        "ttfa_s": round(ttfa_s, 4),
        "total_s": round(total_s, 4),
        "audio_s": round(audio_s, 4),
        "rtf": round(rtf, 4),
        "realtime_debt_s": round(realtime_debt_s, 4),
        "avg_inter_chunk_gap_s": round(avg_gap_s, 4),
        "p95_inter_chunk_gap_s": round(p95_gap_s, 4),
        "max_inter_chunk_gap_s": round(max_gap_s, 4),
        "safe_realtime": rtf >= args.safe_rtf,
        "error": error,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sts_tuning/qwen3_realtime_perf"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tts-language", default="chinese")
    parser.add_argument("--sample-rate", type=int, default=PIPELINE_SAMPLE_RATE)
    parser.add_argument("--safe-rtf", type=float, default=1.05)
    parser.add_argument("--speakers", nargs="+", default=["Serena", "Vivian", "Uncle_Fu"])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--blocksizes", nargs="+", type=int, default=[512])
    parser.add_argument("--dtypes", nargs="+", default=["auto"])
    parser.add_argument("--attn-implementations", nargs="+", default=["eager"])
    parser.add_argument("--parity-modes", nargs="+", default=["false"])
    parser.add_argument("--max-new-tokens", nargs="+", type=int, default=[1536])
    parser.add_argument("--do-sample", nargs="+", default=["true"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.9])
    parser.add_argument("--top-k", nargs="+", type=int, default=[50])
    parser.add_argument("--non-streaming-modes", nargs="+", default=["true", "false"])
    parser.add_argument("--instructs", nargs="*", default=["", "用自然、亲切、清晰的中文口语语气说话。"])
    parser.add_argument("--texts", nargs="*", default=DEFAULT_TEXTS)
    parser.add_argument("--quick", action="store_true", help="Run a smaller matrix for smoke testing.")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    speakers = args.speakers[:1] if args.quick else args.speakers
    chunk_sizes = args.chunk_sizes[:2] if args.quick else args.chunk_sizes
    texts = args.texts[:1] if args.quick else args.texts

    configs = [
        ConfigCase(
            speaker=speaker,
            streaming_chunk_size=chunk_size,
            blocksize=blocksize,
            non_streaming_mode=as_bool(non_streaming_mode),
            parity_mode=as_bool(parity_mode),
            dtype=dtype,
            attn_implementation=attn_impl,
            max_new_tokens=max_new_tokens,
            do_sample=as_bool(do_sample),
            temperature=temperature,
            top_k=top_k,
            instruct=instruct or None,
        )
        for speaker in speakers
        for chunk_size in chunk_sizes
        for blocksize in args.blocksizes
        for non_streaming_mode in args.non_streaming_modes
        for parity_mode in args.parity_modes
        for dtype in args.dtypes
        for attn_impl in args.attn_implementations
        for max_new_tokens in args.max_new_tokens
        for do_sample in args.do_sample
        for temperature in args.temperatures
        for top_k in args.top_k
        for instruct in args.instructs
    ]

    rows: list[dict[str, Any]] = []
    csv_path = run_dir / "qwen3_realtime_perf.csv"
    partial_summary_path = run_dir / "partial_summary.json"
    case_index = 1
    for cfg in configs:
        handler = None
        try:
            print(f"\n[config] {cfg.name}")
            handler = load_handler(args, cfg)
            for text in texts:
                row = run_one(handler, args, cfg, text, case_index)
                rows.append(row)
                status = "SAFE" if row["safe_realtime"] else "SLOW"
                if row["error"]:
                    status = "ERR"
                print(
                    f"{case_index:03d} {status} ttfa={row['ttfa_s']:.3f}s "
                    f"total={row['total_s']:.3f}s audio={row['audio_s']:.3f}s "
                    f"rtf={row['rtf']:.2f} debt={row['realtime_debt_s']:.2f}s "
                    f"max_gap={row['max_inter_chunk_gap_s']:.3f}s"
                )
                case_index += 1
                write_csv(rows, csv_path)
                partial_summary_path.write_text(
                    json.dumps(
                        {
                            "created_at": timestamp,
                            "output_dir": str(run_dir),
                            "csv": str(csv_path),
                            "completed_rows": len(rows),
                            "last_config": cfg.name,
                            "last_case_index": row["case_index"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except Exception as exc:
            rows.append(
                {
                    "case_index": case_index,
                    "config": cfg.name,
                    "speaker": cfg.speaker,
                    "streaming_chunk_size": cfg.streaming_chunk_size,
                    "blocksize": cfg.blocksize,
                    "non_streaming_mode": cfg.non_streaming_mode,
                    "parity_mode": cfg.parity_mode,
                    "dtype": cfg.dtype,
                    "attn_implementation": cfg.attn_implementation,
                    "max_new_tokens": cfg.max_new_tokens,
                    "do_sample": cfg.do_sample,
                    "temperature": cfg.temperature,
                    "top_k": cfg.top_k,
                    "has_instruct": bool(cfg.instruct),
                    "text_chars": 0,
                    "text": "",
                    "chunk_count": 0,
                    "ttfa_s": 0,
                    "total_s": 0,
                    "audio_s": 0,
                    "rtf": 0,
                    "realtime_debt_s": 0,
                    "avg_inter_chunk_gap_s": 0,
                    "p95_inter_chunk_gap_s": 0,
                    "max_inter_chunk_gap_s": 0,
                    "safe_realtime": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{case_index:03d} ERR {cfg.name}: {type(exc).__name__}: {exc}")
            case_index += 1
            write_csv(rows, csv_path)
        finally:
            if handler is not None:
                handler.cleanup()

    summary_path = run_dir / "summary.json"
    write_csv(rows, csv_path)

    ok_rows = [row for row in rows if not row.get("error")]
    by_config: dict[str, dict[str, Any]] = {}
    for cfg_name in sorted({row["config"] for row in ok_rows}):
        cfg_rows = [row for row in ok_rows if row["config"] == cfg_name]
        by_config[cfg_name] = {
            "runs": len(cfg_rows),
            "safe_runs": sum(1 for row in cfg_rows if row["safe_realtime"]),
            "min_rtf": min(row["rtf"] for row in cfg_rows),
            "median_rtf": round(statistics.median(row["rtf"] for row in cfg_rows), 4),
            "max_debt_s": max(row["realtime_debt_s"] for row in cfg_rows),
            "median_ttfa_s": round(statistics.median(row["ttfa_s"] for row in cfg_rows), 4),
        }

    summary = {
        "created_at": timestamp,
        "output_dir": str(run_dir),
        "csv": str(csv_path),
        "safe_rtf": args.safe_rtf,
        "environment": environment_summary(),
        "config_count": len(configs),
        "row_count": len(rows),
        "error_count": sum(1 for row in rows if row.get("error")),
        "safe_count": sum(1 for row in rows if row.get("safe_realtime")),
        "by_config": by_config,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary: {summary_path}")
    print(f"csv: {csv_path}")
    return 1 if summary["error_count"] == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
