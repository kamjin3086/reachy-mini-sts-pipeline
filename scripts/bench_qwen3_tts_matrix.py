#!/usr/bin/env python3
"""
Run real Qwen3-TTS Chinese instruction tests.

Outputs WAV files, a CSV, and a JSON summary under /tmp/sts_tuning by default.
The matrix focuses on whether inline bracket hints are spoken, and whether
Qwen3-TTS follows the official ``instruct`` control path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("GPU_MAX_ALLOC_PERCENT", "100")
os.environ.setdefault("GPU_MAX_HEAP_SIZE", "100")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import multiprocessing

import numpy as np


PIPELINE_SAMPLE_RATE = 16000
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CHINESE_FIRST_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]
CONTROL_SPEAKERS = ["Aiden"]
FULL_SPEAKERS = CHINESE_FIRST_SPEAKERS + CONTROL_SPEAKERS


@dataclass(frozen=True)
class TextCase:
    case_name: str
    text: str
    instruct: str | None = None
    strip_inline: bool = False
    expected_spoken_text: str | None = None
    purpose: str = ""


BASE_CASES = [
    TextCase(
        case_name="plain",
        text="你好，我是 Reachy Mini。",
        expected_spoken_text="你好，我是 Reachy Mini。",
        purpose="Baseline Chinese synthesis.",
    ),
    TextCase(
        case_name="inline_raw_parentheses",
        text="（开心地）你好呀，我已经准备好了。",
        expected_spoken_text="（开心地）你好呀，我已经准备好了。",
        purpose="Check whether raw bracket hints are spoken.",
    ),
    TextCase(
        case_name="explicit_instruct_happy",
        text="你好呀，我已经准备好了。",
        instruct="用开心、轻快、亲切的语气说话。",
        expected_spoken_text="你好呀，我已经准备好了。",
        purpose="Official instruct path for emotion/style control.",
    ),
    TextCase(
        case_name="inline_extracted_happy",
        text="（开心地）你好呀，我已经准备好了。",
        strip_inline=True,
        expected_spoken_text="你好呀，我已经准备好了。",
        purpose="Pipeline-compatible behavior: extract hint to instruct, do not speak it.",
    ),
    TextCase(
        case_name="inline_extracted_quiet",
        text="[小声]别担心，我会轻轻地说。",
        strip_inline=True,
        expected_spoken_text="别担心，我会轻轻地说。",
        purpose="Square-bracket hint extraction.",
    ),
]


def extract_inline_instruct(text: str) -> tuple[str, str | None]:
    cleaned = text.strip()
    hints: list[str] = []
    patterns = (
        r"^[（(]\s*([^（）()]{1,40}?)\s*[）)]\s*",
        r"^[［\[]\s*([^［］\[\]]{1,40}?)\s*[］\]]\s*",
    )
    while cleaned:
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match is None:
                continue
            hint = match.group(1).strip()
            if hint:
                hints.append(hint)
            cleaned = cleaned[match.end() :].lstrip()
            break
        else:
            break
    return (cleaned or text.strip(), "；".join(hints) if hints else None)


def save_wav(audio: np.ndarray, path: Path, sample_rate: int = PIPELINE_SAMPLE_RATE) -> None:
    audio = np.asarray(audio)
    if audio.dtype != np.int16:
        audio = audio.astype(np.float32)
        if audio.size and np.max(np.abs(audio)) <= 1.5:
            audio = audio * 32767.0
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def load_handler(args: argparse.Namespace, speaker: str):
    from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

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
            "model_name": args.model_name,
            "device": args.device,
            "language": args.tts_language,
            "speaker": speaker,
            "streaming_chunk_size": args.streaming_chunk_size,
            "blocksize": args.blocksize,
            "non_streaming_mode": args.non_streaming_mode,
            "instruct": args.default_instruct,
        },
    )
    return handler


def supported_speakers(handler: Any) -> list[str] | None:
    getter = getattr(handler, "_supported_speakers", None)
    if callable(getter):
        speakers = getter()
        if speakers is not None:
            return list(speakers)
    return None


def synthesize_case(
    handler: Any,
    args: argparse.Namespace,
    speaker: str,
    text_case: TextCase,
    case_id: int,
) -> dict[str, Any]:
    from speech_to_speech.pipeline.handler_types import TTSInput

    spoken_text = text_case.text
    inline_instruct = None
    if text_case.strip_inline:
        spoken_text, inline_instruct = extract_inline_instruct(text_case.text)

    original_instruct = handler.instruct
    effective_instruct = text_case.instruct or inline_instruct or args.default_instruct
    handler.instruct = effective_instruct

    t_start = time.perf_counter()
    t_first_audio = None
    chunks: list[np.ndarray] = []
    error = ""
    try:
        for chunk in handler.process(TTSInput(text=spoken_text, language_code=args.tts_language)):
            if not isinstance(chunk, np.ndarray):
                continue
            if t_first_audio is None and chunk.size:
                t_first_audio = time.perf_counter()
            chunks.append(chunk)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        handler.instruct = original_instruct

    total = time.perf_counter() - t_start
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    if not audio.size and not error:
        error = "no_audio_generated"
    audio_s = len(audio) / args.sample_rate if args.sample_rate else 0.0
    ttfa = (t_first_audio - t_start) if t_first_audio else 0.0
    rtf = audio_s / total if total else 0.0

    wav_path = args.output_dir / f"{case_id:03d}_{speaker}_{text_case.case_name}.wav"
    if audio.size:
        save_wav(audio, wav_path, args.sample_rate)

    raw_marker_expected = "（" in text_case.text or "(" in text_case.text or "[" in text_case.text or "［" in text_case.text
    marker_removed_from_tts_input = raw_marker_expected and spoken_text != text_case.text

    return {
        "case_id": case_id,
        "speaker": speaker,
        "case_name": text_case.case_name,
        "purpose": text_case.purpose,
        "source_text": text_case.text,
        "tts_input_text": spoken_text,
        "expected_spoken_text": text_case.expected_spoken_text or spoken_text,
        "effective_instruct": effective_instruct or "",
        "strip_inline": text_case.strip_inline,
        "marker_removed_from_tts_input": marker_removed_from_tts_input,
        "language": args.tts_language,
        "streaming_chunk_size": args.streaming_chunk_size,
        "blocksize": args.blocksize,
        "non_streaming_mode": args.non_streaming_mode,
        "ttfa_s": round(ttfa, 4),
        "total_s": round(total, 4),
        "audio_s": round(audio_s, 4),
        "rtf": round(rtf, 4),
        "wav": str(wav_path) if audio.size else "",
        "error": error,
        "subjective_marker_spoken": "",
        "subjective_style_score_1_5": "",
        "subjective_tail_score_1_5": "",
        "notes": "",
    }


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    try:
        import torch

        summary.update(
            {
                "torch": torch.__version__,
                "hip": getattr(torch.version, "hip", None),
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            }
        )
    except Exception as exc:
        summary["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import importlib.metadata as metadata

        for pkg in ("speech-to-speech", "qwen-tts", "faster-qwen3-tts", "funasr", "numpy"):
            summary[pkg] = metadata.version(pkg)
    except Exception as exc:
        summary["package_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sts_tuning"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tts-language", default="chinese")
    parser.add_argument("--sample-rate", type=int, default=PIPELINE_SAMPLE_RATE)
    parser.add_argument("--streaming-chunk-size", type=int, default=12)
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--non-streaming-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-instruct", default=None)
    parser.add_argument("--quick", action="store_true", help="Run one Chinese speaker and core cases.")
    parser.add_argument("--speakers", nargs="*", help="Speaker names to test.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.output_dir = args.output_dir / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speakers = args.speakers or (["Serena"] if args.quick else FULL_SPEAKERS)
    text_cases = BASE_CASES[:4] if args.quick else BASE_CASES

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": timestamp,
        "output_dir": str(args.output_dir),
        "environment": environment_summary(),
        "model_name": args.model_name,
        "tts_language": args.tts_language,
        "speakers_requested": speakers,
        "supported_speakers": None,
        "rows_csv": str(args.output_dir / "qwen3_tts_matrix.csv"),
    }

    case_id = 1
    for speaker in speakers:
        handler = None
        try:
            handler = load_handler(args, speaker)
            if summary["supported_speakers"] is None:
                summary["supported_speakers"] = supported_speakers(handler)
            for text_case in text_cases:
                row = synthesize_case(handler, args, speaker, text_case, case_id)
                rows.append(row)
                status = "ERR" if row["error"] else "OK"
                print(
                    f"{case_id:03d} {status} {speaker:8s} {text_case.case_name:24s} "
                    f"TTFA={row['ttfa_s']:.3f}s total={row['total_s']:.3f}s RTF={row['rtf']:.2f}x"
                )
                case_id += 1
        except Exception as exc:
            rows.append(
                {
                    "case_id": case_id,
                    "speaker": speaker,
                    "case_name": "handler_load",
                    "purpose": "Load Qwen3-TTS handler.",
                    "source_text": "",
                    "tts_input_text": "",
                    "expected_spoken_text": "",
                    "effective_instruct": "",
                    "strip_inline": "",
                    "marker_removed_from_tts_input": "",
                    "language": args.tts_language,
                    "streaming_chunk_size": args.streaming_chunk_size,
                    "blocksize": args.blocksize,
                    "non_streaming_mode": args.non_streaming_mode,
                    "ttfa_s": 0,
                    "total_s": 0,
                    "audio_s": 0,
                    "rtf": 0,
                    "wav": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "subjective_marker_spoken": "",
                    "subjective_style_score_1_5": "",
                    "subjective_tail_score_1_5": "",
                    "notes": "",
                }
            )
            print(f"{case_id:03d} ERR {speaker:8s} handler_load: {type(exc).__name__}: {exc}")
            case_id += 1
        finally:
            if handler is not None:
                handler.cleanup()

    write_csv(rows, args.output_dir / "qwen3_tts_matrix.csv")
    summary["row_count"] = len(rows)
    summary["error_count"] = sum(1 for row in rows if row.get("error"))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary written: {args.output_dir / 'summary.json'}")
    return 1 if summary["error_count"] == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
