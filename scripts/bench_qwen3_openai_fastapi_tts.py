#!/usr/bin/env python3
"""Benchmark a Qwen3-TTS OpenAI-compatible FastAPI server.

The server endpoint returns raw PCM when response_format=pcm.  For streaming
requests this script records first-byte latency and inter-chunk gaps from the
HTTP response body, which is the useful measurement for realtime playback.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_TEXTS = [
    "你好，我已经准备好了，可以开始和你聊天。",
    "当然可以，我会用简短自然的方式回答你。",
    "这个问题我理解了，我们先确认当前状态，再继续下一步。",
]


def post_speech(
    base_url: str,
    text: str,
    *,
    model: str,
    voice: str,
    language: str,
    instruct: str,
    stream: bool,
    sample_rate: int,
    read_size: int,
) -> dict:
    parsed = urlparse(base_url.rstrip("/"))
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    path = f"{prefix}/v1/audio/speech"

    body = json.dumps(
        {
            "model": model,
            "voice": voice,
            "input": text,
            "language": language,
            "instruct": instruct,
            "response_format": "pcm",
            "stream": stream,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    conn = conn_cls(host, port, timeout=240)
    started = time.perf_counter()
    conn.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Accept": "audio/pcm"},
    )
    response = conn.getresponse()
    headers_at = time.perf_counter()
    if response.status >= 400:
        error = response.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {response.status}: {error}")

    first_byte_at = None
    last_chunk_at = None
    gaps = []
    total_bytes = 0
    chunk_count = 0
    while True:
        chunk = response.read(read_size)
        if not chunk:
            break
        now = time.perf_counter()
        if first_byte_at is None:
            first_byte_at = now
        if last_chunk_at is not None:
            gaps.append(now - last_chunk_at)
        last_chunk_at = now
        chunk_count += 1
        total_bytes += len(chunk)

    finished = time.perf_counter()
    conn.close()

    audio_s = total_bytes / 2 / sample_rate
    total_s = finished - started
    ttfa_s = (first_byte_at - started) if first_byte_at else 0.0
    header_s = headers_at - started
    rtf = audio_s / total_s if total_s else 0.0
    return {
        "text": text,
        "chars": len(text),
        "model": model,
        "voice": voice,
        "language": language,
        "stream": stream,
        "status": response.status,
        "header_s": round(header_s, 4),
        "ttfa_s": round(ttfa_s, 4),
        "total_s": round(total_s, 4),
        "audio_s": round(audio_s, 4),
        "rtf_audio_per_wall": round(rtf, 4),
        "realtime_debt_s": round(max(0.0, total_s - audio_s), 4),
        "bytes": total_bytes,
        "chunks": chunk_count,
        "max_gap_s": round(max(gaps), 4) if gaps else 0.0,
        "median_gap_s": round(statistics.median(gaps), 4) if gaps else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8880")
    parser.add_argument("--model", default="qwen3-tts")
    parser.add_argument("--voice", default="Serena")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="用自然、亲切、清晰的中文口语语气说话。")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--read-size", type=int, default=4096)
    parser.add_argument("--texts", nargs="*", default=DEFAULT_TEXTS)
    parser.add_argument("--modes", nargs="+", choices=["stream", "nonstream"], default=["stream", "nonstream"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "apps/sts-cache/bench/qwen3_openai_fastapi",
    )
    args = parser.parse_args()

    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for mode in args.modes:
        stream = mode == "stream"
        for text in args.texts:
            row = post_speech(
                args.base_url,
                text,
                model=args.model,
                voice=args.voice,
                language=args.language,
                instruct=args.instruct,
                stream=stream,
                sample_rate=args.sample_rate,
                read_size=args.read_size,
            )
            rows.append(row)
            write_csv(run_dir / "qwen3_openai_fastapi_tts.csv", rows)
            print(
                f"{mode} chars={row['chars']} ttfa={row['ttfa_s']:.3f}s "
                f"total={row['total_s']:.3f}s audio={row['audio_s']:.3f}s "
                f"rtf={row['rtf_audio_per_wall']:.2f} chunks={row['chunks']} "
                f"max_gap={row['max_gap_s']:.3f}s"
            )

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "voice": args.voice,
        "language": args.language,
        "rows_csv": str(run_dir / "qwen3_openai_fastapi_tts.csv"),
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
