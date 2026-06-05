"""Smoke test for HuggingFace speech-to-speech realtime WS pipeline.

Sends pre-recorded 24 kHz mono PCM16 clips (English and/or Chinese) and saves
the model's audio reply plus transcripts. Works headless (no mic, no speakers).

Important protocol constraints (from openai-python source):
  - AudioPCM.rate is Literal[24000]; the HF server resamples internally to 16 kHz
  - Voice must be a built-in name ("alloy", "ash", "ballad", "coral", "echo", ...)
  - output_modalities default is ["audio"] (already includes transcript)
  - turn_detection ServerVad fields are all Optional

Server pipeline runs at 16 kHz, so output audio is delivered as 16 kHz PCM
regardless of the declared 24 kHz in session.update.

Usage:
    python test_s2s_ws.py                # run both en + zh
    python test_s2s_ws.py --lang en
    python test_s2s_ws.py --lang zh
"""
import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).parent
WS_URI = "ws://127.0.0.1:8765/v1/realtime"
SAMPLE_RATE = 24000           # wire rate (matches AudioPCM Literal[24000])
PIPELINE_SR = 16000           # rate of the server's internal pipeline
CHUNK_MS = 100                                # audio per input_audio_buffer.append event
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 2400 samples per chunk

LANGS = {
    "en": {
        "input_wav": ROOT / "test_data" / "test_input_en_24k_mono.wav",
        "output_wav": ROOT / "test_data" / "test_output_en_16k_mono.wav",
        "log_path": ROOT / "test_data" / "test_output_en_log.json",
        "instructions": (
            "You are a friendly English conversation partner. "
            "Reply in one or two short sentences only."
        ),
    },
    "zh": {
        "input_wav": ROOT / "test_data" / "test_input_zh_24k_mono.wav",
        "output_wav": ROOT / "test_data" / "test_output_zh_16k_mono.wav",
        "log_path": ROOT / "test_data" / "test_output_zh_log.json",
        "instructions": (
            "你是一个友好的中文对话伙伴。"
            "请用一到两句简短的中文回复。"
        ),
    },
}


def build_session_update(instructions: str) -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": instructions,
            "audio": {
                "input": {
                    "transcription": {"model": "whisper-1"},
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "voice": "aiden",
                    "format": {"type": "audio/pcm", "rate": 24000},
                },
            },
        },
    }


def load_pcm16_mono(path: Path, expected_rate: int) -> bytes:
    data, sr = sf.read(str(path), dtype="int16", always_2d=False)
    assert sr == expected_rate, f"expected {expected_rate} Hz, got {sr}"
    assert data.ndim == 1, f"expected mono, got shape {data.shape}"
    return data.tobytes()


async def run_test(lang: str) -> int:
    cfg = LANGS[lang]
    session_update = build_session_update(cfg["instructions"])

    audio_bytes = load_pcm16_mono(cfg["input_wav"], SAMPLE_RATE)
    duration = len(audio_bytes) / 2 / SAMPLE_RATE
    print(
        f"[client][{lang}] input wav: {cfg['input_wav'].name} "
        f"({duration:.2f}s @ {SAMPLE_RATE}Hz, {len(audio_bytes)} bytes)"
    )

    log: dict = {
        "lang": lang,
        "ws_uri": WS_URI,
        "input_duration_s": duration,
        "wire_sample_rate": SAMPLE_RATE,
        "pipeline_sample_rate": PIPELINE_SR,
        "started_at": time.time(),
        "events": [],
        "output_audio_bytes": 0,
        "user_transcript": None,
    }

    out_chunks: list[bytes] = []
    user_transcript_parts: list[str] = []
    assistant_transcript_parts: list[str] = []
    close_after_response = asyncio.Event()

    async with websockets.connect(WS_URI, open_timeout=15, close_timeout=10) as ws:
        print(f"[client][{lang}] connected to {WS_URI}")

        first_raw = await ws.recv()
        first = json.loads(first_raw)
        log["events"].append(first)
        first_type = first.get("type")
        print(f"[server][{lang}] {first_type}")
        if first_type == "error":
            print(f"[client][{lang}] server rejected connection: {first.get('error')}")
            return 3

        await ws.send(json.dumps(session_update))
        print(f"[client][{lang}] sent session.update (rate=24000)")

        async def sender() -> None:
            try:
                for i in range(0, len(audio_bytes), CHUNK_SAMPLES * 2):
                    if close_after_response.is_set():
                        break
                    chunk = audio_bytes[i : i + CHUNK_SAMPLES * 2]
                    if not chunk:
                        break
                    msg = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                    await ws.send(json.dumps(msg))
                    await asyncio.sleep(CHUNK_MS / 1000.0)
            finally:
                pass

        async def receiver() -> None:
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    et = ev.get("type")
                    log["events"].append(
                        {
                            "type": et,
                            **(
                                {"len": len(ev.get("delta", ""))}
                                if "delta" in ev
                                else {}
                            ),
                        }
                    )

                    if et == "response.output_audio.delta":
                        out_chunks.append(base64.b64decode(ev["delta"]))
                    elif et == "conversation.item.input_audio_transcription.delta":
                        t = ev.get("delta") or ""
                        if t:
                            user_transcript_parts.append(t)
                    elif et == "conversation.item.input_audio_transcription.completed":
                        t = ev.get("transcript") or ""
                        if t:
                            log["user_transcript"] = t
                            print(f"[transcript][{lang}] user (final): {t}")
                    elif et == "response.output_audio_transcript.delta":
                        t = ev.get("delta") or ""
                        if t:
                            assistant_transcript_parts.append(t)
                    elif et == "response.output_audio_transcript.done":
                        t = ev.get("transcript") or ""
                        if t:
                            assistant_transcript_parts.append(t)
                    elif et == "response.done":
                        print(
                            f"[server][{lang}] response.done reason={ev.get('reason')} "
                            f"status={ev.get('status')}"
                        )
                        close_after_response.set()
                    elif et == "error":
                        print(f"[server][{lang}][error] {ev.get('error')}")
                        close_after_response.set()
                    else:
                        if et not in {"session.created", "session.update"}:
                            print(f"[server][{lang}] {et}")
            except websockets.ConnectionClosed as e:
                print(
                    f"[client][{lang}] connection closed: code={e.code} reason={e.reason!r}"
                )

        recv_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(sender())

        try:
            await asyncio.wait_for(close_after_response.wait(), timeout=45.0)
        except asyncio.TimeoutError:
            print(f"[client][{lang}] no response.done within 45s")

        # Stop the sender; receiver will exit when the WS closes
        send_task.cancel()
        try:
            await send_task
        except (asyncio.CancelledError, Exception):
            pass

        try:
            await ws.close(code=1000, reason="client done")
        except Exception:
            pass

        try:
            await asyncio.wait_for(recv_task, timeout=10.0)
        except asyncio.TimeoutError:
            print(f"[client][{lang}] receiver did not exit cleanly - cancelling")
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass

    if user_transcript_parts:
        log["user_transcript_streamed"] = "".join(user_transcript_parts)
    if assistant_transcript_parts:
        log["assistant_transcript"] = "".join(assistant_transcript_parts)
        snippet = log["assistant_transcript"][:400]
        print(
            f"[transcript][{lang}] assistant (full, "
            f"{len(assistant_transcript_parts)} chunks, "
            f"{len(log['assistant_transcript'])} chars): {snippet}"
        )

    if out_chunks:
        pcm = b"".join(out_chunks)
        pcm_arr = np.frombuffer(pcm, dtype=np.int16)
        sf.write(str(cfg["output_wav"]), pcm_arr, PIPELINE_SR, subtype="PCM_16")
        log["output_audio_bytes"] = len(pcm)
        log["output_duration_s"] = len(pcm) / 2 / PIPELINE_SR
        log["output_sample_rate"] = PIPELINE_SR
        print(
            f"[client][{lang}] wrote {cfg['output_wav']} "
            f"({len(pcm)} bytes, {log['output_duration_s']:.2f}s @ {PIPELINE_SR}Hz)"
        )
    else:
        print(f"[client][{lang}][warn] no response.output_audio.delta received")

    cfg["log_path"].write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[client][{lang}] log -> {cfg['log_path']}")
    return 0 if out_chunks else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "both"],
        default="both",
        help="Which language(s) to test (default: both)",
    )
    args = parser.parse_args()

    langs = ["en", "zh"] if args.lang == "both" else [args.lang]
    results: dict[str, int] = {}
    for lang in langs:
        print(f"\n===== {lang} test =====")
        results[lang] = asyncio.run(run_test(lang))

    print("\n===== summary =====")
    for lang, code in results.items():
        status = "OK" if code == 0 else f"FAIL(code={code})"
        print(f"  {lang}: {status}")
    return 0 if all(c == 0 for c in results.values()) else max(results.values())


if __name__ == "__main__":
    sys.exit(main())
