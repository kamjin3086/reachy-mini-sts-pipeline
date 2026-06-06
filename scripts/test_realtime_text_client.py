#!/usr/bin/env python3
"""
Exercise a running speech-to-speech realtime server with text turns.

This client verifies the OpenAI Realtime event path used by the Reachy app:
session tools, response.create, audio deltas, transcript events, and function
call events. It writes raw events and a compact summary to an output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import websockets


DEFAULT_TOOLS = [
    {
        "type": "function",
        "name": "reachy_action",
        "description": (
            "Execute a Reachy Mini action or expression. Use nod for 点头, "
            "happy for 开心表情/高兴表情, and dance for 跳舞."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["nod", "happy", "dance"],
                    "description": "Action to execute: nod=点头, happy=开心表情, dance=跳舞.",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }
]


PROMPTS = [
    "请用开心的语气说：你好，我准备好了。",
    "请点点头，然后用一句自然中文回复我。",
    "做个开心表情，不要把工具调用内容念出来。",
]


INSTRUCTIONS = (
    "你是 Reachy Mini 的中文语音助手。默认中文短回复。"
    "动作、表情和跳舞必须通过工具调用执行；不要朗读 JSON、工具名、参数或动作标记。"
    "点头使用 nod，开心表情使用 happy，跳舞使用 dance。"
    "如需表达语气，可以在回复开头用括号提示语气，例如（开心地），但不要解释这个提示。"
)


async def recv_until_done(ws: Any, timeout_s: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        event = json.loads(raw)
        events.append(event)
        if event.get("type") == "response.done":
            break
        if event.get("type") == "error":
            break
    return events


def summarize_events(events: list[dict[str, Any]], prompt: str, elapsed_s: float) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    first_audio_index = None
    transcripts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, event in enumerate(events):
        event_type = str(event.get("type"))
        by_type[event_type] = by_type.get(event_type, 0) + 1
        if event_type in ("response.audio.delta", "response.output_audio.delta") and first_audio_index is None:
            first_audio_index = index
        if event_type == "response.output_audio_transcript.done":
            transcripts.append(str(event.get("transcript", "")))
        if event_type == "response.function_call_arguments.done":
            tool_calls.append(
                {
                    "name": event.get("name"),
                    "arguments": event.get("arguments"),
                    "call_id": event.get("call_id"),
                }
            )
        if event_type == "error":
            errors.append(event)

    transcript_text = "".join(transcripts)
    return {
        "prompt": prompt,
        "elapsed_s": round(elapsed_s, 4),
        "event_count": len(events),
        "events_by_type": by_type,
        "has_audio_delta": (
            by_type.get("response.audio.delta", 0) + by_type.get("response.output_audio.delta", 0)
        )
        > 0,
        "first_audio_event_index": first_audio_index,
        "transcript": transcript_text,
        "tool_calls": tool_calls,
        "error_count": len(errors),
        "errors": errors,
        "tts_leaked_tool_text": any(token in transcript_text for token in ("reachy_action", "arguments", "{", "}")),
        "tts_leaked_style_marker": any(token in transcript_text for token in ("（", "）", "[", "]")),
    }


async def run(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    raw_path = output_dir / "realtime_events.jsonl"
    prompts = args.prompts or PROMPTS

    async with websockets.connect(args.url, ping_interval=None) as ws:
        initial = await recv_until_done(ws, timeout_s=2.0)

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": args.instructions,
                        "tools": DEFAULT_TOOLS,
                        "tool_choice": "auto",
                        "audio": {"output": {"voice": args.voice}},
                    },
                },
                ensure_ascii=False,
            )
        )
        session_events = await recv_until_done(ws, timeout_s=1.0)

        with raw_path.open("w", encoding="utf-8") as raw_file:
            for event in initial + session_events:
                raw_file.write(json.dumps(event, ensure_ascii=False) + "\n")

            for prompt in prompts:
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": prompt}],
                            },
                        },
                        ensure_ascii=False,
                    )
                )
                await recv_until_done(ws, timeout_s=1.0)
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "instructions": args.instructions,
                                "tools": DEFAULT_TOOLS,
                                "tool_choice": "auto",
                                "output_modalities": ["audio"],
                            },
                        },
                        ensure_ascii=False,
                    )
                )
                started = time.perf_counter()
                events = await recv_until_done(ws, timeout_s=args.timeout)
                elapsed = time.perf_counter() - started
                for event in events:
                    raw_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                summaries.append(summarize_events(events, prompt, elapsed))

    summary = {
        "created_at": timestamp,
        "url": args.url,
        "voice": args.voice,
        "instructions": args.instructions,
        "raw_events": str(raw_path),
        "turns": summaries,
    }
    (output_dir / "realtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary written: {output_dir / 'realtime_summary.json'}")
    return 1 if any(turn["error_count"] for turn in summaries) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "apps/sts-cache/bench/realtime",
    )
    parser.add_argument("--voice", default="Serena")
    parser.add_argument("--instructions", default=INSTRUCTIONS)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--prompts", nargs="*")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
