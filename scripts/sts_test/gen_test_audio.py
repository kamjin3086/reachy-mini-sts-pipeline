"""Generate 24 kHz mono PCM16 WAV test clips (English + Chinese) via edge-tts.

Why 24 kHz: HuggingFace's openai_realtime gateway validates `session.update`
against the OpenAI Pydantic models, where `AudioPCM.rate` is the literal
`24000` only. The server's internal pipeline still operates at 16 kHz, so the
declared rate is the *wire* format and the gateway resamples as needed.
"""
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts

OUT_DIR = Path(__file__).parent / "test_data"

CLIPS = [
    {
        "name": "en",
        "text": (
            "Hello, this is a test of the speech to speech pipeline. "
            "Please respond in a short sentence so I can verify the audio output."
        ),
        "voice": "en-US-AriaNeural",
    },
    {
        "name": "zh",
        "text": (
            "你好,这是一个语音到语音管道的测试。"
            "请用一两句话简短地回复,方便我验证音频输出。"
        ),
        "voice": "zh-CN-XiaoxiaoNeural",
    },
]


async def synth(mp3_path: Path, text: str, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(str(mp3_path))


def to_wav_mono_pcm16(src: Path, dst: Path, rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",          # mono
        "-ar",
        str(rate),
        "-acodec",
        "pcm_s16le",  # PCM 16-bit little-endian
        "-fflags",
        "+bitexact",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(CLIPS, 1):
        wav_path = OUT_DIR / f"test_input_{clip['name']}_24k_mono.wav"
        with tempfile.TemporaryDirectory() as tmp:
            mp3_path = Path(tmp) / "edge_tts.mp3"
            print(f"[{i}/{len(CLIPS)}] {clip['name']}: synth -> {wav_path.name}")
            asyncio.run(synth(mp3_path, clip["text"], clip["voice"]))
            to_wav_mono_pcm16(mp3_path, wav_path, 24000)
        print(f"  OK: {wav_path}  ({wav_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
