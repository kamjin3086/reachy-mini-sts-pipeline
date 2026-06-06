#!/usr/bin/env python3
"""Patch installed speech-to-speech Qwen3 handler with an optional FastAPI bridge.

The bridge is inactive unless STS_QWEN3_OPENAI_FASTAPI_URL is set.  When active,
the Qwen3 TTS handler skips local model loading and streams PCM from an
OpenAI-compatible /v1/audio/speech endpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


IMPORT_OLD = "import logging\nimport os\nimport math\n"
IMPORT_NEW = "import http.client\nimport json\nimport logging\nimport os\nimport math\nfrom urllib.parse import urlparse\n"

SETUP_ANCHOR = """        self._mlx_ref_audio_cache: dict[str, Any] = {}
        self._mlx_temp_ref_audio_files: set[str] = set()

        self.backend = "mlx" if platform == "darwin" else "faster_qwen3_tts"
"""

SETUP_PATCH = """        self._mlx_ref_audio_cache: dict[str, Any] = {}
        self._mlx_temp_ref_audio_files: set[str] = set()

        self.openai_fastapi_url = os.environ.get("STS_QWEN3_OPENAI_FASTAPI_URL", "").strip().rstrip("/")
        self.openai_fastapi_model = os.environ.get("STS_QWEN3_OPENAI_FASTAPI_MODEL", "qwen3-tts").strip()
        self.openai_fastapi_voice = os.environ.get("STS_QWEN3_OPENAI_FASTAPI_VOICE", self.speaker or "Serena").strip()
        fastapi_language = os.environ.get("STS_QWEN3_OPENAI_FASTAPI_LANGUAGE", "").strip()
        if not fastapi_language:
            fastapi_language = "Chinese" if str(self.language).lower() in {"zh", "chinese", "中文"} else self.language
        self.openai_fastapi_language = fastapi_language
        self.openai_fastapi_stream = os.environ.get("STS_QWEN3_OPENAI_FASTAPI_STREAM", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.openai_fastapi_read_size = int(os.environ.get("STS_QWEN3_OPENAI_FASTAPI_READ_SIZE", "4096"))

        self.backend = "mlx" if platform == "darwin" else "faster_qwen3_tts"
        if self.openai_fastapi_url:
            self.backend = "openai_fastapi"
            self.device = device
            self.model_name = model_name
            self.streaming_chunk_size = self._resolve_streaming_chunk_size(streaming_chunk_size)
            self._initial_speaker = self.speaker
            self._initial_ref_audio = self.ref_audio
            logger.info("Using Qwen3-TTS OpenAI FastAPI bridge: %s", self.openai_fastapi_url)
            return
"""

PROCESS_OLD = """        text = coalesced_text or "Hello."

        model_type = self._model_type()
        self._apply_session_voice_override(model_type, runtime_config, response)

        text, inline_instruct = self._extract_inline_instruct(text)
        original_instruct = self.instruct
        if inline_instruct:
            self.instruct = f"{original_instruct}；{inline_instruct}" if original_instruct else inline_instruct

        console.print(f"[green]ASSISTANT: {text}")

        try:
            if self.ref_audio:
"""

PROCESS_NEW = """        text = coalesced_text or "Hello."

        if getattr(self, "backend", None) == "openai_fastapi":
            text, inline_instruct = self._extract_inline_instruct(text)
            original_instruct = self.instruct
            if inline_instruct:
                self.instruct = f"{original_instruct}；{inline_instruct}" if original_instruct else inline_instruct
            console.print(f"[green]ASSISTANT: {text}")
            try:
                yield from self._process_openai_fastapi(text)
            except Exception as e:
                logger.error(f"Error during Qwen3-TTS OpenAI FastAPI bridge generation: {e}", exc_info=True)
            finally:
                self.instruct = original_instruct
            return

        model_type = self._model_type()
        self._apply_session_voice_override(model_type, runtime_config, response)

        text, inline_instruct = self._extract_inline_instruct(text)
        original_instruct = self.instruct
        if inline_instruct:
            self.instruct = f"{original_instruct}；{inline_instruct}" if original_instruct else inline_instruct

        console.print(f"[green]ASSISTANT: {text}")

        try:
            if self.ref_audio:
"""

METHOD_ANCHOR = """    def _process_custom_voice(self, text: str) -> Iterator[bytes | np.ndarray]:
"""

METHOD_PATCH = """    def _process_openai_fastapi(self, text: str) -> Iterator[bytes | np.ndarray]:
        parsed = urlparse(self.openai_fastapi_url)
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        prefix = parsed.path.rstrip("/")
        path = f"{prefix}/v1/audio/speech"
        body = json.dumps(
            {
                "model": self.openai_fastapi_model,
                "voice": self.openai_fastapi_voice or self.speaker or "Serena",
                "input": text,
                "language": self.openai_fastapi_language,
                "instruct": self.instruct,
                "response_format": "pcm",
                "stream": self.openai_fastapi_stream,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        start = perf_counter()
        conn = conn_cls(host, port, timeout=240)
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json", "Accept": "audio/pcm"},
        )
        response = conn.getresponse()
        if response.status >= 400:
            error = response.read().decode("utf-8", errors="replace")
            conn.close()
            raise RuntimeError(f"OpenAI FastAPI TTS HTTP {response.status}: {error}")

        leftover_bytes = b""
        leftover_audio = np.array([], dtype=np.int16)
        total_samples = 0
        first_chunk = True
        try:
            while True:
                data = response.read(self.openai_fastapi_read_size)
                if not data:
                    break
                if first_chunk:
                    logger.info("Qwen3-TTS OpenAI FastAPI TTFA: %.2fs", perf_counter() - start)
                    first_chunk = False
                data = leftover_bytes + data
                if len(data) % 2:
                    leftover_bytes = data[-1:]
                    data = data[:-1]
                else:
                    leftover_bytes = b""
                if not data:
                    continue

                audio_chunk = np.frombuffer(data, dtype="<i2").copy()
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                audio_chunk = self._resample_to_pipeline_sr(audio_chunk, 24000)
                audio_chunk = self._to_int16(audio_chunk)
                audio_chunk = np.concatenate([leftover_audio, audio_chunk])

                n = (len(audio_chunk) // self.blocksize) * self.blocksize
                for i in range(0, n, self.blocksize):
                    yield audio_chunk[i : i + self.blocksize]
                    total_samples += self.blocksize
                leftover_audio = audio_chunk[n:]

            if len(leftover_audio) > 0:
                yield np.pad(leftover_audio, (0, self.blocksize - len(leftover_audio)))
                total_samples += len(leftover_audio)
        finally:
            conn.close()

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            "Qwen3-TTS OpenAI FastAPI generated %.2fs audio in %.2fs (RTF: %.2f)",
            audio_duration,
            generation_time,
            rtf,
        )

""" + METHOD_ANCHOR

CLEANUP_OLD = """    def cleanup(self) -> None:
        try:
            del self.model
"""

CLEANUP_NEW = """    def cleanup(self) -> None:
        try:
            if getattr(self, "backend", None) == "openai_fastapi":
                logger.info("Qwen3-TTS OpenAI FastAPI bridge cleaned up")
                return
            del self.model
"""


def default_handler_path() -> Path:
    spec = importlib.util.find_spec("speech_to_speech.TTS.qwen3_tts_handler")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not import speech_to_speech.TTS.qwen3_tts_handler")
    return Path(spec.origin)


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        print(f"[ok] {label}: already patched")
        return text, False
    if old not in text:
        raise RuntimeError(f"Could not find patch target for {label}")
    print(f"[patch] {label}")
    return text.replace(old, new, 1), True


def patch(path: Path, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    for label, old, new in (
        ("imports", IMPORT_OLD, IMPORT_NEW),
        ("setup bridge", SETUP_ANCHOR, SETUP_PATCH),
        ("process bridge", PROCESS_OLD, PROCESS_NEW),
        ("bridge method", METHOD_ANCHOR, METHOD_PATCH),
        ("cleanup bridge", CLEANUP_OLD, CLEANUP_NEW),
    ):
        text, did = replace_once(text, old, new, label)
        changed = changed or did
    if changed and not dry_run:
        backup = path.with_suffix(path.suffix + ".openai_fastapi_bridge.bak")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = args.target or default_handler_path()
    changed = patch(target, dry_run=args.dry_run)
    print(f"[done] {'would patch' if args.dry_run and changed else 'patched' if changed else 'no changes needed'}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
