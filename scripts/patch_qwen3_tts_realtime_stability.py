#!/usr/bin/env python3
"""
Patch installed speech-to-speech defaults for Qwen3-TTS realtime stability.

This keeps the current venv aligned with scripts/sts_start.sh:
- faster-qwen3-tts keeps full-text prefill enabled to avoid repeated TTFA stalls.
- Realtime audio batches are slightly larger to reduce client playback jitter.
- Barge-in defaults to off unless the client explicitly enables it.
- Optional full-utterance buffering avoids mid-sentence underruns on ROCm devices
  where Qwen3-TTS generation is slower than realtime playback.
- The per-utterance token budget floor is reduced for short voice-assistant
  turns so EOS sampling failures do not spend up to ~30 seconds decoding silence
  or discarded tail audio.
- faster-qwen3-tts generation kwargs are forwarded so sampling can be tuned
  for more stable EOS timing on ROCm.
- STS_QWEN3_TTS_DO_SAMPLE can override faster-qwen3-tts sampling from the
  startup script.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


PATCHES = {
    "speech_to_speech.TTS.qwen3_tts_handler": (
        ("import logging\n", "import logging\nimport os\n"),
        ("non_streaming_mode: bool | None = False,", "non_streaming_mode: bool | None = True,"),
        ("MIN_QWEN3_TTS_UTTERANCE_TOKENS = 360", "MIN_QWEN3_TTS_UTTERANCE_TOKENS = 128"),
        (
            "        self.gen_kwargs = gen_kwargs or {}\n",
            "        self.gen_kwargs = dict(gen_kwargs or {})\n"
            "        env_do_sample = os.environ.get(\"STS_QWEN3_TTS_DO_SAMPLE\")\n"
            "        if env_do_sample is not None and env_do_sample.strip():\n"
            "            self.gen_kwargs[\"do_sample\"] = env_do_sample.strip().lower() in {\n"
            "                \"1\",\n"
            "                \"true\",\n"
            "                \"yes\",\n"
            "                \"on\",\n"
            "            }\n",
        ),
        (
            "        leftover = np.array([], dtype=np.int16)\n",
            "        leftover = np.array([], dtype=np.int16)\n"
            "        buffer_full_utterance = os.environ.get(\n"
            "            \"STS_QWEN3_TTS_BUFFER_FULL_UTTERANCE\", \"\"\n"
            "        ).strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
            "        buffered_chunks: list[np.ndarray] = []\n",
        ),
        (
            "            for i in range(0, n, self.blocksize):\n"
            "                yield audio_chunk[i : i + self.blocksize]\n"
            "                total_samples += self.blocksize\n",
            "            for i in range(0, n, self.blocksize):\n"
            "                block = audio_chunk[i : i + self.blocksize]\n"
            "                if buffer_full_utterance:\n"
            "                    buffered_chunks.append(block)\n"
            "                else:\n"
            "                    yield block\n"
            "                total_samples += self.blocksize\n",
        ),
        (
            "        if len(leftover) > 0:\n"
            "            chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))\n"
            "            yield chunk\n"
            "            total_samples += len(leftover)\n\n"
            "        generation_time = perf_counter() - start\n",
            "        if len(leftover) > 0:\n"
            "            chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))\n"
            "            if buffer_full_utterance:\n"
            "                buffered_chunks.append(chunk)\n"
            "            else:\n"
            "                yield chunk\n"
            "            total_samples += len(leftover)\n\n"
            "        generation_time = perf_counter() - start\n"
            "        if buffer_full_utterance:\n"
            "            logger.info(\n"
            "                \"Qwen3-TTS full-utterance buffering enabled; releasing %d audio chunks\",\n"
            "                len(buffered_chunks),\n"
            "            )\n"
            "            for chunk in buffered_chunks:\n"
            "                yield chunk\n",
        ),
        (
            "                parity_mode=self.parity_mode,\n"
            "                non_streaming_mode=self.non_streaming_mode,\n"
            "            ),\n",
            "                parity_mode=self.parity_mode,\n"
            "                non_streaming_mode=self.non_streaming_mode,\n"
            "                **self.gen_kwargs,\n"
            "            ),\n",
        ),
        (
            "                max_new_tokens=utterance_max_new_tokens,\n"
            "                non_streaming_mode=self.non_streaming_mode,\n"
            "            ),\n"
            "            label=\"custom_voice\",\n",
            "                max_new_tokens=utterance_max_new_tokens,\n"
            "                non_streaming_mode=self.non_streaming_mode,\n"
            "                **self.gen_kwargs,\n"
            "            ),\n"
            "            label=\"custom_voice\",\n",
        ),
        (
            "                max_new_tokens=utterance_max_new_tokens,\n"
            "                non_streaming_mode=self.non_streaming_mode,\n"
            "            ),\n"
            "            label=\"voice_design\",\n",
            "                max_new_tokens=utterance_max_new_tokens,\n"
            "                non_streaming_mode=self.non_streaming_mode,\n"
            "                **self.gen_kwargs,\n"
            "            ),\n"
            "            label=\"voice_design\",\n",
        ),
    ),
    "speech_to_speech.arguments_classes.qwen3_tts_arguments": (
        (
            "qwen3_tts_xvec_only: bool = field(\n        default=True,",
            "qwen3_tts_xvec_only: bool = field(\n        default=False,",
        ),
        (
            "qwen3_tts_non_streaming_mode: Optional[bool] = field(\n        default=False,",
            "qwen3_tts_non_streaming_mode: Optional[bool] = field(\n        default=True,",
        ),
        (
            '"help": "Optional override for Qwen3-TTS text prefill behavior. Default is false, which streams codec chunks as they are decoded on faster-qwen3-tts. Currently ignored on Apple Silicon because mlx-audio does not expose this yet."',
            '"help": "Optional override for Qwen3-TTS text prefill behavior. Default is true, which pre-fills the full target text before decode on faster-qwen3-tts to reduce repeated TTFA stalls. Currently ignored on Apple Silicon because mlx-audio does not expose this yet."',
        ),
    ),
    "speech_to_speech.api.openai_realtime.websocket_router": (
        ("MAX_AUDIO_BATCH_BYTES = 6400", "MAX_AUDIO_BATCH_BYTES = 9600"),
    ),
    "speech_to_speech.api.openai_realtime.runtime_config": (
        ("Defaults to 'True' (OpenAI API default).", "Defaults to 'False' for local speaker/microphone setups to avoid self-interruption."),
        ("if td is None:\n            return True", "if td is None:\n            return False"),
        ('td.get("interrupt_response", True)', 'td.get("interrupt_response", False)'),
        ("return True\n        return val if val is not None else True", "return False\n        return val if val is not None else False"),
    ),
}


def find_module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Could not import {module_name!r}. Activate the target venv first.")
    return Path(spec.origin)


def patch_file(path: Path, replacements: tuple[tuple[str, str], ...], *, dry_run: bool, backup: bool) -> bool:
    source = path.read_text(encoding="utf-8")
    patched = source

    for old, new in replacements:
        if new in patched:
            continue
        if old in patched:
            patched = patched.replace(old, new, 1)
        else:
            raise RuntimeError(f"Could not find patch target in {path}: {old!r}")

    if patched == source:
        print(f"already patched: {path}")
        return False

    if dry_run:
        print(f"would patch: {path}")
        return True

    if backup:
        backup_path = path.with_suffix(path.suffix + ".realtime_stability.bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"backup written: {backup_path}")

    path.write_text(patched, encoding="utf-8")
    print(f"patched: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check without writing.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files.")
    args = parser.parse_args()

    try:
        changed = False
        for module_name, replacements in PATCHES.items():
            changed |= patch_file(
                find_module_path(module_name),
                replacements,
                dry_run=args.dry_run,
                backup=not args.no_backup,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not changed:
        print("no changes needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
