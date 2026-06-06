#!/usr/bin/env python3
"""
Patch installed Qwen3-TTS inline style instruction handling.

The speech-to-speech Qwen3 handler already passes ``qwen3_tts_instruct`` to
Qwen3 CustomVoice/VoiceDesign generation. This patch adds a small compatibility
layer for assistant text like "(开心地)你好" or "[小声]我在这里": leading style
markers are extracted into the TTS instruct field and removed from spoken text.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


IMPORT_MARKER = "from time import perf_counter\n"
IMPORT_PATCH = "from time import perf_counter\nimport re\n"

HELPER_MARKER = "    def _extract_inline_instruct(self, text: str) -> tuple[str, str | None]:\n"
HELPER_BLOCK = '''\
    def _extract_inline_instruct(self, text: str) -> tuple[str, str | None]:
        """Extract leading style hints and keep them out of spoken text."""
        cleaned = text.strip()
        hints: list[str] = []
        patterns = (
            r"^[（(]\\s*([^（）()]{1,40}?)\\s*[）)]\\s*",
            r"^[［\\[]\\s*([^［］\\[\\]]{1,40}?)\\s*[］\\]]\\s*",
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

'''

PROCESS_OLD = '''\
        text = coalesced_text or "Hello."

        model_type = self._model_type()
        self._apply_session_voice_override(model_type, runtime_config, response)

        console.print(f"[green]ASSISTANT: {text}")

        try:
            if self.ref_audio:
                yield from self._process_voice_clone(text)
            elif model_type == "custom_voice":
                yield from self._process_custom_voice(text)
            elif model_type == "voice_design":
                yield from self._process_voice_design(text)
            else:
                raise ValueError(
                    "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                    "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
                )
        except Exception as e:
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)
'''

PROCESS_NEW = '''\
        text = coalesced_text or "Hello."

        model_type = self._model_type()
        self._apply_session_voice_override(model_type, runtime_config, response)

        text, inline_instruct = self._extract_inline_instruct(text)
        original_instruct = self.instruct
        if inline_instruct:
            self.instruct = f"{original_instruct}；{inline_instruct}" if original_instruct else inline_instruct

        console.print(f"[green]ASSISTANT: {text}")

        try:
            if self.ref_audio:
                yield from self._process_voice_clone(text)
            elif model_type == "custom_voice":
                yield from self._process_custom_voice(text)
            elif model_type == "voice_design":
                yield from self._process_voice_design(text)
            else:
                raise ValueError(
                    "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                    "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
                )
        except Exception as e:
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)
        finally:
            self.instruct = original_instruct
'''


def find_installed_handler() -> Path:
    spec = importlib.util.find_spec("speech_to_speech.TTS.qwen3_tts_handler")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not import speech_to_speech.TTS.qwen3_tts_handler. "
            "Activate the target venv first."
        )
    return Path(spec.origin)


def patch_text(source: str) -> tuple[str, bool]:
    patched = source

    if "\nimport re\n" not in patched and IMPORT_PATCH not in patched:
        if IMPORT_MARKER not in patched:
            raise RuntimeError("Could not find time import to patch.")
        patched = patched.replace(IMPORT_MARKER, IMPORT_PATCH, 1)

    if HELPER_MARKER not in patched:
        anchor = "    def _process_voice_clone(self, text: str) -> Iterator[bytes | np.ndarray]:\n"
        if anchor not in patched:
            raise RuntimeError("Could not find Qwen3 voice-clone method anchor.")
        patched = patched.replace(anchor, HELPER_BLOCK + anchor, 1)

    inline_instruct_already_present = (
        "text, inline_instruct = self._extract_inline_instruct(text)" in patched
        and "original_instruct = self.instruct" in patched
        and "self.instruct = original_instruct" in patched
    )
    if PROCESS_NEW not in patched and not inline_instruct_already_present:
        if PROCESS_OLD not in patched:
            raise RuntimeError("Could not find Qwen3 process block to patch.")
        patched = patched.replace(PROCESS_OLD, PROCESS_NEW, 1)

    return patched, patched != source


def patch_file(path: Path, *, dry_run: bool = False, backup: bool = True) -> bool:
    source = path.read_text(encoding="utf-8")
    patched, changed = patch_text(source)

    if not changed:
        print(f"already patched: {path}")
        return False

    if dry_run:
        print(f"would patch: {path}")
        return True

    if backup:
        backup_path = path.with_suffix(path.suffix + ".inline_instruct.bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"backup written: {backup_path}")

    path.write_text(patched, encoding="utf-8")
    print(f"patched: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        help="Path to qwen3_tts_handler.py. Defaults to the active venv install.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check without writing.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak file.")
    args = parser.parse_args()

    try:
        target = args.target or find_installed_handler()
        patch_file(target, dry_run=args.dry_run, backup=not args.no_backup)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
