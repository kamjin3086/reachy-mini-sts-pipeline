#!/usr/bin/env python3
"""
Patch installed speech-to-speech Paraformer live transcription behavior.

The upstream Paraformer handler may emit Transcription for both progressive
and final VAD chunks. In realtime mode that causes partial text such as
"今", "今天", "今天天..." to be treated as separate user turns. This patch
keeps progressive chunks as PartialTranscription for UI subtitles and only
emits final Transcription for final/legacy chunks.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


PATCHED_YIELD = """\
        if getattr(vad_audio, "mode", None) == "progressive":
            yield PartialTranscription(text=pred_text)
        else:
            yield Transcription(text=pred_text)
"""

OLD_YIELD = "        yield Transcription(text=pred_text)\n"
OLD_IMPORT = "from speech_to_speech.pipeline.messages import Transcription\n"
NEW_IMPORT = "from speech_to_speech.pipeline.messages import PartialTranscription, Transcription\n"
OLD_PATH_STRIP = """\
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
"""
PATCHED_PATH_STRIP = """\
        # Preserve absolute/local cache paths so startup can run offline.
        if not model_name.startswith("/") and len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
"""
OLD_AUTOMODEL = "        self.model = AutoModel(model=model_name, device=device)\n"
PATCHED_AUTOMODEL = "        self.model = AutoModel(model=model_name, device=device, disable_update=True)\n"


def find_installed_handler() -> Path:
    spec = importlib.util.find_spec("speech_to_speech.STT.paraformer_handler")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not import speech_to_speech.STT.paraformer_handler. "
            "Activate the target venv first."
        )
    return Path(spec.origin)


def patch_text(source: str) -> tuple[str, bool]:
    patched = source

    if OLD_YIELD in patched and PATCHED_YIELD not in patched:
        if OLD_IMPORT not in patched and NEW_IMPORT not in patched:
            raise RuntimeError("Could not find Paraformer messages import to patch.")
        if NEW_IMPORT not in patched:
            patched = patched.replace(OLD_IMPORT, NEW_IMPORT, 1)
        patched = patched.replace(OLD_YIELD, PATCHED_YIELD, 1)

    if OLD_PATH_STRIP in patched and PATCHED_PATH_STRIP not in patched:
        patched = patched.replace(OLD_PATH_STRIP, PATCHED_PATH_STRIP, 1)

    if OLD_AUTOMODEL in patched and PATCHED_AUTOMODEL not in patched:
        patched = patched.replace(OLD_AUTOMODEL, PATCHED_AUTOMODEL, 1)

    if patched != source:
        return patched, True

    if PATCHED_YIELD in source or PATCHED_PATH_STRIP in source or PATCHED_AUTOMODEL in source:
        if PATCHED_YIELD in source and NEW_IMPORT not in source:
            patched = patched.replace(OLD_IMPORT, NEW_IMPORT, 1)
            return patched, patched != source
        return source, False

    if OLD_IMPORT in source:
        raise RuntimeError("Could not find final Transcription yield to patch.")
    if "AutoModel(" in source:
        raise RuntimeError("Could not find Paraformer AutoModel construction to patch.")
    if "model_name.split" in source:
        raise RuntimeError("Could not find Paraformer model path handling to patch.")
    raise RuntimeError("No known Paraformer patch targets found.")


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
        backup_path = path.with_suffix(path.suffix + ".bak")
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
        help="Path to paraformer_handler.py. Defaults to the active venv install.",
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
