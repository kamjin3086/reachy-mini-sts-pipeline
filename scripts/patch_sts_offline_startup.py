#!/usr/bin/env python3
"""Patch speech-to-speech startup checks to avoid network access after caches exist."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


OLD_NLTK_TAGGER_PATH = '    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")\n'
NEW_NLTK_TAGGER_PATH = '    nltk.data.find("taggers/averaged_perceptron_tagger_eng")\n'


def find_installed_pipeline() -> Path:
    spec = importlib.util.find_spec("speech_to_speech.s2s_pipeline")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not import speech_to_speech.s2s_pipeline. Activate the target venv first.")
    return Path(spec.origin)


def patch_text(source: str) -> tuple[str, bool]:
    if NEW_NLTK_TAGGER_PATH in source:
        return source, False
    if OLD_NLTK_TAGGER_PATH not in source:
        raise RuntimeError("Could not find NLTK tagger resource check to patch.")
    return source.replace(OLD_NLTK_TAGGER_PATH, NEW_NLTK_TAGGER_PATH, 1), True


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
    parser.add_argument("--target", type=Path, help="Path to s2s_pipeline.py. Defaults to active venv install.")
    parser.add_argument("--dry-run", action="store_true", help="Check without writing.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak file.")
    args = parser.parse_args()

    try:
        patch_file(args.target or find_installed_pipeline(), dry_run=args.dry_run, backup=not args.no_backup)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
