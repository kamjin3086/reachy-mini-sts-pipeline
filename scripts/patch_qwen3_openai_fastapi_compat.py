#!/usr/bin/env python3
"""Patch Qwen3-TTS-Openai-Fastapi for the local ROCm/qwen_tts runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        print(f"[ok] {label}: already patched")
        return text, False
    if old not in text:
        raise RuntimeError(f"Could not find patch target for {label}")
    print(f"[patch] {label}")
    return text.replace(old, new, 1), True


def patch_backend(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    text, did = replace_once(text, "import os\nimport yaml\n", "import os\nimport inspect\nimport yaml\n", "inspect import")
    changed = changed or did

    old = """                self.model.enable_streaming_optimizations(
                    decode_window_frames=streaming_opts.get("decode_window_frames", 80),
                    use_compile=True,
                    use_cuda_graphs=opt.get("use_cuda_graphs", False),
                    compile_mode=opt.get("compile_mode", "max-autotune"),
                    use_fast_codebook=opt.get("use_fast_codebook", True),
                    compile_codebook_predictor=opt.get("compile_codebook_predictor", True),
                    compile_talker=opt.get("compile_talker", False),
                )
"""
    new = """                compile_kwargs = {
                    "decode_window_frames": streaming_opts.get("decode_window_frames", 80),
                    "use_compile": True,
                    "use_cuda_graphs": opt.get("use_cuda_graphs", False),
                    "compile_mode": opt.get("compile_mode", "max-autotune"),
                    "use_fast_codebook": opt.get("use_fast_codebook", True),
                    "compile_codebook_predictor": opt.get("compile_codebook_predictor", True),
                    "compile_talker": opt.get("compile_talker", False),
                }
                supported = inspect.signature(self.model.enable_streaming_optimizations).parameters
                compile_kwargs = {k: v for k, v in compile_kwargs.items() if k in supported}
                self.model.enable_streaming_optimizations(**compile_kwargs)
"""
    text, did = replace_once(text, old, new, "streaming optimization kwargs")
    changed = changed or did

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/home/kamjin/apps/Qwen3-TTS-Openai-Fastapi"),
        help="Path to the Qwen3-TTS-Openai-Fastapi checkout.",
    )
    args = parser.parse_args()
    backend = args.repo_dir / "api" / "backends" / "optimized_backend.py"
    if not backend.exists():
        raise SystemExit(f"Missing optimized backend: {backend}")
    changed = patch_backend(backend)
    print(f"[done] {'patched' if changed else 'no changes needed'}: {backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
