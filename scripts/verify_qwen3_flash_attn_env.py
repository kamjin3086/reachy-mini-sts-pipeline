#!/usr/bin/env python3
"""Verify the ROCm flash-attn environment used by the Qwen3-TTS FastAPI path."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-smoke",
        action="store_true",
        help="Run a tiny flash_attn_func call. This may trigger Triton/aiter JIT.",
    )
    args = parser.parse_args()

    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        warn("HSA_OVERRIDE_GFX_VERSION is set; unset it for ROCm 7.13/gfx1151.")

    import flash_attn  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import triton  # noqa: PLC0415

    report: dict[str, Any] = {
        "python": sys.executable,
        "packages": {
            "flash-attn": package_version("flash-attn"),
            "amd-aiter": package_version("amd-aiter"),
            "torch": package_version("torch"),
            "torchaudio": package_version("torchaudio"),
            "torchvision": package_version("torchvision"),
            "triton": package_version("triton"),
            "speech-to-speech": package_version("speech-to-speech"),
            "qwen-tts": package_version("qwen-tts"),
            "faster-qwen3-tts": package_version("faster-qwen3-tts"),
            "numpy": package_version("numpy"),
            "packaging": package_version("packaging"),
        },
        "runtime": {
            "flash_attn_module_version": getattr(flash_attn, "__version__", None),
            "torch_version": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "triton_version": triton.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        },
        "env": {
            "FLASH_ATTENTION_TRITON_AMD_ENABLE": os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE"),
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get(
                "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
            ),
            "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        },
    }

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is false; ROCm is not usable.")
    if "rocm" not in torch.__version__:
        fail(f"torch is not a ROCm build: {torch.__version__}")
    if not getattr(torch.version, "hip", None):
        fail("torch.version.hip is empty; this is not the expected HIP runtime.")
    triton_dist_version = package_version("triton")
    if not triton_dist_version or "rocm" not in triton_dist_version:
        fail(f"triton is not the TheRock ROCm build: {triton_dist_version}")
    if package_version("flash-attn") != "2.8.4":
        fail(f"unexpected flash-attn version: {package_version('flash-attn')}")
    if package_version("amd-aiter") is None:
        fail("amd-aiter is missing; flash-attn ROCm Triton backend will not work.")
    if not any("gfx1151" in arch for arch in report["runtime"]["arch_list"]):
        warn(f"gfx1151 not present in torch arch list: {report['runtime']['arch_list']}")

    if args.kernel_smoke:
        from flash_attn import flash_attn_func  # noqa: PLC0415

        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        torch.cuda.synchronize()
        report["kernel_smoke"] = {
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "finite": bool(torch.isfinite(out).all().item()),
        }
        if not report["kernel_smoke"]["finite"]:
            fail("flash_attn_func returned non-finite values.")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[OK] ROCm flash-attn environment looks usable.")


if __name__ == "__main__":
    main()
