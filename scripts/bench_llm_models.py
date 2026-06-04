#!/usr/bin/env python3
"""
快速对比多个 LLM 模型的 TTFT（首 token 延迟）。
跳过模型加载（首次会很慢），测量已加载模型的稳态 TTFT。
"""
import os
import time
import sys

os.environ.pop("OPENAI_API_KEY", None)

from openai import OpenAI

BASE = "http://127.0.0.1:8101/v1"
PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "用中文写一首关于秋天的四行诗。",
    "解释什么是 ROCm 和 HIP。",
]

CANDIDATES = [
    "Gemma-4-E4B-instruct",
    "Gemma-4-E2B-instruct",
    "Qwen3.5-4b-FLM-instruct",
    "Qwen3.5-9b-FLM-instruct",
    "GPT-OSS-20B",
    "Qwen3.6-35B-A3B-instruct",
    "Step-3.5-Flash-normal",
]


def measure(model: str, prompts: list, *, max_tokens: int = 60):
    client = OpenAI(base_url=BASE, api_key="")
    results = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        first_token_at = None
        tokens = 0
        full_text = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.6,
                stream=True,
            )
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    full_text += chunk.choices[0].delta.content
                    tokens += 1
        except Exception as e:
            return [{"model": model, "error": str(e)}]
        total = time.perf_counter() - t0
        ttft = (first_token_at - t0) if first_token_at else total
        gen_time = total - ttft
        tok_per_s = tokens / gen_time if gen_time > 0 else 0
        results.append({
            "model": model,
            "prompt_idx": i,
            "ttft_s": round(ttft, 3),
            "total_s": round(total, 3),
            "tokens": tokens,
            "tok_per_s": round(tok_per_s, 1),
            "preview": full_text[:40].replace("\n", " "),
        })
    return results


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else CANDIDATES

    client = OpenAI(base_url=BASE, api_key="")
    available = {m.id for m in client.models.list().data}
    print("=" * 70)
    print(f"测试目标: {[t for t in targets if t in available]}")
    print(f"未在 llama-swap 中: {[t for t in targets if t not in available]}")
    print("=" * 70)

    summary = {}
    for model in targets:
        if model not in available:
            continue
        print(f"\n>>> {model}")
        all_runs = measure(model, PROMPTS)
        if all_runs and "error" in all_runs[0]:
            print(f"  ERROR: {all_runs[0]['error'][:200]}")
            summary[model] = {"error": all_runs[0]["error"]}
            continue
        for r in all_runs:
            print(f"  p{r['prompt_idx']}: TTFT={r['ttft_s']:.3f}s  total={r['total_s']:.3f}s  "
                  f"tokens={r['tokens']}  tok/s={r['tok_per_s']:.1f}  | {r['preview']}")
        avg_ttft = sum(r["ttft_s"] for r in all_runs) / len(all_runs)
        avg_tok = sum(r["tok_per_s"] for r in all_runs) / len(all_runs)
        summary[model] = {"avg_ttft": round(avg_ttft, 3), "avg_tok_per_s": round(avg_tok, 1)}

    print("\n" + "=" * 70)
    print("  排名（按 TTFT 升序）")
    print("=" * 70)
    valid = [(m, s["avg_ttft"]) for m, s in summary.items() if "avg_ttft" in s]
    valid.sort(key=lambda x: x[1])
    for m, t in valid:
        marker = " ← 当前" if m == "Gemma-4-E4B-instruct" else ""
        print(f"  {t:6.3f}s  {summary[m]['avg_tok_per_s']:6.1f} tok/s  {m}{marker}")


if __name__ == "__main__":
    main()
