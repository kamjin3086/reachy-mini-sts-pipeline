# Troubleshooting Quick Reference

> [中文版](TROUBLESHOOTING.zh.md) · [← Back to README](README.md)
>
> Full known-issues list: [docs/03 §Known Issues & Workarounds](docs/03-speech-to-speech-status.md). This file is a **quick-lookup** cheat sheet.

## Startup failures

| Symptom | Cause | One-line fix |
|---|---|---|
| `hipErrorInvalidImage` | `HSA_OVERRIDE_GFX_VERSION=11.0.0` triggered on ROCm 7.13 | Remove the variable from `sts_start.sh` |
| `HF_HUB_ENABLE_HF_TRANSFER` error | hf_transfer not installed | `uv pip install hf_transfer` |
| `ModuleNotFoundError: No module named 'df'` | DeepFilterNet installed but not patched | See [DeepFilterNet patch](#deepfilternet-torchaudio-incompat) below |
| `Model 'X' not found` (LLM server side) | Model not registered with your OpenAI-compatible server | Change `--model_name` in `sts_start.sh` to a model your server hosts (e.g. one already loaded into `llama-server`) |
| `unable to start process: upstream command exited prematurely but successfully` | LLM server-side config issue (model command invalid, OOM, etc.) | Check the LLM server's own logs; verify the model loads cleanly when invoked directly |
| `pip` points to Python 3.14, not venv | `~/.local/bin/pip` in PATH takes precedence | `export VIRTUAL_ENV=/home/kamjin/apps/.venv` then use `uv pip` |

## Reachy Mini connection

| Symptom | Cause | Where to look |
|---|---|---|
| App connects but "nothing happens" | See [docs/04](docs/04-reachy-mini-debug-journey.md) | Check server log for WebSocket handshake, mic permissions, whether STT model actually loaded |
| VAD never triggers | Mic permission off / volume too low | Check audio input in Reachy Mini Control app |
| App shows connected but no response | WebSocket protocol mismatch | Confirm `ws://<server-ip>:8765/v1/realtime` path is correct |

## Performance problems

| Symptom | Cause | Fix |
|---|---|---|
| E2E perceived latency 4 s+ | LLM is on NPU model or Step-3.5-Flash | Switch back to `Gemma-4-E4B-instruct` (steady-state TTFT 50 ms) |
| First TTS call 12 s+ | CUDA graph compile | Unavoidable, but can warmup at end of `sts_start.sh` |
| STT reports MPS error | Library bug: `paraformer_handler.py:56` | See [MPS bug patch](#mps-bug) below |
| LLM 401 invalid_api_key | `OPENAI_API_KEY` client env interference | `unset OPENAI_API_KEY` and restart |

## Package install failures

| Symptom | Cause | Fix |
|---|---|---|
| `uv pip install` reports "no wheel for your platform" | Package supports x86 CUDA only, no ROCm equivalent | Find TheRock alternative (torch) or community ROCm wheel |
| `numpy` cannot install 2.x | deepfilternet pins `numpy<2` | Accept 1.26.4 (torch-compatible); see [03 §Package notes](docs/03-speech-to-speech-status.md) for how to force 2.x if needed |
| `flash-attn` cannot install | gfx1151 has no upstream support | **Don't install**, see [03 §2.a](docs/03-speech-to-speech-status.md) |

## Fix patches

### MPS bug

`speech_to_speech/paraformer_handler.py:56` calls `torch.mps.empty_cache()` unconditionally and crashes on ROCm/CUDA:

```bash
sed -i 's/torch.mps.empty_cache()/torch.cuda.empty_cache()/' \
  /home/kamjin/apps/.venv/lib64/python3.12/site-packages/speech_to_speech/STT/paraformer_handler.py
```

### DeepFilterNet torchaudio incompat

`df/io.py:9` references `torchaudio.backend.common.AudioMetaData`, which TheRock 2.10 has removed:

```bash
DFIO=/home/kamjin/apps/.venv/lib64/python3.12/site-packages/df/io.py
cp $DFIO ${DFIO}.bak

python3 - <<'PY'
p = '/home/kamjin/apps/.venv/lib64/python3.12/site-packages/df/io.py'
s = open(p).read()
old = 'from torchaudio.backend.common import AudioMetaData'
new = '''try:
    from torchaudio.backend.common import AudioMetaData  # type: ignore[attr-defined]
except ImportError:
    AudioMetaData = Any  # type: ignore[assignment,misc]'''
if old in s and 'try:' not in s.split(old)[0][-50:]:
    open(p, 'w').write(s.replace(old, new))
    print("patched")
else:
    print("already patched or pattern not found")
PY
```

Verify:

```bash
python3 -c "from df.enhance import init_df; init_df(); print('OK')"
```

## More help

1. See the full [docs/03 §Known Issues](docs/03-speech-to-speech-status.md)
2. See [docs/04 — Debug Journey](docs/04-reachy-mini-debug-journey.md)
3. Open an issue: https://github.com/kamjin3086/reachy-mini-sts-pipeline/issues
4. Upstream issues:
   - speech-to-speech: https://github.com/facebookresearch/speech-to-speech/issues
   - TheRock: https://github.com/ROCm/TheRock/issues
   - DeepFilterNet: https://github.com/Rikorose/DeepFilterNet/issues
