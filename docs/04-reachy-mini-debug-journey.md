# Debug Record: "Connection OK, No Response" Symptom (Resolved)

> [中文版](04-reachy-mini-debug-journey.zh.md) · [← Back to README](../README.md)
>
> This is the preserved diagnostic record for the original "nothing happens" symptom encountered during initial testing on 2026-06-03. The actual root cause and fix are described in the [Epilogue](#epilogue-how-it-was-actually-resolved) at the bottom of this page.

## Symptom

The `speech-to-speech` server starts successfully and the Reachy Mini conversation app connects, but **nothing happens when the user speaks** — no logs, no robot response, no audio output. The pipeline hangs silently after the WebSocket connection is established.

## Setup at the time

- **Robot**: Reachy Mini Lite (assembled 2026-06-01, ~3 hours)
- **Compute**: Strix Halo 128G (AMD GPU, running Linux)
- **Control**: macOS (Reachy Mini Control desktop app)
- **App**: `reachy_mini_conversation_app` installed via Reachy Mini Control

References followed during initial setup:

- [Local Reachy Mini Conversation (Hugging Face blog)](https://huggingface.co/blog/local-reachy-mini-conversation)
- [Reachy Mini Goes Fully Local (r/LocalLLaMA)](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/)

## Pipeline configuration

The CPU-only first attempt used Parakeet TDT for STT and Kokoro for TTS (the built-in `qwen3-tts` only supports CUDA, which was discovered at runtime):

```bash
speech-to-speech \
  --responses_api_base_url "http://127.0.0.1:8101/v1" \
  --responses_api_api_key "" \
  --mode realtime \
  --model_name Gemma-4-E4B-instruct \
  --llm_backend responses-api \
  --tts kokoro \
  --ws_host 0.0.0.0 \
  --ws_port 8765 \
  --stt parakeet-tdt
```

## Server startup logs

The server starts with warnings but no errors:

```
DeepFilterNet not available for audio enhancement: No module named 'df'
[nltk_data] Downloading package averaged_perceptron_tagger_eng to
[nltk_data]     /home/kamjin/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger_eng is already up-to-
[nltk_data]      date!
Using cache found in /home/kamjin/.cache/torch/hub/snakers4_silero-vad_master
2026-06-03 14:28:10,397 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Loading Parakeet TDT model: nvidia/parakeet-tdt-0.6b-v3 on cpu
2026-06-03 14:28:15,336 - speech_to_speech.STT.parakeet_tdt_handler - INFO - nano-parakeet model loaded successfully on cpu
2026-06-03 14:28:15,336 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Live transcription enabled for Parakeet TDT (nano_parakeet)
2026-06-03 14:28:15,336 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Warming up ParakeetTDTSTTHandler
2026-06-03 14:28:15,732 - speech_to_speech.STT.parakeet_tdt_handler - INFO - Model warmed up and ready
2026-06-03 14:28:15,744 - speech_to_speech.LLM.responses_api_language_model - INFO - Warming up ResponsesApiModelHandler
2026-06-03 14:28:17,994 - httpx - INFO - HTTP Request: POST http://127.0.0.1:8101/v1/responses "HTTP/1.1 200 OK"
2026-06-03 14:28:18,019 - speech_to_speech.LLM.responses_api_language_model - INFO - ResponsesApiModelHandler:  warmed up! time: 2.276 s
2026-06-03 14:28:18,020 - speech_to_speech.TTS.kokoro_handler - INFO - Loading Kokoro model: hexgrad/Kokoro-82M on cpu
WARNING: Defaulting repo_id to hexgrad/Kokoro-82M. Pass repo_id='hexgrad/Kokoro-82M' to suppress this warning.
/home/kamjin/apps/.venv/lib64/python3.14/site-packages/torch/nn/modules/rnn.py:1009: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  super().__init__("LSTM", *args, **kwargs)
/home/kamjin/apps/.venv/lib64/python3.14/site-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
2026-06-03 14:28:20,028 - speech_to_speech.TTS.kokoro_handler - INFO - Native Kokoro pipeline loaded successfully
2026-06-03 14:28:20,028 - speech_to_speech.TTS.kokoro_handler - INFO - Warming up KokoroTTSHandler
2026-06-03 14:28:20,626 - speech_to_speech.TTS.kokoro_handler - INFO - KokoroTTSHandler warmed up
2026-06-03 14:28:20,690 - speech_to_speech.api.openai_realtime.server - INFO - OpenAI Realtime API server starting on ws://0.0.0.0:8765/v1/realtime
INFO:     Started server process [3467244]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8765 (Press CTRL+C to quit)
```

After configuring the IP and port in the Reachy Mini conversation app, the logs show a new client connection, but speaking produces zero activity — no VAD triggers, no STT logs, nothing.

## What was verified from logs

- Server starts cleanly and all models load on CPU
- LLM backend (Responses API on port 8101) responds correctly during warmup
- WebSocket connection from the app is established successfully
- The connection handshake itself looks valid

## Suspected (could not be confirmed from server-side logs alone)

- Microphone permissions and audio input routing on the robot
- Whether the app is actually sending audio data over the WebSocket
- Reachy Mini Control desktop app logs for connection errors

## Epilogue: how it was actually resolved

The "nothing happens" symptom persisted into the GPU-accelerated phase. After moving to the Strix Halo 128G workstation, switching to Paraformer-zh + Qwen3-TTS (instead of Parakeet + Kokoro), and modifying the official Reachy Mini conversation app, the same `ws://0.0.0.0:8765/v1/realtime` endpoint started working.

The root cause was never a single bug — it was a combination of:

1. **VAD never triggered** because the audio path on the macOS control side wasn't piping microphone data to the WebSocket (resolved by modifying the conversation app).
2. **Kokoro TTS was CPU-only**, and combined with the broken VAD path, made the system feel completely unresponsive.
3. **Qwen3-TTS does not work on AMD GPUs** out of the box — the original assumption that it "just works" is wrong on ROCm. It required the TheRock gfx1151 PyTorch wheel plus a fix to ROCm 7.13's gfx1151 issues (covered in [01 — ROCm gfx1151 PyTorch Install](01-rocm-gfx1151-pytorch-install.md)).

The forked and modified Reachy Mini conversation app — including the fix for the VAD/audio path — lives at **[kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app)**, and the debugging workflow that uncovered these issues is packaged as a reusable agent skill at **[kamjin3086/reachymini-debug-skill](https://github.com/kamjin3086/reachymini-debug-skill)**.

## Iterating on the Forked App (Editable Install)

When you're debugging the fork and need to make frequent small changes to the app source (VAD thresholds, prompt wording, error messages, etc.), use pip's `-e` (editable / development) install mode so your editor changes take effect immediately — no reinstall, no daemon restart needed for most code changes:

```bash
# Clone your fork locally
git clone https://github.com/kamjin3086/reachy_mini_conversation_app.git
cd reachy_mini_conversation_app

# Install in editable mode (-e) into the SAME Python env the daemon uses
pip install -e .

# Now any edit you save in this directory is picked up live
# by the running Reachy Mini Control app on next call/invocation
```

Why this matters during debug:

- **No reinstall loop** — saves minutes per change. Reachy Mini Control installs apps from its own store path; regular `pip install` of the same package will shadow that path on next import.
- **Same Python env** — make sure `pip` here points to the same interpreter the daemon launches (check with `which python3` and compare to Reachy Mini Control's "App Python path" in settings).
- **Process restart may still be needed** for changes to top-level imports, daemon hooks, or anything loaded at startup. Plan for ~one restart per "session" of edits.

If you don't use `-e` you'll find yourself re-installing after every save, which is the slowest way to debug.
