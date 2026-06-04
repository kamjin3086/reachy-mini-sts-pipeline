# Reachy2 Lite + speech-to-speech Pipeline: Connection OK, No Response — Need Help Debugging

## Background

Hi everyone! I recently assembled a **Reachy2 Lite** from parts (took about 2 hours) and got it connected to my setup. I'm running the full speech-to-speech pipeline on a **Starx Halo 128G** (AMD GPU workstation) connected via my macOS machine.

I followed the excellent guide from the Hugging Face blog ([Local Reachy Mini Conversation](https://huggingface.co/blog/local-reachy-mini-conversation)) and was inspired by this thread: [Reachy Mini Goes Fully Local](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/). Great work by the way — this project has a lot of potential!

## The Problem

The `speech-to-speech` server starts successfully and the Reachy Mini conversation app connects, but **nothing happens when I speak** — no logs, no robot response, no audio output. The pipeline seems to hang silently after the WebSocket connection is established.

## My Setup

- **Robot**: Reachy2 Lite (newly assembled)
- **Compute**: Starx Halo 128G (AMD GPU, running Linux)
- **Control**: macOS (Reachy2 Control desktop app)
- **App**: `reachy_mini_conversation_app` installed via Reachy2 Control

## Pipeline Configuration

Since the built-in `qwen3-tts` only supports CUDA and I discovered this at runtime, I switched to **Kokoro** for TTS. Here's my command:

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

## Server Startup Logs

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

## What I've Checked

- Server starts cleanly and all models load on CPU
- LLM backend (Responses API on port 8101) responds correctly during warmup
- WebSocket connection from the app is established successfully
- Copilot confirmed the connection handshake looks valid

## What I Haven't Been Able to Check

Due to the remote setup, I haven't been able to verify:
- Microphone permissions and audio input routing on the robot
- Whether the app is actually sending audio data over the WebSocket
- Network connectivity between the macOS control machine and the Linux server
- Reachy2 Control desktop app logs for connection errors

I plan to do a thorough in-person debug session tonight with Copilot to inspect all logs (Reachy2 Control desktop, app, server, and robot status).

## Questions

1. Has anyone successfully run this exact pipeline (parakeet-tdt + responses-api LLM + Kokoro TTS) with Reachy2 Lite? Any gotchas?
2. Could the issue be related to running everything on CPU vs GPU? The blog recommends CUDA for Qwen3-TTS, but I'm on AMD.
3. Are there known issues with the Kokoro TTS handler in the realtime WebSocket mode?
4. Any debugging tips for the audio pipeline between the Reachy Mini app and the speech-to-speech server?

Any help or pointers would be greatly appreciated. Happy to share more logs or details once I can access the machine in person!
