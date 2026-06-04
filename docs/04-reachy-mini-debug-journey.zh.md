# 调试记录："连接成功，无响应"症状（已解决）

> [English](04-reachy-mini-debug-journey.md) · [← 返回 README](../README.zh.md)
>
> 这是 2026-06-03 首次测试时遇到的"什么都不发生"症状的诊断记录归档。真正的根因和修复方案见本页末尾的[尾声](#尾声问题最终如何解决)。

## 症状

`speech-to-speech` 服务正常启动，Reachy Mini 对话 app 也连上了，但**用户说话时什么也不发生** —— 没有日志、没有机器人响应、没有任何声音输出。WebSocket 建立后整个管道就静默挂住。

## 当时的配置

- **机器人**：Reachy Mini Lite（2026-06-01 组装，约 3 小时）
- **算力**：Strix Halo 128G（AMD GPU，Linux）
- **控制端**：macOS（Reachy Mini Control 桌面 app）
- **App**：通过 Reachy Mini Control 装的 `reachy_mini_conversation_app`

参考的起步资料：

- [Local Reachy Mini Conversation（Hugging Face 博客）](https://huggingface.co/blog/local-reachy-mini-conversation)
- [Reachy Mini Goes Fully Local（r/LocalLLaMA）](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/)

## 管道配置

第一次 CPU 跑：STT 用 Parakeet TDT，TTS 用 Kokoro（内置的 `qwen3-tts` 只支持 CUDA，是运行时才发现的）：

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

## 服务启动日志

服务启动时只有 warning，没有 error：

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

在 Reachy Mini 对话 app 里填好 IP 和端口后，日志显示有新客户端连上，但说话时什么活动都没有 —— VAD 不触发、STT 不打印、完全没有动静。

## 已经从日志确认的事实

- 服务启动正常，所有模型加载到 CPU
- LLM 后端（8101 端口的 Responses API）warmup 期间响应正常
- App 端的 WebSocket 连接建立成功
- 握手本身看起来正常

## 当时无法从服务端日志判断的疑点

- 机器人的麦克风权限和音频输入路由
- App 是不是真的把音频数据通过 WebSocket 发过来了
- Reachy Mini Control 桌面 app 的连接错误日志

## 尾声：问题最终如何解决

"什么都不发生"这个症状在 GPU 加速阶段仍然存在。搬到 Strix Halo 128G 工作站后，换成 Paraformer-zh + Qwen3-TTS（代替 Parakeet + Kokoro），并修改了官方 Reachy Mini 对话 app，同样的 `ws://0.0.0.0:8765/v1/realtime` 端点才开始正常工作。

根因不是单一 bug，而是多重问题叠加：

1. **VAD 始终未触发**：macOS 控制端的音频路径没有把麦克风数据传到 WebSocket（通过修改对话 app 修复）
2. **Kokoro TTS 是 CPU 推理**，加上 VAD 路径断裂，整个系统感觉完全无响应
3. **Qwen3-TTS 在 AMD GPU 上不直接可用**——原本以为"开箱即用"是错的，ROCm 上需要 TheRock gfx1151 PyTorch wheel + ROCm 7.13 的 gfx1151 修复（见 [01 — ROCm gfx1151 PyTorch 安装](01-rocm-gfx1151-pytorch-install.md)）

Fork 并修改的 Reachy Mini 对话 app（含 VAD/音频路径修复）位于 **[kamjin3086/reachy_mini_conversation_app](https://github.com/kamjin3086/reachy_mini_conversation_app)**，定位这些问题的调试工作流打包成了可复用的 agent skill：**[kamjin3086/reachymini-debug-skill](https://github.com/kamjin3086/reachymini-debug-skill)**。

## 迭代修改 fork 的 app（可编辑模式安装）

调试 fork 阶段需要频繁改动 app 源码（VAD 阈值、prompt 文案、错误信息等）时，用 pip 的 `-e`（可编辑 / 开发模式）安装 —— 编辑器里保存就立即生效，**大多数代码改动不需要重装、也不需要重启 daemon**：

```bash
# 把 fork 克隆到本地
git clone https://github.com/kamjin3086/reachy_mini_conversation_app.git
cd reachy_mini_conversation_app

# 用可编辑模式装到 daemon 用的同一个 Python 环境
pip install -e .

# 现在你在这个目录下任何保存的修改，
# 都会在下次调用时被正在运行的 Reachy Mini Control app 实时加载
```

为什么调试时这个细节很关键：

- **省掉重装循环** —— 每次改动省几分钟。Reachy Mini Control 从自己的 store 路径装 app；普通 `pip install` 同名包会在下次 import 时覆盖那条路径。
- **必须同一个 Python 环境** —— 确认 `pip` 指向 daemon 启动用的解释器（用 `which python3` 比对 Reachy Mini Control 设置里的 "App Python path"）。
- **某些改动仍需重启进程** —— 顶层 import、daemon 钩子、启动期加载的内容改完还是得重启一次。规划好"一波改动"再来一次重启。

不用 `-e` 的话你会发现自己每保存一次就重装一次 —— 这是调试最慢的方式。
