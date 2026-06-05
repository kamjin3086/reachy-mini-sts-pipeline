# 中文 STS Pipeline 与 Qwen3-TTS 实测报告

测试时间：2026-06-05，Asia/Shanghai。  
测试机器：AMD Strix Halo / Radeon 8060S，ROCm 7.13。

## 结论

当前机器可以跑通真实 pipeline：`sts_start.sh` 启动后，Realtime WebSocket、Gemma-4 LLM、Qwen3-TTS 和 function-call 事件链路都能工作。

最关键的问题不是 Gemma-4，而是 TTS 参数和文本边界：

- `qwen3_tts_language=zh` 会让 Qwen3-TTS 后端拒绝生成，表现为 handler 记录错误但没有音频输出。启动脚本已改为 `--qwen3_tts_language chinese`。
- Qwen3-TTS 官方支持的语气控制路径是 `instruct` 参数，不是把“（开心地）”这类文本直接混进待朗读正文。官方 README 的 CustomVoice 示例使用 `language="Chinese"`、`speaker="Vivian"`、`instruct="用特别愤怒的语气说"`；也说明可通过 `get_supported_speakers()` / `get_supported_languages()` 查询支持项。参考：[QwenLM/Qwen3-TTS README](https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md#custom-voice-generate)、[HuggingFace README](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/README.md#custom-voice-generate)。
- 已安装的 Qwen3-TTS handler 已打补丁：开头的 `（...）`、`(...)`、`[...]`、`［...］` 会被抽取到 `instruct`，并从朗读文本中移除。因此推荐让 LLM 输出短句时可带开头语气提示，但最终由 TTS handler 转换，不依赖模型“猜”括号是不是指令。
- Reachy 动作/表情是否触发，主要取决于 Realtime session 的 tool schema、app prompt 和 LLM 工具选择；TTS 只负责把最终 assistant 文本变成音频。Qwen3-TTS 不应该也不能负责执行 Reachy 动作。

## 本次改动

仓库脚本：

- `scripts/sts_start.sh`
  - 固定中文链路：`--language zh`、`--qwen3_tts_language chinese`、`--responses_api_disable_thinking`、`--stream_batch_sentences 1`。
  - 默认关闭 Paraformer live transcription，避免中文 partial 被重复送入 LLM。
  - 默认 speaker 改为 `Serena`，默认 TTS instruct 为自然、亲切、清晰的中文口语。
  - prompt 强化：动作、表情、跳舞必须通过工具 schema 调用，不能朗读工具名、参数或 JSON。
- `scripts/patch_paraformer_live_transcription.py`
  - 幂等修复已安装 `speech_to_speech/STT/paraformer_handler.py`：`mode="progressive"` 输出 `PartialTranscription`，final 输出 `Transcription`。
- `scripts/patch_qwen3_tts_inline_instruct.py`
  - 幂等修复已安装 `speech_to_speech/TTS/qwen3_tts_handler.py`：抽取开头括号语气提示到 `self.instruct`，朗读文本只保留正文。
- `scripts/bench_qwen3_tts_matrix.py`
  - 真实加载 Qwen3-TTS，输出 WAV、CSV、summary 到 `/tmp/sts_tuning/qwen3_direct/`。
- `scripts/test_realtime_text_client.py`
  - 连接真实 Realtime server，发送 text turn，记录 raw event JSONL 与 summary。

已应用到当前 venv：

- Paraformer patch 目标：`/home/kamjin/apps/.venv/lib64/python3.12/site-packages/speech_to_speech/STT/paraformer_handler.py`
- Qwen3-TTS patch 目标：`/home/kamjin/apps/.venv/lib64/python3.12/site-packages/speech_to_speech/TTS/qwen3_tts_handler.py`

## 测试环境

| 项目 | 值 |
|---|---|
| torch | `2.10.0+rocm7.13.0a20260513` |
| HIP | `7.13.26183` |
| GPU | `Radeon 8060S Graphics` |
| `torch.cuda.is_available()` | `True` |
| speech-to-speech | `0.2.9` |
| qwen-tts | `0.1.1` |
| faster-qwen3-tts | `0.2.6` |
| funasr | `1.3.9` |
| numpy | `1.26.4` |
| LLM endpoint | `http://127.0.0.1:8101/v1` |
| LLM model | `Gemma-4-E4B-instruct` |

## Qwen3-TTS 直接测试

命令：

```bash
python3 scripts/bench_qwen3_tts_matrix.py \
  --device cuda \
  --output-dir /tmp/sts_tuning/qwen3_direct \
  --speakers Serena Vivian Aiden
```

输出：

- 详细 CSV：`/tmp/sts_tuning/qwen3_direct/20260605-114715/qwen3_tts_matrix.csv`
- WAV：`/tmp/sts_tuning/qwen3_direct/20260605-114715/*.wav`
- Summary：`/tmp/sts_tuning/qwen3_direct/20260605-114715/summary.json`

15 条样本全部生成成功，`error_count=0`。支持 speaker 列表由 runtime 查询到：`aiden`、`dylan`、`eric`、`ono_anna`、`ryan`、`serena`、`sohee`、`uncle_fu`、`vivian`。

| Speaker | Case | TTFA(s) | Total(s) | Audio(s) | RTF |
|---|---:|---:|---:|---:|---:|
| Serena | plain | 1.216 | 4.793 | 1.664 | 0.347 |
| Serena | inline raw parentheses | 1.212 | 16.264 | 1.824 | 0.112 |
| Serena | explicit instruct happy | 1.219 | 5.691 | 2.016 | 0.354 |
| Serena | inline extracted happy | 1.212 | 2.273 | 1.856 | 0.817 |
| Serena | inline extracted quiet | 1.207 | 7.023 | 2.496 | 0.355 |
| Vivian | plain | 1.210 | 2.367 | 1.888 | 0.798 |
| Vivian | inline raw parentheses | 1.209 | 5.911 | 2.048 | 0.347 |
| Vivian | explicit instruct happy | 1.220 | 24.116 | 2.688 | 0.112 |
| Vivian | inline extracted happy | 1.213 | 6.138 | 2.176 | 0.355 |
| Vivian | inline extracted quiet | 1.212 | 30.186 | 3.456 | 0.115 |
| Aiden | plain | 1.210 | 20.758 | 2.272 | 0.110 |
| Aiden | inline raw parentheses | 1.212 | 2.365 | 1.888 | 0.799 |
| Aiden | explicit instruct happy | 1.213 | 2.273 | 1.792 | 0.788 |
| Aiden | inline extracted happy | 1.206 | 2.266 | 1.792 | 0.791 |
| Aiden | inline extracted quiet | 1.207 | 3.048 | 2.464 | 0.808 |

观察：

- 稳态 TTFA 很稳定，基本在 `1.20s` 左右。
- 总耗时波动较大，尤其是中文 speaker + instruct 组合。`Serena` 更像保守默认；`Aiden` 在部分样本上更快，但官方 native language 是 English，不适合作为中文默认音色。
- 这次矩阵是在 Qwen3-TTS handler patch 已应用后跑的；即使 benchmark case 名为 `inline_raw_parentheses`，真实 handler 仍会在生成前剥离开头括号提示。因此它验证的是“当前 pipeline 不朗读开头提示”的行为，不是未打补丁 Qwen3 原生直接读括号文本的行为。
- 辅助 ASR 检查输出在 `/tmp/sts_tuning/qwen3_direct/20260605-114715/qwen3_tts_asr_check.json`，本次转写文本为空，不能作为内容或语气判断依据。语气质量仍需要人工听 WAV 评分。

语言参数问题：

- 早期快速测试使用 `qwen3_tts_language=zh`，Qwen3-TTS 后端报 `Unsupported languages: ['zh']`，并产生 `no_audio_generated`。
- 当前脚本和文档统一使用 `chinese`。官方示例写法是 `Chinese`；当前安装的 faster-qwen3-tts runtime 接受小写 `chinese`。

## Realtime Pipeline 测试

启动：

```bash
./scripts/sts_start.sh
```

客户端：

```bash
python3 scripts/test_realtime_text_client.py \
  --output-dir /tmp/sts_tuning/realtime \
  --voice Serena \
  --timeout 90
```

输出：

- Raw events：`/tmp/sts_tuning/realtime/20260605-115138/realtime_events.jsonl`
- Summary：`/tmp/sts_tuning/realtime/20260605-115138/realtime_summary.json`

服务端启动结果：

- Paraformer warmup 成功。
- LLM warmup 成功，`POST http://127.0.0.1:8101/v1/responses` 返回 200。
- Qwen3-TTS warmup 成功。
- WebSocket server 运行在 `0.0.0.0:8765`。

三轮测试结果：

| Turn | Prompt | Result | Audio delta | Tool call | Errors |
|---:|---|---|---:|---|---:|
| 1 | `请用开心的语气说：你好，我准备好了。` | transcript: `开心地你好我准备好了` | 15 | none | 0 |
| 2 | `请点点头，然后用一句自然中文回复我。` | no transcript | 0 | `reachy_action {"action":"nod"}` | 0 |
| 3 | `做个开心表情，不要把工具调用内容念出来。` | transcript: `点头好的收到` | 8 | none | 0 |

说明：

- 事件通路正常：第二轮收到了 `response.function_call_arguments.done`，说明服务端能把 Realtime function-call 事件发给 app。
- TTS 没有朗读工具名、参数或 JSON。三轮 `tts_leaked_tool_text=false`，没有工具内容泄漏到 transcript。
- 第一轮 transcript 中出现 `开心地`，但不是括号标记；这来自 LLM 正文输出，不是 Qwen3-TTS 朗读了 `（开心地）`。如果要把“开心地”完全作为语气而非正文，应让 prompt 强制输出 `（开心地）正文`，再由 patched handler 剥离，或在 app 层把语气字段单独传给 TTS。
- 第三轮没有触发 `happy` tool，是当前 prompt/schema 映射不够稳的表现。测试脚本和 `sts_start.sh` 已强化说明；实际 Reachy app 也应该在 tool schema 中明确列出中文动作词到参数的映射。

## 对“括号指令会不会被念出来”的回答

在当前已 patch 的 pipeline 中，开头括号指令不会进入朗读正文：

- `（开心地）你好呀` 会被转换为 `text=你好呀`、`instruct=开心地`。
- `[小声]别担心` 会被转换为 `text=别担心`、`instruct=小声`。

这不是 Qwen3-TTS 原生“解析括号标签”的能力，而是本仓库在 handler 层做的兼容。这样更可控，也更接近 Qwen 官方推荐的 `instruct` 路径。

如果让未 patch 的 TTS 直接收到完整文本，是否会朗读括号内容取决于模型本身和当前 prompt，不能保证稳定。因此生产路径不建议依赖“括号不被读出”这个隐式行为。

## 与 Reachy app prompt 的关系

需要分成两件事看：

- 语气控制：主要属于 TTS handler。LLM 或 app 可以产生 `（开心地）` 这样的语气提示，但必须在 TTS 前剥离并转入 `qwen3_tts_instruct`，否则它只是普通文本。
- 动作/表情：主要属于 Realtime tool schema 和 app prompt。LLM 需要通过 `response.function_call_arguments.done` 输出工具调用，Reachy app 负责执行；TTS 只朗读自然语言回复。

所以，`reachy_mini_conversation_app` 的提示词会影响“是否调用工具、调用哪个工具、是否把动作说成自然语言”，但不会自动让 Qwen3-TTS 理解括号语气。括号语气能稳定工作，是因为当前 venv 的 Qwen3-TTS handler 已做文本抽取。

## 建议默认配置

当前建议继续使用：

```bash
--tts qwen3
--qwen3_tts_language chinese
--qwen3_tts_speaker Serena
--qwen3_tts_instruct "用自然、亲切、清晰的中文口语语气说话。"
--qwen3_tts_streaming_chunk_size 12
--qwen3_tts_blocksize 512
--qwen3_tts_non_streaming_mode
--stream_batch_sentences 1
--responses_api_disable_thinking
--no_enable_live_transcription
```

若要重新打开 Paraformer live transcription，先确认已应用：

```bash
python3 scripts/patch_paraformer_live_transcription.py
```

若要让括号语气稳定走 TTS instruct，先确认已应用：

```bash
python3 scripts/patch_qwen3_tts_inline_instruct.py
```

## 剩余风险

- 语气质量没有完成主观人工听感评分；本次只确认了可生成、时延和事件链路。
- Qwen3-TTS 的总耗时在不同 speaker/instruct 上波动明显，中文实时体验仍可能被 TTS 拖慢。
- Realtime 工具调用链路可用，但 `开心表情` 这类自然语言到 action 参数的映射仍需在实际 app 的 tool schema 和 prompt 中继续收紧。
- 当前 patch 修改的是已安装包，仓库保留的是幂等 patch 脚本，不维护 speech-to-speech fork；升级上游包后需要重新运行 patch。
