# 故障排查速查

> [English](TROUBLESHOOTING.md) · [← 返回 README](README.zh.md)
>
> 详细的已知问题列表见 [docs/03 已知问题与 workaround](docs/03-speech-to-speech-status.md)。本文件是**应急速查表**。

## 启动失败

| 症状 | 原因 | 一行解决 |
|---|---|---|
| `hipErrorInvalidImage` | `HSA_OVERRIDE_GFX_VERSION=11.0.0` 在 ROCm 7.13 触发 | 从 `sts_start.sh` 移除该变量 |
| `HF_HUB_ENABLE_HF_TRANSFER` 报错 | hf_transfer 未装 | `uv pip install hf_transfer` |
| `ModuleNotFoundError: No module named 'df'` | DeepFilterNet 装完但没 patch | 见下面 [DeepFilterNet patch](#deepfilternet-torchaudio-不兼容) |
| `Model 'X' not found`（llama-swap 端） | 模型没注册到 llama-swap | 改 `sts_start.sh` 的 `--model_name` 为 llama-swap 已注册的模型，或在 `config.yaml` 加 |
| `unable to start process: upstream command exited prematurely but successfully` | lemonade 后端问题（模型命令无效、OOM 等） | 看 lemonade 自己的日志；直接 `lemonade serve` 调模型命令确认能干净加载 |
| `pip` 指向 Python 3.14 而非 venv | `~/.local/bin/pip` 在 PATH 抢先 | `export VIRTUAL_ENV=/home/kamjin/apps/.venv` 后用 `uv pip` |

## Reachy Mini 连接

| 症状 | 原因 | 排查方向 |
|---|---|---|
| App 连接后"什么都不发生" | 见 [docs/04](docs/04-reachy-mini-debug-journey.md) | 检查 server log 的 WebSocket 握手、麦克风权限、STT 模型是否实际加载 |
| VAD 不触发 | 麦克风权限未开 / 音量太小 | 在 Reachy Mini Control App 检查音频输入 |
| App 显示连接成功但无响应 | WebSocket 协议不匹配 | 确认 `ws://<server-ip>:8765/v1/realtime` 路径正确 |

## 性能差

| 症状 | 原因 | 解决 |
|---|---|---|
| E2E 感知延迟 4s+ | LLM 选了 NPU 模型或 Step-3.5-Flash | 改回 `Gemma-4-E4B-instruct`（稳态 TTFT 50ms） |
| TTS 首次 12s+ | CUDA graph 编译 | 不可消除，但可在 `sts_start.sh` 末尾加预热请求 |
| STT 报 MPS 错误 | 库内 bug：`paraformer_handler.py:56` | 见下面 [MPS bug patch](#mps-bug) |
| LLM 401 invalid_api_key | `OPENAI_API_KEY` 客户端环境干扰 | `unset OPENAI_API_KEY` 后重启 |

## 包安装失败

| 症状 | 原因 | 解决 |
|---|---|---|
| `uv pip install` 报 "no wheel for your platform" | 包只支持 x86 CUDA，ROCm 无对应 | 找 TheRock 替代（torch）或 TheRock community wheels |
| `numpy` 装不上 2.x | deepfilternet 强制 pin `numpy<2` | 接受 1.26.4 即可（torch 兼容），需要 numpy 2 见 [docs/03 包依赖注记](docs/03-speech-to-speech-status.md) |
| `flash-attn` 装不上 | gfx1151 上游无支持 | **不装**，见 [docs/03 §2.a](docs/03-speech-to-speech-status.md) |

## 修复 patch

### MPS bug

`speech_to_speech/paraformer_handler.py:56` 无条件调用 `torch.mps.empty_cache()`，在 ROCm/CUDA 崩溃：

```bash
sed -i 's/torch.mps.empty_cache()/torch.cuda.empty_cache()/' \
  /home/kamjin/apps/.venv/lib64/python3.12/site-packages/speech_to_speech/STT/paraformer_handler.py
```

### DeepFilterNet torchaudio 不兼容

`df/io.py:9` 引用了 `torchaudio.backend.common.AudioMetaData`，TheRock 2.10 移除了该子包：

```bash
# 备份并 patch
DFIO=/home/kamjin/apps/.venv/lib64/python3.12/site-packages/df/io.py
cp $DFIO ${DFIO}.bak

python3 - <<'PY'
import re
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

验证：

```bash
python3 -c "from df.enhance import init_df; init_df(); print('OK')"
```

## 获取更多帮助

1. 看完整 [docs/03 已知问题](docs/03-speech-to-speech-status.md)
2. 看 [docs/04 调试之旅](docs/04-reachy-mini-debug-journey.md)
3. 提交 issue：https://github.com/kamjin3086/reachy-mini-sts-pipeline/issues
4. 上游问题：
   - speech-to-speech: https://github.com/facebookresearch/speech-to-speech/issues
   - TheRock: https://github.com/ROCm/TheRock/issues
   - DeepFilterNet: https://github.com/Rikorose/DeepFilterNet/issues
