# 维护与记录规范

> 返回：[README.zh.md](../README.zh.md) · 脚本索引：[docs/SCRIPTS.zh.md](SCRIPTS.zh.md)

这个仓库的核心价值不是“通用框架”，而是一个已验证的本地 STS 部署记录。后续维护应优先保证：能复现、能排障、能解释为什么这样选。

## 文档分工

| 文件 | 记录什么 | 不记录什么 |
|---|---|---|
| `README.zh.md` / `README.md` | 项目结论、快速开始、最短可用路径 | 详细调参过程 |
| `TROUBLESHOOTING.zh.md` / `TROUBLESHOOTING.md` | 症状到修复的一行速查 | 长篇背景说明 |
| `docs/01-*` | ROCm + PyTorch 基础环境 | STS 运行细节 |
| `docs/02-*` | STS 安装与组件选型 | 实时性能调优 |
| `docs/03-*` | 当前状态、性能、已知问题 | 一次性流水账 |
| `docs/04-*` | Reachy Mini App 调试记录 | 基础安装步骤 |
| `docs/05-*` | Qwen3-TTS 中文效果和 Realtime 实测 | 启动路径说明 |
| `docs/06-*` | 当前支持的启动路径和离线行为 | 历史废弃路径 |
| `docs/SCRIPTS.zh.md` | 所有脚本入口、变量、用途 | 深入实验结论 |

## 记录一次新实验

新增实验记录时，建议按这个结构写，避免以后只剩不可复现的结论：

```markdown
## YYYY-MM-DD 实验名称

- 目标：
- 环境：硬件、OS、Python、torch、ROCm/HIP、关键包版本
- 命令：
- 输入数据：
- 结果：数字优先，例如 TTFT、RTF、CER、延迟
- 结论：
- 后续动作：保留 / 废弃 / 需要重测
```

## 改启动路径的判断标准

只有同时满足下面条件，才建议把实验路径提升为 README 中的主路径：

- 能从干净 venv 或明确的隔离 venv 复现。
- 启动命令不超过一条，必要配置用环境变量覆盖。
- 离线缓存行为清楚，不会无提示联网。
- 性能提升能用数字说明，而不是主观“感觉更快”。
- 故障时能定位到具体组件：STT、LLM、TTS、WebSocket、Reachy App。

## 删除或降级一条路径

如果路径复杂但收益不明确，应从 README 主流程删除，只在实验文档保留记录。删除前确认：

- 是否还有脚本引用该路径。
- 是否还有文档把它描述为推荐路径。
- 是否需要在 `TROUBLESHOOTING.zh.md` 留一个“已废弃，不建议使用”的说明。

## 提交前检查

```bash
make test
python3 -m py_compile scripts/*.py scripts/sts_test/*.py
git status --short
```

如果改了启动脚本，再至少做一次：

```bash
bash -n scripts/sts_start.sh
bash -n scripts/sts_start_qwen3_openai_fastapi_flash.sh
```

## 常见维护原则

- 保留“稳定路径”和“高性能路径”两条主线，避免启动入口膨胀。
- Patch 脚本要可重复运行；重复运行应显示 `already patched` 或等价结果。
- 文档中的绝对路径可以作为本机默认值，但必须说明可用环境变量覆盖。
- 新增性能数字时，同时写清楚冷启动还是稳态。
- 中文文档优先更新；英文 README 至少同步主入口和脚本索引。
- 可复用的模型、源码 checkout、torch.compile 缓存和 benchmark 结果不要默认写 `/tmp`；统一放到 `/home/kamjin/apps/sts-cache` 或通过 `STS_CACHE_DIR` 覆盖。
