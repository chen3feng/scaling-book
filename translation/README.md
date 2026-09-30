# 中文翻译工具

使用本地 [Ollama](https://ollama.com) 的 `qwen3.5:9b` 模型将本书翻译为简体中文，全程离线、免费。

## 准备

```bash
ollama pull qwen3.5:9b     # 拉取模型（约 6.6 GB）
ollama serve               # 启动服务，默认监听 127.0.0.1:11434
curl http://127.0.0.1:11434/api/tags   # 确认可用
```

## 翻译

在仓库根目录执行：

```bash
# 翻译全书 13 个正文章节，输出到 zh/（保留 frontmatter 与目录结构）
python3 translation/translate_zh.py \
  --src index.md roofline.md tpus.md sharding.md transformers.md \
        training.md applied-training.md inference.md applied-inference.md \
        profiling.md jax-stuff.md conclusion.md gpus.md \
  --out zh --workers 3

# 只翻译单章（调试）
python3 translation/translate_zh.py --src roofline.md --out zh

# 强制全部重译（忽略缓存）
python3 translation/translate_zh.py --src roofline.md --out zh --force
```

## 特性

- **非思考模式**：请求带 `think=false`，并剥离可能的 ` thinking` 段落，输出干净的译文；
- **术语一致**：[glossary.txt](glossary.txt) 作为术语表注入每次请求，全书统一译法
  （如 scaling law＝缩放定律、token＝词元、compute＝算力、shard＝分片）。改术语表后新译文自动生效；
- **翻译记忆**：相同英文句段只译一次，缓存于 `translation/.cache/translation_memory.json`，重复句段天然一致；
- **断点续译**：`translation/.cache/state.json` 记录每个源文件哈希，未改动文件自动跳过；
- **格式保护**：围栏代码块、数学公式（`$`/`$$`/`\begin{align}` 等）、行内代码、下标 `W<sub>in</sub>`、
  链接 URL、图片、HTML 标签、Jekyll Liquid 标签（`{% … %}` / `{{ … }}`）全部经占位符保护后原样还原；
- **关键元数据翻译**：frontmatter 的 `title`/`subtitle`、图注 `caption`、`{% details %}` 标题一并翻译；
- **防幻觉校验**：译文标题数/列表项数/行数与原文不符时自动重译，仍不合规则保留英文原文，绝不输出损坏内容。

## 校对建议

机翻完成后建议人工通读，重点检查：① 术语表标注「依语境」的词（如 inference/reasoning）；② 公式前后的句子；
③ 表格管道符与链接是否完好。