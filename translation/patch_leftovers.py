#!/usr/bin/env python3
"""补译《scaling-book》中文译本的零星英文残留。

用法：在仓库根目录执行  python3 translation/patch_leftovers.py

自动：
  1. 复用 translate_zh.py 的 Translator（术语表 + 翻译记忆 + 防幻觉校验）；
  2. 补译正文中漏译的英文段落/列表项（tpus / transformers / applied-training / profiling）；
  3. 补译 frontmatter 的 description（13 章）；
  4. 修正两处标题：gpus 的幻觉标题、Quiz 5 术语未对齐。

幂等：已译该处的英文原文段已经不存在时，对应步骤自动跳过。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZH = ROOT / "zh"
sys.path.insert(0, str(ROOT / "translation"))

import translate_zh as T  # noqa: E402

MODEL = "qwen3:14b"
GLOSSARY = (ROOT / "translation" / "glossary.txt").read_text(encoding="utf-8")
TM_PATH = ROOT / "translation" / ".cache" / "translation_memory.json"


def translate_lines(fname: str, marker: str, n_lines: int) -> str | None:
    """在 zh/<fname> 中定位含 marker 的行并连同后续 n_lines-1 行一并翻译替换。"""
    path = ZH / fname
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines(keepends=True)
    idx = next((i for i, ln in enumerate(lines) if marker in ln), None)
    if idx is None:
        print(f"  · 跳过 {fname}（已无 {marker!r}）")
        return None
    raw = "".join(lines[idx: idx + n_lines])
    body = raw.strip("\n")
    new = tr.translate_text(body).strip("\n")
    lines[idx: idx + n_lines] = [new + "\n"]
    path.write_text("".join(lines), encoding="utf-8")
    print(f"  ✓ {fname}: {body[:40]!r} → {new[:40]!r}")
    return new


def translate_description(fname: str) -> str | None:
    """补译 frontmatter 的 description 字段（含转义引号还原）。"""
    path = ZH / fname
    txt = path.read_text(encoding="utf-8")
    m = re.search(r'(^description:\s*")(.*)("\s*$)', txt, re.M)
    if not m:
        print(f"  · 跳过 {fname}（无 description）")
        return None
    head, raw, tail = m.group(1), m.group(2), m.group(3)
    if re.search(r"[一-鿿]", raw):
        print(f"  · 跳过 {fname}（description 已是中文）")
        return None
    logical = raw.replace('\\"', '"')          # 还原转义引号用于翻译
    translated = tr.translate_text(logical).strip("\n")
    escaped = translated.replace("\\", "\\\\").replace('"', '\\"')  # 重新转义
    txt = txt.replace(head + raw + tail, head + escaped + tail, 1)
    path.write_text(txt, encoding="utf-8")
    print(f"  ✓ {fname} description")
    return escaped


def direct_replace(fname: str, old: str, new: str) -> None:
    path = ZH / fname
    txt = path.read_text(encoding="utf-8")
    if old not in txt:
        print(f"  · 跳过 {fname}（未找到 {old[:30]!r}）")
        return
    txt = txt.replace(old, new, 1)
    path.write_text(txt, encoding="utf-8")
    print(f"  ✓ {fname}: {old[:36]!r} → {new[:36]!r}")


tr = T.Translator(model=MODEL, glossary=GLOSSARY, tm_path=TM_PATH)

print("=== 1) 补译正文残余英文 ===")
translate_lines("tpus.md", "We have to perform $2BDF$", 1)
translate_lines("transformers.md", "The parameter count of the MLP block dominates", 1)
translate_lines("transformers.md", "The total FLOPs budget during training is well approximated", 1)
translate_lines("applied-training.md", "Why wouldn't we do this?", 1)
translate_lines("profiling.md", "Question 2:** [The Transformer Colab", 1)
translate_lines("profiling.md", "What sharding strategy is this?", 4)  # 4 条列表项
translate_lines("profiling.md", "since this problem was written, the XLA compiler", 1)

print("\n=== 2) 修正标题 ===")
direct_replace(
    "gpus.md",
    "## Collectives on GPUs work by using the GPU's communication infrastructure, "
    "which is typically based on the NVLink or PCIe interconnects.",
    "## 集合通信在 GPU 上是如何工作的？",
)
direct_replace("gpus.md", "### Quiz 5: LLM rooflines", "### Quiz 5: LLM 屋顶线")

print("\n=== 3) 补译 frontmatter description ===")
for f in ["index.md", "roofline.md", "tpus.md", "sharding.md", "transformers.md",
          "training.md", "applied-training.md", "inference.md", "applied-inference.md",
          "profiling.md", "jax-stuff.md", "conclusion.md", "gpus.md"]:
    translate_description(f)

tr._save_tm()
print("\n全部补译完成。")