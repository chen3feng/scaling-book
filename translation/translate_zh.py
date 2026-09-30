#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Markdown / Jupyter Notebook 技术书籍翻译为简体中文。

- 调用本地 Ollama 的 qwen3.5:9b 模型（关闭思考模式 think=false）
- 保护代码块、数学公式、行内代码、链接 URL、图片、HTML 标签
- 通过术语表 (glossary) + 翻译记忆 (translation memory) 保证全书术语一致
- 断点续译：已翻译且源文件未变化的文件自动跳过
- 仅使用 Python 标准库

用法示例：
    python3 scripts/translate_zh.py --src book --out zh/book
    python3 scripts/translate_zh.py --src book --out zh/book --workers 2
    python3 scripts/translate_zh.py --src book --out zh/book --file book/ch01.md
    python3 scripts/translate_zh.py --src book --out zh/book --force   # 强制重译
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

DEFAULT_MODEL = "qwen3.5:9b"
TEXT_EXTS = {".md", ".markdown", ".ipynb"}
ASSET_COPY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf",
    ".csv", ".json", ".yml", ".yaml", ".toml", ".cfg", ".css",
    ".bib", ".tex",
}
MAX_BLOCK_CHARS = 6000          # 单次送译文本上限
MAX_RETRY = 3
PLACEHOLDER_RE = re.compile(r"ZXPH[0-9A-F]{4,}ZX")

# 打印锁
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Ollama 调用
# ---------------------------------------------------------------------------

def ollama_chat(prompt: str, system: str, model: str, temperature: float = 0.2) -> str:
    """调用 Ollama /api/chat，关闭思考模式。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,          # qwen3 系列：关闭思考模式
        "options": {
            "temperature": temperature,
            "num_predict": 8192,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = (body.get("message") or {}).get("content", "")
    # 兜底：若模型仍输出 <think> 段，剥掉
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    return content.strip()


def check_ollama(model: str) -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"[错误] 无法连接 Ollama ({OLLAMA_HOST})：{exc}\n"
            f"请先启动 Ollama：ollama serve"
        )
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names:
        sys.exit(
            f"[错误] 本地 Ollama 没有模型 {model}。\n"
            f"可用模型：{', '.join(sorted(names))}\n"
            f"请先拉取：ollama pull {model}"
        )


# ---------------------------------------------------------------------------
# 行内元素保护：把不能翻译的部分替换为占位符，翻译后还原
# ---------------------------------------------------------------------------

# 图片必须先于普通链接保护，否则 ![alt](url) 会被链接正则切走感叹号
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

INLINE_PATTERNS = [
    # Liquid 模板标签 {% ... %} / 输出 {{ ... }}（Jekyll 站点结构，绝不翻译）
    re.compile(r"\{%[\s\S]*?%\}", re.S),
    re.compile(r"\{\{[\s\S]*?\}\}", re.S),
    # 数学：$$...$$、\[...\]、\(...\)、$...$
    re.compile(r"\$\$.+?\$\$", re.S),
    re.compile(r"\\\[[\s\S]+?\\\]"),
    re.compile(r"\\\([\s\S]+?\\\)"),
    re.compile(r"\$[^$\n]+?\$"),
    # 行内代码
    re.compile(r"`[^`\n]+`"),
    # 下标/上标（变量记号如 W<sub>in</sub>、D<sup>2</sup>）整体保护，先于 HTML 标签
    re.compile(r"<sub>[^<]*</sub>"),
    re.compile(r"<sup>[^<]*</sup>"),
    # 引用键 [@key] 或 [@key, p.1]
    re.compile(r"\[@[^\]]+\]"),
    # 自动链接
    re.compile(r"<https?://[^>\s]+>"),
    # HTML 标签
    re.compile(r"</?[a-zA-Z][^>]*>"),
]

# 普通链接：保留 URL，翻译链接文字
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


class Protector:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.counter = 0

    def _next_token(self) -> str:
        token = f"ZXPH{self.counter:04X}ZX"
        self.counter += 1
        return token

    def protect(self, text: str) -> str:
        # 0) 图片整体保护（须先于普通链接）
        def _image(m: re.Match) -> str:
            token = self._next_token()
            self.store[token] = m.group(0)
            return token

        text = IMAGE_RE.sub(_image, text)

        # 1) 链接 URL 保护：[text](url) -> [text](TOKEN)
        def _link(m: re.Match) -> str:
            label, url = m.group(1), m.group(2)
            token = self._next_token()
            self.store[token] = url
            return f"[{label}]({token})"

        text = LINK_RE.sub(_link, text)

        # 2) 其他行内元素整体保护
        for pat in INLINE_PATTERNS:
            def _sub(m: re.Match, _pat: re.Pattern = pat) -> str:
                token = self._next_token()
                self.store[token] = m.group(0)
                return token

            text = pat.sub(_sub, text)
        return text

    def restore(self, text: str) -> str:
        # 多轮替换，防止模型对占位符做了少量改动时尽量恢复
        for token, original in self.store.items():
            text = text.replace(token, original)
        # 容错：模型可能在占位符中插入了空格，如 ZXPH 0001 ZX
        def _fuzzy(m: re.Match) -> str:
            compact = re.sub(r"\s+", "", m.group(0))
            return self.store.get(compact, m.group(0))

        text = re.sub(r"Z\s*X\s*P\s*H\s*[0-9A-Fa-f\s]{4,}?\s*Z\s*X", _fuzzy, text)
        return text


# ---------------------------------------------------------------------------
# Markdown 分块
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
MATH_ENV_BEGIN_RE = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|eqnarray\*?|multline\*?)\}")
MATH_ENV_END_RE = re.compile(r"\\end\{(equation\*?|align\*?|gather\*?|eqnarray\*?|multline\*?)\}")
HEADING_RE = re.compile(r"^(#{1,6}\s+)(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
# Liquid 结构中需要翻译的中文内容：
#  figure.liquid 的 caption 属性、{% details ... %} 的标题
FIGURE_CAPTION_RE = re.compile(
    r'(\{%\s*include\s+figure\.liquid\b[^%]*?caption=")'
    r'((?:[^"\\]|\\.)*)'
    r'("[^%]*?%\})'
)
DETAILS_TITLE_RE = re.compile(r"(\{%\s*details\s+)(.*?)(\s*%\})")
# 需要翻译的 frontmatter 字段（标题/副标题）
FRONT_TITLE_RE = re.compile(r'^((?:title|subtitle):\s*")((?:[^"\\]|\\.)*)(")', re.M)
# 模型可能附加的引导前缀，例如「译文：」这类
LEADING_JUNK_RE = re.compile(
    r"^\s*(?:译文|翻译|翻译结果|中文翻译|以下是?翻译|以下为译文|以下是译文|翻译如下)\s*[：:]\s*"
)
FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.S)


def split_blocks(md: str) -> list[dict]:
    """把 Markdown 切分为块。每块: {type, text}。
    type: code / math / raw / text（普通段落、标题、列表等可翻译文本）
    """
    front = ""
    m = FRONTMATTER_RE.match(md)
    if m:
        front = m.group(0)
        md = md[m.end():]

    lines = md.splitlines(keepends=True)
    blocks: list[dict] = []
    if front:
        blocks.append({"type": "raw", "text": front})

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # 围栏代码块
        fm = FENCE_RE.match(line)
        if fm:
            fence = fm.group(2)
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                if lines[i].lstrip().startswith(fence[0] * 3):
                    i += 1
                    break
                i += 1
            blocks.append({"type": "code", "text": "".join(buf)})
            continue

        # 数学环境 \begin{...} ... \end{...}
        if MATH_ENV_BEGIN_RE.search(line):
            buf = [line]
            i += 1
            while i < n and not MATH_ENV_END_RE.search(lines[i - 1]):
                buf.append(lines[i])
                i += 1
                if i > 0 and MATH_ENV_END_RE.search(buf[-1]):
                    break
            blocks.append({"type": "math", "text": "".join(buf)})
            continue

        # $$ 块（跨行）
        if line.strip().startswith("$$"):
            buf = [line]
            if line.strip().count("$$") < 2:
                i += 1
                while i < n:
                    buf.append(lines[i])
                    if "$$" in lines[i]:
                        i += 1
                        break
                    i += 1
            else:
                i += 1
            blocks.append({"type": "math", "text": "".join(buf)})
            continue

        # 空行
        if not line.strip():
            blocks.append({"type": "raw", "text": line})
            i += 1
            continue

        # 连续普通文本行聚合成一个段落块
        buf = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            if (
                not nxt.strip()
                or FENCE_RE.match(nxt)
                or MATH_ENV_BEGIN_RE.search(nxt)
                or nxt.strip().startswith("$$")
            ):
                break
            buf.append(nxt)
            i += 1
        blocks.append({"type": "text", "text": "".join(buf)})

    return blocks


def split_long(text: str, limit: int = MAX_BLOCK_CHARS) -> list[str]:
    """过长文本按行边界切分，保留列表等结构；单行仍过长时再按句子切。"""
    if len(text) <= limit:
        return [text]

    def split_paragraph(p: str) -> list[str]:
        if len(p) <= limit:
            return [p]
        sentences = re.split(r"(?<=[.!?。！？])\s+", p)
        out, cur = [], ""
        for s in sentences:
            if cur and len(cur) + 1 + len(s) > limit:
                out.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}" if cur else s
        if cur:
            out.append(cur)
        return out

    chunks, cur = [], ""
    for ln in text.splitlines(keepends=True):
        if len(ln) > limit:
            # 罕见：单行长段落
            if cur:
                chunks.append(cur)
                cur = ""
            tail = ln.endswith("\n")
            body = ln[:-1] if tail else ln
            pieces = split_paragraph(body)
            if tail:
                pieces = [p + "\n" for p in pieces]
            chunks.extend(pieces)
        elif cur and len(cur) + len(ln) > limit:
            chunks.append(cur)
            cur = ln
        else:
            cur += ln
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# 表格处理
# ---------------------------------------------------------------------------

def translate_table_row(row: str, translate_fn) -> str:
    cells = row.strip().strip("|").split("|")
    lead = "|" if row.lstrip().startswith("|") else ""
    trail = "|" if row.rstrip().endswith("|") else ""
    out = []
    for cell in cells:
        out.append(translate_fn(cell.strip()))
    return f"{lead}{'|'.join(out)}{trail}\n"


# ---------------------------------------------------------------------------
# 翻译流程
# ---------------------------------------------------------------------------

class Translator:
    def __init__(self, model: str, glossary: str, tm_path: Path):
        self.model = model
        self.system_prompt = self._build_system_prompt(glossary)
        self.tm_path = tm_path
        self.tm_lock = threading.Lock()
        self.tm: dict[str, str] = {}
        if tm_path.exists():
            try:
                self.tm = json.loads(tm_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log("[警告] 翻译记忆文件损坏，将重建")
                self.tm = {}

    @staticmethod
    def _build_system_prompt(glossary: str) -> str:
        return (
            "你是一名资深技术文档译者，负责将英文机器学习/AI 技术书籍翻译为简体中文。\n"
            "任务规则：\n"
            "1. 逐句翻译给定文本，输出规范、流畅、专业的简体中文技术文档语言，避免翻译腔；\n"
            "2. 严禁扩写、改写、补充内容、添加标题或章节、重复原文；输入有几句话就输出几句话；\n"
            "3. 严格保留所有 Markdown 格式（标题符号、列表符号、加粗、表格管道符 | 等），"
            "标题的 # 符号不会出现在待译正文中，你也绝不能自行添加 #；\n"
            "4. 形如 ZXPH0001ZX 的占位符必须原样保留，位置不变，不得翻译、删除或改写；\n"
            "5. 代码、数学公式、变量名、命令、URL 一律不翻译；\n"
            "6. 严格遵循下面的术语表，全书术语保持一致；术语表未覆盖的术语首次出现时可采用"
            "“中文（English）”形式；\n"
            "7. 只输出译文本身，不要输出任何解释、说明、前后缀，不要复述任务。\n\n"
            "示例：\n"
            "输入：The training loss follows a power law. See [the doc](ZXPH0000ZX) for details.\n"
            "输出：训练损失遵循幂律。详情参见[文档](ZXPH0000ZX)。\n\n"
            "输入：- The model is undertrained on this corpus.\n"
            "输出：- 模型在该语料库上训练不足。\n\n"
            f"术语表：\n{glossary}\n"
        )

    def _save_tm(self) -> None:
        self.tm_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.tm_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.tm, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.tm_path)

    @staticmethod
    def _structure_ok(src: str, out: str) -> bool:
        """粗校验：仅拦截明显扩写（新增小节标题、列表项明显增多、行数膨胀数倍），
        对正常换行/微调格式宽容，避免把合格译文误判后回退英文。"""
        def feats(s: str) -> tuple[int, int, int]:
            headings = len(re.findall(r"(?m)^#{1,6}\s", s))
            items = len(re.findall(r"(?m)^\s*(?:[-*+]|\d+\.)\s+", s))
            lines = len([ln for ln in s.splitlines() if ln.strip()])
            return headings, items, lines

        hs, items, lines = feats(src)
        ho, itemo, lo = feats(out)
        if ho > hs + 1:                      # 新增了多个小节标题 → 扩写
            return False
        if itemo > max(items, 0) + 2:        # 列表项明显增多 → 扩写
            return False
        if lines and lo > lines * 3 + 3:     # 行数膨胀到 3 倍以上 → 扩写
            return False
        return True

    def translate_text(self, src: str) -> str:
        leading_nl = len(src) - len(src.lstrip("\n"))
        trailing_nl = len(src) - len(src.rstrip("\n"))
        body = src.strip("\n")
        if not body.strip():
            return src
        key = hashlib.sha1(body.encode("utf-8")).hexdigest()
        with self.tm_lock:
            cached = self.tm.get(key)
        if cached is not None:
            return "\n" * leading_nl + cached + "\n" * trailing_nl

        protector = Protector()
        protected = protector.protect(body)
        required = set(PLACEHOLDER_RE.findall(protected))

        def _user_prompt(extra: str = "") -> str:
            return (
                "请将 <<<TRANSLATE 与 TRANSLATE>>> 标记之间的英文 Markdown 翻译为简体中文。"
                "逐句翻译，严禁扩写、补充或添加标题，只输出译文。\n"
                f"<<<TRANSLATE\n{protected}\nTRANSLATE>>>\n"
                f"{extra}"
            )

        result: str | None = None
        last_err = None
        for attempt in range(MAX_RETRY):
            extra = ""
            if attempt > 0:
                extra = (
                    "上次输出不合规，请严格重译。注意：\n"
                    "- 不得添加任何 # 标题或额外段落，行数与列表项数必须与原文一致；\n"
                    "- 占位符必须原样保留：" + ", ".join(sorted(required))
                )
            try:
                out = ollama_chat(_user_prompt(extra), self.system_prompt, self.model)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2 * (attempt + 1))
                continue

            # 若模型把分隔标记也输出了，截取标记间内容
            m = re.search(r"<<<TRANSLATE\n?([\s\S]*?)\n?TRANSLATE>>>?", out)
            if m:
                out = m.group(1)

            # 清洗常见前缀（如「译文：」）与可能的包裹代码围栏
            out = out.strip()
            fm = re.match(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", out)
            if fm:
                out = fm.group(1).strip()
            out = LEADING_JUNK_RE.sub("", out)

            missing = required - set(PLACEHOLDER_RE.findall(out))
            if missing:
                continue
            if not self._structure_ok(body, out):
                continue

            result = protector.restore(out)
            break

        if result is None:
            if last_err is not None:
                log(f"[警告] 翻译请求失败（{last_err}），保留英文原文")
            else:
                log("[警告] 多次翻译未通过结构校验（疑似扩写），保留英文原文")
            result = body

        with self.tm_lock:
            self.tm[key] = result
        return "\n" * leading_nl + result + "\n" * trailing_nl

    def translate_liquid_text(self, md: str) -> str:
        """翻译 Liquid 标签中的 caption / details 标题，标签结构本身保持原样。"""

        def _caption(m: re.Match) -> str:
            head, cap, tail = m.group(1), m.group(2), m.group(3)
            return head + self.translate_text(cap).strip() + tail

        def _title(m: re.Match) -> str:
            head, title, tail = m.group(1), m.group(2), m.group(3)
            return head + self.translate_text(title).strip() + tail

        md = FIGURE_CAPTION_RE.sub(_caption, md)
        md = DETAILS_TITLE_RE.sub(_title, md)
        return md

    def translate_markdown(self, md: str) -> str:
        md = self.translate_liquid_text(md)
        blocks = split_blocks(md)
        out_parts: list[str] = []
        for block in blocks:
            btype = block["type"]
            text = block["text"]
            if btype in ("code", "math", "raw"):
                out_parts.append(text)
                continue

            # 表格：逐行处理，分隔线原样保留
            stripped_lines = text.splitlines()
            if any("|" in ln for ln in stripped_lines) and any(
                TABLE_SEP_RE.match(ln) for ln in stripped_lines
            ):
                rendered = []
                for ln in stripped_lines:
                    if TABLE_SEP_RE.match(ln) or not ln.strip():
                        rendered.append(ln)
                    else:
                        rendered.append(
                            translate_table_row(ln, lambda c: self.translate_text(c + "\n").rstrip("\n"))
                        )
                out_parts.append("".join(rendered))
                continue

            # 标题：仅翻译标题文字
            hm = HEADING_RE.match(text)
            if hm and "\n" not in text.rstrip("\n"):
                title = self.translate_text(hm.group(2)).strip()
                # 容错：去掉模型误加的 # 或多行内容
                title = title.lstrip("#").strip().splitlines()[0]
                out_parts.append(hm.group(1) + title + "\n")
                continue

            # 普通文本块（段落 / 列表）：过长则切分
            chunks = split_long(text)
            translated = "".join(self.translate_text(c) for c in chunks)
            # 保留结尾换行情况
            if text.endswith("\n") and not translated.endswith("\n"):
                translated += "\n"
            out_parts.append(translated)

        result = "".join(out_parts)
        result = self.translate_frontmatter(result)
        self._save_tm()
        return result

    def translate_frontmatter(self, md: str) -> str:
        """翻译 frontmatter 中的 title / subtitle 字段值。"""

        def _repl(m: re.Match) -> str:
            head, val, tail = m.group(1), m.group(2), m.group(3)
            return head + self.translate_text(val).strip() + tail

        return FRONT_TITLE_RE.sub(_repl, md)

    def translate_notebook(self, nb_text: str) -> str:
        nb = json.loads(nb_text)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", []))
            translated = self.translate_markdown(source)
            cell["source"] = [translated]
        self._save_tm()
        return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


# ---------------------------------------------------------------------------
# 文件遍历 / 主流程
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_source_files(src_roots: list[Path]):
    seen = set()
    for root in src_roots:
        if root.is_file():
            if root not in seen:
                seen.add(root)
                yield root
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            if "__pycache__" in path.parts:
                continue
            if path not in seen:
                seen.add(path)
                yield path


def mirror_out_path(path: Path, src_roots: list[Path], out_root: Path) -> Path:
    for root in src_roots:
        try:
            rel = path.relative_to(root)
            base = root if root.is_dir() else root.parent
            rel = path.relative_to(base)
            return out_root / rel
        except ValueError:
            continue
    return out_root / path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="使用本地 Ollama 将技术书籍翻译为简体中文")
    parser.add_argument("--src", nargs="+", required=True, type=Path,
                        help="源目录（可多个），如 book；也可传单个文件")
    parser.add_argument("--out", required=True, type=Path,
                        help="译文输出目录，如 zh/book（镜像源目录结构）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama 模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--glossary", type=Path, default=None,
                        help="术语表文件（默认脚本同目录 glossary.txt）")
    parser.add_argument("--workers", type=int, default=2,
                        help="并发线程数（Ollama 本地推理，默认 2）")
    parser.add_argument("--file", type=Path, default=None,
                        help="只翻译指定的单个文件（路径相对 --src 根）")
    parser.add_argument("--force", action="store_true",
                        help="忽略缓存状态，强制重新翻译")
    parser.add_argument("--copy-assets", action="store_true", default=True,
                        help="复制图片等非文本资源到输出目录（默认开启）")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    glossary_path = args.glossary or script_dir / "glossary.txt"
    if not glossary_path.exists():
        sys.exit(f"[错误] 找不到术语表：{glossary_path}")
    glossary = glossary_path.read_text(encoding="utf-8").strip()

    cache_dir = script_dir / ".cache"
    state_path = cache_dir / "state.json"
    tm_path = cache_dir / "translation_memory.json"

    check_ollama(args.model)

    state: dict[str, dict] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}

    translator = Translator(args.model, glossary, tm_path)

    src_roots = args.src
    files = list(iter_source_files(src_roots))
    if args.file:
        files = [p for p in files if p == args.file or str(p).endswith(str(args.file))]
        if not files:
            sys.exit(f"[错误] 找不到文件：{args.file}")

    pending: list[Path] = []
    for path in files:
        rel_key = str(path)
        out_path = mirror_out_path(path, src_roots, args.out)
        if path.suffix.lower() in TEXT_EXTS:
            digest = sha256_file(path)
            st = state.get(rel_key)
            if (
                not args.force
                and st
                and st.get("sha256") == digest
                and out_path.exists()
            ):
                continue
            pending.append(path)
        elif args.copy_assets and path.suffix.lower() in ASSET_COPY_EXTS:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                shutil.copy2(path, out_path)

    total = len(pending)
    log(f"共 {len(files)} 个文件，待翻译 {total} 个；模型 {args.model}（非思考模式）")
    if total == 0:
        log("全部已是最新，无需翻译。")
        return

    done_count = 0
    fail_count = 0

    def process(path: Path) -> tuple[Path, bool]:
        out_path = mirror_out_path(path, src_roots, args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".ipynb":
                translated = translator.translate_notebook(content)
            else:
                translated = translator.translate_markdown(content)
            tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_out.write_text(translated, encoding="utf-8")
            tmp_out.replace(out_path)
            state[str(path)] = {"sha256": sha256_file(path)}
            return path, True
        except Exception as exc:  # noqa: BLE001
            log(f"[失败] {path}: {exc}")
            return path, False

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process, p): p for p in pending}
        for fut in as_completed(futures):
            path, ok = fut.result()
            done_count += 1
            if not ok:
                fail_count += 1
            log(f"[{done_count}/{total}] {'✓' if ok else '✗'} {path}")
            if done_count % 5 == 0:
                cache_dir.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
                )

    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log(f"完成：成功 {total - fail_count}，失败 {fail_count}。译文目录：{args.out}")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
