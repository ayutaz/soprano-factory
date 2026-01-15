import json
import re
from pathlib import Path
from typing import Iterable

_JA_CHAR_RANGES = "\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F\u30FC\uFF01-\uFF5E\uFF65-\uFF9F"
_JA_RE = re.compile(f"[{_JA_CHAR_RANGES}]")


def collect_japanese_chars(paths: Iterable[Path]) -> list[str]:
    chars = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)
        for text, _ in dataset:
            for ch in text:
                if _JA_RE.match(ch):
                    chars.add(ch)
    return sorted(chars)


def add_japanese_tokens(tokenizer, paths: Iterable[Path]) -> int:
    chars = collect_japanese_chars(paths)
    if not chars:
        return 0
    return tokenizer.add_tokens(chars)
