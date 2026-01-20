import json
import re
from pathlib import Path
from typing import Iterable

from phoneme_utils import get_phoneme_vocabulary

_JA_CHAR_RANGES = "\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F\u30FC\uFF01-\uFF5E\uFF65-\uFF9F"
_JA_RE = re.compile(f"[{_JA_CHAR_RANGES}]")


def collect_japanese_chars(paths: Iterable[Path]) -> list[str]:
    """Collect unique Japanese characters from dataset files.

    Args:
        paths: Paths to JSON dataset files

    Returns:
        Sorted list of unique Japanese characters found
    """
    chars = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            dataset = json.load(f)
        for text, _ in dataset:
            for ch in text:
                if _JA_RE.match(ch):
                    chars.add(ch)
    return sorted(chars)


def get_japanese_phoneme_tokens() -> list[str]:
    """Get phoneme tokens with accent info for Japanese.

    Returns:
        List of phoneme tokens (~1000 tokens) instead of characters (~2676)
    """
    return get_phoneme_vocabulary()


def add_japanese_tokens(tokenizer, paths: Iterable[Path], use_phonemes: bool = False) -> int:
    """Add Japanese tokens to tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer to extend
        paths: Paths to JSON dataset files (used for character-based mode)
        use_phonemes: If True, use phoneme tokens; if False, use character tokens

    Returns:
        Number of tokens added
    """
    if use_phonemes:
        tokens = get_japanese_phoneme_tokens()
    else:
        tokens = collect_japanese_chars(paths)

    if not tokens:
        return 0
    return tokenizer.add_tokens(tokens)
