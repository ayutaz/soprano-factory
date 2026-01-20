"""
Phoneme conversion utilities for Japanese text using pyopenjtalk.

Converts Japanese text to phoneme tokens with accent information,
enabling much more efficient learning compared to character-based tokens.
"""
import re

import pyopenjtalk


def extract_fullcontext(text: str, run_marine: bool = False) -> list[str]:
    """Convert Japanese text to fullcontext labels.

    Args:
        text: Japanese text to convert
        run_marine: Whether to run MARINE accent estimation (requires marine package)

    Returns:
        List of fullcontext label strings
    """
    return pyopenjtalk.extract_fullcontext(text, run_marine=run_marine)


def parse_label(label: str) -> dict:
    """Parse a fullcontext label into components.

    Args:
        label: Fullcontext label string (e.g., "sil^n-i+h=o/A:-3+1+7/...")

    Returns:
        Dictionary with parsed components:
        - phoneme: Current phoneme
        - a1: Accent relative position (distance from accent nucleus)
        - a2: Position from accent phrase start
        - a3: Position from accent phrase end
    """
    parts = label.split("/")
    phoneme_part = parts[0]  # e.g., "sil^n-i+h=o"

    # Extract current phoneme (between - and +)
    match = re.search(r"-(.+?)\+", phoneme_part)
    current_phoneme = match.group(1) if match else phoneme_part

    # Parse A: field (accent information)
    a_field = next((p for p in parts if p.startswith("A:")), None)
    a1, a2, a3 = None, None, None
    if a_field:
        a_match = re.match(r"A:(-?\d+|xx)\+(\d+|xx)\+(\d+|xx)", a_field)
        if a_match:
            a1 = a_match.group(1)  # Accent relative position
            a2 = a_match.group(2)  # Forward position in accent phrase
            a3 = a_match.group(3)  # Backward position in accent phrase

    return {
        "phoneme": current_phoneme,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


def text_to_phoneme_tokens(text: str, include_accent: bool = True) -> list[str]:
    """Convert Japanese text to phoneme tokens with optional accent info.

    Args:
        text: Japanese text to convert
        include_accent: Whether to include accent position in tokens

    Returns:
        List of phoneme tokens (e.g., ["k", "o_-3", "N", "n_-2", ...])
    """
    labels = extract_fullcontext(text)
    tokens = []

    for label in labels:
        parsed = parse_label(label)
        ph = parsed["phoneme"]

        # Skip silence markers at boundaries
        if ph in ("sil", "pau"):
            continue

        if include_accent:
            a1 = parsed["a1"]
            # Combine phoneme with accent position
            if a1 and a1 != "xx":
                token = f"{ph}_{a1}"
            else:
                token = ph
        else:
            token = ph

        tokens.append(token)

    return tokens


def text_to_phoneme_string(text: str, include_accent: bool = True) -> str:
    """Convert Japanese text to space-separated phoneme string.

    Args:
        text: Japanese text to convert
        include_accent: Whether to include accent position in tokens

    Returns:
        Space-separated phoneme string for tokenizer
    """
    tokens = text_to_phoneme_tokens(text, include_accent=include_accent)
    return " ".join(tokens)


def get_phoneme_vocabulary() -> list[str]:
    """Get the complete phoneme vocabulary with accent variations.

    Returns:
        List of all phoneme tokens including accent variations (~1000 tokens)
    """
    # Base phonemes from Japanese phonology
    base_phonemes = [
        # Vowels
        "a", "i", "u", "e", "o",
        # Basic consonants
        "k", "s", "t", "n", "h", "m", "y", "r", "w",
        # Voiced consonants
        "g", "z", "d", "b", "p",
        # Palatalized consonants (拗音)
        "ky", "sy", "ty", "ny", "hy", "my", "ry",
        "gy", "zy", "dy", "by", "py",
        # Special consonants
        "ts", "ch", "sh", "j", "f", "v",
        # Special morae
        "N",   # 撥音 (ん)
        "cl",  # 促音 (っ)
        # Long vowels (may appear in some contexts)
        "A", "I", "U", "E", "O",
    ]

    # Accent positions typically range from -10 to +10
    # Negative = before accent nucleus, Positive = after, 0 = nucleus
    accent_positions = [str(i) for i in range(-10, 11)]

    # Build vocabulary
    vocab = list(base_phonemes)  # Plain phonemes (no accent info)

    # Add accent variations for each phoneme
    for ph in base_phonemes:
        for pos in accent_positions:
            vocab.append(f"{ph}_{pos}")

    return vocab


def get_phoneme_token_count() -> int:
    """Get the total number of phoneme tokens.

    Returns:
        Number of tokens in the phoneme vocabulary
    """
    return len(get_phoneme_vocabulary())
