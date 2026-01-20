"""Tests for phoneme_utils module."""
import pytest

from phoneme_utils import (
    extract_fullcontext,
    get_phoneme_vocabulary,
    get_phoneme_token_count,
    parse_label,
    text_to_phoneme_string,
    text_to_phoneme_tokens,
)


class TestExtractFullcontext:
    """Tests for extract_fullcontext function."""

    def test_basic_text(self):
        """Test basic Japanese text conversion."""
        labels = extract_fullcontext("こんにちは")
        assert isinstance(labels, list)
        assert len(labels) > 0
        # All labels should be strings
        assert all(isinstance(label, str) for label in labels)

    def test_empty_text(self):
        """Test empty text returns empty or minimal labels."""
        labels = extract_fullcontext("")
        assert isinstance(labels, list)


class TestParseLabel:
    """Tests for parse_label function."""

    def test_parse_basic_label(self):
        """Test parsing a basic fullcontext label."""
        # Example label with phoneme 'a' and accent info
        label = "sil^k-a+N=N/A:-2+1+5/B:xx-xx_xx/..."
        result = parse_label(label)

        assert result["phoneme"] == "a"
        assert result["a1"] == "-2"
        assert result["a2"] == "1"
        assert result["a3"] == "5"

    def test_parse_label_with_xx(self):
        """Test parsing label with undefined accent (xx)."""
        label = "xx^sil-sil+k=o/A:xx+xx+xx/..."
        result = parse_label(label)

        assert result["phoneme"] == "sil"
        assert result["a1"] == "xx"

    def test_parse_simple_phoneme(self):
        """Test extracting phoneme from various formats."""
        # Standard format
        result = parse_label("k^o-N+n=i/A:-1+2+3")
        assert result["phoneme"] == "N"


class TestTextToPhonemeTokens:
    """Tests for text_to_phoneme_tokens function."""

    def test_basic_conversion(self):
        """Test basic text to phoneme conversion."""
        tokens = text_to_phoneme_tokens("こんにちは")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should not contain silence markers
        assert "sil" not in tokens
        assert "pau" not in tokens

    def test_with_accent(self):
        """Test conversion with accent information."""
        tokens = text_to_phoneme_tokens("あ", include_accent=True)
        assert isinstance(tokens, list)
        # Some tokens should have accent markers (underscore)
        # Not all will have it, but the function should work

    def test_without_accent(self):
        """Test conversion without accent information."""
        tokens = text_to_phoneme_tokens("あ", include_accent=False)
        assert isinstance(tokens, list)
        # No tokens should have accent markers
        for token in tokens:
            assert "_" not in token or token in ("cl",)  # cl is a special case


class TestTextToPhonemeString:
    """Tests for text_to_phoneme_string function."""

    def test_basic_string_conversion(self):
        """Test conversion to space-separated string."""
        result = text_to_phoneme_string("こんにちは")
        assert isinstance(result, str)
        # Should be space-separated
        assert " " in result

    def test_string_roundtrip(self):
        """Test that string can be split back to tokens."""
        tokens = text_to_phoneme_tokens("あいう")
        string = text_to_phoneme_string("あいう")
        reconstructed = string.split()
        assert tokens == reconstructed


class TestGetPhonemeVocabulary:
    """Tests for get_phoneme_vocabulary function."""

    def test_vocabulary_size(self):
        """Test that vocabulary has reasonable size."""
        vocab = get_phoneme_vocabulary()
        # Should have base phonemes plus accent variations
        # Base ~40 phonemes × 21 accent positions + base = ~900 tokens
        assert len(vocab) > 500
        assert len(vocab) < 2000

    def test_vocabulary_contains_basics(self):
        """Test that vocabulary contains basic phonemes."""
        vocab = get_phoneme_vocabulary()
        # Check for vowels
        assert "a" in vocab
        assert "i" in vocab
        assert "u" in vocab
        assert "e" in vocab
        assert "o" in vocab
        # Check for special morae
        assert "N" in vocab  # ん
        assert "cl" in vocab  # っ

    def test_vocabulary_contains_accent_variants(self):
        """Test that vocabulary contains accent variants."""
        vocab = get_phoneme_vocabulary()
        # Should have accent variants like "a_0", "a_-1", etc.
        assert "a_0" in vocab
        assert "a_-1" in vocab
        assert "a_1" in vocab

    def test_no_duplicates(self):
        """Test that vocabulary has no duplicate entries."""
        vocab = get_phoneme_vocabulary()
        assert len(vocab) == len(set(vocab))


class TestGetPhonemeTokenCount:
    """Tests for get_phoneme_token_count function."""

    def test_count_matches_vocabulary(self):
        """Test that count matches vocabulary length."""
        count = get_phoneme_token_count()
        vocab = get_phoneme_vocabulary()
        assert count == len(vocab)


class TestIntegration:
    """Integration tests for the phoneme pipeline."""

    def test_common_phrases(self):
        """Test conversion of common Japanese phrases."""
        phrases = [
            "おはようございます",
            "ありがとう",
            "すみません",
            "こんにちは",
        ]
        for phrase in phrases:
            tokens = text_to_phoneme_tokens(phrase)
            assert len(tokens) > 0, f"Failed for phrase: {phrase}"

    def test_mixed_content(self):
        """Test handling of mixed content."""
        # pyopenjtalk should handle numbers and some punctuation
        text = "今日は2024年です"
        tokens = text_to_phoneme_tokens(text)
        assert len(tokens) > 0
