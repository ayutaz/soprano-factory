import torch

from generate_dataset import ensure_float_audio, parse_metadata_lines


def test_parse_metadata_lines_skips_empty():
    lines = ["a|hello", "", "  ", "b|world"]
    files = parse_metadata_lines(lines)
    assert files == [("a", "hello"), ("b", "world")]


def test_parse_metadata_lines_allows_pipe_in_text():
    lines = ["a|hello|world"]
    files = parse_metadata_lines(lines)
    assert files == [("a", "hello|world")]


def test_ensure_float_audio_int16():
    audio = torch.tensor([0, 32767, -32768], dtype=torch.int16)
    out = ensure_float_audio(audio)
    assert out.dtype == torch.float32
    assert torch.isclose(out[1], torch.tensor(32767 / 32768.0))
    assert torch.isclose(out[2], torch.tensor(-1.0))
