import json

from tokenizer_utils import add_japanese_tokens, collect_japanese_chars


class DummyTokenizer:
    def __init__(self, existing=None):
        self.vocab = set(existing or [])

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab.add(token)
                added += 1
        return added


def test_collect_japanese_chars(tmp_path):
    data = [
        ["ひらがなと漢字", [1, 2, 3]],
        ["カタカナ", [4]],
        ["Mix混在", [5]],
        ["ASCII only", [6]],
    ]
    path = tmp_path / "train.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    chars = collect_japanese_chars([path])
    assert set(chars) == set("ひらがなと漢字カタカナ混在")


def test_add_japanese_tokens(tmp_path):
    data = [
        ["日本語", [1]],
        ["English", [2]],
    ]
    path = tmp_path / "train.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    tokenizer = DummyTokenizer(existing=["英"])
    added = add_japanese_tokens(tokenizer, [path])
    assert added == 3
    assert "日" in tokenizer.vocab
    assert "本" in tokenizer.vocab
    assert "語" in tokenizer.vocab
