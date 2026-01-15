import json

from dataset import AudioDataset


def test_audio_dataset_format(tmp_path):
    path = tmp_path / "train.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([["こんにちは", [1, 2, 3]]], f, ensure_ascii=False)

    dataset = AudioDataset(path)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample == "[STOP][TEXT]こんにちは[START][1][2][3][STOP]"
