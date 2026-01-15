import sys
from pathlib import Path
import wave

from tools.jvs_to_ljspeech import iter_jvs_samples, main


def write_wav(path: Path, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * 10)


def test_iter_jvs_samples_basic(tmp_path):
    jvs_root = tmp_path / "jvs"
    speaker = jvs_root / "jvs001"
    sub = speaker / "parallel100"
    wav_dir = sub / "wav24kHz16bit"
    wav_dir.mkdir(parents=True)
    (sub / "transcripts_utf8.txt").write_text("001:こんにちは\n", encoding="utf-8")
    write_wav(wav_dir / "001.wav")

    samples = list(iter_jvs_samples(jvs_root))
    assert len(samples) == 1
    spk, subset, base, text, wav_path = samples[0]
    assert spk == "jvs001"
    assert subset == "parallel100"
    assert base == "001"
    assert text == "こんにちは"
    assert wav_path.exists()


def test_main_max_samples(tmp_path, monkeypatch):
    jvs_root = tmp_path / "jvs"
    speaker = jvs_root / "jvs001"
    sub = speaker / "parallel100"
    wav_dir = sub / "wav24kHz16bit"
    wav_dir.mkdir(parents=True)
    (sub / "transcripts_utf8.txt").write_text("001:こんにちは\n002:世界\n", encoding="utf-8")
    write_wav(wav_dir / "001.wav")
    write_wav(wav_dir / "002.wav")

    output_dir = tmp_path / "out"
    argv = [
        "jvs_to_ljspeech",
        "--jvs-root",
        str(jvs_root),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    metadata = (output_dir / "metadata.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(metadata) == 1
    assert metadata[0] == "jvs001_parallel100_001|こんにちは"
    assert (output_dir / "wavs" / "jvs001_parallel100_001.wav").exists()
