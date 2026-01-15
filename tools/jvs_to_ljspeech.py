import argparse
import os
from pathlib import Path
import shutil


def find_wav_dir(sub_dir: Path) -> Path | None:
    candidates = ["wav24kHz16bit", "wav48kHz16bit", "wav", "voice"]
    for name in candidates:
        candidate = sub_dir / name
        if candidate.is_dir() and any(p.suffix.lower() == ".wav" for p in candidate.iterdir()):
            return candidate
    if any(p.suffix.lower() == ".wav" for p in sub_dir.iterdir()):
        return sub_dir
    return None


def iter_jvs_samples(jvs_root: Path):
    for speaker_dir in sorted(jvs_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for sub_dir in sorted(speaker_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            transcript_path = sub_dir / "transcripts_utf8.txt"
            if not transcript_path.exists():
                continue
            wav_dir = find_wav_dir(sub_dir)
            if wav_dir is None:
                continue
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        base, text = line.split(":", 1)
                    except ValueError:
                        continue
                    wav_path = wav_dir / f"{base}.wav"
                    if not wav_path.exists():
                        continue
                    yield speaker_dir.name, sub_dir.name, base, text, wav_path


def main():
    parser = argparse.ArgumentParser(description="Convert JVS to LJSpeech-style dataset.")
    parser.add_argument("--jvs-root", required=True, type=Path, help="Path to JVS root (contains jvs001, jvs002, ...)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for LJSpeech-style dataset")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of samples to copy (0 = all)")
    args = parser.parse_args()

    jvs_root = args.jvs_root
    output_dir = args.output_dir
    output_wavs = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wavs.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.txt"

    copied = 0
    with open(metadata_path, "w", encoding="utf-8") as meta_out:
        for speaker, subset, base, text, wav_path in iter_jvs_samples(jvs_root):
            file_id = f"{speaker}_{subset}_{base}"
            dst = output_wavs / f"{file_id}.wav"
            if not dst.exists():
                shutil.copy2(wav_path, dst)
            meta_out.write(f"{file_id}|{text}\n")
            copied += 1
            if args.max_samples and copied >= args.max_samples:
                break

    print(f"Copied {copied} samples to {output_dir}")


if __name__ == "__main__":
    main()
