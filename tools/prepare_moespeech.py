"""
Prepares moe-speech-20speakers-ljspeech dataset for Soprano training.
Converts 3-column metadata (filename|speaker_id|text) to 2-column LJSpeech format (filename|text).
Extracts wavs.zip and resamples audio from 22050Hz to 32000Hz.

Usage:
uv run python tools/prepare_moespeech.py --output-dir path/to/output
"""
import argparse
import pathlib
import zipfile
import shutil
from tqdm import tqdm
import torchaudio
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache/huggingface/hub/datasets--ayousanz--moe-speech-20speakers-ljspeech/snapshots/6cd2b0902884d7da31e93cd7ee3444baf8fafb64",
        help="Path to HuggingFace cache directory containing the dataset"
    )
    parser.add_argument("--output-dir",
        type=pathlib.Path,
        required=True,
        help="Output directory for processed dataset"
    )
    parser.add_argument("--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    return parser.parse_args()


def main():
    args = get_args()
    cache_dir = args.cache_dir
    output_dir = args.output_dir

    metadata_path = cache_dir / "metadata.csv"
    wavs_zip_path = cache_dir / "wavs.zip"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
    if not wavs_zip_path.exists():
        raise FileNotFoundError(f"wavs.zip not found at {wavs_zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    # Read metadata
    print("Reading metadata...")
    with open(metadata_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse and convert to LJSpeech format (filename|text)
    entries = []
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 3:
            filename = parts[0]
            text = parts[2]
            entries.append((filename, text))

    if args.max_samples:
        entries = entries[:args.max_samples]

    print(f"Found {len(entries)} entries")

    # Extract and resample audio
    print("Extracting and resampling audio (22050Hz -> 32000Hz)...")
    new_metadata = []

    with zipfile.ZipFile(wavs_zip_path, 'r') as zf:
        for filename, text in tqdm(entries):
            wav_name = f"{filename}.wav"
            # The wav might be in wavs/ subdirectory inside the zip
            try:
                # Try wavs/filename.wav first
                zip_path = f"wavs/{wav_name}"
                if zip_path not in zf.namelist():
                    zip_path = wav_name

                # Extract to temp location
                temp_path = output_dir / f"temp_{wav_name}"
                with zf.open(zip_path) as source, open(temp_path, 'wb') as target:
                    target.write(source.read())

                # Load and resample
                waveform, sr = torchaudio.load(str(temp_path))
                if sr != 32000:
                    waveform = torchaudio.functional.resample(waveform, sr, 32000)

                # Save resampled audio
                output_path = wavs_dir / wav_name
                torchaudio.save(str(output_path), waveform, 32000)

                # Clean up temp file
                temp_path.unlink()

                new_metadata.append(f"{filename}|{text}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # Write new metadata
    print("Writing metadata.txt...")
    with open(output_dir / "metadata.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_metadata))

    print(f"Done! Processed {len(new_metadata)} samples to {output_dir}")


if __name__ == '__main__':
    main()
