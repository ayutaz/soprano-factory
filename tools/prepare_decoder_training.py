"""
Prepare file lists for decoder training.
Creates train.txt and val.txt with audio file paths.
"""

import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description="Prepare file lists for decoder training")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to LJSpeech-style dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for file lists")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    wavs_dir = os.path.join(args.input_dir, "wavs")

    if not os.path.exists(wavs_dir):
        raise ValueError(f"wavs directory not found: {wavs_dir}")

    # Collect all wav files
    wav_files = []
    for f in os.listdir(wavs_dir):
        if f.endswith(".wav"):
            wav_files.append(os.path.abspath(os.path.join(wavs_dir, f)))

    print(f"Found {len(wav_files)} wav files")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(wav_files)

    val_size = int(len(wav_files) * args.val_ratio)
    val_files = wav_files[:val_size]
    train_files = wav_files[val_size:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Write file lists
    train_path = os.path.join(output_dir, "train_decoder.txt")
    val_path = os.path.join(output_dir, "val_decoder.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_files))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_files))

    print(f"Saved train list: {train_path}")
    print(f"Saved val list: {val_path}")


if __name__ == "__main__":
    main()
