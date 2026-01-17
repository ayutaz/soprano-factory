"""Test script for trained decoder v2 (mel to audio)."""
import argparse
import torch
import torchaudio

from train_decoder_v2 import MelSpecExtractor, VocosDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--decoder-path", "-d", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="test_decoder_v2_output.wav")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    mel_extractor = MelSpecExtractor().to(device)
    model = VocosDecoder().to(device)
    state_dict = torch.load(args.decoder_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load audio
    print(f"Loading audio: {args.input}")
    audio, sr = torchaudio.load(args.input)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 32000:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=32000)
    audio = audio.to(device)

    # Extract mel and decode
    print("Processing...")
    with torch.no_grad():
        mel = mel_extractor(audio)
        print(f"Mel shape: {mel.shape}")
        audio_hat = model(mel)
        print(f"Output shape: {audio_hat.shape}")

    # Save
    audio_hat = audio_hat.cpu()
    torchaudio.save(args.output, audio_hat, 32000)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
