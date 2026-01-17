"""
Test script for trained decoder.
Encodes audio to FSQ codes, then decodes back to audio using the trained decoder.
"""
import argparse
import torch
import torchaudio

from train_decoder import (
    load_encoder,
    SopranoDecoderHF,
    DecoderWithProjection,
)


def main():
    parser = argparse.ArgumentParser(description="Test trained decoder")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input audio file")
    parser.add_argument("--decoder-path", "-d", type=str, required=True, help="Path to trained decoder")
    parser.add_argument("--output", "-o", type=str, default="test_decoder_output.wav", help="Output file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load encoder
    print("Loading encoder...")
    encoder = load_encoder(device)

    # Load trained decoder
    print(f"Loading decoder from: {args.decoder_path}")
    base_decoder = SopranoDecoderHF().to(device)
    model = DecoderWithProjection(base_decoder).to(device)
    state_dict = torch.load(args.decoder_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess audio
    print(f"Loading audio: {args.input}")
    audio, sr = torchaudio.load(args.input)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 32000:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=32000)
    audio = audio.to(device)

    # Encode to FSQ codes
    print("Encoding audio to FSQ codes...")
    with torch.no_grad():
        mel = encoder.preprocess(audio)
        x = encoder.encoder(mel)
        x = x[:, :, ::encoder.downsample_scale]
        x = x.transpose(1, 2)
        x = encoder.downsampler(x)
        fsq_codes = encoder.quant(x)  # (1, L, 5)
        print(f"FSQ codes shape: {fsq_codes.shape}")

    # Decode
    print("Decoding with trained decoder...")
    with torch.no_grad():
        audio_hat = model(fsq_codes)
        print(f"Reconstructed audio shape: {audio_hat.shape}")

    # Save output
    audio_hat = audio_hat.cpu()
    torchaudio.save(args.output, audio_hat, 32000)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
