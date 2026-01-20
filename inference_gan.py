"""
Inference script for fine-tuned Soprano model with GAN decoder enhancement.

This script uses the Soprano TTS model to generate speech, then enhances
the audio quality using a trained GAN decoder (mel-to-audio vocoder).

Usage:
    uv run python inference_gan.py "テキスト" -m weights_moespeech_v5 -g weights_decoder_gan/decoder.pth -o output.wav
"""
import argparse
import torch
import torchaudio
from soprano import SopranoTTS

from train_decoder_gan import MelSpecExtractor, VocosDecoder


def enhance_audio_with_gan(
    audio: torch.Tensor,
    sample_rate: int,
    gan_decoder_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Enhance audio quality using GAN decoder.

    Args:
        audio: Input audio tensor (1, T)
        sample_rate: Sample rate of input audio
        gan_decoder_path: Path to trained GAN decoder weights
        device: Device to use for inference

    Returns:
        Enhanced audio tensor (1, T)
    """
    # Resample to 32kHz if needed
    if sample_rate != 32000:
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=32000)

    # Move to device
    audio = audio.to(device)

    # Load models
    mel_extractor = MelSpecExtractor().to(device)
    gan_decoder = VocosDecoder().to(device)

    state_dict = torch.load(gan_decoder_path, map_location=device, weights_only=True)
    gan_decoder.load_state_dict(state_dict)
    gan_decoder.eval()

    # Extract mel spectrogram and regenerate audio
    with torch.no_grad():
        mel = mel_extractor(audio)
        enhanced_audio = gan_decoder(mel)

    return enhanced_audio.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Soprano model and GAN decoder enhancement"
    )
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--model-path", "-m", type=str, required=True,
                        help="Path to fine-tuned Soprano model")
    parser.add_argument("--gan-decoder", "-g", type=str, required=True,
                        help="Path to trained GAN decoder weights")
    parser.add_argument("--output", "-o", type=str, default="output_gan.wav",
                        help="Output file path")
    parser.add_argument("--device", "-d", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip GAN enhancement (for comparison)")
    parser.add_argument("--save-original", type=str, default=None,
                        help="Save original (non-enhanced) audio to this path")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load Soprano model
    print(f"Loading Soprano model from: {args.model_path}")
    soprano = SopranoTTS(
        backend='transformers',
        device=str(device),
        model_path=args.model_path
    )

    # Generate initial audio
    print(f"Generating speech for: {args.text}")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = os.path.join(tmpdir, "soprano_output.wav")
        soprano.infer(args.text, out_path=tmp_output)

        # Load generated audio
        audio, sr = torchaudio.load(tmp_output)
        print(f"Original audio: {audio.shape}, sample_rate={sr}")

        # Save original if requested
        if args.save_original:
            torchaudio.save(args.save_original, audio, sr)
            print(f"Saved original audio to: {args.save_original}")

    if args.no_enhance:
        # Just save the original
        torchaudio.save(args.output, audio, sr)
        print(f"Saved (no enhancement) to: {args.output}")
        return

    # Enhance with GAN decoder
    print(f"Enhancing audio with GAN decoder: {args.gan_decoder}")
    enhanced_audio = enhance_audio_with_gan(
        audio=audio,
        sample_rate=sr,
        gan_decoder_path=args.gan_decoder,
        device=device,
    )

    # Save enhanced audio
    print(f"Enhanced audio shape: {enhanced_audio.shape}")
    torchaudio.save(args.output, enhanced_audio, 32000)
    print(f"Saved enhanced audio to: {args.output}")


if __name__ == "__main__":
    main()
