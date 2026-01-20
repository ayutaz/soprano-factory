"""Tests for inference_gan.py"""

import os
import sys
import tempfile
import pytest
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestEnhanceAudioWithGan:
    """Tests for enhance_audio_with_gan function."""

    def _create_test_decoder(self, tmpdir):
        """Create a test GAN decoder weights file."""
        from train_decoder_gan import VocosDecoder

        decoder = VocosDecoder()
        decoder_path = os.path.join(tmpdir, "decoder.pth")
        torch.save(decoder.state_dict(), decoder_path)
        return decoder_path

    def test_enhance_basic(self, tmp_path):
        """Test basic audio enhancement."""
        from inference_gan import enhance_audio_with_gan

        decoder_path = self._create_test_decoder(str(tmp_path))

        # Create test audio
        audio = torch.randn(1, 16000) * 0.5  # 0.5 seconds at 32kHz

        device = torch.device("cpu")
        enhanced = enhance_audio_with_gan(
            audio=audio,
            sample_rate=32000,
            gan_decoder_path=decoder_path,
            device=device,
        )

        assert enhanced.dim() == 2
        assert enhanced.size(0) == 1
        assert enhanced.size(1) > 0

    def test_enhance_with_resample(self, tmp_path):
        """Test audio enhancement with resampling."""
        from inference_gan import enhance_audio_with_gan

        decoder_path = self._create_test_decoder(str(tmp_path))

        # Create test audio at different sample rate
        audio = torch.randn(1, 22050)  # 1 second at 22050Hz

        device = torch.device("cpu")
        enhanced = enhance_audio_with_gan(
            audio=audio,
            sample_rate=22050,
            gan_decoder_path=decoder_path,
            device=device,
        )

        assert enhanced.dim() == 2
        assert enhanced.size(1) > 0


class TestImports:
    """Test that all imports work correctly."""

    def test_import_enhance_function(self):
        from inference_gan import enhance_audio_with_gan
        assert enhance_audio_with_gan is not None

    def test_import_main(self):
        from inference_gan import main
        assert main is not None
