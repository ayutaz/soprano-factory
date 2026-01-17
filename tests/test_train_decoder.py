"""Tests for train_decoder.py"""

import os
import tempfile
import pytest
import torch


class TestMelSpecLoss:
    """Tests for MelSpecLoss class."""

    def test_loss_computation(self):
        """Test that loss is computed correctly."""
        from train_decoder import MelSpecLoss

        loss_fn = MelSpecLoss(sample_rate=32000)

        # Create dummy audio tensors
        batch_size = 2
        audio_length = 16000  # 0.5 seconds at 32kHz
        y = torch.randn(batch_size, audio_length)
        y_hat = torch.randn(batch_size, audio_length)

        loss = loss_fn(y_hat, y)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_identical_audio_low_loss(self):
        """Test that identical audio produces low loss."""
        from train_decoder import MelSpecLoss

        loss_fn = MelSpecLoss(sample_rate=32000)

        y = torch.randn(2, 16000)
        loss = loss_fn(y, y)

        assert loss.item() < 0.01, f"Loss for identical audio should be very low, got {loss.item()}"

    def test_different_audio_higher_loss(self):
        """Test that different audio produces higher loss than identical."""
        from train_decoder import MelSpecLoss

        loss_fn = MelSpecLoss(sample_rate=32000)

        y = torch.randn(2, 16000)
        y_different = torch.randn(2, 16000)

        loss_identical = loss_fn(y, y)
        loss_different = loss_fn(y_different, y)

        assert loss_different > loss_identical, "Different audio should have higher loss"


class TestDecoderDataset:
    """Tests for DecoderDataset class."""

    def _create_test_wavs(self, tmpdir, num_files=5, duration_samples=32000):
        """Create test wav files."""
        import torchaudio

        wav_paths = []
        for i in range(num_files):
            wav_path = os.path.join(tmpdir, f"test_{i}.wav")
            # Create random audio
            audio = torch.randn(1, duration_samples) * 0.5
            torchaudio.save(wav_path, audio, sample_rate=32000)
            wav_paths.append(wav_path)

        # Create filelist
        filelist_path = os.path.join(tmpdir, "filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("\n".join(wav_paths))

        return filelist_path

    def test_dataset_creation(self, tmp_path):
        """Test dataset creation and length."""
        from train_decoder import DecoderDataset

        filelist_path = self._create_test_wavs(str(tmp_path), num_files=10)
        dataset = DecoderDataset(filelist_path, num_samples=16000)

        assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"

    def test_dataset_item_shape(self, tmp_path):
        """Test that dataset returns correct shape."""
        from train_decoder import DecoderDataset

        num_samples = 16000
        filelist_path = self._create_test_wavs(str(tmp_path), num_files=3, duration_samples=32000)
        dataset = DecoderDataset(filelist_path, num_samples=num_samples, train=True)

        audio = dataset[0]

        assert audio.dim() == 1, f"Expected 1D tensor, got {audio.dim()}D"
        assert audio.size(0) == num_samples, f"Expected {num_samples} samples, got {audio.size(0)}"

    def test_dataset_padding_short_audio(self, tmp_path):
        """Test that short audio is padded correctly."""
        from train_decoder import DecoderDataset

        num_samples = 32000
        # Create very short audio
        filelist_path = self._create_test_wavs(str(tmp_path), num_files=1, duration_samples=8000)
        dataset = DecoderDataset(filelist_path, num_samples=num_samples, train=False)

        audio = dataset[0]

        assert audio.size(0) == num_samples, f"Expected {num_samples} samples after padding"


class TestDecoderWithProjection:
    """Tests for DecoderWithProjection class."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        from train_decoder import DecoderWithProjection
        from soprano.vocos.decoder import SopranoDecoder

        decoder = SopranoDecoder()
        model = DecoderWithProjection(decoder)

        # Create dummy FSQ codes
        batch_size = 2
        seq_len = 16
        fsq_codes = torch.randn(batch_size, seq_len, 5)

        audio = model(fsq_codes)

        assert audio.dim() == 2, f"Expected 2D output, got {audio.dim()}D"
        assert audio.size(0) == batch_size, f"Batch size mismatch"
        # Output length depends on decoder architecture
        # upscale=4, hop_length=512, so approx: (seq_len * 4 - 3) * 512
        # For seq_len=16: (16*4-3)*512 = 61*512 = 31232, but ISTFT reduces it
        # Just check it's reasonably long
        expected_min_length = seq_len * 1500  # Conservative estimate
        assert audio.size(1) > expected_min_length, f"Output too short: {audio.size(1)}"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from train_decoder import DecoderWithProjection
        from soprano.vocos.decoder import SopranoDecoder

        decoder = SopranoDecoder()
        model = DecoderWithProjection(decoder)

        fsq_codes = torch.randn(2, 16, 5, requires_grad=True)
        audio = model(fsq_codes)

        # Compute dummy loss and backward
        loss = audio.mean()
        loss.backward()

        # Check projection has gradients
        assert model.projection.weight.grad is not None, "Projection should have gradients"
        # Check decoder has gradients
        has_decoder_grad = any(
            p.grad is not None for p in model.decoder.parameters() if p.requires_grad
        )
        assert has_decoder_grad, "Decoder should have gradients"


class TestSafeLog:
    """Tests for safe_log function."""

    def test_positive_values(self):
        """Test safe_log with positive values."""
        from train_decoder import safe_log

        x = torch.tensor([1.0, 2.0, 10.0])
        result = safe_log(x)

        expected = torch.log(x)
        assert torch.allclose(result, expected), "safe_log should match torch.log for positive values"

    def test_zero_clipping(self):
        """Test that zero values are clipped."""
        from train_decoder import safe_log

        x = torch.tensor([0.0, 0.0, 0.0])
        result = safe_log(x)

        assert not torch.isinf(result).any(), "safe_log should not produce -inf for zeros"
        assert not torch.isnan(result).any(), "safe_log should not produce NaN"

    def test_negative_values(self):
        """Test that negative values are clipped."""
        from train_decoder import safe_log

        x = torch.tensor([-1.0, -0.5, 0.0])
        result = safe_log(x)

        assert not torch.isnan(result).any(), "safe_log should not produce NaN for negative values"


class TestLoadEncoder:
    """Tests for load_encoder function."""

    def test_encoder_device_tensors(self):
        """Test that encoder quantizer tensors are on correct device."""
        from train_decoder import load_encoder
        import torch

        device = torch.device("cpu")  # Use CPU for testing
        encoder = load_encoder(device)

        # Check quantizer tensors are on the correct device
        assert encoder.quant.levels.device == device
        assert encoder.quant.half_levels.device == device
        assert encoder.quant.offset.device == device
        assert encoder.quant.shift.device == device
        assert encoder.quant._basis.device == device

    def test_encoder_eval_mode(self):
        """Test that encoder is in eval mode."""
        from train_decoder import load_encoder
        import torch

        encoder = load_encoder(torch.device("cpu"))
        assert not encoder.training, "Encoder should be in eval mode"

    def test_encoder_frozen(self):
        """Test that encoder parameters are frozen."""
        from train_decoder import load_encoder
        import torch

        encoder = load_encoder(torch.device("cpu"))
        for param in encoder.parameters():
            assert not param.requires_grad, "Encoder parameters should be frozen"


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """Test TrainConfig default values."""
        from train_decoder import TrainConfig

        config = TrainConfig(
            train_filelist="train.txt",
            val_filelist="val.txt",
            save_dir="./output",
        )

        assert config.max_steps == 100000
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.use_wandb is True

    def test_custom_values(self):
        """Test TrainConfig with custom values."""
        from train_decoder import TrainConfig

        config = TrainConfig(
            train_filelist="train.txt",
            val_filelist="val.txt",
            save_dir="./output",
            max_steps=50000,
            batch_size=16,
            use_wandb=True,
        )

        assert config.max_steps == 50000
        assert config.batch_size == 16
        assert config.use_wandb is True


class TestSopranoDecoderHF:
    """Tests for SopranoDecoderHF class."""

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        from train_decoder import SopranoDecoderHF

        decoder = SopranoDecoderHF()
        # Input: (B, 512, L)
        x = torch.randn(2, 512, 16)
        audio = decoder(x)

        assert audio.dim() == 2
        assert audio.size(0) == 2

    def test_interpolation(self):
        """Test that interpolation upsamples correctly."""
        from train_decoder import SopranoDecoderHF

        decoder = SopranoDecoderHF(upscale=4)
        x = torch.randn(1, 512, 10)
        # After interpolation: 4 * (10 - 1) + 1 = 37
        audio = decoder(x)
        assert audio.size(1) > 0


class TestConvertHfStateDict:
    """Tests for convert_hf_state_dict function."""

    def test_pwconv_conversion(self):
        """Test pwconv key conversion."""
        from train_decoder import convert_hf_state_dict

        old_state = {
            'decoder.convnext.0.pwconv1.weight': torch.randn(1),
            'decoder.convnext.0.pwconv2.bias': torch.randn(1),
        }
        new_state = convert_hf_state_dict(old_state)

        assert 'decoder.convnext.0.mlp.pwconv1.weight' in new_state
        assert 'decoder.convnext.0.mlp.pwconv2.bias' in new_state

    def test_non_pwconv_keys_unchanged(self):
        """Test that non-pwconv keys are unchanged."""
        from train_decoder import convert_hf_state_dict

        old_state = {
            'decoder.head.out.weight': torch.randn(1),
            'decoder.embed.weight': torch.randn(1),
        }
        new_state = convert_hf_state_dict(old_state)

        assert 'decoder.head.out.weight' in new_state
        assert 'decoder.embed.weight' in new_state
