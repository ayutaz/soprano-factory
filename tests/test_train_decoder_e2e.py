"""Tests for train_decoder_e2e.py"""

import os
import sys
import tempfile
import pytest
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestImports:
    """Test that all imports work correctly."""

    def test_import_mel_extractor(self):
        from train_decoder_e2e import MelSpecExtractor
        extractor = MelSpecExtractor()
        assert extractor is not None

    def test_import_hidden_simulator(self):
        from train_decoder_e2e import HiddenStateSimulator
        simulator = HiddenStateSimulator()
        assert simulator is not None

    def test_import_vocos_decoder_e2e(self):
        from train_decoder_e2e import VocosDecoderE2E
        decoder = VocosDecoderE2E()
        assert decoder is not None


class TestMelSpecExtractor:
    """Tests for MelSpecExtractor."""

    def test_forward_shape(self):
        from train_decoder_e2e import MelSpecExtractor
        extractor = MelSpecExtractor()
        audio = torch.randn(2, 32000)
        mel = extractor(audio)
        assert mel.dim() == 3
        assert mel.size(0) == 2
        assert mel.size(1) == 100  # n_mels


class TestHiddenStateSimulator:
    """Tests for HiddenStateSimulator."""

    def test_forward_shape(self):
        from train_decoder_e2e import HiddenStateSimulator
        simulator = HiddenStateSimulator(n_mels=100, hidden_dim=512)
        mel = torch.randn(2, 100, 64)  # (B, n_mels, T)
        hidden = simulator(mel)
        assert hidden.dim() == 3
        assert hidden.size(0) == 2
        assert hidden.size(1) == 512  # hidden_dim
        assert hidden.size(2) == 64  # T

    def test_gradient_flow(self):
        from train_decoder_e2e import HiddenStateSimulator
        simulator = HiddenStateSimulator()
        mel = torch.randn(2, 100, 32, requires_grad=True)
        hidden = simulator(mel)
        loss = hidden.mean()
        loss.backward()
        assert mel.grad is not None


class TestVocosDecoderE2E:
    """Tests for VocosDecoderE2E."""

    def test_forward_shape(self):
        from train_decoder_e2e import VocosDecoderE2E
        decoder = VocosDecoderE2E(input_dim=512)
        hidden = torch.randn(2, 512, 64)  # (B, hidden_dim, T)
        audio = decoder(hidden)
        assert audio.dim() == 2
        assert audio.size(0) == 2
        assert audio.size(1) > 0

    def test_gradient_flow(self):
        from train_decoder_e2e import VocosDecoderE2E
        decoder = VocosDecoderE2E(input_dim=512)
        hidden = torch.randn(2, 512, 32, requires_grad=True)
        audio = decoder(hidden)
        loss = audio.mean()
        loss.backward()
        assert hidden.grad is not None


class TestLosses:
    """Tests for loss functions."""

    def test_multi_scale_mel_loss(self):
        from train_decoder_e2e import MultiScaleMelLoss
        loss_fn = MultiScaleMelLoss()
        y = torch.randn(2, 16000)
        y_hat = torch.randn(2, 16000)
        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_generator_loss(self):
        from train_decoder_e2e import GeneratorLoss
        loss_fn = GeneratorLoss()
        disc_outputs = [torch.randn(2, 100) for _ in range(5)]
        loss = loss_fn(disc_outputs)
        assert not torch.isnan(loss)

    def test_discriminator_loss(self):
        from train_decoder_e2e import DiscriminatorLoss
        loss_fn = DiscriminatorLoss()
        disc_real = [torch.randn(2, 100) for _ in range(5)]
        disc_fake = [torch.randn(2, 100) for _ in range(5)]
        loss = loss_fn(disc_real, disc_fake)
        assert not torch.isnan(loss)

    def test_feature_matching_loss(self):
        from train_decoder_e2e import FeatureMatchingLoss
        loss_fn = FeatureMatchingLoss()
        fmap_r = [[torch.randn(2, 32, 100) for _ in range(3)] for _ in range(5)]
        fmap_g = [[torch.randn(2, 32, 100) for _ in range(3)] for _ in range(5)]
        loss = loss_fn(fmap_r, fmap_g)
        assert not torch.isnan(loss)


class TestDecoderDataset:
    """Tests for DecoderDataset."""

    def _create_test_wavs(self, tmpdir, num_files=3):
        wav_paths = []
        for i in range(num_files):
            wav_path = os.path.join(tmpdir, f"test_{i}.wav")
            audio = torch.randn(1, 32000) * 0.5
            torchaudio.save(wav_path, audio, sample_rate=32000)
            wav_paths.append(wav_path)
        filelist_path = os.path.join(tmpdir, "filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("\n".join(wav_paths))
        return filelist_path

    def test_dataset_creation(self, tmp_path):
        from train_decoder_e2e import DecoderDataset
        filelist_path = self._create_test_wavs(str(tmp_path))
        dataset = DecoderDataset(filelist_path, num_samples=16000)
        assert len(dataset) == 3

    def test_dataset_item_shape(self, tmp_path):
        from train_decoder_e2e import DecoderDataset
        filelist_path = self._create_test_wavs(str(tmp_path))
        dataset = DecoderDataset(filelist_path, num_samples=16000, train=True)
        audio = dataset[0]
        assert audio.dim() == 1
        assert audio.size(0) == 16000


class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_values(self):
        from train_decoder_e2e import TrainConfig
        config = TrainConfig(
            train_filelist="train.txt",
            val_filelist="val.txt",
            save_dir="./output",
        )
        assert config.max_steps == 100000
        assert config.batch_size == 8
        assert config.mel_loss_coeff == 45.0
        assert config.use_wandb is True


class TestEndToEnd:
    """End-to-end test for the training loop components."""

    def test_training_step(self):
        """Test a single training step."""
        from train_decoder_e2e import (
            MelSpecExtractor, HiddenStateSimulator, VocosDecoderE2E,
            MultiScaleMelLoss, GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss
        )
        from discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator

        # Create models
        mel_extractor = MelSpecExtractor()
        hidden_simulator = HiddenStateSimulator()
        generator = VocosDecoderE2E()
        mpd = MultiPeriodDiscriminator()
        mrd = MultiResolutionDiscriminator()

        # Create loss functions
        mel_loss_fn = MultiScaleMelLoss()
        gen_loss_fn = GeneratorLoss()
        disc_loss_fn = DiscriminatorLoss()
        feat_loss_fn = FeatureMatchingLoss()

        # Create optimizers
        opt_gen = torch.optim.AdamW(
            list(hidden_simulator.parameters()) + list(generator.parameters()),
            lr=1e-4
        )
        opt_disc = torch.optim.AdamW(
            list(mpd.parameters()) + list(mrd.parameters()), lr=1e-4
        )

        # Simulate one training step
        audio = torch.randn(2, 16384)

        with torch.no_grad():
            mel = mel_extractor(audio)

        hidden = hidden_simulator(mel)
        audio_hat = generator(hidden)

        # Match lengths
        min_len = min(audio.size(-1), audio_hat.size(-1))
        audio = audio[:, :min_len]
        audio_hat_matched = audio_hat[:, :min_len]

        # Discriminator step
        opt_disc.zero_grad()
        y_df_r, y_df_g, _, _ = mpd(audio, audio_hat_matched.detach())
        y_ds_r, y_ds_g, _, _ = mrd(audio, audio_hat_matched.detach())
        loss_disc = disc_loss_fn(y_df_r + y_ds_r, y_df_g + y_ds_g)
        loss_disc.backward()
        opt_disc.step()

        # Generator step
        opt_gen.zero_grad()
        loss_mel = mel_loss_fn(audio_hat_matched, audio) * 45.0
        y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(audio, audio_hat_matched)
        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(audio, audio_hat_matched)
        loss_gen = gen_loss_fn(y_df_g + y_ds_g)
        loss_feat = feat_loss_fn(fmap_f_r + fmap_s_r, fmap_f_g + fmap_s_g) * 2.0
        loss_total = loss_mel + loss_gen + loss_feat
        loss_total.backward()
        opt_gen.step()

        # Verify losses are valid
        assert not torch.isnan(loss_disc)
        assert not torch.isnan(loss_mel)
        assert not torch.isnan(loss_gen)
        assert not torch.isnan(loss_feat)
