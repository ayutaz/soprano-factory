"""
Decoder training script with GAN (Discriminator) for high-quality audio.
Based on Vocos training approach.
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from encoder.codec import VocosBackbone, safe_log
from soprano.vocos.heads import ISTFTHead

# Import discriminators directly to avoid encodec dependency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vocos', 'vocos'))
from discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class MelSpecExtractor(nn.Module):
    """Extract mel spectrogram from audio."""
    def __init__(self, sample_rate=32000, n_fft=2048, hop_length=512, n_mels=100):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(audio)
        mel = safe_log(mel)
        return mel


class VocosDecoder(nn.Module):
    """Full Vocos decoder: MelSpec -> Backbone -> ISTFTHead -> Audio"""
    def __init__(
        self,
        n_mels: int = 100,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels=n_mels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            input_kernel_size=7,
            dw_kernel_size=7,
        )
        self.head = ISTFTHead(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.backbone(mel)
        audio = self.head(x)
        return audio


class MultiScaleMelLoss(nn.Module):
    """Multi-scale mel spectrogram reconstruction loss."""
    def __init__(self, sample_rate: int = 32000):
        super().__init__()
        self.mel_specs = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                n_mels=80, center=True, power=1,
            )
            for n_fft, hop_length in [(512, 128), (1024, 256), (2048, 512)]
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for mel_spec in self.mel_specs:
            mel_hat = safe_log(mel_spec(y_hat))
            mel = safe_log(mel_spec(y))
            loss += nn.functional.l1_loss(mel_hat, mel)
        return loss / len(self.mel_specs)


class GeneratorLoss(nn.Module):
    """Hinge loss for generator."""
    def forward(self, disc_outputs: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for dg in disc_outputs:
            loss += torch.mean(torch.clamp(1 - dg, min=0))
        return loss


class DiscriminatorLoss(nn.Module):
    """Hinge loss for discriminator."""
    def forward(self, disc_real: List[torch.Tensor], disc_fake: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for dr, dg in zip(disc_real, disc_fake):
            loss += torch.mean(torch.clamp(1 - dr, min=0))
            loss += torch.mean(torch.clamp(1 + dg, min=0))
        return loss


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss between real and fake feature maps."""
    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = 0.0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss


class DecoderDataset(Dataset):
    """Dataset for decoder training."""
    def __init__(self, filelist_path: str, sample_rate: int = 32000,
                 num_samples: int = 32768, train: bool = True):
        with open(filelist_path, encoding="utf-8") as f:
            self.filelist = [line.strip() for line in f if line.strip()]
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)
        if y.size(-1) < self.num_samples:
            y = torch.nn.functional.pad(y, (0, self.num_samples - y.size(-1)))
        elif self.train:
            start = random.randint(0, y.size(-1) - self.num_samples)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, :self.num_samples]
        return y.squeeze(0)


@dataclass
class TrainConfig:
    train_filelist: str
    val_filelist: str
    save_dir: str
    max_steps: int = 100000
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_samples: int = 32768
    val_freq: int = 1000
    save_freq: int = 5000
    log_freq: int = 100
    num_workers: int = 0
    mel_loss_coeff: float = 45.0
    feat_match_coeff: float = 2.0
    pretrain_mel_steps: int = 0  # Steps to pretrain without GAN
    use_wandb: bool = True
    wandb_project: str = "soprano-decoder-gan"
    wandb_run_name: Optional[str] = None


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.save_dir, exist_ok=True)

    if config.use_wandb and HAS_WANDB:
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))

    # Create models
    print("Creating models...")
    mel_extractor = MelSpecExtractor().to(device)
    generator = VocosDecoder().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    # Create datasets
    print("Creating datasets...")
    train_dataset = DecoderDataset(config.train_filelist, num_samples=config.num_samples, train=True)
    val_dataset = DecoderDataset(config.val_filelist, num_samples=config.num_samples, train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    # Losses
    mel_loss_fn = MultiScaleMelLoss().to(device)
    gen_loss_fn = GeneratorLoss()
    disc_loss_fn = DiscriminatorLoss()
    feat_loss_fn = FeatureMatchingLoss()

    # Optimizers
    opt_gen = torch.optim.AdamW(generator.parameters(), lr=config.learning_rate, betas=(0.8, 0.9))
    opt_disc = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=config.learning_rate, betas=(0.8, 0.9)
    )

    # Schedulers
    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=config.max_steps)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=config.max_steps)

    # Training loop
    print(f"Starting training for {config.max_steps} steps...")
    generator.train()
    mpd.train()
    mrd.train()

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        try:
            audio = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            audio = next(train_iter)

        audio = audio.to(device)

        # Extract mel spectrogram
        with torch.no_grad():
            mel = mel_extractor(audio)

        # Generate audio
        audio_hat = generator(mel)

        # Match lengths
        min_len = min(audio.size(-1), audio_hat.size(-1))
        audio = audio[:, :min_len]
        audio_hat_matched = audio_hat[:, :min_len]

        # Train discriminator
        use_gan = step >= config.pretrain_mel_steps

        if use_gan:
            opt_disc.zero_grad()

            # Discriminator forward
            y_df_r, y_df_g, _, _ = mpd(audio, audio_hat_matched.detach())
            y_ds_r, y_ds_g, _, _ = mrd(audio, audio_hat_matched.detach())

            loss_disc = disc_loss_fn(y_df_r + y_ds_r, y_df_g + y_ds_g)
            loss_disc.backward()
            opt_disc.step()
        else:
            loss_disc = torch.tensor(0.0)

        # Train generator
        opt_gen.zero_grad()

        # Mel loss
        loss_mel = mel_loss_fn(audio_hat_matched, audio) * config.mel_loss_coeff

        if use_gan:
            # Adversarial loss
            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(audio, audio_hat_matched)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(audio, audio_hat_matched)

            loss_gen = gen_loss_fn(y_df_g + y_ds_g)
            loss_feat = feat_loss_fn(fmap_f_r + fmap_s_r, fmap_f_g + fmap_s_g) * config.feat_match_coeff

            loss_total = loss_mel + loss_gen + loss_feat
        else:
            loss_gen = torch.tensor(0.0)
            loss_feat = torch.tensor(0.0)
            loss_total = loss_mel

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_gen.step()

        scheduler_gen.step()
        scheduler_disc.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({
            "mel": f"{loss_mel.item()/config.mel_loss_coeff:.4f}",
            "gen": f"{loss_gen.item() if isinstance(loss_gen, torch.Tensor) else 0:.4f}",
        })

        # Logging
        if step % config.log_freq == 0 and config.use_wandb and HAS_WANDB:
            wandb.log({
                "train/loss_mel": loss_mel.item() / config.mel_loss_coeff,
                "train/loss_gen": loss_gen.item() if isinstance(loss_gen, torch.Tensor) else 0,
                "train/loss_disc": loss_disc.item() if isinstance(loss_disc, torch.Tensor) else 0,
                "train/loss_feat": loss_feat.item() if isinstance(loss_feat, torch.Tensor) else 0,
                "train/lr": scheduler_gen.get_last_lr()[0],
            }, step=step)

        # Validation
        if step % config.val_freq == 0:
            generator.eval()
            val_losses = []
            with torch.no_grad():
                for val_audio in val_loader:
                    val_audio = val_audio.to(device)
                    mel = mel_extractor(val_audio)
                    audio_hat = generator(mel)
                    min_len = min(val_audio.size(-1), audio_hat.size(-1))
                    val_loss = mel_loss_fn(audio_hat[:, :min_len], val_audio[:, :min_len])
                    val_losses.append(val_loss.item())
                    if len(val_losses) >= 10:
                        break

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"\nStep {step}: val_loss = {avg_val_loss:.4f}")

            if config.use_wandb and HAS_WANDB:
                wandb.log({"val/loss": avg_val_loss}, step=step)

            generator.train()

        # Save checkpoint
        if step % config.save_freq == 0:
            checkpoint_path = os.path.join(config.save_dir, f"decoder_step_{step}.pth")
            torch.save({
                'generator': generator.state_dict(),
                'mpd': mpd.state_dict(),
                'mrd': mrd.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
                'step': step,
            }, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")

    # Save final model (generator only for inference)
    final_path = os.path.join(config.save_dir, "decoder.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    if config.use_wandb and HAS_WANDB:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Vocos decoder with GAN")
    parser.add_argument("--train-filelist", type=str, required=True)
    parser.add_argument("--val-filelist", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--val-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mel-loss-coeff", type=float, default=45.0)
    parser.add_argument("--pretrain-mel-steps", type=int, default=0,
                        help="Steps to pretrain with mel loss only before adding GAN")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="soprano-decoder-gan")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        train_filelist=args.train_filelist,
        val_filelist=args.val_filelist,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        val_freq=args.val_freq,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        mel_loss_coeff=args.mel_loss_coeff,
        pretrain_mel_steps=args.pretrain_mel_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train(config)


if __name__ == "__main__":
    main()
