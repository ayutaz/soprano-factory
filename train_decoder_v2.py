"""
Decoder training script for Soprano TTS - Version 2.
Trains the decoder directly from mel spectrogram (no FSQ information loss).
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from encoder.codec import VocosBackbone, safe_log
from soprano.vocos.heads import ISTFTHead

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
    """
    Full Vocos decoder: MelSpec -> Backbone -> ISTFTHead -> Audio
    """
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
        """
        Args:
            mel: Mel spectrogram (B, n_mels, T)
        Returns:
            audio: Reconstructed audio (B, T')
        """
        x = self.backbone(mel)  # (B, dim, T)
        audio = self.head(x)    # (B, T')
        return audio


class MelSpecLoss(nn.Module):
    """Multi-scale mel spectrogram reconstruction loss."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft_list: list = [512, 1024, 2048],
        hop_length_list: list = [128, 256, 512],
        n_mels: int = 80,
    ):
        super().__init__()
        self.mel_specs = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                center=True,
                power=1,
            )
            for n_fft, hop_length in zip(n_fft_list, hop_length_list)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for mel_spec in self.mel_specs:
            mel_hat = safe_log(mel_spec(y_hat))
            mel = safe_log(mel_spec(y))
            loss += nn.functional.l1_loss(mel_hat, mel)
        return loss / len(self.mel_specs)


class DecoderDataset(Dataset):
    """Dataset for decoder training."""

    def __init__(
        self,
        filelist_path: str,
        sample_rate: int = 32000,
        num_samples: int = 32768,
        train: bool = True,
    ):
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
            pad_length = self.num_samples - y.size(-1)
            y = torch.nn.functional.pad(y, (0, pad_length))
        elif self.train:
            start = random.randint(0, y.size(-1) - self.num_samples)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, : self.num_samples]

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
    use_wandb: bool = True
    wandb_project: str = "soprano-decoder-v2"
    wandb_run_name: Optional[str] = None


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.save_dir, exist_ok=True)

    if config.use_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Create models
    print("Creating models...")
    mel_extractor = MelSpecExtractor().to(device)
    model = VocosDecoder().to(device)

    # Create datasets
    print("Creating datasets...")
    train_dataset = DecoderDataset(
        config.train_filelist,
        num_samples=config.num_samples,
        train=True,
    )
    val_dataset = DecoderDataset(
        config.val_filelist,
        num_samples=config.num_samples,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Loss and optimizer
    mel_loss_fn = MelSpecLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.8, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps)

    # Training loop
    print(f"Starting training for {config.max_steps} steps...")
    model.train()
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

        # Decode
        audio_hat = model(mel)

        # Match lengths
        min_len = min(audio.size(-1), audio_hat.size(-1))
        audio = audio[:, :min_len]
        audio_hat = audio_hat[:, :min_len]

        # Compute loss
        loss = mel_loss_fn(audio_hat, audio)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if step % config.log_freq == 0:
            if config.use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=step)

        if step % config.val_freq == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_audio in val_loader:
                    val_audio = val_audio.to(device)
                    mel = mel_extractor(val_audio)
                    audio_hat = model(mel)

                    min_len = min(val_audio.size(-1), audio_hat.size(-1))
                    val_loss = mel_loss_fn(audio_hat[:, :min_len], val_audio[:, :min_len])
                    val_losses.append(val_loss.item())

                    if len(val_losses) >= 10:
                        break

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"\nStep {step}: val_loss = {avg_val_loss:.4f}")

            if config.use_wandb and HAS_WANDB:
                wandb.log({"val/loss": avg_val_loss}, step=step)

            model.train()

        if step % config.save_freq == 0:
            checkpoint_path = os.path.join(config.save_dir, f"decoder_step_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(config.save_dir, "decoder.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    if config.use_wandb and HAS_WANDB:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Vocos decoder (v2 - mel to audio)")
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
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="soprano-decoder-v2")
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
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train(config)


if __name__ == "__main__":
    main()
