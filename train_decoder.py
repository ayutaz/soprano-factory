"""
Decoder training script for Soprano TTS.
Trains the decoder (Vocos-based) on Japanese audio to improve pronunciation quality.
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

# Import Soprano components
from encoder.codec import Encoder, VocosBackbone

# Import vocos head
from soprano.vocos.heads import ISTFTHead


class SopranoDecoderHF(nn.Module):
    """
    Soprano Decoder compatible with HuggingFace Soprano-80M weights.
    Uses input_kernel_size=3 to match HF architecture.
    """
    def __init__(
        self,
        num_input_channels: int = 512,
        decoder_num_layers: int = 8,
        decoder_dim: int = 512,
        decoder_intermediate_dim: int = 1536,
        hop_length: int = 512,
        n_fft: int = 2048,
        upscale: int = 4,
        dw_kernel: int = 3,
    ):
        super().__init__()
        self.decoder_initial_channels = num_input_channels
        self.num_layers = decoder_num_layers
        self.dim = decoder_dim
        self.intermediate_dim = decoder_intermediate_dim
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.upscale = upscale
        self.dw_kernel = dw_kernel

        self.decoder = VocosBackbone(
            input_channels=self.decoder_initial_channels,
            dim=self.dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            input_kernel_size=3,  # Match HuggingFace
            dw_kernel_size=dw_kernel,
        )
        self.head = ISTFTHead(
            dim=self.dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        x = torch.nn.functional.interpolate(
            x, size=self.upscale * (T - 1) + 1, mode='linear', align_corners=True
        )
        x = self.decoder(x)
        reconstructed = self.head(x)
        return reconstructed

# Import for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def safe_log(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    """Safe log function to avoid log(0)."""
    return torch.log(torch.clamp(x, min=clip_val))


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
        """
        Compute multi-scale mel spectrogram loss.

        Args:
            y_hat: Predicted audio (B, T)
            y: Target audio (B, T)

        Returns:
            L1 loss averaged across scales.
        """
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
        num_samples: int = 32768,  # ~1 second at 32kHz
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

        # Mix to mono if stereo
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)

        # Pad or crop
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
    """Training configuration."""
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
    num_workers: int = 0  # Windows compatibility
    use_wandb: bool = True
    wandb_project: str = "soprano-decoder"
    wandb_run_name: Optional[str] = None
    pretrained_decoder: Optional[str] = None


def load_encoder(device: torch.device) -> Encoder:
    """Load the Soprano encoder for tokenization.

    Note: The encoder is used only for encoding audio to FSQ codes.
    It is not trained - only the decoder is trained.
    """
    encoder = Encoder().to(device)

    # Move quantizer tensors to device (they are not registered as buffers)
    if hasattr(encoder.quant, 'levels') and encoder.quant.levels is not None:
        encoder.quant.levels = encoder.quant.levels.to(device)
        encoder.quant.half_levels = encoder.quant.half_levels.to(device)
        encoder.quant.offset = encoder.quant.offset.to(device)
        encoder.quant.shift = encoder.quant.shift.to(device)
    if hasattr(encoder.quant, '_basis'):
        encoder.quant._basis = encoder.quant._basis.to(device)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def convert_hf_state_dict(state_dict: dict) -> dict:
    """Convert HuggingFace state dict keys to local model format.

    HF uses: decoder.convnext.X.pwconv1 / pwconv2
    Local uses: decoder.convnext.X.mlp.pwconv1 / mlp.pwconv2
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # Convert pwconv keys to mlp.pwconv
        if '.convnext.' in k and ('.pwconv1' in k or '.pwconv2' in k):
            # decoder.convnext.0.pwconv1 -> decoder.convnext.0.mlp.pwconv1
            parts = k.split('.')
            for i, part in enumerate(parts):
                if part in ('pwconv1', 'pwconv2'):
                    parts.insert(i, 'mlp')
                    break
            new_key = '.'.join(parts)
        new_state_dict[new_key] = v
    return new_state_dict


def load_decoder(device: torch.device, pretrained_path: Optional[str] = None) -> SopranoDecoderHF:
    """Load the Soprano decoder.

    Note: HuggingFace Soprano-80M uses dim=512, input_kernel_size=3.
    We use SopranoDecoderHF which matches the HF architecture.
    """
    from huggingface_hub import hf_hub_download

    decoder = SopranoDecoderHF().to(device)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        # Check if this is a HF checkpoint (has pwconv without mlp prefix)
        if any('.pwconv1.' in k or k.endswith('.pwconv1.weight') for k in state_dict.keys()):
            if not any('.mlp.pwconv1' in k for k in state_dict.keys()):
                state_dict = convert_hf_state_dict(state_dict)
        decoder.load_state_dict(state_dict)
    else:
        # Load from HuggingFace
        decoder_path = hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth')
        state_dict = torch.load(decoder_path, map_location=device)
        state_dict = convert_hf_state_dict(state_dict)
        decoder.load_state_dict(state_dict)
    return decoder


def get_embeddings_from_audio(encoder: Encoder, audio: torch.Tensor) -> torch.Tensor:
    """
    Encode audio to get FSQ embeddings.

    Args:
        encoder: Soprano encoder
        audio: Audio tensor (B, T)

    Returns:
        Embeddings tensor (B, 512, L) suitable for decoder input
    """
    with torch.no_grad():
        # Preprocess to mel spectrogram
        mel = encoder.preprocess(audio)
        # Encode to get FSQ codes
        x = encoder.encoder(mel)
        x = x[:, :, ::encoder.downsample_scale]
        x = x.transpose(1, 2)
        x = encoder.downsampler(x)
        codes = encoder.quant(x)
        # codes shape: (B, L, 5) - normalized FSQ values

        # Convert to embeddings by expanding dimensions
        # The decoder expects 512 channels, so we need to project
        # For now, we'll use a simple approach: repeat and transform
        # In the actual Soprano pipeline, hidden states from LLM are used

        # Create embeddings similar to LLM hidden states
        # This is a simplified version - the actual embedding layer is in the LLM
        B, L, D = codes.shape
        # Project from 5-dim FSQ to 512-dim embeddings
        # Use a simple learned projection (we'll add this to the decoder)
        embeddings = codes.transpose(1, 2)  # (B, 5, L)

    return embeddings, codes


class DecoderWithProjection(nn.Module):
    """Decoder with input projection for training."""

    def __init__(self, decoder: SopranoDecoderHF, fsq_dim: int = 5, embed_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(fsq_dim, embed_dim)
        self.decoder = decoder

    def forward(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fsq_codes: FSQ codes (B, L, 5)

        Returns:
            Reconstructed audio (B, T)
        """
        # Project to embedding space
        embeddings = self.projection(fsq_codes)  # (B, L, 512)
        embeddings = embeddings.transpose(1, 2)  # (B, 512, L)
        # Decode
        audio = self.decoder(embeddings)
        return audio


def train(config: TrainConfig):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Initialize wandb
    if config.use_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Load models
    print("Loading encoder...")
    encoder = load_encoder(device)

    print("Loading decoder...")
    base_decoder = load_decoder(device, config.pretrained_decoder)
    model = DecoderWithProjection(base_decoder).to(device)

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

        # Encode audio to get FSQ codes
        with torch.no_grad():
            mel = encoder.preprocess(audio)
            x = encoder.encoder(mel)
            x = x[:, :, ::encoder.downsample_scale]
            x = x.transpose(1, 2)
            x = encoder.downsampler(x)
            fsq_codes = encoder.quant(x)  # (B, L, 5)

        # Decode
        audio_hat = model(fsq_codes)

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

        # Logging
        if step % config.log_freq == 0:
            if config.use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=step)

        # Validation
        if step % config.val_freq == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_audio in val_loader:
                    val_audio = val_audio.to(device)

                    mel = encoder.preprocess(val_audio)
                    x = encoder.encoder(mel)
                    x = x[:, :, ::encoder.downsample_scale]
                    x = x.transpose(1, 2)
                    x = encoder.downsampler(x)
                    fsq_codes = encoder.quant(x)

                    audio_hat = model(fsq_codes)

                    min_len = min(val_audio.size(-1), audio_hat.size(-1))
                    val_loss = mel_loss_fn(audio_hat[:, :min_len], val_audio[:, :min_len])
                    val_losses.append(val_loss.item())

                    if len(val_losses) >= 10:  # Limit validation batches
                        break

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"\nStep {step}: val_loss = {avg_val_loss:.4f}")

            if config.use_wandb and HAS_WANDB:
                wandb.log({"val/loss": avg_val_loss}, step=step)

            model.train()

        # Save checkpoint
        if step % config.save_freq == 0:
            checkpoint_path = os.path.join(config.save_dir, f"decoder_step_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)  # Save full model including projection
            print(f"\nSaved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(config.save_dir, "decoder.pth")
    torch.save(model.state_dict(), final_path)  # Save full model including projection
    print(f"Saved final model: {final_path}")

    if config.use_wandb and HAS_WANDB:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Soprano decoder on Japanese audio")
    parser.add_argument("--train-filelist", type=str, required=True, help="Path to training file list")
    parser.add_argument("--val-filelist", type=str, required=True, help="Path to validation file list")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=32768, help="Audio samples per batch item")
    parser.add_argument("--val-freq", type=int, default=1000, help="Validation frequency")
    parser.add_argument("--save-freq", type=int, default=5000, help="Checkpoint save frequency")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows compatibility)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="soprano-decoder", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--pretrained-decoder", type=str, default=None, help="Path to pretrained decoder")
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
        pretrained_decoder=args.pretrained_decoder,
    )

    train(config)


if __name__ == "__main__":
    main()
