"""
Training script for Soprano.

Usage:
python train.py --input-dir path/to/files --save-dir path/to/weights

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
--save-dir: Path to directory to save weights

Adapted from https://github.com/karpathy/nanoGPT
"""
import argparse
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import AudioDataset
from tokenizer_utils import add_japanese_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    parser.add_argument("--save-dir",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument("--device",
        default="cuda:0",
        help="Device to use (e.g., cuda:0 or cpu)"
    )
    parser.add_argument("--seed",
        type=int,
        default=1337,
        help="Random seed"
    )
    parser.add_argument("--max-steps",
        type=int,
        default=50000,
        help="Number of training steps"
    )
    parser.add_argument("--batch-size",
        type=int,
        default=4,
        help="Batch size for packed sequences"
    )
    parser.add_argument("--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument("--seq-len",
        type=int,
        default=1024,
        help="Sequence length for packed tokens"
    )
    parser.add_argument("--val-freq",
        type=int,
        default=250,
        help="Validation frequency in steps (0 to disable)"
    )
    parser.add_argument("--text-factor",
        type=float,
        default=0.5,
        help="Weight for text-token loss (0.0 disables text loss)"
    )
    parser.add_argument("--transition-factor",
        type=float,
        default=1.0,
        help="Weight for text→audio transition loss"
    )
    parser.add_argument("--attention-factor",
        type=float,
        default=0.1,
        help="Weight for attention-based text-audio alignment loss"
    )
    parser.add_argument("--attention-warmup-steps",
        type=int,
        default=1000,
        help="Steps before enabling attention loss"
    )
    parser.add_argument("--lr",
        type=float,
        default=1e-4,
        help="Maximum learning rate"
    )
    parser.add_argument("--embedding-lr-factor",
        type=float,
        default=1.0,
        help="Learning rate multiplier for new token embeddings (e.g., 10.0 for 10x LR)"
    )
    parser.add_argument("--contrastive-factor",
        type=float,
        default=0.0,
        help="Weight for contrastive text-audio alignment loss"
    )
    parser.add_argument("--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument("--wandb-project",
        type=str,
        default="soprano-factory",
        help="WandB project name"
    )
    parser.add_argument("--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (optional)"
    )
    parser.add_argument("--no-phonemes",
        action="store_true",
        help="Disable phoneme tokens (use raw Japanese characters instead)"
    )
    return parser.parse_args()

args = get_args()

# training hyperparameters
device = args.device
seed = args.seed
max_lr = args.lr
warmup_ratio = 0.1
cooldown_ratio = 0.1
min_lr = 0.1 * max_lr
batch_size = args.batch_size
grad_accum_steps = args.grad_accum_steps
seq_len = args.seq_len
val_freq = args.val_freq
text_factor = args.text_factor
transition_factor = args.transition_factor
attention_factor = args.attention_factor
attention_warmup_steps = args.attention_warmup_steps
embedding_lr_factor = args.embedding_lr_factor
contrastive_factor = args.contrastive_factor
max_steps = args.max_steps
betas = (0.9, 0.95)
weight_decay = 0.1
train_dataset_path = f'{args.input_dir}/train.json'
val_dataset_path = f'{args.input_dir}/val.json'
save_path = args.save_dir

def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_lr(it): # WSD schedule
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it<max_steps-cooldown_steps:
        return max_lr
    return min_lr + (max_lr-min_lr) * ((max_steps-it) / cooldown_steps)

def collate_pack(texts):
    tokens_batch = tokenizer(texts, padding=False, truncation=False)
    batch = []
    cur_sample, cur_size = [], 0
    for i in range(len(texts)):
        tokens = torch.tensor(tokens_batch['input_ids'][i], dtype=torch.long)
        cur_size += tokens.size(0)
        cur_sample.append(tokens)
        if cur_size >= seq_len + 1:
            batch.append(torch.cat(cur_sample)[: seq_len + 1])
            cur_sample, cur_size = [], 0
            if len(batch) == batch_size:
                break
    if cur_sample and not batch: # add partial sample if there isn't enough data
        batch.append(torch.cat(cur_sample + [torch.zeros(seq_len, dtype=torch.long)])[: seq_len + 1])
    if len(batch) < batch_size:
        # pad up to batch_size for consistency
        pad = batch[-1]
        while len(batch) < batch_size:
            batch.append(pad)
    batch = torch.stack(batch)
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y

def compute_attention_alignment_loss(attentions, x, y):
    """音声位置からテキスト位置へのattentionを促進する損失"""
    text_token_id = tokenizer.convert_tokens_to_ids('[TEXT]')
    start_token_id = tokenizer.convert_tokens_to_ids('[START]')

    batch_size_local, seq_len_local = x.shape
    device_local = x.device

    # 最後4層のattentionを平均（効率化）
    attn_stack = torch.stack(attentions[-4:])
    avg_attn = attn_stack.mean(dim=(0, 2))  # (batch, seq, seq)

    # テキスト位置マスク（[TEXT]〜[START]の間）
    text_mask = torch.zeros(batch_size_local, seq_len_local, dtype=torch.bool, device=device_local)
    audio_positions = torch.zeros(batch_size_local, seq_len_local, dtype=torch.bool, device=device_local)

    for b in range(batch_size_local):
        text_pos = (x[b] == text_token_id).nonzero(as_tuple=True)[0]
        start_pos = (x[b] == start_token_id).nonzero(as_tuple=True)[0]

        if len(text_pos) > 0 and len(start_pos) > 0:
            t_start = text_pos[0].item() + 1
            t_end = start_pos[0].item()
            if t_end > t_start:
                text_mask[b, t_start:t_end] = True
            audio_positions[b, start_pos[0].item() + 1:] = True

    # 音声トークンマスクと組み合わせ
    audio_token_mask = torch.logical_and(y >= 3, y <= 8003)
    audio_positions = torch.logical_and(audio_positions, audio_token_mask)

    # 各音声位置からテキスト位置への注目度を計算
    attention_to_text = (avg_attn * text_mask.unsqueeze(1).float()).sum(dim=-1)

    # 高いattentionを促進: -log(attention)
    eps = 1e-8
    log_attn = -torch.log(attention_to_text + eps)

    masked_loss = log_attn * audio_positions.float()
    num_audio = audio_positions.float().sum()

    if num_audio > 0:
        return masked_loss.sum() / num_audio
    return torch.tensor(0.0, device=device_local)

def compute_contrastive_alignment_loss(hidden_states, x, y, temperature=0.07):
    """
    テキストと音声の表現を対照学習で整列させる損失。

    各サンプルについて：
    - テキスト表現: テキスト位置の隠れ状態を平均プーリング
    - 音声表現: 音声位置の隠れ状態を平均プーリング

    正例ペア: (text_i, audio_i) 同じサンプル
    負例ペア: (text_i, audio_j) i != j
    """
    batch_size_local, seq_len_local, hidden_dim = hidden_states.shape
    device_local = hidden_states.device

    text_token_id = tokenizer.convert_tokens_to_ids('[TEXT]')
    start_token_id = tokenizer.convert_tokens_to_ids('[START]')

    text_embeddings = []
    audio_embeddings = []

    for b in range(batch_size_local):
        text_pos = (x[b] == text_token_id).nonzero(as_tuple=True)[0]
        start_pos = (x[b] == start_token_id).nonzero(as_tuple=True)[0]

        if len(text_pos) > 0 and len(start_pos) > 0:
            t_start = text_pos[0].item() + 1
            t_end = start_pos[0].item()

            # テキスト隠れ状態をプーリング
            if t_end > t_start:
                text_emb = hidden_states[b, t_start:t_end].mean(dim=0)
            else:
                text_emb = hidden_states[b, t_start]

            # 音声隠れ状態をプーリング（最初の50トークン）
            audio_start = t_end + 1
            audio_end = min(audio_start + 50, seq_len_local)
            if audio_end > audio_start:
                audio_emb = hidden_states[b, audio_start:audio_end].mean(dim=0)
            else:
                continue

            text_embeddings.append(text_emb)
            audio_embeddings.append(audio_emb)

    if len(text_embeddings) < 2:
        return torch.tensor(0.0, device=device_local)

    # 埋め込みをスタック
    text_emb = torch.stack(text_embeddings)
    audio_emb = torch.stack(audio_embeddings)

    # L2正規化
    text_emb = F.normalize(text_emb, dim=-1)
    audio_emb = F.normalize(audio_emb, dim=-1)

    # 類似度行列を計算
    logits_cont = torch.matmul(text_emb, audio_emb.T) / temperature

    # ラベル: 対角線が正解
    labels = torch.arange(len(text_embeddings), device=device_local)

    # 双方向クロスエントロピー
    loss_t2a = F.cross_entropy(logits_cont, labels)
    loss_a2t = F.cross_entropy(logits_cont.T, labels)

    return (loss_t2a + loss_a2t) / 2

def compute_loss(logits, y, x, num_steps, attentions=None):
    pred = logits.view(-1, logits.size(-1))
    labels = y.reshape(-1)
    loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')

    # 音声損失（既存）
    audio_mask = torch.logical_and(y>=3, y<=8003).view(-1)
    audio_loss = loss[audio_mask].mean()
    text_loss = loss[~audio_mask].mean()
    acc = (logits.argmax(dim=-1) == y).view(-1)[audio_mask].to(torch.float32).mean()

    # 遷移損失: [START]→最初の音声トークン
    start_token_id = tokenizer.convert_tokens_to_ids('[START]')
    batch_size, seq_len_y = y.shape

    # xの中で[START]トークンの位置を特定
    start_mask = (x == start_token_id)
    transition_indices = start_mask.nonzero(as_tuple=False)

    if len(transition_indices) > 0:
        batch_idx = transition_indices[:, 0]
        seq_idx = transition_indices[:, 1]
        # yの対応位置（[START]の次=最初の音声トークン）の損失
        transition_loss = loss.view(batch_size, seq_len_y)[batch_idx, seq_idx].mean()
    else:
        transition_loss = torch.tensor(0.0, device=y.device)

    # Attention alignment loss
    if attentions is not None:
        attention_loss = compute_attention_alignment_loss(attentions, x, y)
    else:
        attention_loss = torch.tensor(0.0, device=y.device)

    audio_loss = audio_loss / num_steps
    text_loss = text_loss / num_steps
    acc = acc / num_steps
    transition_loss = transition_loss / num_steps
    attention_loss = attention_loss / num_steps

    return audio_loss, text_loss, acc, transition_loss, attention_loss

def evaluate(val_dataloader, step=None, use_wandb=False):
    model.eval()
    val_dataloader_it = iter(val_dataloader)
    with torch.no_grad():
        val_audio_loss_accum = torch.tensor(0.0).to(device)
        val_text_loss_accum = torch.tensor(0.0).to(device)
        val_acc_accum = torch.tensor(0.0).to(device)
        val_transition_loss_accum = torch.tensor(0.0).to(device)
        val_attention_loss_accum = torch.tensor(0.0).to(device)
        val_contrastive_loss_accum = torch.tensor(0.0).to(device)
        val_loss_steps = 1
        for _ in range(val_loss_steps):
            x, y = next(val_dataloader_it)
            x, y = x.to(device), y.to(device)
            if use_autocast:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    outputs = model(x, output_attentions=True, output_hidden_states=(contrastive_factor > 0))
                    logits = outputs.logits
                    attentions = outputs.attentions
                    audio_loss, text_loss, acc, transition_loss, attention_loss = compute_loss(logits, y, x, val_loss_steps, attentions)
                    if contrastive_factor > 0:
                        hidden_states = outputs.hidden_states[-1]
                        contrastive_loss = compute_contrastive_alignment_loss(hidden_states, x, y) / val_loss_steps
                    else:
                        contrastive_loss = torch.tensor(0.0, device=device)
            else:
                outputs = model(x, output_attentions=True, output_hidden_states=(contrastive_factor > 0))
                logits = outputs.logits
                attentions = outputs.attentions
                audio_loss, text_loss, acc, transition_loss, attention_loss = compute_loss(logits, y, x, val_loss_steps, attentions)
                if contrastive_factor > 0:
                    hidden_states = outputs.hidden_states[-1]
                    contrastive_loss = compute_contrastive_alignment_loss(hidden_states, x, y) / val_loss_steps
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
            val_audio_loss_accum += audio_loss.detach()
            val_text_loss_accum += text_loss.detach()
            val_acc_accum += acc.detach()
            val_transition_loss_accum += transition_loss.detach()
            val_attention_loss_accum += attention_loss.detach()
            val_contrastive_loss_accum += contrastive_loss.detach()
        print(f"val text: {val_text_loss_accum.item():.4f} | val audio: {val_audio_loss_accum.item():.4f} | val contr: {val_contrastive_loss_accum.item():.4f} | val acc: {val_acc_accum.item():.4f}")
        if use_wandb:
            wandb.log({
                "val/text_loss": val_text_loss_accum.item(),
                "val/audio_loss": val_audio_loss_accum.item(),
                "val/transition_loss": val_transition_loss_accum.item(),
                "val/attention_loss": val_attention_loss_accum.item(),
                "val/contrastive_loss": val_contrastive_loss_accum.item(),
                "val/acc": val_acc_accum.item(),
            }, step=step)
    model.train()
    return val_audio_loss_accum.item()


tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
original_vocab_size = len(tokenizer)  # Store original vocab size before adding tokens
use_phonemes = not args.no_phonemes
added_tokens = add_japanese_tokens(tokenizer, [train_dataset_path, val_dataset_path], use_phonemes=use_phonemes)
if added_tokens:
    token_type = "phoneme" if use_phonemes else "Japanese character"
    print(f"Added {added_tokens} {token_type} tokens to tokenizer.")
if __name__ == '__main__':
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_autocast = device_type == "cuda"
    model_dtype = torch.bfloat16 if use_autocast else torch.float32
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    print(f"Save Path: {save_path}")

    # Initialize WandB
    use_wandb = args.wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "max_steps": max_steps,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "grad_accum_steps": grad_accum_steps,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "warmup_ratio": warmup_ratio,
                "cooldown_ratio": cooldown_ratio,
                "text_factor": text_factor,
                "transition_factor": transition_factor,
                "attention_factor": attention_factor,
                "attention_warmup_steps": attention_warmup_steps,
                "embedding_lr_factor": embedding_lr_factor,
                "seed": seed,
                "device": device,
                "use_phonemes": use_phonemes,
            }
        )

    # lr schedule
    warmup_steps = int(max_steps * warmup_ratio)
    cooldown_steps = int(max_steps * cooldown_ratio)

    # model
    model = AutoModelForCausalLM.from_pretrained('ekwek/Soprano-80M')
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    # Enable eager attention for output_attentions support
    if attention_factor > 0:
        model.set_attn_implementation('eager')
    model.to(model_dtype).to(device)
    model.train()

    # dataset
    dataset = AudioDataset(train_dataset_path)
    # we need batch_size * 16 to have enough tokens after packing
    dataloader = DataLoader(dataset,
        batch_size=batch_size * 16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    dataloader_it = iter(dataloader)
    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(val_dataset,
        batch_size=batch_size * 16,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), max_lr, betas=betas, weight_decay=weight_decay, fused=True)

    # Gradient hook to scale gradients for new Japanese token embeddings
    if added_tokens > 0 and embedding_lr_factor != 1.0:
        print(f"Applying {embedding_lr_factor}x learning rate to {added_tokens} new token embeddings (indices {original_vocab_size} to {len(tokenizer)-1})")

        def scale_embedding_grad(grad):
            """Scale gradients for new token indices by embedding_lr_factor"""
            scaled_grad = grad.clone()
            scaled_grad[original_vocab_size:] *= embedding_lr_factor
            return scaled_grad

        # Register hook on input embeddings (Qwen3 uses model.model.embed_tokens)
        model.model.embed_tokens.weight.register_hook(scale_embedding_grad)
        # Also register on output layer (lm_head) if it exists and is separate
        if hasattr(model, 'lm_head') and model.lm_head.weight is not model.model.embed_tokens.weight:
            model.lm_head.weight.register_hook(scale_embedding_grad)

    # Best model tracking
    best_val_loss = float('inf')
    best_step = 0
    best_save_path = save_path / "best"
    best_save_path.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(0, max_steps), ncols=200, dynamic_ncols=True)
    for step in pbar:
        start = time.time()
        if val_freq>0 and (step % val_freq == 0 or step==max_steps-1):
            val_loss = evaluate(val_dataloader, step=step, use_wandb=use_wandb)
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                print(f"New best model at step {step} with val_loss={val_loss:.4f}. Saving...")
                model.save_pretrained(best_save_path)
                tokenizer.save_pretrained(best_save_path)

        opt.zero_grad()
        audio_loss_accum = 0.0
        text_loss_accum = 0.0
        acc_accum = 0.0
        transition_loss_accum = 0.0
        attention_loss_accum = 0.0
        contrastive_loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(dataloader_it)
            except:
                dataloader_it = iter(dataloader)
                x, y = next(dataloader_it)
            x, y = x.to(device), y.to(device)

            if use_autocast:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    outputs = model(x, output_attentions=True, output_hidden_states=(contrastive_factor > 0))
                    logits = outputs.logits
                    attentions = outputs.attentions
                    audio_loss, text_loss, acc, transition_loss, attention_loss = compute_loss(logits, y, x, grad_accum_steps, attentions)
                    if contrastive_factor > 0:
                        hidden_states = outputs.hidden_states[-1]
                        contrastive_loss = compute_contrastive_alignment_loss(hidden_states, x, y) / grad_accum_steps
                    else:
                        contrastive_loss = torch.tensor(0.0, device=device)
            else:
                outputs = model(x, output_attentions=True, output_hidden_states=(contrastive_factor > 0))
                logits = outputs.logits
                attentions = outputs.attentions
                audio_loss, text_loss, acc, transition_loss, attention_loss = compute_loss(logits, y, x, grad_accum_steps, attentions)
                if contrastive_factor > 0:
                    hidden_states = outputs.hidden_states[-1]
                    contrastive_loss = compute_contrastive_alignment_loss(hidden_states, x, y) / grad_accum_steps
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
            audio_loss_accum += audio_loss.detach()
            text_loss_accum += text_loss.detach()
            acc_accum += acc.detach()
            transition_loss_accum += transition_loss.detach()
            attention_loss_accum += attention_loss.detach()
            contrastive_loss_accum += contrastive_loss.detach()

            # Total loss with attention warmup
            if step >= attention_warmup_steps:
                total_loss = audio_loss + text_factor*text_loss + transition_factor*transition_loss + attention_factor*attention_loss + contrastive_factor*contrastive_loss
            else:
                total_loss = audio_loss + text_factor*text_loss + transition_factor*transition_loss + contrastive_factor*contrastive_loss
            total_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        opt.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        total_tokens = step * batch_size*seq_len*grad_accum_steps
        end = time.time()
        dt = (end-start)*1000
        tokens_per_second = (batch_size*seq_len*grad_accum_steps) / (end-start)
        tqdm_log = f'text: {text_loss_accum.item():.3f} | audio: {audio_loss_accum.item():.3f} | contr: {contrastive_loss_accum.item():.3f} | acc: {acc_accum.item():.4f} | lr: {lr:.2e} | {dt:.0f}ms'
        pbar.set_description(tqdm_log)

        if use_wandb:
            wandb.log({
                "train/text_loss": text_loss_accum.item(),
                "train/audio_loss": audio_loss_accum.item(),
                "train/transition_loss": transition_loss_accum.item(),
                "train/attention_loss": attention_loss_accum.item(),
                "train/contrastive_loss": contrastive_loss_accum.item(),
                "train/acc": acc_accum.item(),
                "train/lr": lr,
                "train/grad_norm": norm.item(),
                "train/tokens_per_second": tokens_per_second,
            }, step=step)

    print(f"Training complete. Saving final model at {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Saving done.")
    print(f"Best model saved at step {best_step} with val_loss={best_val_loss:.4f}")
    print(f"Best model path: {best_save_path}")

    if use_wandb:
        wandb.finish()
