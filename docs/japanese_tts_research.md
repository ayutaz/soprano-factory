# Soprano日本語TTS調査・訓練結果まとめ

## 概要

Soprano-80M（英語TTSモデル）を日本語に対応させるためのファインチューニングと調査結果をまとめた文書です。

---

## 1. プロジェクト構成

### Sopranoアーキテクチャ

```
[テキスト入力] → [トークナイザー] → [LLM (Soprano-80M)] → [音声トークン] → [デコーダー] → [音声波形]
                                         ↑                               ↑
                                    訓練対象                        英語用（問題箇所）
```

### 主要コンポーネント

| コンポーネント | ファイル | 説明 |
|---------------|---------|------|
| エンコーダー | `encoder/codec.py` | ConvNeXtベースのVocosBackboneで音声特徴抽出 |
| 量子化 | `encoder/quantizer.py` | FSQ（有限スカラー量子化）でトークン生成 |
| デコーダー | `decoder.pth` | VocosBackbone + ISTFTHeadで音声再構成 |
| LLM | `model.safetensors` | テキストから音声トークンを生成 |

### デコーダー構造（SopranoDecoder）

```python
class SopranoDecoder(nn.Module):
    # パラメータ
    num_input_channels = 512      # 入力チャンネル数
    decoder_num_layers = 8        # レイヤー数
    decoder_dim = 768             # 次元数（HF版は512）
    hop_length = 512              # ホップ長
    n_fft = 2048                  # FFTサイズ
    upscale = 4                   # アップスケール係数

    # アーキテクチャ
    VocosBackbone → ISTFTHead → 音声波形
```

### VocosDecoder（train_decoder_v2/gan用）

```python
class VocosDecoder(nn.Module):
    # パラメータ
    n_mels = 100                  # 入力メルスペクトログラム次元
    dim = 512                     # 隠れ次元
    intermediate_dim = 1536       # 中間次元
    num_layers = 8                # ConvNeXtブロック数
    hop_length = 512              # ホップ長
    n_fft = 2048                  # FFTサイズ

    # アーキテクチャ
    MelSpec → VocosBackbone(ConvNeXt×8) → ISTFTHead → 音声波形
```

---

## 2. 日本語ファインチューニング結果

### 使用データセット

| データセット | サンプル数 | 特徴 |
|-------------|-----------|------|
| MoeSpeech | 約60,000 | アニメ風女性音声、高品質 |
| JSUT | 約7,000 | 女性音声、標準的な日本語 |
| JVS | 約30,000 | 複数話者、研究用 |

### 訓練実験結果

| バージョン | 設定 | val_loss | val_acc | 備考 |
|-----------|------|----------|---------|------|
| v1 | max-steps=5000, text-factor=0 | 5.16 | ~8-10% | 初期実験 |
| v2 | max-steps=7000, text-factor=0 | 5.09 | ~11% | やや改善 |
| v3 | text-factor=0.5 | - | - | テキスト損失追加 |
| v4 | text-factor=1.0 | - | - | さらにテキスト重視 |
| v5 | 最新 | - | - | 継続実験 |

### 訓練コマンド例

```bash
# 基本訓練
python train.py --input-dir moespeech_dataset --save-dir weights_moespeech_v5 \
    --max-steps 10000 --batch-size 4 --wandb

# テキスト損失を含める
python train.py --input-dir moespeech_dataset --save-dir weights_moespeech_v5 \
    --max-steps 10000 --text-factor 0.5 --wandb
```

---

## 3. 問題点の分析

### 主要な問題: 滑舌の悪さ

日本語音声の滑舌が悪い原因を調査した結果：

1. **デコーダーが英語用**
   - `decoder.pth`は英語音声データで訓練済み
   - 日本語の音素体系（モーラ、促音、長音等）に最適化されていない

2. **LLMファインチューニングの限界**
   - val_acc ~11%は低い（英語では通常30-50%以上）
   - Soprano-80Mは英語のテキスト-音声対応を学習済み

3. **トークナイザーの拡張**
   - 日本語文字（ひらがな、カタカナ、漢字）を語彙に追加
   - 約2,800トークン追加（10868 - 8000）
   - 新規トークンの埋め込みは未学習

### 検証精度が低い理由

```
英語モデル:  テキスト → 音声トークン（学習済みマッピング）
日本語追加: テキスト → ???（新規マッピングが必要）
```

---

## 4. 日本語デコーダー訓練

### デコーダー訓練スクリプト比較

| スクリプト | 入力 | 損失関数 | 音質 | 速度 |
|-----------|------|----------|------|------|
| `train_decoder.py` | FSQコード (5次元) | Mel損失のみ | 低（機械音） | 75 it/s |
| `train_decoder_v2.py` | メルスペクトログラム (100次元) | Mel損失のみ | 中（やや機械音） | 80 it/s |
| `train_decoder_gan.py` | メルスペクトログラム (100次元) | Mel + GAN + Feature Matching | 高（自然） | 5 it/s |

### 訓練結果

| バージョン | 最終 val_loss | 備考 |
|-----------|---------------|------|
| train_decoder.py (100k steps) | 1.0982 | FSQからの情報損失が大きい |
| train_decoder_v2.py (50k steps) | 0.1650 | Melベースで大幅改善 |
| train_decoder_gan.py (100k steps) | TBD | GAN訓練進行中 |

### 推奨: GAN訓練 (`train_decoder_gan.py`)

**訓練手順:**

1. ファイルリスト生成
   ```bash
   uv run python tools/prepare_decoder_training.py --input-dir path/to/dataset
   ```

2. GAN訓練実行
   ```bash
   uv run python train_decoder_gan.py \
       --train-filelist path/to/train_decoder.txt \
       --val-filelist path/to/val_decoder.txt \
       --save-dir weights_decoder_gan \
       --max-steps 100000
   ```

3. テスト
   ```bash
   uv run python test_decoder_v2.py \
       --input path/to/audio.wav \
       --decoder-path weights_decoder_gan/decoder.pth \
       --output test_output.wav
   ```

### GAN訓練の構成

- **Generator**: VocosDecoder (VocosBackbone + ISTFTHead)
- **Discriminators**:
  - MultiPeriodDiscriminator (5つの周期: 2, 3, 5, 7, 11)
  - MultiResolutionDiscriminator (3つのFFTサイズ: 2048, 1024, 512)
- **損失関数**:
  - Mel再構成損失 (係数: 45.0)
  - Generator損失 (Hinge loss)
  - Feature Matching損失 (係数: 2.0)
  - Discriminator損失 (Hinge loss)

---

## 5. 技術的詳細

### 音声トークン形式

```
[STOP][TEXT]<テキストプロンプト>[START][token1][token2]...[STOP]
```

### トークナイザー拡張

```python
# tokenizer_utils.py
def collect_japanese_chars(texts):
    """日本語文字を収集"""
    chars = set()
    for text in texts:
        for char in text:
            if is_japanese_char(char):
                chars.add(char)
    return sorted(chars)

def add_japanese_tokens(tokenizer, chars):
    """トークナイザーに日本語トークンを追加"""
    tokenizer.add_tokens(list(chars))
    return len(chars)
```

### デコーダーの内部構造

```
入力: [batch, 512, seq_len]  # 音声トークン埋め込み
  ↓
線形補間（4倍アップサンプル）
  ↓
VocosBackbone (8層ConvNeXt)
  - dim: 768
  - intermediate_dim: 2304
  ↓
ISTFTHead
  - n_fft: 2048
  - hop_length: 512
  ↓
出力: [batch, audio_samples]  # 32kHz音声
```

---

## 6. 必要リソース

| 項目 | 推定値 |
|------|--------|
| GPU | RTX 4090 (24GB VRAM) - 十分 |
| 日本語音声データ | 10-100時間 |
| ディスク容量 | 50GB以上 |
| 訓練時間 | 数時間〜数日 |

---

## 7. 今後の課題

### 完了

1. ✅ **日本語デコーダーの訓練環境構築**
   - `train_decoder.py` - FSQベース（非推奨）
   - `train_decoder_v2.py` - Melベース
   - `train_decoder_gan.py` - GAN訓練（推奨）

2. ✅ **MoeSpeechでの訓練実行**
   - train_decoder.py: 100k steps, val_loss 1.0982
   - train_decoder_v2.py: 50k steps, val_loss 0.1650
   - train_decoder_gan.py: 進行中

### 優先度: 高

3. **GAN訓練の完了と評価**
   - 100kステップ訓練の完了
   - 音声品質の主観評価

4. **Sopranoへの統合**
   - 訓練済みデコーダーをSopranoパイプラインに統合
   - 推論スクリプトの更新

### 優先度: 中

5. **ハイパーパラメータ調整**
   - バッチサイズ、学習率の最適化
   - Mel損失係数、Feature Matching係数の調整

6. **より長い訓練**
   - 200k-500kステップでの品質向上検証

### 優先度: 低

7. **代替アプローチの検討**
   - VITS等の日本語TTSモデル
   - より大規模な日本語データセット

---

## 8. 参考リンク

- [Soprano](https://github.com/ekwek1/soprano) - 元のTTSモデル
- [Soprano-Factory](https://github.com/ekwek1/soprano-factory) - 訓練フレームワーク
- [Vocos](https://github.com/gemelo-ai/vocos) - デコーダーのベースアーキテクチャ
- [MoeSpeech](https://huggingface.co/datasets/litagin/moespeech) - 日本語音声データセット
- [JSUT](https://sites.google.com/site/shinaborutakahashi/jsut) - 日本語音声コーパス
- [JVS](https://sites.google.com/site/shinaborutakahashi/jvs) - 日本語多話者コーパス

---

## 9. WandBログ

訓練の進捗はWandBで確認可能：
- プロジェクト: `soprano-factory`
- 実行例: `https://wandb.ai/yousan/soprano-factory/runs/9a184mrs`

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-01-17 | 初版作成、調査結果まとめ |
| 2026-01-17 | train_decoder.py修正（num_workers=0、wandbデフォルト有効） |
| 2026-01-17 | train_decoder.py 100kステップ訓練完了（val_loss: 1.0982） |
| 2026-01-17 | train_decoder_v2.py作成（Melベース、情報損失削減） |
| 2026-01-17 | train_decoder_v2.py 50kステップ訓練完了（val_loss: 0.1650） |
| 2026-01-17 | train_decoder_gan.py作成（GAN訓練、高品質音声生成） |
| 2026-01-17 | テストスクリプト作成（test_decoder_v2.py, test_decoder_inference.py） |
| 2026-01-17 | ドキュメント更新（CLAUDE.md, japanese_tts_research.md） |
