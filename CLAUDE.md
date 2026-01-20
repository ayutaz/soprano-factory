# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Soprano-Factoryは、Soprano音声合成(TTS)モデルをファインチューニングするための軽量（約600行）トレーニングフレームワークです。カスタムデータセットを使用して、ユーザー自身のハードウェアでTTSモデルを訓練できます。

## よく使うコマンド

### データセット生成
```bash
python generate_dataset.py --input-dir path/to/dataset
```
LJSpeech形式のデータセットから`train.json`と`val.json`を生成します。
デフォルトで音素変換が有効（`--no-phonemes`で無効化可能）。

音素変換の利点:
- 語彙サイズ: 2,676文字 → 約1,000トークン
- アクセント情報を含むため韻律生成が改善
- 未知漢字も読み仮名経由で処理可能

### モデル訓練
```bash
python train.py --input-dir path/to/dataset --save-dir path/to/weights
```

主な訓練引数:
- `--max-steps` - 訓練ステップ数（デフォルト: 50000）
- `--batch-size` - バッチサイズ（デフォルト: 4）
- `--seq-len` - シーケンス長（デフォルト: 1024）
- `--val-freq` - 検証頻度（デフォルト: 250、0で無効化）
- `--text-factor` - テキストトークン損失の重み（デフォルト: 0.5）
- `--transition-factor` - テキスト→音声遷移損失の重み（デフォルト: 1.0）
- `--contrastive-factor` - Contrastive損失の重み（デフォルト: 0.0）
- `--lr` - 学習率（デフォルト: 1e-4）
- `--no-phonemes` - 音素変換を無効化（文字ベースに戻す）
- `--wandb` - WandBログを有効化
- `--wandb-project` - WandBプロジェクト名（デフォルト: soprano-factory）
- `--wandb-run-name` - WandB実行名（オプション）

### テスト実行
```bash
uv run pytest tests/
uv run pytest tests/test_tokenizer_utils.py -v  # 単一ファイル実行
```

### デコーダー訓練（日本語音質改善用）

1. ファイルリスト生成:
```bash
uv run python tools/prepare_decoder_training.py --input-dir path/to/dataset
```

2. デコーダー訓練（3つのバージョンあり）:

#### train_decoder.py（FSQベース、非推奨）
FSQコード（5次元）からデコードするため情報損失が大きい。
```bash
uv run python train_decoder.py \
    --train-filelist path/to/train_decoder.txt \
    --val-filelist path/to/val_decoder.txt \
    --save-dir path/to/weights
```

#### train_decoder_v2.py（Melベース）
メルスペクトログラム（100次元）から直接デコード。情報損失が少ない。
```bash
uv run python train_decoder_v2.py \
    --train-filelist path/to/train_decoder.txt \
    --val-filelist path/to/val_decoder.txt \
    --save-dir path/to/weights \
    --max-steps 50000
```

#### train_decoder_gan.py（GAN訓練、推奨）
Discriminatorを使用した敵対的訓練。最も自然な音声を生成。
```bash
uv run python train_decoder_gan.py \
    --train-filelist path/to/train_decoder.txt \
    --val-filelist path/to/val_decoder.txt \
    --save-dir path/to/weights \
    --max-steps 100000
```

主な訓練引数:
- `--max-steps` - 訓練ステップ数（デフォルト: 100000）
- `--batch-size` - バッチサイズ（デフォルト: 8）
- `--learning-rate` - 学習率（デフォルト: 1e-4）
- `--val-freq` - 検証頻度（デフォルト: 1000）
- `--save-freq` - チェックポイント保存頻度（デフォルト: 5000）
- `--no-wandb` - WandBログを無効化（デフォルトは有効）
- `--mel-loss-coeff` - メル損失係数（GAN版のみ、デフォルト: 45.0）

### Soprano統合推論（GAN強化）
訓練したGANデコーダーでSopranoの音声品質を向上させる：
```bash
uv run python inference_gan.py "テキスト" \
    -m weights_moespeech_v5 \
    -g weights_decoder_gan/decoder.pth \
    -o output.wav
```

引数:
- `-m` - ファインチューニング済みSopranoモデルのパス
- `-g` - 訓練済みGANデコーダーのパス
- `-o` - 出力ファイルパス
- `--save-original` - オリジナル音声も保存（比較用）
- `--no-enhance` - GAN強化をスキップ

### JVSデータセット変換
```bash
python -m tools.jvs_to_ljspeech --jvs-root path/to/jvs --output-dir output
```

## アーキテクチャ

### データフロー
1. **入力**: LJSpeech形式（`wavs/`ディレクトリ + `metadata.txt`）
2. **前処理** (`generate_dataset.py`): 音声を32kHzにリサンプル → Sopranoエンコーダでトークン化
3. **訓練** (`train.py`): HuggingFaceからSoprano-80Mをロード → ファインチューニング
4. **出力**: 訓練済みモデルとトークナイザー

### 主要コンポーネント
- **encoder/codec.py**: ConvNeXtベースのVocosBackboneで音声特徴抽出
- **encoder/quantizer.py**: FSQ（有限スカラー量子化）でトークン生成
- **tokenizer_utils.py**: 日本語文字/音素トークンの語彙拡張
- **phoneme_utils.py**: pyopenjtalkによる日本語音素変換とアクセント情報抽出
- **train_decoder.py**: デコーダー訓練（FSQベース、非推奨）
- **train_decoder_v2.py**: デコーダー訓練（Melベース）
- **train_decoder_gan.py**: デコーダー訓練（GAN、推奨）
- **train_decoder_e2e.py**: エンドツーエンドデコーダー訓練（512-dim入力対応）
- **inference_gan.py**: GAN強化推論スクリプト
- **tools/jvs_to_ljspeech.py**: JVSコーパスをLJSpeech形式に変換
- **tools/prepare_decoder_training.py**: デコーダー訓練用ファイルリスト生成

### 音声トークン形式
```
[STOP][TEXT]<テキストプロンプト>[START][token1][token2]...[STOP]
```

## 日本語サポート

### 音素ベース（デフォルト）
pyopenjtalk-plusによる音素変換がデフォルトで有効:

```
日本語テキスト → フルコンテキストラベル → 音素+アクセントトークン
例: "こんにちは" → "k_-4 o_-4 N_-3 n_-2 i_-2 ch_-1 i_-1 w_0 a_0"
```

利点:
- 語彙サイズ削減: 2,676文字 → 約1,000トークン
- アクセント情報（A:フィールド）による韻律生成改善
- 未知漢字も読み仮名経由で処理可能

### 文字ベース（従来方式）
`--no-phonemes`フラグで音素変換を無効化し、日本語文字をそのまま使用。
訓練時にデータセット内の日本語文字（ひらがな、カタカナ、漢字等）を自動検出し、トークナイザーの語彙に追加します。

### 推論
```bash
uv run python inference.py "こんにちは" \
    -m weights/best \
    -o output.wav
```
音素変換はデフォルトで有効。文字ベースモデルには`--no-phonemes`を指定。

## 入力データ形式

LJSpeech形式:
```
dataset_dir/
├── metadata.txt    # 形式: filename|text
└── wavs/
    ├── sample1.wav
    └── ...
```

## ドキュメント

- `docs/japanese_tts_research.md` - 日本語TTS調査・訓練結果まとめ
