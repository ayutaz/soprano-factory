# Soprano日本語TTS調査・訓練結果まとめ

## 概要

Soprano-80M（英語TTSモデル）を日本語に対応させるためのファインチューニングと調査結果をまとめた文書です。

---

## 1. プロジェクト構成

### Sopranoアーキテクチャ

```
[テキスト入力] → [トークナイザー] → [LLM (Soprano-80M)] → [音声トークン] → [デコーダー] → [音声波形]
                                         ↑                               ↑
                                  ★真のボトルネック★                  訓練済み（問題なし）
```

**重要な発見（2026-01-18）**: 当初デコーダーが問題と考えていたが、実際の問題は**LLMが日本語の音声トークンを正しく生成できていない**ことが判明。デコーダーは正常に動作している。

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

### 主要な問題: LLMが日本語を学習できていない

**結論（2026-01-18確定）**: 問題はデコーダーではなく**LLM**にある。

### 根本原因1: text_factor=0.0（最重要）★

**場所**: `train.py:77, 317`

```python
# デフォルト設定（train.py:77）
parser.add_argument("--text-factor", default=0.0, ...)

# 損失計算（train.py:317）
total_loss = audio_loss + text_factor * text_loss
# text_factor=0.0 の場合、text_loss が完全に無視される！
```

**トークンID範囲**:
- 音声トークン: ID 4-8003（`[0]`〜`[7999]`）
- 英語テキスト: ID 8004-8191
- 日本語テキスト: ID 8192-10867（2,676文字）

**問題**: 日本語トークンは `audio_mask` の範囲外（3-8003）なので `text_loss` に含まれる。しかし `text_factor=0.0` のため、**日本語テキストに対する勾配が一切流れていない**。

### 根本原因2: 訓練ステップ不足

- デフォルト: 10,000ステップ
- データセット: ~60,000サンプル
- 2,676の新規埋め込みを学習するには不十分

### 根本原因3: 新規トークン埋め込みの初期化

- 2,676の日本語文字がランダム初期化
- HuggingFaceの「mean resizing」で初期化されるが、std=0.022（元の半分）
- 英語モデルからの転移学習が効かない

### 検証精度が低い理由（詳細）

```
英語モデル:  テキスト → 音声トークン（学習済みマッピング）
日本語追加: テキスト → 勾配なし（text_factor=0） → 学習されない
```

| 要因 | 深刻度 | 影響 |
|------|--------|------|
| text_factor=0.0 | ★★★ 致命的 | テキスト理解の最適化信号なし |
| 訓練ステップ不足 | ★★ 高 | 新規埋め込み学習に不十分 |
| 埋め込み初期化 | ★ 中 | 収束が遅い |
| 言語ミスマッチ | ★ 中 | 英語→日本語の根本的課題 |

---

## 4. 日本語デコーダー訓練

### デコーダー訓練スクリプト比較

| スクリプト | 入力 | 損失関数 | 音質 | 速度 |
|-----------|------|----------|------|------|
| `train_decoder.py` | FSQコード (5次元) | Mel損失のみ | 低（機械音） | 75 it/s |
| `train_decoder_v2.py` | メルスペクトログラム (100次元) | Mel損失のみ | 中（やや機械音） | 80 it/s |
| `train_decoder_gan.py` | メルスペクトログラム (100次元) | Mel + GAN + Feature Matching | 高（自然） | 5 it/s |

### 訓練結果

| バージョン | 最終 val_loss | 音質 | 備考 |
|-----------|---------------|------|------|
| train_decoder.py (100k steps) | 1.0982 | 低（機械音） | FSQからの情報損失が大きい |
| train_decoder_v2.py (50k steps) | 0.1650 | 中（やや機械音） | Melベースで大幅改善 |
| **train_decoder_gan.py (100k steps)** | **0.2593** | **高（自然）** | **GAN訓練で最高品質** |

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

### Soprano統合

訓練したGANデコーダーをSopranoに統合する方法：

**課題**: 次元の不一致
- Soprano decoder: 512-dim LLM hidden states → 音声
- GAN decoder: 100-dim メルスペクトログラム → 音声

**解決策A: Post-processing approach（推奨、即使用可）**
```
Soprano TTS → 音声 → メルスペクトログラム → GAN Decoder → 高品質音声
```

使用方法:
```bash
uv run python inference_gan.py "こんにちは" \
    -m weights_moespeech_v5 \
    -g weights_decoder_gan/decoder.pth \
    -o output.wav
```

**解決策B: End-to-end approach（追加訓練必要）**
```
LLM hidden states (512-dim) → Projection → GAN Decoder → 音声
```

訓練方法:
```bash
uv run python train_decoder_e2e.py \
    --train-filelist path/to/train_decoder.txt \
    --val-filelist path/to/val_decoder.txt \
    --save-dir weights_decoder_e2e \
    --max-steps 100000
```

### 統合テスト結果（2026-01-18）

**テスト内容**: `inference_gan.py` を使用してSoprano + GANデコーダーで日本語音声生成をテスト

**結果**: ❌ 失敗
- 生成された音声は意味不明な音の羅列
- オリジナルSoprano出力もGAN強化後も同様に意味不明
- 「hallucination」警告が発生

**原因分析**:
```
期待: テキスト → LLM → 正しい音声トークン → デコーダー → 日本語音声
実際: テキスト → LLM → 間違った音声トークン → デコーダー → 意味不明な音声
                  ↑
              ここが問題
```

**結論**:
- デコーダー（Vocos）は正常に動作している（音声→メル→音声の再構成は高品質）
- 問題はLLMが日本語テキスト→音声トークンのマッピングを学習できていないこと
- val_acc ~11% は非常に低く、LLMの学習が不十分

**デコーダー訓練の評価**:
| 観点 | 評価 |
|------|------|
| ボコーダーとしての性能 | ✅ 高品質（音声再構成は成功） |
| 日本語TTS改善への貢献 | ⚠️ LLMが修正されるまで効果なし |
| 将来的な価値 | ✅ LLM修正後に音質向上に貢献予定 |

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
   - train_decoder_gan.py: 100k steps, val_loss 0.2593 ✅

3. ✅ **GAN訓練の完了と評価**
   - 100kステップ訓練完了（val_loss: 0.2593）
   - 音声品質: 自然な音声を生成（ボコーダーとして）

4. ✅ **Sopranoへの統合調査**
   - `inference_gan.py` - GAN強化推論スクリプト作成
   - `train_decoder_e2e.py` - 512-dim入力対応のE2Eデコーダー訓練スクリプト作成
   - **結果**: LLMがボトルネックと判明、デコーダー改善では解決しない

### 優先度: 最高 ⚠️

5. **LLMの日本語対応改善**（未着手・最重要課題）
   - 現状: val_acc ~11%（非常に低い）
   - 目標: val_acc 30%以上
   - 選択肢:
     - A. より長い訓練（50k-100kステップ）
     - B. より大きな学習率
     - C. より多くの訓練データ
     - D. 音素(phoneme)ベースのアプローチ
     - E. 別のTTSモデル（VITS、Style-Bert-VITS2等）

### 優先度: 中（LLM修正後）

6. **デコーダーのハイパーパラメータ調整**
   - バッチサイズ、学習率の最適化
   - Mel損失係数、Feature Matching係数の調整
   - ※LLMが修正されてから意味がある

7. **より長いデコーダー訓練**
   - 200k-500kステップでの品質向上検証
   - ※LLMが修正されてから意味がある

### 優先度: 低

8. **代替アプローチの検討**
   - VITS等の日本語TTSモデルへの移行
   - Sopranoベースのアプローチを諦める選択肢

---

## 8. LLM改善プラン（v6）

### Option A 実験結果 ❌ 失敗

**実行した訓練**:
```bash
uv run python train.py \
    --input-dir moespeech_dataset \
    --save-dir weights_moespeech_v6 \
    --max-steps 50000 \
    --batch-size 4 \
    --text-factor 0.5 \
    --lr 1e-4 \
    --wandb \
    --wandb-run-name "v6-textfactor-05"
```

**結果**:
| 指標 | 訓練前 | 訓練後 | 期待値 |
|------|--------|--------|--------|
| val_acc | 11% | 10.3% | 20-30% |
| 音声品質 | 意味不明 | 意味不明 | 聞き取れる |

**結論**: text_factor を有効にしても改善しなかった。より深い問題がある。

---

### 根本原因（深層分析）★重要★

#### アーキテクチャの構造的欠陥

**問題**: モデルは「音声→音声」の予測を学習しており、「テキスト→音声」のマッピングを学習していない。

```
訓練シーケンス:
[TEXT]日本語テキスト[START][audio1][audio2][audio3]...[STOP]
  ↓
損失計算:
  audio_mask = (token_id >= 3) AND (token_id <= 8003)
  audio_loss = CrossEntropy(model_output, target)[audio_mask].mean()
  ↓
実際に学習される遷移:
  [audio1] → [audio2] ✅ 学習される
  [audio2] → [audio3] ✅ 学習される
  ...
  ↓
決定的な問題:
  [START] → [audio1] の遷移
  ↑
  この遷移の入力コンテキストに日本語テキストが含まれるが、
  損失関数は「audio_mask」された位置のみを見るため、
  テキスト→音声の因果関係が明示的に訓練されない
```

**train.py:158-169 の問題コード**:
```python
audio_mask = torch.logical_and(y>=3, y<=8003).view(-1)
audio_loss = loss[audio_mask].mean()  # 音声位置のみの損失
text_loss = loss[~audio_mask].mean()  # テキスト位置のみの損失
# 問題: テキスト→音声の遷移を測る指標がない
acc = (logits.argmax(dim=-1) == y).view(-1)[audio_mask].to(torch.float32).mean()
# ↑ 精度もaudio_maskのみ = 音声→音声の予測精度
```

**text_factor=0.5 が効かない理由**:
- `text_loss` はテキスト位置での次トークン予測損失（テキスト→テキスト）
- 日本語テキストの予測精度が上がっても、音声生成能力は向上しない
- 必要なのは「テキストコンテキスト→音声トークン」の遷移損失

**val_acc が常に低い理由**:
- `acc` は `audio_mask` 位置のみを測定
- 実質的に「音声→音声」の予測精度
- テキスト理解能力を反映していない

---

### 推奨アプローチ（更新版）

#### Option B: 音素ベースアプローチ ★推奨★

```
現状:   日本語テキスト → [2,676新規トークン] → LLM → 音声
改善:   日本語テキスト → pyopenjtalk → [~50音素] → LLM → 音声
```

**利点**:
- 語彙サイズを大幅削減（2,676 → ~50）
- 成功実績あり（VITS, GPT-SoVITS, Style-Bert-VITS2で使用）
- 既存コードへの変更が限定的

**必要な変更**:
| ファイル | 変更内容 |
|---------|---------|
| `pyproject.toml` | pyopenjtalk 依存追加 |
| `phoneme_utils.py` | 新規作成（音素変換ユーティリティ） |
| `tokenizer_utils.py` | 音素トークン追加ロジック |
| `dataset.py` | 音素前処理パイプライン |
| `generate_dataset.py` | 音素変換統合 |

**期待効果**: val_acc 30-50%

#### Option C: 遷移損失の追加

**概念**: テキスト→音声遷移を明示的に訓練する損失関数を追加

```python
# 提案: 遷移損失の追加
start_token_pos = find_start_token_position(sequence)
transition_loss = CrossEntropy(
    logits[start_token_pos],  # [START]位置での予測
    target[start_token_pos]   # 最初の音声トークン
)
total_loss = audio_loss + text_factor*text_loss + transition_factor*transition_loss
```

**必要な変更**:
| ファイル | 変更内容 |
|---------|---------|
| `train.py` | 遷移損失の計算と適用 |

**期待効果**: val_acc 20-35%
**リスク**: 効果は限定的な可能性

#### Option D: 代替モデル（Style-Bert-VITS2等）

- 日本語ネイティブのTTSモデルに完全移行
- Sopranoアーキテクチャを放棄
- 期待効果: 高品質な日本語TTS

---

## 13. Contrastive Learning アプローチ（最新）

### 背景と失敗分析

これまで試行したすべてのアプローチ（Option A-F）は val_acc ~10-12% で失敗：

| Option | アプローチ | 失敗理由 |
|--------|-----------|---------|
| A | text_factor=0.5 | text→text予測を改善、text→audio改善なし |
| B | フォネームトークン | 語彙削減したが同じアーキテクチャ問題 |
| C | 遷移損失 | 1位置 vs 200音声位置、圧倒的に弱い信号 |
| E | Attention損失 | 95%テキスト注目達成、しかし何を生成すべきかの勾配なし |
| F | Embedding 10x LR | 収束は速いが同じ弱い信号問題 |

### 根本原因（調査で特定）

1. **損失関数構造**: audio→audio予測を最適化、text→audioは最適化されない
2. **勾配比率問題**: 音声損失が遷移損失を8000:1で圧倒
3. **埋め込み初期化**: 新トークンはstd=0.022（元の半分）

### 解決策：Contrastive Text-Audio Alignment Loss

**理論的根拠**（MM-TTS, CLAP, HiStyle研究より）：

1. **Contrastive Loss**: テキスト表現と対応する音声表現を明示的に近づける
2. **共有意味空間**: 日本語テキスト埋め込みが、生成すべき音声と同じ領域を占めるよう学習
3. **Attention Lossとの違い**: Attentionは「どこを見るか」、Contrastiveは「何にマップするか」を教える

### 実装詳細

**新規引数**:
```bash
--contrastive-factor 0.5  # Contrastive損失の重み
```

**損失関数** (`train.py:238-295`):
```python
def compute_contrastive_alignment_loss(hidden_states, x, y, temperature=0.07):
    """
    テキストと音声の表現を対照学習で整列させる損失。

    各サンプルについて：
    - テキスト表現: テキスト位置の隠れ状態を平均プーリング
    - 音声表現: 音声位置の隠れ状態を平均プーリング

    正例ペア: (text_i, audio_i) 同じサンプル
    負例ペア: (text_i, audio_j) i != j
    """
    # L2正規化 → 類似度行列 → 双方向クロスエントロピー
    ...
```

### 訓練コマンド

```bash
# MoeSpeechで検証（50Kステップ）
uv run python train.py \
    --input-dir moespeech_dataset \
    --save-dir weights_v11_contrastive_moespeech \
    --max-steps 50000 \
    --batch-size 8 \
    --text-factor 0.5 \
    --transition-factor 5.0 \
    --contrastive-factor 0.5 \
    --attention-factor 0.0 \
    --wandb \
    --wandb-run-name "v11-contrastive-moespeech"
```

### 期待される改善

| メトリクス | 現在 | 期待値 |
|-----------|------|--------|
| val_acc | 10-12% | 25-40% |
| Text-Audio整列 | ランダム | 構造化 |
| 最初の音声トークン精度 | ~5% | 20-35% |

### WandB監視指標

- `train/contrastive_loss`: 減少することを確認
- `val/contrastive_loss`: 検証用Contrastive損失
- `val_acc`: 15%以上への上昇を期待

---

### 実行順序の推奨

1. **Option B（音素ベース）** を最初に試す
   - 最も実績のあるアプローチ
   - 語彙削減による学習効率向上
   - 2-3日で実装可能

2. Option Bで不十分な場合、Option Cを追加

3. それでも不十分な場合、Option Dを検討

### 成功基準

| 指標 | 現状 | 最低目標 | 理想 |
|------|------|----------|------|
| val_acc | 10.3% | 30% | 50%+ |
| 音声品質 | 意味不明 | 聞き取れる | 自然 |

---

## 9. OpenJTalkフルコンテキストラベル仕様

### ラベル形式

```
sil^n-i+h=o/A:-3+1+7/B:xx-xx_xx/C:02_xx+xx/D:02+xx_xx/E:xx_xx!xx_xx-xx/F:7_4#0_xx@1_3|1_12/G:4_4%0_xx_1/H:xx_xx/I:3-12@1+2&1-8|1+41/J:5_29/K:2+8-41
```

### フィールド定義

| フィールド | 形式 | 内容 |
|-----------|------|------|
| **音素** | `p2^p1-c+n1=n2` | 5音素コンテキスト（前2、現在、後2） |
| **A** | `a1+a2+a3` | モーラ情報（アクセント相対位置、前方位置、後方位置） |
| **B** | `b1-b2_b3` | 前の単語（品詞、活用型、活用形） |
| **C** | `c1_c2+c3` | 現在の単語 |
| **D** | `d1+d2_d3` | 次の単語 |
| **E** | `e1_e2!e3_e4-e5` | 前のアクセント句 |
| **F** | `f1_f2#f3_f4@f5_f6\|f7_f8` | 現在のアクセント句（モーラ数、アクセント位置等） |
| **G** | `g1_g2%g3_g4_g5` | 次のアクセント句 |
| **H** | `h1_h2` | 前の呼吸グループ |
| **I** | `i1-i2@i3+i4&i5-i6\|i7+i8` | 現在の呼吸グループ |
| **J** | `j1_j2` | 次の呼吸グループ |
| **K** | `k1+k2-k3` | 発話レベル情報 |

### TTS向け重要フィールド

**A: モーラ情報（必須）**
- `a1`: アクセント核との相対位置（-10〜+10）→ ピッチ決定に重要
- `a2`: アクセント句内の前方位置（1〜49）
- `a3`: アクセント句内の後方位置（1〜49）

**F: アクセント句情報（推奨）**
- `f1`: アクセント句内のモーラ数
- `f2`: アクセント位置（アクセント核の位置）

### 使用ライブラリ

| ライブラリ | 特徴 |
|-----------|------|
| [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) | プリビルドwheel、Python 3.11-3.14対応、改良版辞書 |
| [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) | オリジナル版、ビルド必要 |
| [jpreprocess](https://github.com/jpreprocess/jpreprocess) | Rust製OpenJTalk互換 |

### 参考実装

- [piper-plus](https://github.com/ayutaz/piper-plus) - OpenJTalk統合の日本語TTS
- [VOICEVOX](https://github.com/VOICEVOX/voicevox_engine) - フルコンテキストラベル処理

---

## 10. 参考リンク

- [Soprano](https://github.com/ekwek1/soprano) - 元のTTSモデル
- [Soprano-Factory](https://github.com/ekwek1/soprano-factory) - 訓練フレームワーク
- [Vocos](https://github.com/gemelo-ai/vocos) - デコーダーのベースアーキテクチャ
- [MoeSpeech](https://huggingface.co/datasets/litagin/moespeech) - 日本語音声データセット
- [JSUT](https://sites.google.com/site/shinaborutakahashi/jsut) - 日本語音声コーパス
- [JVS](https://sites.google.com/site/shinaborutakahashi/jvs) - 日本語多話者コーパス

---

## 11. WandBログ

訓練の進捗はWandBで確認可能：
- プロジェクト: `soprano-factory`
- 実行例: `https://wandb.ai/yousan/soprano-factory/runs/9a184mrs`

---

## 12. 更新履歴

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
| 2026-01-18 | train_decoder_gan.py 100kステップ訓練完了（val_loss: 0.2593、音質良好） |
| 2026-01-18 | Soprano統合調査完了、inference_gan.py作成（Post-processing approach） |
| 2026-01-18 | train_decoder_e2e.py作成（512-dim入力対応、E2E訓練） |
| 2026-01-18 | 統合テスト実施：LLMがボトルネックと判明（デコーダーは正常動作） |
| 2026-01-18 | ドキュメント更新：問題の根本原因と今後の優先順位を明確化 |
| 2026-01-18 | **根本原因特定**: text_factor=0.0により日本語テキストの勾配が流れていなかった |
| 2026-01-18 | LLM改善プラン（v6）策定：text_factor=0.5, max_steps=50000で再訓練予定 |
| 2026-01-18 | **Option A訓練実施**: 50kステップ完了、しかしval_acc=10.3%で改善なし |
| 2026-01-18 | **根本原因特定（深層）**: アーキテクチャがテキスト→音声遷移を訓練しない構造的欠陥 |
| 2026-01-18 | **Option B調査**: pyopenjtalk-plus + アクセントラベル（a1-a3）によるアプローチ策定 |
| 2026-01-20 | **Option B-F実施**: 全アプローチ失敗（val_acc ~10-12%）、根本原因を再調査 |
| 2026-01-20 | **新アプローチ**: Contrastive Text-Audio Alignment Lossを実装 |
