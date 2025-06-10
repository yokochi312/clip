# CLIP Project

このプロジェクトは、OpenAIのCLIPモデルを使用して画像と文章の関連性を学習・分析するための実装です。

## プロジェクト構造

```
clip/
├── src/           # ソースコード
├── tests/         # テストコード
├── docs/          # ドキュメント
├── examples/      # 使用例
├── data/          # データセット（自動ダウンロード）
└── models/        # 保存されたモデル
```

## セットアップ

1. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

## 機能

このプロジェクトでは、以下の機能を提供しています：

1. **ゼロショット画像分類**: 事前学習済みCLIPモデルを使用した画像分類
2. **CLIPモデルのファインチューニング**: 独自のデータセットでCLIPモデルをファインチューニング
3. **画像キャプション検索**: ファインチューニングしたモデルを使用して画像に最適なキャプションを検索
4. **画像生成プロンプト作成**: 画像から画像生成AIのためのプロンプトを作成

## 使用方法

### サンプルデータの準備

Flickr8kデータセットのサンプルを準備します：

```bash
python examples/prepare_sample_data.py
```

### CLIPモデルのファインチューニング

```bash
python src/clip.py --csv data/flickr8k_sample/captions.csv --images data/flickr8k_sample/images --output models/finetuned_clip
```

オプション：
- `--batch-size`: バッチサイズ（デフォルト: 4）
- `--epochs`: エポック数（デフォルト: 3）
- `--lr`: 学習率（デフォルト: 1e-5）

### ゼロショット画像分類

基本的な分類：
```bash
python examples/zero_shot_classification.py
```

高度な分類（複数のプロンプトテンプレートと視覚化）：
```bash
python examples/zero_shot_classification_advanced.py
```

ドメイン別の分類：
```bash
python examples/custom_classification.py --image path/to/image.jpg --domain animals
```

### ファインチューニングしたモデルのテスト

単一画像に対するキャプション検索：
```bash
python examples/test_finetuned_clip.py --image path/to/image.jpg --model models/finetuned_clip
```

データセット全体の評価：
```bash
python examples/test_finetuned_clip.py --mode evaluate --model models/finetuned_clip
```

### 画像生成プロンプトの作成

```bash
python examples/generate_image_prompts.py --image path/to/image.jpg --model models/finetuned_clip
```

オプション：
- `--num-prompts`: 生成するプロンプトの数（デフォルト: 3）
- `--output`: 結果の保存先パス

## ファインチューニングとゼロショット分類の違い

- **ゼロショット分類**: 事前学習済みCLIPモデルを使用して、特定のタスクに対して明示的に学習していなくても分類を行う方法
- **ファインチューニング**: 事前学習済みモデルを特定のデータセットで追加学習し、特定のタスクに適応させる方法

## 結果の例

### ゼロショット分類
- 様々なカテゴリの画像を分類
- プロンプトテンプレートの影響を確認

### ファインチューニング後
- 画像に対する最適なキャプションの検索
- 画像生成AIのためのプロンプト作成 