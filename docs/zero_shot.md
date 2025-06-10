# ゼロショット画像分類

このセクションでは、事前学習済みのCLIPモデルを使用したゼロショット画像分類について説明します。

## 基本的な使用方法

```bash
python examples/zero_shot_classification.py --image path/to/image.jpg
```

## 高度な使用方法

複数のプロンプトテンプレートと視覚化機能を使用した高度な分類：

```bash
python examples/zero_shot_classification_advanced.py --image path/to/image.jpg
```

## ドメイン別分類

特定のドメイン向けに最適化された分類：

```bash
python examples/custom_classification.py --image path/to/image.jpg --domain animals
```

利用可能なドメイン：
- `general`: 一般的なカテゴリ（人、動物、乗り物など）
- `animals`: 動物カテゴリ（猫、犬、鳥など）
- `food`: 食品カテゴリ（ピザ、ハンバーガー、パスタなど）
- `vehicles`: 乗り物カテゴリ（車、トラック、バイクなど）
- `custom`: カスタムラベル（`--custom-labels`オプションで指定）

## カスタムラベルの使用

独自のラベルセットを使用した分類：

```bash
python examples/custom_classification.py --image path/to/image.jpg --domain custom --custom-labels "シャム猫,アメリカンショートヘア,メインクーン,ペルシャ猫,雑種猫"
```

## 結果の保存

分類結果を画像ファイルとして保存：

```bash
python examples/custom_classification.py --image path/to/image.jpg --domain animals --output results/animal_classification.png
``` 