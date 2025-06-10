# CLIPモデルのゼロショット分類の例

このディレクトリには、OpenAIのCLIPモデルを使用したゼロショット画像分類の例が含まれています。

## ファイル構成

- `zero_shot_classification.py`: 基本的なゼロショット分類の実装
- `zero_shot_classification_advanced.py`: 複数のプロンプトテンプレートと視覚化機能を含む高度な実装
- `custom_classification.py`: コマンドライン引数を使用した使いやすいゼロショット分類ツール

## 必要なパッケージ

以下のパッケージが必要です：
```bash
pip install torch torchvision transformers pillow numpy matplotlib tqdm
```

## 使用方法

### 基本的なゼロショット分類

```bash
python zero_shot_classification.py
```

このスクリプトは、指定した画像に対して事前定義されたラベルセットでゼロショット分類を行います。使用前に、コード内の`sample_image_path`を実際の画像ファイルのパスに変更してください。

### 高度なゼロショット分類

```bash
python zero_shot_classification_advanced.py
```

このスクリプトは以下の機能を提供します：

1. 複数のプロンプトテンプレートを使用した分類
2. 結果の視覚化（画像と予測スコアのグラフを含む）
3. データセット全体の評価と結果のJSON出力

### ドメイン特化型分類ツール

```bash
python custom_classification.py --image path/to/image.jpg --domain animals
```

このスクリプトは、コマンドライン引数を使用して簡単に画像分類を行うことができます：

- `--image`: 分類する画像のパス（必須）
- `--domain`: 分類ドメイン（general, animals, food, vehicles, custom）
- `--custom_labels`: カスタムラベル（カンマ区切り、domain=customの場合）
- `--output`: 結果の画像を保存するパス

例：
```bash
# 一般的なカテゴリで分類
python custom_classification.py --image cat.jpg --domain general

# 動物カテゴリで分類
python custom_classification.py --image cat.jpg --domain animals

# カスタムラベルで分類
python custom_classification.py --image cat.jpg --domain custom --custom_labels "シャム猫,アメリカンショートヘア,メインクーン,ペルシャ猫,雑種猫"

# 結果を保存
python custom_classification.py --image cat.jpg --domain animals --output results/cat_result.png
```

## 分類対象のカスタマイズ

両方のスクリプトでは、`candidate_labels`リストを変更することで、分類対象のクラスを変更できます。例：

```python
candidate_labels = ["cat", "dog", "car", "house", "person", "bird", "flower"]
```

## プロンプトテンプレートのカスタマイズ

高度な実装では、プロンプトテンプレートをカスタマイズできます：

```python
prompt_templates = [
    "a photo of a {}",
    "a {} in the wild",
    "a close-up of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}"
]
```

## データセット評価

実際のデータセットで評価するには、以下の設定を変更します：

1. `image_dir`: 画像が含まれるディレクトリ
2. `true_labels`: 各画像の正解ラベル（ファイル名をキーとする辞書）
3. `candidate_labels`: 候補となるクラスラベルのリスト

## 出力

高度な実装では、`output`ディレクトリに以下のファイルが生成されます：

- 各画像の分類結果の視覚化（`{image_name}_results.png`）
- データセット全体の評価結果（`classification_results.json`） 