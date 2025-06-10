# 画像キャプション検索

このセクションでは、ファインチューニングしたCLIPモデルを使用して画像に最適なキャプションを検索する方法について説明します。

## 単一画像のキャプション検索

```bash
python examples/test_finetuned_clip.py --image path/to/image.jpg --model models/finetuned_clip
```

このコマンドは、指定した画像に最も類似したキャプションを検索し、結果を視覚化します。

## データセット全体の評価

```bash
python examples/test_finetuned_clip.py --mode evaluate --model models/finetuned_clip
```

このコマンドは、データセット全体に対してキャプション検索の性能を評価し、以下の指標を計算します：
- Top-1精度: 正解キャプションが1位にランクされる割合
- Top-5精度: 正解キャプションが上位5位以内にランクされる割合
- 平均ランク: 正解キャプションの平均ランク

## カスタムキャプションCSVの使用

独自のキャプションCSVファイルを使用する場合：

```bash
python examples/test_finetuned_clip.py --image path/to/image.jpg --model models/finetuned_clip --captions-csv path/to/your/captions.csv
```

## 結果の保存

結果を特定のパスに保存する場合：

```bash
python examples/test_finetuned_clip.py --image path/to/image.jpg --model models/finetuned_clip --output path/to/output.png
```

評価結果をJSONファイルとして保存する場合：

```bash
python examples/test_finetuned_clip.py --mode evaluate --model models/finetuned_clip --output path/to/evaluation_results.json
```

## テストデータの割合の調整

評価に使用するデータの割合を調整する場合：

```bash
python examples/test_finetuned_clip.py --mode evaluate --model models/finetuned_clip --test-split 0.5
```

このコマンドは、データセットの50%をテストに使用します（デフォルトは20%）。 