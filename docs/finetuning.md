# CLIPモデルのファインチューニング

このセクションでは、CLIPモデルを独自のデータセットでファインチューニングする方法について説明します。

## サンプルデータの準備

まず、Flickr8kデータセットのサンプルを準備します：

```bash
python examples/prepare_sample_data.py
```

このスクリプトは以下の処理を行います：
1. Flickr8kデータセットをダウンロード
2. 画像とキャプションのデータを抽出
3. サンプルとして100枚の画像とそのキャプションを準備

## ファインチューニングの実行

準備したデータセットを使用してCLIPモデルをファインチューニングします：

```bash
python src/clip.py --csv data/flickr8k_sample/captions.csv --images data/flickr8k_sample/images --output models/finetuned_clip
```

オプション：
- `--batch-size`: バッチサイズ（デフォルト: 4）
- `--epochs`: エポック数（デフォルト: 3）
- `--lr`: 学習率（デフォルト: 1e-5）

## 独自のデータセットの使用

独自のデータセットを使用する場合は、以下の形式でCSVファイルを準備します：

```csv
filename,caption
image1.jpg,画像のキャプション1
image2.jpg,画像のキャプション2
...
```

そして、以下のコマンドでファインチューニングを実行します：

```bash
python src/clip.py --csv path/to/your/captions.csv --images path/to/your/images --output models/your_finetuned_model
```

## ファインチューニングのパラメータ調整

より良い結果を得るためのパラメータ調整：

```bash
# より多くのエポック数で学習
python src/clip.py --csv data/flickr8k_sample/captions.csv --images data/flickr8k_sample/images --output models/finetuned_clip --epochs 10

# より小さい学習率で学習
python src/clip.py --csv data/flickr8k_sample/captions.csv --images data/flickr8k_sample/images --output models/finetuned_clip --lr 5e-6

# より大きいバッチサイズで学習（GPUメモリに余裕がある場合）
python src/clip.py --csv data/flickr8k_sample/captions.csv --images data/flickr8k_sample/images --output models/finetuned_clip --batch-size 8
``` 