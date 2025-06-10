# 画像生成プロンプト作成

このセクションでは、ファインチューニングしたCLIPモデルを使用して画像生成AIのためのプロンプトを作成する方法について説明します。

## 基本的な使用方法

```bash
python examples/generate_image_prompts.py --image path/to/image.jpg --model models/finetuned_clip
```

このコマンドは、指定した画像に基づいて画像生成用のプロンプトを作成します。デフォルトでは3つのプロンプトが生成されます。

## プロンプト数の調整

生成するプロンプトの数を調整する場合：

```bash
python examples/generate_image_prompts.py --image path/to/image.jpg --model models/finetuned_clip --num-prompts 5
```

このコマンドは、5つのプロンプトを生成します。

## カスタムキャプションCSVの使用

独自のキャプションCSVファイルを使用する場合：

```bash
python examples/generate_image_prompts.py --image path/to/image.jpg --model models/finetuned_clip --captions-csv path/to/your/captions.csv
```

## 結果の保存

結果を特定のパスに保存する場合：

```bash
python examples/generate_image_prompts.py --image path/to/image.jpg --model models/finetuned_clip --output path/to/output.png
```

このコマンドは、視覚化された結果を画像ファイルとして保存します。また、プロンプトのテキストも同じディレクトリに`.txt`ファイルとして保存されます。

## プロンプト強化の仕組み

生成されるプロンプトは、以下のプロセスで強化されます：

1. 画像に最も類似したキャプションを検索
2. 画像品質を向上させるための修飾語（high quality, detailed, sharp focusなど）をランダムに2-3個追加
3. 強化されたプロンプトを生成

## 画像生成AIとの使用

生成されたプロンプトは、以下のような画像生成AIで使用できます：

- Stable Diffusion
- Midjourney
- DALL-E

例えば、Stable Diffusionのコマンドラインインターフェースでは：

```bash
python stable_diffusion_cli.py --prompt "$(cat output/image_generation_prompts.txt | head -n 1)" --output generated_image.png
``` 