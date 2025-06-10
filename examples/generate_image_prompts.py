import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import json
import random

def get_image_embeddings(model, processor, image_path, device="cpu"):
    """
    画像の埋め込みベクトルを取得
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # 画像を処理
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 埋め込みを取得
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        
    # 正規化
    embeddings = outputs.detach().cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings[0]

def get_text_embeddings(model, processor, texts, device="cpu"):
    """
    テキストの埋め込みベクトルを取得
    """
    # テキストを処理
    inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # 埋め込みを取得
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    # 正規化
    embeddings = outputs.detach().cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings

def image_text_similarity(image_embedding, text_embeddings):
    """
    画像とテキストの類似度を計算
    """
    # コサイン類似度を計算
    similarities = np.dot(text_embeddings, image_embedding)
    return similarities

def find_most_similar_captions(model, processor, image_path, captions, top_n=5, device="cpu"):
    """
    画像に最も類似したキャプションを見つける
    """
    # 画像の埋め込みを取得
    image_embedding = get_image_embeddings(model, processor, image_path, device)
    
    # テキストの埋め込みを取得
    text_embeddings = get_text_embeddings(model, processor, captions, device)
    
    # 類似度を計算
    similarities = image_text_similarity(image_embedding, text_embeddings)
    
    # 類似度でソート
    indices = np.argsort(similarities)[::-1]
    
    # 上位N件を返す
    top_indices = indices[:top_n]
    top_similarities = similarities[top_indices]
    top_captions = [captions[i] for i in top_indices]
    
    return list(zip(top_captions, top_similarities))

def enhance_caption_for_image_generation(caption):
    """
    画像生成用にキャプションを強化する
    """
    # 画像品質を向上させるための修飾語
    quality_enhancers = [
        "high quality", "detailed", "sharp focus", "professional", 
        "4K", "high resolution", "beautiful", "stunning"
    ]
    
    # ランダムに2-3個の修飾語を選択
    num_enhancers = random.randint(2, 3)
    selected_enhancers = random.sample(quality_enhancers, num_enhancers)
    
    # 修飾語をキャプションに追加
    enhanced_caption = caption + ", " + ", ".join(selected_enhancers)
    
    return enhanced_caption

def create_image_generation_prompts(model, processor, image_path, captions_df, num_prompts=3, device="cpu"):
    """
    画像生成用のプロンプトを作成する
    """
    # キャプションのリストを取得
    captions = captions_df["caption"].tolist()
    
    # 画像に最も類似したキャプションを見つける
    similar_captions = find_most_similar_captions(
        model, processor, image_path, captions, top_n=num_prompts, device=device
    )
    
    # 画像生成用にキャプションを強化
    generation_prompts = []
    for caption, similarity in similar_captions:
        enhanced_caption = enhance_caption_for_image_generation(caption)
        generation_prompts.append({
            "original_caption": caption,
            "enhanced_caption": enhanced_caption,
            "similarity": float(similarity)
        })
    
    return generation_prompts

def visualize_image_and_prompts(image_path, prompts, title=None, output_path=None):
    """
    画像と生成プロンプトを視覚化
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 画像を表示
    ax1.imshow(image)
    ax1.set_title(title or "入力画像")
    ax1.axis("off")
    
    # プロンプトを表示
    ax2.axis("off")
    ax2.text(0.01, 0.99, "画像生成用プロンプト:", fontsize=12, fontweight="bold", va="top")
    
    y_pos = 0.92
    for i, prompt in enumerate(prompts):
        ax2.text(0.01, y_pos, f"{i+1}. 類似度: {prompt['similarity']:.4f}", fontsize=10, va="top", color="blue")
        y_pos -= 0.05
        ax2.text(0.01, y_pos, f"元のキャプション:", fontsize=9, va="top", color="gray")
        y_pos -= 0.05
        ax2.text(0.01, y_pos, f"{prompt['original_caption']}", fontsize=9, va="top", wrap=True)
        y_pos -= 0.08
        ax2.text(0.01, y_pos, f"強化されたプロンプト:", fontsize=9, va="top", color="green")
        y_pos -= 0.05
        ax2.text(0.01, y_pos, f"{prompt['enhanced_caption']}", fontsize=9, va="top", wrap=True)
        y_pos -= 0.1
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ファインチューニングしたCLIPモデルを使用して画像生成用プロンプトを作成")
    parser.add_argument("--image", type=str, required=True, help="入力画像のパス")
    parser.add_argument("--model", type=str, default="models/finetuned_clip", help="ファインチューニングしたモデルのパス")
    parser.add_argument("--captions-csv", type=str, default="data/flickr8k_sample/captions.csv", help="キャプションCSVファイルのパス")
    parser.add_argument("--num-prompts", type=int, default=3, help="生成するプロンプトの数")
    parser.add_argument("--output", type=str, help="結果の保存先パス")
    args = parser.parse_args()
    
    # デバイスを設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    # 画像ファイルの確認
    if not os.path.exists(args.image):
        print(f"画像ファイル '{args.image}' が見つかりません。")
        return
    
    # モデルとプロセッサをロード
    print(f"モデルをロード中: {args.model}")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    
    # キャプションを読み込む
    print(f"キャプションを読み込み中: {args.captions_csv}")
    captions_df = pd.read_csv(args.captions_csv)
    
    # 画像生成用プロンプトを作成
    print("画像生成用プロンプトを作成中...")
    prompts = create_image_generation_prompts(
        model, processor, args.image, captions_df, num_prompts=args.num_prompts, device=device
    )
    
    # 結果を表示
    print("\n画像生成用プロンプト:")
    for i, prompt in enumerate(prompts):
        print(f"\n{i+1}. 類似度: {prompt['similarity']:.4f}")
        print(f"   元のキャプション: {prompt['original_caption']}")
        print(f"   強化されたプロンプト: {prompt['enhanced_caption']}")
    
    # 視覚化
    if args.output:
        output_path = args.output
    else:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "image_generation_prompts.png")
    
    visualize_image_and_prompts(
        args.image, prompts, "ファインチューニングしたCLIPモデルによる画像生成プロンプト", output_path
    )
    print(f"結果を保存しました: {output_path}")
    
    # プロンプトをテキストファイルとしても保存
    txt_output_path = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i+1}. {prompt['enhanced_caption']}\n\n")
    
    print(f"プロンプトをテキストファイルとして保存しました: {txt_output_path}")

if __name__ == "__main__":
    main() 