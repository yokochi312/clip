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

def visualize_image_caption_matches(image_path, caption_similarities, title=None, output_path=None):
    """
    画像とキャプションのマッチングを視覚化
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # キャプションと類似度を分離
    captions = [item[0] for item in caption_similarities]
    similarities = [item[1] for item in caption_similarities]
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 画像を表示
    ax1.imshow(image)
    ax1.set_title(title or "入力画像")
    ax1.axis("off")
    
    # 類似度を水平バーチャートで表示
    bars = ax2.barh(range(len(similarities)), similarities, align="center")
    ax2.set_yticks(range(len(similarities)))
    ax2.set_yticklabels([f"{i+1}. {cap[:50]}..." if len(cap) > 50 else f"{i+1}. {cap}" for i, cap in enumerate(captions)])
    ax2.set_xlabel("類似度")
    ax2.set_title("最も類似したキャプション")
    
    # バーにラベルを追加
    for i, bar in enumerate(bars):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{similarities[i]:.4f}", va="center")
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    plt.show()
    plt.close()

def generate_caption(model, processor, image_path, candidate_captions, device="cpu"):
    """
    画像に対して最も適切なキャプションを生成（類似度に基づく選択）
    """
    # 最も類似したキャプションを見つける
    similar_captions = find_most_similar_captions(
        model, processor, image_path, candidate_captions, top_n=1, device=device
    )
    
    # 最も類似したキャプションを返す
    return similar_captions[0]

def evaluate_caption_retrieval(model, processor, test_data, device="cpu"):
    """
    キャプション検索の性能を評価
    
    Args:
        model: CLIPモデル
        processor: CLIPプロセッサ
        test_data: テスト用データ（画像パスとキャプションのペア）
        device: デバイス
    
    Returns:
        評価結果の辞書
    """
    results = {
        "top1_accuracy": 0,
        "top5_accuracy": 0,
        "mean_rank": 0,
        "detailed_results": []
    }
    
    total_images = len(test_data)
    
    # すべての画像に対してキャプションを取得
    all_captions = [item["caption"] for item in test_data]
    
    for idx, item in enumerate(tqdm(test_data, desc="キャプション検索を評価中")):
        image_path = item["image_path"]
        true_caption = item["caption"]
        
        # 画像に最も類似したキャプションを見つける
        similar_captions = find_most_similar_captions(
            model, processor, image_path, all_captions, top_n=len(all_captions), device=device
        )
        
        # 正解キャプションのランクを取得
        caption_ranks = [cap for cap, _ in similar_captions]
        true_caption_rank = caption_ranks.index(true_caption) + 1
        
        # Top-1とTop-5の精度を計算
        in_top1 = true_caption_rank == 1
        in_top5 = true_caption_rank <= 5
        
        results["top1_accuracy"] += int(in_top1)
        results["top5_accuracy"] += int(in_top5)
        results["mean_rank"] += true_caption_rank
        
        # 詳細な結果を記録
        detailed_result = {
            "image_path": image_path,
            "true_caption": true_caption,
            "true_caption_rank": true_caption_rank,
            "in_top1": in_top1,
            "in_top5": in_top5,
            "top5_captions": [cap for cap, sim in similar_captions[:5]]
        }
        results["detailed_results"].append(detailed_result)
    
    # 平均値を計算
    results["top1_accuracy"] /= total_images
    results["top5_accuracy"] /= total_images
    results["mean_rank"] /= total_images
    
    return results

def main():
    parser = argparse.ArgumentParser(description="ファインチューニングしたCLIPモデルのテスト")
    parser.add_argument("--image", type=str, help="テストする画像のパス")
    parser.add_argument("--model", type=str, default="models/finetuned_clip", help="ファインチューニングしたモデルのパス")
    parser.add_argument("--captions-csv", type=str, default="data/flickr8k_sample/captions.csv", help="キャプションCSVファイルのパス")
    parser.add_argument("--output", type=str, help="結果の保存先パス")
    parser.add_argument("--mode", type=str, choices=["single", "evaluate"], default="single", help="テストモード (single: 単一画像, evaluate: データセット評価)")
    parser.add_argument("--test-split", type=float, default=0.2, help="テスト用データの割合")
    args = parser.parse_args()
    
    # デバイスを設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    # モデルとプロセッサをロード
    print(f"モデルをロード中: {args.model}")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    
    # キャプションを読み込む
    print(f"キャプションを読み込み中: {args.captions_csv}")
    captions_df = pd.read_csv(args.captions_csv)
    captions = captions_df["caption"].tolist()
    
    if args.mode == "single":
        # 単一画像のテスト
        if not args.image:
            print("画像パスを指定してください (--image)")
            return
        
        if not os.path.exists(args.image):
            print(f"画像ファイル '{args.image}' が見つかりません。")
            return
        
        # 画像に最も類似したキャプションを見つける
        print("画像に最も類似したキャプションを検索中...")
        similar_captions = find_most_similar_captions(
            model, processor, args.image, captions, top_n=5, device=device
        )
        
        # 結果を表示
        print("\n画像に最も類似したキャプション:")
        for i, (caption, similarity) in enumerate(similar_captions):
            print(f"{i+1}. [{similarity:.4f}] {caption}")
        
        # 視覚化
        if args.output:
            output_path = args.output
        else:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "caption_matches.png")
        
        visualize_image_caption_matches(
            args.image, similar_captions, "ファインチューニングしたCLIPモデル", output_path
        )
        print(f"結果を保存しました: {output_path}")
    
    elif args.mode == "evaluate":
        # データセット全体の評価
        print("データセット全体の評価を実行中...")
        
        # 画像ディレクトリを取得
        image_dir = os.path.dirname(args.captions_csv)
        if "flickr8k_sample" in image_dir:
            image_dir = os.path.join(image_dir, "images")
        
        # テスト用データを準備
        test_data = []
        for _, row in captions_df.iterrows():
            image_path = os.path.join(image_dir, row["filename"])
            if os.path.exists(image_path):
                test_data.append({
                    "image_path": image_path,
                    "caption": row["caption"]
                })
        
        # テスト用データをシャッフルして一部を使用
        np.random.shuffle(test_data)
        test_size = int(len(test_data) * args.test_split)
        test_data = test_data[:test_size]
        
        print(f"テスト用データ数: {len(test_data)}")
        
        # 評価を実行
        results = evaluate_caption_retrieval(model, processor, test_data, device=device)
        
        # 結果を表示
        print("\n評価結果:")
        print(f"Top-1 精度: {results['top1_accuracy']:.4f}")
        print(f"Top-5 精度: {results['top5_accuracy']:.4f}")
        print(f"平均ランク: {results['mean_rank']:.2f}")
        
        # 結果を保存
        if args.output:
            output_path = args.output
        else:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "evaluation_results.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"評価結果を保存しました: {output_path}")

if __name__ == "__main__":
    main() 