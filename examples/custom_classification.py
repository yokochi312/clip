import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"デバイス: {device}")

def zero_shot_classify(model, processor, image_path, labels, templates):
    """
    画像をゼロショット分類する
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    all_probs = []
    
    # 各テンプレートでスコアを計算
    for template in templates:
        texts = [template.format(label) for label in labels]
        
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    
    # すべてのテンプレートの平均を取る
    avg_probs = np.mean(all_probs, axis=0)
    
    return {
        "scores": {label: float(score) for label, score in zip(labels, avg_probs)},
        "best_label": labels[np.argmax(avg_probs)],
        "best_score": float(avg_probs[np.argmax(avg_probs)])
    }

def visualize_results(image_path, results, title=None, output_path=None):
    """
    分類結果を視覚化
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # スコアをソート
    labels = list(results["scores"].keys())
    scores = [results["scores"][label] for label in labels]
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # 上位5件のみ表示
    top_n = min(5, len(sorted_labels))
    sorted_labels = sorted_labels[:top_n]
    sorted_scores = sorted_scores[:top_n]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 画像
    ax1.imshow(image)
    ax1.set_title(title or "入力画像")
    ax1.axis("off")
    
    # スコア
    bars = ax2.barh(range(top_n), sorted_scores, align="center")
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(sorted_labels)
    ax2.set_xlabel("確率")
    ax2.set_title("分類結果")
    
    # バーにラベル追加
    for i, bar in enumerate(bars):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{sorted_scores[i]:.4f}", va="center")
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ゼロショット画像分類")
    parser.add_argument("--image", type=str, required=True, help="分類する画像のパス")
    parser.add_argument("--domain", type=str, default="general", choices=["general", "animals", "food", "vehicles", "custom"], help="分類ドメイン")
    parser.add_argument("--custom_labels", type=str, help="カンマ区切りのカスタムラベル (--domain customの場合)")
    parser.add_argument("--output", type=str, help="結果の保存先パス")
    args = parser.parse_args()
    
    # モデルとプロセッサをロード
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # ドメインごとのラベル
    domain_labels = {
        "general": ["person", "animal", "vehicle", "building", "nature", "food", "clothing", "furniture", "electronics", "sports"],
        "animals": ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "monkey"],
        "food": ["pizza", "burger", "pasta", "sushi", "salad", "steak", "soup", "dessert", "fruit", "vegetable", "bread", "rice"],
        "vehicles": ["car", "truck", "motorcycle", "bicycle", "bus", "train", "airplane", "boat", "helicopter", "scooter"]
    }
    
    # テンプレート
    domain_templates = {
        "general": ["a photo of a {}", "a picture of a {}", "{}"],
        "animals": ["a photo of a {}", "a picture of a {}", "a {} in the wild", "a close-up of a {}"],
        "food": ["a photo of {}", "a picture of {}", "a plate of {}", "a serving of {}", "delicious {}"],
        "vehicles": ["a photo of a {}", "a picture of a {}", "a {} on the road", "a {} vehicle"]
    }
    
    # ラベルとテンプレートを設定
    if args.domain == "custom" and args.custom_labels:
        labels = [label.strip() for label in args.custom_labels.split(",")]
        templates = ["a photo of a {}", "a picture of a {}", "{}"]
    else:
        if args.domain not in domain_labels:
            print(f"指定されたドメイン '{args.domain}' は無効です。'general'を使用します。")
            args.domain = "general"
        
        labels = domain_labels[args.domain]
        templates = domain_templates[args.domain]
    
    # 画像ファイルの確認
    if not os.path.exists(args.image):
        print(f"画像ファイル '{args.image}' が見つかりません。")
        return
    
    # 分類実行
    print(f"画像を分類中... ({args.domain} ドメイン)")
    result = zero_shot_classify(model, processor, args.image, labels, templates)
    
    # 結果表示
    print(f"分類結果: {result['best_label']} (確率: {result['best_score']:.4f})")
    print("各ラベルのスコア:")
    for label, score in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {score:.4f}")
    
    # 視覚化
    title = f"分類ドメイン: {args.domain}"
    output_path = args.output if args.output else None
    visualize_results(args.image, result, title, output_path)
    
    if output_path:
        print(f"結果を保存しました: {output_path}")

if __name__ == "__main__":
    main() 