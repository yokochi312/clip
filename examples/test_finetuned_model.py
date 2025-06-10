import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse

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
        
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        
        # GPUが利用可能な場合はデバイスに転送
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model = model.to("cuda")
        
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

def compare_models(image_path, labels, templates, original_model_name="openai/clip-vit-base-patch32", finetuned_model_path="models/finetuned_clip"):
    """
    元のモデルとファインチューニングしたモデルを比較する
    """
    # 元のモデルをロード
    print(f"元のモデルをロード中: {original_model_name}")
    original_model = CLIPModel.from_pretrained(original_model_name)
    original_processor = CLIPProcessor.from_pretrained(original_model_name)
    
    # ファインチューニングしたモデルをロード
    print(f"ファインチューニングしたモデルをロード中: {finetuned_model_path}")
    finetuned_model = CLIPModel.from_pretrained(finetuned_model_path)
    finetuned_processor = CLIPProcessor.from_pretrained(finetuned_model_path)
    
    # 元のモデルで分類
    print("元のモデルで分類中...")
    original_result = zero_shot_classify(original_model, original_processor, image_path, labels, templates)
    
    # ファインチューニングしたモデルで分類
    print("ファインチューニングしたモデルで分類中...")
    finetuned_result = zero_shot_classify(finetuned_model, finetuned_processor, image_path, labels, templates)
    
    # 結果を表示
    print("\n元のモデルの結果:")
    print(f"最も確率の高いラベル: {original_result['best_label']} (確率: {original_result['best_score']:.4f})")
    print("各ラベルのスコア:")
    for label, score in sorted(original_result["scores"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {score:.4f}")
    
    print("\nファインチューニングしたモデルの結果:")
    print(f"最も確率の高いラベル: {finetuned_result['best_label']} (確率: {finetuned_result['best_score']:.4f})")
    print("各ラベルのスコア:")
    for label, score in sorted(finetuned_result["scores"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {score:.4f}")
    
    # 視覚化
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 元のモデルの結果を視覚化
    visualize_results(
        image_path, 
        original_result, 
        "元のモデル", 
        os.path.join(output_dir, "original_model_result.png")
    )
    
    # ファインチューニングしたモデルの結果を視覚化
    visualize_results(
        image_path, 
        finetuned_result, 
        "ファインチューニングしたモデル", 
        os.path.join(output_dir, "finetuned_model_result.png")
    )
    
    return original_result, finetuned_result

def main():
    parser = argparse.ArgumentParser(description="ファインチューニングしたCLIPモデルのテスト")
    parser.add_argument("--image", type=str, required=True, help="分類する画像のパス")
    parser.add_argument("--finetuned-model", type=str, default="models/finetuned_clip", help="ファインチューニングしたモデルのパス")
    parser.add_argument("--domain", type=str, default="general", choices=["general", "animals", "food", "vehicles", "custom"], help="分類ドメイン")
    parser.add_argument("--custom-labels", type=str, help="カンマ区切りのカスタムラベル (--domain customの場合)")
    args = parser.parse_args()
    
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
    
    # モデルの比較を実行
    compare_models(
        image_path=args.image,
        labels=labels,
        templates=templates,
        finetuned_model_path=args.finetuned_model
    )

if __name__ == "__main__":
    main() 