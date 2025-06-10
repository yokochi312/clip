import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm
import json
import argparse

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"デバイス: {device}")

# モデルとプロセッサをロード
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def zero_shot_classify(image_path, candidate_labels, prompt_templates=None):
    """
    複数のプロンプトテンプレートを使用したゼロショット分類
    
    Args:
        image_path: 画像ファイルへのパス
        candidate_labels: 候補となるクラスラベルのリスト
        prompt_templates: プロンプトテンプレートのリスト（Noneの場合はデフォルトを使用）
    
    Returns:
        dict: 分類結果
    """
    # デフォルトのプロンプトテンプレート
    if prompt_templates is None:
        prompt_templates = [
            "a photo of a {}",
            "a photograph of a {}",
            "an image of a {}",
            "a picture of a {}",
            "{}"
        ]
    
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    all_probs = []
    
    # 各プロンプトテンプレートで推論
    for template in prompt_templates:
        # テンプレートを適用
        texts = [template.format(label) for label in candidate_labels]
        
        # 画像とテキストを処理
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 画像とテキストの類似度スコアを取得
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    
    # すべてのプロンプトの結果を平均
    avg_probs = np.mean(all_probs, axis=0)
    best_idx = np.argmax(avg_probs)
    
    # 各テンプレートの結果も記録
    template_results = {}
    for i, template in enumerate(prompt_templates):
        template_probs = all_probs[i]
        template_best_idx = np.argmax(template_probs)
        template_results[template] = {
            "scores": {label: float(score) for label, score in zip(candidate_labels, template_probs)},
            "best_label": candidate_labels[template_best_idx],
            "best_score": float(template_probs[template_best_idx])
        }
    
    return {
        "scores": {label: float(score) for label, score in zip(candidate_labels, avg_probs)},
        "best_label": candidate_labels[best_idx],
        "best_score": float(avg_probs[best_idx]),
        "template_results": template_results
    }

def visualize_results(image_path, results, save_path):
    """
    分類結果を視覚化
    
    Args:
        image_path: 画像ファイルへのパス
        results: 分類結果
        save_path: 結果を保存するパス
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # 結果のスコアを取得
    labels = list(results["scores"].keys())
    scores = [results["scores"][label] for label in labels]
    
    # 降順にソート
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    # 上位5件のみ表示
    top_n = min(5, len(sorted_labels))
    sorted_labels = sorted_labels[:top_n]
    sorted_scores = sorted_scores[:top_n]
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 画像を表示
    ax1.imshow(image)
    ax1.set_title("入力画像")
    ax1.axis("off")
    
    # スコアを水平バーチャートで表示
    bars = ax2.barh(range(top_n), sorted_scores, align="center")
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(sorted_labels)
    ax2.set_xlabel("確率")
    ax2.set_title("分類結果")
    
    # バーにラベルを追加
    for i, bar in enumerate(bars):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{sorted_scores[i]:.4f}", va="center")
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path)
    plt.close()

def evaluate_dataset_with_templates(image_dir, true_labels, candidate_labels, prompt_templates=None, output_file=None):
    """
    データセット全体の精度を評価し、詳細な結果を出力
    
    Args:
        image_dir: 画像が含まれるディレクトリ
        true_labels: 各画像の正解ラベル (画像ファイル名をキーとする辞書)
        candidate_labels: 候補となるすべてのクラスラベルのリスト
        prompt_templates: プロンプトテンプレートのリスト
        output_file: 結果を保存するJSONファイルのパス
    
    Returns:
        accuracy: 正解率
        results: 各画像の分類結果
    """
    correct = 0
    total = 0
    results = {}
    
    for filename in tqdm(os.listdir(image_dir)):
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            continue
            
        image_path = os.path.join(image_dir, filename)
        
        if filename not in true_labels:
            continue
            
        true_label = true_labels[filename]
        
        # 分類実行
        classification = zero_shot_classify(image_path, candidate_labels, prompt_templates)
        pred_label = classification["best_label"]
        
        # 正解判定
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        total += 1
        
        # 結果を記録
        results[filename] = {
            "true_label": true_label,
            "pred_label": pred_label,
            "scores": classification["scores"],
            "correct": is_correct
        }
        
        # 視覚化
        visualize_results(image_path, classification, f"output/{os.path.splitext(filename)[0]}_results.png")
    
    accuracy = correct / total if total > 0 else 0
    
    # 結果をJSONファイルとして保存
    if output_file:
        output_data = {
            "accuracy": accuracy,
            "results": results,
            "candidate_labels": candidate_labels,
            "prompt_templates": prompt_templates
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return accuracy, results

# 使用例
if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="高度なゼロショット画像分類")
    parser.add_argument("--image", type=str, default="path/to/your/image.jpg", help="分類する画像ファイルのパス")
    parser.add_argument("--labels", type=str, default="cat,dog,car,house,person", help="カンマ区切りの候補ラベル")
    parser.add_argument("--output", type=str, default="output/classification_results.png", help="結果の可視化を保存するパス")
    args = parser.parse_args()
    
    # 候補ラベルをリストに変換
    if args.labels:
        candidate_labels = args.labels.split(",")
    else:
        # デフォルトの候補ラベル
        candidate_labels = ["cat", "dog", "car", "house", "person"]
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 画像ファイルが存在するか確認
    if os.path.exists(args.image):
        # カスタムプロンプトテンプレート
        prompt_templates = [
            "a photo of a {}",
            "a {} in the scene",
            "looking at a {}",
            "this is a {}",
            "{} in the image"
        ]
        
        # 複数のプロンプトテンプレートを使用した分類
        result = zero_shot_classify(args.image, candidate_labels, prompt_templates)
        
        print(f"分類結果: {result['best_label']} (確率: {result['best_score']:.4f})")
        print("各ラベルのスコア:")
        for label, score in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {score:.4f}")
        
        # 可視化
        visualize_results(args.image, result, save_path=args.output)
        print(f"結果の視覚化が {args.output} に保存されました")
        
    else:
        print(f"サンプル画像 {args.image} が見つかりません。パスを確認してください。")
    
    # データセット全体の評価例（コメントアウト）
    """
    image_dir = "path/to/your/images"
    
    # 各画像の正解ラベル（実際のデータに合わせて調整）
    true_labels = {
        "image1.jpg": "cat",
        "image2.jpg": "dog",
        # ...
    }
    
    # 候補ラベル
    candidate_labels = ["cat", "dog", "car", "house", "person"]
    
    # 評価実行
    accuracy, results = evaluate_dataset_with_templates(
        image_dir,
        true_labels,
        candidate_labels,
        prompt_templates,
        output_file="output/classification_results.json"
    )
    print(f"全体の精度: {accuracy:.4f}")
    """ 