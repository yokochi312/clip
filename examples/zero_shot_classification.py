import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm
import argparse

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"デバイス: {device}")

# モデルとプロセッサをロード
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def classify_image(image_path, candidate_labels):
    """
    画像ファイルをゼロショット分類する
    
    Args:
        image_path: 画像ファイルへのパス
        candidate_labels: 候補となるクラスラベルのリスト
    
    Returns:
        scores: 各ラベルの確率スコア
        best_label: 最も確率の高いラベル
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # 候補ラベルにプロンプトテンプレートを適用
    texts = [f"a photo of a {label}" for label in candidate_labels]
    
    # 画像とテキストを処理
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 画像とテキストの類似度スコアを取得
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    # 最も確率の高いラベルのインデックスを取得
    best_idx = np.argmax(probs)
    
    return {
        "scores": {label: float(score) for label, score in zip(candidate_labels, probs)},
        "best_label": candidate_labels[best_idx],
        "best_score": float(probs[best_idx])
    }

def evaluate_dataset(image_dir, true_labels, candidate_labels):
    """
    データセット全体の精度を評価
    
    Args:
        image_dir: 画像が含まれるディレクトリ
        true_labels: 各画像の正解ラベル (画像ファイル名をキーとする辞書)
        candidate_labels: 候補となるすべてのクラスラベルのリスト
    
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
        classification = classify_image(image_path, candidate_labels)
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
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results

# 使用例
if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="ゼロショット画像分類")
    parser.add_argument("--image", type=str, default="path/to/your/image.jpg", help="分類する画像ファイルのパス")
    parser.add_argument("--labels", type=str, default="cat,dog,car,house,person", help="カンマ区切りの候補ラベル")
    args = parser.parse_args()
    
    # 候補ラベルをリストに変換
    candidate_labels = args.labels.split(",")
    
    # 画像ファイルが存在するか確認
    if os.path.exists(args.image):
        result = classify_image(args.image, candidate_labels)
        print(f"分類結果: {result['best_label']} (確率: {result['best_score']:.4f})")
        print("各ラベルのスコア:")
        for label, score in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {score:.4f}")
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
    accuracy, results = evaluate_dataset(image_dir, true_labels, candidate_labels)
    print(f"全体の精度: {accuracy:.4f}")
    """ 