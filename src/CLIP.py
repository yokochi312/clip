import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
from tqdm import tqdm
import argparse

# データセット定義
class ImageTextDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")
        text = row['caption']
        
        # 画像とテキストを個別に処理（パディングはDataLoaderのcollate_fnで行う）
        return {
            "image": image,
            "text": text
        }

# カスタムcollate関数
def clip_collate_fn(batch):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # バッチ全体を一度に処理してパディング
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return inputs

def train_clip(csv_path, image_dir, output_dir, batch_size=4, num_epochs=3, lr=1e-5):
    """
    CLIPモデルをファインチューニングする
    
    Args:
        csv_path: キャプションCSVファイルのパス
        image_dir: 画像ディレクトリのパス
        output_dir: モデル保存先ディレクトリ
        batch_size: バッチサイズ
        num_epochs: エポック数
        lr: 学習率
    """
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    # モデル・プロセッサ
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)

    # データローダー
    dataset = ImageTextDataset(csv_path, image_dir, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=clip_collate_fn
    )
    
    print(f"データセットサイズ: {len(dataset)}画像")

    # 最適化・学習
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"エポック {epoch+1}/{num_epochs}"):
            # バッチをデバイスに転送
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # モデル実行
            outputs = model(**inputs)
            
            # デバッグ: モデル出力の構造を確認
            print("モデル出力の属性:")
            for attr in dir(outputs):
                if not attr.startswith('_'):
                    print(f"- {attr}")
            
            # 損失計算
            # CLIPモデルの損失を計算（画像とテキストの類似度に基づく対照損失）
            if hasattr(outputs, 'logits_per_image'):
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # 対角要素が正例（同じインデックスの画像とテキストが対応）
                labels = torch.arange(len(logits_per_image)).to(device)
                
                # 画像→テキストとテキスト→画像の両方向の損失を計算
                loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2.0
            else:
                print("Error: モデル出力に logits_per_image がありません")
                print(f"出力の型: {type(outputs)}")
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                    print(f"outputs.loss: {loss}")
                else:
                    print("outputs.loss も存在しません")
                    # 一時的に損失をゼロにして続行
                    loss = torch.tensor(0.0, requires_grad=True, device=device)

            # 勾配計算と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"エポック {epoch+1} 損失: {avg_loss:.4f}")

    # モデル保存
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ モデルを保存しました: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="CLIPモデルのファインチューニング")
    parser.add_argument("--csv", type=str, default="data/flickr8k_sample/captions.csv", help="キャプションCSVファイルのパス")
    parser.add_argument("--images", type=str, default="data/flickr8k_sample/images", help="画像ディレクトリのパス")
    parser.add_argument("--output", type=str, default="models/finetuned_clip", help="モデル保存先ディレクトリ")
    parser.add_argument("--batch-size", type=int, default=4, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学習率")
    args = parser.parse_args()
    
    # ファインチューニング実行
    train_clip(
        csv_path=args.csv,
        image_dir=args.images,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr
    )

if __name__ == "__main__":
    main()