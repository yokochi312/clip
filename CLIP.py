import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
from tqdm import tqdm

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
        return self.processor(text=[text], images=image, return_tensors="pt", padding=True)

# 設定
csv_path = "your_dataset/captions.csv"
image_dir = "your_dataset/images"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
lr = 1e-5
num_epochs = 3

# モデル・プロセッサ
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# データローダー
dataset = ImageTextDataset(csv_path, image_dir, processor)
dataloader = DataLoader(dataset, batch_size=batch_size)

# 最適化・学習
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = {k: v.squeeze(0).to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# モデル保存
model.save_pretrained("finetuned_clip_model")
processor.save_pretrained("finetuned_clip_model")
print("✅ モデル保存完了")