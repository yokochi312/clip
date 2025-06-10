import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm
import shutil
import random

def download_file(url, save_path):
    """
    ファイルをダウンロードする
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(save_path, 'wb') as f:
        for data in tqdm(response.iter_content(block_size), total=total_size//block_size, desc=f"ダウンロード中: {os.path.basename(save_path)}"):
            f.write(data)

def extract_zip(zip_path, extract_to):
    """
    ZIPファイルを解凍する
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"解凍中: {os.path.basename(zip_path)}"):
            zip_ref.extract(member, extract_to)

def prepare_flickr8k_sample():
    """
    Flickr8kデータセットのサンプルを準備する
    """
    # データ保存用ディレクトリ
    data_dir = "data"
    sample_dir = os.path.join(data_dir, "flickr8k_sample")
    os.makedirs(data_dir, exist_ok=True)
    
    # Flickr8kの画像とキャプションのURLを設定
    # 注意: 実際にはこのURLは変更されている可能性があります
    flickr8k_images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    flickr8k_text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # ダウンロード先のパス
    images_zip_path = os.path.join(data_dir, "Flickr8k_Dataset.zip")
    text_zip_path = os.path.join(data_dir, "Flickr8k_text.zip")
    
    # 画像をダウンロード
    if not os.path.exists(images_zip_path):
        print("画像データをダウンロードしています...")
        try:
            download_file(flickr8k_images_url, images_zip_path)
        except Exception as e:
            print(f"画像のダウンロードに失敗しました: {e}")
            print("手動でダウンロードして data/ ディレクトリに配置してください。")
    
    # テキストデータをダウンロード
    if not os.path.exists(text_zip_path):
        print("テキストデータをダウンロードしています...")
        try:
            download_file(flickr8k_text_url, text_zip_path)
        except Exception as e:
            print(f"テキストデータのダウンロードに失敗しました: {e}")
            print("手動でダウンロードして data/ ディレクトリに配置してください。")
    
    # 解凍
    images_dir = os.path.join(data_dir, "Flickr8k_Dataset")
    text_dir = os.path.join(data_dir, "Flickr8k_text")
    
    if not os.path.exists(images_dir) and os.path.exists(images_zip_path):
        print("画像データを解凍しています...")
        extract_zip(images_zip_path, data_dir)
    
    if not os.path.exists(text_dir) and os.path.exists(text_zip_path):
        print("テキストデータを解凍しています...")
        extract_zip(text_zip_path, data_dir)
    
    # サンプルデータ用のディレクトリを作成
    sample_images_dir = os.path.join(sample_dir, "images")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    # キャプションファイルを読み込む
    captions_file = os.path.join(data_dir, "Flickr8k.token.txt")
    if not os.path.exists(captions_file):
        captions_file = os.path.join(text_dir, "Flickr8k.token.txt")
        if not os.path.exists(captions_file):
            print(f"キャプションファイルが見つかりません: {captions_file}")
            return False
    
    # キャプションを読み込む
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_caption_id, caption = parts
                image_id = image_caption_id.split('#')[0]
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
    
    # 画像ディレクトリ
    original_images_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    if not os.path.exists(original_images_dir):
        # 別のパスも試す
        original_images_dir = os.path.join(data_dir, "Flicker8k_Dataset")
        if not os.path.exists(original_images_dir):
            print(f"画像ディレクトリが見つかりません: {original_images_dir}")
            return False
    
    # サンプル数を設定
    num_samples = 100
    
    # ランダムにサンプル画像を選択
    all_image_ids = list(captions.keys())
    if len(all_image_ids) > num_samples:
        sample_image_ids = random.sample(all_image_ids, num_samples)
    else:
        sample_image_ids = all_image_ids
    
    # サンプル画像をコピー
    print(f"{len(sample_image_ids)}枚の画像をサンプルとしてコピーしています...")
    sample_captions = []
    
    for image_id in tqdm(sample_image_ids, desc="画像をコピー中"):
        src_path = os.path.join(original_images_dir, image_id)
        dst_path = os.path.join(sample_images_dir, image_id)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            
            # 各画像の最初のキャプションを使用
            if image_id in captions and captions[image_id]:
                sample_captions.append({
                    "filename": image_id,
                    "caption": captions[image_id][0]
                })
    
    # キャプションをCSVとして保存
    captions_df = pd.DataFrame(sample_captions)
    captions_csv_path = os.path.join(sample_dir, "captions.csv")
    captions_df.to_csv(captions_csv_path, index=False)
    
    print(f"サンプルデータの準備が完了しました。")
    print(f"画像: {sample_images_dir}")
    print(f"キャプション: {captions_csv_path}")
    
    return True

if __name__ == "__main__":
    prepare_flickr8k_sample() 