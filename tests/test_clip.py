import unittest
import torch
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import tempfile
import pandas as pd

class TestCLIPModel(unittest.TestCase):
    """CLIPモデルの基本機能をテストするクラス"""
    
    @classmethod
    def setUpClass(cls):
        """テスト開始前に一度だけ実行される処理"""
        # モデルとプロセッサをロード
        cls.model_name = "openai/clip-vit-base-patch32"
        cls.model = CLIPModel.from_pretrained(cls.model_name)
        cls.processor = CLIPProcessor.from_pretrained(cls.model_name)
        
        # テスト用の一時ディレクトリを作成
        cls.test_dir = tempfile.mkdtemp()
        
        # テスト用の画像を作成
        cls.test_image = Image.new('RGB', (224, 224), color='red')
        cls.test_image_path = os.path.join(cls.test_dir, 'test_image.jpg')
        cls.test_image.save(cls.test_image_path)
        
        # テスト用のキャプションCSVを作成
        cls.test_captions = pd.DataFrame({
            'filename': ['test_image.jpg'],
            'caption': ['a red image']
        })
        cls.test_csv_path = os.path.join(cls.test_dir, 'test_captions.csv')
        cls.test_captions.to_csv(cls.test_csv_path, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """テスト終了後に一度だけ実行される処理"""
        # テスト用の一時ディレクトリを削除
        import shutil
        shutil.rmtree(cls.test_dir)
    
    def test_model_loading(self):
        """モデルが正しくロードされることをテスト"""
        self.assertIsInstance(self.model, CLIPModel)
        self.assertIsInstance(self.processor, CLIPProcessor)
    
    def test_image_processing(self):
        """画像の処理が正しく行われることをテスト"""
        inputs = self.processor(images=self.test_image, return_tensors="pt")
        self.assertIn('pixel_values', inputs)
        self.assertEqual(inputs.pixel_values.shape[0], 1)  # バッチサイズ
    
    def test_text_processing(self):
        """テキストの処理が正しく行われることをテスト"""
        inputs = self.processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True)
        self.assertIn('input_ids', inputs)
        self.assertEqual(inputs.input_ids.shape[0], 2)  # バッチサイズ
    
    def test_image_text_similarity(self):
        """画像とテキストの類似度計算をテスト"""
        inputs = self.processor(
            text=["a red image", "a blue image"],
            images=self.test_image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).numpy()
        
        # 赤い画像は「a red image」の方が「a blue image」よりも類似度が高いはず
        self.assertGreater(probs[0][0], probs[0][1])
    
    def test_batch_processing(self):
        """バッチ処理が正しく行われることをテスト"""
        # 2つの画像を用意
        images = [self.test_image, self.test_image]
        texts = ["a red image", "a photo of something"]
        
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        self.assertEqual(inputs.pixel_values.shape[0], 2)  # 画像のバッチサイズ
        self.assertEqual(inputs.input_ids.shape[0], 2)     # テキストのバッチサイズ

if __name__ == '__main__':
    unittest.main() 