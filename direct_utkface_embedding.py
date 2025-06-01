#!/usr/bin/env python3
"""
UTKFace'den doğrudan embedding çıkarma
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
import cv2
import numpy as np
import insightface
from config import Config

def extract_embedding_directly():
    """UTKFace resimlerinden doğrudan embedding çıkar"""
    
    # UTKFace klasörü
    utkface_dir = "storage/models/age/archive/UTKFace"
    
    if not os.path.exists(utkface_dir):
        print(f"❌ UTKFace klasörü bulunamadı: {utkface_dir}")
        return
    
    # İlk dosyayı al
    files = [f for f in os.listdir(utkface_dir) if f.lower().endswith('.jpg')][:5]
    
    if not files:
        print("❌ Hiç jpg dosyası bulunamadı")
        return
    
    print(f"🔍 Test edilen dosyalar: {files}")
    
    # Flask app context
    app = create_app()
    with app.app_context():
        # InsightFace direkt model
        try:
            model_path = Config.INSIGHTFACE_AGE_MODEL_ACTIVE_PATH
            rec_model_path = os.path.join(model_path, "w600k_r50.onnx")
            
            if not os.path.exists(rec_model_path):
                print(f"❌ Recognition model bulunamadı: {rec_model_path}")
                return
            
            print(f"✅ Recognition model: {rec_model_path}")
            
            # InsightFace model
            rec_model = insightface.model_zoo.get_model(
                rec_model_path, 
                providers=['CPUExecutionProvider']
            )
            
            print("✅ Recognition model yüklendi")
            
            for filename in files:
                print(f"\n📸 Test: {filename}")
                
                try:
                    # Dosya adından yaş bilgisini çıkar
                    age_str = filename.split('_')[0]
                    age = int(age_str)
                    print(f"  📅 Yaş: {age}")
                    
                    # Resmi yükle
                    image_path = os.path.join(utkface_dir, filename)
                    img = cv2.imread(image_path)
                    
                    if img is None:
                        print(f"  ❌ Resim yüklenemedi")
                        continue
                    
                    print(f"  ✅ Resim yüklendi: {img.shape}")
                    
                    # UTKFace resimlerinin zaten aligned olduğunu varsayalım
                    # Boyutu 112x112'ye getir (InsightFace standart)
                    face_img = cv2.resize(img, (112, 112))
                    
                    # Embedding çıkar
                    try:
                        embedding = rec_model.get_feat(face_img)
                        
                        if embedding is not None:
                            print(f"  ✅ Embedding çıkarıldı: shape={embedding.shape}")
                            print(f"  📊 Embedding range: {embedding.min():.3f} - {embedding.max():.3f}")
                        else:
                            print(f"  ❌ Embedding çıkarılamadı: None döndü")
                    except Exception as e:
                        print(f"  ❌ Embedding hatası: {e}")
                
                except Exception as e:
                    print(f"  ❌ Genel hata: {e}")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")

if __name__ == "__main__":
    extract_embedding_directly() 