#!/usr/bin/env python3
"""
UTKFace tek dosya debug
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
import cv2
import logging

# Logging ayarla
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_single_utkface():
    """Tek bir UTKFace dosyası ile debug"""
    
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
        from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
        
        # Age estimator başlat
        estimator = InsightFaceAgeEstimator()
        
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
                    print(f"  ❌ Resim yüklenemedi: {image_path}")
                    continue
                
                print(f"  ✅ Resim yüklendi: {img.shape}")
                
                # Face detection
                faces = estimator.model.get(img)
                print(f"  👤 Tespit edilen yüz sayısı: {len(faces) if faces else 0}")
                
                if not faces:
                    print(f"  ❌ Yüz tespit edilemedi")
                    continue
                
                face = faces[0]
                print(f"  ✅ Yüz bbox: {face.bbox}")
                
                # Embedding çıkar
                try:
                    face_embedding = estimator._extract_face_embedding(img, face)
                    if face_embedding is not None:
                        print(f"  ✅ Embedding çıkarıldı: shape={face_embedding.shape}")
                    else:
                        print(f"  ❌ Embedding çıkarılamadı: None döndü")
                except Exception as e:
                    print(f"  ❌ Embedding hatası: {e}")
                
            except Exception as e:
                print(f"  ❌ Genel hata: {e}")
        
        # Cleanup
        if hasattr(estimator, 'cleanup'):
            estimator.cleanup()

if __name__ == "__main__":
    debug_single_utkface() 