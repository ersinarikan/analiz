#!/usr/bin/env python3
"""
Final Baby Test - Normalizasyon Fix'i Doğrulama
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
import cv2

def main():
    print("👶 FINAL BABY TEST - Normalizasyon Fix Doğrulaması")
    print("=" * 55)
    
    # Flask app context
    app = create_app()
    with app.app_context():
        
        # Test dosyası
        test_image = "storage/uploads/bebek.jpg"
        
        if not os.path.exists(test_image):
            print(f"❌ Test dosyası bulunamadı: {test_image}")
            return
        
        print(f"🔍 Test dosyası: {test_image}")
        
        # Age Estimator başlat
        print("🤖 Age Estimator başlatılıyor...")
        estimator = InsightFaceAgeEstimator()
        
        # Resmi yükle
        img = cv2.imread(test_image)
        if img is None:
            print("❌ Resim yüklenemedi!")
            return
        
        # Yüz tespit et
        print("\n🎯 YAŞ TAHMİNİ:")
        faces = estimator.app.get(img)
        
        if faces:
            face = faces[0]
            result = estimator.estimate_age(img, face)
            
            if result:
                final_age = result['age']
                confidence = result['confidence']
                
                print(f"🎂 Final Yaş: {final_age}")
                print(f"🔒 Güven: {confidence:.4f}")
                
                # Doğrulama
                if final_age < 15:
                    print("✅ BAŞARILI! Çocuk yaşı doğru tespit edildi!")
                    print(f"📈 Normalizasyon fix'i çalışıyor!")
                else:
                    print("❌ HATA! Çocuk yaşı yanlış tespit edildi!")
                    print(f"⚠️  Normalizasyon sorunu devam ediyor...")
                    
            else:
                print("❌ Yaş tahmini başarısız!")
        else:
            print("❌ Yüz tespit edilemedi!")
        
        # Cleanup
        estimator.cleanup()

if __name__ == "__main__":
    main() 