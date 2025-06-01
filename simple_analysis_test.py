#!/usr/bin/env python3
"""
Basit Analiz Test - Geri Bildirim Tablosuna Veri Ekleme
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
import cv2

def main():
    print("🔍 BASİT ANALİZ TESTİ")
    print("=" * 40)
    print("📊 Amaç: Geri bildirim tablosuna örnek veri ekleme")
    print()
    
    # Flask app context
    app = create_app()
    with app.app_context():
        
        # Test resmi
        test_image = "storage/uploads/bebek.jpg"
        
        if not os.path.exists(test_image):
            print(f"❌ Test dosyası bulunamadı: {test_image}")
            return
        
        print(f"🔍 Analiz edilen dosya: {test_image}")
        
        # Age Estimator başlat
        estimator = InsightFaceAgeEstimator()
        
        # Resmi yükle
        img = cv2.imread(test_image)
        if img is None:
            print("❌ Resim yüklenemedi!")
            return
        
        print(f"📐 Resim boyutu: {img.shape}")
        
        # Yüz tespit et
        faces = estimator.model.get(img)
        
        if faces:
            print(f"👥 Tespit edilen yüz sayısı: {len(faces)}")
            
            for i, face in enumerate(faces):
                print(f"\n👤 Yüz {i+1}:")
                print(f"   📍 Bbox: {face.bbox}")
                print(f"   🎯 Güven: {face.det_score:.3f}")
                
                # Yaş tahmini yap
                result = estimator.estimate_age(img, face)
                
                if result:
                    # result 3 değer döndürüyor: (age, confidence, pseudo_label_data)
                    if len(result) == 3:
                        final_age, confidence, _ = result
                    else:
                        final_age, confidence = result
                    
                    print(f"   🎂 Tahmin edilen yaş: {final_age}")
                    print(f"   🔒 Model güveni: {confidence:.4f}")
                    
                    # Bu noktada manuel geri bildirim ekleyebiliriz
                    print(f"\n💡 Geri bildirim örneği:")
                    print(f"   - Tahmin: {final_age} yaş")
                    print(f"   - Gerçek yaş (manuel): Kullanıcı 3 yaş girerse")
                    print(f"   - Bu feedback tablosuna kaydedilebilir")
                    
                else:
                    print("   ❌ Yaş tahmini başarısız!")
        else:
            print("❌ Hiç yüz tespit edilemedi!")
        
        # Cleanup
        if hasattr(estimator, 'cleanup'):
            estimator.cleanup()
        
        print(f"\n✅ Analiz tamamlandı!")
        print(f"📝 Şimdi manuel geri bildirim ekleyebilirsiniz")
        print(f"🌐 Web arayüzünden: http://localhost:5000")

if __name__ == "__main__":
    main() 