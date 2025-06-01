#!/usr/bin/env python3
"""
Feedback Enhanced Training - UTKFace + Geri Bildirimler
- UTKFace dataset (1979 örnek)
- Kullanıcı feedback'leri (1 örnek)
- Başarılı parametreler korunuyor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.age_training_service import AgeTrainingService
import numpy as np
import cv2
import logging
from tqdm import tqdm

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_utkface_data_feedback_enhanced(image_dir, max_samples=2000):
    """
    UTKFace verilerini yükle (feedback enhanced training için)
    """
    logger.info(f"UTKFace verilerini yüklüyorum: {image_dir}")
    
    # Flask app context
    app = create_app()
    with app.app_context():
        # Doğrudan InsightFace recognition model yükle
        import insightface
        from config import Config
        
        model_path = Config.INSIGHTFACE_AGE_MODEL_ACTIVE_PATH
        rec_model_path = os.path.join(model_path, "w600k_r50.onnx")
        
        if not os.path.exists(rec_model_path):
            logger.error(f"Recognition model bulunamadı: {rec_model_path}")
            return None
        
        logger.info(f"Recognition model yükleniyor: {rec_model_path}")
        rec_model = insightface.model_zoo.get_model(
            rec_model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # UTKFace dosyalarını al
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        logger.info(f"Toplam {len(image_files)} UTKFace dosyası bulundu")
        
        if max_samples > 0 and len(image_files) > max_samples:
            image_files = image_files[:max_samples]
            logger.info(f"İlk {max_samples} dosya ile sınırlandırıldı")
        
        embeddings = []
        ages = []
        valid_count = 0
        
        for filename in tqdm(image_files, desc="UTKFace verilerini işliyorum"):
            try:
                # Dosya adından yaş bilgisini çıkar
                age_str = filename.split('_')[0]
                age = int(age_str)
                
                # Yaş aralığı: 0-100 (başarılı parametre!)
                if not (0 <= age <= 100):
                    continue
                
                # Resmi yükle
                image_path = os.path.join(image_dir, filename)
                img = cv2.imread(image_path)
                
                if img is None:
                    continue
                
                # UTKFace resimleri zaten kırpılmış, doğrudan embedding çıkar
                face_img = cv2.resize(img, (112, 112))
                
                # Recognition model ile doğrudan embedding çıkar
                embedding_result = rec_model.get_feat(face_img)
                face_embedding = embedding_result.flatten() if embedding_result is not None else None
                
                if face_embedding is not None:
                    embeddings.append(face_embedding)
                    ages.append(age)
                    valid_count += 1
                    
                    if valid_count % 100 == 0:
                        logger.info(f"İşlenen geçerli veri: {valid_count}")
                
            except Exception as e:
                logger.debug(f"Hata {filename}: {e}")
                continue
        
        # Model cleanup
        del rec_model
    
    logger.info(f"Toplam UTKFace verisi: {valid_count}")
    
    if valid_count > 0:
        logger.info(f"Yaş aralığı: {min(ages):.1f} - {max(ages):.1f}")
        logger.info(f"Ortalama yaş: {np.mean(ages):.1f}")
        
        return {
            'embeddings': np.array(embeddings),
            'ages': np.array(ages),
            'sources': ['utkface'] * len(embeddings),
            'confidence_scores': np.ones(len(embeddings)),  # UTKFace verilerine tam güven
            'feedback_ids': list(range(len(embeddings)))  # Dummy IDs
        }
    else:
        logger.error("Hiç UTKFace verisi işlenemedi!")
        return None

def combine_data(utkface_data, feedback_data):
    """
    UTKFace ve Feedback verilerini birleştir
    """
    logger.info("UTKFace + Feedback verileri birleştiriliyor...")
    
    if utkface_data is None:
        logger.error("UTKFace verisi bulunamadı!")
        return feedback_data
    
    if feedback_data is None or len(feedback_data['embeddings']) == 0:
        logger.warning("Feedback verisi bulunamadı, sadece UTKFace kullanılacak")
        return utkface_data
    
    # Verileri birleştir
    combined_embeddings = np.vstack([utkface_data['embeddings'], feedback_data['embeddings']])
    combined_ages = np.concatenate([utkface_data['ages'], feedback_data['ages']])
    combined_sources = utkface_data['sources'] + feedback_data['sources']
    combined_confidence = np.concatenate([utkface_data['confidence_scores'], feedback_data['confidence_scores']])
    combined_feedback_ids = utkface_data['feedback_ids'] + feedback_data['feedback_ids']
    
    logger.info(f"Birleştirilmiş veri:")
    logger.info(f"  UTKFace: {len(utkface_data['embeddings'])} örnek")
    logger.info(f"  Feedback: {len(feedback_data['embeddings'])} örnek")
    logger.info(f"  Toplam: {len(combined_embeddings)} örnek")
    
    return {
        'embeddings': combined_embeddings,
        'ages': combined_ages,
        'sources': combined_sources,
        'confidence_scores': combined_confidence,
        'feedback_ids': combined_feedback_ids
    }

def main():
    print("🔄 Feedback Enhanced Training - UTKFace + Geri Bildirimler")
    print("=" * 65)
    print("📊 UTKFace dataset (1979 örnek)")
    print("💬 Kullanıcı feedback'leri (1 örnek)")
    print("⚙️  Başarılı parametreler: epochs=20, batch_size=64, lr=0.001")
    print("🏗️  Network: [512, 256, 128]")
    print("🔧 Normalizasyon KORUNUYOR")
    print()
    
    # UTKFace klasörü
    utkface_dir = "storage/models/age/archive/UTKFace"
    
    if not os.path.exists(utkface_dir):
        print(f"❌ UTKFace klasörü bulunamadı: {utkface_dir}")
        return
    
    # Flask app context
    app = create_app()
    with app.app_context():
        
        # 1. UTKFace verilerini yükle
        print("\n📊 UTKFACE VERİLERİ YÜKLENİYOR:")
        print("-" * 45)
        
        utkface_data = load_utkface_data_feedback_enhanced(utkface_dir, max_samples=2000)
        
        if utkface_data is None:
            print("❌ UTKFace verisi yüklenemedi")
            return
        
        print(f"✅ UTKFace verisi yüklendi: {len(utkface_data['embeddings'])} örnek")
        
        # 2. Feedback verilerini yükle
        print("\n💬 FEEDBACK VERİLERİ YÜKLENİYOR:")
        print("-" * 45)
        
        service = AgeTrainingService()
        feedback_data = service.prepare_training_data(min_samples=1)
        
        if feedback_data is None or len(feedback_data['embeddings']) == 0:
            print("⚠️  Feedback verisi bulunamadı, sadece UTKFace kullanılacak")
            combined_data = utkface_data
        else:
            print(f"✅ Feedback verisi yüklendi: {len(feedback_data['embeddings'])} örnek")
            
            # 3. Verileri birleştir
            print("\n🔄 VERİLER BİRLEŞTİRİLİYOR:")
            print("-" * 45)
            
            combined_data = combine_data(utkface_data, feedback_data)
        
        print(f"✅ Toplam eğitim verisi: {len(combined_data['embeddings'])} örnek")
        
        # 4. Model eğitimi - BAŞARILI PARAMETRELERLİ
        print("\n🎯 FEEDBACK ENHANCED MODEL EĞİTİMİ:")
        print("-" * 45)
        
        # BAŞARILI PARAMETRELERİ KULLAN!
        params = {
            'epochs': 20,  # Başarılı: 20 epoch
            'batch_size': 64,  # Başarılı: 64 batch size
            'learning_rate': 0.001,  # Başarılı: 0.001 learning rate
            'hidden_dims': [512, 256, 128],  # Başarılı: [512, 256, 128] network
            'test_size': 0.2,  # Başarılı: 0.2 test split
            'early_stopping_patience': 5  # Başarılı: 5 patience
        }
        
        print(f"⚙️  Feedback Enhanced eğitim parametreleri:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Eğitimi başlat
        print(f"\n🚀 Feedback Enhanced eğitim başlıyor ({len(combined_data['embeddings'])} örnek ile)...")
        result = service.train_model(combined_data, params)
        
        if result:
            print("\n✅ FEEDBACK ENHANCED EĞİTİM BAŞARILI!")
            print("📊 Final Metrikler:")
            metrics = result['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            
            # Performans değerlendirmesi
            mae = metrics['mae']
            within_3 = metrics['within_3_years']
            
            print("\n📈 PERFORMANS DEĞERLENDİRMESİ:")
            print(f"   Önceki Model: MAE: 1.652, 3-yıl: 80.1%")
            
            if mae < 1.7:
                print(f"   🟢 YENİ MODEL MAE: {mae:.3f} - MÜKEMMEL!")
            elif mae < 2.0:
                print(f"   🟡 YENİ MODEL MAE: {mae:.3f} - İYİ")
            else:
                print(f"   🔴 YENİ MODEL MAE: {mae:.3f} - KÖTÜ")
            
            if within_3 > 0.80:
                print(f"   🟢 YENİ MODEL 3-yıl: {within_3:.3f} - MÜKEMMEL!")
            elif within_3 > 0.75:
                print(f"   🟡 YENİ MODEL 3-yıl: {within_3:.3f} - İYİ")
            else:
                print(f"   🔴 YENİ MODEL 3-yıl: {within_3:.3f} - KÖTÜ")
            
            # Model versiyonu kaydet
            print("\n💾 Model kaydediliyor...")
            version_name = f"feedback_enhanced_v1_{len(combined_data['embeddings'])}_samples"
            version = service.save_model_version(result['model'], result, version_name)
            print(f"✅ Feedback Enhanced model versiyonu kaydedildi: {version.version_name}")
            
            # Model'i aktifleştir
            print(f"\n🔄 Model aktifleştiriliyor...")
            activation_result = service.activate_model_version(version.id)
            if activation_result:
                print(f"✅ Model başarıyla aktifleştirildi!")
            else:
                print(f"⚠️  Model aktifleştirme sorunu yaşandı")
            
            # Performance test
            print("\n👶 BABY YAŞ TAHMİNİ TESTİ:")
            test_baby_prediction_feedback_enhanced()
            
        else:
            print("❌ Feedback Enhanced eğitim başarısız!")

def test_baby_prediction_feedback_enhanced():
    """Feedback Enhanced baby yaş tahmini testi"""
    test_image = "storage/uploads/bebek.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test dosyası bulunamadı: {test_image}")
        return
    
    print(f"🔍 Test dosyası: {test_image}")
    
    from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
    
    # Age Estimator başlat (Feedback Enhanced model ile)
    estimator = InsightFaceAgeEstimator()
    
    # Resmi yükle
    img = cv2.imread(test_image)
    if img is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Yüz tespit et
    faces = estimator.model.get(img)
    
    if faces:
        face = faces[0]
        result = estimator.estimate_age(img, face)
        
        if result:
            # result 3 değer döndürüyor: (age, confidence, pseudo_label_data)
            if len(result) == 3:
                final_age, confidence, _ = result
            else:
                final_age, confidence = result
            
            print(f"🎂 Feedback Enhanced Model Yaş Tahmini: {final_age}")
            print(f"🔒 Güven: {confidence:.4f}")
            
            # Karşılaştırma
            print(f"📊 Önceki Model Tahmini: 9 yaş")
            print(f"📊 Kullanıcı Geri Bildirimi: 3 yaş (gerçek)")
            
            # Başarı değerlendirmesi
            if final_age < 5:
                print("✅ MÜKEMMEL! Feedback Enhanced model gerçek yaşa çok yakın!")
            elif final_age < 8:
                print("✅ İYİ! Feedback Enhanced model iyileşme gösteriyor!")
            elif final_age < 12:
                print("🟡 ORTA! Hala çocuk yaş aralığında")
            else:
                print("🔴 KÖTÜ! Feedback etkisi görülmüyor")
                
        else:
            print("❌ Yaş tahmini başarısız!")
    else:
        print("❌ Yüz tespit edilemedi!")
    
    # Cleanup
    if hasattr(estimator, 'cleanup'):
        estimator.cleanup()

if __name__ == "__main__":
    main() 