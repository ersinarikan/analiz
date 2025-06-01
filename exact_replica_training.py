#!/usr/bin/env python3
"""
EXACT REPLICA - İlk Başarılı UTKFace Eğitiminin Tam Kopyası
- Yaş aralığı: 0-100 (önceki başarılı gibi!)
- Aynı parametreler: epochs=20, batch_size=64, lr=0.001
- Aynı network: [512, 256, 128]
- Normalizasyon KORUNUYOR
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

def load_utkface_data_exact_replica(image_dir, max_samples=2000):
    """
    EXACT REPLICA - İlk başarılı eğitiminin tam kopyası
    - Yaş aralığı: 0-100 (KRİTİK!)
    - Aynı veri işleme
    - Aynı embedding çıkarma
    """
    logger.info(f"EXACT REPLICA - UTKFace verilerini yüklüyorum: {image_dir}")
    
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
                # Format: age_gender_race_timestamp.jpg.chip.jpg
                age_str = filename.split('_')[0]
                age = int(age_str)
                
                # EXACT REPLICA: Yaş aralığı 0-100 (önceki başarılı gibi!)
                if not (0 <= age <= 100):
                    continue
                
                # Resmi yükle
                image_path = os.path.join(image_dir, filename)
                img = cv2.imread(image_path)
                
                if img is None:
                    continue
                
                # UTKFace resimleri zaten kırpılmış, doğrudan embedding çıkar
                # Boyutu 112x112'ye getir (InsightFace standart)
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
        
        # Model cleanup (basit)
        del rec_model
    
    logger.info(f"Toplam geçerli veri: {valid_count}")
    
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
        logger.error("Hiç geçerli veri işlenemedi!")
        return None

def main():
    print("🎯 EXACT REPLICA - İlk Başarılı UTKFace Eğitiminin Tam Kopyası")
    print("=" * 65)
    print("🔥 Yaş aralığı: 0-100 (önceki başarılı gibi!)")
    print("⚙️  AYNI parametreler: epochs=20, batch_size=64, lr=0.001")
    print("🏗️  AYNI network: [512, 256, 128]")
    print("🔧 Normalizasyon KORUNUYOR (kritik!)")
    print("🎯 Hedef: MAE < 2.0, 3-yıl doğruluk > 75%")
    print()
    
    # UTKFace klasörü
    utkface_dir = "storage/models/age/archive/UTKFace"
    
    if not os.path.exists(utkface_dir):
        print(f"❌ UTKFace klasörü bulunamadı: {utkface_dir}")
        print("💡 Dataset'i storage/models/age/archive/UTKFace altına kopyalayın")
        return
    
    # Flask app context
    app = create_app()
    with app.app_context():
        
        # 1. UTKFace verilerini TAM AYNI şekilde yükle
        print("\n📊 EXACT REPLICA UTKFACE VERİLERİ YÜKLENİYOR:")
        print("-" * 50)
        
        # AYNI boyutta eğitim (2K örnek - önceki başarılı ile aynı)
        training_data = load_utkface_data_exact_replica(utkface_dir, max_samples=2000)
        
        if training_data is None or len(training_data['embeddings']) < 50:
            print("❌ Yeterli UTKFace verisi yüklenemedi")
            return
        
        print(f"✅ EXACT REPLICA UTKFace verisi yüklendi: {len(training_data['embeddings'])} örnek")
        
        # 2. Model eğitimi - TAM AYNI PARAMETRELERLİ
        print("\n🎯 EXACT REPLICA MODEL EĞİTİMİ:")
        print("-" * 50)
        
        service = AgeTrainingService()
        
        # EXACT REPLICA PARAMETERS - İlk başarılı eğitimin TAM AYNISI!
        params = {
            'epochs': 20,  # EXACT: 20 epoch
            'batch_size': 64,  # EXACT: 64 batch size
            'learning_rate': 0.001,  # EXACT: 0.001 learning rate
            'hidden_dims': [512, 256, 128],  # EXACT: [512, 256, 128] network
            'test_size': 0.2,  # EXACT: 0.2 test split
            'early_stopping_patience': 5  # EXACT: 5 patience
        }
        
        print(f"⚙️  EXACT REPLICA eğitim parametreleri:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Eğitimi başlat
        print(f"\n🚀 EXACT REPLICA eğitim başlıyor ({len(training_data['embeddings'])} örnek ile)...")
        result = service.train_model(training_data, params)
        
        if result:
            print("\n✅ EXACT REPLICA EĞİTİM BAŞARILI!")
            print("📊 Final Metrikler:")
            metrics = result['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            
            # Performans değerlendirmesi - EXACT COMPARISON
            mae = metrics['mae']
            within_3 = metrics['within_3_years']
            
            print("\n📈 PERFORMANS KARŞILAŞTIRMASI:")
            print(f"   İlk Başarılı Model: MAE: 1.661, 3-yıl: 80.1%")
            
            if mae < 2.0:
                print(f"   🟢 YENİ MODEL MAE: {mae:.3f} - MÜKEMMEL! (hedef: <2.0)")
            elif mae < 3.0:
                print(f"   🟡 YENİ MODEL MAE: {mae:.3f} - İYİ (hedef: <2.0)")
            else:
                print(f"   🔴 YENİ MODEL MAE: {mae:.3f} - KÖTÜ (hedef: <2.0)")
            
            if within_3 > 0.75:
                print(f"   🟢 YENİ MODEL 3-yıl: {within_3:.3f} - MÜKEMMEL! (hedef: >75%)")
            elif within_3 > 0.60:
                print(f"   🟡 YENİ MODEL 3-yıl: {within_3:.3f} - İYİ (hedef: >75%)")
            else:
                print(f"   🔴 YENİ MODEL 3-yıl: {within_3:.3f} - KÖTÜ (hedef: >75%)")
            
            # Model versiyonu kaydet
            print("\n💾 Model kaydediliyor...")
            version_name = f"exact_replica_v1_{len(training_data['embeddings'])}_samples"
            version = service.save_model_version(result['model'], result, version_name)
            print(f"✅ EXACT REPLICA model versiyonu kaydedildi: {version.version_name}")
            
            # Model'i aktifleştir
            print(f"\n🔄 Model aktifleştiriliyor...")
            activation_result = service.activate_model_version(version.id)
            if activation_result:
                print(f"✅ Model başarıyla aktifleştirildi!")
            else:
                print(f"⚠️  Model aktifleştirme sorunu yaşandı")
            
            # Performance test
            print("\n👶 BABY YAŞ TAHMİNİ TESTİ:")
            test_baby_prediction_exact_replica()
            
        else:
            print("❌ EXACT REPLICA eğitim başarısız!")

def test_baby_prediction_exact_replica():
    """EXACT REPLICA baby yaş tahmini testi"""
    test_image = "storage/uploads/bebek.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test dosyası bulunamadı: {test_image}")
        return
    
    print(f"🔍 Test dosyası: {test_image}")
    
    from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
    
    # Age Estimator başlat (EXACT REPLICA model ile)
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
            
            print(f"🎂 EXACT REPLICA Model Yaş Tahmini: {final_age}")
            print(f"🔒 Güven: {confidence:.4f}")
            
            # Karşılaştırma
            print(f"📊 İlk Başarılı Model Tahmini: 9 yaş (Buffalo seçilmişti)")
            
            # Başarı değerlendirmesi
            if final_age < 8:
                print("✅ MÜKEMMEL! EXACT REPLICA modeli bebek yaşını çok doğru tespit etti!")
            elif final_age < 12:
                print("✅ İYİ! EXACT REPLICA modeli çocuk yaşını başarıyla tespit etti!")
            elif final_age < 18:
                print("🟡 ORTA! Genç yaş tahmin etti")
            else:
                print("🔴 KÖTÜ! Hala yüksek yaş tahmin ediyor")
                
        else:
            print("❌ Yaş tahmini başarısız!")
    else:
        print("❌ Yüz tespit edilemedi!")
    
    # Cleanup
    if hasattr(estimator, 'cleanup'):
        estimator.cleanup()

if __name__ == "__main__":
    main() 