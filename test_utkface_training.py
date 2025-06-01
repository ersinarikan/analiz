#!/usr/bin/env python3
"""
UTKFace Dataset ile Yaş Tahmini Model Eğitimi Test
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

def load_utkface_data(image_dir, max_samples=5000):
    """
    UTKFace dataset'inden yaş verilerini yükle
    
    Args:
        image_dir: UTKFace resimlerinin bulunduğu klasör
        max_samples: Maksimum örnek sayısı
        
    Returns:
        dict: Eğitim verisi (embeddings, ages)
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
                # Format: age_gender_race_timestamp.jpg.chip.jpg
                age_str = filename.split('_')[0]
                age = int(age_str)
                
                # Yaş validasyonu
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
    print("🎓 UTKFace Dataset ile Yaş Tahmini Model Eğitimi")
    print("=" * 55)
    
    # UTKFace klasörü
    utkface_dir = "storage/models/age/archive/UTKFace"
    
    if not os.path.exists(utkface_dir):
        print(f"❌ UTKFace klasörü bulunamadı: {utkface_dir}")
        print("💡 Dataset'i storage/models/age/archive/UTKFace altına kopyalayın")
        return
    
    # Flask app context
    app = create_app()
    with app.app_context():
        
        # 1. UTKFace verilerini yükle
        print("\n📊 UTKFACE VERİLERİ YÜKLENİYOR:")
        print("-" * 40)
        
        training_data = load_utkface_data(utkface_dir, max_samples=2000)  # Test için 2K sample
        
        if training_data is None or len(training_data['embeddings']) < 50:
            print("❌ Yeterli UTKFace verisi yüklenemedi")
            return
        
        print(f"✅ UTKFace verisi yüklendi: {len(training_data['embeddings'])} örnek")
        
        # 2. Model eğitimi
        print("\n🎯 MODEL EĞİTİMİ:")
        print("-" * 40)
        
        service = AgeTrainingService()
        
        # Eğitim parametreleri
        params = {
            'epochs': 20,  # UTKFace için daha çok epoch
            'batch_size': 64,
            'learning_rate': 0.001,
            'hidden_dims': [512, 256, 128],  # Daha derin network
            'test_size': 0.2,
            'early_stopping_patience': 5
        }
        
        print(f"⚙️  Eğitim parametreleri: {params}")
        
        # Eğitimi başlat
        result = service.train_model(training_data, params)
        
        if result:
            print("\n✅ EĞİTİM BAŞARILI!")
            print("📊 Final Metrikler:")
            metrics = result['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            
            # Model kaydet
            print("\n💾 Model kaydediliyor...")
            version = service.save_model_version(
                result['model'], 
                result, 
                f"utkface_v1_{len(training_data['embeddings'])}_samples"
            )
            print(f"✅ UTKFace model versiyonu kaydedildi: {version.version_name}")
            
            # Test baby yaş tahmini
            print("\n👶 BABY YAŞ TAHMİNİ TESTİ:")
            test_baby_prediction(service)
            
        else:
            print("❌ Eğitim başarısız!")

def test_baby_prediction(service):
    """Baby yaş tahmini testi"""
    test_image = "storage/uploads/bebek.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test dosyası bulunamadı: {test_image}")
        return
    
    print(f"🔍 Test dosyası: {test_image}")
    
    from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
    
    # Age Estimator başlat
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
            # result tuple olarak dönüyor: (age, confidence)
            final_age, confidence = result
            
            print(f"🎂 UTKFace Model Yaş Tahmini: {final_age}")
            print(f"🔒 Güven: {confidence:.4f}")
            
            # Doğrulama
            if final_age < 15:
                print("✅ BAŞARILI! UTKFace modeli çocuk yaşını doğru tespit etti!")
            else:
                print("⚠️  Dikkat: UTKFace modeli yüksek yaş tahmin etti")
                
        else:
            print("❌ Yaş tahmini başarısız!")
    else:
        print("❌ Yüz tespit edilemedi!")
    
    # Cleanup
    if hasattr(estimator, 'cleanup'):
        estimator.cleanup()

if __name__ == "__main__":
    main() 