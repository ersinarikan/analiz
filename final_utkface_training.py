#!/usr/bin/env python3
"""
Kapsamlı UTKFace Dataset ile Custom Age Head Eğitimi
- Normalizasyon KORUNUYOR (kritik!)
- Daha fazla veri ile eğitim
- Model performans optimizasyonu
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

def load_comprehensive_utkface_data(image_dir, max_samples=10000):
    """
    UTKFace dataset'inden kapsamlı yaş verilerini yükle
    
    Args:
        image_dir: UTKFace resimlerinin bulunduğu klasör
        max_samples: Maksimum örnek sayısı (0 = hepsi)
        
    Returns:
        dict: Eğitim verisi (embeddings, ages)
    """
    logger.info(f"Kapsamlı UTKFace verilerini yüklüyorum: {image_dir}")
    
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
        
        # Yaş gruplarına göre dengeli seçim yapalım
        age_groups = {}
        
        # Önce dosyaları yaş gruplarına ayır
        for filename in image_files:
            try:
                age_str = filename.split('_')[0]
                age = int(age_str)
                
                # Pratik yaş aralığı: 0-85 yaş (100+ yaş çok nadir ve gürültülü)
                if not (0 <= age <= 85):
                    continue
                
                # Yaş grupları: 0-10, 11-20, 21-30, ..., 80-85
                age_group = (age // 10) * 10
                
                if age_group not in age_groups:
                    age_groups[age_group] = []
                age_groups[age_group].append((filename, age))
                
            except (ValueError, IndexError):
                continue
        
        logger.info(f"Yaş grupları: {[(k, len(v)) for k, v in age_groups.items()]}")
        
        # Her yaş grubundan dengeli seçim yap
        selected_files = []
        if max_samples > 0:
            samples_per_group = max(50, max_samples // len(age_groups))  # Minimum 50, maksimum eşit dağıtım
            logger.info(f"Her yaş grubundan maksimum {samples_per_group} örnek seçiliyor")
            
            for age_group, files in age_groups.items():
                selected_count = min(len(files), samples_per_group)
                selected_files.extend(files[:selected_count])
                logger.info(f"Yaş grubu {age_group}-{age_group+9}: {selected_count}/{len(files)} seçildi")
        else:
            # Tüm dosyaları kullan
            for files in age_groups.values():
                selected_files.extend(files)
        
        logger.info(f"Toplam seçilen dosya sayısı: {len(selected_files)}")
        
        embeddings = []
        ages = []
        valid_count = 0
        failed_count = 0
        
        for filename, age in tqdm(selected_files, desc="UTKFace verilerini işliyorum"):
            try:
                # Resmi yükle
                image_path = os.path.join(image_dir, filename)
                img = cv2.imread(image_path)
                
                if img is None:
                    failed_count += 1
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
                    
                    if valid_count % 500 == 0:
                        logger.info(f"İşlenen geçerli veri: {valid_count}")
                else:
                    failed_count += 1
                
            except Exception as e:
                logger.debug(f"Hata {filename}: {e}")
                failed_count += 1
                continue
        
        # Model cleanup
        del rec_model
    
    logger.info(f"Toplam geçerli veri: {valid_count}")
    logger.info(f"Başarısız işlem: {failed_count}")
    
    if valid_count > 0:
        logger.info(f"Yaş aralığı: {min(ages):.1f} - {max(ages):.1f}")
        logger.info(f"Ortalama yaş: {np.mean(ages):.1f}")
        
        # Yaş dağılımını göster
        age_distribution = {}
        for age in ages:
            decade = (age // 10) * 10
            age_distribution[decade] = age_distribution.get(decade, 0) + 1
        
        logger.info("Yaş dağılımı:")
        for decade in sorted(age_distribution.keys()):
            logger.info(f"  {decade}-{decade+9} yaş: {age_distribution[decade]} örnek")
        
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
    print("🎓 Kapsamlı UTKFace Dataset ile Custom Age Head Eğitimi")
    print("=" * 60)
    print("🔧 Normalizasyon KORUNUYOR (kritik!)")
    print("📊 Dengeli yaş dağılımı ile eğitim")
    print("🚀 Optimized model parametreleri")
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
        
        # 1. UTKFace verilerini yükle
        print("\n📊 KAPSAMLI UTKFACE VERİLERİ YÜKLENİYOR:")
        print("-" * 50)
        
        # Daha fazla veri ile eğitim (maksimum 15K örnek)
        training_data = load_comprehensive_utkface_data(utkface_dir, max_samples=15000)
        
        if training_data is None or len(training_data['embeddings']) < 100:
            print("❌ Yeterli UTKFace verisi yüklenemedi")
            return
        
        print(f"✅ Kapsamlı UTKFace verisi yüklendi: {len(training_data['embeddings'])} örnek")
        
        # 2. Model eğitimi
        print("\n🎯 KAPSAMLI MODEL EĞİTİMİ:")
        print("-" * 50)
        
        service = AgeTrainingService()
        
        # Optimized eğitim parametreleri
        params = {
            'epochs': 100,  # Daha fazla epoch (early stopping var)
            'batch_size': 128,  # Daha büyük batch size
            'learning_rate': 0.0005,  # Biraz daha düşük learning rate
            'hidden_dims': [512, 256, 128, 64],  # Daha derin network
            'test_size': 0.15,  # Daha fazla training data
            'early_stopping_patience': 15  # Daha fazla patience
        }
        
        print(f"⚙️  Optimized eğitim parametreleri:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Eğitimi başlat
        print(f"\n🚀 Eğitim başlıyor ({len(training_data['embeddings'])} örnek ile)...")
        result = service.train_model(training_data, params)
        
        if result:
            print("\n✅ KAPSAMLI EĞİTİM BAŞARILI!")
            print("📊 Final Metrikler:")
            metrics = result['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            
            # Model versiyonu kaydet
            print("\n💾 Model kaydediliyor...")
            version_name = f"utkface_comprehensive_v1_{len(training_data['embeddings'])}_samples"
            version = service.save_model_version(result['model'], result, version_name)
            print(f"✅ Kapsamlı UTKFace model versiyonu kaydedildi: {version.version_name}")
            
            # Model'i aktifleştir
            print(f"\n🔄 Model aktifleştiriliyor...")
            activation_result = service.activate_model_version(version.id)
            if activation_result:
                print(f"✅ Model başarıyla aktifleştirildi!")
            else:
                print(f"⚠️  Model aktifleştirme sorunu yaşandı")
            
            # Performance test
            print("\n👶 BABY YAŞ TAHMİNİ TESTİ:")
            test_baby_prediction_comprehensive()
            
        else:
            print("❌ Kapsamlı eğitim başarısız!")

def test_baby_prediction_comprehensive():
    """Kapsamlı baby yaş tahmini testi"""
    test_image = "storage/uploads/bebek.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test dosyası bulunamadı: {test_image}")
        return
    
    print(f"🔍 Test dosyası: {test_image}")
    
    from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
    
    # Age Estimator başlat (yeni model ile)
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
            
            print(f"🎂 Kapsamlı UTKFace Model Yaş Tahmini: {final_age}")
            print(f"🔒 Güven: {confidence:.4f}")
            
            # Başarı değerlendirmesi
            if final_age < 12:
                print("✅ MÜKEMMEL! Kapsamlı UTKFace modeli çocuk yaşını çok doğru tespit etti!")
            elif final_age < 18:
                print("✅ İYİ! Kapsamlı UTKFace modeli genç yaşını başarıyla tespit etti!")
            else:
                print("⚠️  Dikkat: Model hala yüksek yaş tahmin ediyor")
                
        else:
            print("❌ Yaş tahmini başarısız!")
    else:
        print("❌ Yüz tespit edilemedi!")
    
    # Cleanup
    if hasattr(estimator, 'cleanup'):
        estimator.cleanup()

if __name__ == "__main__":
    main() 