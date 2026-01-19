"""
Custom Age modelini geri bildirimlerle eğitme test scripti
"""
import os
import sys
import logging

# Proje kök dizinini Python path'e ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app import create_app, db
from app.services.age_training_service import AgeTrainingService
from app.models.feedback import Feedback
from sqlalchemy import inspect
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_age_training():
    """Age training sistemini test eder"""
    app, socketio_direct = create_app(return_socketio=True)  # (app, socketio)
    
    with app.app_context():
        # ModelVersion tablosunun var olduğunu kontrol et
        inspector = db.inspect(db.engine)
        if 'model_versions' not in inspector.get_table_names():
            print("HATA: model_versions tablosu bulunamadı!")
            print("Lütfen veritabanını yeniden oluşturun veya app.py'yi başlatın.")
            return
            
        # Servisi başlat
        trainer = AgeTrainingService()
        
        # 1. Mevcut veri istatistiklerini göster
        print("\n=== Mevcut Eğitim Verisi İstatistikleri ===")
        training_data = trainer.prepare_training_data(min_samples=1)
        
        if training_data is None or len(training_data['embeddings']) < 2:
            print("Henüz yeterli geri bildirim verisi yok veya örnek sayısı yetersiz.")
            
            # En az 2 örnek oluştur (test amaçlı)
            print("\n=== Test Verisi Oluşturuluyor ===")
            create_sample_feedback_data(min_samples=2)
            
            # Tekrar dene
            training_data = trainer.prepare_training_data(min_samples=2)
        
        if training_data:
            manual_count = training_data['sources'].count('manual')
            pseudo_count = training_data['sources'].count('pseudo')
            ages = training_data['ages']
            
            print(f"Toplam örnek: {len(training_data['embeddings'])}")
            print(f"Manuel geri bildirim: {manual_count}")
            print(f"Pseudo-label: {pseudo_count}")
            print(f"Yaş aralığı: {ages.min():.1f} - {ages.max():.1f}")
            print(f"Ortalama yaş: {ages.mean():.1f} ± {ages.std():.1f}")
            
            # 2. Model eğitimi başlat
            print("\n=== Model Eğitimi Başlatılıyor ===")
            params = Config.DEFAULT_TRAINING_PARAMS.copy()
            # Test için istenirse override yapılabilir, örn: params['epochs'] = 10
            result = trainer.train_model(training_data, params)
            
            # 3. Sonuçları göster
            print("\n=== Eğitim Sonuçları ===")
            print(f"Final MAE: {result['metrics']['mae']:.2f} yaş")
            print(f"Final RMSE: {result['metrics']['rmse']:.2f} yaş")
            print(f"±3 yaş doğruluğu: {result['metrics']['within_3_years']*100:.1f}%")
            print(f"±5 yaş doğruluğu: {result['metrics']['within_5_years']*100:.1f}%")
            print(f"±10 yaş doğruluğu: {result['metrics']['within_10_years']*100:.1f}%")
            
            # 4. Model versiyonunu kaydet
            print("\n=== Model Kaydediliyor ===")
            model_version = trainer.save_model_version(result['model'], result)
            print(f"Model kaydedildi: {model_version.version_name} (ID: {model_version.id})")
            
            # 5. Model versiyonlarını listele
            print("\n=== Tüm Model Versiyonları ===")
            versions = trainer.get_model_versions()
            for v in versions:
                active_str = " [AKTİF]" if v['is_active'] else ""
                print(f"- {v['version_name']} (v{v['version']}){active_str}")
                print(f"  MAE: {v['metrics'].get('mae', 'N/A'):.2f} yaş")
                print(f"  Örnekler: {v['training_samples']} eğitim, {v['validation_samples']} doğrulama")
                print()

def create_sample_feedback_data(min_samples=1):
    """Test için örnek geri bildirim verisi oluşturur"""
    import numpy as np
    
    # Örnek embedding (512 boyutlu)
    sample_embedding = np.random.randn(512)
    embedding_str = ",".join(str(float(x)) for x in sample_embedding)
    
    # Manuel geri bildirim örneği
    feedback1 = Feedback(
        feedback_type='age',
        feedback_source='MANUAL_USER',
        corrected_age=25,
        embedding=embedding_str,
        frame_path='test/frame1.jpg',
        person_id='test_person_1'
    )
    
    # Pseudo-label örneği
    feedback2 = Feedback(
        feedback_type='age_pseudo',
        feedback_source='PSEUDO_BUFFALO_HIGH_CONF',
        pseudo_label_original_age=30.5,
        pseudo_label_clip_confidence=0.85,
        embedding=embedding_str,
        frame_path='test/frame2.jpg',
        person_id='test_person_2'
    )
    
    db.session.add(feedback1)
    db.session.add(feedback2)
    db.session.commit()
    
    print("Test verileri oluşturuldu.")

if __name__ == "__main__":
    test_age_training() 