#!/usr/bin/env python3
"""
Model Senkronizasyon Test Script'i
VT ve processed klasörü temizlendikten sonra model versiyonlarının 
otomatik olarak veritabanına kaydedilip kaydedilmediğini test eder.
"""

import os
import shutil
import sqlite3
from datetime import datetime

def cleanup_database_and_processed():
    """VT ve processed klasörünü temizle"""
    print("🧹 VT ve processed klasörü temizleniyor...")
    
    # Veritabanını sil
    db_paths = [
        'app/wsanaliz_dev.db',
        'wsanaliz_dev.db',
        'instance/wsanaliz_dev.db'
    ]
    
    for db_path in db_paths:
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"   ✓ {db_path} silindi")
    
    # Processed klasörünü temizle
    processed_path = 'storage/processed'
    if os.path.exists(processed_path):
        # Logs klasörünü koru
        logs_path = os.path.join(processed_path, 'logs')
        logs_backup = None
        
        if os.path.exists(logs_path):
            logs_backup = 'logs_backup_temp'
            shutil.move(logs_path, logs_backup)
        
        # Processed klasörünü temizle
        shutil.rmtree(processed_path)
        os.makedirs(processed_path, exist_ok=True)
        
        # Logs klasörünü geri koy
        if logs_backup and os.path.exists(logs_backup):
            shutil.move(logs_backup, logs_path)
        
        print(f"   ✓ {processed_path} temizlendi (logs korundu)")

def create_test_model_versions():
    """Test için örnek model versiyonları oluştur"""
    print("📦 Test model versiyonları oluşturuluyor...")
    
    # Age model test versiyonları
    age_versions_dir = 'storage/models/age/custom_age_head/versions'
    os.makedirs(age_versions_dir, exist_ok=True)
    
    # v1_20250527_120000 versiyonu
    v1_path = os.path.join(age_versions_dir, 'v1_20250527_120000')
    os.makedirs(v1_path, exist_ok=True)
    
    # training_details.json oluştur
    training_details = {
        "training_date": "2025-05-27T12:00:00",
        "metrics": {"mae": 3.2, "accuracy": 0.85},
        "training_samples": 1000,
        "validation_samples": 200,
        "history": {
            "train_loss": [0.5, 0.3, 0.2],
            "val_loss": [0.6, 0.4, 0.3]
        }
    }
    
    with open(os.path.join(v1_path, 'training_details.json'), 'w') as f:
        import json
        json.dump(training_details, f, indent=2)
    
    # Dummy model dosyası
    with open(os.path.join(v1_path, 'model.pth'), 'w') as f:
        f.write("dummy model file")
    
    print(f"   ✓ Age model v1 oluşturuldu: {v1_path}")
    
    # v2_20250527_150000 versiyonu
    v2_path = os.path.join(age_versions_dir, 'v2_20250527_150000')
    os.makedirs(v2_path, exist_ok=True)
    
    training_details_v2 = {
        "training_date": "2025-05-27T15:00:00",
        "metrics": {"mae": 2.8, "accuracy": 0.88},
        "training_samples": 1200,
        "validation_samples": 250,
        "history": {
            "train_loss": [0.4, 0.25, 0.18],
            "val_loss": [0.5, 0.35, 0.25]
        }
    }
    
    with open(os.path.join(v2_path, 'training_details.json'), 'w') as f:
        json.dump(training_details_v2, f, indent=2)
    
    with open(os.path.join(v2_path, 'model.pth'), 'w') as f:
        f.write("dummy model file v2")
    
    print(f"   ✓ Age model v2 oluşturuldu: {v2_path}")
    
    # CLIP model test versiyonları
    clip_versions_dir = 'storage/models/clip/versions'
    os.makedirs(clip_versions_dir, exist_ok=True)
    
    # v20250527_140000 versiyonu
    clip_v1_path = os.path.join(clip_versions_dir, 'v20250527_140000')
    os.makedirs(clip_v1_path, exist_ok=True)
    
    clip_metadata = {
        "created_at": "2025-05-27T14:00:00",
        "training_start": "2025-05-27T14:00:00",
        "training_end": "2025-05-27T14:30:00",
        "feedback_count": 80,
        "training_params": {
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 5
        },
        "performance_metrics": {
            "final_loss": 0.15,
            "accuracy": 0.92
        }
    }
    
    with open(os.path.join(clip_v1_path, 'metadata.json'), 'w') as f:
        json.dump(clip_metadata, f, indent=2)
    
    with open(os.path.join(clip_v1_path, 'pytorch_model.bin'), 'w') as f:
        f.write("dummy clip model file")
    
    print(f"   ✓ CLIP model v1 oluşturuldu: {clip_v1_path}")
    
    # v20250527_160000 versiyonu
    clip_v2_path = os.path.join(clip_versions_dir, 'v20250527_160000')
    os.makedirs(clip_v2_path, exist_ok=True)
    
    clip_metadata_v2 = {
        "created_at": "2025-05-27T16:00:00",
        "training_start": "2025-05-27T16:00:00",
        "training_end": "2025-05-27T16:45:00",
        "feedback_count": 120,
        "training_params": {
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 8
        },
        "performance_metrics": {
            "final_loss": 0.12,
            "accuracy": 0.94
        }
    }
    
    with open(os.path.join(clip_v2_path, 'metadata.json'), 'w') as f:
        json.dump(clip_metadata_v2, f, indent=2)
    
    with open(os.path.join(clip_v2_path, 'pytorch_model.bin'), 'w') as f:
        f.write("dummy clip model file v2")
    
    print(f"   ✓ CLIP model v2 oluşturuldu: {clip_v2_path}")

def check_database_after_startup():
    """Uygulama başlatıldıktan sonra VT'yi kontrol et"""
    print("🔍 Veritabanı kontrol ediliyor...")
    
    db_path = 'app/wsanaliz_dev.db'
    if not os.path.exists(db_path):
        print("   ❌ Veritabanı bulunamadı!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Age model versiyonlarını kontrol et
        cursor.execute("SELECT * FROM model_versions WHERE model_type='age'")
        age_versions = cursor.fetchall()
        print(f"   📊 Age model versiyonları: {len(age_versions)} adet")
        
        for version in age_versions:
            print(f"      - {version[3]} (v{version[2]}) - Aktif: {bool(version[9])}")
        
        # CLIP training sessions kontrol et
        cursor.execute("SELECT * FROM clip_training_sessions")
        clip_sessions = cursor.fetchall()
        print(f"   🤖 CLIP training sessions: {len(clip_sessions)} adet")
        
        for session in clip_sessions:
            print(f"      - {session[1]} (ID: {session[0]}) - Aktif: {bool(session[11])}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Veritabanı kontrol hatası: {e}")
        conn.close()
        return False

def main():
    """Ana test fonksiyonu"""
    print("🧪 Model Senkronizasyon Test Script'i")
    print("=" * 50)
    
    # 1. Temizlik
    cleanup_database_and_processed()
    
    # 2. Test model versiyonları oluştur
    create_test_model_versions()
    
    print("\n✅ Test hazırlığı tamamlandı!")
    print("📋 Sonraki adımlar:")
    print("   1. Uygulamayı başlatın: python app.py")
    print("   2. Uygulama başladıktan sonra bu script'i tekrar çalıştırın:")
    print("      python test_model_sync.py --check")
    print("   3. Veya web arayüzünden kontrol edin:")
    print("      http://localhost:5000/model-management")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_database_after_startup()
    else:
        main() 