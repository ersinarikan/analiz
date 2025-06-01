#!/usr/bin/env python3
"""
Mevcut model versiyonlarını dosya sisteminden veritabanına senkronize eder.
"""

import os
import json
from datetime import datetime
from app import create_app, db
from app.models.content import ModelVersion

def sync_age_model_versions():
    """Yaş modeli versiyonlarını senkronize eder"""
    app = create_app()
    
    with app.app_context():
        # Mevcut veritabanı kayıtlarını kontrol et
        existing_versions = ModelVersion.query.filter_by(model_type='age').all()
        print(f"Veritabanında mevcut {len(existing_versions)} yaş modeli versiyonu bulundu")
        
        # Dosya sistemindeki versiyonları kontrol et
        versions_dir = os.path.join('storage', 'models', 'age', 'custom_age_head', 'versions')
        
        if not os.path.exists(versions_dir):
            print(f"Versiyonlar klasörü bulunamadı: {versions_dir}")
            return
        
        version_folders = [d for d in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, d))]
        print(f"Dosya sisteminde {len(version_folders)} versiyon klasörü bulundu: {version_folders}")
        
        for version_folder in version_folders:
            version_path = os.path.join(versions_dir, version_folder)
            training_details_path = os.path.join(version_path, 'training_details.json')
            model_path = os.path.join(version_path, 'model.pth')
            
            # Eğitim detaylarını oku
            if os.path.exists(training_details_path):
                with open(training_details_path, 'r') as f:
                    training_details = json.load(f)
                
                # Versiyon numarasını belirle (base model v0, custom models v1+)
                if version_folder.startswith('v') and '_' in version_folder:
                    base_version = int(version_folder.split('_')[0][1:])  # v1_20250526_230854 -> 1
                    version_num = base_version + 1  # Custom model olduğu için +1
                elif version_folder.startswith('v'):
                    base_version = int(version_folder[1:])  # v1 -> 1
                    version_num = base_version + 1  # Custom model olduğu için +1
                else:
                    version_num = 2  # Default custom model
                
                # Bu versiyon veritabanında var mı kontrol et
                existing = ModelVersion.query.filter_by(
                    model_type='age',
                    version=version_num,
                    version_name=version_folder
                ).first()
                
                if existing:
                    print(f"Versiyon {version_folder} zaten veritabanında mevcut")
                    continue
                
                # Yeni model versiyonu oluştur
                model_version = ModelVersion(
                    model_type='age',
                    version=version_num,
                    version_name=version_folder,
                    file_path=version_path,
                    weights_path=model_path,
                    metrics=training_details.get('metrics', {}),
                    training_samples=training_details.get('training_samples', 0),
                    validation_samples=training_details.get('validation_samples', 0),
                    epochs=len(training_details.get('history', {}).get('train_loss', [])),
                    is_active=False,  # Başlangıçta aktif değil
                    created_at=datetime.fromisoformat(training_details.get('training_date', datetime.now().isoformat())),
                    used_feedback_ids=[]
                )
                
                db.session.add(model_version)
                print(f"Versiyon {version_folder} veritabanına eklendi")
            else:
                print(f"Eğitim detayları bulunamadı: {training_details_path}")
        
        # En son versiyonu aktif yap
        if version_folders:
            latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
            latest_model = ModelVersion.query.filter_by(
                model_type='age',
                version_name=latest_version
            ).first()
            
            if latest_model:
                # Diğer tüm versiyonları pasif yap
                ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                
                # En son versiyonu aktif yap
                latest_model.is_active = True
                print(f"En son versiyon {latest_version} aktif olarak ayarlandı")
        
        # Base model (v0) kaydını ekle
        base_model = ModelVersion.query.filter_by(
            model_type='age',
            version=0
        ).first()
        
        if not base_model:
            base_model_version = ModelVersion(
                model_type='age',
                version=0,
                version_name='base_utkface',
                file_path='storage/models/age/buffalo_l/base_model',
                weights_path='storage/models/age/buffalo_l/base_model/w600k_r50.onnx',
                metrics={'description': 'UTKFace pre-trained base model'},
                training_samples=0,
                validation_samples=0,
                epochs=0,
                is_active=False,
                created_at=datetime(2025, 1, 1),  # Sabit tarih
                used_feedback_ids=[]
            )
            db.session.add(base_model_version)
            print("Base model (v0) kaydı eklendi")
        
        db.session.commit()
        print("Senkronizasyon tamamlandı!")
        
        # Sonuçları göster
        final_versions = ModelVersion.query.filter_by(model_type='age').order_by(ModelVersion.version).all()
        print(f"\nSonuç: Veritabanında {len(final_versions)} yaş modeli versiyonu:")
        for v in final_versions:
            status = "AKTİF" if v.is_active else "Pasif"
            mae_info = f"MAE: {v.metrics.get('mae', 'N/A')}" if 'mae' in v.metrics else v.metrics.get('description', 'Base Model')
            print(f"- {v.version_name} (v{v.version}) - {status} - {mae_info}")

def sync_clip_model_versions():
    """CLIP modeli versiyonlarını senkronize eder"""
    app = create_app()
    
    with app.app_context():
        from app.models.clip_training import CLIPTrainingSession
        
        # Mevcut veritabanı kayıtlarını kontrol et
        existing_sessions = CLIPTrainingSession.query.all()
        print(f"Veritabanında mevcut {len(existing_sessions)} CLIP training session'ı bulundu")
        
        # Dosya sistemindeki versiyonları kontrol et
        versions_dir = os.path.join('storage', 'models', 'clip', 'versions')
        
        if not os.path.exists(versions_dir):
            print(f"CLIP versiyonlar klasörü bulunamadı: {versions_dir}")
            return
        
        version_folders = [d for d in os.listdir(versions_dir) 
                          if os.path.isdir(os.path.join(versions_dir, d)) and d.startswith('v')]
        print(f"Dosya sisteminde {len(version_folders)} CLIP versiyon klasörü bulundu: {version_folders}")
        
        for version_folder in version_folders:
            version_path = os.path.join(versions_dir, version_folder)
            metadata_path = os.path.join(version_path, 'metadata.json')
            model_path = os.path.join(version_path, 'pytorch_model.bin')
            
            # Metadata dosyasını oku
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Bu versiyon veritabanında var mı kontrol et
                existing = CLIPTrainingSession.query.filter_by(
                    version_name=version_folder
                ).first()
                
                if existing:
                    print(f"CLIP versiyon {version_folder} zaten veritabanında mevcut")
                    continue
                
                # Yeni CLIP training session oluştur
                training_session = CLIPTrainingSession(
                    version_name=version_folder,
                    feedback_count=metadata.get('feedback_count', 0),
                    training_start=datetime.fromisoformat(metadata.get('training_start', datetime.now().isoformat())),
                    training_end=datetime.fromisoformat(metadata.get('training_end', datetime.now().isoformat())),
                    status='completed',
                    model_path=model_path,
                    is_active=False,  # Başlangıçta aktif değil
                    is_successful=True,
                    created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat()))
                )
                
                # Training parametrelerini ayarla
                if 'training_params' in metadata:
                    training_session.set_training_params(metadata['training_params'])
                
                # Performance metriklerini ayarla
                if 'performance_metrics' in metadata:
                    training_session.set_performance_metrics(metadata['performance_metrics'])
                
                db.session.add(training_session)
                print(f"CLIP versiyon {version_folder} veritabanına eklendi")
            else:
                print(f"CLIP metadata bulunamadı: {metadata_path}")
        
        # En son versiyonu aktif yap
        if version_folders:
            latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
            latest_session = CLIPTrainingSession.query.filter_by(
                version_name=latest_version
            ).first()
            
            if latest_session:
                # Diğer tüm versiyonları pasif yap
                CLIPTrainingSession.query.update({'is_active': False})
                
                # En son versiyonu aktif yap
                latest_session.is_active = True
                print(f"En son CLIP versiyon {latest_version} aktif olarak ayarlandı")
        
        db.session.commit()
        print("CLIP senkronizasyon tamamlandı!")
        
        # Sonuçları göster
        final_sessions = CLIPTrainingSession.query.order_by(CLIPTrainingSession.created_at).all()
        print(f"\nSonuç: Veritabanında {len(final_sessions)} CLIP training session'ı:")
        for s in final_sessions:
            status = "AKTİF" if s.is_active else "Pasif"
            print(f"- {s.version_name} (ID: {s.id}) - {status} - {s.feedback_count} feedback")

def sync_all_model_versions():
    """Tüm model versiyonlarını senkronize eder"""
    print("🔄 Model versiyonları senkronize ediliyor...")
    
    try:
        print("\n📊 Yaş modeli versiyonları senkronize ediliyor...")
        sync_age_model_versions()
        
        print("\n🤖 CLIP modeli versiyonları senkronize ediliyor...")
        sync_clip_model_versions()
        
        print("\n✅ Tüm model versiyonları başarıyla senkronize edildi!")
        
    except Exception as e:
        print(f"❌ Senkronizasyon hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sync_all_model_versions() 