#!/usr/bin/env python3
"""
v1 modelini (UTKFace ile eğitilmiş) manuel olarak veritabanına ekler.
"""

import os
from datetime import datetime
from app import create_app, db
from app.models.content import ModelVersion

def add_v1_model():
    """v1 modelini veritabanına ekler"""
    app = create_app()
    
    with app.app_context():
        # v1 zaten var mı kontrol et
        existing_v1 = ModelVersion.query.filter_by(
            model_type='age',
            version=1
        ).first()
        
        if existing_v1:
            print("v1 modeli zaten mevcut")
            return
        
        # v1 model dosyası var mı kontrol et
        v1_model_path = os.path.join('storage', 'models', 'age', 'custom_age_head', 'versions', 'v1', 'custom_age_head.pth')
        
        if not os.path.exists(v1_model_path):
            print(f"v1 model dosyası bulunamadı: {v1_model_path}")
            return
        
        # v1 model kaydını oluştur
        v1_model_version = ModelVersion(
            model_type='age',
            version=1,
            version_name='v1_utkface_trained',
            file_path='storage/models/age/custom_age_head/versions/v1',
            weights_path=v1_model_path,
            metrics={
                'description': 'UTKFace dataset ile eğitilmiş custom model',
                'training_method': 'UTKFace full dataset training'
            },
            training_samples=0,  # UTKFace dataset boyutu bilinmiyor
            validation_samples=0,
            epochs=0,  # Epoch sayısı bilinmiyor
            is_active=False,  # Başlangıçta aktif değil
            created_at=datetime(2025, 5, 25),  # Tahmini tarih
            used_feedback_ids=[]
        )
        
        db.session.add(v1_model_version)
        db.session.commit()
        
        print("v1 modeli başarıyla eklendi!")
        
        # Tüm versiyonları göster
        all_versions = ModelVersion.query.filter_by(model_type='age').order_by(ModelVersion.version).all()
        print(f"\nTüm yaş modeli versiyonları:")
        for v in all_versions:
            status = "AKTİF" if v.is_active else "Pasif"
            description = v.metrics.get('description', f"MAE: {v.metrics.get('mae', 'N/A')}")
            print(f"- v{v.version}: {v.version_name} - {status} - {description}")

if __name__ == "__main__":
    add_v1_model() 