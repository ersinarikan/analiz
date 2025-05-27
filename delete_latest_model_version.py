#!/usr/bin/env python3
"""
En son model versiyonunu hem dosya sisteminden hem de veritabanından siler.
Kullanım: python delete_latest_model_version.py --model-type age
"""

import os
import shutil
import argparse
from datetime import datetime
from app import create_app, db
from app.models.content import ModelVersion

def delete_latest_model_version(model_type='age', dry_run=False):
    """
    Belirtilen model tipinin en son versiyonunu siler.
    
    Args:
        model_type: Silinecek model tipi ('age' veya 'content')
        dry_run: True ise sadece ne yapılacağını gösterir, silmez
        
    Returns:
        tuple: (success, message)
    """
    app = create_app()
    
    with app.app_context():
        # En son versiyonu bul
        latest_version = ModelVersion.query.filter_by(
            model_type=model_type
        ).order_by(ModelVersion.version.desc()).first()
        
        if not latest_version:
            return False, f"{model_type} tipinde hiç model versiyonu bulunamadı."
        
        # Base model (v0) silinmemeli
        if latest_version.version == 0:
            return False, "Base model (v0) silinemez!"
        
        # Aktif model silinmemeli
        if latest_version.is_active:
            return False, f"Aktif model versiyonu (v{latest_version.version}) silinemez! Önce başka bir versiyonu aktif yapın."
        
        print(f"\nSilinecek versiyon bilgileri:")
        print(f"- Model Tipi: {latest_version.model_type}")
        print(f"- Versiyon: v{latest_version.version}")
        print(f"- Versiyon Adı: {latest_version.version_name}")
        print(f"- Oluşturulma Tarihi: {latest_version.created_at}")
        print(f"- Dosya Yolu: {latest_version.file_path}")
        
        if latest_version.metrics:
            print(f"- Metrikler:")
            for key, value in latest_version.metrics.items():
                print(f"  - {key}: {value}")
        
        if dry_run:
            print("\n[DRY RUN] Yukarıdaki versiyon silinecek.")
            return True, "Dry run tamamlandı."
        
        # Kullanıcıdan onay al
        confirmation = input("\nBu versiyonu silmek istediğinizden emin misiniz? (evet/hayır): ")
        if confirmation.lower() not in ['evet', 'e', 'yes', 'y']:
            return False, "İşlem iptal edildi."
        
        try:
            # Dosya sisteminden sil
            if latest_version.file_path and os.path.exists(latest_version.file_path):
                print(f"\nDosya sisteminden siliniyor: {latest_version.file_path}")
                shutil.rmtree(latest_version.file_path)
                print("✓ Dosyalar silindi")
            else:
                print(f"\n! Dosya yolu bulunamadı veya zaten silinmiş: {latest_version.file_path}")
            
            # Veritabanından sil
            db.session.delete(latest_version)
            db.session.commit()
            print("✓ Veritabanı kaydı silindi")
            
            # Kalan versiyonları göster
            remaining_versions = ModelVersion.query.filter_by(
                model_type=model_type
            ).order_by(ModelVersion.version.desc()).all()
            
            print(f"\nKalan {len(remaining_versions)} versiyon:")
            for v in remaining_versions:
                status = "AKTİF" if v.is_active else "Pasif"
                mae_info = f"MAE: {v.metrics.get('mae', 'N/A')}" if 'mae' in v.metrics else v.metrics.get('description', 'N/A')
                print(f"- {v.version_name} (v{v.version}) - {status} - {mae_info}")
            
            return True, f"Model versiyonu v{latest_version.version} başarıyla silindi."
            
        except Exception as e:
            db.session.rollback()
            return False, f"Silme işlemi sırasında hata oluştu: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='En son model versiyonunu sil')
    parser.add_argument('--model-type', type=str, default='age', 
                        choices=['age', 'content'],
                        help='Silinecek model tipi (varsayılan: age)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Sadece ne yapılacağını göster, silme')
    
    args = parser.parse_args()
    
    success, message = delete_latest_model_version(
        model_type=args.model_type,
        dry_run=args.dry_run
    )
    
    print(f"\n{'✓' if success else '✗'} {message}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 