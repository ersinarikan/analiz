import os
import sys
from pathlib import Path

# Proje kök dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.ai.cnn_age_model import create_age_model, save_model

def init_age_model():
    """Başlangıç CNN yaş tahmin modelini oluşturur ve kaydeder"""
    try:
        # Model dizini
        model_dir = project_root / 'storage' / 'models' / 'age'
        model_path = model_dir / 'age_model.h5'
        
        # Dizin yoksa oluştur
        os.makedirs(model_dir, exist_ok=True)
        
        # Model oluştur
        model = create_age_model()
        
        # Modeli kaydet
        save_model(model, str(model_path))
        print(f"Başlangıç CNN yaş tahmin modeli oluşturuldu: {model_path}")
        
    except Exception as e:
        print(f"Model oluşturma hatası: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    init_age_model() 