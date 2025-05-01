import os
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
import json

# Proje kök dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.ai.cnn_age_model import create_age_model, train_model, save_model

def process_image(image_path):
    """Görüntüyü işler ve yaş etiketini çıkarır"""
    try:
        # Dosya adından yaşı çıkar (format: [age]_[gender]_[race]_[date&time].jpg)
        age = int(image_path.stem.split('_')[0])
        
        # Görüntüyü oku ve yeniden boyutlandır
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
            
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        
        return image, age
        
    except Exception as e:
        print(f"Görüntü işleme hatası ({image_path}): {str(e)}")
        return None, None

def prepare_dataset():
    """Veri setini hazırlar"""
    dataset_dir = project_root / 'storage' / 'datasets' / 'UTKFace'
    
    # Görüntüleri işle
    images = []
    ages = []
    
    print("Görüntüler işleniyor...")
    image_files = list(dataset_dir.glob('*.jpg'))
    
    for image_path in tqdm(image_files):
        image, age = process_image(image_path)
        if image is not None and age is not None:
            images.append(image)
            ages.append(age)
    
    return np.array(images), np.array(ages)

def main():
    """Ana eğitim fonksiyonu"""
    try:
        print("Veri seti hazırlanıyor...")
        print("UTKFace veri seti hakkında:")
        print("- 20,000+ yüz görüntüsü")
        print("- Yaş aralığı: 0-116")
        print("- Hizalanmış ve kırpılmış yüzler")
        print("- Her görüntü için yaş, cinsiyet ve etnik köken bilgisi")
        print("- Kaynak: https://susanqq.github.io/UTKFace/")
        print()
        
        X, y = prepare_dataset()
        
        if len(X) == 0:
            print("Veri seti hazırlanamadı!")
            return
        
        print(f"Toplam {len(X)} görüntü işlendi")
        
        # Verileri karıştır
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Eğitim ve doğrulama setlerine ayır
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Model oluştur
        print("Model oluşturuluyor...")
        model = create_age_model()
        
        # Modeli eğit
        print("Model eğitiliyor...")
        history = train_model(
            model,
            (X_train, y_train),
            (X_val, y_val),
            epochs=100,  # Daha uzun eğitim
            batch_size=32,  # Batch size optimizasyonu
            learning_rate=0.001  # Learning rate optimizasyonu
        )
        
        # Modeli kaydet
        model_path = project_root / 'storage' / 'models' / 'age' / 'age_model.h5'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(model_path))
        print(f"Model kaydedildi: {model_path}")
        
        # Eğitim istatistiklerini kaydet
        stats = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }
        
        stats_path = model_path.parent / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        print("Eğitim tamamlandı!")
        print(f"Son doğrulama MAE: {history.history['val_mae'][-1]:.2f} yıl")
        
    except Exception as e:
        print(f"Eğitim hatası: {str(e)}")

if __name__ == "__main__":
    main() 