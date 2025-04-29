import os
import cv2
import numpy as np
import logging
from app.utils.image_utils import load_image

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_training_data(source_dir, target_dir):
    """
    Eğitim verilerini hazırlar
    
    Args:
        source_dir: Kaynak görüntülerin bulunduğu dizin
        target_dir: Hazırlanan verilerin kaydedileceği dizin
    """
    try:
        # Hedef dizini oluştur
        os.makedirs(target_dir, exist_ok=True)
        
        # Kaynak dizindeki tüm dosyaları tara
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Görüntüyü yükle
                        image_path = os.path.join(root, file)
                        image = load_image(image_path)
                        
                        if image is None:
                            continue
                            
                        # Görüntüyü yeniden boyutlandır
                        image = cv2.resize(image, (224, 224))
                        
                        # Yaş etiketini al (dosya adından)
                        age = int(file.split('_')[0])
                        
                        # Yeni dosya adı oluştur
                        new_filename = f"{age}_{file}"
                        target_path = os.path.join(target_dir, new_filename)
                        
                        # Görüntüyü kaydet
                        cv2.imwrite(target_path, image)
                        logger.info(f"Görüntü hazırlandı: {target_path}")
                        
                    except Exception as e:
                        logger.error(f"Veri hazırlama hatası ({file}): {str(e)}")
                        continue
                        
        logger.info("Eğitim verileri hazırlama tamamlandı")
        
    except Exception as e:
        logger.error(f"Eğitim verisi hazırlama hatası: {str(e)}")

if __name__ == "__main__":
    # Kaynak ve hedef dizinleri belirle
    source_dir = "raw_data"  # Ham verilerin bulunduğu dizin
    target_dir = "storage/data/training"  # Hazırlanan verilerin kaydedileceği dizin
    
    prepare_training_data(source_dir, target_dir) 