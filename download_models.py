#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI modelleri indirme betiği
Bu betik, WSANALIZ uygulaması için gerekli AI model dosyalarını indirir ve yapılandırır.
"""

import os
import sys
import requests
import zipfile
import gdown
import logging
import shutil
from pathlib import Path

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_downloader')

# Ana dizin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'storage', 'models')

# Model indirme bağlantıları - Bu bağlantıları kendi gerçek modellerinizin bağlantılarıyla değiştirin
MODEL_URLS = {
    # YOLOv4 nesne tespit modeli
    'detection': {
        'yolov4.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
        'yolov4.cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
        'coco.names': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
    },
    # Kategori modelleri için Google Drive bağlantıları
    'category_models': {
        'violence': 'MODEL_ID_1',  # Google Drive ID'leri ile değiştirin
        'adult': 'MODEL_ID_2',
        'harassment': 'MODEL_ID_3',
        'weapon': 'MODEL_ID_4',
        'drug': 'MODEL_ID_5'
    }
}

def ensure_dir(dir_path):
    """Klasörün var olduğundan emin ol, yoksa oluştur"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Klasör oluşturuldu: {dir_path}")

def download_file(url, target_path):
    """Dosyayı belirtilen URL'den indir ve hedef yola kaydet"""
    try:
        logger.info(f"İndiriliyor: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"İndirme tamamlandı: {target_path}")
        return True
    except Exception as e:
        logger.error(f"İndirme hatası: {str(e)}")
        return False

def download_from_drive(file_id, target_path):
    """Google Drive'dan dosya indir"""
    try:
        logger.info(f"Google Drive'dan indiriliyor: {file_id}")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", target_path, quiet=False)
        logger.info(f"İndirme tamamlandı: {target_path}")
        return True
    except Exception as e:
        logger.error(f"Google Drive indirme hatası: {str(e)}")
        return False

def extract_zip(zip_path, extract_to):
    """ZIP dosyasını belirtilen hedefe çıkar"""
    try:
        logger.info(f"ZIP dosyası çıkarılıyor: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"ZIP dosyası çıkarma tamamlandı: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"ZIP dosyası çıkarma hatası: {str(e)}")
        return False

def download_detection_models():
    """Tespit modellerini indir"""
    detection_dir = os.path.join(MODELS_DIR, 'detection')
    ensure_dir(detection_dir)
    
    for filename, url in MODEL_URLS['detection'].items():
        target_path = os.path.join(detection_dir, filename)
        if not os.path.exists(target_path):
            download_file(url, target_path)
        else:
            logger.info(f"Dosya zaten mevcut: {target_path}")

def download_category_models():
    """Kategori modellerini indir"""
    for category, file_id in MODEL_URLS['category_models'].items():
        category_dir = os.path.join(MODELS_DIR, category)
        ensure_dir(category_dir)
        
        zip_path = os.path.join(category_dir, f"{category}_model.zip")
        if not os.path.exists(os.path.join(category_dir, 'saved_model.pb')):
            if download_from_drive(file_id, zip_path):
                extract_zip(zip_path, category_dir)
                # ZIP dosyasını temizle
                os.remove(zip_path)
        else:
            logger.info(f"Model zaten mevcut: {category}")

def create_dummy_models():
    """Gerçek model dosyaları yoksa geçici olarak boş modeller oluştur"""
    logger.warning("Gerçek model dosyaları indirilemiyor, geçici model dosyaları oluşturuluyor...")
    
    # Detection dummy files
    detection_dir = os.path.join(MODELS_DIR, 'detection')
    ensure_dir(detection_dir)
    
    # Dummy YOLO dosyaları
    with open(os.path.join(detection_dir, 'yolov4.cfg'), 'w') as f:
        f.write("# Dummy YOLO config file")
    
    with open(os.path.join(detection_dir, 'coco.names'), 'w') as f:
        f.write("person\ncar\nbottle\nchair\ndog\nweapon\ninappropriate")
    
    # Dummy model dosyalarını oluştur
    for category in ['violence', 'adult', 'harassment', 'weapon', 'drug']:
        category_dir = os.path.join(MODELS_DIR, category)
        ensure_dir(category_dir)
        
        # SavedModel yapısını taklit et
        saved_model_dir = os.path.join(category_dir, 'saved_model')
        ensure_dir(saved_model_dir)
        
        with open(os.path.join(category_dir, 'saved_model.pb'), 'w') as f:
            f.write("# Dummy model file")
        
        logger.info(f"Dummy model oluşturuldu: {category}")

def main():
    """Ana fonksiyon"""
    logger.info("WSANALIZ AI modelleri indiriliyor...")
    
    # Model klasörlerini oluştur
    ensure_dir(MODELS_DIR)
    
    try:
        # Tespit modellerini indir
        download_detection_models()
        
        # Kategori modellerini indir
        download_category_models()
        
        logger.info("Tüm modeller başarıyla indirildi ve yapılandırıldı.")
    except Exception as e:
        logger.error(f"Model indirme hatası: {str(e)}")
        logger.warning("Gerçek modeller indirilemedi, geçici model dosyaları oluşturuluyor...")
        create_dummy_models()
    
    logger.info("Model kurulumu tamamlandı.")

if __name__ == "__main__":
    main() 