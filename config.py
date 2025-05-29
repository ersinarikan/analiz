#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Konfigürasyon Ayarları
==============================

Bu dosya uygulamanın farklı ortamlar (development, production, testing)
için konfigürasyon ayarlarını içerir.
"""

import os
from dotenv import load_dotenv
from datetime import timedelta

# .env dosyasını yükle
load_dotenv()

# TensorFlow uyarılarını bastır (tüm modüller için geçerli)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN uyarılarını kapat

# Ultralytics uyarılarını bastır
os.environ['YOLO_VERBOSE'] = 'False'

# Albumentations günceleme kontrolünü devre dışı bırak
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Model state'i import et (Flask debug mode bu dosyayı izleyecek)
try:
    from app.utils.model_state import MODEL_STATE, LAST_UPDATE as MODEL_LAST_UPDATE
except ImportError:
    MODEL_STATE = {}
    MODEL_LAST_UPDATE = None

# Settings state'i import et (Flask debug mode bu dosyayı izleyecek)
try:
    from app.utils.settings_state import SETTINGS_STATE, LAST_UPDATE as SETTINGS_LAST_UPDATE
except ImportError:
    SETTINGS_STATE = {}
    SETTINGS_LAST_UPDATE = None

class Config:
    """Temel konfigürasyon sınıfı - tüm ortamlar için ortak ayarlar"""
    
    # Güvenlik Ayarları
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # Veritabanı Ayarları
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///wsanaliz.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_timeout': 20,
        'pool_recycle': -1,
        'pool_pre_ping': True
    }
    
    # Dosya Yükleme Ayarları
    STORAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'uploads')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'processed')
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'models')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB maksimum dosya boyutu
    
    # İzin verilen dosya uzantıları
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', '3gp'}
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    
    # Model Ayarları
    AGE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'age')
    CONTENT_MODEL_PATH = os.path.join(MODELS_FOLDER, 'clip')
    DETECTION_MODEL_PATH = os.path.join(MODELS_FOLDER, 'detection')
    
    # Eğitim Ayarları
    MIN_TRAINING_SAMPLES = 5  # Minimum eğitim örneği sayısı
    CLEANUP_TRAINING_DATA_AFTER_TRAINING = False  # Eğitim sonrası veri temizleme
    
    # CORS Ayarları
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Analiz Performans Ayarları
    MAX_CONCURRENT_ANALYSES = 3  # Eş zamanlı maksimum analiz sayısı
    ANALYSIS_TIMEOUT = 1800  # Analiz zaman aşımı (30 dakika)
    
    # Log Ayarları
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'processed', 'logs', 'app.log')
    
    # Yapay Zeka Modeli Ayarları
    USE_GPU = os.environ.get('USE_GPU', 'True').lower() in ('true', '1', 't')
    
    # === Merkezi Model Yolları ===
    # InsightFace Age Model
    INSIGHTFACE_AGE_MODEL_TYPE = 'age'
    INSIGHTFACE_AGE_MODEL_NAME = 'buffalo_l'
    INSIGHTFACE_AGE_MODEL_BASE_PATH = os.path.join(MODELS_FOLDER, INSIGHTFACE_AGE_MODEL_TYPE, INSIGHTFACE_AGE_MODEL_NAME, 'base_model')
    INSIGHTFACE_AGE_MODEL_VERSIONS_PATH = os.path.join(MODELS_FOLDER, INSIGHTFACE_AGE_MODEL_TYPE, INSIGHTFACE_AGE_MODEL_NAME, 'versions')
    INSIGHTFACE_AGE_MODEL_ACTIVE_PATH = os.path.join(MODELS_FOLDER, INSIGHTFACE_AGE_MODEL_TYPE, INSIGHTFACE_AGE_MODEL_NAME, 'active_model') # Başlangıçta base_model'i işaret edebilir veya boş olabilir

    # OpenCLIP Model (ViT-H-14-378-quickgelu, pretrained: dfn5b)
    OPENCLIP_MODEL_TYPE = 'clip'
    OPENCLIP_MODEL_NAME = 'ViT-H-14-378-quickgelu_dfn5b' # model_name ve pretrained birleştirildi
    OPENCLIP_MODEL_BASE_PATH = os.path.join(MODELS_FOLDER, OPENCLIP_MODEL_TYPE, OPENCLIP_MODEL_NAME, 'base_model')
    OPENCLIP_MODEL_VERSIONS_PATH = os.path.join(MODELS_FOLDER, OPENCLIP_MODEL_TYPE, OPENCLIP_MODEL_NAME, 'versions')
    OPENCLIP_MODEL_ACTIVE_PATH = os.path.join(MODELS_FOLDER, OPENCLIP_MODEL_TYPE, OPENCLIP_MODEL_NAME, 'active_model') # Başlangıçta base_model'i işaret edebilir

    # Custom Age Head Model
    AGE_MODEL_TYPE = 'age'
    AGE_MODEL_NAME = 'custom_age_head'
    AGE_MODEL_BASE_PATH = os.path.join(MODELS_FOLDER, AGE_MODEL_TYPE, AGE_MODEL_NAME, 'base_model')
    AGE_MODEL_VERSIONS_PATH = os.path.join(MODELS_FOLDER, AGE_MODEL_TYPE, AGE_MODEL_NAME, 'versions')
    AGE_MODEL_ACTIVE_PATH = os.path.join(MODELS_FOLDER, AGE_MODEL_TYPE, AGE_MODEL_NAME, 'active_model')

    # Content Analysis Model (OpenCLIP with Classification Head)
    CONTENT_MODEL_TYPE = 'content'
    CONTENT_MODEL_NAME = 'openclip_classifier'
    CONTENT_MODEL_BASE_PATH = os.path.join(MODELS_FOLDER, CONTENT_MODEL_TYPE, CONTENT_MODEL_NAME, 'base_model')
    CONTENT_MODEL_VERSIONS_PATH = os.path.join(MODELS_FOLDER, CONTENT_MODEL_TYPE, CONTENT_MODEL_NAME, 'versions')
    CONTENT_MODEL_ACTIVE_PATH = os.path.join(MODELS_FOLDER, CONTENT_MODEL_TYPE, CONTENT_MODEL_NAME, 'active_model')

    # Diğer modeller için benzer tanımlamalar eklenebilir (örn: YOLO)
    YOLO_MODEL_TYPE = 'detection'
    YOLO_MODEL_NAME = 'yolov8x' # veya 'yolov8n' gibi
    YOLO_MODEL_BASE_PATH = os.path.join(MODELS_FOLDER, YOLO_MODEL_TYPE, YOLO_MODEL_NAME, 'base_model')
    YOLO_MODEL_VERSIONS_PATH = os.path.join(MODELS_FOLDER, YOLO_MODEL_TYPE, YOLO_MODEL_NAME, 'versions')
    YOLO_MODEL_ACTIVE_PATH = os.path.join(MODELS_FOLDER, YOLO_MODEL_TYPE, YOLO_MODEL_NAME, 'active_model')
    # === Merkezi Model Yolları Sonu ===

    # Analiz Ayarları
    DEFAULT_FRAMES_PER_SECOND = 1  # Video analizi için varsayılan saniyede kare sayısı
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Nesne tespiti için varsayılan güven eşiği
    # MIN_FACE_SIZE = 20 # Piksel cinsinden minimum yüz boyutu (KALDIRILDI)
    # CONTENT_ANALYSIS_PROFILE = "balanced" # Analiz profili: balanced, fast, detailed (KALDIRILDI)
    # VIDEO_FRAME_SAMPLING_RATE = 1 # Videolardan saniyede kaç kare analiz edileceği (KALDIRILDI)

    # Yeni Global Analiz Parametreleri (Kullanıcının resmindeki)
    FACE_DETECTION_CONFIDENCE = 0.5 # (0.1 - 1.0)
    TRACKING_RELIABILITY_THRESHOLD = 0.5 # (0.1 - 0.9)
    ID_CHANGE_THRESHOLD = 0.45 # (0.1 - 0.8)
    MAX_LOST_FRAMES = 30 # (5 - 300)
    EMBEDDING_DISTANCE_THRESHOLD = 0.4 # (0.1 - 0.8)

    # Görüntü işleme
    CLIP_ADULT_THRESHOLD = 0.6 # Örnek eşik değeri, ihtiyaca göre ayarlayın
    CLIP_VIOLENCE_THRESHOLD = 0.7
    CLIP_HARASSMENT_THRESHOLD = 0.7 # Yeni eklendi

    # Yaş tahmini için CLIP güven eşiği (sözde etiketleme veri kaydı için)
    PSEUDO_LABEL_RECORD_CLIP_THRESHOLD = 0.75 # Yeni İngilizce standart isim
    
    # Eğitim Verisi Saklama Politikası
    TRAINING_DATA_RETENTION_POLICY = {
        'pseudo_label_max_age_days': 180,
        'max_feedback_per_person': 3,
        'keep_manual_feedback': True
    }

class DevelopmentConfig(Config):
    """Geliştirme ortamı konfigürasyonu"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.getcwd(), 'wsanaliz_dev.db')
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Test ortamı konfigürasyonu"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # Bellekte test veritabanı
    WTF_CSRF_ENABLED = False  # Test için CSRF korumasını kapat
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production ortamı konfigürasyonu"""
    DEBUG = False
    
    # Production veritabanı - çevre değişkeninden al
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.getcwd(), 'wsanaliz_production.db')
    
    # Güvenlik için güçlü secret key zorunlu
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("Production ortamında SECRET_KEY çevre değişkeni zorunludur!")
    
    # Production güvenlik ayarları
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Performance optimizasyonları
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20
    }
    
    # Log seviyesi
    LOG_LEVEL = 'WARNING'
    
    # CORS - production domainleri
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []

# Konfigürasyon sözlüğü
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 