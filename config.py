import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class Config:
    # Uygulama Ayarları
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'guvenli-anahtar-buraya'
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Veritabanı Ayarları
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///wsanaliz.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Dosya Yükleme Ayarları
    STORAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'uploads')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'processed')
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'models')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max yükleme boyutu
    
    # Yapay Zeka Modeli Ayarları
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static', 'models')
    CONTENT_MODEL_PATH = os.path.join(DEFAULT_MODEL_PATH, 'content_model')
    OBJECT_DETECTION_MODEL_PATH = os.path.join(DEFAULT_MODEL_PATH, 'object_detection')
    USE_GPU = os.environ.get('USE_GPU', 'True').lower() in ('true', '1', 't')
    
    # Analiz Ayarları
    DEFAULT_FRAMES_PER_SECOND = 1  # Video analizi için varsayılan saniyede kare sayısı
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Nesne tespiti için varsayılan güven eşiği

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///wsanaliz_dev.db'

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///wsanaliz_test.db'

class ProductionConfig(Config):
    DEBUG = False
    
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 