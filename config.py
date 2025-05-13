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
    USE_GPU = os.environ.get('USE_GPU', 'True').lower() in ('true', '1', 't')
    
    # Analiz Ayarları
    DEFAULT_FRAMES_PER_SECOND = 1  # Video analizi için varsayılan saniyede kare sayısı
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Nesne tespiti için varsayılan güven eşiği
    # MIN_FACE_SIZE = 20 # Piksel cinsinden minimum yüz boyutu (KALDIRILDI)
    # CONTENT_ANALYSIS_PROFILE = "balanced" # Analiz profili: balanced, fast, detailed (KALDIRILDI)
    # VIDEO_FRAME_SAMPLING_RATE = 1 # Videolardan saniyede kaç kare analiz edileceği (KALDIRILDI)

    # Yeni Global Analiz Parametreleri (Kullanıcının resmindeki)
    FACE_DETECTION_CONFIDENCE = 0.2 # (0.1 - 1.0)
    TRACKING_RELIABILITY_THRESHOLD = 0.25 # (0.1 - 0.9)
    ID_CHANGE_THRESHOLD = 0.5 # (0.1 - 0.8)
    MAX_LOST_FRAMES = 30 # (5 - 300)
    EMBEDDING_DISTANCE_THRESHOLD = 0.45 # (0.1 - 0.8)

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