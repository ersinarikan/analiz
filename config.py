import os
from dotenv import load_dotenv

"""
Uygulama konfigürasyon dosyası.
- Ortam değişkenleri, model ve ayar state yönetimi içerir.
"""

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
    from app.utils.model_state import get_model_state, LAST_UPDATE as MODEL_LAST_UPDATE
    MODEL_STATE = get_model_state()  # Thread-safe getter kullan
except ImportError:
    MODEL_STATE = {}
    MODEL_LAST_UPDATE = None

# Settings state'i import et (Flask debug mode bu dosyayı izleyecek)
try:
    from app.utils.settings_state import SETTINGS_STATE, LAST_UPDATE as SETTINGS_LAST_UPDATE
except ImportError:
    SETTINGS_STATE = {}
    SETTINGS_LAST_UPDATE = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'wsanaliz_dev.db')}"

class Config:
    # Uygulama Ayarları
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'guvenli-anahtar-buraya'
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Logging Ayarları
    SHOW_HTTP_LOGS = os.environ.get('SHOW_HTTP_LOGS', 'False').lower() in ('true', '1', 't')
    
    # Veritabanı Ayarları
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Dosya Yükleme Ayarları
    STORAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'uploads')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'processed')
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'models')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max yükleme boyutu
    
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
    FACE_DETECTION_CONFIDENCE = 0.1 # (0.1 - 1.0) - Daha hassas yüz tespiti
    TRACKING_RELIABILITY_THRESHOLD = 0.3 # (0.1 - 0.9)
    ID_CHANGE_THRESHOLD = 0.45 # (0.1 - 0.8)
    MAX_LOST_FRAMES = None # (5 - 300)
    EMBEDDING_DISTANCE_THRESHOLD = 0.3 # (0.1 - 0.8)

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
    
    # Eğitim sonrası temizlik ayarı
    CLEANUP_TRAINING_DATA_AFTER_TRAINING = True  # Eğitim sonrası kullanılan verileri tamamen siler (VT + dosyalar)

    # Merkezi fabrika ayarları (ör: settings_routes.py ve analysis_service.py'de kullanılıyor)
    FACTORY_DEFAULTS = {
        "FACE_DETECTION_CONFIDENCE": 0.1,  # Daha hassas yüz tespiti için güncellendi
        "TRACKING_RELIABILITY_THRESHOLD": 0.5,
        "ID_CHANGE_THRESHOLD": 0.45,
        "MAX_LOST_FRAMES": 30,
        "EMBEDDING_DISTANCE_THRESHOLD": 0.4
    }

    # Güncellenebilecek parametreler ve tipleri (ör: settings_routes.py)
    UPDATABLE_PARAMS = {
        "FACE_DETECTION_CONFIDENCE": float,
        "TRACKING_RELIABILITY_THRESHOLD": float,
        "ID_CHANGE_THRESHOLD": float,
        "MAX_LOST_FRAMES": int,
        "EMBEDDING_DISTANCE_THRESHOLD": float
    }

    # Model yolları (ör: model_service.py)
    MODEL_PATHS = {
        'violence_detection': os.path.join(MODELS_FOLDER, 'violence'),
        'harassment_detection': os.path.join(MODELS_FOLDER, 'harassment'),
        'adult_content_detection': os.path.join(MODELS_FOLDER, 'adult_content'),
        'weapon_detection': os.path.join(MODELS_FOLDER, 'weapon'),
        'substance_detection': os.path.join(MODELS_FOLDER, 'substance'),
        'age_estimation': os.path.join(MODELS_FOLDER, 'age')
    }

    # Eğitim için varsayılan parametreler (ör: model_service.py, training fonksiyonları)
    DEFAULT_TRAINING_PARAMS = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'test_size': 0.2,
        'hidden_dims': [256, 128],
        'early_stopping_patience': 10
    }

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI
    
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 