import os 
from dotenv import load_dotenv 

"""
Uygulama konfigürasyon dosyası.
- Ortam değişkenleri, model ve ayar state yönetimi içerir.
"""

# ERSIN .env dosyasını yükle
load_dotenv ()

# ERSIN TensorFlow uyarılarını bastır, tüm modüller için geçerli
os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'# ERSIN 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os .environ ['TF_ENABLE_ONEDNN_OPTS']='0'# ERSIN oneDNN uyarılarını kapat

# ERSIN Ultralytics uyarılarını bastır
os .environ ['YOLO_VERBOSE']='False'

# ERSIN Albumentations güncelleme kontrolünü devre dışı bırak
os .environ ['NO_ALBUMENTATIONS_UPDATE']='1'

# ERSIN Model state'i import et, Flask debug mode bu dosyayı izleyecek
try :
    from app .utils .model_state import get_model_state ,LAST_UPDATE as MODEL_LAST_UPDATE
    MODEL_STATE =get_model_state ()# ERSIN Thread-safe getter kullan
    _ = MODEL_LAST_UPDATE  # re-export / side-effect for Flask debug reload
except ImportError :
    MODEL_STATE ={}
    MODEL_LAST_UPDATE =None 

    # ERSIN Settings state'i import et, Flask debug mode bu dosyayı izleyecek
try :
    from app .utils .settings_state import LAST_UPDATE as SETTINGS_LAST_UPDATE
    _ = SETTINGS_LAST_UPDATE  # re-export / side-effect for Flask debug reload
except ImportError :
    SETTINGS_STATE ={}
    SETTINGS_LAST_UPDATE =None 

BASE_DIR =os .path .dirname (os .path .abspath (__file__ ))
# ERSIN Mutlak path: worker/subprocess farklı cwd ile aynı DB dosyasına baksın
_db_path =os .path .abspath (os .path .join (BASE_DIR ,"wsanaliz_dev.db"))
SQLALCHEMY_DATABASE_URI =f"sqlite:///{_db_path}"

class Config :
# ERSIN Uygulama Ayarları
    SECRET_KEY =os .environ .get ('SECRET_KEY')or 'guvenli-anahtar-buraya'
    DEBUG =os .environ .get ('DEBUG','False').lower ()in ('true','1','t')

    # ERSIN Auth / Session
    # ERSIN PAM sadece sunucuda çalışır, browser auth session cookie olarak implement edilir
    WSANALIZ_AUTH_DISABLED =(os .environ .get ("WSANALIZ_AUTH_DISABLED","")or "").strip ().lower ()in {"1","true","yes","y","on"}
    # ERSIN "su" kullan, /etc/pam.d/login içinde pam_lastlog hatası ve ek oturum modülleri var
    # ERSIN "su" sadece common-auth (pam_unix) kullanır, TTY/oturum bağımsız çalışır
    WSANALIZ_PAM_SERVICE =(os .environ .get ("WSANALIZ_PAM_SERVICE")or "su").strip ()

    # ERSIN Flask-Session yapılandırması, server-side sessions
    SESSION_TYPE =(os .environ .get ("WSANALIZ_SESSION_TYPE")or "filesystem").strip ()# ERSIN Aciklama.
    SESSION_PERMANENT =True 
    SESSION_USE_SIGNER =True 
    SESSION_COOKIE_NAME =os .environ .get ("WSANALIZ_SESSION_COOKIE_NAME")or "wsanaliz_session"
    SESSION_COOKIE_HTTPONLY =True 
    SESSION_COOKIE_SAMESITE =os .environ .get ("WSANALIZ_SESSION_COOKIE_SAMESITE")or "Lax"
    # ERSIN Production'da HTTPS arkasında WSANALIZ_SESSION_COOKIE_SECURE=1 ayarla
    SESSION_COOKIE_SECURE =(os .environ .get ("WSANALIZ_SESSION_COOKIE_SECURE","")or "").strip ().lower ()in {"1","true","yes","y","on"}

    # ERSIN Logging Ayarları
    SHOW_HTTP_LOGS =os .environ .get ('SHOW_HTTP_LOGS','False').lower ()in ('true','1','t')

    # ERSIN Veritabanı Ayarları
    SQLALCHEMY_DATABASE_URI =os .environ .get ('DATABASE_URL')or SQLALCHEMY_DATABASE_URI 
    SQLALCHEMY_TRACK_MODIFICATIONS =False 

    # ERSIN Dosya Yükleme Ayarları
    STORAGE_FOLDER =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'storage')
    UPLOAD_FOLDER =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'storage','uploads')
    PROCESSED_FOLDER =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'storage','processed')
    MODELS_FOLDER =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'storage','models')
    MAX_CONTENT_LENGTH =500 *1024 *1024 # ERSIN 500 MB max yükleme boyutu

    # ERSIN Yapay Zeka Modeli Ayarları
    USE_GPU =os .environ .get ('USE_GPU','True').lower ()in ('true','1','t')

    # ERSIN === Merkezi Model Yolları ===
    # ERSIN InsightFace Age Model
    INSIGHTFACE_AGE_MODEL_TYPE ='age'
    INSIGHTFACE_AGE_MODEL_NAME ='buffalo_l'
    INSIGHTFACE_AGE_MODEL_BASE_PATH =os .path .join (MODELS_FOLDER ,INSIGHTFACE_AGE_MODEL_TYPE ,INSIGHTFACE_AGE_MODEL_NAME ,'base_model')
    INSIGHTFACE_AGE_MODEL_VERSIONS_PATH =os .path .join (MODELS_FOLDER ,INSIGHTFACE_AGE_MODEL_TYPE ,INSIGHTFACE_AGE_MODEL_NAME ,'versions')
    INSIGHTFACE_AGE_MODEL_ACTIVE_PATH =os .path .join (MODELS_FOLDER ,INSIGHTFACE_AGE_MODEL_TYPE ,INSIGHTFACE_AGE_MODEL_NAME ,'active_model')# ERSIN Başlangıçta base_model'i işaret edebilir veya boş olabilir

    # ERSIN OpenCLIP Model (ViT-H-14-378-quickgelu, pretrained: dfn5b)
    OPENCLIP_MODEL_TYPE ='clip'
    OPENCLIP_MODEL_NAME ='ViT-H-14-378-quickgelu_dfn5b'# ERSIN model_name ve pretrained birleştirildi
    OPENCLIP_MODEL_BASE_PATH =os .path .join (MODELS_FOLDER ,OPENCLIP_MODEL_TYPE ,OPENCLIP_MODEL_NAME ,'base_model')
    OPENCLIP_MODEL_VERSIONS_PATH =os .path .join (MODELS_FOLDER ,OPENCLIP_MODEL_TYPE ,OPENCLIP_MODEL_NAME ,'versions')
    OPENCLIP_MODEL_ACTIVE_PATH =os .path .join (MODELS_FOLDER ,OPENCLIP_MODEL_TYPE ,OPENCLIP_MODEL_NAME ,'active_model')# ERSIN Başlangıçta base_model'i işaret edebilir

    # ERSIN Custom Age Head Model
    AGE_MODEL_TYPE ='age'
    AGE_MODEL_NAME ='custom_age_head'
    AGE_MODEL_BASE_PATH =os .path .join (MODELS_FOLDER ,AGE_MODEL_TYPE ,AGE_MODEL_NAME ,'base_model')
    AGE_MODEL_VERSIONS_PATH =os .path .join (MODELS_FOLDER ,AGE_MODEL_TYPE ,AGE_MODEL_NAME ,'versions')
    AGE_MODEL_ACTIVE_PATH =os .path .join (MODELS_FOLDER ,AGE_MODEL_TYPE ,AGE_MODEL_NAME ,'active_model')

    # ERSIN Content Analysis Model (OpenCLIP with Classification Head)
    CONTENT_MODEL_TYPE ='content'
    CONTENT_MODEL_NAME ='openclip_classifier'
    CONTENT_MODEL_BASE_PATH =os .path .join (MODELS_FOLDER ,CONTENT_MODEL_TYPE ,CONTENT_MODEL_NAME ,'base_model')
    CONTENT_MODEL_VERSIONS_PATH =os .path .join (MODELS_FOLDER ,CONTENT_MODEL_TYPE ,CONTENT_MODEL_NAME ,'versions')
    CONTENT_MODEL_ACTIVE_PATH =os .path .join (MODELS_FOLDER ,CONTENT_MODEL_TYPE ,CONTENT_MODEL_NAME ,'active_model')

    # ERSIN Diğer modeller için benzer tanımlamalar eklenebilir (örn: YOLO)
    YOLO_MODEL_TYPE ='detection'
    YOLO_MODEL_NAME ='yolov8x'# ERSIN veya 'yolov8n' gibi
    YOLO_MODEL_BASE_PATH =os .path .join (MODELS_FOLDER ,YOLO_MODEL_TYPE ,YOLO_MODEL_NAME ,'base_model')
    YOLO_MODEL_VERSIONS_PATH =os .path .join (MODELS_FOLDER ,YOLO_MODEL_TYPE ,YOLO_MODEL_NAME ,'versions')
    YOLO_MODEL_ACTIVE_PATH =os .path .join (MODELS_FOLDER ,YOLO_MODEL_TYPE ,YOLO_MODEL_NAME ,'active_model')
    # ERSIN === Merkezi Model Yolları Sonu ===

    # ERSIN Analiz Ayarları
    DEFAULT_FRAMES_PER_SECOND =1 # ERSIN Video analizi için varsayılan saniyede kare sayısı
    DEFAULT_CONFIDENCE_THRESHOLD =0.5 # ERSIN Nesne tespiti için varsayılan güven eşiği
    # ERSIN Video kareleri için maksimum uzun kenar (px), 0 = orijinal çözünürlük
    VIDEO_FRAME_MAX_DIM =int (os .environ .get ('VIDEO_FRAME_MAX_DIM','720'))
    # ERSIN MIN_FACE_SIZE = 20 # Piksel cinsinden minimum yüz boyutu (KALDIRILDI)
    # ERSIN CONTENT_ANALYSIS_PROFILE = "balanced" # Analiz profili: balanced, fast, detailed (KALDIRILDI)
    # ERSIN VIDEO_FRAME_SAMPLING_RATE = 1 # Videolardan saniyede kaç kare analiz edileceği (KALDIRILDI)

    # ERSIN Yeni Global Analiz Parametreleri, önerilen fabrika varsayılanları ile uyumlu
    # ERSIN Not: 0.1 çok agresifti (false positive artışı), 0.25 daha dengeli başlangıç
    FACE_DETECTION_CONFIDENCE = 0.25 # ERSIN (0.1 - 1.0)
    TRACKING_RELIABILITY_THRESHOLD = 0.5 # ERSIN (0.1 - 0.9)
    ID_CHANGE_THRESHOLD = 0.45 # ERSIN (0.1 - 0.8)
    # ERSIN Not: 1 FPS gibi senaryolarda 30 frame ~= 30sn, 45 daha stabil takip sağlar
    MAX_LOST_FRAMES = 45 # ERSIN (5 - 300)
    EMBEDDING_DISTANCE_THRESHOLD = 0.4 # ERSIN (0.1 - 0.8)

    # ERSIN Görüntü işleme
    CLIP_ADULT_THRESHOLD =0.6 # ERSIN Örnek eşik değeri, ihtiyaca göre ayarlayın
    CLIP_VIOLENCE_THRESHOLD =0.7 
    CLIP_HARASSMENT_THRESHOLD =0.7 # ERSIN Yeni eklendi

    # ERSIN NSFW Model Ayarları
    NSFW_ENABLED =os .environ .get ('NSFW_ENABLED','False').lower ()in ('true','1','t')
    # ERSIN Falconsai modeli daha başarılı (%100 vs %5.4) ve daha hızlı (11.6ms vs 131ms)
    NSFW_MODEL_PATH =os .path .join (MODELS_FOLDER ,'nsfw','nsfw-detector-224.onnx')# ERSIN ÖNCEKİ: nsfw-detector-384.onnx
    NSFW_THRESHOLD =float (os .environ .get ('NSFW_THRESHOLD','0.3'))# ERSIN NSFW pozitif eşiği (0.3-0.5 önerilir)
    NSFW_USE_ONNX =True # ERSIN ONNX Runtime kullan, daha hızlı
    NSFW_INPUT_SIZE =224 # ERSIN Model input boyutu (Falconsai 224x224, önceki Marqo 384x384)

    # ERSIN Yaş tahmini için CLIP güven eşiği, sözde etiketleme veri kaydı için
    PSEUDO_LABEL_RECORD_CLIP_THRESHOLD =0.75 # ERSIN Yeni İngilizce standart isim

    # ERSIN Eğitim Verisi Saklama Politikası
    TRAINING_DATA_RETENTION_POLICY ={
    'pseudo_label_max_age_days':180 ,
    'max_feedback_per_person':3 ,
    'keep_manual_feedback':True 
    }

    # ERSIN Eğitim sonrası temizlik ayarı
    CLEANUP_TRAINING_DATA_AFTER_TRAINING =True # ERSIN Eğitim sonrası kullanılan verileri tamamen siler (VT + dosyalar)

    # ERSIN Merkezi fabrika ayarları, ör: settings_routes.py ve analysis_service.py'de kullanılıyor
    FACTORY_DEFAULTS ={
    "FACE_DETECTION_CONFIDENCE":0.25 ,
    "TRACKING_RELIABILITY_THRESHOLD":0.5 ,
    "ID_CHANGE_THRESHOLD":0.45 ,
    "MAX_LOST_FRAMES":45 ,
    "EMBEDDING_DISTANCE_THRESHOLD":0.4 
    }

    # ERSIN Güncellenebilecek parametreler ve tipleri, ör: settings_routes.py
    UPDATABLE_PARAMS ={
    "FACE_DETECTION_CONFIDENCE":float ,
    "TRACKING_RELIABILITY_THRESHOLD":float ,
    "ID_CHANGE_THRESHOLD":float ,
    "MAX_LOST_FRAMES":int ,
    "EMBEDDING_DISTANCE_THRESHOLD":float 
    }

    # ERSIN Model yolları, ör: model_service.py
    MODEL_PATHS ={
    'violence_detection':os .path .join (MODELS_FOLDER ,'violence'),
    'harassment_detection':os .path .join (MODELS_FOLDER ,'harassment'),
    'adult_content_detection':os .path .join (MODELS_FOLDER ,'adult_content'),
    'weapon_detection':os .path .join (MODELS_FOLDER ,'weapon'),
    'substance_detection':os .path .join (MODELS_FOLDER ,'substance'),
    'age_estimation':os .path .join (MODELS_FOLDER ,'age')
    }

    # ERSIN Eğitim için varsayılan parametreler, ör: model_service.py, training fonksiyonları
    DEFAULT_TRAINING_PARAMS ={
    'epochs':10 ,
    'batch_size':32 ,
    'learning_rate':0.001 ,
    'test_size':0.2 ,
    'hidden_dims':[256 ,128 ],
    'early_stopping_patience':10 
    }

class DevelopmentConfig (Config ):
    DEBUG =True 
    SQLALCHEMY_DATABASE_URI =Config .SQLALCHEMY_DATABASE_URI 

class TestingConfig (Config ):
    TESTING =True 
    SQLALCHEMY_DATABASE_URI =Config .SQLALCHEMY_DATABASE_URI 

class ProductionConfig (Config ):
    DEBUG =False 
    SQLALCHEMY_DATABASE_URI =Config .SQLALCHEMY_DATABASE_URI 

config ={
'development':DevelopmentConfig ,
'testing':TestingConfig ,
'production':ProductionConfig ,
'default':DevelopmentConfig 
}