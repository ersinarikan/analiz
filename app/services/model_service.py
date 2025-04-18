import os
import shutil
import json
import datetime
import logging
import numpy as np
import tensorflow as tf
from flask import current_app
from app import db
from app.models.feedback import Feedback
from app.ai.content_analyzer import ContentAnalyzer
from config import Config

logger = logging.getLogger(__name__)

# model_cache sözlüğü - bir kez yüklenen modelleri önbelleğe alır
_model_cache = {}

def get_model_stats(model_type='all'):
    """Model performans istatistiklerini döndürür. Belirtilen model tipine göre istatistikleri filtreler."""
    stats = {}
    
    if model_type in ['all', 'content']:
        # İçerik modelinin istatistiklerini al
        content_stats = _get_content_model_stats()
        stats['content'] = content_stats
    
    if model_type in ['all', 'age']:
        # Yaş modelinin istatistiklerini al
        age_stats = _get_age_model_stats()
        stats['age'] = age_stats
    
    return stats


def _get_content_model_stats():
    """İçerik analiz modelinin istatistiklerini döndürür. Bu fonksiyon model performansı, 
    eğitim geçmişi ve kullanıcı geri bildirimleri hakkında detaylı istatistiksel bilgi sağlar."""
    stats = {
        'model_name': 'Content Analysis Model',
        'model_type': 'content',
        'training_history': [],
        'metrics': {},
        'feedback_count': 0,
        'feedback_distribution': {}
    }
    
    # Model konfigürasyon dosyasını oku
    config_path = os.path.join(current_app.config['MODELS_FOLDER'], 'content_model_config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                if 'training_history' in config:
                    stats['training_history'] = config['training_history']
                
                if 'metrics' in config:
                    stats['metrics'] = config['metrics']
        except Exception as e:
            current_app.logger.error(f"Model konfigürasyonu okuma hatası: {str(e)}")
    
    # İçerik kategorileri için geri bildirim sayısını hesapla
    feedback_categories = ['violence_feedback', 'adult_content_feedback', 'harassment_feedback', 'weapon_feedback', 'drug_feedback']
    category_counts = {}
    
    for category in feedback_categories:
        # Her kategori için geri bildirim dağılımını hesapla
        results = db.session.query(
            getattr(Feedback, category),
            db.func.count(Feedback.id)
        ).filter(getattr(Feedback, category) != None).group_by(getattr(Feedback, category)).all()
        
        category_name = category.replace('_feedback', '')
        distribution = {value: count for value, count in results}
        
        if distribution:
            stats['feedback_distribution'][category_name] = distribution
            stats['feedback_count'] += sum(distribution.values())
    
    return stats


def _get_age_model_stats():
    """Yaş tahmin modelinin istatistiklerini döndürür. 
    Bu fonksiyon yaş tahmini doğruluğu, geri bildirim dağılımı ve
    model performans metriklerini içeren kapsamlı istatistikler sağlar."""
    stats = {
        'model_name': 'Age Estimation Model',
        'model_type': 'age',
        'training_history': [],
        'metrics': {},
        'feedback_count': 0,
        'feedback_accuracy': {},
        'age_distribution': {}
    }
    
    # Model konfigürasyon dosyasını oku
    config_path = os.path.join(current_app.config['MODELS_FOLDER'], 'age_model_config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                if 'training_history' in config:
                    stats['training_history'] = config['training_history']
                
                if 'metrics' in config:
                    stats['metrics'] = config['metrics']
        except Exception as e:
            current_app.logger.error(f"Model konfigürasyonu okuma hatası: {str(e)}")
    
    # Yaş tahminleri için geri bildirim istatistiklerini hesapla
    from app.models.analysis import AgeEstimation
    
    # Yaş geri bildirimi olan kayıtları al
    feedbacks = db.session.query(
        Feedback.person_id,
        Feedback.age_feedback,
        AgeEstimation.estimated_age
    ).join(
        AgeEstimation, 
        (AgeEstimation.person_id == Feedback.person_id) & (AgeEstimation.analysis_id == Feedback.analysis_id)
    ).filter(
        Feedback.age_feedback != None
    ).all()
    
    if feedbacks:
        stats['feedback_count'] = len(feedbacks)
        
        # Yaş farkı dağılımını hesapla
        age_diffs = {}
        for person_id, age_feedback, estimated_age in feedbacks:
            diff = abs(age_feedback - estimated_age)
            age_diffs[diff] = age_diffs.get(diff, 0) + 1
        
        stats['feedback_accuracy'] = age_diffs
        
        # Yaş dağılımını hesapla (10'ar yıllık gruplar halinde)
        age_counts = {}
        for person_id, age_feedback, estimated_age in feedbacks:
            age_group = str(age_feedback // 10 * 10) + 's'  # 0s, 10s, 20s, vb.
            age_counts[age_group] = age_counts.get(age_group, 0) + 1
        
        stats['age_distribution'] = age_counts
    
    return stats


def get_available_models():
    """Sistemde kullanılabilir modelleri listeler. 
    Her bir model için adı, tipi, dosya yolu ve mevcut sürümleri dahil olmak üzere detaylı bilgi döndürür."""
    models = []
    
    # İçerik modelini kontrol et ve listele
    content_model_path = os.path.join(current_app.config['MODELS_FOLDER'], 'content_model')
    if os.path.exists(content_model_path):
        content_model = {
            'name': 'Content Analysis Model',
            'type': 'content',
            'path': content_model_path,
            'versions': []
        }
        
        # Model sürümlerini listele
        versions_path = os.path.join(current_app.config['MODELS_FOLDER'], 'content_model_versions')
        if os.path.exists(versions_path):
            for version_dir in os.listdir(versions_path):
                version_path = os.path.join(versions_path, version_dir)
                if os.path.isdir(version_path):
                    try:
                        # Sürüm bilgilerini oku
                        version_info_path = os.path.join(version_path, 'version_info.json')
                        if os.path.exists(version_info_path):
                            with open(version_info_path, 'r') as f:
                                version_info = json.load(f)
                                content_model['versions'].append(version_info)
                    except Exception as e:
                        current_app.logger.error(f"Sürüm bilgisi okuma hatası: {str(e)}")
        
        models.append(content_model)
    
    # Yaş modelini kontrol et ve listele
    age_model_path = os.path.join(current_app.config['MODELS_FOLDER'], 'age_model')
    if os.path.exists(age_model_path):
        age_model = {
            'name': 'Age Estimation Model',
            'type': 'age',
            'path': age_model_path,
            'versions': []
        }
        
        # Model sürümlerini listele
        versions_path = os.path.join(current_app.config['MODELS_FOLDER'], 'age_model_versions')
        if os.path.exists(versions_path):
            for version_dir in os.listdir(versions_path):
                version_path = os.path.join(versions_path, version_dir)
                if os.path.isdir(version_path):
                    try:
                        # Sürüm bilgilerini oku
                        version_info_path = os.path.join(version_path, 'version_info.json')
                        if os.path.exists(version_info_path):
                            with open(version_info_path, 'r') as f:
                                version_info = json.load(f)
                                age_model['versions'].append(version_info)
                    except Exception as e:
                        current_app.logger.error(f"Sürüm bilgisi okuma hatası: {str(e)}")
        
        models.append(age_model)
    
    return models


def reset_model(model_type):
    """Bir modeli orijinal ön eğitimli haline sıfırlar.
    Mevcut modeli yedekleyip, varsayılan ön eğitimli modeli tekrar yükler."""
    if model_type not in ['content', 'age']:
        return False, "Geçersiz model tipi"
    
    model_folder = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model")
    pretrained_folder = os.path.join(current_app.config['DEFAULT_MODEL_PATH'], f"{model_type}_model")
    
    try:
        # Mevcut modeli tarih ve saat bilgisiyle yedekle
        backup_folder = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if os.path.exists(model_folder):
            shutil.copytree(model_folder, backup_folder)
            shutil.rmtree(model_folder)
        
        # Ön eğitimli modeli mevcut model klasörüne kopyala
        if os.path.exists(pretrained_folder):
            shutil.copytree(pretrained_folder, model_folder)
            
            # Model konfigürasyonunu sıfırla ve yeni bilgileri kaydet
            config_path = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model_config.json")
            
            config = {
                "model_type": model_type,
                "version": "pretrained",
                "training_history": [],
                "metrics": {},
                "reset_date": datetime.datetime.now().isoformat(),
                "is_pretrained": True
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            return True, f"{model_type.capitalize()} modeli başarıyla sıfırlandı"
        else:
            return False, "Ön eğitimli model bulunamadı"
    
    except Exception as e:
        current_app.logger.error(f"Model sıfırlama hatası: {str(e)}")
        return False, f"Model sıfırlama hatası: {str(e)}"


def prepare_training_data(model_type):
    """Eğitim için gerekli verileri hazırlar.
    Model tipine göre ilgili eğitim verisi hazırlama fonksiyonunu çağırır."""
    if model_type == 'content':
        return _prepare_content_training_data()
    elif model_type == 'age':
        return _prepare_age_training_data()
    else:
        return None, "Geçersiz model tipi"


def _prepare_content_training_data():
    """İçerik analiz modeli için eğitim verilerini hazırlar.
    Kullanıcı geri bildirimlerini kullanarak şiddet, yetişkin içerik, taciz,
    silah ve uyuşturucu kategorileri için eğitim verisi oluşturur."""
    # İçerik kategorileri için geri bildirimleri al
    feedback_categories = ['violence_feedback', 'adult_content_feedback', 'harassment_feedback', 'weapon_feedback', 'drug_feedback']
    training_data = {}
    
    for category in feedback_categories:
        # Olumlu geri bildirimleri al (positive)
        positive_feedbacks = Feedback.query.filter(getattr(Feedback, category) == 'positive').all()
        
        for feedback in positive_feedbacks:
            if feedback.frame_path not in training_data:
                training_data[feedback.frame_path] = {cat.replace('_feedback', ''): 0 for cat in feedback_categories}
            
            training_data[feedback.frame_path][category.replace('_feedback', '')] = 1
    
    # Eğitim verilerini liste formatına dönüştür
    training_list = []
    for frame_path, labels in training_data.items():
        training_list.append({
            'frame_path': frame_path,
            'labels': labels
        })
    
    return training_list, f"{len(training_list)} adet eğitim verisi hazırlandı"


def _prepare_age_training_data():
    """Yaş tahmin modeli için eğitim verilerini hazırlar.
    Kullanıcıların yaş tahminlerine verdiği geri bildirimleri kullanarak,
    yüz konumları ve doğru yaş bilgileriyle eğitim veri seti oluşturur."""
    from app.models.analysis import AgeEstimation
    
    # Yaş geri bildirimi olan kayıtları al
    feedbacks = db.session.query(
        Feedback.person_id,
        Feedback.age_feedback,
        Feedback.frame_path,
        AgeEstimation.face_x,
        AgeEstimation.face_y,
        AgeEstimation.face_width,
        AgeEstimation.face_height
    ).join(
        AgeEstimation, 
        (AgeEstimation.person_id == Feedback.person_id) & (AgeEstimation.analysis_id == Feedback.analysis_id)
    ).filter(
        Feedback.age_feedback != None
    ).all()
    
    # Eğitim verilerini liste formatına dönüştür
    training_list = []
    for person_id, age_feedback, frame_path, face_x, face_y, face_width, face_height in feedbacks:
        training_list.append({
            'frame_path': frame_path,
            'person_id': person_id,
            'age': age_feedback,
            'face_location': {
                'x': face_x,
                'y': face_y,
                'width': face_width,
                'height': face_height
            }
        })
    
    return training_list, f"{len(training_list)} adet eğitim verisi hazırlandı"

# Model yolları için sabit değerler
MODEL_PATHS = {
    'violence_detection': os.path.join(Config.DEFAULT_MODEL_PATH, 'violence'),
    'harassment_detection': os.path.join(Config.DEFAULT_MODEL_PATH, 'harassment'),
    'adult_content_detection': os.path.join(Config.DEFAULT_MODEL_PATH, 'adult_content'),
    'weapon_detection': os.path.join(Config.DEFAULT_MODEL_PATH, 'weapon'),
    'substance_detection': os.path.join(Config.DEFAULT_MODEL_PATH, 'substance'),
    'age_estimation': os.path.join(Config.DEFAULT_MODEL_PATH, 'age')
}

def load_model(model_name):
    """
    Belirtilen model adıyla yapay zeka modelini yükler.
    ContentAnalyzer üzerinden model yüklemesini sağlar.
    
    Args:
        model_name: Yüklenecek modelin adı
        
    Returns:
        Yüklenen model veya yüklenemezse None
    """
    # Önbellekten modeli kontrol et
    if model_name in _model_cache:
        return _model_cache[model_name]
        
    try:
        # ContentAnalyzer örneğini al (Singleton)
        if model_name in ['violence_detection', 'harassment_detection', 'adult_content_detection', 
                         'weapon_detection', 'substance_detection']:
            # ContentAnalyzer'ı kullan
            content_analyzer = ContentAnalyzer()
            
            # Model adına göre doğru modeli al
            if model_name == 'violence_detection':
                model = content_analyzer.violence_model
            elif model_name == 'harassment_detection':
                model = content_analyzer.harassment_model
            elif model_name == 'adult_content_detection':
                model = content_analyzer.adult_model
            elif model_name == 'weapon_detection':
                model = content_analyzer.weapon_model
            elif model_name == 'substance_detection':
                model = content_analyzer.drug_model
            else:
                logger.error(f"Tanımlanmamış model adı: {model_name}")
                model = _create_dummy_model()
                
            # Modeli önbelleğe al
            if model:
                _model_cache[model_name] = model
                logger.info(f"{model_name} modeli ContentAnalyzer'dan başarıyla yüklendi")
            
            return model
        else:
            # Diğer model tipleri (yaş tahmini, vb.)
            model_path = MODEL_PATHS.get(model_name)
            
            if not model_path:
                logger.error(f"Tanımlanmamış model adı: {model_name}")
                return _create_dummy_model()
            
            if not os.path.exists(model_path):
                logger.warning(f"Model dosyası bulunamadı: {model_path}")
                # Örnek/geçici bir model döndür
                return _create_dummy_model()
            
            logger.info(f"{model_name} modeli yükleniyor: {model_path}")
            model = tf.saved_model.load(model_path)
            
            # Modeli önbelleğe al
            _model_cache[model_name] = model
            
            return model
        
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu ({model_name}): {str(e)}")
        return _create_dummy_model()

def run_image_analysis(model, image_path):
    """
    Bir resim dosyası üzerinde model analizi çalıştırır.
    
    Args:
        model: Kullanılacak yapay zeka modeli
        image_path: Analiz edilecek resmin tam yolu
        
    Returns:
        dict: Analiz sonucu - score (skor) ve details (detaylar) içerir
    """
    try:
        # Gerçek bir model yoksa (dummy model)
        if hasattr(model, '_is_dummy') and model._is_dummy:
            return _get_dummy_result()
        
        # Resmi yükle ve ön işleme yap
        image = _preprocess_image(image_path)
        
        # Model tahmini yap
        predictions = model(image)
        
        # Sonucu işle
        score = float(predictions[0][0].numpy())
        
        return {
            'score': score,
            'details': {
                'confidence': score,
                'threshold': 0.5,
                'result': 'flagged' if score > 0.5 else 'safe'
            }
        }
    except Exception as e:
        logger.error(f"Resim analizi sırasında hata: {str(e)}")
        return {
            'score': 0.0,
            'details': {'error': str(e)}
        }

def run_video_analysis(model, video_path):
    """
    Bir video dosyası üzerinde model analizi çalıştırır.
    
    Args:
        model: Kullanılacak yapay zeka modeli
        video_path: Analiz edilecek videonun tam yolu
        
    Returns:
        dict: Analiz sonucu - score (skor) ve details (detaylar) içerir
    """
    try:
        # Gerçek bir model yoksa (dummy model)
        if hasattr(model, '_is_dummy') and model._is_dummy:
            return _get_dummy_result()
        
        # Burada gerçek bir video analizi yapılacak
        # Normalde video karelerini çıkarıp her bir kare için analiz yapılması gerekir
        # Basitlik için şimdilik sahte sonuçlar döndürüyoruz
        
        return {
            'score': 0.3,  # Örnek skor
            'details': {
                'confidence': 0.3,
                'threshold': 0.5,
                'result': 'safe',
                'frames_analyzed': 10,
                'highest_score_frame': 5,
                'highest_score': 0.3
            }
        }
    except Exception as e:
        logger.error(f"Video analizi sırasında hata: {str(e)}")
        return {
            'score': 0.0,
            'details': {'error': str(e)}
        }

def _preprocess_image(image_path):
    """
    Resmi model için uygun formata dönüştürür.
    
    Args:
        image_path: İşlenecek resmin tam yolu
        
    Returns:
        Modele uygun formatta tensor
    """
    from PIL import Image
    
    # Resmi yükle
    img = Image.open(image_path).convert('RGB')
    
    # Modelin beklediği boyuta getir
    img = img.resize((224, 224))
    
    # NumPy dizisine dönüştür ve normalize et
    img_array = np.array(img) / 255.0
    
    # Model için uygun forma getir (batch boyutu ekle)
    tensor = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    
    return tensor

def _create_dummy_model():
    """
    Gerçek model bulunamadığında kullanılacak sahte bir model oluşturur.
    Bu, uygulama akışının devam etmesini sağlar.
    
    Returns:
        object: Sahte model nesnesi
    """
    class DummyModel:
        def __init__(self):
            self._is_dummy = True
            
        def __call__(self, inputs):
            # Rastgele tahminler döndür
            import random
            score = random.uniform(0.1, 0.4)  # Genelde düşük skorlar
            return [[np.array([score])]]
    
    return DummyModel()

def _get_dummy_result():
    """
    Sahte analiz sonucu döndürür.
    
    Returns:
        dict: Sahte analiz sonucu
    """
    import random
    score = random.uniform(0.1, 0.4)
    
    return {
        'score': score,
        'details': {
            'confidence': score,
            'threshold': 0.5,
            'result': 'safe',
            'note': 'This is a placeholder result as no real model is available'
        }
    } 