import os
import shutil
import json
import datetime
import logging
import numpy as np
import time

# TensorFlow uyarılarını bastır
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
# Güncel TensorFlow 2.x logging API'si kullan
tf.get_logger().setLevel('ERROR')

# Yaş modeli için torch ve CustomAgeHead import'ları
import torch

from flask import current_app
from app import db
from app.models.feedback import Feedback
from app.ai.content_analyzer import ContentAnalyzer
from config import Config
from app.models.content import ModelVersion
from app.services import db_service
from app.ai.insightface_age_estimator import CustomAgeHead

logger = logging.getLogger(__name__)

# model_cache sözlüğü - bir kez yüklenen modelleri önbelleğe alır
_model_cache = {}

def load_age_model(model_path):
    """
    Belirtilen yoldan yaş tahmin modelini yükler (CustomAgeHead)

    Args:
        model_path: Model dosyasının tam yolu (.pth dosyası)

    Returns:
        Yüklenen CustomAgeHead modeli veya None
    """
    try:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and
            current_app.config.get('USE_GPU', True) else "cpu"
        )

        logger.info(f"Yaş modeli yükleniyor: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Model dosyası bulunamadı: {model_path}")
            return None

        # Model checkpoint'ini yükle
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

        # Model konfigürasyonunu al
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            model = CustomAgeHead(
                input_size=model_config['input_dim'],
                hidden_dims=model_config['hidden_dims'],
                output_dim=model_config['output_dim']
            )
        else:
            # Varsayılan konfigürasyon
            model = CustomAgeHead(input_size=512, hidden_dims=[256, 128], output_dim=1)

        # Model ağırlıklarını yükle
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Eski formatta kaydedilmiş olabilir
            model.load_state_dict(checkpoint)

        model.eval()  # Evaluation moduna geç
        model.to(device)

        logger.info(f"Yaş modeli başarıyla yüklendi: {model_path}")
        return model

    except Exception as e:
        logger.error(f"Yaş modeli yüklenirken hata: {str(e)}")
        return None

def load_content_model(model_path):
    """
    Belirtilen yoldan içerik analiz modelini yükler (CLIP tabanlı)

    Args:
        model_path: Model dosyasının tam yolu

    Returns:
        Yüklenen içerik modeli veya None
    """
    try:
        logger.info(f"İçerik modeli yükleniyor: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Model dosyası bulunamadı: {model_path}")
            return None

        # ContentAnalyzer'ın CLIP tabanlı modelini yükle
        content_analyzer = ContentAnalyzer()

        if content_analyzer.initialized:
            logger.info(f"İçerik analiz modeli başarıyla yüklendi: {model_path}")
            return content_analyzer
        else:
            logger.error(f"İçerik analiz modeli yüklenemedi: {model_path}")
            return None

    except Exception as e:
        logger.error(f"İçerik modeli yüklenirken hata: {str(e)}")
        return None

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
    content_feedbacks = Feedback.query.filter(
        Feedback.feedback_type.in_([
            'violence', 'adult', 'harassment', 'weapon', 'drug'
        ])
    ).all()

    if content_feedbacks:
        stats['feedback_count'] = len(content_feedbacks)

        # Kategori dağılımını hesapla
        category_counts = {}
        for feedback in content_feedbacks:
            category = feedback.feedback_type
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

        stats['feedback_distribution'] = category_counts

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

    # Yaş geri bildirimi olan kayıtları al
    feedbacks = Feedback.query.filter(
        (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo'),
        Feedback.corrected_age.isnot(None)
    ).all()

    if feedbacks:
        stats['feedback_count'] = len(feedbacks)

        # Yaş dağılımını hesapla (10'ar yıllık gruplar halinde)
        age_counts = {}
        for feedback in feedbacks:
            age = feedback.corrected_age or feedback.pseudo_label_original_age
            if age:
                age_group = str(int(age) // 10 * 10) + 's'  # 0s, 10s, 20s, vb.
                age_counts[age_group] = age_counts.get(age_group, 0) + 1

        stats['age_distribution'] = age_counts

        # Manuel vs pseudo feedback dağılımı
        manual_count = len([f for f in feedbacks if f.feedback_source == 'MANUAL_USER'])
        pseudo_count = len([f for f in feedbacks if f.feedback_source != 'MANUAL_USER'])
        stats['feedback_sources'] = {
            'manual': manual_count,
            'pseudo': pseudo_count
        }

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

    try:
        if model_type == 'age':
            # Yaş modeli için özel reset işlemi
            from app.services.age_training_service import AgeTrainingService
            trainer = AgeTrainingService()

            # Tüm versiyonları pasif yap
            from app.models.content import ModelVersion
            db.session.query(ModelVersion).filter_by(
                model_type='age',
                is_active=True
            ).update({'is_active': False})
            db.session.commit()

            # Base modeli tekrar aktif et
            success = trainer.reset_to_base_model()

            if success:
                return True, "Yaş tahmin modeli başarıyla sıfırlandı"
            else:
                return False, "Yaş tahmin modeli sıfırlanırken hata oluştu"

        elif model_type == 'content':
            # CLIP İçerik modeli için özel reset işlemi
            from app.models.clip_training import CLIPTrainingSession
            from app.models.content import ModelVersion

            # Tüm content model versiyonlarını pasif yap
            db.session.query(ModelVersion).filter_by(
                model_type='content',
                is_active=True
            ).update({'is_active': False})

            # Tüm CLIP training sessions'ları pasif yap
            db.session.query(CLIPTrainingSession).update({'is_active': False})

            # Base OpenCLIP modelini aktif yap
            base_session = CLIPTrainingSession.query.filter_by(
                version_name='base_openclip'
            ).first()

            # Base session yoksa oluştur
            if not base_session:
                base_session = CLIPTrainingSession(
                    version_name='base_openclip',
                    session_name='Base OpenCLIP Model',
                    status='completed',
                    is_active=True,
                    created_at=datetime.datetime.now(),
                    model_path=current_app.config['OPENCLIP_MODEL_BASE_PATH'],
                    training_data={
                        'total_pairs': 0,
                        'train_pairs': 0,
                        'val_pairs': 0
                    },
                    performance_metrics={
                        'type': 'base_model',
                        'description': 'Original pretrained OpenCLIP model'
                    }
                )
                db.session.add(base_session)
            else:
                base_session.is_active = True

            db.session.commit()

            # Active model klasörünü base model ile güncelle
            active_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
            base_model_path = current_app.config['OPENCLIP_MODEL_BASE_PATH']

            # Base model var mı kontrol et
            if not os.path.exists(base_model_path):
                return False, "Base OpenCLIP modeli bulunamadı"

            # Mevcut active model'i backup'la
            backup_path = os.path.join(
                current_app.config['MODELS_FOLDER'],
                'clip',
                'backups',
                f"active_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            if os.path.exists(active_model_path):
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copytree(active_model_path, backup_path)
                shutil.rmtree(active_model_path)

            # Base model'i active olarak kopyala
            shutil.copytree(base_model_path, active_model_path)

            # Active model metadata'sını güncelle
            metadata = {
                'reset_date': datetime.datetime.now().isoformat(),
                'source': 'base_model',
                'backup_path': backup_path if os.path.exists(backup_path) else None,
                'model_type': 'base_openclip',
                'session_id': base_session.id,
                'version_name': 'base_openclip'
            }

            metadata_path = os.path.join(active_model_path, 'version_info.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            return True, "İçerik analiz modeli başarıyla base OpenCLIP modeline sıfırlandı"
        else:
            # Diğer model tipleri için standart reset işlemi
            model_folder = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model")
            pretrained_folder = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model_pretrained")

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
        Feedback.age_feedback.isnot(None)
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
    'violence_detection': os.path.join(Config.MODELS_FOLDER, 'violence'),
    'harassment_detection': os.path.join(Config.MODELS_FOLDER, 'harassment'),
    'adult_content_detection': os.path.join(Config.MODELS_FOLDER, 'adult_content'),
    'weapon_detection': os.path.join(Config.MODELS_FOLDER, 'weapon'),
    'substance_detection': os.path.join(Config.MODELS_FOLDER, 'substance'),
    'age_estimation': os.path.join(Config.MODELS_FOLDER, 'age')
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
        if model_name in [
            'violence_detection', 'harassment_detection', 'adult_content_detection',
            'weapon_detection', 'substance_detection'
        ]:
            # ContentAnalyzer artık CLIP tabanlı çalışıyor
            content_analyzer = ContentAnalyzer()
            if content_analyzer.initialized:
                logger.info("ContentAnalyzer başarıyla yüklendi (model: %s)", model_name)
                _model_cache[model_name] = content_analyzer
                return content_analyzer
            else:
                logger.error(f"ContentAnalyzer yükleme başarısız oldu: initialized=False")
                return None
        elif model_name == 'detection':
            # Nesne tespiti için YOLO modelini al
            content_analyzer = ContentAnalyzer()
            if content_analyzer.initialized and hasattr(content_analyzer, 'yolo_model'):
                logger.info("YOLO detection modeli başarıyla yüklendi")
                _model_cache[model_name] = content_analyzer.yolo_model
                return content_analyzer.yolo_model
            else:
                logger.error("YOLO detection modeli yüklenemedi")
                return None
        elif model_name == 'clip':
            # CLIP modeli için
            content_analyzer = ContentAnalyzer()
            if content_analyzer.initialized and hasattr(content_analyzer, 'clip_model'):
                logger.info("CLIP modeli başarıyla yüklendi")
                _model_cache[model_name] = content_analyzer.clip_model
                return content_analyzer.clip_model
            else:
                logger.error("CLIP modeli yüklenemedi")
                return None
        else:
            logger.error(f"Bilinmeyen model adı: {model_name}")
            return None

    except Exception as e:
        logger.error(f"Model yükleme hatası ({model_name}): {str(e)}")
        return None

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
        # Burada gerçek bir video analizi yapılacak
        # Normalde video karelerini çıkarıp her bir kare için analiz yapılması gerekir
        # (Burada gerçek analiz kodu olmalı)
        logger.error("Gerçek video analizi fonksiyonu henüz uygulanmadı.")
        return {
            'score': 0.0,
            'details': {'error': 'Gerçek video analizi fonksiyonu eksik.'}
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

def train_with_feedback(model_type, params=None):
    """
    Kullanıcı geri bildirimleriyle model eğitimi yapar
    """
    logging.info(f"Geri bildirimlerle model eğitimi başlatılıyor: {model_type}")
    logging.debug(f"Eğitim parametreleri: {params}")

    # Varsayılan parametreleri ayarla
    if not params:
        params = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'test_size': 0.2,
            'hidden_dims': [256, 128],
            'early_stopping_patience': 10
        }
        logging.debug("Varsayılan parametreler kullanılıyor")
    else:
        # Eksik parametreleri varsayılan değerlerle tamamla
        default_params = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'test_size': 0.2,
            'hidden_dims': [256, 128],
            'early_stopping_patience': 10
        }
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        logging.debug(f"Parametreler tamamlandı: {params}")

    start_time = time.time()

    try:
        if model_type == 'age':
            # Custom Age modelini eğit
            from app.services.age_training_service import AgeTrainingService

            trainer = AgeTrainingService()

            # Veriyi hazırla
            training_data = trainer.prepare_training_data(min_samples=10)

            if training_data is None:
                return {
                    "success": False,
                    "message": "Yeterli sayıda geri bildirim verisi bulunamadı. En az 10 geri bildirim gerekli."
                }

            logging.info(f"Eğitim verisi hazırlandı: {len(training_data['embeddings'])} örnek")

            # Modeli eğit
            training_result = trainer.train_model(training_data, params)

            # Model versiyonunu kaydet
            model_version = trainer.save_model_version(
                training_result['model'],
                training_result
            )

            # Eğitim sonrası temizlik (opsiyonel)
            cleanup_enabled = current_app.config.get('CLEANUP_TRAINING_DATA_AFTER_TRAINING', True)
            if cleanup_enabled:
                logging.info("Starting post-training cleanup...")
                cleanup_report = trainer.cleanup_used_training_data(
                    training_result['used_feedback_ids'],
                    model_version.version_name
                )
                logging.info(f"Cleanup completed: {cleanup_report}")

            duration = time.time() - start_time

            logging.info(f"Model eğitimi tamamlandı. Süre: {duration:.2f} saniye")

            return {
                "success": True,
                "version": model_version.version,
                "version_name": model_version.version_name,
                "duration": duration,
                "metrics": training_result['metrics'],
                "training_samples": training_result['training_samples'],
                "validation_samples": training_result['validation_samples'],
                "epochs": len(training_result['history']['train_loss']),
                "model_id": model_version.id
            }

        elif model_type == 'content':
            # İçerik modeli eğitimi
            from app.services.content_training_service import ContentTrainingService

            content_trainer = ContentTrainingService()

            # Eğitim verisini hazırla
            training_data = content_trainer.prepare_training_data(min_samples=params.get('min_samples', 50))
            if not training_data:
                return {
                    "success": False,
                    "message": "Yeterli eğitim verisi bulunamadı"
                }

            # Modeli eğit
            training_result = content_trainer.train_model(training_data, params)

            # Modeli kaydet
            model_version = content_trainer.save_model_version(
                training_result['model'],
                training_result
            )

            return {
                "success": True,
                "message": f"İçerik modeli başarıyla eğitildi (v{model_version.version})",
                "model_id": model_version.id,
                "version": model_version.version,
                "metrics": training_result['metrics'],
                "training_samples": training_result['training_samples'],
                "validation_samples": training_result['validation_samples']
            }
        else:
            return {
                "success": False,
                "message": f"Geçersiz model tipi: {model_type}"
            }

    except Exception as e:
        logging.error(f"Model eğitimi sırasında hata: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Eğitim sırasında hata: {str(e)}"
        }

def prepare_feedback_data(feedback_data, model_type):
    """
    Geri bildirim verilerini eğitim için hazırlar
    """
    if model_type == 'content':
        return prepare_content_feedback(feedback_data)
    else:  # age
        return prepare_age_feedback(feedback_data)

def prepare_content_feedback(feedback_data):
    """
    İçerik analizi için geri bildirim verilerini hazırla
    """
    # İçerik ve kategori verileri
    train_data = []
    val_data = []

    # Veriyi eğitim (%80) ve doğrulama (%20) olarak ayır
    split_idx = int(len(feedback_data) * 0.8)

    # Karıştır
    np.random.shuffle(feedback_data)

    for i, feedback in enumerate(feedback_data):
        # İçerik ID'ye göre resim/video karesini bul
        content = db_service.get_content_by_id(feedback.content_id)
        if not content:
            continue

        # Kategori geri bildirimleri
        category_data = feedback.category_feedback

        # Eğitim verisi hazırla
        item = {
            "content_id": feedback.content_id,
            "frame_path": content.frame_path,
            "category_scores": {
                "violence": float(category_data.get("violence", 0)),
                "adult_content": float(category_data.get("adult_content", 0)),
                "harassment": float(category_data.get("harassment", 0)),
                "weapon": float(category_data.get("weapon", 0)),
                "drug": float(category_data.get("drug", 0))
            }
        }

        # Eğitim/doğrulama ayırma
        if i < split_idx:
            train_data.append(item)
        else:
            val_data.append(item)

    return {
        "train_data": train_data,
        "val_data": val_data
    }

def prepare_age_feedback(feedback_data):
    """
    Yaş tahmini için geri bildirim verilerini hazırla
    """
    # Yaş tahmini verileri
    train_data = []
    val_data = []

    # Veriyi eğitim (%80) ve doğrulama (%20) olarak ayır
    split_idx = int(len(feedback_data) * 0.8)

    # Karıştır
    np.random.shuffle(feedback_data)

    for i, feedback in enumerate(feedback_data):
        person_data = db_service.get_person_by_id(feedback.person_id)
        if not person_data:
            continue

        # Yaş verisi hazırla
        item = {
            "person_id": feedback.person_id,
            "face_image_path": person_data.face_image_path,
            "corrected_age": feedback.corrected_age
        }

        # Eğitim/doğrulama ayırma
        if i < split_idx:
            train_data.append(item)
        else:
            val_data.append(item)

    return {
        "train_data": train_data,
        "val_data": val_data
    }

def create_model_version(model_type, metrics, training_samples, validation_samples, epochs, feedback_ids):
    """
    Yeni bir model versiyonu oluşturur
    """
    # Son sürüm numarasını bul
    last_version = db.session.query(ModelVersion).filter_by(
        model_type=model_type
    ).order_by(ModelVersion.version.desc()).first()

    new_version_num = 1
    if last_version:
        new_version_num = last_version.version + 1

    # Tüm aktif sürümleri devre dışı bırak
    db.session.query(ModelVersion).filter_by(
        model_type=model_type,
        is_active=True
    ).update({ModelVersion.is_active: False})

    # Yeni sürüm oluştur
    model_version = ModelVersion(
        model_type=model_type,
        version=new_version_num,
        created_at=datetime.now(),
        metrics=metrics,
        is_active=True,
        training_samples=training_samples,
        validation_samples=validation_samples,
        epochs=epochs,
        file_path=f"models/{model_type}/version_{new_version_num}",
        weights_path=f"models/{model_type}/version_{new_version_num}/weights.pth",
        used_feedback_ids=feedback_ids
    )

    db.session.add(model_version)
    db.session.commit()

    logging.info(f"Yeni model versiyonu oluşturuldu: {model_type} v{new_version_num}")

    return model_version

def get_model_versions(model_type):
    """
    Belirli bir model tipi için tüm versiyonları getirir
    """
    if model_type == 'age':
        return get_age_model_versions()
    elif model_type == 'content':
        return get_content_model_versions()
    else:
        logger.warning(f"Desteklenmeyen model tipi: {model_type}")
        return {'success': False, 'error': 'Desteklenmeyen model tipi'}

def get_age_model_versions():
    """Yaş modeli versiyonlarını getir"""
    try:
        # Veritabanından versiyonları al
        versions_query = db.session.query(ModelVersion).filter_by(
            model_type='age'
        ).order_by(ModelVersion.version.desc())

        db_versions = versions_query.all()

        # Versions klasöründen fiziksel versiyonları kontrol et
        versions_path = current_app.config['AGE_MODEL_VERSIONS_PATH']
        physical_versions = []

        if os.path.exists(versions_path):
            for version_dir in os.listdir(versions_path):
                version_full_path = os.path.join(versions_path, version_dir)
                if os.path.isdir(version_full_path):
                    # metadata.json dosyasını kontrol et
                    metadata_path = os.path.join(version_full_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            physical_versions.append({
                                'version_name': version_dir,
                                'path': version_full_path,
                                'metadata': metadata
                            })
                        except Exception:
                            continue

        # Aktif model versiyonunu belirle
        active_model_path = current_app.config['AGE_MODEL_ACTIVE_PATH']
        active_version = None

        if os.path.exists(active_model_path):
            # active_model klasöründeki version_info.json'u oku
            active_info_path = os.path.join(active_model_path, 'version_info.json')
            if os.path.exists(active_info_path):
                try:
                    with open(active_info_path, 'r') as f:
                        active_info = json.load(f)
                        active_version = active_info.get('version_name')
                except Exception:
                    pass

        # Base model bilgisi - Buffalo + custom_age kombinasyonu v0 olarak kabul edilir
        base_model_path = current_app.config['AGE_MODEL_BASE_PATH']
        buffalo_path = current_app.config['INSIGHTFACE_AGE_MODEL_BASE_PATH']

        # Custom age head modeli + Buffalo modeli = v0 base model
        custom_age_exists = os.path.exists(os.path.join(base_model_path, 'model.pth'))
        buffalo_exists = os.path.exists(os.path.join(buffalo_path, 'w600k_r50.onnx'))
        base_model_exists = custom_age_exists and buffalo_exists

        # Versiyonları hazırla
        versions_list = []

        # Eğer hiç eğitilmiş model yoksa veya aktif versiyon belirlenmemişse, base model aktiftir
        has_active_custom_version = any(v.is_active for v in db_versions)
        base_is_active = not has_active_custom_version and active_version is None

        # Base model'i her zaman ilk versiyon olarak ekle
        if base_model_exists:
            versions_list.append({
                'id': 0,  # Base model için özel ID
                'version': 0,
                'version_name': 'base_model',
                'created_at': None,  # Base model için yaratılma tarihi yok
                'is_active': base_is_active,
                'training_samples': 0,
                'validation_samples': 0,
                'epochs': 0,
                'metrics': {
                    'type': 'base_pretrained',
                    'description': 'Buffalo-L + Custom Age Head (UTKFace eğitimli)',
                    'mae': 1.696  # Temel model için gerçek MAE değeri
                },
                'model_type': 'age'
            })

        # Veritabanındaki custom versiyonları ekle
        for v in db_versions:
            versions_list.append({
                'id': getattr(v, 'id', 0),
                'version': getattr(v, 'version', 0),
                'version_name': getattr(v, 'version_name', 'unknown'),
                'created_at': getattr(v, 'created_at', None).isoformat() if getattr(v, 'created_at', None) else None,
                'is_active': getattr(v, 'is_active', False) or (getattr(v, 'version_name', '') == active_version),
                'training_samples': getattr(v, 'training_samples', 0),
                'validation_samples': getattr(v, 'validation_samples', 0),
                'epochs': getattr(v, 'epochs', 0),
                'metrics': getattr(v, 'metrics', {}),
                'model_type': 'age'
            })

        return {
            'success': True,
            'versions': versions_list,
            'physical_versions': physical_versions,
            'active_version': active_version if active_version else ('base_model' if base_is_active else None),
            'base_model_exists': base_model_exists,
            'versions_path': versions_path,
            'active_path': active_model_path,
            'base_path': base_model_path
        }

    except Exception as e:
        logger.error(f"Age model versiyonları alınırken hata: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def get_content_model_versions():
    """İçerik modeli versiyonlarını getir"""
    try:
        # Veritabanından versiyonları al
        versions_query = db.session.query(ModelVersion).filter_by(
            model_type='content'
        ).order_by(ModelVersion.version.desc())

        db_versions = versions_query.all()

        # Versions klasöründen fiziksel versiyonları kontrol et
        versions_path = current_app.config['OPENCLIP_MODEL_VERSIONS_PATH']
        physical_versions = []

        if os.path.exists(versions_path):
            for version_dir in os.listdir(versions_path):
                version_full_path = os.path.join(versions_path, version_dir)
                if os.path.isdir(version_full_path):
                    # metadata.json dosyasını kontrol et
                    metadata_path = os.path.join(version_full_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            physical_versions.append({
                                'version_name': version_dir,
                                'path': version_full_path,
                                'metadata': metadata
                            })
                        except Exception:
                            continue

        # Aktif model versiyonunu belirle
        active_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
        active_version = None

        if os.path.exists(active_model_path):
            # active_model klasöründeki version_info.json varsa onu oku
            active_info_path = os.path.join(active_model_path, 'version_info.json')
            if os.path.exists(active_info_path):
                try:
                    with open(active_info_path, 'r') as f:
                        active_info = json.load(f)
                        active_version = active_info.get('version_name')
                except Exception:
                    pass

        # Base model bilgisi
        base_model_path = current_app.config['OPENCLIP_MODEL_BASE_PATH']
        base_model_exists = os.path.exists(os.path.join(base_model_path, 'open_clip_pytorch_model.bin'))

        # Versiyonları hazırla
        versions_list = []

        # Eğer hiç eğitilmiş model yoksa veya aktif versiyon belirlenmemişse, base model aktiftir
        has_active_custom_version = any(v.is_active for v in db_versions)
        base_is_active = not has_active_custom_version and active_version is None

        # Base OpenCLIP modelini her zaman ilk versiyon olarak ekle
        if base_model_exists:
            versions_list.append({
                'id': 0,  # Base model için özel ID
                'version': 0,
                'version_name': 'base_openclip',
                'created_at': None,  # Base model için yaratılma tarihi yok
                'is_active': base_is_active,
                'training_samples': 0,
                'validation_samples': 0,
                'epochs': 0,
                'metrics': {
                    'type': 'base_pretrained',
                    'description': 'OpenCLIP ViT-H/14 önceden eğitilmiş model'
                },
                'model_type': 'content'
            })

        # Veritabanındaki custom versiyonları ekle
        for v in db_versions:
            versions_list.append({
                'id': getattr(v, 'id', 0),
                'version': getattr(v, 'version', 0),
                'version_name': getattr(v, 'version_name', 'unknown'),
                'created_at': getattr(v, 'created_at', None).isoformat() if getattr(v, 'created_at', None) else None,
                'is_active': getattr(v, 'is_active', False) or (getattr(v, 'version_name', '') == active_version),
                'training_samples': getattr(v, 'training_samples', 0),
                'validation_samples': getattr(v, 'validation_samples', 0),
                'epochs': getattr(v, 'epochs', 0),
                'metrics': getattr(v, 'metrics', {}),
                'model_type': 'content'
            })

        return {
            'success': True,
            'versions': versions_list,
            'physical_versions': physical_versions,
            'active_version': active_version if active_version else ('base_openclip' if base_is_active else None),
            'base_model_exists': base_model_exists,
            'versions_path': versions_path,
            'active_path': active_model_path,
            'base_path': base_model_path
        }

    except Exception as e:
        logger.error(f"Content model versiyonları alınırken hata: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def update_model_state_file(model_type, version_id):
    """
    Model state'i thread-safe şekilde günceller (Flask auto-reload için)
    """
    try:
        from app.utils.model_state import update_model_state

        # Thread-safe update fonksiyonunu kullan
        update_model_state(model_type, version_id)

        logger.info(f"Model state thread-safe güncellendi: {model_type} -> version {version_id}")
        return True

    except Exception as e:
        logger.error(f"Model state güncellenirken hata: {str(e)}")
        return False

def activate_model_version(version_id):
    """
    Belirli bir model versiyonunu aktif hale getirir
    """
    try:
        # Versiyonu bul
        version = db.session.query(ModelVersion).filter_by(id=version_id).first()

        if not version:
            return {
                "success": False,
                "message": "Belirtilen model versiyonu bulunamadı"
            }

        # Aynı tipteki tüm aktif modelleri devre dışı bırak
        db.session.query(ModelVersion).filter_by(
            model_type=version.model_type,
            is_active=True
        ).update({ModelVersion.is_active: False})

        # Bu versiyonu aktif yap
        version.is_active = True
        db.session.commit()

        # Modeli yükle (uygulama tarafından kullanılmak üzere)
        load_specific_model(version.model_type, version.version)

        # Model state dosyasını güncelle (Flask auto-reload için)
        update_model_state_file(version.model_type, version_id)

        return {
            "success": True,
            "version": version.version,
            "model_type": version.model_type
        }
    except Exception as e:
        logging.error(f"Model versiyonu aktifleştirme hatası: {str(e)}")
        return {
            "success": False,
            "message": f"Model aktifleştirme hatası: {str(e)}"
        }

def load_specific_model(model_type, version):
    """
    Belirli bir versiyon numarasına sahip modeli yükler
    """
    try:
        # Versiyonu doğrula
        model_version = db.session.query(ModelVersion).filter_by(
            model_type=model_type,
            version=version
        ).first()

        if not model_version:
            logging.error(f"Belirtilen model versiyonu bulunamadı: {model_type} v{version}")
            return False

        # Model dosyasını kontrol et
        if not os.path.exists(model_version.weights_path):
            logging.error(f"Model dosyası bulunamadı: {model_version.weights_path}")
            return False

        # Modeli yükle
        if model_type == 'content':
            model = load_content_model(model_version.weights_path)
        else:  # age
            model = load_age_model(model_version.weights_path)

        # Global model değişkenine ata
        if model_type == 'content':
            global content_model
            content_model = model
        else:  # age
            global age_model
            age_model = model

        logging.info(f"{model_type} modeli v{version} başarıyla yüklendi")
        return True
    except Exception as e:
        logging.error(f"Model yükleme hatası: {str(e)}")
        return False

def calculate_metrics(model, training_data, model_type):
    """
    Model performans metriklerini hesaplar
    """
    metrics = {}

    try:
        if model_type == 'content':
            # İçerik analiz metriklerini hesapla
            val_data = training_data['val_data']

            # Tahminleri hesapla
            predictions = []
            ground_truth = []

            for item in val_data:
                # Model tahmini yap (bu fonksiyon implement edilmeli)
                # input_data = prepare_input_for_prediction(item['frame_path'])
                # pred = model.predict(input_data)
                pred = {'violence': 0.5, 'adult_content': 0.5, 'harassment': 0.5, 'weapon': 0.5, 'drug': 0.5}

                # Tahmin ve gerçek değerleri topla
                for category in ['violence', 'adult_content', 'harassment', 'weapon', 'drug']:
                    predictions.append(pred[category])
                    ground_truth.append(item['category_scores'][category])

            # Metrikleri hesapla
            metrics = {
                'accuracy': calculate_accuracy(predictions, ground_truth, threshold=0.5),
                'precision': calculate_precision(predictions, ground_truth, threshold=0.5),
                'recall': calculate_recall(predictions, ground_truth, threshold=0.5),
                'f1': calculate_f1(predictions, ground_truth, threshold=0.5)
            }

            # Kategori bazlı metrikler
            category_metrics = {}
            for i, category in enumerate(['violence', 'adult_content', 'harassment', 'weapon', 'drug']):
                cat_preds = [predictions[j] for j in range(len(predictions)) if j % 5 == i]
                cat_truth = [ground_truth[j] for j in range(len(ground_truth)) if j % 5 == i]

                category_metrics[category] = {
                    'accuracy': calculate_accuracy(cat_preds, cat_truth, threshold=0.5),
                    'precision': calculate_precision(cat_preds, cat_truth, threshold=0.5),
                    'recall': calculate_recall(cat_preds, cat_truth, threshold=0.5),
                    'f1': calculate_f1(cat_preds, cat_truth, threshold=0.5)
                }

            metrics['category_metrics'] = category_metrics

        else:  # age
            # Yaş tahmin metriklerini hesapla
            val_data = training_data['val_data']

            # Tahminleri hesapla
            predictions = []
            ground_truth = []

            for item in val_data:
                # Model tahmini yap (bu fonksiyon implement edilmeli)
                # input_data = prepare_input_for_prediction(item['face_image_path'])
                # pred = model.predict_age(input_data)
                pred = 25.0  # Placeholder value

                # Tahmin ve gerçek değerleri topla
                predictions.append(pred)
                ground_truth.append(item['corrected_age'])

            # MAE (Mean Absolute Error) hesapla
            mae = sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(predictions)

            # ±3 yaş doğruluğunu hesapla
            accuracy_3years = sum(1 for p, g in zip(predictions, ground_truth) if abs(p - g) <= 3) / len(predictions)

            metrics = {
                'mae': mae,
                'accuracy': accuracy_3years,
                'count': len(predictions)
            }

            # Yaş dağılımı
            age_distribution = {}
            age_ranges = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

            for age in ground_truth:
                range_idx = min(age // 10, 7)  # 70+ için 7. indeks
                age_range = age_ranges[range_idx]
                age_distribution[age_range] = age_distribution.get(age_range, 0) + 1

            metrics['age_distribution'] = age_distribution

                                # Hata dağılımı
            error_distribution = {}

            for p, g in zip(predictions, ground_truth):
                error = abs(p - g)

                if error <= 1:
                    error_range = '0-1'
                elif error <= 3:
                    error_range = '2-3'
                elif error <= 5:
                    error_range = '4-5'
                elif error <= 10:
                    error_range = '6-10'
                else:
                    error_range = '10+'

                error_distribution[error_range] = error_distribution.get(error_range, 0) + 1

            metrics['error_distribution'] = error_distribution

    except Exception as e:
        logging.error(f"Metrik hesaplama hatası: {str(e)}")
        metrics = {'error': str(e)}

    return metrics

# Yardımcı metrik hesaplama fonksiyonları
def calculate_accuracy(predictions, ground_truth, threshold=0.5):
    """İkili sınıflandırma için doğruluk hesaplar"""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if (p >= threshold) == (g >= threshold))
    return correct / len(predictions) if predictions else 0

def calculate_precision(predictions, ground_truth, threshold=0.5):
    """Kesinlik (doğru pozitiflerin tüm pozitiflere oranı)"""
    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p >= threshold and g >= threshold)
    predicted_positives = sum(1 for p in predictions if p >= threshold)
    return true_positives / predicted_positives if predicted_positives else 0

def calculate_recall(predictions, ground_truth, threshold=0.5):
    """Duyarlılık (doğru pozitiflerin tüm gerçek pozitiflere oranı)"""
    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p >= threshold and g >= threshold)
    actual_positives = sum(1 for g in ground_truth if g >= threshold)
    return true_positives / actual_positives if actual_positives else 0

def calculate_f1(predictions, ground_truth, threshold=0.5):
    """F1 skoru (kesinlik ve duyarlılığın harmonik ortalaması)"""
    precision = calculate_precision(predictions, ground_truth, threshold)
    recall = calculate_recall(predictions, ground_truth, threshold)

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)

def get_model_version(model_name):
    """
    Belirtilen modelin version bilgisini döndürür

    Args:
        model_name: Version bilgisi alınacak model adı

    Returns:
        str: Model version bilgisi
    """
    model_versions = {
        'violence_detection': 'CLIP-integrated-v1.0',
        'harassment_detection': 'CLIP-integrated-v1.0',
        'adult_content_detection': 'CLIP-integrated-v1.0',
        'weapon_detection': 'CLIP-integrated-v1.0',
        'substance_detection': 'CLIP-integrated-v1.0',
        'detection': 'YOLOv8n-v1.0',
        'clip': 'ViT-L/14@336px'
    }

    return model_versions.get(model_name, 'unknown')

def delete_latest_version(model_type):
    """
    Belirtilen model tipinin en son versiyonunu siler.

    Args:
        model_type: Silinecek model tipi ('age' veya 'content')

    Returns:
        dict: İşlem sonucu - success (bool) ve message (str) içerir
    """
    try:
        # En son versiyonu bul
        latest_version = db.session.query(ModelVersion).filter_by(
            model_type=model_type
        ).order_by(ModelVersion.version.desc()).first()

        if not latest_version:
            return {
                "success": False,
                "message": f"{model_type} tipinde hiç model versiyonu bulunamadı."
            }

        # Base model (v0) silinmemeli
        if latest_version.version == 0:
            return {
                "success": False,
                "message": "Base model (v0) silinemez!"
            }

        # Aktif model silinmemeli
        if latest_version.is_active:
            return {
                "success": False,
                "message": f"Aktif model versiyonu (v{latest_version.version}) silinemez! Önce başka bir versiyonu aktif yapın."
            }

        logger.info(f"Silinecek versiyon: {model_type} v{latest_version.version} ({latest_version.version_name})")

        # Dosya sisteminden sil
        if latest_version.file_path and os.path.exists(latest_version.file_path):
            logger.info(f"Dosya sisteminden siliniyor: {latest_version.file_path}")
            shutil.rmtree(latest_version.file_path)
            logger.info("Dosyalar silindi")
        else:
            logger.warning(f"Dosya yolu bulunamadı veya zaten silinmiş: {latest_version.file_path}")

        # Veritabanından sil
        version_info = {
            "version": latest_version.version,
            "version_name": latest_version.version_name,
            "created_at": latest_version.created_at.isoformat() if latest_version.created_at else None
        }

        db.session.delete(latest_version)
        db.session.commit()
        logger.info("Veritabanı kaydı silindi")

        return {
            "success": True,
            "message": f"Model versiyonu v{version_info['version']} ({version_info['version_name']}) başarıyla silindi.",
            "deleted_version": version_info
        }

    except Exception as e:
        db.session.rollback()
        logger.error(f"Model versiyonu silme hatası: {str(e)}")
        return {
            "success": False,
            "message": f"Silme işlemi sırasında hata oluştu: {str(e)}"
        }

def get_model_dashboard_stats(model_type):
    """
    Model dashboard için istatistikleri getirir

    Args:
        model_type: 'age' veya 'content'

    Returns:
        dict: Model dashboard istatistikleri
    """
    try:
        if model_type == 'age':
            return _get_age_model_stats()
        elif model_type == 'content':
            return _get_content_model_stats()
        else:
            logger.error(f"Geçersiz model tipi: {model_type}")
            return None

    except Exception as e:
        logger.error(f"Model dashboard istatistikleri alınırken hata: {str(e)}")
        return None

def load_specific_model_by_version_id(model_type, version_id):
    """
    Belirli versiyon ID'si ile modeli yükler ve aktif yapar

    Args:
        model_type: Model tipi ('age' veya 'content')
        version_id: Aktif yapılacak versiyon ID'si

    Returns:
        tuple: (success, message)
    """
    try:
        # Versiyonu bul
        version = db.session.query(ModelVersion).filter_by(id=version_id).first()

        if not version:
            return False, "Belirtilen model versiyonu bulunamadı"

        if version.model_type != model_type:
            return False, "Model tipi uyuşmuyor"

        # Activate model version fonksiyonunu kullan
        result = activate_model_version(version_id)

        if result['success']:
            return True, f"{model_type} modeli v{result['version']} başarıyla aktifleştirildi"
        else:
            return False, result.get('message', 'Bilinmeyen hata')

    except Exception as e:
        logger.error(f"Model versiyon yükleme hatası: {str(e)}")
        return False, f"Model yükleme hatası: {str(e)}"
