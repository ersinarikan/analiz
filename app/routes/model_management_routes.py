from flask import Blueprint, jsonify, request, current_app
from app.services import model_service
import logging
import os
from app.models import Feedback
from app.models.content import ModelVersion
from app import db
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)
model_management_bp = Blueprint('model_management_bp', __name__, url_prefix='/api/models')

@model_management_bp.route('/stats/<string:model_type>', methods=['GET'])
def get_model_stats_route(model_type):
    """ Belirtilen model tipi için istatistikleri döndürür. """
    if model_type not in ['content', 'age']:
        return jsonify({"error": "Invalid model type"}), 400
    try:
        stats = model_service.get_model_dashboard_stats(model_type)
        if stats is None:
            return jsonify({"error": f"{model_type} için istatistikler bulunamadı."}), 404
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"/{model_type}/stats endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"İstatistikler alınırken sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/versions/<string:model_type>', methods=['GET'])
def get_model_versions_route(model_type):
    """ Belirtilen model tipi için versiyonları döndürür. """
    if model_type not in ['content', 'age']:
        return jsonify({"error": "Invalid model type"}), 400
    try:
        versions_data = model_service.get_model_versions(model_type)
        if versions_data.get('success', False):
            return jsonify(versions_data), 200
        else:
            return jsonify({"error": versions_data.get('error', 'Bilinmeyen hata')}), 500
    except Exception as e:
        logger.error(f"/{model_type}/versions endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"Versiyonlar alınırken sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/metrics/<string:model_type>', methods=['GET'])
def get_model_metrics_route(model_type):
    """ Belirtilen model tipi için metrikleri döndürür. """
    if model_type not in ['content', 'age']:
        return jsonify({"error": "Invalid model type"}), 400
    try:
        stats = model_service.get_model_dashboard_stats(model_type)
        if stats is None:
            return jsonify({"error": f"{model_type} için metrikler bulunamadı."}), 404
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"/{model_type}/metrics endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"Metrikler alınırken sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/activate/<int:version_id>', methods=['POST'])
def activate_model_version_route(version_id):
    """ Belirtilen versiyon ID'sine sahip modeli aktifleştirir. """
    try:
        result = model_service.activate_model_version(version_id)
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify({"error": result.get('message', 'Bilinmeyen hata')}), 500
    except Exception as e:
        logger.error(f"/activate/{version_id} endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"Model aktivasyonunda sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/train/<string:model_type>', methods=['POST'])
def train_model_route(model_type):
    """ Belirtilen model tipini geri bildirimlerle eğitir. """
    if model_type not in ['content', 'age']:
        return jsonify({"error": "Invalid model type"}), 400
    try:
        # Eğitim parametreleri request body'den alınabilir (örn: epoch sayısı vb.)
        # Şimdilik varsayılan parametrelerle çalışacak
        # training_params = request.json if request.is_json else {}
        
        current_app.logger.info(f"{model_type} modeli için eğitim süreci başlatılıyor (manuel tetikleme).")
        success, message_or_metrics = model_service.train_with_feedback(model_type) # training_params buraya eklenebilir
        
        if success:
            return jsonify({"message": f"{model_type} modeli için eğitim başarıyla tamamlandı/başlatıldı.", "details": message_or_metrics}), 200
        else:
            return jsonify({"error": f"{model_type} modeli eğitimi başarısız oldu veya başlatılamadı.", "details": message_or_metrics}), 500
    except Exception as e:
        logger.error(f"/{model_type}/train endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"Eğitim sırasında sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/load_version/<string:model_type>/<int:version_id>', methods=['POST'])
def load_model_version_route(model_type, version_id):
    """ Belirtilen model tipinin belirli bir versiyonunu yükler. """
    if model_type not in ['content', 'age']:
        return jsonify({"error": "Invalid model type"}), 400
    try:
        success, message = model_service.load_specific_model_by_version_id(model_type, version_id)
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 404 # veya 500 duruma göre
    except Exception as e:
        logger.error(f"/{model_type}/load_version/{version_id} endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"Model versiyonu yüklenirken sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/reset/<string:model_type>', methods=['POST'])
def reset_model_route(model_type):
    """ Belirtilen model tipini önceden eğitilmiş (pretrained) haline sıfırlar. """
    if model_type not in ['content', 'age']:
        return jsonify({"error": "Invalid model type"}), 400
    try:
        success, message = model_service.reset_model(model_type)
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        logger.error(f"/{model_type}/reset endpoint hatası: {str(e)}", exc_info=True)
        return jsonify({"error": f"Model sıfırlanırken sunucu hatası: {str(e)}"}), 500

@model_management_bp.route('/reload-content-analyzer', methods=['POST'])
def reload_content_analyzer():
    """ContentAnalyzer'ı yeniden yükler"""
    try:
        from app.ai.content_analyzer import ContentAnalyzer
        ContentAnalyzer.reset_instance()
        
        # Yeni instance oluştur
        analyzer = ContentAnalyzer()
        if analyzer.initialized:
            return jsonify({"message": "ContentAnalyzer başarıyla yeniden yüklendi"}), 200
        else:
            return jsonify({"error": "ContentAnalyzer yeniden yükleme başarısız"}), 500
    except Exception as e:
        logger.error(f"ContentAnalyzer yeniden yükleme hatası: {str(e)}")
        return jsonify({"error": f"Yeniden yükleme hatası: {str(e)}"}), 500 

@model_management_bp.route('/api/model-management/cleanup', methods=['POST'])
def cleanup_system():
    """Sistem temizliği endpoint'i"""
    try:
        data = request.get_json() or {}
        
        # Temizlik konfigürasyonu
        cleanup_config = {
            'age_model_versions': data.get('age_model_versions', 5),
            'content_model_versions': data.get('content_model_versions', 5),
            'age_feedback_records': data.get('age_feedback_records', 100),
            'content_feedback_records': data.get('content_feedback_records', 100),
            'ensemble_age_versions': data.get('ensemble_age_versions', 3),
            'ensemble_content_versions': data.get('ensemble_content_versions', 3),
            'unused_frames_days': data.get('unused_frames_days', 30),
            'vacuum_database': data.get('vacuum_database', True)
        }
        
        # Kapsamlı temizlik gerçekleştir
        model_service = ModelService()
        result = model_service.comprehensive_cleanup(cleanup_config)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Sistem temizliği hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@model_management_bp.route('/api/model-management/cleanup/feedback', methods=['POST'])
def cleanup_feedback():
    """Feedback kayıtlarını temizle"""
    try:
        data = request.get_json() or {}
        model_type = data.get('model_type', 'age')  # 'age' veya 'content'
        keep_count = data.get('keep_count', 100)
        
        if model_type not in ['age', 'content']:
            return jsonify({
                "success": False,
                "error": "Model türü 'age' veya 'content' olmalıdır"
            }), 400
        
        model_service = ModelService()
        result = model_service.cleanup_ensemble_feedback_records(model_type, keep_count)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Feedback temizliği hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@model_management_bp.route('/api/model-management/cleanup/ensemble-files', methods=['POST'])
def cleanup_ensemble_files():
    """Ensemble model dosyalarını temizle"""
    try:
        data = request.get_json() or {}
        model_type = data.get('model_type', 'age')  # 'age' veya 'content'
        keep_count = data.get('keep_count', 3)
        
        if model_type not in ['age', 'content']:
            return jsonify({
                "success": False,
                "error": "Model türü 'age' veya 'content' olmalıdır"
            }), 400
        
        model_service = ModelService()
        result = model_service.cleanup_ensemble_model_files(model_type, keep_count)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Ensemble dosya temizliği hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@model_management_bp.route('/api/model-management/cleanup/unused-frames', methods=['POST'])
def cleanup_unused_frames():
    """Kullanılmayan frame'leri temizle"""
    try:
        data = request.get_json() or {}
        days_old = data.get('days_old', 30)
        
        model_service = ModelService()
        result = model_service.cleanup_unused_analysis_frames(days_old)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Kullanılmayan frame temizliği hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@model_management_bp.route('/api/model-management/database/vacuum', methods=['POST'])
def vacuum_database():
    """Veritabanını optimize et"""
    try:
        model_service = ModelService()
        success = model_service.vacuum_database()
        
        if success:
            return jsonify({
                "success": True,
                "message": "Veritabanı başarıyla optimize edildi",
                "database_size": model_service.get_database_size()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Veritabanı optimize edilemedi"
            }), 500
        
    except Exception as e:
        logger.error(f"Veritabanı optimize hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@model_management_bp.route('/api/model-management/database/size', methods=['GET'])
def get_database_size():
    """Veritabanı boyutunu getir"""
    try:
        model_service = ModelService()
        size_mb = model_service.get_database_size()
        
        return jsonify({
            "success": True,
            "size_mb": size_mb
        })
        
    except Exception as e:
        logger.error(f"Veritabanı boyutu alma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@model_management_bp.route('/api/model-management/cleanup/status', methods=['GET'])
def get_cleanup_status():
    """Temizlik durumunu getir"""
    try:
        model_service = ModelService()
        
        # Ensemble feedback istatistikleri
        age_feedback_count = db.session.query(Feedback).filter(
            Feedback.used_in_ensemble == True,
            Feedback.feedback_type == 'age'
        ).count()
        
        content_feedback_count = db.session.query(Feedback).filter(
            Feedback.used_in_ensemble == True,
            Feedback.feedback_type == 'content'
        ).count()
        
        # Model versiyon sayıları
        age_versions = db.session.query(ModelVersion).filter(
            ModelVersion.model_type == 'age'
        ).count()
        
        content_versions = db.session.query(ModelVersion).filter(
            ModelVersion.model_type == 'content'
        ).count()
        
        # Ensemble klasör boyutları
        ensemble_age_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'age',
            'ensemble_versions'
        )
        
        ensemble_content_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'content',
            'ensemble_versions'
        )
        
        def get_dir_size(path):
            if not os.path.exists(path):
                return 0
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # MB
        
        return jsonify({
            "success": True,
            "status": {
                "database_size_mb": model_service.get_database_size(),
                "feedback_records": {
                    "age": age_feedback_count,
                    "content": content_feedback_count
                },
                "model_versions": {
                    "age": age_versions,
                    "content": content_versions
                },
                "ensemble_storage_mb": {
                    "age": round(get_dir_size(ensemble_age_dir), 2),
                    "content": round(get_dir_size(ensemble_content_dir), 2)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Temizlik durumu alma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500 