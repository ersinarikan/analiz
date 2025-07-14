from flask import Blueprint, jsonify, request, current_app
from app.services import model_service
import logging

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