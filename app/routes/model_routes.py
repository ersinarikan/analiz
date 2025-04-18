from flask import Blueprint, request, jsonify, current_app
from app.tasks.training_tasks import train_model_task
from app.services.model_service import get_model_stats, get_available_models, reset_model
from app.services.model_service import prepare_training_data
import logging
import threading
import uuid
from app.services import model_service
from app.models.content import ModelVersion
from app import db

bp = Blueprint('model', __name__, url_prefix='/api/model')
logger = logging.getLogger(__name__)

@bp.route('/metrics/<model_type>', methods=['GET'])
def get_metrics(model_type):
    """
    Model metrikleri API endpoint'i.
    
    Args:
        model_type: Metrik bilgileri alınacak modelin tipi ('content', 'age' veya 'all')
        
    Returns:
        JSON: Model metrikleri
    """
    try:
        metrics = get_model_stats(model_type)
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Model metrikleri alınırken hata: {str(e)}")
        return jsonify({'error': f'Model metrikleri alınırken bir hata oluştu: {str(e)}'}), 500

@bp.route('/available', methods=['GET'])
def available_models():
    """
    Kullanılabilir modelleri listeler.
    
    Returns:
        JSON: Kullanılabilir modeller listesi
    """
    try:
        models = get_available_models()
        return jsonify(models), 200
    except Exception as e:
        logger.error(f"Kullanılabilir modeller listelenirken hata: {str(e)}")
        return jsonify({'error': f'Kullanılabilir modeller listelenirken bir hata oluştu: {str(e)}'}), 500

@bp.route('/reset', methods=['POST'])
def reset_model_endpoint():
    """
    Modeli ön eğitimli (pretrained) haline sıfırlar.
    
    Request Body:
        - model_type: Sıfırlanacak modelin tipi ('content' veya 'age')
        
    Returns:
        JSON: İşlem sonucu
    """
    try:
        data = request.json
        
        if not data or 'model_type' not in data:
            return jsonify({'error': 'model_type alanı gereklidir'}), 400
            
        model_type = data['model_type']
        
        if model_type not in ['content', 'age']:
            return jsonify({'error': 'Geçersiz model tipi. Desteklenen tipler: content, age'}), 400
            
        success, message = reset_model(model_type)
        
        if success:
            return jsonify({'message': message}), 200
        else:
            return jsonify({'error': message}), 500
            
    except Exception as e:
        logger.error(f"Model sıfırlama hatası: {str(e)}")
        return jsonify({'error': f'Model sıfırlanırken bir hata oluştu: {str(e)}'}), 500

@bp.route('/train', methods=['POST'])
def train_model():
    """
    Model eğitimi başlatır.
    
    Request Body:
        - model_type: Eğitilecek modelin tipi ('content' veya 'age')
        - epochs: Eğitim turlarının sayısı
        - batch_size: Batch boyutu
        - learning_rate: Öğrenme oranı
        
    Returns:
        JSON: Eğitim ID'si ve durum
    """
    try:
        data = request.json
        
        if not data or 'model_type' not in data:
            return jsonify({'error': 'model_type alanı gereklidir'}), 400
            
        model_type = data['model_type']
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 16)
        learning_rate = data.get('learning_rate', 0.001)
        
        if model_type not in ['content', 'age']:
            return jsonify({'error': 'Geçersiz model tipi. Desteklenen tipler: content, age'}), 400
            
        # Eğitim verisini hazırla
        training_data, message = prepare_training_data(model_type)
        
        if not training_data:
            return jsonify({'error': f'Eğitim verisi hazırlanamadı: {message}'}), 400
            
        # Benzersiz model ID oluştur
        model_id = str(uuid.uuid4())
        dataset_path = f"training_data_{model_type}"
        parameters = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        # Eğitimi ayrı bir thread'de başlat
        threading.Thread(
            target=train_model_task,
            args=(model_id, dataset_path),
            kwargs={'parameters': parameters}
        ).start()
        
        return jsonify({
            'message': f'Eğitim başlatıldı. {len(training_data)} örnek ile eğitim yapılacak.',
            'model_id': model_id,
            'status': 'pending'
        }), 200
            
    except Exception as e:
        logger.error(f"Model eğitim hatası: {str(e)}")
        return jsonify({'error': f'Model eğitimi başlatılırken bir hata oluştu: {str(e)}'}), 500

@bp.route('/training-status/<training_id>', methods=['GET'])
def training_status(training_id):
    """
    Eğitim durumunu kontrol eder.
    
    Args:
        training_id: Eğitim ID'si
        
    Returns:
        JSON: Eğitim durumu
    """
    try:
        # Simülasyon - Gerçek bir uygulamada veritabanından veya Redis'ten durumu alır
        # Gerçek durum kontrolü kodu burada olacak
        
        # Örnek için sabit bir ilerleme değeri
        progress = 75
        
        return jsonify({
            'training_id': training_id,
            'status': 'in_progress',
            'progress': progress,
            'status_message': f'Eğitim devam ediyor... (Epoch 3/4)'
        }), 200
            
    except Exception as e:
        logger.error(f"Eğitim durumu kontrolü hatası: {str(e)}")
        return jsonify({'error': f'Eğitim durumu kontrol edilirken bir hata oluştu: {str(e)}'}), 500

@bp.route('/versions/<model_type>', methods=['GET'])
def get_model_versions(model_type):
    """
    Belirli bir model tipi için tüm versiyonları döndürür
    """
    if model_type not in ['content', 'age']:
        return jsonify({'error': 'Geçersiz model tipi'}), 400
    
    versions = model_service.get_model_versions(model_type)
    
    # Verileri JSON formatına dönüştür
    versions_data = []
    for version in versions:
        versions_data.append({
            'id': version.id,
            'model_type': version.model_type,
            'version': version.version,
            'created_at': version.created_at.isoformat(),
            'is_active': version.is_active,
            'metrics': version.metrics,
            'training_samples': version.training_samples,
            'validation_samples': version.validation_samples,
            'epochs': version.epochs
        })
    
    return jsonify({
        'success': True,
        'model_type': model_type,
        'versions': versions_data
    })

@bp.route('/activate/<int:version_id>', methods=['POST'])
def activate_model_version(version_id):
    """
    Belirli bir model versiyonunu aktif hale getirir
    """
    result = model_service.activate_model_version(version_id)
    return jsonify(result)

@bp.route('/reset', methods=['POST'])
def reset_model():
    """
    Modeli sıfırlar (ön eğitimli orijinal modele geri döner)
    """
    data = request.json
    model_type = data.get('model_type')
    
    if model_type not in ['content', 'age']:
        return jsonify({'error': 'Geçersiz model tipi'}), 400
    
    result = model_service.reset_model(model_type)
    return jsonify(result)

@bp.route('/train-with-feedback', methods=['POST'])
def train_with_feedback():
    """
    Geri bildirim verilerini kullanarak modeli yeniden eğitir
    """
    data = request.json
    model_type = data.get('model_type')
    
    if model_type not in ['content', 'age']:
        return jsonify({'error': 'Geçersiz model tipi'}), 400
    
    # Eğitim parametreleri
    params = {
        'epochs': data.get('epochs', 10),
        'batch_size': data.get('batch_size', 32),
        'learning_rate': data.get('learning_rate', 0.001)
    }
    
    # Arka planda eğitim başlat
    # Not: Gerçek uygulamada bu işlem Celery ile asenkron çalıştırılmalı
    result = model_service.train_with_feedback(model_type, params)
    
    return jsonify(result) 