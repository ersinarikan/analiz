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
            return jsonify({'success': True, 'message': message}), 200
        else:
            return jsonify({'success': False, 'error': message}), 500
            
    except Exception as e:
        logger.error(f"Model sıfırlama hatası: {str(e)}")
        return jsonify({'success': False, 'error': f'Model sıfırlanırken bir hata oluştu: {str(e)}'}), 500

@bp.route('/reset/<model_type>', methods=['POST'])
def reset_model_by_type(model_type):
    """
    Belirtilen model tipini sıfırlar (URL parametreli versiyon)
    
    Args:
        model_type: Sıfırlanacak modelin tipi ('content' veya 'age')
        
    Returns:
        JSON: İşlem sonucu
    """
    try:
        if model_type not in ['content', 'age']:
            return jsonify({'success': False, 'error': 'Geçersiz model tipi. Desteklenen tipler: content, age'}), 400
            
        success, message = reset_model(model_type)
        
        if success:
            response_data = {
                'success': True, 
                'message': message
            }
            
            # Yaş modeli sıfırlandığında sistem yeniden başlatılmalı
            if model_type == 'age':
                logger.info(f"{model_type} modeli sıfırlandı")
                
                # Model state dosyasını güncelle
                from app.services.model_service import update_model_state_file
                update_model_state_file(model_type, 0)  # 0 = base model
                
                response_data['restart_required'] = True
                response = jsonify(response_data)
                
                return response, 200
            else:
                return jsonify(response_data), 200
        else:
            return jsonify({'success': False, 'error': message}), 500
            
    except Exception as e:
        logger.error(f"Model sıfırlama hatası: {str(e)}")
        return jsonify({'success': False, 'error': f'Model sıfırlanırken bir hata oluştu: {str(e)}'}), 500

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
    
    # Frontend'in beklediği formata uygun yanıt döndür
    if 'success' in result:
        if result['success']:
            # Model başarıyla aktifleştirildi
            logger.info(f"Model versiyonu {version_id} aktifleştirildi")
            
            # model_state.py güncellendiği için Flask otomatik restart yapacak
            response = jsonify({
                **result,
                'message': 'Model başarıyla aktifleştirildi. Sistem otomatik olarak yeniden başlatılacak...',
                'restart_required': True
            })
            
            return response, 200
        else:
            return jsonify(result), 400
    else:
        # Eğer result dictionary'de success key'i yoksa ekle
        return jsonify({'success': True, **result}), 200

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

@bp.route('/age/versions', methods=['GET'])
def get_age_model_versions():
    """
    Custom Age model versiyonlarını listeler
    """
    try:
        from app.services.age_training_service import AgeTrainingService
        trainer = AgeTrainingService()
        versions = trainer.get_model_versions()
        
        return jsonify({
            'success': True,
            'versions': versions
        }), 200
        
    except Exception as e:
        logger.error(f"Age model versiyonları alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/age/activate/<int:version_id>', methods=['POST'])
def activate_age_model_version(version_id):
    """
    Belirli bir Custom Age model versiyonunu aktif hale getirir
    """
    try:
        from app.services.age_training_service import AgeTrainingService
        trainer = AgeTrainingService()
        
        success = trainer.activate_model_version(version_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model version {version_id} activated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to activate model version'
            }), 400
            
    except Exception as e:
        logger.error(f"Model versiyonu aktifleştirme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/age/training-data-stats', methods=['GET'])
def get_age_training_data_stats():
    """
    Mevcut yaş eğitim verisi istatistiklerini döndürür
    """
    try:
        from app.services.age_training_service import AgeTrainingService
        trainer = AgeTrainingService()
        
        # Veriyi hazırla ama eğitme
        training_data = trainer.prepare_training_data(min_samples=1)
        
        if training_data is None:
            return jsonify({
                'success': True,
                'stats': {
                    'total_samples': 0,
                    'manual_samples': 0,
                    'pseudo_samples': 0,
                    'age_range': None,
                    'mean_age': None
                }
            }), 200
        
        manual_count = training_data['sources'].count('manual')
        pseudo_count = training_data['sources'].count('pseudo')
        ages = training_data['ages']
        
        return jsonify({
            'success': True,
            'stats': {
                'total_samples': len(training_data['embeddings']),
                'manual_samples': manual_count,
                'pseudo_samples': pseudo_count,
                'age_range': {
                    'min': float(ages.min()),
                    'max': float(ages.max())
                },
                'mean_age': float(ages.mean()),
                'std_age': float(ages.std())
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Eğitim verisi istatistikleri alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/delete-latest/<model_type>', methods=['DELETE'])
def delete_latest_model_version(model_type):
    """
    Belirtilen model tipinin en son versiyonunu siler.
    
    Args:
        model_type: Silinecek model tipi ('age' veya 'content')
        
    Returns:
        JSON: İşlem sonucu
    """
    try:
        if model_type not in ['content', 'age']:
            return jsonify({'error': 'Geçersiz model tipi. Desteklenen tipler: content, age'}), 400
        
        result = model_service.delete_latest_version(model_type)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Model versiyonu silme hatası: {str(e)}")
        return jsonify({'error': f'Model versiyonu silinirken bir hata oluştu: {str(e)}'}), 500 