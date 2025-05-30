from flask import Blueprint, request, jsonify, current_app
from app.tasks.training_tasks import train_model_task
from app.services.model_service import get_model_stats, get_available_models, reset_model
from app.services.model_service import prepare_training_data
import logging
import threading
import uuid
from app.services import model_service
from app.models.content import ModelVersion
from app.models.feedback import Feedback
from app import db, socketio

bp = Blueprint('model', __name__, url_prefix='/api/model')
# Root logger'ı kullan (terminalde görünmesi için)
logger = logging.getLogger('app.model_routes')

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
    
    result = model_service.get_model_versions(model_type)
    
    # model_service.get_model_versions zaten bir dictionary döndürüyor
    if result.get('success', False):
        return jsonify(result)
    else:
        return jsonify({'error': result.get('error', 'Bilinmeyen hata')}), 500

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

@bp.route('/train-web', methods=['POST'])
def train_model_web():
    """
    Web arayüzünden model eğitimi başlatır (her iki model türü için)
    
    Request body:
    {
        "model_type": "content" | "age",
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
        "patience": 5,
        ... diğer parametreler
    }
    """
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'content')
        
        # Parametreleri hazırla
        params = {
            'epochs': data.get('epochs', 20),
            'batch_size': data.get('batch_size', 16),
            'learning_rate': data.get('learning_rate', 0.001),
            'patience': data.get('patience', 5),
            'min_samples': data.get('min_samples', 50)
        }
        
        if model_type == 'content':
            # Content model training
            from app.services.content_training_service import ContentTrainingService
            
            trainer = ContentTrainingService()
            
            # Veriyi hazırla
            training_data = trainer.prepare_training_data(
                min_samples=params['min_samples'],
                validation_strategy=params.get('validation_strategy', 'strict')
            )
            
            if training_data is None:
                return jsonify({
                    'success': False,
                    'error': f'Yeterli eğitim verisi bulunamadı. En az {params["min_samples"]} örnek gerekli.'
                }), 400
            
            # WebSocket ile progress tracking için session ID oluştur
            training_session_id = str(uuid.uuid4())
            
            # Background task olarak eğitimi başlat
            from threading import Thread
            training_thread = Thread(
                target=_run_content_training,
                args=(trainer, training_data, params, training_session_id, current_app._get_current_object())
            )
            training_thread.daemon = True
            training_thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Content model eğitimi başlatıldı',
                'session_id': training_session_id,
                'training_samples': training_data['total_samples'],
                'estimated_duration': _estimate_training_duration(training_data['total_samples'], params['epochs'])
            })
            
        elif model_type == 'age':
            # Age model training - Artık destekleniyor!
            from app.services.age_training_service import AgeTrainingService
            
            trainer = AgeTrainingService()
            
            # Veriyi hazırla
            training_data = trainer.prepare_training_data(min_samples=10)
            
            if training_data is None:
                return jsonify({
                    'success': False,
                    'error': 'Yeterli yaş eğitim verisi bulunamadı. En az 10 geri bildirim gerekli.'
                }), 400
            
            # WebSocket ile progress tracking için session ID oluştur
            training_session_id = str(uuid.uuid4())
            
            # Background task olarak eğitimi başlat
            from threading import Thread
            training_thread = Thread(
                target=_run_age_training,
                args=(trainer, training_data, params, training_session_id, current_app._get_current_object())
            )
            training_thread.daemon = True
            training_thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Yaş modeli eğitimi başlatıldı',
                'session_id': training_session_id,
                'training_samples': len(training_data['embeddings']),
                'estimated_duration': _estimate_training_duration(len(training_data['embeddings']), params['epochs'])
            })
        
        else:
            return jsonify({
                'success': False,
                'error': f'Desteklenmeyen model türü: {model_type}'
            }), 400
            
    except Exception as e:
        logger.error(f"Web eğitimi başlatma hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Eğitim başlatılamadı: {str(e)}'
        }), 500

def _run_content_training(trainer, training_data, params, session_id, app):
    """
    Background thread'de content model eğitimi çalıştırır
    """
    # Flask app context'ini background thread'e taşı
    with app.app_context():
        try:
            # WebSocket ile progress gönder
            socketio.emit('training_started', {
                'session_id': session_id,
                'model_type': 'content',
                'total_samples': training_data['total_samples']
            })
            
            # Session ID'yi params'a ekle
            params['session_id'] = session_id
            
            # Eğitimi başlat
            training_result = trainer.train_model(training_data, params)
            
            # Model versiyonunu kaydet
            model_version = trainer.save_model_version(
                training_result['model'], 
                training_result
            )
            
            # Başarı mesajı
            socketio.emit('training_completed', {
                'session_id': session_id,
                'success': True,
                'model_version': model_version.version_name,
                'metrics': training_result['metrics'],
                'training_samples': training_result['training_samples'],
                'validation_samples': training_result['validation_samples'],
                'conflicts_resolved': training_result['conflicts_detected']
            })
            
        except Exception as e:
            # Hata mesajı
            socketio.emit('training_error', {
                'session_id': session_id,
                'error': str(e)
            })
            logger.error(f"Training thread error: {str(e)}")

def _run_age_training(trainer, training_data, params, session_id, app):
    """
    Yaş modeli eğitimini arka planda çalıştırır
    
    Args:
        trainer: AgeTrainingService instance
        training_data: Hazırlanmış eğitim verisi  
        params: Eğitim parametreleri
        session_id: WebSocket session ID
        app: Flask app instance
    """
    # Flask app context'ini background thread'e taşı
    with app.app_context():
        try:
            logger.info(f"Yaş modeli eğitimi başlatıldı: session_id={session_id}")
            
            # WebSocket ile eğitim başlatıldığını bildir
            socketio.emit('training_started', {
                'session_id': session_id,
                'model_type': 'age',
                'total_samples': len(training_data['embeddings'])
            })
            
            # Eğitim parametrelerini hazırla
            training_params = {
                'epochs': params.get('epochs', 50),
                'batch_size': params.get('batch_size', 32),
                'learning_rate': params.get('learning_rate', 0.001),
                'hidden_dims': params.get('hidden_dims', [256, 128]),
                'test_size': 0.2,
                'early_stopping_patience': params.get('patience', 10)
            }
            
            logger.info(f"Eğitim parametreleri: {training_params}")
            
            # İlerleme callback fonksiyonu
            def progress_callback(epoch, total_epochs, metrics=None):
                progress_percent = (epoch / total_epochs) * 100
                socketio.emit('training_progress', {
                    'session_id': session_id,
                    'progress': progress_percent,
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'metrics': metrics
                })
                logger.info(f"Eğitim ilerlemesi: Epoch {epoch}/{total_epochs} ({progress_percent:.1f}%)")
            
            # Parametrelere callback ekle
            training_params['progress_callback'] = progress_callback
            
            # Modeli eğit
            logger.info("Model eğitimi başlatılıyor...")
            result = trainer.train_model(training_data, training_params)
            
            # Model versiyonunu kaydet
            logger.info("Model versiyonu kaydediliyor...")
            model_version = trainer.save_model_version(result['model'], result)
            
            # Başarı durumunda WebSocket event gönder
            final_metrics = {
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'], 
                'within_3_years': result['metrics']['within_3_years'],
                'within_5_years': result['metrics']['within_5_years'],
                'training_samples': result['training_samples'],
                'validation_samples': result['validation_samples']
            }
            
            socketio.emit('training_completed', {
                'session_id': session_id,
                'success': True,
                'model_type': 'age',
                'model_version': model_version.version_name,
                'metrics': final_metrics
            })
            
            logger.info(f"Yaş modeli eğitimi tamamlandı: {model_version.version_name}")
            
        except Exception as e:
            logger.error(f"Yaş modeli eğitimi hatası: {str(e)}", exc_info=True)
            
            # Hata durumunda WebSocket event gönder
            socketio.emit('training_error', {
                'session_id': session_id,
                'error': str(e),
                'model_type': 'age'
            })
            
            # Re-raise the exception for logging
            raise

def _estimate_training_duration(samples, epochs):
    """
    Eğitim süresini tahmin eder
    
    Args:
        samples: Eğitim örnek sayısı
        epochs: Epoch sayısı
        
    Returns:
        str: Tahmini süre (readable format)
    """
    # Basit tahmin - gerçek değerler deneyime göre ayarlanmalı
    seconds_per_sample_per_epoch = 0.1  # Saniye
    total_seconds = samples * epochs * seconds_per_sample_per_epoch
    
    if total_seconds < 60:
        return f"{int(total_seconds)} saniye"
    elif total_seconds < 3600:
        return f"{int(total_seconds / 60)} dakika"
    else:
        return f"{int(total_seconds / 3600)} saat {int((total_seconds % 3600) / 60)} dakika"

@bp.route('/training-stats/<model_type>', methods=['GET'])
def get_training_stats(model_type):
    """
    Model eğitimi için veri istatistiklerini döndürür
    """
    try:
        if model_type == 'content':
            from app.services.content_training_service import ContentTrainingService
            
            trainer = ContentTrainingService()
            training_data = trainer.prepare_training_data(min_samples=1, validation_strategy='all')
            
            if training_data is None:
                return jsonify({
                    'success': True,
                    'stats': {
                        'total_feedbacks': 0,
                        'total_samples': 0,
                        'category_stats': {},
                        'message': 'Henüz feedback verisi bulunmuyor'
                    }
                })
            
            # Çelişki analizi
            conflicts = trainer.detect_feedback_conflicts(training_data)
            
            return jsonify({
                'success': True,
                'stats': {
                    'total_feedbacks': training_data['feedbacks_processed'],
                    'total_samples': training_data['total_samples'],
                    'category_stats': training_data['category_stats'],
                    'conflicts_detected': len(conflicts),
                    'conflicts': conflicts[:10]  # İlk 10 çelişki
                }
            })
            
        elif model_type == 'age':
            # Age model stats - AgeTrainingService kullan
            from app.services.age_training_service import AgeTrainingService
            from app.models.feedback import Feedback
            
            # Raw feedback sayılarını al (conflict resolution öncesi)
            raw_manual_feedbacks = Feedback.query.filter(
                Feedback.feedback_type == 'age',
                Feedback.feedback_source == 'MANUAL_USER'
            ).count()
            
            raw_pseudo_feedbacks = Feedback.query.filter(
                Feedback.feedback_type == 'age_pseudo',
                Feedback.feedback_source == 'PSEUDO_BUFFALO_HIGH_CONF'
            ).count()
            
            total_raw_feedbacks = raw_manual_feedbacks + raw_pseudo_feedbacks
            
            trainer = AgeTrainingService()
            training_data = trainer.prepare_training_data(min_samples=1)
            
            if training_data is None:
                return jsonify({
                    'success': True,
                    'stats': {
                        'total_feedbacks': total_raw_feedbacks,
                        'total_samples': 0,
                        'manual_samples': raw_manual_feedbacks,
                        'pseudo_samples': raw_pseudo_feedbacks,
                        'age_range': None,
                        'mean_age': None,
                        'message': 'Henüz yaş eğitim verisi bulunmuyor'
                    }
                })
            
            manual_count = training_data['sources'].count('manual')
            pseudo_count = training_data['sources'].count('pseudo')
            ages = training_data['ages']
            
            # Yaş dağılımını hesapla
            age_distribution = {}
            for age in ages:
                age_group = f"{(int(age) // 10) * 10}s"
                age_distribution[age_group] = age_distribution.get(age_group, 0) + 1
            
            return jsonify({
                'success': True,
                'stats': {
                    'total_feedbacks': total_raw_feedbacks,  # Raw toplam (manuel + pseudo)
                    'total_samples': len(training_data['embeddings']),  # Conflict resolution sonrası
                    'manual_samples': raw_manual_feedbacks,  # Raw manuel sayısı
                    'pseudo_samples': raw_pseudo_feedbacks,  # Raw pseudo sayısı
                    'age_range': {
                        'min': float(ages.min()),
                        'max': float(ages.max())
                    },
                    'mean_age': float(ages.mean()),
                    'std_age': float(ages.std()),
                    'age_distribution': age_distribution
                }
            })
            
        else:
            return jsonify({
                'success': False,
                'error': f'Desteklenmeyen model türü: {model_type}'
            }), 400
            
    except Exception as e:
        logger.error(f"Training stats error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'İstatistikler alınamadı: {str(e)}'
        }), 500 

@bp.route('/analyze-conflicts/<model_type>', methods=['GET'])
def analyze_conflicts(model_type):
    """
    Belirtilen model türü için çelişki analizi yapar
    
    Args:
        model_type: 'content' veya 'age'
        
    Returns:
        JSON: Çelişki analizi sonuçları
    """
    try:
        if model_type == 'content':
            from app.services.content_training_service import ContentTrainingService
            
            trainer = ContentTrainingService()
            training_data = trainer.prepare_training_data(min_samples=1, validation_strategy='all')
            
            if training_data is None:
                return jsonify({
                    'success': True,
                    'conflicts': [],
                    'total_conflicts': 0,
                    'high_severity': 0,
                    'summary': {
                        'categories_affected': 0,
                        'avg_score_diff': 0.0
                    },
                    'message': 'Henüz analiz edilecek feedback verisi bulunmuyor'
                })
            
            # Çelişkileri tespit et
            conflicts = trainer.detect_feedback_conflicts(training_data)
            
            # Özet istatistikleri hesapla
            high_severity_count = len([c for c in conflicts if c.get('severity') == 'high'])
            categories_affected = len(set(c['category'] for c in conflicts))
            avg_score_diff = sum(c.get('score_diff', 0) for c in conflicts) / len(conflicts) if conflicts else 0.0
            
            return jsonify({
                'success': True,
                'conflicts': conflicts,
                'total_conflicts': len(conflicts),
                'high_severity': high_severity_count,
                'summary': {
                    'categories_affected': categories_affected,
                    'avg_score_diff': avg_score_diff
                }
            })
            
        elif model_type == 'age':
            # Yaş modeli için çelişki analizi gerekli değil
            return jsonify({
                'success': True,
                'conflicts': [],
                'total_conflicts': 0,
                'high_severity': 0,
                'summary': {
                    'categories_affected': 0,
                    'avg_score_diff': 0.0
                },
                'message': 'Yaş modeli için çelişki analizi gerekli değildir'
            })
        
        else:
            return jsonify({
                'success': False,
                'error': f'Desteklenmeyen model türü: {model_type}'
            }), 400
            
    except Exception as e:
        logger.error(f"Çelişki analizi hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Çelişki analizi yapılırken hata oluştu: {str(e)}'
        }), 500 