from flask import Blueprint, request, jsonify, current_app
import logging
from app.services.content_training_service import ContentTrainingService
from app.services.clip_training_service import ClipTrainingService

logger = logging.getLogger('app.clip_training_routes')

clip_training_bp = Blueprint('clip_training', __name__, url_prefix='/api/clip-training')

@clip_training_bp.route('/status', methods=['GET'])
def get_training_status():
    """Training durumunu getir"""
    try:
        content_service = ContentTrainingService()
        clip_service = ClipTrainingService()
        
        # Training readiness analizi
        readiness = content_service.analyze_training_readiness()
        
        # Training statistics
        stats = clip_service.get_training_statistics()
        
        # Training history
        history = content_service.get_training_history(limit=5)
        
        return jsonify({
            'success': True,
            'readiness': readiness,
            'statistics': stats,
            'recent_history': history
        })
        
    except Exception as e:
        logger.error(f"Training status hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/analyze', methods=['POST'])
def analyze_training_data():
    """Training verilerini analiz et"""
    try:
        content_service = ContentTrainingService()
        
        # Detaylı analiz
        analysis = content_service.analyze_training_readiness()
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Training analizi hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/prepare', methods=['POST'])
def prepare_training():
    """Training session hazırla"""
    try:
        data = request.get_json() or {}
        training_params = data.get('training_params')
        
        content_service = ContentTrainingService()
        
        # Session hazırla
        preparation = content_service.prepare_training_session(training_params)
        
        return jsonify({
            'success': preparation['success'],
            **preparation
        })
        
    except Exception as e:
        logger.error(f"Training hazırlama hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/train', methods=['POST'])
def start_training():
    """CLIP modelini eğit"""
    try:
        data = request.get_json() or {}
        training_params = data.get('training_params')
        
        # Özel parametreler
        if training_params:
            # Güvenlik kontrolü - makul limitler
            training_params['epochs'] = min(training_params.get('epochs', 10), 50)
            training_params['batch_size'] = min(training_params.get('batch_size', 16), 64)
            training_params['learning_rate'] = max(
                min(training_params.get('learning_rate', 1e-4), 1e-2), 1e-6
            )
        
        content_service = ContentTrainingService()
        
        # WebSocket session ID oluştur ve training başlama bildirimi
        import uuid
        session_id = str(uuid.uuid4())
        
        try:
            from app.routes.websocket_routes import emit_training_started
            emit_training_started(session_id, 'CLIP', 100)  # Yaklaşık sample sayısı
        except Exception as ws_err:
            logger.warning(f"WebSocket training started event hatası: {str(ws_err)}")
        
        # Training'i başlat
        result = content_service.execute_training(training_params)
        
        if result['success']:
            # WebSocket ile training tamamlanma bildirimi
            try:
                from app.routes.websocket_routes import emit_training_completed
                emit_training_completed(
                    session_id,  # Bizim oluşturduğumuz session_id'yi kullan
                    'CLIP Model', 
                    result['performance']
                )
            except Exception as ws_err:
                logger.warning(f"WebSocket training completed event hatası: {str(ws_err)}")
            
            return jsonify({
                'success': True,
                'message': 'CLIP model training başarıyla tamamlandı',
                'training_session_id': session_id,  # Bizim session_id'yi döndür
                'performance': result['performance'],
                'model_updated': True
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        logger.error(f"Training başlatma hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/history', methods=['GET'])
def get_training_history():
    """Training geçmişini getir"""
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Maksimum 100 kayıt
        
        content_service = ContentTrainingService()
        history = content_service.get_training_history(limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'total_count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Training history hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/statistics', methods=['GET'])
def get_detailed_statistics():
    """Detaylı training istatistikleri"""
    try:
        clip_service = ClipTrainingService()
        content_service = ContentTrainingService()
        
        # CLIP statistics
        clip_stats = clip_service.get_training_statistics()
        
        # Training readiness
        readiness = content_service.analyze_training_readiness()
        
        # Combined statistics
        combined_stats = {
            'clip_model': clip_stats,
            'training_readiness': readiness,
            'system_info': {
                'device': 'cuda' if current_app.config.get('USE_GPU', True) else 'cpu',
                'models_folder': current_app.config.get('MODELS_FOLDER'),
                'openclip_active_path': current_app.config.get('OPENCLIP_MODEL_ACTIVE_PATH')
            }
        }
        
        return jsonify({
            'success': True,
            'statistics': combined_stats
        })
        
    except Exception as e:
        logger.error(f"Detaylı istatistik hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/test-training', methods=['POST'])
def test_training_pipeline():
    """Training pipeline'ını test et (dry run)"""
    try:
        content_service = ContentTrainingService()
        clip_service = ClipTrainingService()
        
        # Test verilerini hazırla
        test_data = clip_service.prepare_training_data(min_samples=1)
        
        if not test_data:
            return jsonify({
                'success': False,
                'error': 'Test için yeterli veri yok'
            }), 400
        
        # Test istatistikleri
        test_stats = {
            'data_preparation': 'OK',
            'total_samples': test_data['total_samples'],
            'train_samples': test_data['train_samples'],
            'val_samples': test_data['val_samples'],
            'sample_data': {
                'first_positive_caption': test_data['train_positive_captions'][0] if test_data['train_positive_captions'] else None,
                'first_negative_caption': test_data['train_negative_captions'][0] if test_data['train_negative_captions'] else None,
                'first_labels': test_data['train_labels'][0] if test_data['train_labels'] else None
            }
        }
        
        # Model yükleme testi
        try:
            model_loaded = clip_service.load_base_model()
            test_stats['model_loading'] = 'OK' if model_loaded else 'FAILED'
        except Exception as e:
            test_stats['model_loading'] = f'FAILED: {str(e)}'
        
        return jsonify({
            'success': True,
            'test_results': test_stats,
            'message': 'Training pipeline test tamamlandı'
        })
        
    except Exception as e:
        logger.error(f"Training test hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/feedback-analysis', methods=['GET'])
def analyze_feedback_for_training():
    """Training için feedback analizi"""
    try:
        content_service = ContentTrainingService()
        
        # Sadece feedback analizini çalıştır
        analysis = content_service._analyze_feedback_data()
        quality = content_service._check_data_quality()
        
        return jsonify({
            'success': True,
            'feedback_analysis': analysis,
            'data_quality': quality
        })
        
    except Exception as e:
        logger.error(f"Feedback analizi hatası: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 