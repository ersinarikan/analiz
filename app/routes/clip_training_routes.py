from flask import Blueprint, request, jsonify, current_app
from app.services.clip_training_service import CLIPTrainingService
from app.services.clip_version_service import CLIPVersionService
from app.models.clip_training import CLIPTrainingSession
from app import db
import logging
import threading
import time
from datetime import datetime

# Blueprint oluştur
clip_training_bp = Blueprint('clip_training', __name__, url_prefix='/api/clip-training')

logger = logging.getLogger(__name__)

@clip_training_bp.route('/statistics', methods=['GET'])
def get_training_statistics():
    """Eğitim için mevcut istatistikleri getir"""
    try:
        training_service = CLIPTrainingService()
        stats, error = training_service.get_training_statistics()
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Training istatistikleri alınırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/feedbacks', methods=['GET'])
def get_available_feedbacks():
    """Eğitim için kullanılabilir feedback'leri getir"""
    try:
        training_service = CLIPTrainingService()
        min_count = request.args.get('min_count', 50, type=int)
        
        feedbacks, error = training_service.get_available_feedbacks(min_count)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Feedback'leri serialize et
        feedback_data = []
        for feedback in feedbacks:
            feedback_data.append({
                'id': feedback.id,
                'image_path': feedback.image_path,
                'feedback_type': feedback.feedback_type,
                'category_feedback': feedback.category_feedback,
                'created_at': feedback.created_at.isoformat() if feedback.created_at else None
            })
        
        return jsonify({
            'success': True,
            'data': {
                'feedbacks': feedback_data,
                'total_count': len(feedback_data)
            }
        })
        
    except Exception as e:
        logger.error(f"Feedback'ler alınırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/prepare-data', methods=['POST'])
def prepare_training_data():
    """Training data'yı hazırla ve contrastive pairs oluştur"""
    try:
        training_service = CLIPTrainingService()
        data = request.get_json()
        categories = data.get('categories', ['violence', 'adult_content', 'harassment', 'weapon', 'drug'])
        train_split = data.get('train_split', 0.8)
        min_feedback_count = data.get('min_feedback_count', 50)
        
        # Feedback'leri al
        feedbacks, error = training_service.get_available_feedbacks(min_feedback_count)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Contrastive pairs oluştur
        pairs, error = training_service.create_contrastive_pairs(feedbacks, categories)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Training data'yı hazırla
        training_data, error = training_service.prepare_training_data(pairs, train_split)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'total_feedbacks': len(feedbacks),
                'total_pairs': training_data['total_pairs'],
                'train_pairs': len(training_data['train_pairs']),
                'val_pairs': len(training_data['val_pairs']),
                'category_stats': training_data['category_stats'],
                'ready_for_training': True
            }
        })
        
    except Exception as e:
        logger.error(f"Training data hazırlanırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/versions', methods=['GET'])
def get_model_versions():
    """Tüm CLIP model versiyonlarını listele"""
    try:
        version_service = CLIPVersionService()
        versions, error = version_service.get_all_versions()
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'versions': versions,
                'total_count': len(versions)
            }
        })
        
    except Exception as e:
        logger.error(f"Model versiyonları alınırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/versions/create', methods=['POST'])
def create_model_version():
    """Yeni model versiyonu oluştur"""
    try:
        version_service = CLIPVersionService()
        data = request.get_json()
        version_name = data.get('version_name')
        source_model = data.get('source_model', 'base')
        
        if not version_name:
            return jsonify({
                'success': False,
                'error': 'Version name gerekli'
            }), 400
        
        version_path, error = version_service.create_new_version(version_name, source_model)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'version_name': version_name,
                'version_path': version_path,
                'source_model': source_model
            }
        })
        
    except Exception as e:
        logger.error(f"Model versiyonu oluşturulurken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/versions/<version_name>/activate', methods=['POST'])
def activate_model_version(version_name):
    """Belirtilen versiyonu aktif model yap"""
    try:
        version_service = CLIPVersionService()
        import os
        version_path = os.path.join(version_service.versions_path, version_name)
        
        success, error = version_service.set_active_model(version_path)
        
        if not success:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Model versiyonu {version_name} aktif hale getirildi'
        })
        
    except Exception as e:
        logger.error(f"Model versiyonu aktif hale getirilirken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/versions/<version_name>', methods=['DELETE'])
def delete_model_version(version_name):
    """Model versiyonunu sil"""
    try:
        version_service = CLIPVersionService()
        success, error = version_service.delete_version(version_name)
        
        if not success:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Model versiyonu {version_name} silindi'
        })
        
    except Exception as e:
        logger.error(f"Model versiyonu silinirken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/sessions', methods=['GET'])
def get_training_sessions():
    """Training session'larını listele"""
    try:
        limit = request.args.get('limit', 10, type=int)
        sessions = CLIPTrainingSession.get_training_history(limit)
        
        session_data = []
        for session in sessions:
            session_data.append(session.to_dict())
        
        return jsonify({
            'success': True,
            'data': {
                'sessions': session_data,
                'total_count': len(session_data)
            }
        })
        
    except Exception as e:
        logger.error(f"Training session'ları alınırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/sessions/<session_id>', methods=['GET'])
def get_training_session(session_id):
    """Belirli bir training session'ın detaylarını getir"""
    try:
        session = CLIPTrainingSession.query.filter_by(id=session_id).first()
        
        if not session:
            return jsonify({
                'success': False,
                'error': 'Training session bulunamadı'
            }), 404
        
        return jsonify({
            'success': True,
            'data': session.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Training session detayları alınırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/start-training', methods=['POST'])
def start_training():
    """CLIP fine-tuning eğitimini başlat"""
    try:
        training_service = CLIPTrainingService()
        data = request.get_json()
        
        # Training parametreleri
        training_params = {
            'learning_rate': data.get('learning_rate', 1e-5),
            'batch_size': data.get('batch_size', 16),
            'epochs': data.get('epochs', 5),
            'categories': data.get('categories', ['violence', 'adult_content', 'harassment', 'weapon', 'drug']),
            'train_split': data.get('train_split', 0.8),
            'min_feedback_count': data.get('min_feedback_count', 50)
        }
        
        # Feedback'leri kontrol et
        feedbacks, error = training_service.get_available_feedbacks(training_params['min_feedback_count'])
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Aktif training session var mı kontrol et
        active_session = CLIPTrainingSession.get_active_session()
        if active_session:
            return jsonify({
                'success': False,
                'error': 'Zaten aktif bir training session var'
            }), 400
        
        # Database session oluştur (training_service içinde yapılacak)
        logger.info(f"CLIP Fine-tuning başlatılıyor: {len(feedbacks)} feedback ile")
        
        # Asenkron training başlat
        training_service.start_training_async(training_params)
        
        # Kısa bir süre bekleyip session ID'yi al
        time.sleep(1)
        
        # En son oluşturulan session'ı bul
        latest_session = CLIPTrainingSession.query.order_by(CLIPTrainingSession.created_at.desc()).first()
        
        if latest_session:
            return jsonify({
                'success': True,
                'data': {
                    'session_id': latest_session.id,
                    'version_name': latest_session.version_name,
                    'status': latest_session.status,
                    'message': 'CLIP Fine-tuning başlatıldı, arka planda devam ediyor'
                }
            })
        else:
            return jsonify({
                'success': True,
                'data': {
                    'session_id': 'unknown',
                    'version_name': 'unknown',
                    'status': 'starting',
                    'message': 'CLIP Fine-tuning başlatıldı'
                }
            })
        
    except Exception as e:
        logger.error(f"Training başlatılırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/sessions/<session_id>/stop', methods=['POST'])
def stop_training(session_id):
    """Training'i durdur"""
    try:
        session = CLIPTrainingSession.query.filter_by(id=session_id).first()
        
        if not session:
            return jsonify({
                'success': False,
                'error': 'Training session bulunamadı'
            }), 404
        
        if session.status != 'training':
            return jsonify({
                'success': False,
                'error': 'Training zaten durmuş durumda'
            }), 400
        
        # Training'i durdur (gerçek implementasyon Faz 4'te)
        session.status = 'stopped'
        session.training_end = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Training durduruldu'
        })
        
    except Exception as e:
        logger.error(f"Training durdurulurken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@clip_training_bp.route('/active-model', methods=['GET'])
def get_active_model():
    """Aktif model bilgilerini getir"""
    try:
        version_service = CLIPVersionService()
        info, error = version_service.get_active_model_info()
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Aktif training session'ı da ekle
        active_session = CLIPTrainingSession.get_active_session()
        
        return jsonify({
            'success': True,
            'data': {
                'model_info': info,
                'active_session': active_session.to_dict() if active_session else None
            }
        })
        
    except Exception as e:
        logger.error(f"Aktif model bilgisi alınırken hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 