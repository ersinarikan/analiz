from flask import Blueprint, request, jsonify, current_app
from app import db, socketio
from app.models.feedback import Feedback
from app.models.analysis import FaceTracking
import logging

bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')
logger = logging.getLogger(__name__)

@bp.route('/submit', methods=['POST'])
def submit_feedback():
    """
    İçerik analizi için geri bildirim gönderir.
    
    Bu fonksiyon, kullanıcıdan gelen içerik analizi geri bildirimlerini işler ve veritabanına kaydeder.
    Geri bildirimler, model eğitimi için kullanılabilir.
    
    Post Body:
        {
            "content_id": "uuid",
            "rating": 1-5,
            "comment": "string",
            "category_feedback": {
                "violence": "accurate/under_estimated/over_estimated/false_positive/false_negative",
                "adult_content": "...",
                "harassment": "...",
                "weapon": "...",
                "drug": "..."
            },
            "category_correct_values": {
                "violence": 0-100,
                "adult_content": 0-100,
                "harassment": 0-100,
                "weapon": 0-100,
                "drug": 0-100
            }
        }
    """
    try:
        data = request.json
        
        if not data or 'content_id' not in data:
            return jsonify({'error': 'Geçersiz istek formatı'}), 400
        
        # Geri bildirim oluştur
        feedback = Feedback(
            content_id=data['content_id'],
            rating=data.get('rating'),
            comment=data.get('comment'),
            category_feedback=data.get('category_feedback', {}),
            category_correct_values=data.get('category_correct_values', {})
        )
        
        # Veritabanına kaydet
        db.session.add(feedback)
        db.session.commit()
        
        logger.info(f"Geri bildirim kaydedildi, ID: {feedback.id}, içerik ID: {data['content_id']}")
        
        return jsonify({
            'success': True, 
            'feedback_id': feedback.id,
            'message': 'Geri bildirim başarıyla kaydedildi'
        }), 201
    
    except Exception as e:
        logger.error(f"Geri bildirim kaydedilirken hata: {str(e)}")
        return jsonify({'error': f'Geri bildirim kaydedilemedi: {str(e)}'}), 500

@bp.route('/age', methods=['POST'])
def submit_age_feedback():
    """
    Yaş tahmini için geri bildirim gönderir.
    
    Bu fonksiyon, kullanıcıdan gelen yaş tahmini geri bildirimlerini işler ve veritabanına kaydeder.
    Yaş geri bildirimleri, yaş tahmin modelinin eğitimi için kullanılabilir.
    
    Post Body:
        {
            "person_id": "uuid", 
            "corrected_age": int,
            "is_age_range_correct": bool,
            "analysis_id": "uuid" (opsiyonel)
        }
    """
    try:
        data = request.json
        
        if not data or 'person_id' not in data or 'corrected_age' not in data:
            return jsonify({'error': 'Geçersiz istek formatı'}), 400
        
        person_id = data['person_id']
        corrected_age = data['corrected_age']
        is_age_range_correct = data.get('is_age_range_correct', False)
        analysis_id = data.get('analysis_id')
        
        # Kişi ID'si geçerli mi kontrol et
        face = FaceTracking.query.filter_by(person_id=person_id).first()
        
        if not face:
            return jsonify({'error': f'Belirtilen person_id bulunamadı: {person_id}'}), 404
        
        # Yaş geri bildirimi oluştur
        feedback = Feedback(
            person_id=person_id,
            analysis_id=analysis_id or face.analysis_id,
            corrected_age=corrected_age,
            is_age_range_correct=is_age_range_correct,
            feedback_type='age'
        )
        
        # Veritabanına kaydet
        db.session.add(feedback)
        db.session.commit()
        
        logger.info(f"Yaş geri bildirimi kaydedildi, ID: {feedback.id}, kişi ID: {person_id}, düzeltilmiş yaş: {corrected_age}")
        
        return jsonify({
            'success': True, 
            'feedback_id': feedback.id,
            'message': 'Yaş geri bildirimi başarıyla kaydedildi'
        }), 201
    
    except Exception as e:
        logger.error(f"Yaş geri bildirimi kaydedilirken hata: {str(e)}")
        return jsonify({'error': f'Yaş geri bildirimi kaydedilemedi: {str(e)}'}), 500

@bp.route('/content/<content_id>', methods=['GET'])
def get_content_feedback(content_id):
    """
    Belirli bir içerik için geri bildirimleri getirir.
    
    Args:
        content_id: İçerik ID'si
        
    Returns:
        JSON: Geri bildirim listesi
    """
    try:
        # İçerik için tüm geri bildirimleri bul
        feedbacks = Feedback.query.filter_by(content_id=content_id).all()
        
        return jsonify([feedback.to_dict() for feedback in feedbacks]), 200
            
    except Exception as e:
        logger.error(f"Geri bildirim getirme hatası: {str(e)}")
        return jsonify({'error': f'Geri bildirimler getirilirken bir hata oluştu: {str(e)}'}), 500 