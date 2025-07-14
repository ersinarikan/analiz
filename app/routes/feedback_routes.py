from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models.feedback import Feedback
from app.models.analysis import FaceTracking, AgeEstimation
import logging
import os
from app.utils.path_utils import to_rel_path

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')
"""
Geri bildirim işlemleri için blueprint.
- Kullanıcı geri bildirimi gönderme ve yönetme endpointlerini içerir.
"""
logger = logging.getLogger(__name__)

@feedback_bp.route('/submit', methods=['POST'])
def submit_feedback():
    """
    İçerik analizi için geri bildirim gönderir.
    
    Bu fonksiyon, kullanıcıdan gelen içerik analizi geri bildirimlerini işler ve veritabanına kaydeder.
    Geri bildirimler, model eğitimi için kullanılabilir.
    
    Post Body:
        {
            "content_id": "uuid",
            "analysis_id": "uuid",
            "frame_path": "string",
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
        
        if not data or 'content_id' not in data or 'analysis_id' not in data:
            return jsonify({'error': 'Geçersiz istek formatı. content_id ve analysis_id gereklidir.'}), 400
        
        feedback = Feedback(
            content_id=data['content_id'],
            analysis_id=data['analysis_id'],
            frame_path=to_rel_path(data.get('frame_path')),
            rating=data.get('rating'),
            comment=data.get('comment'),
            category_feedback=data.get('category_feedback', {}),
            category_correct_values=data.get('category_correct_values', {}),
            feedback_type='content',  # Content feedback olarak işaretle
            feedback_source='MANUAL_USER_CONTENT_CORRECTION'  # Web arayüzünden gelen feedback
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        logger.info(f"İçerik geri bildirimi kaydedildi, ID: {feedback.id}, içerik ID: {data['content_id']}, analiz ID: {data['analysis_id']}")
        
        return jsonify({
            'success': True, 
            'feedback_id': feedback.id,
            'message': 'Geri bildirim başarıyla kaydedildi'
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Geri bildirim kaydedilirken hata: {str(e)}")
        return jsonify({'error': f'Geri bildirim kaydedilemedi: {str(e)}'}), 500

@feedback_bp.route('/age', methods=['POST'])
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
            "analysis_id": "uuid",
            "frame_path": "string"
        }
    """
    try:
        data = request.json
        
        # Gerekli alanları kontrol et
        required_fields = ['person_id', 'corrected_age', 'analysis_id', 'frame_path']
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({'error': f'Geçersiz istek formatı. {field} alanı gereklidir ve boş olamaz.'}), 400
        
        person_id = data['person_id']
        corrected_age = data['corrected_age']
        is_age_range_correct = data.get('is_age_range_correct', False)
        analysis_id = data['analysis_id']
        frame_path = to_rel_path(data['frame_path'])
        
        face = FaceTracking.query.filter_by(person_id=person_id, analysis_id=analysis_id).first()
        
        if not face:
            # Alternatif olarak sadece person_id ile de kontrol edilebilir eğer analysis_id her zaman face ile gelmiyorsa
            # Ancak hem person_id hem analysis_id daha spesifik bir yüzü hedefler.
            logger.warning(f"Belirtilen person_id ({person_id}) ve analysis_id ({analysis_id}) ile eşleşen FaceTracking kaydı bulunamadı.")
            # return jsonify({'error': f'Belirtilen person_id ({person_id}) ve analysis_id ({analysis_id}) ile yüz takibi kaydı bulunamadı.'}), 404
            # Şimdilik kayda devam et, bu durum loglanmış oldu.

        # Embedding'i AgeEstimation tablosundan bul
        embedding_str = None
        try:
            # _confidence_score sütunu ile sıralama
            age_est = AgeEstimation.query.filter_by(analysis_id=analysis_id, person_id=person_id).order_by(AgeEstimation._confidence_score.desc()).first()
            if age_est and hasattr(age_est, 'embedding') and age_est.embedding:
                emb = age_est.embedding
                if isinstance(emb, str):
                    embedding_str = emb
                elif hasattr(emb, 'tolist'):
                    embedding_str = ",".join(str(float(x)) for x in emb.tolist())
                elif isinstance(emb, (list, tuple)):
                    embedding_str = ",".join(str(float(x)) for x in emb)
                else:
                    embedding_str = str(emb)
        except Exception as emb_err:
            logger.warning(f"Embedding alınırken hata: {str(emb_err)}")

        feedback = Feedback(
            person_id=person_id,
            analysis_id=analysis_id,
            corrected_age=corrected_age,
            is_age_range_correct=is_age_range_correct,
            feedback_type='age',
            frame_path=frame_path,
            embedding=embedding_str
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        logger.info(f"Yaş geri bildirimi kaydedildi, ID: {feedback.id}, kişi ID: {person_id}, analiz ID: {analysis_id}, düzeltilmiş yaş: {corrected_age}")
        
        return jsonify({
            'success': True, 
            'feedback_id': feedback.id,
            'message': 'Yaş geri bildirimi başarıyla kaydedildi'
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Yaş geri bildirimi kaydedilirken hata: {str(e)}")
        return jsonify({'error': f'Yaş geri bildirimi kaydedilemedi: {str(e)}'}), 500

@feedback_bp.route('/content/<content_id>', methods=['GET'])
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

bp = feedback_bp 