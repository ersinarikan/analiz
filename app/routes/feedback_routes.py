from flask import Blueprint, request, jsonify, current_app
from app import db, socketio
from app.models.feedback import Feedback
import logging

bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')
logger = logging.getLogger(__name__)

@bp.route('/submit', methods=['POST'])
def submit_feedback():
    """
    İçerik analizi için geri bildirim gönderme endpoint'i.
    
    Request Body:
        - content_id: İçerik ID'si
        - rating: Genel puanlama (1-5)
        - comment: Yorumlar
        - category_feedback: Kategori geri bildirimleri
            - violence: Şiddet tespit geri bildirimi ('correct', 'false_positive', 'false_negative')
            - adult_content: Yetişkin içeriği tespit geri bildirimi
            - harassment: Taciz tespit geri bildirimi
            - weapon: Silah tespit geri bildirimi
            - drug: Madde kullanımı tespit geri bildirimi
        
    Returns:
        JSON: Başarı veya hata mesajı
    """
    try:
        data = request.json
        
        if not data or 'content_id' not in data:
            return jsonify({'error': 'content_id alanı gereklidir'}), 400
            
        # Geri bildirim verilerini al
        content_id = data['content_id']
        rating = data.get('rating', 3)
        comment = data.get('comment', '')
        category_feedback = data.get('category_feedback', {})
        
        # Geri bildirim nesnesi oluştur
        feedback = Feedback(
            content_id=content_id,
            rating=rating,
            comment=comment,
            violence_feedback=category_feedback.get('violence'),
            adult_content_feedback=category_feedback.get('adult_content'),
            harassment_feedback=category_feedback.get('harassment'),
            weapon_feedback=category_feedback.get('weapon'),
            drug_feedback=category_feedback.get('drug')
        )
        
        # Veritabanına kaydet
        db.session.add(feedback)
        db.session.commit()
        
        # WebSocket aracılığıyla gerçek zamanlı bildirim gönder
        socketio.emit('new_feedback', {
            'content_id': content_id,
            'feedback_id': feedback.id
        })
        
        return jsonify({
            'message': 'Geri bildirim başarıyla kaydedildi',
            'feedback_id': feedback.id
        }), 201
            
    except Exception as e:
        logger.error(f"Geri bildirim gönderme hatası: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Geri bildirim gönderilirken bir hata oluştu: {str(e)}'}), 500

@bp.route('/age', methods=['POST'])
def submit_age_feedback():
    """
    Yaş tahmini için geri bildirim gönderme endpoint'i.
    
    Request Body:
        - person_id: Kişi ID'si
        - corrected_age: Düzeltilmiş yaş değeri
        
    Returns:
        JSON: Başarı veya hata mesajı
    """
    try:
        data = request.json
        
        if not data or 'person_id' not in data or 'corrected_age' not in data:
            return jsonify({'error': 'person_id ve corrected_age alanları gereklidir'}), 400
            
        # Geri bildirim verilerini al
        person_id = data['person_id']
        corrected_age = data['corrected_age']
        
        if not isinstance(corrected_age, int) or corrected_age < 1 or corrected_age > 100:
            return jsonify({'error': 'Geçersiz yaş değeri. 1-100 arasında bir tam sayı olmalıdır.'}), 400
            
        # İlgili kişiyi bul ve yaş geri bildirimini kaydet
        # Gerçek uygulamada veritabanı modeline göre kod değişir
        # Örnek kod:
        # from app.models.age_estimation import AgeEstimation
        # age_estimation = AgeEstimation.query.filter_by(person_id=person_id).first()
        # 
        # if not age_estimation:
        #     return jsonify({'error': 'Kişi bulunamadı'}), 404
        # 
        # age_estimation.user_feedback_age = corrected_age
        # db.session.commit()
        
        # WebSocket aracılığıyla gerçek zamanlı bildirim gönder
        socketio.emit('age_feedback', {
            'person_id': person_id,
            'corrected_age': corrected_age
        })
        
        return jsonify({
            'message': 'Yaş geri bildirimi başarıyla kaydedildi',
            'person_id': person_id,
            'corrected_age': corrected_age
        }), 201
            
    except Exception as e:
        logger.error(f"Yaş geri bildirimi gönderme hatası: {str(e)}")
        return jsonify({'error': f'Yaş geri bildirimi gönderilirken bir hata oluştu: {str(e)}'}), 500

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