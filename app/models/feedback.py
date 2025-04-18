from datetime import datetime
from app import db

class Feedback(db.Model):
    """
    Kullanıcı geri bildirimlerini saklayan model.
    Bu model, içerik analizi sonuçlarına verilen geri bildirimleri depolar.
    """
    __tablename__ = 'feedbacks'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.String(36), nullable=False)  # İçerik UUID'si
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Genel geri bildirim
    rating = db.Column(db.Integer)  # 1-5 arası puanlama
    comment = db.Column(db.Text)  # Yorumlar
    
    # Kategorik geri bildirimler - 'correct', 'false_positive', 'false_negative'
    violence_feedback = db.Column(db.String(20))
    adult_content_feedback = db.Column(db.String(20))
    harassment_feedback = db.Column(db.String(20))
    weapon_feedback = db.Column(db.String(20))
    drug_feedback = db.Column(db.String(20))
    
    # Kişi bazlı geri bildirimler
    person_id = db.Column(db.String(50))  # Eğer belirli bir kişi için geri bildirim ise
    age_feedback = db.Column(db.Integer)  # Düzeltilen yaş değeri
    
    def to_dict(self):
        """
        Geri bildirimi sözlük formatında döndürür.
        
        Returns:
            dict: Geri bildirimin tüm özellikleriyle sözlük temsili
        """
        return {
            'id': self.id,
            'content_id': self.content_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'rating': self.rating,
            'comment': self.comment,
            'category_feedback': {
                'violence': self.violence_feedback,
                'adult_content': self.adult_content_feedback,
                'harassment': self.harassment_feedback,
                'weapon': self.weapon_feedback,
                'drug': self.drug_feedback
            },
            'person_id': self.person_id,
            'age_feedback': self.age_feedback
        } 