import uuid
from datetime import datetime
from app import db
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey

class Feedback(db.Model):
    """İçerik analizlerine kullanıcı geri bildirimleri için model."""
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # İçerik ile ilgili alanlar (İçerik analizi geri bildirimleri için)
    content_id = db.Column(db.String(36), nullable=True)  # İçerik için UUID
    analysis_id = db.Column(db.String(36), nullable=True)  # Analiz için UUID
    
    # Yaş tahmini ile ilgili alanlar
    person_id = db.Column(db.String(36), nullable=True)  # Kişi için UUID 
    corrected_age = db.Column(db.Integer, nullable=True)  # Düzeltilmiş yaş değeri
    is_age_range_correct = db.Column(db.Boolean, default=False)  # Yaş aralığı doğru mu?
    
    # Geri bildirim türü - 'content' veya 'age'
    feedback_type = db.Column(db.String(20), default='content')
    
    # Genel geri bildirim
    rating = db.Column(db.Integer, nullable=True)  # 1-5 arası puan
    comment = db.Column(db.Text, nullable=True)  # Kullanıcı yorumu
    
    # Kategori geri bildirimleri
    # JSON formatında saklanan kategori değerlendirmeleri
    # Örn: {'violence': 'accurate', 'adult_content': 'over_estimated'} vs.
    category_feedback = db.Column(db.JSON, nullable=True)
    
    # Kullanıcı tarafından girilen doğru değerler
    # JSON formatında saklanan 0-100 arası değerler
    # Örn: {'violence': 75, 'adult_content': 30} vs.
    category_correct_values = db.Column(db.JSON, nullable=True)
    
    def __repr__(self):
        if self.feedback_type == 'age':
            return f"<AgeFeedback(id={self.id}, person_id={self.person_id}, corrected_age={self.corrected_age})>"
        else:
            return f"<ContentFeedback(id={self.id}, content_id={self.content_id}, rating={self.rating})>"
    
    def to_dict(self):
        """Geri bildirimi sözlük formatına dönüştürür"""
        result = {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'feedback_type': self.feedback_type,
            'rating': self.rating,
            'comment': self.comment
        }
        
        if self.feedback_type == 'content':
            result.update({
                'content_id': self.content_id,
                'analysis_id': self.analysis_id,
                'category_feedback': self.category_feedback,
                'category_correct_values': self.category_correct_values
            })
        else:  # age
            result.update({
                'person_id': self.person_id,
                'corrected_age': self.corrected_age,
                'is_age_range_correct': self.is_age_range_correct
            })
        
        return result 