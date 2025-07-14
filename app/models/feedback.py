import uuid
from datetime import datetime
from app import db
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, LargeBinary

class Feedback(db.Model):
    """
    Kullanıcı geri bildirimi modeli.
    - Analiz sonuçlarına verilen yaş, içerik, doğruluk gibi geri bildirimleri tutar.
    """
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Görüntü ve Yüz Bilgileri
    frame_path = db.Column(db.String(1024), nullable=True)
    face_bbox = db.Column(db.String(255), nullable=True) # "x1,y1,x2,y2" veya JSON
    embedding = db.Column(db.Text, nullable=True) # Artık virgül ile ayrılmış float string olarak saklanacak

    # İçerik ve Analiz ID'leri
    content_id = db.Column(db.String(36), nullable=True)
    analysis_id = db.Column(db.String(36), nullable=True)
    
    # Kişi ID'si (yaş geri bildirimi için)
    person_id = db.Column(db.String(36), nullable=True, index=True)
    
    # Yaş Geri Bildirimleri
    corrected_age = db.Column(db.Integer, nullable=True) # Kullanıcının girdiği yaş
    pseudo_label_original_age = db.Column(db.Float, nullable=True) # BuffaloL'nin sözde etiket yaşı
    pseudo_label_clip_confidence = db.Column(db.Float, nullable=True) # BuffaloL sözde etiketinin CLIP güveni
    is_age_range_correct = db.Column(db.Boolean, nullable=True) # default=False kaldırıldı

    # Geri Bildirim Türü ve Kaynağı
    feedback_type = db.Column(db.String(50), nullable=True, index=True) # örn: 'age', 'content', 'general'
    feedback_source = db.Column(db.String(50), nullable=True, default='MANUAL_USER', index=True) 
                                # örn: 'MANUAL_USER_AGE_CORRECTION', 'PSEUDO_BUFFALO_HIGH_CONF'
    
    # Genel Derecelendirme ve Yorum
    rating = db.Column(db.Integer, nullable=True)
    comment = db.Column(db.Text, nullable=True)
    
    # Kategori Bazlı Geri Bildirimler (JSON)
    category_feedback = db.Column(db.JSON, nullable=True) # PostgreSQL için JSON, diğer DB'ler için db.Text veya db.String
    category_correct_values = db.Column(db.JSON, nullable=True) # PostgreSQL için JSON, diğer DB'ler için db.Text veya db.String
    
    # Eğitim Durumu Alanları
    training_status = db.Column(db.String(50), nullable=True, index=True) # 'used_in_training', 'archived', vb.
    used_in_model_version = db.Column(db.String(100), nullable=True) # Hangi model versiyonunda kullanıldı
    training_used_at = db.Column(db.DateTime, nullable=True) # Ne zaman eğitimde kullanıldı
    is_archived = db.Column(db.Boolean, default=False, index=True) # Arşivlenmiş mi
    archive_reason = db.Column(db.String(100), nullable=True) # Arşivleme nedeni
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, type='{self.feedback_type}', source='{self.feedback_source}')>"
    
    def to_dict(self) -> dict:
        data = {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'frame_path': self.frame_path,
            'face_bbox': self.face_bbox,
            # embedding'i to_dict'e eklemek genellikle iyi bir fikir değil, büyük olabilir.
            'content_id': self.content_id,
            'analysis_id': self.analysis_id,
            'person_id': self.person_id,
            'corrected_age': self.corrected_age,
            'pseudo_label_original_age': self.pseudo_label_original_age,
            'pseudo_label_clip_confidence': self.pseudo_label_clip_confidence,
            'is_age_range_correct': self.is_age_range_correct,
            'feedback_type': self.feedback_type,
            'feedback_source': self.feedback_source,
            'rating': self.rating,
            'comment': self.comment,
            'category_feedback': self.category_feedback,
            'category_correct_values': self.category_correct_values,
            'training_status': self.training_status,
            'used_in_model_version': self.used_in_model_version,
            'training_used_at': self.training_used_at.isoformat() if self.training_used_at else None,
            'is_archived': self.is_archived,
            'archive_reason': self.archive_reason
        }
        return data 