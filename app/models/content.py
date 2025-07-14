import enum
from datetime import datetime
import json
from app import db
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

class ContentType(enum.Enum):
    """
    İçerik türü modeli.
    - İçerik kategorilerini ve ilişkili analizleri tanımlar.
    """
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"

class Content(db.Model):
    """İçerik bilgilerini tutan model sınıfı."""
    __tablename__ = 'contents'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID formatında
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(1024), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    mime_type = db.Column(db.String(50), nullable=False)
    content_type = db.Column(db.Enum(ContentType), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.now)
    thumbnail = db.Column(db.LargeBinary, nullable=True)
    
    # İlişkiler
    analysis_results = db.relationship('AnalysisResult', backref='content', lazy=True)
    
    def add_analysis_result(self, category: str, score: float, details: dict | None = None) -> 'AnalysisResult':
        """İçerik için analiz sonucu ekler."""
        result = AnalysisResult(
            content_id=self.id,
            category=category,
            score=score,
            details=json.dumps(details) if details else None
        )
        self.analysis_results.append(result)
        return result

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'content_type': self.content_type.value,
            'user_id': self.user_id,
            'upload_date': self.upload_date.isoformat(),
            'thumbnail': self.thumbnail.decode('utf-8') if self.thumbnail else None
        }

class AnalysisResult(db.Model):
    """Bir içerik için analiz sonuçlarını tutan model sınıfı."""
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.String(36), db.ForeignKey('contents.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # violence, adult_content, harassment, vb.
    score = db.Column(db.Float, nullable=False)  # 0-1 arası tespit skoru
    details = db.Column(db.Text, nullable=True)  # JSON formatında detay bilgisi
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def get_details(self) -> dict:
        """Detayları JSON nesnesine dönüştürür."""
        if self.details:
            try:
                return json.loads(self.details)
            except:
                return {}
        return {}

class ContentCategory(enum.Enum):
    """Analiz edilecek içerik kategorilerini temsil eden enum sınıfı."""
    VIOLENCE = "violence"
    ADULT_CONTENT = "adult_content"
    HARASSMENT = "harassment"
    WEAPON = "weapon"
    DRUG = "drug"

class ModelVersion(db.Model):
    """
    Eğitilmiş modellerin sürüm bilgilerini saklar.
    Bu tablo, model eğitim geçmişini ve metriklerini takip etmeyi sağlar.
    """
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)  # 'content' veya 'age'
    version = db.Column(db.Integer, nullable=False)
    version_name = db.Column(db.String(100), nullable=True)  # Örn: "v1_20240101_120000"
    created_at = db.Column(db.DateTime, default=datetime.now)
    metrics = db.Column(db.JSON) # Doğruluk, F1, Kesinlik, Geri çağırma skorları
    is_active = db.Column(db.Boolean, default=True)
    training_samples = db.Column(db.Integer) # Eğitim için kullanılan örnek sayısı
    validation_samples = db.Column(db.Integer) # Doğrulama için kullanılan örnek sayısı
    epochs = db.Column(db.Integer) # Eğitimde kullanılan epoch sayısı
    
    # Model dosya yolları
    file_path = db.Column(db.String(255))
    weights_path = db.Column(db.String(255))
    
    # İlişkili geri bildirimler (eğitimde kullanılan)
    used_feedback_ids = db.Column(db.JSON) # Eğitimde kullanılan geri bildirim ID'leri
    
    def __repr__(self):
        return f"<ModelVersion(id={self.id}, model_type='{self.model_type}', version={self.version}, active={self.is_active})>" 