import enum
from datetime import datetime
import json
from app import db

class ContentType(enum.Enum):
    """İçerik türlerini temsil eden enum sınıfı."""
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
    
    def add_analysis_result(self, category, score, details=None):
        """İçerik için analiz sonucu ekler."""
        result = AnalysisResult(
            content_id=self.id,
            category=category,
            score=score,
            details=json.dumps(details) if details else None
        )
        self.analysis_results.append(result)
        return result

class AnalysisResult(db.Model):
    """Bir içerik için analiz sonuçlarını tutan model sınıfı."""
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.String(36), db.ForeignKey('contents.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # violence, adult_content, harassment, vb.
    score = db.Column(db.Float, nullable=False)  # 0-1 arası tespit skoru
    details = db.Column(db.Text, nullable=True)  # JSON formatında detay bilgisi
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def get_details(self):
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