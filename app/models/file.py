import datetime
from app import db

class File(db.Model):
    """
    Yüklenen dosya modeli.
    - Dosya yolu, türü, yüklenme zamanı ve ilişkili analizlerle bağlantı sağlar.
    """
    
    __tablename__ = 'files'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False, unique=True)
    file_size = db.Column(db.Integer, nullable=False)  # bytes cinsinden
    mime_type = db.Column(db.String(128), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'image' veya 'video'
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, nullable=True)  # Kullanıcı ID referansı
    
    # İlişkiler - çakışmayı engellemek için backref'leri kaldırdık
    analyses = db.relationship('Analysis', 
                              foreign_keys='Analysis.file_id', 
                              lazy='dynamic',
                              primaryjoin="func.cast(File.id, String) == Analysis.file_id",
                              back_populates="file")
    
    # Latest analysis tek bir analiz erişimi için kullanılabilir
    # Çakışmayı önlemek için analysis ilişkisi kaldırıldı
    
    def __init__(self, filename: str, original_filename: str, file_path: str, file_size: int, mime_type: str, user_id: int | None = None):
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.mime_type = mime_type
        self.user_id = user_id
        
        # MIME tipine göre dosya türünü belirle
        if mime_type.startswith('image/'):
            self.file_type = 'image'
        elif mime_type.startswith('video/'):
            self.file_type = 'video'
        else:
            self.file_type = 'unknown'
    
    def to_dict(self) -> dict:
        """Modeli JSON'a dönüştürmek için dict temsilini döndürür."""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'file_type': self.file_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'has_analysis': self.analyses.count() > 0,
            'user_id': self.user_id
        } 