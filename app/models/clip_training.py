from app import db
from datetime import datetime
import json

class CLIPTrainingSession(db.Model):
    """CLIP Fine-tuning Training Sessions"""
    
    __tablename__ = 'clip_training_sessions'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    version_name = db.Column(db.String(100), nullable=False)
    feedback_count = db.Column(db.Integer, nullable=False)
    training_start = db.Column(db.DateTime, nullable=True)
    training_end = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default='preparing')  # preparing, training, completed, failed
    
    # Training parametreleri (JSON)
    training_params = db.Column(db.Text, nullable=True)
    
    # Performance metrikleri (JSON)
    performance_metrics = db.Column(db.Text, nullable=True)
    
    # Model paths
    model_path = db.Column(db.String(500), nullable=True)
    backup_path = db.Column(db.String(500), nullable=True)
    
    # Flags
    is_active = db.Column(db.Boolean, default=False)
    is_successful = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<CLIPTrainingSession {self.id}: {self.version_name}>'
    
    def to_dict(self):
        """Model'i dictionary'ye çevir"""
        return {
            'id': self.id,
            'version_name': self.version_name,
            'feedback_count': self.feedback_count,
            'training_start': self.training_start.isoformat() if self.training_start else None,
            'training_end': self.training_end.isoformat() if self.training_end else None,
            'status': self.status,
            'training_params': json.loads(self.training_params) if self.training_params else {},
            'performance_metrics': json.loads(self.performance_metrics) if self.performance_metrics else {},
            'model_path': self.model_path,
            'backup_path': self.backup_path,
            'is_active': self.is_active,
            'is_successful': self.is_successful,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'duration_minutes': self._calculate_duration()
        }
    
    def _calculate_duration(self):
        """Training süresini hesapla (dakika)"""
        if self.training_start and self.training_end:
            duration = self.training_end - self.training_start
            return round(duration.total_seconds() / 60, 2)
        return None
    
    def set_training_params(self, params_dict):
        """Training parametrelerini ayarla"""
        self.training_params = json.dumps(params_dict)
    
    def get_training_params(self):
        """Training parametrelerini getir"""
        if self.training_params:
            return json.loads(self.training_params)
        return {}
    
    def set_performance_metrics(self, metrics_dict):
        """Performance metriklerini ayarla"""
        self.performance_metrics = json.dumps(metrics_dict)
    
    def get_performance_metrics(self):
        """Performance metriklerini getir"""
        if self.performance_metrics:
            return json.loads(self.performance_metrics)
        return {}
    
    def start_training(self):
        """Training'i başlat"""
        self.status = 'training'
        self.training_start = datetime.utcnow()
        db.session.commit()
    
    def complete_training(self, success=True, metrics=None):
        """Training'i tamamla"""
        self.status = 'completed' if success else 'failed'
        self.training_end = datetime.utcnow()
        self.is_successful = success
        
        if metrics:
            self.set_performance_metrics(metrics)
        
        db.session.commit()
    
    def set_as_active(self):
        """Bu versiyonu aktif yap"""
        # Diğer tüm versiyonları pasif yap
        CLIPTrainingSession.query.update({'is_active': False})
        
        # Bu versiyonu aktif yap
        self.is_active = True
        db.session.commit()
    
    @staticmethod
    def get_active_session():
        """Aktif training session'ı getir"""
        return CLIPTrainingSession.query.filter_by(is_active=True).first()
    
    @staticmethod
    def get_latest_successful():
        """En son başarılı training session'ı getir"""
        return CLIPTrainingSession.query.filter_by(
            is_successful=True
        ).order_by(CLIPTrainingSession.created_at.desc()).first()
    
    @staticmethod
    def get_training_history(limit=10):
        """Training geçmişini getir"""
        return CLIPTrainingSession.query.order_by(
            CLIPTrainingSession.created_at.desc()
        ).limit(limit).all() 