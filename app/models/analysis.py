from datetime import datetime
import json
from app import db
from app.models.file import File
import logging
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from app.json_encoder import json_dumps_numpy, NumPyJSONEncoder
import numpy as np
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import traceback
from flask import current_app
from app.utils.serialization_utils import convert_numpy_types_to_python, debug_serialization
import uuid

logger = logging.getLogger(__name__)

class Analysis(db.Model):
    """
    İçerik analiz sonuçlarını saklayan model.
    Resim ve video dosyalarının şiddet, yetişkin içerik, taciz, silah ve madde kullanımı 
    kategorilerindeki analiz sonuçlarını içerir.
    """
    __tablename__ = 'analyses'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id = db.Column(db.String(36), nullable=False)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    
    start_time = db.Column(db.DateTime, default=datetime.now)
    end_time = db.Column(db.DateTime, nullable=True)
    
    frames_analyzed = db.Column(db.Integer, default=0)
    frames_per_second = db.Column(db.Float, default=1.0)
    
    error_message = db.Column(db.Text, nullable=True)
    
    include_age_analysis = db.Column(db.Boolean, default=False)  # Yaş tahmini yapılsın mı?
    
    # Genel kategorik skorlar (0-1 arası)
    overall_violence_score = db.Column(db.Float, default=0)
    overall_adult_content_score = db.Column(db.Float, default=0)
    overall_harassment_score = db.Column(db.Float, default=0)
    overall_weapon_score = db.Column(db.Float, default=0)
    overall_drug_score = db.Column(db.Float, default=0)
    
    # En yüksek riskli çerçeve bilgileri
    highest_risk_frame = db.Column(db.String(255))  # En riskli kare dosya yolu
    highest_risk_frame_timestamp = db.Column(db.Float)  # Kare zaman damgası (videolar için)
    highest_risk_score = db.Column(db.Float)  # En yüksek risk skoru
    highest_risk_category = db.Column(db.String(20))  # En riskli kategori ('violence', 'adult_content', vb.)
    
    # İlişkiler - Çakışmayı önlemek için file_ref ve file ilişkilerini kaldırdık
    # Bunun yerine, tek bir file ilişkisi kullanacağız
    file = db.relationship('File', foreign_keys=[file_id])
    content_detections = db.relationship('ContentDetection', backref='analysis', lazy=True, cascade="all, delete-orphan")
    age_estimations = db.relationship('AgeEstimation', backref='analysis', lazy=True, cascade="all, delete-orphan")
    
    def start_analysis(self):
        """Analiz sürecini başlatır ve durumu 'processing' olarak günceller."""
        self.status = 'processing'
        self.progress = 0
        self.status_message = 'Analiz başlatıldı'
        db.session.commit()
    
    def update_progress(self, progress):
        """
        Analiz ilerleme durumunu günceller.
        
        Args:
            progress: 0-100 arası ilerleme yüzdesi
        """
        self.progress = min(progress, 100)  # 100'den büyük değerler kabul edilmez
        db.session.commit()
    
    def complete_analysis(self):
        """Analizi başarıyla tamamlandı olarak işaretler."""
        self.status = 'completed'
        self.progress = 100
        self.status_message = 'Analiz tamamlandı'
        db.session.commit()
    
    def fail_analysis(self, message):
        """
        Analizi başarısız olarak işaretler.
        
        Args:
            message: Başarısızlık nedeni
        """
        self.status = 'failed'
        self.status_message = message
        db.session.commit()
    
    def to_dict(self):
        """
        Analiz verilerini sözlük formatında döndürür.
        
        Returns:
            dict: Analizin tüm özellikleriyle sözlük temsili
        """
        file_info = None
        if self.file:
            file_info = {
                'filename': self.file.original_filename,
                'file_type': self.file.file_type
            }
            
        return {
            'id': self.id,
            'file_id': self.file_id,
            'file_info': file_info,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'status': self.status,
            'progress': self.progress,
            'status_message': self.status_message,
            'frames_per_second': self.frames_per_second,
            'include_age_analysis': self.include_age_analysis,
            'overall_violence_score': self.overall_violence_score,
            'overall_adult_content_score': self.overall_adult_content_score,
            'overall_harassment_score': self.overall_harassment_score,
            'overall_weapon_score': self.overall_weapon_score,
            'overall_drug_score': self.overall_drug_score,
            'highest_risk_frame': self.highest_risk_frame,
            'highest_risk_frame_timestamp': self.highest_risk_frame_timestamp,
            'highest_risk_score': self.highest_risk_score,
            'highest_risk_category': self.highest_risk_category
        }


class ContentDetection(db.Model):
    """
    Her bir kare veya resim için içerik analiz sonuçlarını saklar.
    Şiddet, yetişkin içerik, taciz, silah ve madde kullanımı kategorilerindeki
    tespit skorlarını ve tespit edilen nesneleri içerir.
    """
    __tablename__ = 'content_detections'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id'), nullable=False)
    frame_path = db.Column(db.String(255))  # Analiz edilen karenin dosya yolu
    frame_timestamp = db.Column(db.Float)  # Karenin video içindeki zaman damgası (saniye)
    
    # Kategorik skorlar (0-1 arası)
    violence_score = db.Column(db.Float, default=0)
    adult_content_score = db.Column(db.Float, default=0)
    harassment_score = db.Column(db.Float, default=0)
    weapon_score = db.Column(db.Float, default=0)
    drug_score = db.Column(db.Float, default=0)
    
    # Tespit edilen nesneler JSON formatında
    _detected_objects = db.Column(db.Text)  # JSON olarak saklanan tespit edilen nesneler
    
    def set_detected_objects(self, detected_objects):
        """
        Algılanan nesneleri JSON olarak ayarlar. NumPy tiplerini düzgün bir şekilde işler.
        
        Args:
            detected_objects: Algılanan nesne listesi veya sözlüğü
        """
        try:
            from app.utils.serialization_utils import convert_numpy_types_to_python, debug_serialization
            
            # NumPy türlerini Python türlerine dönüştür
            python_objects = convert_numpy_types_to_python(detected_objects)
            
            # JSON'a çevir
            self.detected_objects_json = json.dumps(python_objects)
            
        except TypeError as e:
            logging.error(f"TypeError detected_objects serileştirirken: {str(e)}")
            logging.error(f"detected_objects türü: {type(detected_objects)}")
            
            # Detaylı hata ayıklama bilgisi
            debug_serialization(detected_objects, "detected_objects")
            
            # Hata durumunda boş bir JSON nesnesi kaydet
            self.detected_objects_json = "{}"
        except Exception as e:
            logging.error(f"Hata detected_objects serileştirirken: {str(e)}")
            self.detected_objects_json = "{}"
    
    def get_detected_objects(self):
        """
        Tespit edilen nesneleri JSON formatından döndürür.
        
        Returns:
            list: Tespit edilen nesnelerin listesi
        """
        if self._detected_objects:
            return json.loads(self._detected_objects)
        return []
    
    @property
    def detected_objects_json(self):
        """Return the detected objects as a JSON string."""
        return self._detected_objects

    @detected_objects_json.setter 
    def detected_objects_json(self, value):
        """Set the detected objects JSON string."""
        self._detected_objects = value
    
    def to_dict(self):
        """
        İçerik tespitini sözlük formatında döndürür.
        
        Returns:
            dict: İçerik tespitinin tüm özellikleriyle sözlük temsili
        """
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'frame_path': self.frame_path,
            'frame_timestamp': self.frame_timestamp,
            'violence_score': self.violence_score,
            'adult_content_score': self.adult_content_score,
            'harassment_score': self.harassment_score,
            'weapon_score': self.weapon_score,
            'drug_score': self.drug_score,
            'detected_objects': self.get_detected_objects()
        }


class AgeEstimation(db.Model):
    """
    Her bir tespit edilen yüz için yaş tahmini sonuçlarını saklar.
    Resim veya video karelerinde tespit edilen yüzlerin konum ve yaş bilgilerini içerir.
    """
    __tablename__ = 'age_estimations'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id'), nullable=False)
    person_id = db.Column(db.String(36), nullable=False)  # Kişi için benzersiz ID
    frame_path = db.Column(db.String(255), nullable=True)
    frame_timestamp = db.Column(db.Float, nullable=False)  # Karenin video içindeki zaman damgası (saniye)
    
    # Yüz konumu 
    _face_location = db.Column(db.String(100))  # JSON olarak saklanır: [x, y, width, height]
    
    # Yaş tahmini
    estimated_age = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float, default=0.0)
    
    # JSON formatında saklanan yaş tahmini verileri
    age_estimations_json = db.Column(db.Text)
    
    def set_face_location(self, x, y, width, height):
        """
        Yüz konumunu ayarlar ve JSON formatında saklar.
        
        Args:
            x: Yüzün sol üst köşesinin x koordinatı
            y: Yüzün sol üst köşesinin y koordinatı
            width: Yüzün genişliği
            height: Yüzün yüksekliği
        """
        # Koordinatların int olduğundan emin ol
        x, y, width, height = int(x), int(y), int(width), int(height)
        self._face_location = json.dumps([x, y, width, height])
    
    def get_face_location(self):
        """
        Yüz konumunu JSON formatından döndürür.
        
        Returns:
            list: [x, y, width, height] formatında yüz konumu
        """
        if self._face_location:
            return json.loads(self._face_location)
        return None
    
    face_location = property(get_face_location, set_face_location)
    
    def set_age_estimations(self, age_estimations):
        """
        Yaş tahminlerini JSON olarak ayarlar. NumPy tiplerini düzgün bir şekilde işler.
        
        Args:
            age_estimations: Yaş tahmini listesi veya sözlüğü
        """
        try:
            from app.utils.serialization_utils import convert_numpy_types_to_python, debug_serialization
            
            # NumPy türlerini Python türlerine dönüştür
            python_objects = convert_numpy_types_to_python(age_estimations)
            
            # JSON'a çevir
            self.age_estimations_json = json.dumps(python_objects)
            
        except TypeError as e:
            logging.error(f"TypeError age_estimations serileştirirken: {str(e)}")
            logging.error(f"age_estimations türü: {type(age_estimations)}")
            
            # Detaylı hata ayıklama bilgisi
            debug_serialization(age_estimations, "age_estimations")
            
            # Hata durumunda boş bir JSON nesnesi kaydet
            self.age_estimations_json = "{}"
        except Exception as e:
            logging.error(f"Hata age_estimations serileştirirken: {str(e)}")
            self.age_estimations_json = "{}"
    
    def get_age_estimations(self):
        """
        Yaş tahminlerini JSON formatından döndürür.
        
        Returns:
            str: Yaş tahminlerinin JSON string formatı
        """
        return self.age_estimations_json
    
    def to_dict(self):
        """
        Yaş tahminini sözlük formatında döndürür.
        
        Returns:
            dict: Yaş tahmininin tüm özellikleriyle sözlük temsili
        """
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'person_id': self.person_id,
            'frame_path': self.frame_path,
            'frame_timestamp': self.frame_timestamp,
            'face_location': self.get_face_location(),
            'estimated_age': self.estimated_age,
            'confidence_score': self.confidence_score,
            'age_estimations': self.get_age_estimations()
        }


class AnalysisFeedback(db.Model):
    """
    Kullanıcılardan gelen analiz geri bildirimleri tablosu.
    Bu, AI modelini geliştirmek için kullanılacaktır.
    """
    __tablename__ = 'analysis_feedbacks'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False)
    analysis = db.relationship('Analysis', backref='feedbacks')
    
    # Geri bildirim verileri
    feedback_type = db.Column(db.String(50), nullable=False)  # correct, false_positive, false_negative
    category = db.Column(db.String(50), nullable=False)  # violence, adult_content, harassment, weapons, substance_abuse
    comment = db.Column(db.Text, nullable=True)
    
    # JSON formatında saklanan tespit ve yaş tahmini verileri
    _detected_objects = db.Column(db.Text)
    _age_estimations = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalysisFeedback {self.id}: {self.feedback_type} - {self.category}>"

    def set_age_estimations(self, age_estimations):
        """
        Yaş tahminlerini JSON string olarak ayarlar.
        """
        try:
            # Yeni utils modülünden convert fonksiyonunu kullan
            converted_estimations = convert_numpy_types_to_python(age_estimations)
            
            # JSON uyumluluğunu test et
            try:
                # Doğrudan JSON serileştirmeyi dene
                json_string = json.dumps(converted_estimations)
                self._age_estimations = json_string
            except TypeError as type_error:
                # Tür hatası durumunda daha fazla hata ayıklama bilgisi
                logger.error(f"JSON serileştirme hatası: {str(type_error)}")
                debug_serialization(age_estimations, "age_estimations")
                
                # Sorunlu objeyi boş liste ile değiştir
                self._age_estimations = "[]"
                raise ValueError(f"Yaş tahminlerini JSON'a dönüştürürken tür hatası: {str(type_error)}")
                
        except Exception as e:
            current_app.logger.error(f"JSON dönüşüm hatası: {str(e)}")
            current_app.logger.error(f"Nesne türü: {type(age_estimations)}")
            current_app.logger.error(f"Nesne içeriği: {str(age_estimations)}")
            # Varsayılan boş değer at
            self._age_estimations = "[]"
            raise ValueError(f"Yaş tahminlerini JSON'a dönüştürürken hata: {str(e)}")

    def to_dict(self):
        """Analiz nesnesini sözlüğe dönüştürür."""
        try:
            detected_objects = json.loads(self._detected_objects) if self._detected_objects else []
        except json.JSONDecodeError:
            logger.error(f"Tespit edilen nesneler JSON decode hatası: {self._detected_objects[:100]}")
            detected_objects = []
            
        try:
            age_estimations = json.loads(self._age_estimations) if self._age_estimations else []
        except json.JSONDecodeError:
            logger.error(f"Yaş tahminleri JSON decode hatası: {self._age_estimations[:100]}")
            age_estimations = []
        
        return {
            'id': self.id,
            'file_id': self.file_id,
            'status': self.status,
            'error': self.error,
            'progress': self.progress,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'violence_score': float(self.violence_score) if self.violence_score is not None else None,
            'weapon_score': float(self.weapon_score) if self.weapon_score is not None else None,
            'adult_score': float(self.adult_score) if self.adult_score is not None else None,
            'substance_score': float(self.substance_score) if self.substance_score is not None else None,
            'harassment_score': float(self.harassment_score) if self.harassment_score is not None else None,
            'overall_score': float(self.overall_score) if self.overall_score is not None else None,
            'detected_objects': detected_objects,
            'age_estimations': age_estimations,
            'is_processed': self.is_processed,
            'processing_seconds': self.processing_seconds
        }

class FaceTracking(db.Model):
    """Görüntülerdeki yüzleri takip eden model."""
    __tablename__ = 'face_tracking'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id'), nullable=False)
    
    person_id = db.Column(db.String(36), nullable=False, unique=True)  # Benzersiz kişi ID'si
    first_appearance = db.Column(db.Float, nullable=False)  # İlk göründüğü zaman (saniye)
    last_appearance = db.Column(db.Float, nullable=False)  # Son göründüğü zaman (saniye)
    
    # En iyi yüz görüntüsü
    best_frame_path = db.Column(db.String(255), nullable=True)
    best_frame_number = db.Column(db.Integer, nullable=True)
    best_frame_confidence = db.Column(db.Float, default=0.0)
    
    # Kişiyle ilgili veri
    total_appearances = db.Column(db.Integer, default=0)  # Toplam göründüğü kare sayısı
    
    # Yaş tahmini (en yüksek güvenilirliğe sahip tahmin)
    best_age_estimation = db.Column(db.Float, nullable=True)
    best_age_confidence = db.Column(db.Float, default=0.0)
    
    # Yüz özellikleri
    face_features = db.Column(db.Text, nullable=True)  # JSON olarak saklanabilir
    
    def __repr__(self):
        return f"<FaceTracking(id={self.id}, person_id='{self.person_id}', analysis_id='{self.analysis_id}')>"