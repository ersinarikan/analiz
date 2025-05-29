#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Feedback Model
======================

Bu modül kullanıcı geri bildirimlerini ve model eğitimi için 
pseudo-label verilerini saklayan Feedback modelini içerir.
"""

import uuid
from datetime import datetime
from app import db
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, LargeBinary

class Feedback(db.Model):
    """
    İçerik analizlerine ve yaş tahminlerine kullanıcı geri bildirimleri için veritabanı modeli.
    
    Bu model şu tiplerde geri bildirim verilerini saklar:
    - Kullanıcı manuel düzeltmeleri (yaş, içerik kategorileri)
    - Pseudo-label verileri (yüksek güvenli otomatik etiketler)
    - Model eğitimi için training set verileri
    - Geri bildirim durumu ve arşivleme bilgileri
    """
    __tablename__ = 'feedback'
    
    # Birincil anahtar ve zaman damgası
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Görüntü ve yüz tespit bilgileri
    frame_path = db.Column(db.String(1024), nullable=True)  # Analiz edilen karenin dosya yolu
    face_bbox = db.Column(db.String(255), nullable=True)    # Yüz koordinatları "x1,y1,x2,y2" formatında
    embedding = db.Column(db.Text, nullable=True)           # Yüz embedding'i (virgül ayrılmış float değerler)

    # İlişkili analiz bilgileri
    content_id = db.Column(db.String(36), nullable=True)    # İçeriğin benzersiz ID'si
    analysis_id = db.Column(db.String(36), nullable=True)   # Analizin benzersiz ID'si
    
    # Kişi takip bilgileri (video analizlerinde)
    person_id = db.Column(db.String(36), nullable=True, index=True)  # Takip edilen kişinin ID'si
    
    # Yaş tahmin geri bildirimleri
    corrected_age = db.Column(db.Integer, nullable=True)           # Kullanıcının girdiği doğru yaş
    pseudo_label_original_age = db.Column(db.Float, nullable=True) # BuffaloL modelinin tahmin ettiği yaş
    pseudo_label_clip_confidence = db.Column(db.Float, nullable=True) # CLIP güven skoru (pseudo-label için)
    is_age_range_correct = db.Column(db.Boolean, nullable=True)    # Yaş aralığı doğru mu?

    # Geri bildirim meta bilgileri
    feedback_type = db.Column(db.String(50), nullable=True, index=True)   # 'age', 'content', 'age_pseudo' vb.
    feedback_source = db.Column(db.String(50), nullable=True, default='MANUAL_USER', index=True)  
    # Kaynak türleri: 'MANUAL_USER', 'PSEUDO_BUFFALO_HIGH_CONF', 'AUTO_GENERATED' vb.
    
    # Genel değerlendirme
    rating = db.Column(db.Integer, nullable=True)     # 1-5 arası genel puanlama
    comment = db.Column(db.Text, nullable=True)       # Kullanıcı yorumu
    
    # İçerik kategori geri bildirimleri (JSON formatında)
    category_feedback = db.Column(db.JSON, nullable=True)      # Hangi kategorilerde düzeltme yapıldı
    category_correct_values = db.Column(db.JSON, nullable=True) # Düzeltilmiş değerler
    
    # Model eğitimi ve veri yönetimi
    training_status = db.Column(db.String(50), nullable=True, index=True)  # 'used_in_training', 'pending', 'archived'
    used_in_model_version = db.Column(db.String(100), nullable=True)       # Hangi model versiyonunda kullanıldı
    training_used_at = db.Column(db.DateTime, nullable=True)               # Eğitimde kullanma tarihi
    is_archived = db.Column(db.Boolean, default=False, index=True)         # Arşivlenmiş mi?
    archive_reason = db.Column(db.String(100), nullable=True)              # Arşivleme sebebi
    
    def __repr__(self):
        """Model için string temsili"""
        return f"<Feedback(id={self.id}, type='{self.feedback_type}', source='{self.feedback_source}')>"
    
    def to_dict(self):
        """
        Modeli JSON serileştirme için sözlük formatına dönüştürür
        
        Returns:
            dict: Geri bildirim verilerini içeren sözlük
        """
        data = {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'frame_path': self.frame_path,
            'face_bbox': self.face_bbox,
            # Not: embedding büyük olabileceği için to_dict'e dahil edilmez
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