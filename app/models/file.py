#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ File Model
==================

Bu modül sistemde analiz edilecek dosyaları temsil eden File modelini içerir.
Resim ve video dosyalarının metadata bilgilerini saklar.
"""

import os
import datetime
from app import db
from sqlalchemy import func, String

class File(db.Model):
    """
    Sistem tarafından analiz edilecek dosyaları temsil eden veritabanı modeli.
    
    Bu model şu bilgileri saklar:
    - Dosya adı ve orijinal adı
    - Dosya yolu ve boyutu  
    - MIME tipi ve dosya türü
    - Yükleme tarihi
    - İlişkili analizler
    """
    
    __tablename__ = 'files'
    
    # Birincil anahtar
    id = db.Column(db.Integer, primary_key=True)
    
    # Dosya bilgileri
    filename = db.Column(db.String(255), nullable=False)  # Sistemde saklanan dosya adı
    original_filename = db.Column(db.String(255), nullable=False)  # Kullanıcının yüklediği orijinal ad
    file_path = db.Column(db.String(512), nullable=False, unique=True)  # Dosyanın tam yolu
    file_size = db.Column(db.Integer, nullable=False)  # Dosya boyutu (bytes)
    
    # İçerik bilgileri
    mime_type = db.Column(db.String(128), nullable=False)  # application/json, image/jpeg vb.
    file_type = db.Column(db.String(10), nullable=False)  # 'image', 'video' veya 'unknown'
    
    # Zaman damgaları
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)  # Yükleme tarihi
    
    # Kullanıcı ilişkisi (gelecekte kullanım için)
    user_id = db.Column(db.Integer, nullable=True)  # Dosyayı yükleyen kullanıcı ID'si
    
    # Analiz ilişkileri - bir dosyanın birden fazla analizi olabilir
    analyses = db.relationship('Analysis', 
                              foreign_keys='Analysis.file_id', 
                              lazy='dynamic',
                              primaryjoin="func.cast(File.id, String) == Analysis.file_id",
                              back_populates="file")
    
    def __init__(self, filename, original_filename, file_path, file_size, mime_type, user_id=None):
        """
        Yeni bir File instance'ı oluşturur
        
        Args:
            filename: Sistemde saklanan dosya adı
            original_filename: Kullanıcının yüklediği orijinal dosya adı
            file_path: Dosyanın disk üzerindeki tam yolu
            file_size: Dosya boyutu (bytes cinsinden)
            mime_type: Dosyanın MIME tipi
            user_id: Dosyayı yükleyen kullanıcının ID'si (opsiyonel)
        """
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.mime_type = mime_type
        self.user_id = user_id
        
        # MIME tipine göre dosya türünü otomatik belirle
        if mime_type.startswith('image/'):
            self.file_type = 'image'
        elif mime_type.startswith('video/'):
            self.file_type = 'video'
        else:
            self.file_type = 'unknown'
    
    def to_dict(self):
        """
        Modeli JSON serileştirme için sözlük formatına dönüştürür
        
        Returns:
            dict: Dosya bilgilerini içeren sözlük
        """
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'file_type': self.file_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'has_analysis': self.analyses.count() > 0,  # Bu dosyanın analizi var mı?
            'user_id': self.user_id
        } 