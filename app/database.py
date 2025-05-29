#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Veritabanı Yardımcıları
===============================

Bu modül veritabanı işlemleri için yardımcı fonksiyonları içerir.
"""

from flask_sqlalchemy import SQLAlchemy

# Global veritabanı instance'ı
db = SQLAlchemy()

def init_db(app):
    """
    Veritabanını Flask uygulaması ile başlatır
    
    Args:
        app: Flask uygulaması
    """
    db.init_app(app)

def reset_db():
    """Veritabanını sıfırlar."""
    db.drop_all()
    db.create_all()

def get_engine():
    """SQLAlchemy engine'i döndürür."""
    return db.engine 