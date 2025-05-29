#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Path Utilities
=======================

Bu modül dosya yolu işlemleri için yardımcı fonksiyonları içerir.
Mutlak ve göreceli yol dönüşümleri, güvenlik kontrolleri ve
cross-platform uyumluluk sağlar.
"""

import os
from flask import current_app

# Proje kök dizinini belirle (WSANALIZ ana klasörü)
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def to_rel_path(abs_path):
    """
    Mutlak dosya yolunu proje köküne göre göreceli yola dönüştürür
    
    Windows ve Unix sistemlerde uyumlu çalışması için tüm backslash'leri
    forward slash'e çevirir.
    
    Args:
        abs_path (str): Dönüştürülecek mutlak dosya yolu
        
    Returns:
        str: Proje köküne göre göreceli yol (Unix format)
        
    Example:
        '/home/user/WSANALIZ/storage/uploads/file.jpg' -> 'storage/uploads/file.jpg'
    """
    rel_path = os.path.relpath(abs_path, BASE_DIR)
    return rel_path.replace("\\", "/")

def to_abs_path(rel_path):
    """
    Göreceli dosya yolunu proje köküne göre mutlak yola dönüştürür
    
    Args:
        rel_path (str): Dönüştürülecek göreceli dosya yolu
        
    Returns:
        str: Mutlak dosya yolu
        
    Example:
        'storage/uploads/file.jpg' -> '/home/user/WSANALIZ/storage/uploads/file.jpg'
    """
    return os.path.abspath(os.path.join(BASE_DIR, rel_path))

def is_subpath(path, base=BASE_DIR):
    """
    Verilen yolun proje dizini içinde olup olmadığını kontrol eder
    
    Bu fonksiyon güvenlik amaçlı kullanılır ve path traversal
    saldırılarını önlemeye yardımcı olur.
    
    Args:
        path (str): Kontrol edilecek dosya yolu
        base (str): Base dizin (varsayılan: proje kök dizini)
        
    Returns:
        bool: Yol güvenli dizin içinde mi?
        
    Example:
        is_subpath('storage/uploads/file.jpg') -> True
        is_subpath('../../../etc/passwd') -> False
    """
    return os.path.commonpath([os.path.abspath(path), base]) == base 