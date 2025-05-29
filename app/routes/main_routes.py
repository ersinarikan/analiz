#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Ana Route'lar
=====================

Bu modül uygulamanın ana sayfaları ve dashboard'ları için
route tanımlarını içerir.
"""

from flask import Blueprint, render_template
from flask import current_app as app

# Ana routes için blueprint
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """
    Ana sayfa - Dosya yükleme ve analiz arayüzü
    
    Kullanıcıların dosya yükleyip analiz başlatabileceği
    ana arayüzü sunar.
    
    Returns:
        Rendered HTML template
    """
    return render_template('index.html')

@bp.route('/metrics')
def metrics():
    """
    Model performans metrikleri sayfası
    
    Model doğruluk oranları, eğitim istatistikleri ve
    sistem performans metriklerini gösterir.
    
    Returns:
        Rendered HTML template
    """
    return render_template('metrics.html')

@bp.route('/model-management')
def model_management():
    """
    Model yönetimi dashboard'u
    
    Model versiyonları, eğitim durumu, model değiştirme
    ve yönetim işlemlerinin yapıldığı sayfa.
    
    Returns:
        Rendered HTML template
    """
    return render_template('model_management.html') 

@bp.route('/clip-monitoring')
def clip_monitoring():
    """
    CLIP model eğitimi izleme dashboard'u
    
    CLIP modelinin eğitim sürecini, loss değerlerini ve
    eğitim ilerlemesini gerçek zamanlı olarak izlemek için.
    
    Returns:
        Rendered HTML template
    """
    return render_template('clip_monitoring.html') 