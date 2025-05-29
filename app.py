#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ - Web Tabanlı Yapay Zeka Analiz Sistemi
===============================================

Bu uygulama, görüntü ve video dosyalarında içerik analizi, yaş tahmini ve yüz tanıma
işlemlerini gerçekleştiren Flask tabanlı bir web uygulamasıdır.

Özellikler:
- Video/görüntü içerik analizi (şiddet, yetişkin içerik, taciz, silah, uyuşturucu)
- Yapay zeka destekli yaş tahmini
- CLIP model ile risk skorlaması
- Gerçek zamanlı analiz takibi
- Model eğitimi ve versiyonlama
"""

import sys
import os
import logging
from pathlib import Path

def ensure_virtual_env():
    """Virtual environment kontrolü yapar ve gerekirse kullanıcıyı uyarır"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️ Virtual environment aktif değil!")
        print("💡 Önce virtual environment'ı aktifleştirin:")
        print("   venv\\Scripts\\activate  (Windows)")
        print("   source venv/bin/activate  (Linux/Mac)")
        return False
    return True

# Virtual environment kontrolü
ensure_virtual_env()

# TensorFlow loglarını production seviyesine ayarla
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("⚠️ TensorFlow bulunamadı, devam ediliyor...")

# Flask uygulamasını import et
try:
    from app import create_app, socketio, initialize_app
except ImportError as e:
    print(f"❌ Flask uygulaması import edilemedi: {e}")
    print("💡 Virtual environment'ı aktifleştirip tekrar deneyin:")
    print("   venv\\Scripts\\activate  (Windows)")
    print("   source venv/bin/activate  (Linux/Mac)")
    sys.exit(1)

if __name__ == "__main__":
    try:
        print("🚀 WSANALIZ Flask Uygulaması Başlatılıyor...")
        
        app = create_app()
        initialize_app(app)  # Uygulama başlangıç işlemleri
        
        # Production için log seviyelerini ayarla
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)
        
        print("✅ Uygulama başarıyla başlatıldı!")
        print("🌐 Erişim: http://localhost:5000")
        print("📊 Model Yönetimi: http://localhost:5000/model-management")
        print("🤖 CLIP Monitoring: http://localhost:5000/clip-monitoring")
        print("⏹️  Durdurmak için: Ctrl+C")
        
        # Production modunda debug=False
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        socketio.run(app, debug=debug_mode, host="0.0.0.0", port=5000, log_output=False)
        
    except Exception as e:
        print(f"❌ Uygulama başlatılırken hata: {e}")
        print("💡 Çözüm önerileri:")
        print("   1. Virtual environment'ı aktifleştirin")
        print("   2. Gerekli paketleri yükleyin: pip install -r requirements.txt")
        print("   3. Veya flask run --debug komutunu kullanın")
        sys.exit(1) 