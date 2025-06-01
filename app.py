#!/usr/bin/env python3
"""
WSANALIZ Flask Application Entry Point
"""

import sys
import os
import logging

# Virtual environment kontrolü ve aktivasyonu
def ensure_virtual_env():
    """Virtual environment'ın aktif olduğundan emin ol"""
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    
    # Virtual environment var mı kontrol et
    if os.path.exists(venv_path):
        # Windows için Scripts, Linux/Mac için bin
        if os.name == 'nt':  # Windows
            activate_script = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:  # Linux/Mac
            activate_script = os.path.join(venv_path, 'bin', 'python')
        
        # Eğer virtual environment'daki Python kullanılmıyorsa
        if sys.executable != activate_script and os.path.exists(activate_script):
            print(f"🔄 Virtual environment Python'ı kullanılıyor: {activate_script}")
            # Virtual environment'daki Python ile yeniden çalıştır
            os.execv(activate_script, [activate_script] + sys.argv)

# Virtual environment kontrolü
ensure_virtual_env()

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO ve WARNING loglarını gizle

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Sadece ERROR loglarını göster
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
        initialize_app(app)  # Sadece ana süreçte çalıştırılacak
        
        # Werkzeug HTTP request loglarını kapat
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        print("✅ Uygulama başarıyla başlatıldı!")
        print("🌐 Erişim: http://localhost:5000")
        print("📊 Model Yönetimi: http://localhost:5000/model-management")
        print("🤖 CLIP Monitoring: http://localhost:5000/clip-monitoring")
        print("⏹️  Durdurmak için: Ctrl+C")
        
        socketio.run(app, debug=False, host="0.0.0.0", port=5000, log_output=False)
        
    except Exception as e:
        print(f"❌ Uygulama başlatılırken hata: {e}")
        print("💡 Çözüm önerileri:")
        print("   1. Virtual environment'ı aktifleştirin")
        print("   2. Gerekli paketleri yükleyin: pip install -r requirements.txt")
        print("   3. Veya flask run --debug komutunu kullanın")
        sys.exit(1) 