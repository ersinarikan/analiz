#!/usr/bin/env python3
"""
WSANALIZ Flask Application Entry Point
"""
import eventlet
eventlet.monkey_patch()
import sys
import os
import logging
import signal
import threading
import atexit

# Thread-safe logging lock
_log_lock = threading.Lock()

logger = logging.getLogger("wsanaliz.app")
logging.basicConfig(level=logging.INFO)

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO ve WARNING loglarını gizle

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Sadece ERROR loglarını göster
except ImportError:
    tf = None
    logger.warning("TensorFlow bulunamadı, devam ediliyor...")

# Flask uygulamasını import et
try:
    from app import create_app, initialize_app, socketio
except ImportError as import_err:
    create_app = None
    initialize_app = None
    socketio = None
    logger.error(f"Flask uygulaması import edilemedi: {import_err}")
    logger.error("Virtual environment'ı aktifleştirip tekrar deneyin:")
    logger.error("   venv\\Scripts\\activate  (Windows)")
    logger.error("   source venv/bin/activate  (Linux/Mac)")
    sys.exit(1)

PID_FILE = "wsanaliz.pid"

def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_pid():
    try:
        os.remove(PID_FILE)
    except Exception:
        pass

def signal_handler(_signum, _frame):
    """Graceful shutdown handler"""
    logger.info("Shutdown signal alındı...")
    try:
        # Background services'ları kapat
        logger.info("Background servisler kapatılıyor...")
        # Queue service'yi kapat
        try:
            from app.services.queue_service import cleanup_queue_service
            cleanup_queue_service()
        except Exception as queue_err:
            logger.warning(f"Queue service kapatma hatası: {queue_err}")
        # Memory cleanup
        logger.info("Memory cleanup yapılıyor...")
        import gc
        gc.collect()
        logger.info("Graceful shutdown tamamlandı!")
    except Exception as shutdown_err:
        logger.error(f"Shutdown sırasında hata: {shutdown_err}", exc_info=True)
    finally:
        sys.exit(0)  # Force exit

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "-stop":
            from app.utils.restart_helper import stop_app
            stop_app()
        elif sys.argv[1] == "-start":
            from app.utils.restart_helper import start_app
            start_app()
        else:
            print("Kullanım: python app.py [-stop|-start]")
    else:
        # Signal handlers ekle
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Terminal
        
        try:
            write_pid()
            logger.info("WSANALIZ Flask Uygulaması Başlatılıyor...")
            
            app = create_app()
            initialize_app(app)  # Sadece ana süreçte çalıştırılacak
            
            # Werkzeug HTTP request loglarını kapat
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            # Environment'a göre debug mode belirle
            environment = os.environ.get('FLASK_ENV', 'development')
            is_debug = environment == 'development'
            
            if is_debug:
                logger.info("Development mode: Debug ve auto-reload aktif")
            else:
                logger.info("Production mode: Debug kapalı, performans optimized")
            
            logger.info("Uygulama başarıyla başlatıldı!")
            logger.info("Erişim: http://localhost:5000")
            logger.info("Model Yönetimi: http://localhost:5000/model-management")
            logger.info("CLIP Monitoring: http://localhost:5000/clip-monitoring")
            logger.info("Durdurmak için: Ctrl+C")
            
            # SocketIO server başlatılıyor - temiz WebSocket sistemi
            if socketio:
                logger.info("SocketIO server başlatılıyor...")
                socketio.run(app, debug=is_debug, host="0.0.0.0", port=5000)
            else:
                logger.warning("SocketIO bulunamadı, normal Flask server kullanılıyor")
                app.run(debug=is_debug, host="0.0.0.0", port=5000)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt alındı...")
            signal_handler(None, None)
        except Exception as main_err:
            logger.error(f"Uygulama başlatılırken hata: {main_err}", exc_info=True)
            logger.error("Çözüm önerileri:")
            logger.error("   1. Virtual environment'ı aktifleştirin")
            logger.error("   2. Gerekli paketleri yükleyin: pip install -r requirements.txt")
            logger.error("   3. Veya flask run --debug komutunu kullanın")
            sys.exit(1) 