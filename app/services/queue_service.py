import threading
import queue
import logging
import time
from flask import current_app
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global analiz kuyruğu
analysis_queue = queue.Queue()
# İşleme kilidi
processing_lock = threading.Lock()
is_processing = False

@contextmanager
def database_session(app):
    """
    Thread-safe database session context manager
    Her thread için ayrı session yönetimi sağlar
    """
    try:
        with app.app_context():
            from app import db
            
            # Yeni session başlat
            session = db.session
            
            # İşlem başlangıcında session'ı temizle
            session.rollback()  # Önceki işlemlerden kalan uncommitted changes'i temizle
            session.close()     # Connection pool'a geri döndür
            
            # Fresh session başlat
            yield session
            
            # Başarılı işlem sonrası commit
            session.commit()
            
    except Exception as e:
        # Hata durumunda rollback
        try:
            session.rollback()
            logger.error(f"Database session rollback yapıldı: {str(e)}")
        except:
            pass
        raise
        
    finally:
        # Her durumda session'ı temizle
        try:
            session.close()
        except:
            pass

def add_to_queue(analysis_id):
    """
    Analiz işini kuyruğa ekler ve işleyiciyi başlatır
    
    Args:
        analysis_id: Eklenecek analiz ID'si
    """
    logger.info(f"Analiz kuyruğa ekleniyor: {analysis_id}")
    analysis_queue.put(analysis_id)
    
    # Kuyruk işleyiciyi başlat
    start_processor()
    
    # Kuyruk durumu bildirimi gönder  
    emit_queue_status()

def emit_queue_status():
    """
    WebSocket ile kuyruk durum bilgilerini istemcilere gönderir
    """
    try:
        status = get_queue_status()
        
        # WebSocket ile kuyruk durumu bildirimi gönder
        try:
            from app.socketio_instance import get_socketio
            running_socketio = get_socketio()
            if running_socketio:
                running_socketio.emit('queue_status', status)
                logger.debug(f"Kuyruk durumu WebSocket ile gönderildi: {status}")
        except Exception as ws_err:
            logger.warning(f"WebSocket kuyruk durumu bildirimi hatası: {str(ws_err)}")
            
        logger.debug(f"Kuyruk durumu mevcut: {status}")
        # HTTP endpoint /api/queue/status hala mevcut
    except Exception as e:
        logger.warning(f"Kuyruk durumu güncellemesi hatası: {str(e)}")

def start_processor():
    """
    Kuyruk işleyici thread'i başlatır (henüz çalışmıyorsa)
    """
    global is_processing
    with processing_lock:
        if not is_processing:
            is_processing = True
            thread = threading.Thread(target=process_queue)
            thread.daemon = True
            thread.start()
            logger.info("Kuyruk işleyici başlatıldı")

def process_queue():
    """
    Kuyruktan sırayla analiz işlerini işler - Thread-safe database management ile
    """
    global is_processing
    
    try:
        # Ana Flask app'i globalden al ve context aç
        from app import global_flask_app, db
        from app.socketio_instance import get_socketio
        logger.info("Kuyruk işleyici çalışıyor. Global Flask app context açılıyor.")
        with global_flask_app.app_context():
            while not analysis_queue.empty():
                # Kuyruk durumu bildirimi gönder
                emit_queue_status()
                # Sıradaki analizi al
                analysis_id = analysis_queue.get()
                logger.info(f"Analiz işleme başlıyor: #{analysis_id}, Kalan işler: {analysis_queue.qsize()}")
                try:
                    # Thread-safe database session kullan
                    with database_session(global_flask_app) as session:
                        from app.models.analysis import Analysis
                        from app.services.analysis_service import analyze_file
                        analysis = Analysis.query.get(analysis_id)
                        if not analysis:
                            logger.error(f"Analiz bulunamadı: {analysis_id}")
                            analysis_queue.task_done()
                            continue
                        logger.info(f"Analiz #{analysis_id} kuyruğa alındı, status: {analysis.status}")
                    # Session bitti, şimdi analizi gerçekleştir (ayrı session'da)
                    start_time = time.time()
                    success, message = analyze_file(analysis_id)
                    elapsed_time = time.time() - start_time
                    
                    # Sonuç bildirim
                    logger.info(f"Analiz #{analysis_id} tamamlandı: {'Başarılı' if success else 'Başarısız'}, "
                               f"Süre: {elapsed_time:.2f}s, Mesaj: {message}")
                    
                    # Final durumu için yeni session
                    analysis_file_id = None
                    with database_session(global_flask_app) as session:
                        analysis = Analysis.query.get(analysis_id)
                        if analysis:
                            analysis_file_id = analysis.file_id  # file_id'yi önceden al
                            if success:
                                # analyze_file zaten status'u 'completed' yapmış olmalı
                                pass  # WebSocket ile bildirim gönderilecek
                            else:
                                analysis.status = 'failed'
                            session.commit()
                    
                    # Socket bildirim gönder - completed/failed
                    _emit_analysis_completion(analysis_id, analysis_file_id, success, elapsed_time, message)
                    
                except Exception as e:
                    logger.error(f"Analiz işleme hatası: #{analysis_id}, {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Hata durumunda analizi başarısız olarak işaretle - yeni session ile
                    try:
                        error_analysis_file_id = None
                        with database_session(global_flask_app) as session:
                            analysis = Analysis.query.get(analysis_id)
                            if analysis:
                                error_analysis_file_id = analysis.file_id  # file_id'yi önceden al
                                analysis.status = 'failed'
                                session.commit()
                                
                            # Hata bildirimi
                            _emit_analysis_completion(analysis_id, error_analysis_file_id, 
                                                    False, 0, f"İşlem hatası: {str(e)}")
                            
                    except Exception as db_err:
                        logger.error(f"Hata durumunda DB güncelleme hatası: {str(db_err)}")
                        
                finally:
                    # Kuyruk işlemi tamamlandı
                    analysis_queue.task_done()
                    logger.info(f"Analiz #{analysis_id} işlemi tamamlandı ve kuyruktan çıkarıldı.")
                    
                    # Kuyruk durumu bildirimi gönder
                    emit_queue_status()
                    
                    # Gecikmeli olarak bir sonraki analizi başlat (DB'nin nefes alması için)
                    time.sleep(1)
            
            logger.info("Tüm analizler tamamlandı, kuyruk boş.")
            
            # Son kuyruk durumu bildirimi
            emit_queue_status()
            
    except Exception as e:
        logger.error(f"Kuyruk işleyici genel hatası: {str(e)}", exc_info=True)
        
    finally:
        # İşleme durumunu sıfırla
        with processing_lock:
            is_processing = False
            logger.info("Kuyruk işleyici durduruldu.")
            
            # Eğer kuyrukta hala eleman varsa, yeni bir işleyici başlat
            if not analysis_queue.empty():
                start_processor()

def _emit_analysis_status(analysis_id, file_id, status, progress, message):
    """Analiz durumu WebSocket bildirimi (eski fonksiyon - artık kullanılmıyor)"""
    try:
        # Bu fonksiyon artık kullanılmıyor - yeni WebSocket sistem aktif
        logger.info(f"Analiz durumu güncellendi: {analysis_id} - {status} ({progress}%)")
        
    except Exception as e:
        logger.warning(f"Analiz durumu güncelleme hatası: {str(e)}")

def _emit_analysis_completion(analysis_id, file_id, success, elapsed_time, message):
    """Analiz tamamlanma WebSocket bildirimi"""
    try:
        from app.routes.websocket_routes import emit_analysis_completed
        status_text = "completed" if success else "failed"
        final_message = f"Analiz {status_text} ({elapsed_time:.2f}s): {message}"
        
        emit_analysis_completed(analysis_id, final_message, file_id)
        logger.info(f"Analiz tamamlandı - WebSocket bildirimi: {analysis_id} - {status_text} ({elapsed_time:.2f}s)")
        
    except Exception as e:
        logger.warning(f"Analiz tamamlanma WebSocket bildirimi hatası: {str(e)}")

def get_queue_status():
    """
    Kuyruk durumu bilgilerini döndürür
    
    Returns:
        dict: Kuyruk durum bilgileri
    """
    return {
        'queue_size': analysis_queue.qsize(),
        'is_processing': is_processing,
        'timestamp': time.time()
    }

def get_queue_stats():
    """
    Kuyruk istatistiklerini döndürür
    
    Returns:
        dict: Kuyruk istatistikleri
    """
    return {
        'queue_size': analysis_queue.qsize(),
        'is_processing': is_processing,
        'active_analyses': 1 if is_processing else 0,
        'timestamp': time.time()
    }

def cleanup_queue_service():
    """
    Queue service'yi temizle ve background thread'leri durdur
    """
    global is_processing
    
    try:
        logger.info("🧹 Queue service cleanup başlatılıyor...")
        
        # İşleme durumunu durdur
        with processing_lock:
            is_processing = False
            
        # Kuyruktaki bekleyen işleri temizle
        while not analysis_queue.empty():
            try:
                analysis_id = analysis_queue.get_nowait()
                logger.info(f"Kuyruktan temizlenen analiz: {analysis_id}")
                analysis_queue.task_done()
            except queue.Empty:
                break
                
        logger.info("✅ Queue service cleanup tamamlandı!")
        
    except Exception as e:
        logger.error(f"⚠️ Queue service cleanup hatası: {e}") 

def clear_queue():
    """Kuyruktaki tüm analizleri temizle"""
    global analysis_queue, is_processing
    
    cleared_count = 0
    
    # Önce işleme durduralım
    with processing_lock:
        is_processing = False
        
        # Kuyrukta bekleyen tüm analizleri temizle
        try:
            while True:
                analysis_queue.get_nowait()
                analysis_queue.task_done()
                cleared_count += 1
        except queue.Empty:
            pass
    
    logger.info(f"Kuyruk temizlendi: {cleared_count} analiz silindi")
    return cleared_count