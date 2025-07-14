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
    HTTP API üzerinden kuyruk durum bilgilerini sağlar
    SocketIO yerine HTTP polling kullanılıyor
    """
    try:
        status = get_queue_status()
        logger.debug(f"Kuyruk durumu mevcut: {status}")
        # HTTP endpoint /api/queue/status üzerinden erişilebilir
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
        # Flask uygulama bağlamını oluştur
        from app import create_app
        app = create_app()
        
        logger.info("Kuyruk işleyici çalışıyor. Uygulama bağlamı oluşturuldu.")
        
        while not analysis_queue.empty():
            # Kuyruk durumu bildirimi gönder
            emit_queue_status()
            
            # Sıradaki analizi al
            analysis_id = analysis_queue.get()
            logger.info(f"Analiz işleme başlıyor: #{analysis_id}, Kalan işler: {analysis_queue.qsize()}")
            
            try:
                # Thread-safe database session kullan
                with database_session(app) as session:
                    # Gerekli modülleri import et
                    from app.models.analysis import Analysis
                    from app.services.analysis_service import analyze_file
                    
                    # Analiz nesnesini al ve durumunu güncelle
                    analysis = Analysis.query.get(analysis_id)
                    if not analysis:
                        logger.error(f"Analiz bulunamadı: {analysis_id}")
                        analysis_queue.task_done()
                        continue
                    
                    # İlk durum güncellemesi
                    analysis.status = 'processing'
                    analysis.status_message = 'Analiz işlemi başlatılıyor...'
                    analysis.progress = 5
                    session.commit()  # İlk commit
                    
                    # Socket bildirim gönder - status update
                    _emit_analysis_status(analysis_id, analysis.file_id, 'processing', 5, 'Analiz işlemi başlatılıyor...')
                    
                # Session bitti, şimdi analizi gerçekleştir (ayrı session'da)
                start_time = time.time()
                success, message = analyze_file(analysis_id)
                elapsed_time = time.time() - start_time
                
                # Sonuç bildirim
                logger.info(f"Analiz #{analysis_id} tamamlandı: {'Başarılı' if success else 'Başarısız'}, "
                           f"Süre: {elapsed_time:.2f}s, Mesaj: {message}")
                
                # Final durumu için yeni session
                analysis_file_id = None
                with database_session(app) as session:
                    analysis = Analysis.query.get(analysis_id)
                    if analysis:
                        analysis_file_id = analysis.file_id  # file_id'yi önceden al
                        if success:
                            # analyze_file zaten status'u 'completed' yapmış olmalı
                            analysis.status_message = message or 'Analiz başarıyla tamamlandı'
                        else:
                            analysis.status = 'failed'
                            analysis.status_message = message or 'Analiz başarısız'
                        session.commit()
                
                # Socket bildirim gönder - completed/failed
                _emit_analysis_completion(analysis_id, analysis_file_id, success, elapsed_time, message)
                
            except Exception as e:
                logger.error(f"Analiz işleme hatası: #{analysis_id}, {str(e)}")
                logger.error(traceback.format_exc())
                
                # Hata durumunda analizi başarısız olarak işaretle - yeni session ile
                try:
                    error_analysis_file_id = None
                    with database_session(app) as session:
                        analysis = Analysis.query.get(analysis_id)
                        if analysis:
                            error_analysis_file_id = analysis.file_id  # file_id'yi önceden al
                            analysis.status = 'failed'
                            analysis.status_message = f"İşlem sırasında hata: {str(e)}"[:250]
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
        logger.error(f"Kuyruk işleyici kritik hatası: {str(e)}")
        logger.error(traceback.format_exc())
        
    finally:
        # İşleme durumunu sıfırla
        with processing_lock:
            is_processing = False
            logger.info("Kuyruk işleyici durduruldu.")
            
            # Eğer kuyrukta hala eleman varsa, yeni bir işleyici başlat
            if not analysis_queue.empty():
                start_processor()

def _emit_analysis_status(analysis_id, file_id, status, progress, message):
    """Analiz durumu - HTTP API üzerinden erişilebilir"""
    try:
        logger.info(f"Analiz durumu güncellendi: {analysis_id} - {status} ({progress}%)")
        # HTTP endpoint /api/analysis/{analysis_id}/status üzerinden erişilebilir
    except Exception as e:
        logger.warning(f"Analiz durumu güncelleme hatası: {str(e)}")

def _emit_analysis_completion(analysis_id, file_id, success, elapsed_time, message):
    """Analiz tamamlanma - HTTP API üzerinden erişilebilir"""
    try:
        status_text = "completed" if success else "failed"
        logger.info(f"Analiz tamamlandı: {analysis_id} - {status_text} ({elapsed_time:.2f}s)")
        # HTTP endpoint /api/analysis/{analysis_id}/result üzerinden erişilebilir
    except Exception as e:
        logger.warning(f"Analiz tamamlanma bildirimi hatası: {str(e)}")

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