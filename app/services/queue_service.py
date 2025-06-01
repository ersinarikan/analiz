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
    Socket.io aracılığıyla kuyruk durum bilgilerini gönderir
    """
    try:
        from app import socketio
        status = get_queue_status()
        socketio.emit('queue_status', status)
        logger.debug(f"Kuyruk durumu bildirildi: {status}")
    except Exception as e:
        logger.warning(f"Kuyruk durumu bildirimi gönderilemedi: {str(e)}")

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
    """Socket.io analiz durumu bildirimi helper fonksiyonu"""
    try:
        from app import socketio
        socketio.emit('analysis_status_update', {
            'analysis_id': analysis_id,
            'file_id': file_id,
            'status': status,
            'progress': progress,
            'message': message
        })
        
        # Özel olarak durumu da bildirelim (eski client'lar için)
        socketio.emit('analysis_progress', {
            'analysis_id': analysis_id,
            'file_id': file_id,
            'current_frame': 0,
            'total_frames': 100, 
            'progress': progress,
            'message': message
        })
    except Exception as socket_err:
        logger.warning(f"Socket bildirim hatası: {str(socket_err)}")

def _emit_analysis_completion(analysis_id, file_id, success, elapsed_time, message):
    """Socket.io analiz tamamlanma bildirimi helper fonksiyonu"""
    try:
        from app import socketio
        if success:
            # Tamamlandığında klasik completed eventi
            socketio.emit('analysis_completed', {
                'analysis_id': analysis_id,
                'file_id': file_id,
                'elapsed_time': elapsed_time,
                'message': message
            })
            
            # Eski client'lar için analysis_status_update de gönder
            socketio.emit('analysis_status_update', {
                'analysis_id': analysis_id, 
                'file_id': file_id,
                'status': 'completed',
                'progress': 100,
                'message': 'Tamamlandı'
            })
        else:
            # Klasik failed eventi
            socketio.emit('analysis_failed', {
                'analysis_id': analysis_id,
                'file_id': file_id,
                'elapsed_time': elapsed_time,
                'error': message
            })
            
            # Eski client'lar için analysis_status_update de gönder
            socketio.emit('analysis_status_update', {
                'analysis_id': analysis_id,
                'file_id': file_id,
                'status': 'failed',
                'progress': 0,
                'message': message
            })
    except Exception as socket_err:
        logger.warning(f"Socket bildirim hatası: {str(socket_err)}")

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