import threading
import queue
import logging
import time
from flask import current_app
import traceback

logger = logging.getLogger(__name__)

# Global analiz kuyruğu
analysis_queue = queue.Queue()
# İşleme kilidi
processing_lock = threading.Lock()
is_processing = False

def add_to_queue(analysis_id):
    """
    Analiz işini kuyruğa ekler ve işleyiciyi başlatır
    
    Args:
        analysis_id: Eklenecek analiz ID'si
    """
    analysis_queue.put(analysis_id)
    logger.info(f"Analiz #{analysis_id} kuyruğa eklendi. Kuyruk boyutu: {analysis_queue.qsize()}")
    
    # Kuyruk durumu bildirimi gönder
    emit_queue_status()
    
    # İşleyici başlat
    start_processor()

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
    Kuyruktan sırayla analiz işlerini işler
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
                with app.app_context():
                    # Gerekli modülleri import et
                    from app import db
                    from app.models.analysis import Analysis
                    from app.services.analysis_service import analyze_file
                    
                    # Analiz nesnesini al ve durumunu güncelle
                    analysis = Analysis.query.get(analysis_id)
                    if not analysis:
                        logger.error(f"Analiz bulunamadı: {analysis_id}")
                        analysis_queue.task_done()
                        continue
                    
                    # DB session'ı temizle (önceki işlemlerden kalan bağlantıları kapat)
                    db.session.close()
                    
                    # Yeni bir transaction başlat
                    analysis = Analysis.query.get(analysis_id)
                    analysis.status = 'processing'
                    analysis.status_message = 'Analiz işlemi başlatılıyor...'
                    analysis.progress = 5
                    db.session.commit()
                    
                    # Socket bildirim gönder - status update
                    try:
                        from app import socketio
                        socketio.emit('analysis_status_update', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id,
                            'status': 'processing',
                            'progress': 5,
                            'message': 'Analiz işlemi başlatılıyor...'
                        })
                        
                        # Özel olarak durumu da bildirelim (eski client'lar için)
                        socketio.emit('analysis_progress', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id,
                            'current_frame': 0,
                            'total_frames': 100, 
                            'progress': 5,
                            'message': 'Analiz başlatılıyor'
                        })
                    except Exception as socket_err:
                        logger.warning(f"Socket bildirim hatası: {str(socket_err)}")
                    
                    # Analizi gerçekleştir
                    start_time = time.time()
                    success, message = analyze_file(analysis_id)
                    elapsed_time = time.time() - start_time
                    
                    # Sonuç bildirim
                    logger.info(f"Analiz #{analysis_id} tamamlandı: {'Başarılı' if success else 'Başarısız'}, "
                               f"Süre: {elapsed_time:.2f}s, Mesaj: {message}")
                    
                    # Socket bildirim gönder - completed/failed
                    try:
                        from app import socketio
                        if success:
                            # Tamamlandığında klasik completed eventi
                            socketio.emit('analysis_completed', {
                                'analysis_id': analysis_id,
                                'file_id': analysis.file_id,
                                'elapsed_time': elapsed_time,
                                'message': message
                            })
                            
                            # Eski client'lar için analysis_status_update de gönder
                            socketio.emit('analysis_status_update', {
                                'analysis_id': analysis_id, 
                                'file_id': analysis.file_id,
                                'status': 'completed',
                                'progress': 100,
                                'message': 'Tamamlandı'
                            })
                        else:
                            # Klasik failed eventi
                            socketio.emit('analysis_failed', {
                                'analysis_id': analysis_id,
                                'file_id': analysis.file_id,
                                'elapsed_time': elapsed_time,
                                'error': message
                            })
                            
                            # Eski client'lar için analysis_status_update de gönder
                            socketio.emit('analysis_status_update', {
                                'analysis_id': analysis_id,
                                'file_id': analysis.file_id,
                                'status': 'failed',
                                'progress': 0,
                                'message': message
                            })
                    except Exception as socket_err:
                        logger.warning(f"Socket bildirim hatası: {str(socket_err)}")
                
            except Exception as e:
                logger.error(f"Analiz işleme hatası: #{analysis_id}, {str(e)}")
                logger.error(traceback.format_exc())
                
                # Hata durumunda analizi başarısız olarak işaretle
                try:
                    with app.app_context():
                        from app import db
                        from app.models.analysis import Analysis
                        
                        analysis = Analysis.query.get(analysis_id)
                        if analysis:
                            analysis.status = 'failed'
                            analysis.status_message = f"İşlem sırasında hata: {str(e)}"[:250]
                            db.session.commit()
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

def get_queue_status():
    """
    Kuyruk durumunu döndürür
    
    Returns:
        dict: Kuyruk durumu bilgileri
    """
    return {
        'active': is_processing,
        'size': analysis_queue.qsize(),
        'empty': analysis_queue.empty()
    } 