import threading
import queue
import logging
import time
from flask import current_app

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
    
    # İşleyici başlat
    start_processor()

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
        from app import db
        from app.models.analysis import Analysis
        from app.services.analysis_service import analyze_file
        
        while not analysis_queue.empty():
            # Sıradaki analizi al
            analysis_id = analysis_queue.get()
            
            try:
                logger.info(f"Analiz işleme başlıyor: #{analysis_id}")
                
                # Analiz nesnesini al ve durumunu güncelle
                with current_app.app_context():
                    analysis = Analysis.query.get(analysis_id)
                    if not analysis:
                        logger.error(f"Analiz bulunamadı: {analysis_id}")
                        continue
                    
                    analysis.status = 'processing'
                    analysis.status_message = 'Analiz işlemi başlatılıyor...'
                    analysis.progress = 5
                    db.session.commit()
                
                # Socket bildirim gönder
                try:
                    from app import socketio
                    socketio.emit('analysis_status_update', {
                        'analysis_id': analysis_id,
                        'status': 'processing',
                        'progress': 5,
                        'message': 'Analiz işlemi başlatılıyor...'
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
                
                # Socket bildirim gönder
                try:
                    from app import socketio
                    if success:
                        socketio.emit('analysis_completed', {
                            'analysis_id': analysis_id,
                            'elapsed_time': elapsed_time,
                            'message': message
                        })
                    else:
                        socketio.emit('analysis_failed', {
                            'analysis_id': analysis_id,
                            'elapsed_time': elapsed_time,
                            'error': message
                        })
                except Exception as socket_err:
                    logger.warning(f"Socket bildirim hatası: {str(socket_err)}")
                
            except Exception as e:
                logger.error(f"Analiz işleme hatası: #{analysis_id}, {str(e)}")
            finally:
                # Kuyruk işlemi tamamlandı
                analysis_queue.task_done()
        
        logger.info("Tüm analizler tamamlandı")
    except Exception as e:
        logger.error(f"Kuyruk işleyici hatası: {str(e)}")
    finally:
        # İşleme durumunu sıfırla
        with processing_lock:
            is_processing = False

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