import threading
import queue
import logging
import time
import subprocess
import sys
import os
import errno
import fcntl
from flask import current_app
import traceback
from contextlib import contextmanager
from typing import Tuple

logger = logging.getLogger(__name__)

# Queue backend
# - "memory": mevcut in-process queue (tek proses için uygun)
# - "redis": web ve worker proseslerini ayırmak için (önerilen prod)
QUEUE_BACKEND = os.environ.get("WSANALIZ_QUEUE_BACKEND", "redis").strip().lower()
REDIS_URL = os.environ.get("WSANALIZ_REDIS_URL", "redis://localhost:6379/0").strip()
REDIS_QUEUE_KEY = os.environ.get("WSANALIZ_QUEUE_KEY", "wsanaliz:analysis_queue").strip()
REDIS_WORKER_ACTIVE_KEY = os.environ.get("WSANALIZ_WORKER_ACTIVE_KEY", "wsanaliz:worker:active_analyses").strip()
REDIS_WORKER_PROCESSING_KEY = os.environ.get("WSANALIZ_WORKER_PROCESSING_KEY", "wsanaliz:worker:is_processing").strip()
REDIS_WORKER_HEARTBEAT_KEY = os.environ.get("WSANALIZ_WORKER_HEARTBEAT_KEY", "wsanaliz:worker:last_heartbeat").strip()

_redis_client = None


def is_redis_backend() -> bool:
    return QUEUE_BACKEND == "redis"


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # type: ignore

        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        return _redis_client
    except Exception as e:
        raise RuntimeError(f"Redis queue backend seçildi ama redis client init edilemedi: {e}")


def _set_worker_state(is_processing_value: bool, active_analyses: int):
    """Worker state'i Redis'e yazar (queue stats endpoint için)."""
    if not is_redis_backend():
        return
    try:
        r = _get_redis()
        pipe = r.pipeline()
        pipe.set(REDIS_WORKER_PROCESSING_KEY, "1" if is_processing_value else "0", ex=60)
        pipe.set(REDIS_WORKER_ACTIVE_KEY, str(active_analyses), ex=60)
        pipe.set(REDIS_WORKER_HEARTBEAT_KEY, str(time.time()), ex=60)
        pipe.execute()
    except Exception as e:
        logger.warning(f"Worker state Redis'e yazılamadı: {e}")


# Global analiz kuyruğu (memory backend)
analysis_queue = queue.Queue()
# İşleme kilidi (memory backend)
processing_lock = threading.Lock()
is_processing = False

_GPU_LOCK_PATH = os.environ.get("WSANALIZ_GPU_LOCK_PATH", "/tmp/wsanaliz_gpu_analysis.lock")


def _acquire_gpu_lock():
    """
    Cross-process GPU lock.

    Toplu analizde birden fazla Gunicorn worker aynı anda subprocess başlatıp
    CUDA OOM'a neden olabiliyordu. Bu lock, tüm prosesler arasında aynı anda
    sadece 1 analiz subprocess'inin GPU üzerinde çalışmasını sağlar.

    Returns:
        file descriptor (must be kept open to hold the lock)
    """
    fd = os.open(_GPU_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)

    # Non-blocking acquire loop (eventlet uyumlu)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise

            # Lock busy -> yield/sleep
            try:
                import eventlet  # type: ignore

                eventlet.sleep(0.5)
            except Exception:
                time.sleep(0.5)


def _release_gpu_lock(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(fd)
        except Exception:
            pass

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
    if is_redis_backend():
        # Cross-process queue: web sadece enqueue eder; worker BLPOP ile tüketir.
        r = _get_redis()
        r.rpush(REDIS_QUEUE_KEY, str(analysis_id))
        emit_queue_status()
        return

    # Fallback: in-process queue (dev)
    analysis_queue.put(analysis_id)
    start_processor()
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
    if is_redis_backend():
        # Redis backend'te queue processing ayrı worker process'te yapılır.
        logger.info("Redis queue backend aktif: in-process queue processor başlatılmıyor.")
        return
    with processing_lock:
        if not is_processing:
            is_processing = True
            # Gunicorn eventlet worker altında OS thread yerine eventlet greenlet kullanmak daha güvenli.
            # Özellikle SocketIO emit'leri background task içinde çalıştığı için, eventlet hub ile uyumlu olmalı.
            try:
                import eventlet  # type: ignore
                eventlet.spawn_n(process_queue)
                logger.info("Kuyruk işleyici başlatıldı (eventlet greenlet)")
            except Exception:
                thread = threading.Thread(target=process_queue)
                thread.daemon = True
                thread.start()
                logger.info("Kuyruk işleyici başlatıldı (thread)")

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
                analysis_id = analysis_queue.get()
                logger.info(f"Analiz işleme başlıyor: #{analysis_id}, Kalan işler: {analysis_queue.qsize()}")
                try:
                    process_one_analysis(str(analysis_id), global_flask_app)
                finally:
                    analysis_queue.task_done()
                    emit_queue_status()
                    try:
                        import eventlet  # type: ignore
                        eventlet.sleep(1)
                    except Exception:
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


def process_one_analysis(analysis_id: str, app=None) -> Tuple[bool, str]:
    """
    Tek bir analysis_id için analizi çalıştırır (subprocess izolasyonu + GPU lock).
    Hem in-process queue (dev) hem de ayrı worker proses (prod) tarafından kullanılır.
    """
    # Worker state (redis) - best effort
    _set_worker_state(True, 1)

    from app import global_flask_app
    target_app = app or global_flask_app
    if target_app is None:
        raise RuntimeError("Flask app bulunamadı (global_flask_app None). create_app() çağrılmış olmalı.")

    try:
        with target_app.app_context():
            # Analizin varlığını/iptalini kontrol et
            from app.models.analysis import Analysis
            analysis_file_id = None

            with database_session(target_app) as session:
                analysis = Analysis.query.get(analysis_id)
                if not analysis:
                    logger.error(f"Analiz bulunamadı: {analysis_id}")
                    return False, "Analiz bulunamadı"
                analysis_file_id = analysis.file_id
                if getattr(analysis, "is_cancelled", False):
                    logger.info(f"🚫 Analiz #{analysis_id} iptal edilmiş, atlanıyor")
                    return False, "Analiz iptal edildi"

                # Mark as processing + set start_time for observability.
                try:
                    if getattr(analysis, "status", None) != "processing":
                        analysis.status = "processing"
                    if not getattr(analysis, "start_time", None):
                        from datetime import datetime
                        analysis.start_time = datetime.utcnow()
                except Exception:
                    pass

            start_time = time.time()
            gpu_lock_fd = None
            try:
                logger.info(f"🔒 GPU lock bekleniyor (analysis_id={analysis_id})")
                gpu_lock_fd = _acquire_gpu_lock()
                logger.info(f"🔓 GPU lock alındı (analysis_id={analysis_id})")

                # NOTE:
                # subprocess.run(...) blocks for the whole duration of the analysis.
                # Our worker heartbeat keys in Redis have TTL (ex=60). For longer analyses (videos),
                # the keys expire and /api/queue/stats shows worker_last_heartbeat=null although
                # the worker is alive and processing. Use Popen + poll loop to refresh heartbeat.
                logs_dir = os.environ.get("WSANALIZ_SUBPROCESS_LOG_DIR", "/opt/wsanaliz/logs")
                os.makedirs(logs_dir, exist_ok=True)
                stdout_path = os.path.join(logs_dir, f"analysis_subprocess_{analysis_id}.stdout.log")
                stderr_path = os.path.join(logs_dir, f"analysis_subprocess_{analysis_id}.stderr.log")

                proc = subprocess.Popen(
                    [sys.executable, "-m", "app.services.analysis_subprocess_runner", str(analysis_id)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                stdout_lines: list[str] = []
                stderr_lines: list[str] = []
                stdout_file = None
                stderr_file = None

                try:
                    stdout_file = open(stdout_path, "w", encoding="utf-8")
                    stderr_file = open(stderr_path, "w", encoding="utf-8")
                except Exception:
                    stdout_file = None
                    stderr_file = None

                def _stream_reader(stream, lines, file_obj):
                    try:
                        for line in iter(stream.readline, ""):
                            lines.append(line.rstrip("\n"))
                            if file_obj:
                                file_obj.write(line)
                                file_obj.flush()
                    except Exception:
                        pass
                    finally:
                        try:
                            stream.close()
                        except Exception:
                            pass

                if proc.stdout is not None:
                    threading.Thread(
                        target=_stream_reader,
                        args=(proc.stdout, stdout_lines, stdout_file),
                        daemon=True,
                    ).start()
                if proc.stderr is not None:
                    threading.Thread(
                        target=_stream_reader,
                        args=(proc.stderr, stderr_lines, stderr_file),
                        daemon=True,
                    ).start()

                start_wait = time.time()
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break

                    # Refresh heartbeat while processing (best effort)
                    _set_worker_state(True, 1)

                    # Hard timeout (1 hour)
                    if (time.time() - start_wait) > (60 * 60):
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise subprocess.TimeoutExpired(cmd=proc.args, timeout=60 * 60)

                    time.sleep(5)

                # Ensure process is fully terminated and streams drained
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass

                # NOTE: Some native deps (e.g. insightface/onnxruntime) can print to stdout,
                # which can break the "last-line-is-JSON" assumption. Parse the LAST valid JSON
                # object we can find in stdout, and fall back to stderr if needed.
                # We already streamed outputs into stdout_lines/stderr_lines.

                try:
                    if stdout_file:
                        stdout_file.flush()
                        stdout_file.close()
                    if stderr_file:
                        stderr_file.flush()
                        stderr_file.close()
                except Exception:
                    pass

                def _extract_last_json(lines: list[str]):
                    import json as _json

                    last_obj = None
                    for line in lines:
                        line_s = (line or "").strip()
                        if not line_s:
                            continue
                        if not (line_s.startswith("{") and line_s.endswith("}")):
                            continue
                        try:
                            obj = _json.loads(line_s)
                            if isinstance(obj, dict) and ("success" in obj or "message" in obj):
                                last_obj = obj
                        except Exception:
                            continue
                    return last_obj

                out = _extract_last_json(stdout_lines) or _extract_last_json(stderr_lines)
                if out is not None:
                    success = bool(out.get("success", False))
                    message = str(out.get("message", ""))
                else:
                    # Give a helpful debug snippet (last non-empty line).
                    last_line = ""
                    for line in reversed(stdout_lines):
                        if (line or "").strip():
                            last_line = (line or "").strip()
                            break
                    if not last_line:
                        for line in reversed(stderr_lines):
                            if (line or "").strip():
                                last_line = (line or "").strip()
                                break
                    success = False
                    message = f"Subprocess çıktı parse edilemedi (rc={proc.returncode}). Son satır: {last_line[:400]}"
            except subprocess.TimeoutExpired:
                success = False
                message = "Analiz subprocess timeout (1 saat)"
            except Exception as sub_err:
                success = False
                message = f"Analiz subprocess başlatılamadı: {sub_err}"
            finally:
                if gpu_lock_fd is not None:
                    _release_gpu_lock(gpu_lock_fd)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Analiz #{analysis_id} tamamlandı: {'Başarılı' if success else 'Başarısız'}, "
                f"Süre: {elapsed_time:.2f}s, Mesaj: {message}"
            )

            # Final durumu güncelle
            with database_session(target_app) as session:
                analysis = Analysis.query.get(analysis_id)
                if analysis:
                    from datetime import datetime

                    if success:
                        analysis.status = "completed"
                    else:
                        analysis.status = "failed"
                        # keep message for UI/debugging
                        try:
                            analysis.error_message = message
                        except Exception:
                            pass

                    if not getattr(analysis, "end_time", None):
                        analysis.end_time = datetime.utcnow()

            _emit_analysis_completion(analysis_id, analysis_file_id, success, elapsed_time, message)
            return success, message

    except Exception as e:
        logger.error(f"Analiz işleme hatası: #{analysis_id}, {str(e)}")
        logger.error(traceback.format_exc())
        try:
            from app.models.analysis import Analysis
            with database_session(target_app) as session:
                analysis = Analysis.query.get(analysis_id)
                error_analysis_file_id = analysis.file_id if analysis else None
                if analysis:
                    from datetime import datetime
                    analysis.status = 'failed'
                    try:
                        analysis.error_message = f"İşlem hatası: {str(e)}"
                    except Exception:
                        pass
                    if not getattr(analysis, "end_time", None):
                        analysis.end_time = datetime.utcnow()
            _emit_analysis_completion(analysis_id, error_analysis_file_id, False, 0, f"İşlem hatası: {str(e)}")
        except Exception as db_err:
            logger.error(f"Hata durumunda DB güncelleme hatası: {str(db_err)}")
        return False, str(e)
    finally:
        _set_worker_state(False, 0)

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

def remove_cancelled_from_queue(app=None):
    """
    Kuyruktaki iptal edilmiş analizleri temizler
    
    Returns:
        int: Temizlenen analiz sayısı
    """
    try:
        # Prefer existing Flask app context if available; otherwise use provided app or global fallback.
        from flask import current_app as _current_app, has_app_context
        target_app = None
        if has_app_context():
            target_app = _current_app
        else:
            try:
                from app import global_flask_app as _global_flask_app
            except Exception:
                _global_flask_app = None
            target_app = app or _global_flask_app

        if target_app is None:
            logger.warning("remove_cancelled_from_queue: Flask app bulunamadı (no app_context, app param None, global_flask_app None)")
            return 0

        if is_redis_backend():
            # Redis list üzerinde basit bir filtreleme (küçük kuyruklarda yeterli)
            from app.models.analysis import Analysis
            r = _get_redis()
            removed_count = 0

            if has_app_context():
                items = r.lrange(REDIS_QUEUE_KEY, 0, -1) or []
                kept = []
                for analysis_id in items:
                    analysis = Analysis.query.get(analysis_id)
                    if analysis and analysis.is_cancelled:
                        removed_count += 1
                    else:
                        kept.append(analysis_id)

                if removed_count:
                    pipe = r.pipeline()
                    pipe.delete(REDIS_QUEUE_KEY)
                    if kept:
                        pipe.rpush(REDIS_QUEUE_KEY, *kept)
                    pipe.execute()
            else:
                with target_app.app_context():
                    items = r.lrange(REDIS_QUEUE_KEY, 0, -1) or []
                    kept = []
                    for analysis_id in items:
                        analysis = Analysis.query.get(analysis_id)
                        if analysis and analysis.is_cancelled:
                            removed_count += 1
                        else:
                            kept.append(analysis_id)

                    if removed_count:
                        pipe = r.pipeline()
                        pipe.delete(REDIS_QUEUE_KEY)
                        if kept:
                            pipe.rpush(REDIS_QUEUE_KEY, *kept)
                        pipe.execute()

            if removed_count:
                logger.info(f"✅ Redis kuyruğundan {removed_count} iptal edilmiş analiz temizlendi")
            return removed_count

        from app.models.analysis import Analysis
        
        removed_count = 0
        temp_queue = queue.Queue()
        
        # Kuyruktaki tüm analizleri kontrol et
        if has_app_context():
            while not analysis_queue.empty():
                try:
                    analysis_id = analysis_queue.get_nowait()
                    
                    # Analizin iptal edilip edilmediğini kontrol et
                    analysis = Analysis.query.get(analysis_id)
                    if analysis and analysis.is_cancelled:
                        logger.info(f"🗑️ Kuyruktan iptal edilmiş analiz temizlendi: #{analysis_id}")
                        removed_count += 1
                    else:
                        # İptal edilmemişse geri kuyruğa koy
                        temp_queue.put(analysis_id)
                        
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Kuyruk temizleme hatası: {str(e)}")
                    break
            
            # Temizlenmiş kuyruğu geri yükle
            while not temp_queue.empty():
                analysis_queue.put(temp_queue.get())
        else:
            with target_app.app_context():
                while not analysis_queue.empty():
                    try:
                        analysis_id = analysis_queue.get_nowait()

                        # Analizin iptal edilip edilmediğini kontrol et
                        analysis = Analysis.query.get(analysis_id)
                        if analysis and analysis.is_cancelled:
                            logger.info(f"🗑️ Kuyruktan iptal edilmiş analiz temizlendi: #{analysis_id}")
                            removed_count += 1
                        else:
                            # İptal edilmemişse geri kuyruğa koy
                            temp_queue.put(analysis_id)

                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Kuyruk temizleme hatası: {str(e)}")
                        break

                # Temizlenmiş kuyruğu geri yükle
                while not temp_queue.empty():
                    analysis_queue.put(temp_queue.get())
        
        if removed_count > 0:
            logger.info(f"✅ Kuyruktan {removed_count} iptal edilmiş analiz temizlendi")
            
        return removed_count
        
    except Exception as e:
        logger.error(f"❌ Kuyruk temizleme hatası: {str(e)}")
        return 0

def get_queue_status():
    """
    Kuyruk durumu bilgilerini döndürür
    
    Returns:
        dict: Kuyruk durum bilgileri
    """
    if is_redis_backend():
        try:
            r = _get_redis()
            qsize = int(r.llen(REDIS_QUEUE_KEY) or 0)
            is_proc = (r.get(REDIS_WORKER_PROCESSING_KEY) or "0") == "1"
            return {
                'queue_size': qsize,
                'is_processing': is_proc,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"Redis queue status okunamadı: {e}")
            return {
                'queue_size': 0,
                'is_processing': False,
                'timestamp': time.time(),
                'error': str(e)
            }
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
    if is_redis_backend():
        try:
            r = _get_redis()
            qsize = int(r.llen(REDIS_QUEUE_KEY) or 0)
            is_proc = (r.get(REDIS_WORKER_PROCESSING_KEY) or "0") == "1"
            active = int(r.get(REDIS_WORKER_ACTIVE_KEY) or "0")
            heartbeat = r.get(REDIS_WORKER_HEARTBEAT_KEY)
            return {
                'queue_size': qsize,
                'is_processing': is_proc,
                'active_analyses': active,
                'worker_last_heartbeat': float(heartbeat) if heartbeat else None,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"Redis queue stats okunamadı: {e}")
            return {
                'queue_size': 0,
                'is_processing': False,
                'active_analyses': 0,
                'timestamp': time.time(),
                'error': str(e)
            }
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

        if is_redis_backend():
            # Redis backend'te cleanup worker prosesin sorumluluğunda.
            logger.info("Redis queue backend aktif: in-process cleanup atlandı.")
            return
        
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

    if is_redis_backend():
        try:
            r = _get_redis()
            # Del -> önce uzunluğu al
            cleared_count = int(r.llen(REDIS_QUEUE_KEY) or 0)
            r.delete(REDIS_QUEUE_KEY)
            _set_worker_state(False, 0)
            logger.info(f"Redis kuyruğu temizlendi: {cleared_count} analiz silindi")
            return cleared_count
        except Exception as e:
            logger.error(f"Redis kuyruğu temizlenemedi: {e}")
            return 0
    
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