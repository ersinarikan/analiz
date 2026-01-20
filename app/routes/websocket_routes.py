"""
WebSocket Event Handler Routes
Tüm WebSocket event'leri burada yönetilir.
"""

import logging
from datetime import datetime
from flask import request
from flask_socketio import emit, join_room, disconnect

from app.socketio_instance import get_socketio

logger = logging.getLogger(__name__)

# İstemcinin oda katılımı sonrası "hazırım" mesajı bekleniyor
client_analysis_ready = {}

def register_websocket_handlers(socketio_instance):
    """WebSocket handler'larını register eder"""
    # Handler'lar zaten decorator ile tanımlandı, sadece confirm et
    return True

def register_websocket_handlers_in_context(socketio_instance):
    """App_context içinde WebSocket handler'larını register eder"""
    
    # ✅ DIRECT HANDLER REGISTRATION - APP_CONTEXT İÇİNDE
    @socketio_instance.on('connect')
    def handle_connect_in_context():
        """WebSocket bağlantısı kurulduğunda çalışır"""
        try:
            from flask import has_request_context, session
            from flask import current_app
            
            # SocketIO handlers should have a request context, but be defensive
            if not has_request_context():
                return False

            # Reject WS connections when not authenticated (unless auth is disabled)
            app = current_app
            if not app.config.get("WSANALIZ_AUTH_DISABLED", False) and not session.get("pam_user"):
                return False
                
            session_id = request.sid  # type: ignore[attr-defined]
            logger.info(f"📡 WebSocket connected: {session_id}")
            emit('connected', {'status': 'WebSocket bağlantısı başarılı'})
        except Exception as e:
            logger.error(f"Connect handler error: {e}")

    @socketio_instance.on('disconnect')
    def handle_disconnect_in_context():
        """WebSocket bağlantısı kesildiğinde çalışır - cleanup logic ile"""
        try:
            from flask import current_app, has_app_context
            from app import db, global_flask_app
            
            session_id = request.sid  # type: ignore[attr-defined]
            logger.info(f"📡 WebSocket disconnected: {session_id}")
            
            # WebSocket session'ı ile ilişkili çalışan analizleri bul ve iptal et
            try:
                # Prefer Flask context proxy if available; otherwise fall back to global app instance
                if has_app_context():
                    from flask import Flask as _Flask
                    app_obj = getattr(current_app, '_get_current_object', lambda: current_app)()  # type: ignore[attr-defined]
                else:
                    app_obj = global_flask_app

                if app_obj is None:
                    logger.warning(
                        "WebSocket disconnect cleanup: Flask app bulunamadı (no app_context + global_flask_app None). "
                        "Shutdown sırasında normal olabilir; cleanup atlanıyor."
                    )
                    return

                # Validate app_obj looks like a Flask app instance
                try:
                    from flask import Flask as _Flask
                    if not isinstance(app_obj, _Flask):
                        logger.error(
                            f"WebSocket disconnect cleanup: app_obj Flask değil (type={type(app_obj)}). "
                            "Cleanup atlanıyor."
                        )
                        return
                except Exception:
                    pass

                # DB işlemlerini explicit app_context içinde yap
                try:
                    app_ctx = app_obj.app_context()
                except Exception as ctx_err:
                    logger.warning(f"WebSocket disconnect cleanup: app_context oluşturulamadı (muhtemel shutdown). Hata: {ctx_err}")
                    return

                with app_ctx:
                    from app.models.analysis import Analysis
                    from sqlalchemy.exc import OperationalError

                    # 1. Veritabanındaki ilişkili analizleri bul
                    try:
                        active_analyses = Analysis.query.filter(
                            Analysis.websocket_session_id == session_id,
                            Analysis.status.in_(['pending', 'processing'])  # type: ignore[attr-defined]
                        ).all()
                    except OperationalError as op_err:
                        logger.warning(
                            f"WebSocket disconnect cleanup: DB sorgusu çalışmadı (muhtemel schema eksikliği). "
                            f"Session: {session_id}. Hata: {op_err}"
                        )
                        return
                    
                    cancelled_count = 0
                    for analysis in active_analyses:
                        logger.info(f"🚫 WebSocket session {session_id} kesildi - Analiz #{analysis.id} iptal ediliyor")
                        analysis.cancel_analysis("WebSocket bağlantısı kesildi")
                        cancelled_count += 1

                    # Persist cancellations before queue cleanup
                    if cancelled_count > 0:
                        db.session.commit()
                    
                    # 2. Kuyruktaki analizleri de kontrol et
                    from app.services.queue_service import remove_cancelled_from_queue
                    queue_removed = remove_cancelled_from_queue(app=app_obj)
                    
                    if cancelled_count > 0 or queue_removed > 0:
                        total_cancelled = cancelled_count + queue_removed
                        logger.info(
                            f"✅ WebSocket disconnect: {total_cancelled} analiz iptal edildi "
                            f"(DB: {cancelled_count}, Queue: {queue_removed}) (session: {session_id})"
                        )
                    else:
                        logger.info(
                            f"ℹ️ WebSocket disconnect: Bu session ile ilişkili aktif analiz yok (session: {session_id})"
                        )
                    
            except Exception as e:
                logger.error(f"❌ WebSocket disconnect cleanup hatası: {str(e)}", exc_info=True)
                # Rollback must run inside an app context
                try:
                    from flask import current_app as _current_app, has_app_context as _has_app_context
                    if _has_app_context():
                        _app_obj = getattr(_current_app, '_get_current_object', lambda: _current_app)()  # type: ignore[attr-defined]
                    else:
                        from app import global_flask_app as _app_obj

                    if _app_obj is not None:
                        with _app_obj.app_context():
                            try:
                                db.session.rollback()
                            except Exception:
                                pass
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Disconnect handler error: {e}", exc_info=True)

    @socketio_instance.on('ping')
    def handle_ping_in_context(data):
        """Ping-pong test için"""
        try:
            logger.debug(f"🏓 Ping received: {request.sid} - Data: {data}")  # type: ignore[attr-defined]
            
            # timestamp ekle
            pong_data = {
                'message': 'PONG',
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            emit('pong', pong_data)
        except Exception as e:
            logger.error(f"Ping handler error: {e}")

    # JOIN_ANALYSIS handler (Analysis için room katılımı)
    @socketio_instance.on('join_analysis')
    def handle_join_analysis_in_context(data):
        """Analysis room'una katılım"""
        try:
            logger.info(f"📡 JOIN_ANALYSIS: {request.sid} - Data: {data}")  # type: ignore[attr-defined]
            
            if data and 'analysis_id' in data:
                analysis_id = data['analysis_id']
                room = f"analysis_{analysis_id}"
                
                # Room'a katıl
                join_room(room)
                
                logger.debug(f"Client {request.sid} joined room {room}")  # type: ignore[attr-defined]
                
                # Başarı mesajı gönder
                emit('joined_analysis', {
                    'analysis_id': analysis_id,
                    'room': room,
                    'message': f'Analysis {analysis_id} room\'una katıldınız'
                })
            else:
                logger.warning("JOIN_ANALYSIS: analysis_id eksik")
        except Exception as e:
            logger.error(f"Join_analysis handler error: {e}")

    # JOIN_TRAINING handler (Training için room katılımı) 
    @socketio_instance.on('join_training')
    def handle_join_training_in_context(data):
        """Training room'una katılım"""
        try:
            logger.info(f"📡 JOIN_TRAINING: {request.sid} - Data: {data}")  # type: ignore[attr-defined]
            
            if data and 'session_id' in data:
                session_id = data['session_id']
                room = f"training_{session_id}"
                
                # Room'a katıl
                join_room(room)
                
                logger.debug(f"Client {request.sid} joined training room {room}")  # type: ignore[attr-defined]
                
                # Başarı mesajı gönder
                emit('joined_training', {
                    'session_id': session_id,
                    'room': room,
                    'message': f'Training {session_id} room\'una katıldınız'
                })
            else:
                logger.warning("JOIN_TRAINING: session_id eksik")
        except Exception as e:
            logger.error(f"Join_training handler error: {e}")
    
    return True

# EMIT FONKSİYONLARI - Harici kullanım için
def emit_analysis_progress(analysis_id, progress, message="İşleniyor...", file_id=None):
    """Analysis progress event'ini emit eder"""
    try:
        if not analysis_id:
            logger.error("emit_analysis_progress: analysis_id eksik!")
            return False
        
        room = f"analysis_{analysis_id}"
        data = {
            'analysis_id': analysis_id,
            'file_id': file_id,
            'progress': progress,
            'message': message,
            'status': 'processing'
        }
        
        # Centralized SocketIO instance kullan
        try:
            running_socketio = get_socketio()
            if running_socketio is None:
                error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
                logger.error(error_msg)
                return False

            # NOTE: server.manager.get_participants() public API değil ve eventlet altında
            # güvenli olmayabiliyor. Ayrıca progress event'ini iki kez (broadcast + room)
            # emit etmek duplicate/yan etkilere sebep oluyordu.
            # Bu yüzden progress event'ini sadece room'a emit ediyoruz.
            running_socketio.emit('analysis_progress', data, room=room)
            
            logger.info(f"Emit successful: room={room}, data={data}")
            
            return True
            
        except Exception as e:
            error_msg = f"emit_analysis_progress ERROR - analysis_id: {analysis_id}, error: {str(e)}"
            logger.error(error_msg)
            return False
    except Exception as e:
        logger.error(f"emit_analysis_progress OUTER EXCEPTION: {e}")
        return False

def emit_analysis_started(analysis_id, message="Analiz başlatıldı", file_id=None):
    """Analysis started event'ini emit eder"""
    try:
        data = {
            'analysis_id': analysis_id,
            'file_id': file_id,
            'status': 'started',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        running_socketio = get_socketio()
        if running_socketio is None:
            error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
            logger.error(error_msg)
            return False
        
        # Broadcast emit
        running_socketio.emit('analysis_started', data)  # type: ignore[union-attr]
        
        # Room-specific emit
        running_socketio.emit('analysis_started', data, room=f"analysis_{analysis_id}")  # type: ignore[union-attr]
        
        logger.info(f"Analysis started emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_analysis_started error: {e}")
        return False

def emit_analysis_completed(analysis_id, message="Analiz tamamlandı", file_id=None):
    """Analysis completed event'ini emit eder"""
    try:
        data = {
            'analysis_id': analysis_id,
            'file_id': file_id,
            'status': 'completed',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        running_socketio = get_socketio()
        if running_socketio is None:
            error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
            logger.error(error_msg)
            return False
        
        # Sadece room-specific emit (duplicate önlemek için global broadcast kaldırıldı)
        running_socketio.emit('analysis_completed', data, room=f"analysis_{analysis_id}")  # type: ignore[union-attr]
        
        logger.info(f"Analysis completed emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_analysis_completed error: {e}")
        return False

def emit_training_progress(session_id, progress, message="Eğitim devam ediyor...", epoch=None, total_epochs=None, metrics=None):
    """Training progress event'ini emit eder"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'progress': progress,
            'message': message,
            'current_epoch': epoch,
            'total_epochs': total_epochs,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        running_socketio = get_socketio()
        if running_socketio is None:
            error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
            logger.error(error_msg)
            return False
        
        # Broadcast emit
        running_socketio.emit('training_progress', data)  # type: ignore[union-attr]
        # Room-specific emit
        running_socketio.emit('training_progress', data, room=room_name)  # type: ignore[union-attr]
        
        logger.info(f"Training progress emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_training_progress error: {e}")
        return False

def emit_training_started(session_id, model_type=None, sample_count=None, message="Eğitim başlatıldı"):
    """Training started event'ini emit eder"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'status': 'started',
            'model_type': model_type,
            'sample_count': sample_count,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        running_socketio = get_socketio()
        if running_socketio is None:
            error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
            logger.error(error_msg)
            return False
        
        # Broadcast emit
        running_socketio.emit('training_started', data)  # type: ignore[union-attr]
        # Room-specific emit
        running_socketio.emit('training_started', data, room=room_name)  # type: ignore[union-attr]
        logger.info(f"Training started emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_training_started error: {e}")
        return False

def emit_training_completed(session_id, model_path=None, metrics=None, message="Eğitim tamamlandı"):
    """Training completed event'ini emit eder"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'status': 'completed',
            'message': message,
            'model_path': model_path,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        running_socketio = get_socketio()
        if running_socketio is None:
            error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
            logger.error(error_msg)
            return False
        
        # Broadcast emit
        running_socketio.emit('training_completed', data)  # type: ignore[union-attr]
        # Room-specific emit  
        running_socketio.emit('training_completed', data, room=room_name)  # type: ignore[union-attr]
        
        logger.info(f"Training completed emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_training_completed error: {e}")
        return False

def emit_training_error(session_id, error_message, error_details=None):
    """Training error event'ini emit eder"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'status': 'error',
            'error': error_message,
            'error_details': error_details,
            'timestamp': datetime.now().isoformat()
        }
        
        running_socketio = get_socketio()
        if running_socketio is None:
            error_msg = "CRITICAL: get_socketio() None döndürdü! Emit edilemiyor!"
            logger.error(error_msg)
            return False
        
        # Broadcast emit
        running_socketio.emit('training_error', data)  # type: ignore[union-attr]
        # Room-specific emit
        running_socketio.emit('training_error', data, room=room_name)  # type: ignore[union-attr]
        
        logger.info(f"Training error emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_training_error error: {e}")
        return False 