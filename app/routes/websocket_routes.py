"""
WebSocket Event Handler Routes
Tüm WebSocket event'leri burada yönetilir.
"""

import logging
from datetime import datetime
from flask import request
from flask_socketio import emit, join_room, disconnect
from threading import Timer

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
            logger.info(f"📡 WebSocket connected (in-context): {request.sid}")
            # Standard emit kullan
            emit('connected', {'status': 'WebSocket bağlantısı başarılı (in-context)'})
        except Exception as e:
            logger.error(f"In-context connect handler error: {e}")

    @socketio_instance.on('disconnect')
    def handle_disconnect_in_context():
        """WebSocket bağlantısı kesildiğinde çalışır"""
        try:
            logger.info(f"📡 WebSocket disconnected (in-context): {request.sid}")
        except Exception as e:
            logger.error(f"In-context disconnect handler error: {e}")

    @socketio_instance.on('ping')
    def handle_ping_in_context(data):
        """Ping-pong test için"""
        try:
            logger.debug(f"🏓 Ping received (in-context): {request.sid} - Data: {data}")
            
            # timestamp ekle
            pong_data = {
                'message': 'PONG from in-context handler',
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'handler': 'in-context'
            }
            
            emit('pong', pong_data)
        except Exception as e:
            logger.error(f"In-context ping handler error: {e}")

    # JOIN_ANALYSIS handler (Analysis için room katılımı)
    @socketio_instance.on('join_analysis')
    def handle_join_analysis_in_context(data):
        """Analysis room'una katılım"""
        try:
            logger.info(f"📡 JOIN_ANALYSIS (in-context): {request.sid} - Data: {data}")
            
            if data and 'analysis_id' in data:
                analysis_id = data['analysis_id']
                room = f"analysis_{analysis_id}"
                
                # Room'a katıl
                join_room(room)
                
                # Başarı mesajı gönder
                emit('joined_analysis', {
                    'analysis_id': analysis_id,
                    'room': room,
                    'message': f'Analysis {analysis_id} room\'una katıldınız (in-context)',
                    'handler': 'in-context'
                })
            else:
                logger.warning("JOIN_ANALYSIS: analysis_id eksik")
        except Exception as e:
            logger.error(f"In-context join_analysis handler error: {e}")

    # JOIN_TRAINING handler (Training için room katılımı) 
    @socketio_instance.on('join_training')
    def handle_join_training_in_context(data):
        """Training room'una katılım"""
        try:
            logger.info(f"📡 JOIN_TRAINING (in-context): {request.sid} - Data: {data}")
            
            if data and 'session_id' in data:
                session_id = data['session_id']
                room = f"training_{session_id}"
                
                # Room'a katıl
                join_room(room)
                
                # Başarı mesajı gönder
                emit('joined_training', {
                    'session_id': session_id,
                    'room': room,
                    'message': f'Training {session_id} room\'una katıldınız (in-context)',
                    'handler': 'in-context'
                })
            else:
                logger.warning("JOIN_TRAINING: session_id eksik")
        except Exception as e:
            logger.error(f"In-context join_training handler error: {e}")
    
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
            
            # Room membership kontrol et
            try:
                room_members = running_socketio.server.manager.get_participants(namespace='/', room=room)
                room_members_list = list(room_members)
                logger.debug(f"Room {room} members: {room_members_list}")
            except Exception as room_err:
                logger.debug(f"Room membership check failed: {room_err}")
            
            # Broadcast emit (tüm connected clientlara)
            running_socketio.emit('analysis_progress', data)
            logger.debug("BROADCAST emit completed")
            
            # Room-specific emit 
            running_socketio.emit('analysis_progress', data, room=room)
            
            logger.info(f"Emit successful with centralized socketio, data sent: {data}")
            logger.info(f"Emit successful, data sent: {data}")
            logger.info(f"Room {room} - emit completed")
            
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
        
        # Broadcast emit
        running_socketio.emit('analysis_started', data)
        
        # Room-specific emit
        running_socketio.emit('analysis_started', data, room=f"analysis_{analysis_id}")
        
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
        
        # Sadece room-specific emit (duplicate önlemek için global broadcast kaldırıldı)
        running_socketio.emit('analysis_completed', data, room=f"analysis_{analysis_id}")
        
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
        
        # Broadcast emit
        running_socketio.emit('training_progress', data)
        # Room-specific emit
        running_socketio.emit('training_progress', data, room=room_name)
        
        logger.info(f"Training progress emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_training_progress error: {e}")
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
        
        # Broadcast emit
        running_socketio.emit('training_completed', data)
        # Room-specific emit  
        running_socketio.emit('training_completed', data, room=room_name)
        
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
        
        # Broadcast emit
        running_socketio.emit('training_error', data)
        # Room-specific emit
        running_socketio.emit('training_error', data, room=room_name)
        
        logger.info(f"Training error emit successful: {data}")
        return True
    except Exception as e:
        logger.error(f"emit_training_error error: {e}")
        return False 