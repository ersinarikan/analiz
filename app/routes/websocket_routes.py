"""
WebSocket event handlers - Temiz ve minimal implementasyon
"""
import logging
from flask_socketio import emit, join_room, leave_room, disconnect
from app import socketio

logger = logging.getLogger(__name__)

@socketio.on('connect')
def handle_connect():
    """Client bağlandığında"""
    logger.info(f"WebSocket client connected")
    emit('connected', {'status': 'WebSocket bağlantısı başarılı'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client bağlantısı kesildiğinde"""
    logger.info(f"WebSocket client disconnected")

@socketio.on('ping')
def handle_ping():
    """Ping-pong test"""
    logger.info("WebSocket ping received")
    emit('pong', {'message': 'WebSocket çalışıyor'})

@socketio.on('join_analysis')
def handle_join_analysis(data):
    """Analiz odasına katıl"""
    analysis_id = data.get('analysis_id')
    if analysis_id:
        room = f"analysis_{analysis_id}"
        join_room(room)
        logger.info(f"Client joined analysis room: {room}")
        emit('joined_analysis', {'analysis_id': analysis_id, 'room': room})

@socketio.on('join_training')  
def handle_join_training(data):
    """Eğitim odasına katıl"""
    session_id = data.get('session_id')
    if session_id:
        room = f"training_{session_id}"
        join_room(room)
        logger.info(f"Client joined training room: {room}")
        emit('joined_training', {'session_id': session_id, 'room': room})

# Utility fonksiyonlar
def emit_analysis_progress(analysis_id, progress, message, status='processing'):
    """Analiz progress'i client'lara gönder"""
    room = f"analysis_{analysis_id}"
    data = {
        'analysis_id': analysis_id,
        'progress': progress,
        'message': message,
        'status': status
    }
    socketio.emit('analysis_progress', data, room=room)
    logger.info(f"Analysis progress emitted to {room}: {progress}%")

def emit_analysis_completed(analysis_id, message='Analiz tamamlandı'):
    """Analiz tamamlandı bildirimi"""
    room = f"analysis_{analysis_id}"
    data = {
        'analysis_id': analysis_id,
        'status': 'completed',
        'progress': 100,
        'message': message
    }
    socketio.emit('analysis_completed', data, room=room)
    logger.info(f"Analysis completed emitted to {room}")

def emit_training_started(session_id, model_type, total_samples):
    """Eğitim başlatıldı bildirimi"""
    room = f"training_{session_id}"
    data = {
        'session_id': session_id,
        'model_type': model_type,
        'status': 'started',
        'total_samples': total_samples,
        'message': f'{model_type.upper()} model eğitimi başlatıldı'
    }
    socketio.emit('training_started', data, room=room)
    logger.info(f"Training started emitted to {room}: {model_type} model")

def emit_training_progress(session_id, epoch, total_epochs, metrics=None):
    """Eğitim progress'i gönder"""
    room = f"training_{session_id}"
    data = {
        'session_id': session_id,
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'progress': (epoch / total_epochs) * 100 if total_epochs > 0 else 0,
        'metrics': metrics or {}
    }
    socketio.emit('training_progress', data, room=room)
    logger.info(f"Training progress emitted to {room}: {epoch}/{total_epochs}")

def emit_training_completed(session_id, model_version, metrics=None):
    """Eğitim tamamlandı bildirimi"""
    room = f"training_{session_id}"
    data = {
        'session_id': session_id,
        'status': 'completed',
        'model_version': model_version,
        'metrics': metrics or {}
    }
    socketio.emit('training_completed', data, room=room)
    logger.info(f"Training completed emitted to {room}: {model_version}")

def emit_training_error(session_id, error_message):
    """Eğitim hatası bildirimi"""
    room = f"training_{session_id}"
    data = {
        'session_id': session_id,
        'status': 'error',
        'error': error_message
    }
    socketio.emit('training_error', data, room=room)
    logger.error(f"Training error emitted to {room}: {error_message}") 