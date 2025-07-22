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

print("🔥🔥🔥 WEBSOCKET_ROUTES.PY IMPORTED! STANDARD DECORATOR PATTERN AKTİF!")

# İstemcinin oda katılımı sonrası "hazırım" mesajı bekleniyor
client_analysis_ready = {}

def register_websocket_handlers(socketio_instance):
    """WebSocket handler'larını register eder"""
    print(f"🔥🔥🔥 REGISTERING WEBSOCKET HANDLERS - Standard decorator pattern")
    print(f"🔥🔥🔥 SocketIO instance: {socketio_instance}")
    
    # Handler'lar zaten decorator ile tanımlandı, sadece confirm et
    print("🔥🔥🔥 Standard decorator handlers are now active!")
    return True

def register_websocket_handlers_in_context(socketio_instance):
    """App_context içinde WebSocket handler'larını register eder"""
    print(f"🔥🔥🔥 REGISTERING IN CONTEXT - SocketIO instance: {socketio_instance}")
    print(f"🔥🔥🔥 REGISTERING IN CONTEXT - Instance type: {type(socketio_instance)}")
    
    # ✅ DIRECT HANDLER REGISTRATION - APP_CONTEXT İÇİNDE
    @socketio_instance.on('connect')
    def handle_connect_in_context():
        """WebSocket bağlantısı kurulduğunda çalışır"""
        try:
            print(f"🔥🔥🔥 IN-CONTEXT CONNECT HANDLER - Session ID: {request.sid}")
            logger.info(f"📡 WebSocket connected (in-context): {request.sid}")
            
            # Standard emit kullan
            emit('connected', {'status': 'WebSocket bağlantısı başarılı (in-context)'})
            print(f"🔥🔥🔥 IN-CONTEXT CONNECTED EVENT SENT!")
            
        except Exception as e:
            print(f"🔥🔥🔥 IN-CONTEXT CONNECT ERROR: {e}")
            logger.error(f"In-context connect handler error: {e}")

    @socketio_instance.on('disconnect')
    def handle_disconnect_in_context():
        """WebSocket bağlantısı kesildiğinde çalışır"""
        try:
            print(f"🔥🔥🔥 IN-CONTEXT DISCONNECT HANDLER - Session ID: {request.sid}")
            logger.info(f"📡 WebSocket disconnected (in-context): {request.sid}")
        except Exception as e:
            print(f"🔥🔥🔥 IN-CONTEXT DISCONNECT ERROR: {e}")
            logger.error(f"In-context disconnect handler error: {e}")

    @socketio_instance.on('ping')
    def handle_ping_in_context(data):
        """Ping event'ini işler"""
        try:
            print(f"🔥🔥🔥 IN-CONTEXT PING HANDLER - Session ID: {request.sid}")
            print(f"🔥🔥🔥 IN-CONTEXT PING DATA: {data}")
            logger.info(f"📡 Ping received (in-context) from {request.sid}: {data}")
            
            # Standard emit kullan
            emit('pong', {
                'response': 'pong', 
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'source': 'in-context-handler'
            })
            print(f"🔥🔥🔥 IN-CONTEXT PONG EVENT SENT!")
            
        except Exception as e:
            print(f"🔥🔥🔥 IN-CONTEXT PING ERROR: {e}")
            logger.error(f"In-context ping handler error: {e}")

    # @socketio_instance.on('join_analysis')  # DEVRE DIŞI - Minimal handler kullanılıyor
    def handle_join_analysis_in_context(data):
        """Analiz room'una katılır - DEVRE DIŞI"""
        print("=" * 80)
        print("❌❌❌ IN-CONTEXT HANDLER ÇAĞRILDI - BU OLMAMALI!")
        print(f"⚠️ ESKI HANDLER ÇAĞRILDI! Bu çalışmamalı!")
        print("=" * 80)
        return  # Erken çıkış
        
        try:
            print(f"🔥��🔥 IN-CONTEXT JOIN_ANALYSIS - Session ID: {request.sid}")
            print(f"🔥🔥🔥 IN-CONTEXT JOIN_ANALYSIS DATA: {data}")
            
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
                    'source': 'in-context-handler'
                })
                print(f"🔥🔥🔥 IN-CONTEXT JOINED_ANALYSIS EVENT SENT for room {room}")
                logger.info(f"📡 Client {request.sid} joined analysis room (in-context): {room}")
            else:
                print(f"🔥🔥🔥 IN-CONTEXT JOIN_ANALYSIS: No analysis_id in data")
                
        except Exception as e:
            print(f"🔥🔥🔥 IN-CONTEXT JOIN_ANALYSIS ERROR: {e}")
            logger.error(f"In-context join analysis handler error: {e}")

    @socketio_instance.on('join_training')
    def handle_join_training_in_context(data):
        """Training room'una katılır"""
        try:
            print(f"🔥🔥🔥 IN-CONTEXT JOIN_TRAINING - Session ID: {request.sid}")
            print(f"🔥🔥🔥 IN-CONTEXT JOIN_TRAINING DATA: {data}")
            
            if data and 'session_id' in data:
                training_session_id = data['session_id']
                room = f"training_{training_session_id}"
                
                # Room'a katıl
                join_room(room)
                
                # Başarı mesajı gönder
                emit('joined_training', {
                    'session_id': training_session_id,
                    'room': room,
                    'message': f'Training {training_session_id} room\'una katıldınız (in-context)',
                    'source': 'in-context-handler'
                })
                print(f"🔥🔥🔥 IN-CONTEXT JOINED_TRAINING EVENT SENT for room {room}")
                logger.info(f"📡 Client {request.sid} joined training room (in-context): {room}")
            else:
                print(f"🔥🔥🔥 IN-CONTEXT JOIN_TRAINING: No session_id in data")
                
        except Exception as e:
            print(f"🔥🔥🔥 IN-CONTEXT JOIN_TRAINING ERROR: {e}")
            logger.error(f"In-context join training handler error: {e}")
    
    print("🔥🔥🔥 IN-CONTEXT HANDLERS REGISTERED SUCCESSFULLY!")
    return True

# ===============================
# 🎯 EMIT FUNCTIONS - Diğer modüller için
# ===============================
def emit_analysis_progress(analysis_id, progress, message, status='processing'):
    """
    WebSocket üzerinden analiz progress'ini emit eder
    """
    print(f"🔥🔥🔥 [DEBUG] emit_analysis_progress STARTED - analysis_id: {analysis_id}, progress: {progress}")
    logger.info(f"🔥🔥🔥 [DEBUG] emit_analysis_progress STARTED - analysis_id: {analysis_id}, progress: {progress}")
    
    room = f"analysis_{analysis_id}"
    data = {
        'analysis_id': analysis_id,
        'progress': progress,
        'message': message,
        'status': status
    }
    
    try:
        logger.info(f"🔥 Emitting analysis_progress to room {room}: {progress}% - {message}")
        print(f"🔥 Emitting analysis_progress to room {room}: {progress}% - {message}")
        
        print(f"🔥🔥🔥 [DEBUG] About to find running socketio...")
        
        # CRITICAL: Centralized SocketIO instance kullan!
        running_socketio = get_socketio()
        
        if running_socketio is None:
            print(f"🚨 CRITICAL ERROR: No global socketio instance set!")
            logger.error(f"🚨 CRITICAL ERROR: No global socketio instance set!")
            return
            
        print(f"🔥 FOUND centralized socketio: {running_socketio} (ID: {id(running_socketio)})")
        
        print(f"🔥🔥🔥 [DEBUG] running_socketio check: {running_socketio}")
        print(f"🔥🔥🔥 [DEBUG] type of running_socketio: {type(running_socketio)}")
        print(f"🔥🔥🔥 [DEBUG] ENTERING running_socketio block!")
        
        # Room members kontrolü - DEBUG amaçlı
        try:
            room_members = running_socketio.server.manager.get_participants(namespace='/', room=room)
            room_members_list = list(room_members)
            print(f"🔥 DEBUG: Room {room} members: {room_members_list}")
            logger.info(f"🔥 DEBUG: Room {room} members: {room_members_list}")
        except Exception as room_err:
            print(f"🔥 DEBUG: Room membership check failed: {room_err}")
        
        # Broadcast emit (no room - for testing)
        print(f"🔥 DEBUG: About to emit broadcast analysis_progress...")
        get_socketio().emit('analysis_progress', data)
        print(f"🔥 DEBUG: BROADCAST emit (no room) completed")
        logger.info(f"🔥 DEBUG: BROADCAST emit completed, data: {data}")
        
        # Room emit (targeted)
        print(f"🔥 DEBUG: About to emit room analysis_progress to {room}...")
        get_socketio().emit('analysis_progress', data, room=room)
        
        print(f"🔥 SUCCESS: Used CENTRALIZED socketio! (ID: {id(running_socketio)})")
        logger.info(f"🔥 Emit successful with centralized socketio, data sent: {data}")
        
        print(f"🔥 Room {room} - emit completed with centralized socketio")
        logger.info(f"🔥 Emit successful, data sent: {data}")
        print(f"🔥 Emit successful, data sent: {data}")
        
        logger.info(f"🔥 Room {room} - emit completed")
        print(f"🔥 Room {room} - emit completed")
        
    except Exception as e:
        error_msg = f"WebSocket emit hatası: {str(e)}"
        logger.error(error_msg)
        print(f"🔥 ERROR: {error_msg}")
        print(f"🔥 EXCEPTION: {e}")

def emit_analysis_completed(analysis_id, message):
    socketio = get_socketio()
    data = {
        'analysis_id': analysis_id,
        'message': message,
        'status': 'completed'
    }
    
    # 🔥 BROADCAST emit - Room join sorunları için
    print(f"🔥 Emitting BROADCAST analysis_completed...")
    socketio.emit('analysis_completed', data)
    print(f"🔥 BROADCAST analysis_completed completed")
    
    # Room emit (eski mantık - ek güvenlik için)
    print(f"🔥 Emitting analysis_completed to room analysis_{analysis_id}...")
    socketio.emit('analysis_completed', data, room=f"analysis_{analysis_id}")
    print(f"🔥 Room analysis_completed completed")

def emit_training_progress(session_id, progress, message, status='training'):
    """Training ilerlemesini WebSocket ile bildirir"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'progress': progress,
            'message': message,
            'status': status
        }
        
        logger.info(f"🔥 Emitting training_progress to room {room_name}: {progress}% - {message}")
        print(f"🔥 Emitting training_progress to room {room_name}: {progress}% - {message}")
        
        # SocketIO instance ile emit
        get_socketio().emit('training_progress', data, room=room_name)
        
        logger.info(f"🔥 Training progress emit successful: {data}")
        print(f"🔥 Training progress emit successful: {data}")
        
    except Exception as e:
        logger.error(f"❌ Training progress emit hatası: {str(e)}")
        print(f"❌ Training progress emit hatası: {str(e)}")

def emit_training_completed(session_id, results):
    """Training tamamlandığını WebSocket ile bildirir"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'status': 'completed',
            'results': results
        }
        
        logger.info(f"🔥 Emitting training_completed to room {room_name}")
        print(f"🔥 Emitting training_completed to room {room_name}")
        
        # SocketIO instance ile emit
        get_socketio().emit('training_completed', data, room=room_name)
        
        logger.info(f"🔥 Training completed emit successful: {data}")
        print(f"🔥 Training completed emit successful: {data}")
        
    except Exception as e:
        logger.error(f"❌ Training completed emit hatası: {str(e)}")
        print(f"❌ Training completed emit hatası: {str(e)}")

def emit_training_error(session_id, error_message):
    """Training hatasını WebSocket ile bildirir"""
    try:
        room_name = f"training_{session_id}"
        data = {
            'session_id': session_id,
            'status': 'error',
            'error': error_message
        }
        
        logger.info(f"🔥 Emitting training_error to room {room_name}: {error_message}")
        print(f"🔥 Emitting training_error to room {room_name}: {error_message}")
        
        # SocketIO instance ile emit
        get_socketio().emit('training_error', data, room=room_name)
        
        logger.info(f"🔥 Training error emit successful: {data}")
        print(f"🔥 Training error emit successful: {data}")
        
    except Exception as e:
        logger.error(f"❌ Training error emit hatası: {str(e)}")
        print(f"❌ Training error emit hatası: {str(e)}") 