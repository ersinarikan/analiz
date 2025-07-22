"""
SocketIO instance - Circular import'u önlemek için ayrı dosya
"""
from flask_socketio import SocketIO
 
# Global SocketIO instance holder - ZORUNLU TEK NOKTA!
socketio = None

def get_socketio():
    """
    CRITICAL: Tek global SocketIO instance döndürür
    """
    global socketio
    if socketio is None:
        print("🚨 WARNING: socketio instance henüz set edilmemiş!")
        return None
    print(f"🔥 get_socketio() -> ID: {id(socketio)}")
    return socketio

def set_socketio(socketio_instance):
    """
    CRITICAL: Global SocketIO instance'ını set eder - SADECE BURADA!
    """
    global socketio
    if socketio is not None:
        print(f"🚨 WARNING: socketio instance değiştiriliyor! Eski: {id(socketio)}, Yeni: {id(socketio_instance)}")
    socketio = socketio_instance
    print(f"🔥 set_socketio() -> ID: {id(socketio)}")
    
def reset_socketio():
    """
    Test amaçlı socketio'yu reset eder
    """
    global socketio
    old_id = id(socketio) if socketio else None
    socketio = None
    print(f"🔥 reset_socketio() -> Eski ID: {old_id}") 