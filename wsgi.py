"""
WSGI Configuration for production deployment with Gunicorn + Eventlet
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Eventlet monkey patching for SocketIO (must be done before other imports)
import eventlet
# NOTE: PyTorch/OpenCLIP gibi yoğun native kütüphaneler eventlet'in thread patch'i ile
# uyumsuz olabiliyor ve worker içinde segfault'a yol açabiliyor.
# SocketIO için eventlet async_mode yeterli; thread patch'i kapalı tutuyoruz.
eventlet.monkey_patch(thread=False)

from app import create_app as _create_app, initialize_app

# IMPORTANT:
# Gunicorn master process, worker'ları fork etmeden önce bu modülü import edebilir.
# Eğer burada create_app() çağırıp CUDA/torch init yapılırsa, fork-sonrası worker içinde
# ilk GPU işlemlerinde SIGSEGV (code 139) görülebilir.
#
# Bu yüzden app'i LAZY olarak, worker process içinde ilk istek geldiğinde oluşturuyoruz.
_flask_app = None
_socketio = None


def _ensure_app_initialized():
    global _flask_app, _socketio
    if _flask_app is not None:
        return

    _flask_app, _socketio = _create_app()  # Tuple'ı unpack et

    # Worker içinde app init (DB, klasörler, queue processor vs.)
    try:
        initialize_app(_flask_app)
    except Exception as e:
        import logging
        logger = logging.getLogger("wsanaliz.wsgi")
        logger.warning(f"App initialization warning: {e}")


class _LazyWSGIApp:
    def __call__(self, environ, start_response):
        _ensure_app_initialized()
        return _flask_app(environ, start_response)


# Export app for Gunicorn as a WSGI callable
app = _LazyWSGIApp()

# Export app for Gunicorn
# Gunicorn with eventlet worker will handle SocketIO automatically
# The 'app' object is the Flask WSGI application

if __name__ == "__main__":
    # Direct run (development)
    _ensure_app_initialized()
    _socketio.run(_flask_app, host='0.0.0.0', port=5000)