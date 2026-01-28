"""
WSGI Configuration for production deployment with Gunicorn + Eventlet
"""
import os 
import sys 

# ERSIN Proje dizinini Python path'ine ekle
sys .path .insert (0 ,os .path .dirname (__file__ ))

# ERSIN Eventlet monkey patching SocketIO için, diğer import'lardan önce yapılmalı
import eventlet 
# ERSIN Not: PyTorch/OpenCLIP gibi yoğun native kütüphaneler eventlet'in thread patch'i ile uyumsuz olabiliyor
# ERSIN Worker içinde segfault'a yol açabiliyor, SocketIO için eventlet async_mode yeterli, thread patch'i kapalı tutuyoruz
eventlet .monkey_patch (thread =False )

from app import create_app as _create_app ,initialize_app 

# ERSIN Önemli: Gunicorn master process worker'ları fork etmeden önce bu modülü import edebilir
# ERSIN Eğer burada create_app() çağırıp CUDA/torch init yapılırsa, fork-sonrası worker içinde ilk GPU işlemlerinde SIGSEGV (code 139) görülebilir
# ERSIN Bu yüzden app'i LAZY olarak, worker process içinde ilk istek geldiğinde oluşturuyoruz
# ERSIN ANCAK: DB tabloları queue worker subprocess'te analiz yaparken gerekli, bu yüzden DB init'i hemen yapıyoruz
_flask_app =None 
_socketio =None 
_db_initialized =False 


def _ensure_app_initialized ():
    global _flask_app ,_socketio ,_db_initialized
    if _flask_app is not None :
        return 

    _flask_app ,_socketio =_create_app (return_socketio =True )# ERSIN Tuple'ı unpack et

    # ERSIN Worker içinde app init (DB, klasörler, queue processor vs.)
    try :
        initialize_app (_flask_app )
        _db_initialized =True 
    except Exception as e :
        import logging 
        logger =logging .getLogger ("wsanaliz.wsgi")
        logger .warning (f"App initialization warning: {e }")


# ERSIN Modül yüklendiğinde DB'yi hemen initialize et (queue worker için gerekli)
# ERSIN Model yükleme lazy kalır, sadece DB tabloları oluşturulur
try :
    _ensure_app_initialized ()
except Exception as e :
    import logging 
    logger =logging .getLogger ("wsanaliz.wsgi")
    logger .warning (f"Early DB initialization warning (will retry on first request): {e }")


class _LazyWSGIApp :
    def __call__ (self ,environ ,start_response ):
        _ensure_app_initialized ()
        assert _flask_app is not None ,"Flask app must be initialized"
        return _flask_app (environ ,start_response )


        # ERSIN Gunicorn için WSGI callable olarak app export et
app =_LazyWSGIApp ()

# ERSIN Gunicorn için app export et
# ERSIN Eventlet worker ile Gunicorn SocketIO'yu otomatik handle eder
# ERSIN 'app' objesi Flask WSGI uygulamasıdır

if __name__ =="__main__":
# ERSIN Direct run, development için
    _ensure_app_initialized ()
    if _socketio is None or _flask_app is None :
        raise RuntimeError ("App must be initialized")
    _socketio .run (_flask_app ,host ='0.0.0.0',port =5000 )