"""
WSANALIZ Flask Uygulaması - Ana Modül
"""
# pyright: reportImportCycles=false
# ERSIN Flask uygulamalarında circular import'lar yaygın ve genelde sorun yaratmaz
import logging 
import os 
import shutil 
import importlib 
from datetime import datetime 
from typing import Any ,Literal ,overload 

from flask import Flask ,send_from_directory 
from flask_sqlalchemy import SQLAlchemy 
from flask_migrate import Migrate 
from config import config 

from app .json_encoder import CustomJSONEncoder 
# ERSIN socketio proxy burada import edilmez, modüller create_app tamamlanmadan import ederse sorun çıkar
# ERSIN get_socketio fonksiyonunu app

# ERSIN Global minimal socketio referansı runtime sırasında ayarlanır
_current_running_socketio =None 
global_flask_app =None # ERSIN Ana Flask app nesnesi, background thread'ler için

# ERSIN bellek yardımcıları - isteğe bağlı import
try :
    from app .utils .memory_utils import initialize_memory_management 
except ImportError :
    initialize_memory_management =None 

    # ERSIN Global Flask eklentileri
db =SQLAlchemy ()
migrate =Migrate ()

# ERSIN SQLite dayanıklılığı, çoklu proses ortamlarında veritabanı kilitlemeyi azaltmak ...
# ERSIN SQLite olmayan veritabanları için güvenli no-op
try :
    import sqlite3 
    from sqlalchemy import event 
    from sqlalchemy .engine import Engine 

    @event .listens_for (Engine ,"connect")
    def _set_sqlite_pragmas (dbapi_connection ,connection_record ):# ERSIN SQLAlchemy event listener function signature
        try :
            if not isinstance (dbapi_connection ,sqlite3 .Connection ):
                return 
            cursor =dbapi_connection .cursor ()
            cursor .execute ("PRAGMA journal_mode=WAL;")
            cursor .execute ("PRAGMA synchronous=NORMAL;")
            cursor .execute ("PRAGMA busy_timeout=30000;")# ERSIN 30s
            cursor .close ()
        except Exception :
        # ERSIN Best-effort yaklaşım, boot'u engellemez
            pass 
except Exception :
    pass 

logger =logging .getLogger ("wsanaliz.app_init")
# ERSIN logging.basicConfig main.py içinde ortama göre çağrılır, burada çağırmak main.py'nin doğru seviyeyi ayarlamasını engeller

# ERSIN ===============================
# ERSIN 🎯 STANDARD FLASK-SOCKETIO PATTERN
# ERSIN ===============================
# ERSIN DİKKAT: SocketIO instance'ı SADECE burada, uygulama başlatılırken oluşturulur ve set edilir.
# ERSIN Başka hiçbir yerde yeni SocketIO instance'ı yaratılmayacak veya set edilmeyecek!

def register_blueprints_from_list (app ,blueprint_defs ):
    """
    blueprint_defs: List of tuples (import_path, attr_name, alias)
    - import_path: Python import path as string (e.g. 'app.routes.main_routes')
    - attr_name: Attribute name in the module (e.g. 'main_bp' or 'bp')
    - alias: Optional alias for logging (e.g. 'file_bp'), can be None
    """
    import traceback 
    logger =logging .getLogger ("wsanaliz.app_init")
    blueprints_to_register =[]
    for import_path ,attr_name ,alias in blueprint_defs :
        try :
            module =importlib .import_module (import_path )
            bp =getattr (module ,attr_name )
            blueprints_to_register .append (bp )
            logger .info (f"Blueprint imported: {import_path }.{attr_name } as {alias or attr_name }")
        except ImportError as e :
            logger .error (f"{import_path } import failed: {e }")
            logger .debug (f"Import traceback: {traceback .format_exc ()}")
        except AttributeError as e :
            logger .error (f"{import_path } has no attribute {attr_name }: {e }")
            logger .debug (f"AttributeError traceback: {traceback .format_exc ()}")
        except Exception as e :
            logger .error (f"{import_path } failed with unexpected error: {e }")
            logger .debug (f"Unexpected error traceback: {traceback .format_exc ()}")
    for bp in blueprints_to_register :
        try :
            app .register_blueprint (bp )
            logger .info (f"Blueprint registered: {bp .name }")
        except Exception as e :
            logger .error (f"Failed to register blueprint {bp .name }: {e }")
    logger .info (f"Total blueprints registered: {len (blueprints_to_register )}")
    return blueprints_to_register 

@overload 
def create_app (config_name :str ="default",*,return_socketio :Literal [False ]=False )->Flask :...


@overload 
def create_app (config_name :str ="default",*,return_socketio :Literal [True ])->tuple [Flask ,Any ]:...


def create_app (config_name :str ="default",*,return_socketio :bool =False )->Flask |tuple [Flask ,Any ]:
    """
    Flask uygulaması fabrikası.
    Args:
        config_name: Kullanılacak konfigürasyon adı.
        return_socketio: True ise (flask_app, socketio) tuple döner.
    Returns:
        return_socketio=False: sadece Flask app.
        return_socketio=True: (flask_app, minimal_socketio)
    """
    flask_app =Flask (__name__ )
    flask_app .config .from_object (config [config_name ])

    # ERSIN Set global app reference as early as possible için background/SocketIO helpers.
    # ERSIN Not: Some helpers historically relied on `global_flask_app` (legacy). Keep it safe.
    global global_flask_app 
    global_flask_app =flask_app 

    # ERSIN Server-side session store (cookie holds session id only)
    # ERSIN Not: Redis is optional; filesystem sessions work out of the box.
    try :
        from flask_session import Session 

        if (flask_app .config .get ("SESSION_TYPE")or "").strip ().lower ()=="redis":
            try :
            # ERSIN Redis is an optional dependency, import safely
                import redis 

                redis_url =(os .environ .get ("WSANALIZ_REDIS_URL")or "redis://localhost:6379/0").strip ()
                flask_app .config ["SESSION_REDIS"]=redis .Redis .from_url (redis_url )
            except Exception as e :
                logger .warning (f"SESSION_TYPE=redis ama Redis init başarısız, filesystem'e düşülüyor: {e }")
                flask_app .config ["SESSION_TYPE"]="filesystem"

        Session (flask_app )
        logger .info (f"Flask-Session initialized (type={flask_app .config .get ('SESSION_TYPE')})")
    except Exception as e :
        logger .warning (f"Flask-Session init atlandı: {e }")

        # ERSIN CORS configuration için cross-origin API requests
        # ERSIN Kritik: Required for frontend applications to access API endpoints
    try :
        from flask_cors import CORS 
        # ERSIN Allow all origins için development; in production, configure specific origins
        # ERSIN via CORS_ORIGINS environment variable veya config
        cors_origins =flask_app .config .get ('CORS_ORIGINS','*')
        if cors_origins =='*'or not cors_origins :
        # ERSIN Development mode: allow all origins
        # ERSIN Kritik: CORS spec forbids "origins: *" with "supports_credentials=True"
        # ERSIN Browser'lar credentialed request'leri reddeder, authentication bozulur
        # ERSIN Solution: Disable credentials support when allowing all origins
            CORS (flask_app ,resources ={r"/api/*":{"origins":"*"}},supports_credentials =False )
            logger .info ("CORS initialized: All origins allowed (development mode, credentials disabled per CORS spec)")
        else :
        # ERSIN Production mode: specific origins
            if isinstance (cors_origins ,str ):
            # ERSIN Split by comma ve strip whitespace from each origin
            # ERSIN This önlemeks CORS validation failures due to leading/trailing spaces
                origins_list =[origin .strip ()for origin in cors_origins .split (',')if origin .strip ()]
            else :
                origins_list =cors_origins 
            CORS (flask_app ,resources ={r"/api/*":{"origins":origins_list }},supports_credentials =True )
            logger .info (f"CORS initialized: Allowed origins: {origins_list }")
    except ImportError :
        logger .warning ("Flask-CORS not available - cross-origin requests may be blocked")
    except Exception as e :
        logger .warning (f"CORS initialization failed: {e } - cross-origin requests may be blocked")

        # ERSIN Flask uzantılarını başlat
    db .init_app (flask_app )
    # ERSIN Flask-Migrate'ı da initialize et (CLI migrations için gerekli)
    migrate .init_app (flask_app ,db )
    
    # ERSIN Template context processor: F5 load balancer için SocketIO path + API base
    @flask_app .context_processor 
    def inject_socketio_config ():
        from flask import request
        socketio_path =os .environ .get ("SOCKETIO_PATH","/socket.io/")
        socketio_url =os .environ .get ("SOCKETIO_URL","")
        # ERSIN F5 path prefix: request.script_root veya APPLICATION_ROOT (F5 path strip ediyorsa)
        api_base =""
        if request :
            api_base =request .script_root or ""
        if not api_base and os .environ .get ("APPLICATION_ROOT"):
            api_base =(os .environ .get ("APPLICATION_ROOT")or "").strip ().rstrip ("/")
        return {
            "socketio_path":socketio_path ,
            "socketio_url":socketio_url ,
            "api_base":api_base
        }

    # ERSIN ✅ MİNİMAL PATTERN: Optimize SocketIO kurulumu
    from flask_socketio import SocketIO 

    # ERSIN Cross-process SocketIO events:
    # ERSIN - If analysis runs in another process, you may want a message_queue (Redis...
    # ERSIN - But Redis gerekli be optional için backward compatibility.
    def _parse_bool_env (name :str )->bool |None :
        if name not in os .environ :
            return None 
        val =(os .environ .get (name )or "").strip ().lower ()
        if val in {"1","true","yes","y","on"}:
            return True 
        if val in {"0","false","no","n","off",""}:
            return False 
        return None 

    require_redis =_parse_bool_env ("SOCKETIO_REQUIRE_REDIS")is True 

    socketio_message_queue :str |None =None 
    if os .environ .get ("SOCKETIO_MESSAGE_QUEUE"):
        socketio_message_queue =os .environ .get ("SOCKETIO_MESSAGE_QUEUE")
    else :
    # ERSIN Redis message_queue sadece queue backend Redis olduğunda otomatik etkinleştirilir
        if (os .environ .get ("WSANALIZ_QUEUE_BACKEND")or "").strip ().lower ()=="redis":
            socketio_message_queue =os .environ .get ("WSANALIZ_REDIS_URL","redis://localhost:6379/0")

            # ERSIN Message_queue yapılandırılmışsa Redis erişilebilirliğini doğrula
    if socketio_message_queue :
        try :
            import redis # ERSIN Optional dependency, imported safely

            r =redis .Redis .from_url (socketio_message_queue ,socket_timeout =0.2 ,socket_connect_timeout =0.2 )
            r .ping ()
        except Exception as e :
            if require_redis :
                raise 
            logger .warning (
            f"SocketIO message_queue devre dışı bırakıldı (Redis erişilemiyor): {socketio_message_queue }. Hata: {e }"
            )
            socketio_message_queue =None 

            # ERSIN Determine async mode: prefer eventlet if available, otherwise auto-detect
    # ERSIN Type checker için socketio_kwargs'ı doğru tiplerle tanımla
    from typing import Any 
    
    # ERSIN F5 load balancer desteği için CORS origins environment variable'dan al
    cors_origins_socketio =flask_app .config .get ('CORS_ORIGINS','*')
    if cors_origins_socketio =='*'or not cors_origins_socketio :
        socketio_cors_origins ="*"
    else :
        if isinstance (cors_origins_socketio ,str ):
            socketio_cors_origins =[origin .strip ()for origin in cors_origins_socketio .split (',')if origin .strip ()]
        else :
            socketio_cors_origins =cors_origins_socketio 
    
    # ERSIN Debug logging için environment variable kontrolü
    enable_engineio_logger =(os .environ .get ("SOCKETIO_ENGINEIO_LOGGER","")or "").strip ().lower ()in {"1","true","yes","y","on"}
    
    socketio_kwargs :dict [str ,Any ]={
    'cors_allowed_origins':socketio_cors_origins ,
    'ping_timeout':720 ,# ERSIN Uzun analizler için 12 dakika timeout
    'ping_interval':60 ,# ERSIN Her dakika ping ile browser arka plan uyumluluğu
    'logger':False ,# ERSIN Verbose logging kapat
    'engineio_logger':enable_engineio_logger ,# ERSIN F5 debug için environment variable ile kontrol
    'message_queue':socketio_message_queue ,
    'allow_upgrades':True ,# ERSIN F5 load balancer için WebSocket upgrade desteği
    'cors_credentials':True if socketio_cors_origins !="*"else False ,# ERSIN CORS spec: credentials sadece spesifik origins ile
    'always_connect':False ,# ERSIN F5 arkasında polling transport için
    'transports':['polling','websocket'],# ERSIN F5 arkasında önce polling, sonra websocket upgrade
    }

    try :
        import eventlet
        _ = eventlet  # side-effect import for async_mode
        socketio_kwargs ['async_mode']='eventlet'
        logger .info ("Eventlet detected, using eventlet async mode")
    except ImportError :
        logger .warning ("Eventlet not available, Flask-SocketIO will auto-detect async mode")
        # ERSIN async_mode parametresini geçme, Flask-SocketIO en iyi modu otomatik tespit eder

    # ERSIN Type checker için socketio_kwargs'ı Any olarak cast et
    from typing import cast 
    minimal_socketio =SocketIO (flask_app ,**cast (dict [str ,Any ],socketio_kwargs ))

    # ERSIN Global instance'ı güncelleyelim - emit_analysis_progress için
    # ERSIN socketio_instance kasıtlı olarak küçük tutulur, döngüsel import önlemek için
    from app .socketio_instance import set_socketio 
    set_socketio (minimal_socketio )# ERSIN TEK NOKTA SET!

    # ERSIN Geriye dönük uyumluluk: bazı kodlar Flask app nesnesinden socketio'ya erişebilir
    # ERSIN Cross-module kullanım için canonical referans app.socketio_instance (proxy) üzerinden olmalı
    # ERSIN Yukarıda set_socketio(minimal_socketio) ile ayarlanır
    # ERSIN Dynamic attribute assignment - Flask allows this at runtime
    # ERSIN Use setattr to dynamically add attributes to Flask app instance
    setattr (flask_app ,'minimal_socketio',minimal_socketio )

    # ERSIN JSON encoder'ı ayarla
    setattr (flask_app ,'json_encoder',CustomJSONEncoder )

    # ERSIN Performans için memory management başlat
    try :
    # ERSIN Offline tools / one-shot scripts (e.g. prompt_sanity_check) may run on machines where
    # ERSIN CUDA init yavaş veya istenmeyen olabilir, environment variable ile devre dışı bırakılabilir
        disable_mm =(os .environ .get ("WSANALIZ_DISABLE_MEMORY_MANAGEMENT","")or "").strip ().lower ()in {"1","true","yes","y","on"}
        if initialize_memory_management :
            if disable_mm :
                logger .warning ("Memory management disabled via WSANALIZ_DISABLE_MEMORY_MANAGEMENT=1")
            else :
                initialize_memory_management ()
                logger .info ("Memory management initialized")
        else :
            logger .warning ("Memory management not available (optional dependency)")
    except Exception as e :
        logger .warning (f"Memory management initialization failed: {e }",exc_info =True )

        # ERSIN Blueprint'leri error handling ile kaydet (refaktör)
        # ERSIN Önemli: Register blueprints BEFORE middleware initialization to avoid circular imports
        # ERSIN Tüm route'ların kayıtlı olduğundan emin ol
    blueprint_defs =[
    ("app.routes.auth_routes","auth_bp",None ),
    ("app.routes.main_routes","main_bp",None ),
    ("app.routes.file_routes","bp","file_bp"),
    ("app.routes.analysis_routes","bp","analysis_bp"),
    ("app.routes.feedback_routes","bp","feedback_bp"),
    ("app.routes.settings_routes","bp","settings_bp"),
    ("app.routes.model_management_routes","model_management_bp",None ),
    ("app.routes.model_routes","bp","model_bp"),
    ("app.routes.queue_routes","queue_bp",None ),
    ("app.routes.performance_routes","performance_bp",None ),
    ("app.routes.debug_routes","bp","debug_bp"),
    ("app.routes.ensemble_routes","ensemble_bp",None ),
    ("app.routes.clip_training_routes","clip_training_bp",None ),
    ]
    registered_blueprints =register_blueprints_from_list (flask_app ,blueprint_defs )

    # ERSIN Kritik: Verify auth routes blueprint was successfully registered
    # ERSIN Eğer auth routes eksikse ama AuthMiddleware aktifse, kullanıcılar /login'e yönlendirilecek
    # ERSIN /login route'u yoksa authentication sistemi bozulur
    auth_bp_registered =any (bp .name =='auth'for bp in registered_blueprints )
    if not auth_bp_registered :
        error_msg =(
        "❌ CRITICAL: Auth routes blueprint failed to import/register. "
        "Authentication system is BROKEN - /login route is unavailable. "
        "AuthMiddleware will not be initialized to prevent redirect loops."
        )
        logger .error (error_msg )
        environment =os .environ .get ('FLASK_ENV','development')
        is_production =environment =='production'
        if is_production :
            raise RuntimeError (error_msg )
        else :
            logger .warning (
            "Development environment - application will continue but authentication is BROKEN. "
            "Fix auth_routes import/registration before deploying to production."
            )

            # ERSIN Security middleware başlat, blueprint'lerden sonra döngüsel import önlemek için
    from app .middleware .security_middleware import SecurityMiddleware 
    SecurityMiddleware (flask_app )

    # ERSIN Auth middleware (redirects unauthenticated users to /login)
    # ERSIN Önemli: Blueprint kaydından SONRA başlat, auth routes'un mevcut olduğundan emin ol
    # ERSIN Middleware'in before_request hook'u kaydedildiğinde
    # ERSIN Kritik: Only initialize if auth routes are successfully registered
    if auth_bp_registered :
        from app .middleware .auth_middleware import AuthMiddleware 
        AuthMiddleware (flask_app )
        logger .info ("AuthMiddleware initialized successfully (auth routes available)")
    else :
        logger .warning ("AuthMiddleware NOT initialized - auth routes unavailable")

        # ERSIN WebSocket event handlers registration
        # ERSIN Register comprehensive WebSocket handlers from websocket_routes module
        # ERSIN Kritik: WebSocket functionality is essential for real-time features
        # ERSIN If registration fails, fail fast in production to prevent silent deployment failures
    with flask_app .app_context ():
        from app .routes .websocket_routes import register_websocket_handlers_in_context 
        try :
            register_websocket_handlers_in_context (minimal_socketio )
            logger .info ("[OK] WebSocket handlers registered successfully!")
        except Exception as e :
            error_msg =(
            f"❌ CRITICAL: Failed to register WebSocket handlers: {e }. "
            f"WebSocket functionality (real-time updates, progress tracking) is BROKEN. "
            )
            logger .error (error_msg ,exc_info =True )
            # ERSIN Fail fast in production to prevent silent deployment failures
            # ERSIN In development, allow continuation için debugging but log prominently
            environment =os .environ .get ('FLASK_ENV','development')
            is_production =environment =='production'
            if is_production :
                logger .error ("Production environment detected - failing fast to prevent broken deployment")
                raise RuntimeError (error_msg )from e 
            else :
                logger .warning (
                "Development environment - application will continue but WebSocket features are BROKEN. "
                "Fix the registration error before deploying to production."
                )

                # ERSIN Error handlers (404/500) – previously disabled için circular-import concerns
                # ERSIN Bu fonksiyon kendi kendine yeterlidir ve sadece handler'lar içinde jsonify import eder
    register_error_handlers (flask_app )

    # ERSIN Global/static route'lar app creation zamanında kaydedilir, aksi halde bazı entrypoint'ler ve testler bunları kaçırabilir
    register_global_routes (flask_app )

    # ERSIN WebSocket handler'ları SocketIO initialization sırasında yukarıda kaydedilir

    # ERSIN Önemli: Import all models to register them with SQLAlchemy metadata
    # ERSIN Bu create_app() içinde olmalı, böylece initialize_app() çağrıldığında metadata mevcut olur
    # ERSIN Ancak db.create_all() sadece initialize_app() içinde çağrılmalı, multi-worker setup'larda race condition ve testlerde duplicate initialization önlemek için
    with flask_app .app_context ():
        try :
            from app import models
            _ = models  # side-effect: registers models with SQLAlchemy
            logger .debug ("Models imported for SQLAlchemy metadata registration")
        except Exception as e :
            logger .error (f"Model import failed: {str (e )}",exc_info =True )
            # ERSIN Don't fail app creation - models might be imported later

            # ERSIN Geriye uyumluluk: default sadece Flask app döndür.
            # ERSIN SocketIO instance'a ihtiyaç olan yerler return_socketio=True ile tuple alabilir.
    return (flask_app ,minimal_socketio )if return_socketio else flask_app 


    # ERSIN Açıklama: Not:
    # ERSIN `initialize_app()` calls `check_and_run_migrations()`
    # ERSIN Python'da geçerli, modül önce tamamen yüklenir, bazı reviewer'lar bunu kafa karıştırıcı bulabilir
    # ERSIN Küçük wrapper burada tut, fonksiyon adı initialize_app'den önce tanımlı olsun
def check_and_run_migrations ():
    return _check_and_run_migrations ()

def initialize_app (app ):
    """
    Uygulamayı başlatır ve gerekli temizlik işlemlerini yapar.
    Bu fonksiyon sadece ana süreçte çağrılmalıdır.
    
    Args:
        app: Flask uygulaması
    """
    with app .app_context ():
    # ERSIN ERSIN Önemli: Import all models before db
        from app import models
        _ = models  # side-effect: registers models with SQLAlchemy

        # ERSIN SQLAlchemy'nin kullanacağı gerçek veritabanı yolunu al
        # ERSIN SQLAlchemy relative path'leri current working directory'den çözer, app.root_path'ten değil
        # ERSIN Engine initialize edildikten sonra gerçek yolu engine'den almak gerekir
        db_uri =app .config .get ('SQLALCHEMY_DATABASE_URI','')
        is_sqlite =db_uri .startswith ('sqlite:///')

        # ERSIN Initialize db_path to None to önlemek NameError if exception occurs
        db_path =None 

        if is_sqlite :
        # ERSIN Extract path from SQLite URI
            db_path_from_uri =db_uri .replace ('sqlite:///','')
            # ERSIN SQLAlchemy resolves relative paths from CWD, so we need to check the actual path
            # ERSIN Get the actual path from the engine's URL
            try :
            # ERSIN After db.init_app(), engine mevcuttur
                engine_url =db .engine .url 
                if hasattr (engine_url ,'database')and engine_url .database :
                # ERSIN For SQLite, database is the file path
                    actual_db_path =engine_url .database 
                    if actual_db_path and actual_db_path !=':memory:':
                        db_path =actual_db_path 
                    else :
                    # ERSIN Fallback to manual resolution
                        db_path =db_path_from_uri if os .path .isabs (db_path_from_uri )else os .path .abspath (db_path_from_uri )
                else :
                # ERSIN Fallback to manual resolution
                    db_path =db_path_from_uri if os .path .isabs (db_path_from_uri )else os .path .abspath (db_path_from_uri )
            except Exception as e :
                logger .warning (f"Could not determine actual database path from engine: {e }, using URI-based path")
                # ERSIN Fallback to manual resolution
                db_path =db_path_from_uri if os .path .isabs (db_path_from_uri )else os .path .abspath (db_path_from_uri )
        else :
        # ERSIN sqlite olmayan database, path check uygulanamaz
            db_path =None 

        if db_path :
            logger .info (f"Veritabanı yolu: {db_path }")

            # ERSIN Check if database exists
            # ERSIN For SQLite: check if file exists
            # ERSIN Diğer veritabanları için: bağlantı başarılıysa veritabanının var olduğunu varsay
        db_exists =False 
        if is_sqlite and db_path :
            db_exists =os .path .exists (db_path )
        else :
        # ERSIN For sqlite olmayan databases, try to connect to verify existence
        # ERSIN Bağlantı başarılıysa veritabanı var (veya sunucu tarafından oluşturulacak)
        # ERSIN Kritik: Catch all exceptions including engine creation failures,
        # ERSIN Geçersiz URI'ler, erişilemeyen sunucular vb. için db_exists her zaman set edilir
            try :
            # ERSIN Bağlantı denemeden önce engine'in mevcut olduğundan emin ol
                if not hasattr (db ,'engine')or db .engine is None :
                    logger .warning ("Database engine not available, assuming database does not exist")
                    db_exists =False 
                else :
                # ERSIN Bağlantıyı test et, çalışırsa veritabanı var veya erişilebilir
                # ERSIN Bağlantının düzgün kapatıldığından emin olmak için context manager kullan
                # ERSIN Not: Exception context manager entry (__enter__) sırasında veya bağlantı kurulurken oluşabilir, tüm exception'ları yakala
                    with db .engine .connect ():
                    # ERSIN Bağlantı başarılı, veritabanı var veya erişilebilir
                        pass 
                    db_exists =True 
            except Exception as conn_err :
            # ERSIN Connection failed - database might not exist yet, URI invalid, or server unreachable
            # ERSIN Debug için error'u log'la ama fail etme, db.create_all() bunu handle eder
                logger .warning (
                f"Database connection check failed (assuming database does not exist): {conn_err }. "
                f"This is normal for new databases and will be handled by db.create_all()."
                )
                db_exists =False 

                # ERSIN Create tables if needed
                # ERSIN db.create_all() idempotenttir ve birden fazla kez güvenle çağrılabilir
        if not db_exists :
            logger .info ("Veritabanı bulunamadı veya erişilemiyor, yeni veritabanı/tablolar oluşturuluyor.")
        else :
            logger .info ("Mevcut veritabanı kullanılıyor.")

        db .create_all ()

        # ERSIN Run migrations to add missing columns (SQLite-specific, skips için other databases)
        check_and_run_migrations ()

        # ERSIN Klasörlerin oluşturulması ve temizlenmesi
        os .makedirs (app .config ['UPLOAD_FOLDER'],exist_ok =True )
        os .makedirs (app .config ['PROCESSED_FOLDER'],exist_ok =True )
        os .makedirs (app .config ['MODELS_FOLDER'],exist_ok =True )

        # ERSIN Upload klasörü temizlemeyi devre dışı bırak - Eğitim verisi güvenliği için
        # ERSIN clean_folder(app.config['UPLOAD_FOLDER'])  # ← EĞİTİM VERİSİ GÜVENLİĞİ İÇİN KAPATILDI
        # ERSIN clean_folder(app.config['PROCESSED_FOLDER'])  # Analiz sonuçlarını korumak için devre dışı

        # ERSIN Eski analiz sonuçlarını temizle (7 günden eski olanları)
        # ERSIN Pass app parameter to ensure it's available even if global_flask_app isn't set yet
        cleanup_old_analysis_results (days_old =7 ,app =app )

        # ERSIN Model versiyonlarını senkronize et (VT oluşturulduktan sonra)
        sync_model_versions_on_startup ()

        # ERSIN Worker crash recovery: "processing" durumunda olan ama uzun süredir ilerlemeyen analizleri kurtar
        # ERSIN Pass app parameter to ensure it's available even if global_flask_app isn't set yet
        recover_stuck_analyses (app =app )

        # ERSIN Analiz kuyruğu servisini başlat (sadece memory backend'te)
        # ERSIN Redis backend'te queue processing ayrı worker prosesinde yapılır.
        try :
            from app .services import queue_service as _queue_service 
            if not getattr (_queue_service ,"is_redis_backend",lambda :False )():
                from app .services .queue_service import start_processor 
                logger .info ("Analiz kuyruğu servisi (memory backend) başlatılıyor...")
                start_processor ()
                logger .info ("Analiz kuyruğu servisi başlatıldı.")
            else :
                logger .info ("Redis queue backend aktif: web prosesinde queue processor başlatılmıyor.")
        except Exception as e :
            logger .warning (f"Queue processor init atlandı: {e }")

            # ERSIN Global route'lar create_app() içinde kaydedilir

def clean_folder (folder_path ):
    """
    Belirtilen klasörü temizler.
    
    Args:
        folder_path (str): Temizlenecek klasörün yolu.
    """
    if os .path .exists (folder_path ):
        for filename in os .listdir (folder_path ):
            file_path =os .path .join (folder_path ,filename )

            # ERSIN Eğer 'logs' klasörüyse (veya 'logs' klasörünün içindeysek) silme
            if filename =='logs'and os .path .isdir (file_path ):
                logger .warning (f"'{file_path }' log klasörü atlanıyor, silinmeyecek.")
                continue # ERSIN logs klasörünü silme, içini de boşaltma

            if os .path .isfile (file_path )or os .path .islink (file_path ):
                try :
                    os .unlink (file_path )
                except Exception as e :
                    logger .warning (f"Dosya silinirken hata (atlanıyor): {file_path }, Hata: {e }",exc_info =True )
            elif os .path .isdir (file_path ):
                try :
                    shutil .rmtree (file_path )
                except Exception as e :
                    logger .warning (f"Klasör silinirken hata (atlanıyor): {file_path }, Hata: {e }",exc_info =True )
    else :
        os .makedirs (folder_path ,exist_ok =True )

def _check_and_run_migrations ():
    """
    Veritabanı migration kontrolü yapar ve gerekli kolumları ekler.
    Requires an active Flask application context.
    """
    from flask import current_app as _current_app ,has_app_context as _has_app_context 

    # ERSIN Ensure we have an app context
    if not _has_app_context ():
        logger .error ("_check_and_run_migrations() requires an active Flask app context")
        raise RuntimeError ("_check_and_run_migrations() called without Flask app context")

    conn =None 
    conn_closed =False # ERSIN Track connection state to önlemek double-close
    cursor =None # ERSIN Initialize cursor to None to önlemek NameError if connection fails

    def _parse_bool_env (name :str )->bool |None :
        if name not in os .environ :
            return None 
        val =(os .environ .get (name )or "").strip ().lower ()
        if val in {"1","true","yes","y","on"}:
            return True 
        if val in {"0","false","no","n","off",""}:
            return False 
        return None 

    strict_env =_parse_bool_env ("WSANALIZ_MIGRATIONS_STRICT")
    strict =(not _current_app .config .get ("TESTING",False ))if strict_env is None else bool (strict_env )

    # ERSIN Veritabanı backend'ini kontrol et, migration'lar SQLite-specific
    try :
        db_dialect =db .engine .dialect .name 
        is_sqlite_backend =db_dialect =='sqlite'
    except Exception :
    # ERSIN If we can't determine dialect, assume sqlite olmayan ve skip migrations
        logger .warning ("⚠️ Migration: Database backend belirlenemedi, migration'lar atlanıyor.")
        return 

    if not is_sqlite_backend :
        logger .info (f"ℹ️ Migration: Database backend '{db_dialect }' - SQLite-specific migration'lar atlanıyor. "
        f"PostgreSQL/MySQL gibi database'ler için Alembic migration'ları kullanın.")
        return 
    try :
        logger .info ("🔄 Veritabanı migration kontrolü yapılıyor (SQLite)...")

        # ERSIN Use SQLAlchemy's engine connection to avoid separate sqlite3 path handling/locking surprises.
        conn =db .engine .raw_connection ()
        cursor =conn .cursor ()

        # ERSIN First, check if the analyses table exists (SQLite-specific query)
        cursor .execute ("SELECT name FROM sqlite_master WHERE type='table' AND name='analyses'")
        table_exists =cursor .fetchone ()is not None 

        if not table_exists :
            logger .warning ("⚠️ Migration: 'analyses' tablosu henüz oluşturulmamış. Migration'lar tablo oluşturulduktan sonra uygulanacak.")
            conn .close ()
            conn_closed =True 
            return 

            # ERSIN analyses tablosundaki kolonları kontrol et (SQLite-specific PRAGMA)
        cursor .execute ("PRAGMA table_info(analyses)")
        columns =[column [1 ]for column in cursor .fetchall ()]

        migrations_needed =[]
        errors :list [str ]=[]

        # ERSIN websocket_session_id kolonu var mı?
        if 'websocket_session_id'not in columns :
            migrations_needed .append (('websocket_session_id','TEXT'))

            # ERSIN is_cancelled kolonu var mı?
        if 'is_cancelled'not in columns :
        # ERSIN SQLite has no native BOOLEAN; use INTEGER affinity.
            migrations_needed .append (('is_cancelled','INTEGER DEFAULT 0'))

            # ERSIN Migration'ları uygula
            # ERSIN ERSIN Kritik: All migrations must succeed or none should be committed to avoid p...
        applied =0 
        failed_migrations =[]

        for column_name ,column_def in migrations_needed :
            try :
                sql =f"ALTER TABLE analyses ADD COLUMN {column_name } {column_def }"
                cursor .execute (sql )
                applied +=1 
                logger .info (f"✅ Migration: {column_name } kolonu eklendi")
            except Exception as e :
                msg =str (e )
                # ERSIN Duplicate column güvenle ignore edilebilir, column zaten var, schema tutarlı
                if "duplicate column name"in msg .lower ():
                    logger .info (f"ℹ️ Migration: {column_name } zaten var (duplicate), atlanıyor")
                    # ERSIN Column zaten var olduğu için applied olarak say (schema tutarlı)
                    applied +=1 
                elif "no such table"in msg .lower ():
                    logger .error (f"❌ Migration hatası ({column_name }): Tablo bulunamadı. Migration atlanıyor.")
                    errors .append (f"{column_name }: Table does not exist")
                    # ERSIN Tablo yoksa commit yapma, bu fatal error
                    # ERSIN Track as failed migration to ensure rollback happens
                    failed_migrations .append (column_name )
                    try :
                        if not conn_closed :
                            conn .rollback ()
                    except Exception :
                        pass 
                    if not conn_closed :
                        conn .close ()
                        conn_closed =True 
                        # ERSIN Break loop since table doesn't exist - can't proceed ile more migrations
                    break 
                else :
                # ERSIN Non-duplicate error, bu gerçek bir hata ve schema'yı tutarsız bırakabilir
                    logger .error (f"❌ Migration hatası ({column_name }): {msg }",exc_info =True )
                    errors .append (f"{column_name }: {msg }")
                    failed_migrations .append (column_name )
                    # ERSIN Continue loop to collect all errors, but we won't commit if any failed

                    # ERSIN Only commit if ALL migrations succeeded (or were duplicates)
                    # ERSIN If any migration failed, rollback to prevent partial/inconsistent schema
        if failed_migrations :
            logger .error (
            f"❌ Migration başarısız: {len (failed_migrations )} migration uygulanamadı "
            f"({', '.join (failed_migrations )}). Rollback yapılıyor, schema değişiklikleri uygulanmadı."
            )
            if not conn_closed :
                try :
                    conn .rollback ()
                except Exception :
                    pass 
                    # ERSIN Don't commit - schema tutarsız olur
        else :
        # ERSIN All migrations succeeded (or were duplicates), safe to commit
        # ERSIN Kritik: Only commit if no migrations failed, regardless of applied count
            if not conn_closed :
                try :
                    conn .commit ()
                except Exception as commit_err :
                    logger .error (f"❌ Migration commit hatası: {commit_err }",exc_info =True )
                    try :
                        conn .rollback ()
                    except Exception :
                        pass 
                    raise 

        if migrations_needed :
            if failed_migrations :
                logger .warning (f"⚠️ Migration kısmen başarısız: {applied }/{len (migrations_needed )} uygulandı, {len (failed_migrations )} başarısız")
            else :
                logger .info (f"🎉 {applied }/{len (migrations_needed )} migration uygulandı!")
        else :
            logger .info ("✅ Veritabanı şeması güncel, migration gerekmiyor")

            # ERSIN Verify columns exist after migrations (db.create_all() won't add columns).
            # ERSIN SQLite-specific PRAGMA query
            # ERSIN Sadece bağlantı mevcutsa doğrula
            # ERSIN Kritik: cursor may be undefined if connection failed before cursor creation
        if not conn_closed and cursor is not None :
            try :
                cursor .execute ("PRAGMA table_info(analyses)")
                final_cols ={column [1 ]for column in cursor .fetchall ()}
            except Exception as verify_err :
                logger .warning (f"Migration verification failed: {verify_err }")
                final_cols =set ()
        else :
        # ERSIN Connection was closed veya cursor unavailable, skip verification
            final_cols =set ()
            # ERSIN Bu column'lar cancellation/session tracking için kullanılır, app bunlara bağımlıysa gracefully degrade etmeli
        optional_cols ={"websocket_session_id","is_cancelled"}
        missing_optional =sorted (optional_cols -final_cols )
        if missing_optional :
            errors .append (f"missing optional columns after migration: {missing_optional }")

        if errors :
            err_msg =f"Database migrations incomplete/failed: {errors }"
            # ERSIN Optional-column sorunları için app'i çökertme, sadece log'la
            logger .error (f"⚠️ {err_msg }")

    except Exception as e :
        logger .error (f"❌ Migration kontrolü hatası: {str (e )}",exc_info =True )
        # ERSIN Only fail-fast için truly fatal migration issues (e
        # ERSIN önlemek the service from starting.
        if strict :
            raise 
    finally :
        try :
            if conn is not None and not conn_closed :
                conn .close ()
        except Exception :
            pass 

def recover_stuck_analyses (app =None ):
    """
    Worker crash recovery: "processing" durumunda olan ama uzun süredir ilerlemeyen 
    analizleri kontrol edip "failed" yapar.
    
    Bu, worker process segfault veya crash olduğunda analizlerin takılı kalmasını önler.
    
    Args:
        app: Flask uygulaması (opsiyonel). Verilirse bu app kullanılır, yoksa 
             app_context veya global_flask_app'ten çözülmeye çalışılır.
    
    Requires an active Flask application context or app parameter.
    """
    from flask import current_app as _current_app ,has_app_context as _has_app_context 

    def _resolve_app_obj ():
    # ERSIN Priority 1: Explicit app parameter
        if app is not None :
            return app 
            # ERSIN Priority 2: Current app context
        if _has_app_context ():
        # ERSIN Flask _get_current_object() exists but type checker doesn't recognize it
        # ERSIN Use getattr to safely access the method, then call it if it exists
        # ERSIN If it doesn't exist, fallback to _current_app directly (don't call it as a function)
            _get_current_object =getattr (_current_app ,'_get_current_object',None )
            if _get_current_object is not None and callable (_get_current_object ):
                result =_get_current_object ()
                # ERSIN result is Flask app instance at runtime
                return result if isinstance (result ,Flask )else None 
            else :
            # ERSIN Fallback: _get_current_object doesn't exist veya isn't callable, use _current_app directly
                return _current_app if isinstance (_current_app ,Flask )else None 
                # ERSIN Priority 3: Global flask app (fallback için background threads)
        try :
            from app import global_flask_app as _global_flask_app 
            return _global_flask_app 
        except Exception :
            return None 

    app_obj =_resolve_app_obj ()
    if app_obj is None :
        logger .error (
        "recover_stuck_analyses: Flask app bulunamadı (no app parameter, no app_context, global_flask_app None). "
        "CRITICAL: Stuck analysis recovery skipped - this may leave analyses in 'processing' state indefinitely."
        )
        return 

        # ERSIN Type narrowing: app_obj is Flask at this point (not None)
    if not isinstance (app_obj ,Flask ):
        logger .error ("recover_stuck_analyses: app_obj is not Flask instance")
        return 

        # ERSIN Check if we're already in an app context to avoid nested contexts
    already_in_context =_has_app_context ()

    def _execute_recovery ():
        from app .models .analysis import Analysis 

        # ERSIN Redis queue mimarisinde dedicated worker processing'den sorumludur
        # ERSIN Worker heartbeat fresh ise web startup'ta "processing" analizleri otomatik fail etme
        # ERSIN this causes false-failures + SQLite lock contention.
        try :
            if (os .environ .get ("WSANALIZ_QUEUE_BACKEND")or "").strip ().lower ()=="redis":
                import time as _time 
                import redis # ERSIN Optional dependency, imported safely

                redis_url =(os .environ .get ("WSANALIZ_REDIS_URL")or "redis://localhost:6379/0").strip ()
                heartbeat_key =(os .environ .get ("WSANALIZ_WORKER_HEARTBEAT_KEY")or "wsanaliz:worker:last_heartbeat").strip ()
                r =redis .Redis .from_url (redis_url ,decode_responses =True )
                hb =r .get (heartbeat_key )
                if hb :
                    try :
                    # ERSIN Redis get() returns str when decode_responses=True
                        if isinstance (hb ,str ):
                            hb_ts =float (hb )
                            time_since_heartbeat =_time .time ()-hb_ts 
                            # ERSIN Time difference'i pozitif aralıkta doğrula (0-30 saniye)
                            # ERSIN This önlemeks negative values from clock skew (future timestamps) from inc...
                            # ERSIN Use <= 30 to include exactly 30 seconds old heartbeats as potentially stale
                            # ERSIN Accounts için clock skew ve heartbeat intervals to avoid false negatives
                            if 0 <=time_since_heartbeat <=30 :
                                logger .info ("🔍 Worker crash recovery: Worker heartbeat taze (Redis). Recovery atlandı (false-fail + DB lock önleme).")
                                return 
                            elif time_since_heartbeat <0 :
                            # ERSIN Clock skew tespit edildi: heartbeat timestamp gelecekte
                                logger .warning (
                                f"⚠️ Worker crash recovery: Clock skew detected (heartbeat timestamp {hb_ts } is in the future, "
                                f"time difference: {time_since_heartbeat :.2f}s). Proceeding with recovery to prevent stuck analyses."
                                )
                    except Exception :
                        pass 
        except Exception :
        # ERSIN Best-effort; if Redis mevcuttur, fall back to DB-only recovery.
            pass 

            # ERSIN "processing" durumunda olan analizleri bul
        stuck_analyses =Analysis .query .filter (
        Analysis .status =='processing'
        ).all ()

        if not stuck_analyses :
            logger .info ("🔍 Worker crash recovery: Takılı analiz bulunamadı.")
            return 

        recovered_count =0 
        timeout_minutes =10 # ERSIN 10 dakikadan fazla "processing" durumunda olan analizler takılı sayılır

        # ERSIN Use UTC consistently; stored timestamps olmalı naive-UTC.
        # ERSIN Kritik: Handle both naive and timezone-aware datetimes from database
        # ERSIN SQLAlchemy may return timezone-aware datetimes depending on database backend
        # ERSIN TypeError occurs when comparing naive ve aware datetimes
        now_naive =datetime .utcnow ()

        def normalize_datetime (dt ):
            """Convert timezone-aware datetime to naive UTC, or return naive datetime as-is"""
            if dt is None :
                return None 
            if dt .tzinfo is not None :
            # ERSIN Kritik: Properly convert timezone-aware to UTC, then remove tzinfo
            # ERSIN Simply replacing tzinfo=None would keep the same wall time but wrong UTC moment
            # ERSIN Example: 2024-01-01 15:00:00+05:00 -> 2024-01-01 10:00:00 (UTC equivalent)
                from datetime import timezone 
                utc_dt =dt .astimezone (timezone .utc )
                return utc_dt .replace (tzinfo =None )
            return dt 

        for analysis in stuck_analyses :
        # ERSIN start_time mevcut DB'lerde NULL olabilir, created_at'e (veya updated_at'e) fallback yap
            start_time =normalize_datetime (analysis .start_time )
            created_at =normalize_datetime (getattr (analysis ,"created_at",None ))
            updated_at =normalize_datetime (getattr (analysis ,"updated_at",None ))

            # ERSIN Prefer start_time (if sane, i.e., değil in future), then updated_at, then created_at.
            # ERSIN Guard against mixed timezone origins / clock skew where timestamps olabilir end up in  future.
            # ERSIN İlk timestamp'i seç, şu <= now (gelecekte değil)
            ref_time =None 
            if start_time and start_time <=now_naive :
                ref_time =start_time 
            elif updated_at and updated_at <=now_naive :
                ref_time =updated_at 
            elif created_at and created_at <=now_naive :
                ref_time =created_at 

                # ERSIN If no valid (non-future) timestamp found, check if all timestamps dır in future
            if ref_time is None :
            # ERSIN Check if we have any timestamps at all
                has_any_timestamp =start_time is not None or updated_at is not None or created_at is not None 

                if not has_any_timestamp :
                # ERSIN No timestamps at all için açıklama
                # ERSIN 1. A new analysis için açıklama
                # ERSIN 2. A corrupted analysis record
                # ERSIN Since we're only checking 'processing' analyses, if it has no timestamps,
                # ERSIN Muhtemelen yeni analiz az önce oluşturuldu ama başlatılmadı
                # ERSIN Do NOT mark as failed - this would corrupt legitimate pending analyses.
                    logger .info (
                    f"ℹ️ Worker crash recovery: Analiz #{analysis .id } 'processing' ama timestamp yok "
                    f"(start_time/created_at/updated_at NULL). Bu yeni bir analiz olabilir, atlanıyor."
                    )
                    # ERSIN Skip this analysis - don't mark as failed to avoid corrupting legitimate pending analyses
                    continue 
                else :
                # ERSIN All timestamps dır in future - likely clock skew
                # ERSIN Do NOT mark as failed - this could be a valid in-progress analysis
                # ERSIN Log  issue ve skip bu analysis to avoid corrupting legitimate work
                    logger .warning (
                    "⚠️ Worker crash recovery: Analiz #%s 'processing' ama tüm timestamp'ler gelecekte "
                    "(start_time=%s, updated_at=%s, created_at=%s, now=%s). "
                    "Muhtemel timezone/clock skew; analiz atlanıyor (geçerli bir in-progress analiz olabilir).",
                    analysis .id ,
                    start_time ,
                    updated_at ,
                    created_at ,
                    now_naive ,
                    )
                    # ERSIN Skip this analysis - don't mark as failed to avoid corrupting legitimate work
                    continue 

            elapsed =now_naive -ref_time 
            elapsed_minutes =elapsed .total_seconds ()/60 

            if elapsed_minutes >timeout_minutes :
                logger .warning (
                f"🔧 Worker crash recovery: Analiz #{analysis .id } "
                f"{elapsed_minutes :.1f} dakikadır 'processing' durumunda, "
                f"'failed' olarak işaretleniyor (muhtemelen worker crash)."
                )
                analysis .status ='failed'
                analysis .error_message =(
                f"Worker process crash nedeniyle analiz başarısız oldu. "
                f"Analiz {elapsed_minutes :.1f} dakikadır işleniyordu."
                )
                analysis .end_time =now_naive 
                recovered_count +=1 

        if recovered_count >0 :
            db .session .commit ()
            logger .info (f"✅ Worker crash recovery: {recovered_count } takılı analiz 'failed' olarak işaretlendi.")
        else :
            logger .info ("ℹ️ Worker crash recovery: Tüm 'processing' analizler aktif görünüyor.")

    try :
    # ERSIN If already in context, execute directly without opening nested context
        if already_in_context :
            _execute_recovery ()
        else :
        # ERSIN Context yoksa, yeni bir context aç
        # ERSIN Type narrowing: app_obj is Flask at this point (checked above)
            if isinstance (app_obj ,Flask ):
                with app_obj .app_context ():
                    _execute_recovery ()
            else :
                logger .error ("recover_stuck_analyses: app_obj is not Flask instance")
    except Exception as e :
        logger .error (f"❌ Worker crash recovery hatası: {e }",exc_info =True )
        # ERSIN Only attempt rollback if we're in a context
        if already_in_context :
            try :
                db .session .rollback ()
            except Exception :
                pass 
        else :
        # ERSIN Type narrowing: app_obj is Flask at this point (checked above)
            if isinstance (app_obj ,Flask ):
                try :
                    with app_obj .app_context ():
                        try :
                            db .session .rollback ()
                        except Exception :
                            pass 
                except Exception :
                    pass 

def cleanup_old_analysis_results (days_old =7 ,app =None ):
    """
    Belirli bir yaştan eski analiz sonuçlarını ve ilgili dosyaları temizler.
    
    Args:
        days_old: Kaç günden eski analizlerin temizleneceği (varsayılan: 7)
        app: Flask uygulaması (opsiyonel). Verilirse bu app kullanılır, yoksa 
             app_context veya global_flask_app'ten çözülmeye çalışılır.
    
    Raises:
        ValueError: If days_old is negative or zero
    """
    # ERSIN Validate days_old parameter
    if days_old <=0 :
        raise ValueError (f"days_old must be positive, got {days_old }")

    from flask import current_app as _current_app ,has_app_context as _has_app_context 

    def _resolve_app_obj ()->Flask |None :
    # ERSIN Priority 1: Explicit app parameter
        if app is not None :
            return app 
            # ERSIN Priority 2: Current app context
        if _has_app_context ():
        # ERSIN Flask _get_current_object() exists but type checker doesn't recognize it
        # ERSIN Use getattr to safely access the method, then call it if it exists
        # ERSIN If it doesn't exist, fallback to _current_app directly (don't call it as a function)
            _get_current_object =getattr (_current_app ,'_get_current_object',None )
            if _get_current_object is not None and callable (_get_current_object ):
                result =_get_current_object ()
                # ERSIN result is Flask app instance at runtime
                return result if isinstance (result ,Flask )else None 
            else :
            # ERSIN Fallback: _get_current_object doesn't exist veya isn't callable, use _current_app directly
                return _current_app if isinstance (_current_app ,Flask )else None 
                # ERSIN Priority 3: Global flask app (fallback için background threads)
        try :
            from app import global_flask_app as _global_flask_app 
            return _global_flask_app 
        except Exception :
            return None 

    app_obj :Flask |None =_resolve_app_obj ()
    if app_obj is None :
        logger .error (
        "cleanup_old_analysis_results: Flask app bulunamadı (no app parameter, no app_context, global_flask_app None). "
        "CRITICAL: Old analysis cleanup skipped - this may lead to disk space issues."
        )
        return 

        # ERSIN Type narrowing: app_obj is Flask at this point (not None)
    if not isinstance (app_obj ,Flask ):
        logger .error ("cleanup_old_analysis_results: app_obj is not Flask instance")
        return 

        # ERSIN Check if we're already in an app context to avoid nested contexts
    already_in_context =_has_app_context ()

    def _execute_cleanup ():
        from datetime import datetime ,timedelta 
        from app .models .analysis import Analysis 

        cutoff_date =datetime .utcnow ()-timedelta (days =days_old )
        logger .info (f"Eski analiz sonuçları temizleniyor: {cutoff_date } tarihinden eski olanlar")

        # ERSIN Eski analizleri bul (created_at yerine start_time kullan)
        old_analyses =Analysis .query .filter (Analysis .start_time <cutoff_date ).all ()

        if not old_analyses :
            logger .info ("Temizlenecek eski analiz bulunamadı.")
            return 

        logger .info (f"{len (old_analyses )} eski analiz bulundu, temizleniyor...")

        processed_root =app_obj .config ['PROCESSED_FOLDER']

        # ERSIN Her analiz için ilgili dosyaları temizle
        # ERSIN Kritik: Delete files FIRST, then delete from database only if file cleanup succeeds
        # ERSIN This prevents database records from being deleted when file cleanup fails
        cleaned_count =0 
        failed_count =0 
        for analysis in old_analyses :
            try :
                analysis_folder =os .path .join (processed_root ,f"frames_{analysis .id }")
                if os .path .exists (analysis_folder ):
                    shutil .rmtree (analysis_folder )
                    logger .info (f"Analiz klasörü silindi: {analysis_folder }")

                    # ERSIN İşlenmiş resim dosyasını sil (alan mevcutsa)
                processed_image_rel =getattr (analysis ,'processed_image_path',None )
                if processed_image_rel :
                    processed_file =os .path .join (processed_root ,processed_image_rel )
                    if os .path .exists (processed_file ):
                        os .unlink (processed_file )
                        logger .info (f"İşlenmiş resim silindi: {processed_file }")

                        # ERSIN En yüksek riskli kare dosyasını sil
                if analysis .highest_risk_frame :
                    risk_frame_file =os .path .join (processed_root ,analysis .highest_risk_frame )
                    if os .path .exists (risk_frame_file ):
                        os .unlink (risk_frame_file )
                        logger .info (f"En yüksek riskli kare silindi: {risk_frame_file }")

                        # ERSIN Veritabanından analizi sil (cascade ile ilgili kayıtlar da silinir)
                        # ERSIN Only delete from database AFTER file cleanup succeeds
                        # ERSIN If file cleanup fails, exception is raised and database deletion is not queued
                db .session .delete (analysis )
                cleaned_count +=1 

            except Exception as e :
                logger .warning (f"Analiz {analysis .id } temizlenirken hata: {e }",exc_info =True )
                # ERSIN File cleanup failed - do NOT delete from database to maintain consistency
                # ERSIN Continue to next analysis without queuing database deletion
                failed_count +=1 
                continue 

                # ERSIN Değişiklikleri kaydet
        db .session .commit ()
        if failed_count >0 :
            logger .info (f"{cleaned_count }/{len (old_analyses )} eski analiz başarıyla temizlendi. {failed_count } analiz temizlenemedi.")
        else :
            logger .info (f"{cleaned_count } eski analiz başarıyla temizlendi.")

            # ERSIN Artık kullanılmayan dosyaları da temizle
            # ERSIN Wrap in try/except to önlemek exceptions from triggering outer rollback
            # ERSIN since commit() already succeeded
        try :
            cleanup_orphaned_files ()
        except Exception as cleanup_err :
            logger .warning (f"Yetim dosya temizliği sırasında hata (analiz temizliği tamamlandı): {cleanup_err }",exc_info =True )
            # ERSIN Re-raise yapma, cleanup best-effort ve main operation'ı fail etmemeli

    try :
    # ERSIN If already in context, execute directly without opening nested context
        if already_in_context :
            _execute_cleanup ()
        else :
        # ERSIN Context yoksa, yeni bir context aç
        # ERSIN Type narrowing: app_obj is Flask at this point (checked above)
            if isinstance (app_obj ,Flask ):
                with app_obj .app_context ():
                    _execute_cleanup ()
            else :
                logger .error ("cleanup_old_analysis_results: app_obj is not Flask instance")
    except Exception as e :
        logger .error (f"Eski analiz sonuçları temizlenirken hata: {e }",exc_info =True )
        # ERSIN Only attempt rollback if we're in a context
        if already_in_context :
            try :
                db .session .rollback ()
            except Exception :
                pass 
        else :
        # ERSIN Type narrowing: app_obj is Flask at this point (checked above)
            if isinstance (app_obj ,Flask ):
                try :
                    with app_obj .app_context ():
                        try :
                            db .session .rollback ()
                        except Exception :
                            pass 
                except Exception :
                    pass 

def cleanup_orphaned_files ():
    """
    Veritabanında kaydı olmayan yetim dosyaları temizler.
    """
    from flask import current_app as _current_app ,has_app_context as _has_app_context 

    def _resolve_app_obj ()->Flask |None :
        if _has_app_context ():
        # ERSIN Flask _get_current_object() exists but type checker doesn't recognize it
        # ERSIN Use getattr to safely access the method, then call it if it exists
        # ERSIN If it doesn't exist, fallback to _current_app directly (don't call it as a function)
            _get_current_object =getattr (_current_app ,'_get_current_object',None )
            if _get_current_object is not None and callable (_get_current_object ):
                result =_get_current_object ()
                # ERSIN result is Flask app instance at runtime
                return result if isinstance (result ,Flask )else None 
            else :
            # ERSIN Fallback: _get_current_object doesn't exist veya isn't callable, use _current_app directly
                return _current_app if isinstance (_current_app ,Flask )else None 
        try :
            from app import global_flask_app as _global_flask_app 
            return _global_flask_app 
        except Exception :
            return None 

    app_obj :Flask |None =_resolve_app_obj ()
    if app_obj is None :
        logger .warning ("cleanup_orphaned_files: Flask app bulunamadı (no app_context + global_flask_app None). Atlanıyor.")
        return 

        # ERSIN Type narrowing: app_obj is not None at this point
    assert app_obj is not None ,"app_obj should not be None after check"

    # ERSIN Eğer zaten bir app context içindeysek, nested context açmayalım
    # ERSIN Bu, database session corruption'ı önler
    # ERSIN Check context status immediately before execution to avoid race conditions
    def _execute_cleanup ():
        from app .models .analysis import Analysis 

        processed_folder =app_obj .config ['PROCESSED_FOLDER']

        if not os .path .exists (processed_folder ):
            return 

            # ERSIN Processed klasöründeki tüm dosya ve klasörleri kontrol et
        for item in os .listdir (processed_folder ):
            item_path =os .path .join (processed_folder ,item )

            # ERSIN Logs klasörünü atla
            if item =='logs':
                continue 

                # ERSIN frames_ ile başlayan klasörleri kontrol et
            if os .path .isdir (item_path )and item .startswith ('frames_'):
                analysis_id =item .replace ('frames_','')

                # ERSIN Validate analysis_id: gerekli be non-empty ve valid UUID format (36 chars ile dashes)
                # ERSIN bu prevents deleting folders ile invalid names like "frames_" veya "frames_invalid"
                if not analysis_id or len (analysis_id )!=36 or analysis_id .count ('-')!=4 :
                    logger .warning (f"Geçersiz analiz ID formatı, klasör atlanıyor: {item_path } (extracted ID: '{analysis_id }')")
                    continue 

                    # ERSIN Bu analiz ID'si veritabanında var mı kontrol et
                analysis_exists =Analysis .query .filter_by (id =analysis_id ).first ()

                if not analysis_exists :
                    logger .warning (f"Yetim analiz klasörü bulundu, siliniyor: {item_path }")
                    try :
                        shutil .rmtree (item_path )
                    except Exception as e :
                        logger .warning (f"Yetim klasör silinirken hata: {e }",exc_info =True )

        logger .info ("Yetim dosya temizliği tamamlandı.")

    try :
    # ERSIN Check context status immediately before execution to avoid race conditions
    # ERSIN If we're already in a context, execute directly without opening a new one
        if _has_app_context ():
            _execute_cleanup ()
        else :
        # ERSIN Context yoksa, yeni bir context aç
        # ERSIN Type narrowing: app_obj is Flask at this point (checked above)
            if isinstance (app_obj ,Flask ):
                with app_obj .app_context ():
                    _execute_cleanup ()
            else :
                logger .error ("cleanup_orphaned_files: app_obj is not Flask instance")
    except Exception as e :
        logger .error (f"Yetim dosya temizliği sırasında hata: {e }",exc_info =True )

def sync_model_versions_on_startup ():
    """
    Uygulama başlangıcında model versiyonlarını senkronize eder.
    Dosya sistemindeki v1, v2 gibi versiyonları veritabanına kaydeder.
    """
    try :
        logger .info ("🔄 Model versiyonları senkronize ediliyor...")

        # ERSIN Age model versiyonlarını senkronize et
        sync_age_model_versions_startup ()

        # ERSIN CLIP model versiyonlarını senkronize et
        sync_clip_model_versions_startup ()

        logger .info ("[OK] Model versiyonları başarıyla senkronize edildi!")

    except Exception as e :
        logger .error (f"❌ Model senkronizasyon hatası: {e }",exc_info =True )

def sync_age_model_versions_startup ():
    """Yaş modeli versiyonlarını startup'ta senkronize eder"""
    try :
        from app .models .content import ModelVersion 
        import json 

        # ERSIN Dosya sistemindeki versiyonları kontrol et
        versions_dir =os .path .join ('storage','models','age','custom_age_head','versions')

        if not os .path .exists (versions_dir ):
            logger .warning ("📊 Yaş modeli versiyonlar klasörü bulunamadı, atlanıyor...")
            return 

        version_folders =[d for d in os .listdir (versions_dir )
        if os .path .isdir (os .path .join (versions_dir ,d ))and d .startswith ('v')]

        if not version_folders :
            logger .warning ("📊 Yaş modeli versiyonu bulunamadı")
            return 

        logger .info (f"📊 {len (version_folders )} yaş modeli versiyonu bulundu: {version_folders }")

        for version_folder in version_folders :
            version_path =os .path .join (versions_dir ,version_folder )
            training_details_path =os .path .join (version_path ,'training_details.json')
            model_path =os .path .join (version_path ,'model.pth')

            # ERSIN Bu versiyon veritabanında var mı kontrol et
            existing =ModelVersion .query .filter_by (
            model_type ='age',
            version_name =version_folder 
            ).first ()

            if existing :
                logger .info (f"   ✓ {version_folder } zaten veritabanında mevcut")
                continue 

                # ERSIN Eğitim detaylarını oku
            if os .path .exists (training_details_path ):
                with open (training_details_path ,'r')as f :
                    training_details =json .load (f )

                    # ERSIN Versiyon numarasını belirle
                if version_folder .startswith ('v')and '_'in version_folder :
                    base_version =int (version_folder .split ('_')[0 ][1 :])
                    version_num =base_version +1 
                elif version_folder .startswith ('v'):
                    base_version =int (version_folder [1 :])
                    version_num =base_version +1 
                else :
                    version_num =2 

                    # ERSIN Yeni model versiyonu oluştur
                model_version =ModelVersion (
                model_type ='age',
                version =version_num ,
                version_name =version_folder ,
                file_path =version_path ,
                weights_path =model_path ,
                metrics =training_details .get ('metrics',{}),
                training_samples =training_details .get ('training_samples',0 ),
                validation_samples =training_details .get ('validation_samples',0 ),
                epochs =len (training_details .get ('history',{}).get ('train_loss',[])),
                is_active =False ,
                created_at =datetime .fromisoformat (training_details .get ('training_date',datetime .now ().isoformat ())),
                used_feedback_ids =[]
                )

                db .session .add (model_version )
                logger .info (f"   + {version_folder } veritabanına eklendi")
            else :
                logger .warning (f"   ⚠ {version_folder } için training_details.json bulunamadı")

                # ERSIN En son versiyonu aktif yap - ama model state'i kontrol et
        if version_folders :
        # ERSIN Model state dosyasını kontrol et
            try :
            # ERSIN Model state modülünü fresh reload et (cache'den değil)
                import app .utils .model_state 
                importlib .reload (app .utils .model_state )
                from app .utils .model_state import get_model_state 

                current_model_state =get_model_state ('age')# ERSIN thread-safe getter
                current_age_version =current_model_state .get ('active_version')
                logger .info (f"   📊 Model state age version: {current_age_version }")

                # ERSIN Eğer model state version 0 (reset) ise, en son versiyonu aktif yapma
                if current_age_version ==0 :
                    logger .info (f"   🔄 Model state version 0 (reset) - en son versiyon aktif yapılmıyor")
                    # ERSIN Base model'i aktif yap (eğer varsa)
                    base_model =ModelVersion .query .filter_by (model_type ='age',version =0 ).first ()
                    if base_model :
                        ModelVersion .query .filter_by (model_type ='age',is_active =True ).update ({'is_active':False })
                        base_model .is_active =True 
                        logger .info (f"   🎯 Base model (v0) aktif olarak ayarlandı")
                elif current_age_version is not None :
                # ERSIN Model state'te belirli bir versiyon var - onu aktif tut
                    logger .info (f"   🔒 Model state version {current_age_version } korunuyor - startup sync atlanıyor")
                    target_model =ModelVersion .query .filter_by (model_type ='age',version =current_age_version ).first ()
                    if target_model :
                        ModelVersion .query .filter_by (model_type ='age',is_active =True ).update ({'is_active':False })
                        target_model .is_active =True 
                        logger .info (f"   🎯 Version {current_age_version } ({target_model .version_name }) model state'e göre aktif tutuldu")
                    else :
                        logger .warning (f"   ⚠ Model state version {current_age_version } veritabanında bulunamadı, en son versiyon aktif yapılacak")
                        # ERSIN Fallback to latest
                        latest_version =max (version_folders ,key =lambda x :os .path .getctime (os .path .join (versions_dir ,x )))
                        latest_model =ModelVersion .query .filter_by (model_type ='age',version_name =latest_version ).first ()
                        if latest_model :
                            ModelVersion .query .filter_by (model_type ='age',is_active =True ).update ({'is_active':False })
                            latest_model .is_active =True 
                            logger .info (f"   🎯 {latest_version } aktif olarak ayarlandı (fallback)")
                else :
                # ERSIN Model state'te versiyon yok - normal durum, en son versiyonu aktif yap
                    latest_version =max (version_folders ,key =lambda x :os .path .getctime (os .path .join (versions_dir ,x )))
                    latest_model =ModelVersion .query .filter_by (
                    model_type ='age',
                    version_name =latest_version 
                    ).first ()

                    if latest_model :
                        ModelVersion .query .filter_by (model_type ='age',is_active =True ).update ({'is_active':False })
                        latest_model .is_active =True 
                        logger .info (f"   🎯 {latest_version } aktif olarak ayarlandı (varsayılan)")
            except Exception as e :
                logger .warning (f"   ⚠ Model state kontrol hatası: {e }",exc_info =True )
                # ERSIN Fallback - en son versiyonu aktif yap
                latest_version =max (version_folders ,key =lambda x :os .path .getctime (os .path .join (versions_dir ,x )))
                latest_model =ModelVersion .query .filter_by (
                model_type ='age',
                version_name =latest_version 
                ).first ()

                if latest_model :
                    ModelVersion .query .filter_by (model_type ='age',is_active =True ).update ({'is_active':False })
                    latest_model .is_active =True 
                    logger .info (f"   🎯 {latest_version } aktif olarak ayarlandı (fallback)")

        db .session .commit ()

    except Exception as e :
        logger .error (f"❌ Yaş modeli senkronizasyon hatası: {e }",exc_info =True )

def sync_clip_model_versions_startup ():
    """CLIP modeli versiyonlarını startup'ta senkronize eder"""
    try :
        from app .models .clip_training import CLIPTrainingSession 
        import json 

        # ERSIN Base model kaydını kontrol et ve ekle
        base_model =CLIPTrainingSession .query .filter_by (
        version_name ='base_openclip'
        ).first ()

        if not base_model :
        # ERSIN Base OpenCLIP model kaydını oluştur
            base_session =CLIPTrainingSession (
            version_name ='base_openclip',
            feedback_count =0 ,
            training_start =datetime (2025 ,1 ,1 ),# ERSIN Sabit tarih
            training_end =datetime (2025 ,1 ,1 ),
            status ='completed',
            model_path ='storage/models/clip/ViT-H-14-378-quickgelu_dfn5b/active_model',
            is_active =True ,# ERSIN Base model aktif olarak başlasın
            is_successful =True ,
            created_at =datetime (2025 ,1 ,1 )
            )

            # ERSIN Base model parametrelerini ayarla
            base_session .set_training_params ({
            'model_type':'ViT-H-14-378-quickgelu',
            'pretrained':'dfn5b',
            'description':'Base OpenCLIP model - pre-trained'
            })

            base_session .set_performance_metrics ({
            'description':'Pre-trained OpenCLIP model',
            'model_size':'ViT-H-14-378-quickgelu'
            })

            db .session .add (base_session )
            logger .info ("   + Base OpenCLIP model kaydı oluşturuldu")
        else :
            logger .info ("   ✓ Base OpenCLIP model zaten veritabanında mevcut")

            # ERSIN Dosya sistemindeki versiyonları kontrol et
        versions_dir =os .path .join ('storage','models','clip','versions')

        if not os .path .exists (versions_dir ):
            logger .warning ("🤖 CLIP versiyonlar klasörü bulunamadı, sadece base model kullanılacak")
            db .session .commit ()
            return 

        version_folders =[d for d in os .listdir (versions_dir )
        if os .path .isdir (os .path .join (versions_dir ,d ))and d .startswith ('v')]

        if not version_folders :
            logger .warning ("🤖 Fine-tuned CLIP versiyonu bulunamadı, sadece base model kullanılacak")
            db .session .commit ()
            return 

        logger .info (f"🤖 {len (version_folders )} fine-tuned CLIP versiyonu bulundu: {version_folders }")

        for version_folder in version_folders :
            version_path =os .path .join (versions_dir ,version_folder )
            metadata_path =os .path .join (version_path ,'metadata.json')
            model_path =os .path .join (version_path ,'pytorch_model.bin')

            # ERSIN Bu versiyon veritabanında var mı kontrol et
            existing =CLIPTrainingSession .query .filter_by (
            version_name =version_folder 
            ).first ()

            if existing :
                logger .info (f"   ✓ {version_folder } zaten veritabanında mevcut")
                continue 

                # ERSIN Metadata dosyasını oku
            if os .path .exists (metadata_path ):
                with open (metadata_path ,'r')as f :
                    metadata =json .load (f )

                    # ERSIN Yeni CLIP training session oluştur
                training_session =CLIPTrainingSession (
                version_name =version_folder ,
                feedback_count =metadata .get ('feedback_count',0 ),
                training_start =datetime .fromisoformat (metadata .get ('training_start',datetime .now ().isoformat ())),
                training_end =datetime .fromisoformat (metadata .get ('training_end',datetime .now ().isoformat ())),
                status ='completed',
                model_path =model_path ,
                is_active =False ,
                is_successful =True ,
                created_at =datetime .fromisoformat (metadata .get ('created_at',datetime .now ().isoformat ()))
                )

                # ERSIN Training parametrelerini ayarla
                if 'training_params'in metadata :
                    training_session .set_training_params (metadata ['training_params'])

                    # ERSIN Performance metriklerini ayarla
                if 'performance_metrics'in metadata :
                    training_session .set_performance_metrics (metadata ['performance_metrics'])

                db .session .add (training_session )
                logger .info (f"   + {version_folder } veritabanına eklendi")
            else :
                logger .warning (f"   ⚠ {version_folder } için metadata.json bulunamadı")

                # ERSIN En son fine-tuned versiyonu aktif yap (varsa)
        if version_folders :
            latest_version =max (version_folders ,key =lambda x :os .path .getctime (os .path .join (versions_dir ,x )))
            latest_session =CLIPTrainingSession .query .filter_by (
            version_name =latest_version 
            ).first ()

            if latest_session :
            # ERSIN Tüm versiyonları pasif yap (base model dahil)
                CLIPTrainingSession .query .update ({'is_active':False })
                # ERSIN En son fine-tuned versiyonu aktif yap
                latest_session .is_active =True 
                logger .info (f"   🎯 {latest_version } aktif olarak ayarlandı (base model yerine)")

        db .session .commit ()

    except Exception as e :
        logger .error (f"❌ CLIP senkronizasyon hatası: {e }",exc_info =True )

def register_global_routes (app ):
    """
    Global route'ları kaydeder.
    
    Args:
        app: Flask uygulaması.
    """
    @app .route ('/processed/<path:filename>')
    def serve_processed_file (filename ):
        processed_folder =os .path .join (app .config ['STORAGE_FOLDER'],'processed')
        return send_from_directory (processed_folder ,filename )

def register_error_handlers (flask_app ):
    """Register error handlers"""
    @flask_app .errorhandler (404 )
    def page_not_found (e ):
        from flask import jsonify 
        return jsonify ({'error':'Not found'}),404 

    @flask_app .errorhandler (500 )
    def internal_server_error (e ):
        from flask import jsonify 
        return jsonify ({'error':'Internal server error'}),500 