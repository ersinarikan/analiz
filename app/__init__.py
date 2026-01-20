"""
WSANALIZ Flask Uygulaması - Ana Modül
"""
import logging
import os
import shutil
import importlib
from datetime import datetime
from typing import Any, Literal, overload

from flask import Flask, send_from_directory, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import config

from app.json_encoder import CustomJSONEncoder
from app.socketio_instance import socketio

# Global minimal socketio reference (runtime'da set edilecek)
_current_running_socketio = None
global_flask_app = None  # ERSIN Ana Flask app nesnesi, background thread'ler için

# ERSIN Memory utils - isteğe bağlı import
try:
    from app.utils.memory_utils import initialize_memory_management
except ImportError:
    initialize_memory_management = None

# ERSIN Global Flask eklentileri
db = SQLAlchemy()
migrate = Migrate()

# SQLite robustness in multi-process (web + worker) setups:
# - Reduce "database is locked" by enabling WAL + busy_timeout and letting SQLite wait.
# - Safe no-op for non-SQLite databases.
try:
    import sqlite3
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):  # type: ignore[no-redef]
        try:
            if not isinstance(dbapi_connection, sqlite3.Connection):
                return
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")  # 30s
            cursor.close()
        except Exception:
            # Best-effort; do not prevent boot.
            pass
except Exception:
    pass

logger = logging.getLogger("wsanaliz.app_init")
logging.basicConfig(level=logging.INFO)

# ERSIN ===============================
# ERSIN 🎯 STANDARD FLASK-SOCKETIO PATTERN
# ERSIN ===============================
# ERSIN DİKKAT: SocketIO instance'ı SADECE burada, uygulama başlatılırken oluşturulur ve set edilir.
# ERSIN Başka hiçbir yerde yeni SocketIO instance'ı yaratılmayacak veya set edilmeyecek!

def register_blueprints_from_list(app, blueprint_defs):
    """
    blueprint_defs: List of tuples (import_path, attr_name, alias)
    - import_path: Python import path as string (e.g. 'app.routes.main_routes')
    - attr_name: Attribute name in the module (e.g. 'main_bp' or 'bp')
    - alias: Optional alias for logging (e.g. 'file_bp'), can be None
    """
    import importlib
    import traceback
    logger = logging.getLogger("wsanaliz.app_init")
    blueprints_to_register = []
    for import_path, attr_name, alias in blueprint_defs:
        try:
            module = importlib.import_module(import_path)
            bp = getattr(module, attr_name)
            blueprints_to_register.append(bp)
            logger.info(f"Blueprint imported: {import_path}.{attr_name} as {alias or attr_name}")
        except ImportError as e:
            logger.error(f"{import_path} import failed: {e}")
            logger.debug(f"Import traceback: {traceback.format_exc()}")
        except AttributeError as e:
            logger.error(f"{import_path} has no attribute {attr_name}: {e}")
            logger.debug(f"AttributeError traceback: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"{import_path} failed with unexpected error: {e}")
            logger.debug(f"Unexpected error traceback: {traceback.format_exc()}")
    for bp in blueprints_to_register:
        try:
            app.register_blueprint(bp)
            logger.info(f"Blueprint registered: {bp.name}")
        except Exception as e:
            logger.error(f"Failed to register blueprint {bp.name}: {e}")
    logger.info(f"Total blueprints registered: {len(blueprints_to_register)}")
    return blueprints_to_register

@overload
def create_app(config_name: str = "default", return_socketio: Literal[False] = False) -> Flask: ...


@overload
def create_app(config_name: str = "default", *, return_socketio: Literal[True]) -> tuple[Flask, Any]: ...


def create_app(config_name: str = "default", return_socketio: bool = False) -> Flask | tuple[Flask, Any]:
    """
    Flask uygulaması fabrikası.
    Args:
        config_name: Kullanılacak konfigürasyon adı.
        return_socketio: True ise (flask_app, socketio) tuple döner.
    Returns:
        return_socketio=False: sadece Flask app.
        return_socketio=True: (flask_app, minimal_socketio)
    """
    flask_app = Flask(__name__)
    flask_app.config.from_object(config[config_name])

    # Set global app reference as early as possible for background/SocketIO helpers.
    # NOTE: Some helpers historically relied on `global_flask_app` (legacy). Keep it safe.
    global global_flask_app
    global_flask_app = flask_app

    # Server-side session store (cookie holds session id only)
    # NOTE: Redis is optional; filesystem sessions work out of the box.
    try:
        from flask_session import Session

        if (flask_app.config.get("SESSION_TYPE") or "").strip().lower() == "redis":
            try:
                import redis  # type: ignore

                redis_url = (os.environ.get("WSANALIZ_REDIS_URL") or "redis://localhost:6379/0").strip()
                flask_app.config["SESSION_REDIS"] = redis.Redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"SESSION_TYPE=redis ama Redis init başarısız, filesystem'e düşülüyor: {e}")
                flask_app.config["SESSION_TYPE"] = "filesystem"

        Session(flask_app)
        logger.info(f"Flask-Session initialized (type={flask_app.config.get('SESSION_TYPE')})")
    except Exception as e:
        logger.warning(f"Flask-Session init atlandı: {e}")
    
    # ERSIN Flask uzantılarını başlat
    db.init_app(flask_app)
    # Flask-Migrate'ı da initialize et (CLI migrations için gerekli)
    migrate.init_app(flask_app, db)
    
    # ERSIN ✅ MİNİMAL PATTERN: Optimize SocketIO kurulumu
    from flask_socketio import SocketIO

    # Cross-process SocketIO events:
    # - If analysis runs in another process, you may want a message_queue (Redis) so emits propagate.
    # - But Redis must be optional for backward compatibility.
    def _parse_bool_env(name: str) -> bool | None:
        if name not in os.environ:
            return None
        val = (os.environ.get(name) or "").strip().lower()
        if val in {"1", "true", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "no", "n", "off", ""}:
            return False
        return None

    require_redis = _parse_bool_env("SOCKETIO_REQUIRE_REDIS") is True

    socketio_message_queue: str | None = None
    if os.environ.get("SOCKETIO_MESSAGE_QUEUE"):
        socketio_message_queue = os.environ.get("SOCKETIO_MESSAGE_QUEUE")
    else:
        # Auto-enable Redis message_queue only when queue backend is Redis.
        if (os.environ.get("WSANALIZ_QUEUE_BACKEND") or "").lower() == "redis":
            socketio_message_queue = os.environ.get("WSANALIZ_REDIS_URL", "redis://localhost:6379/0")

    # Validate Redis availability if message_queue is configured.
    if socketio_message_queue:
        try:
            import redis  # type: ignore

            r = redis.from_url(socketio_message_queue, socket_timeout=0.2, socket_connect_timeout=0.2)
            r.ping()
        except Exception as e:
            if require_redis:
                raise
            logger.warning(
                f"SocketIO message_queue devre dışı bırakıldı (Redis erişilemiyor): {socketio_message_queue}. Hata: {e}"
            )
            socketio_message_queue = None

    minimal_socketio = SocketIO(
        flask_app,
        cors_allowed_origins="*",
        ping_timeout=720,  # ERSIN Uzun analizler için 12 dakika timeout
        ping_interval=60,  # ERSIN Her dakika ping ile browser arka plan uyumluluğu
        logger=False,      # ERSIN Verbose logging kapat
        engineio_logger=False,
        message_queue=socketio_message_queue,
        async_mode='eventlet'  # ERSIN Eventlet async mode
    )
    
    # ERSIN Global instance'ı güncelleyelim - emit_analysis_progress için
    # socketio_instance is intentionally tiny to avoid circular imports.
    from app.socketio_instance import set_socketio
    set_socketio(minimal_socketio)  # TEK NOKTA SET!
    
    # ERSIN ✅ MİNİMAL PATTERN: Event handler kayıtları
    
    @minimal_socketio.on('connect')
    def handle_connect():
        from flask import has_request_context, request, session
        from flask_socketio import emit
        # SocketIO handlers should have a request context, but be defensive:
        # if there is no request context, Flask session proxy would raise RuntimeError.
        if not has_request_context():
            return False

        # Reject WS connections when not authenticated (unless auth is disabled)
        if not flask_app.config.get("WSANALIZ_AUTH_DISABLED", False) and not session.get("pam_user"):
            return False
        # Flask-SocketIO request.sid exists but type checker doesn't recognize it
        session_id = getattr(request, 'sid', None)  # type: ignore[attr-defined]
        print(f"🎉🎉🎉 MİNİMAL CONNECT! Session: {session_id}")
        emit('connected', {'message': 'Minimal pattern bağlantısı başarılı!'})
        
    @minimal_socketio.on('disconnect')  
    def handle_disconnect():
        from flask import request
        import logging
        from flask import current_app, has_app_context
        logger = logging.getLogger(__name__)
        
        # Flask-SocketIO request.sid exists but type checker doesn't recognize it
        session_id = getattr(request, 'sid', None)  # type: ignore[attr-defined]
        print(f"❌❌❌ MİNİMAL DISCONNECT! Session: {session_id}")
        
        # ERSIN Bu WebSocket session'ı ile ilişkili çalışan analizleri bul ve iptal et
        try:
            # SocketIO event handler'ları her zaman Flask request/app context garantisi vermeyebilir.
            # Prefer Flask context proxy if available; otherwise fall back to global app instance.
            if has_app_context():
                # Flask _get_current_object() exists but type checker doesn't recognize it
                app_obj = getattr(current_app, '_get_current_object', lambda: current_app)()  # type: ignore[attr-defined]
            else:
                from app import global_flask_app as app_obj

            if app_obj is None:
                logger.warning(
                    "WebSocket disconnect cleanup: Flask app bulunamadı (no app_context + global_flask_app None). "
                    "Shutdown sırasında normal olabilir; cleanup atlanıyor."
                )
                return

            # Validate app_obj looks like a Flask app instance (avoid calling app_context() on a bad object).
            try:
                from flask import Flask as _Flask
                if not isinstance(app_obj, _Flask):
                    logger.error(
                        f"WebSocket disconnect cleanup: app_obj Flask değil (type={type(app_obj)}). "
                        "Cleanup atlanıyor."
                    )
                    return
            except Exception:
                # If Flask import/type-check fails, still attempt best-effort below.
                pass

            # DB işlemlerini explicit app_context içinde yap.
            try:
                app_ctx = app_obj.app_context()
            except Exception as ctx_err:
                logger.warning(f"WebSocket disconnect cleanup: app_context oluşturulamadı (muhtemel shutdown). Hata: {ctx_err}")
                return

            with app_ctx:
                from app.models.analysis import Analysis
                from sqlalchemy.exc import OperationalError

                # ERSIN 1. Veritabanındaki ilişkili analizleri bul
                try:
                    active_analyses = Analysis.query.filter(
                        Analysis.websocket_session_id == session_id,
                        Analysis.status.in_(['pending', 'processing'])  # type: ignore[attr-defined]
                    ).all()
                except OperationalError as op_err:
                    # Schema drift case (e.g., websocket_session_id missing) should not crash the service.
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

                # Persist cancellations before queue cleanup so fresh queries see is_cancelled.
                if cancelled_count > 0:
                    db.session.commit()
                
                # ERSIN 2. Kuyruktaki analizleri de kontrol et
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
            logger.error(f"❌ WebSocket disconnect cleanup hatası: {str(e)}")
            # Rollback must run inside an app context; otherwise Flask-SQLAlchemy may raise
            # "Working outside of application context" and hide the original error.
            try:
                from flask import current_app as _current_app, has_app_context as _has_app_context
                if _has_app_context():
                    # Flask _get_current_object() exists but type checker doesn't recognize it
                    _app_obj = getattr(_current_app, '_get_current_object', lambda: _current_app)()  # type: ignore[attr-defined]
                else:
                    from app import global_flask_app as _app_obj

                if _app_obj is None:
                    return

                with _app_obj.app_context():
                    try:
                        db.session.rollback()
                    except Exception:
                        pass
            except Exception:
                pass
        
    @minimal_socketio.on('ping')
    def handle_ping(data):
        from flask import request
        from flask_socketio import emit
        # Flask-SocketIO request.sid exists but type checker doesn't recognize it
        session_id = getattr(request, 'sid', None)  # type: ignore[attr-defined]
        print(f"🏓🏓🏓 MİNİMAL PING! Session: {session_id}, Data: {data}")
        emit('pong', {'message': 'Minimal PONG!', 'data': data})

    @minimal_socketio.on('join_analysis')
    def handle_join_analysis(data):
        from flask import request
        from flask_socketio import emit, join_room
        # Flask-SocketIO request.sid exists but type checker doesn't recognize it
        session_id = getattr(request, 'sid', None)  # type: ignore[attr-defined]
        print(f"🔍🔍🔍 MİNİMAL JOIN_ANALYSIS! Session: {session_id}, Data: {data}")
        
        if data and 'analysis_id' in data:
            analysis_id = data['analysis_id']
            room = f"analysis_{analysis_id}"
            
            # ERSIN Room'a katıl
            join_room(room)
            
            # ERSIN DEBUG: Room membership kontrol et
            try:
                # SocketIO server attribute exists but type checker doesn't recognize it
                server = getattr(minimal_socketio, 'server', None)  # type: ignore[attr-defined]
                if server is None:
                    raise AttributeError("SocketIO server not available")
                room_members = server.manager.get_participants(namespace='/', room=room)
                room_members_list = list(room_members)
                print(f"🔍🔍🔍 MİNİMAL JOIN: Room {room} members after join: {room_members_list}")
            except Exception as room_err:
                print(f"🔍 MİNİMAL JOIN: Room membership check failed: {room_err}")
            
            # ERSIN Başarı mesajı gönder
            emit('joined_analysis', {
                'analysis_id': analysis_id,
                'room': room,
                'message': f'Analysis {analysis_id} room\'una katıldınız (minimal)',
                'source': 'minimal-handler'
            })
        else:
            print(f"❌ MİNİMAL JOIN_ANALYSIS: No analysis_id in data")
            
    logger.info("[OK] Minimal pattern SocketIO handlers registered!")
    print("[OK] Minimal pattern SocketIO handlers registered!")
    
    # Backward-compatible attachment: some code may look this up from the Flask app object.
    # The canonical reference for cross-module usage should be via app.socketio_instance (proxy),
    # which is set above via set_socketio(minimal_socketio).
    # Dynamic attribute assignment - Flask allows this
    setattr(flask_app, 'minimal_socketio', minimal_socketio)  # type: ignore[attr-defined]
    
    # ERSIN JSON encoder'ı ayarla
    setattr(flask_app, 'json_encoder', CustomJSONEncoder)  # type: ignore[attr-defined]
    
    # ERSIN Security middleware başlat
    from app.middleware.security_middleware import SecurityMiddleware
    SecurityMiddleware(flask_app)

    # Auth middleware (redirects unauthenticated users to /login)
    from app.middleware.auth_middleware import AuthMiddleware
    AuthMiddleware(flask_app)
    
    # ERSIN Performans için memory management başlat
    try:
        # Offline tools / one-shot scripts (e.g. prompt_sanity_check) may run on machines where
        # CUDA init is slow or undesired. Allow opting out via env.
        disable_mm = (os.environ.get("WSANALIZ_DISABLE_MEMORY_MANAGEMENT", "") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
        if initialize_memory_management:
            if disable_mm:
                logger.warning("Memory management disabled via WSANALIZ_DISABLE_MEMORY_MANAGEMENT=1")
            else:
                initialize_memory_management()
                logger.info("Memory management initialized")
        else:
            logger.warning("Memory management not available (optional dependency)")
    except Exception as e:
        logger.warning(f"Memory management initialization failed: {e}", exc_info=True)
    
    # ERSIN Blueprint'leri error handling ile kaydet (refaktör)
    blueprint_defs = [
        ("app.routes.auth_routes", "auth_bp", None),
        ("app.routes.main_routes", "main_bp", None),
        ("app.routes.file_routes", "bp", "file_bp"),
        ("app.routes.analysis_routes", "bp", "analysis_bp"),
        ("app.routes.feedback_routes", "bp", "feedback_bp"),
        ("app.routes.settings_routes", "bp", "settings_bp"),
        ("app.routes.model_management_routes", "model_management_bp", None),
        ("app.routes.model_routes", "bp", "model_bp"),
        ("app.routes.queue_routes", "queue_bp", None),
        ("app.routes.performance_routes", "performance_bp", None),
        ("app.routes.debug_routes", "bp", "debug_bp"),
        ("app.routes.ensemble_routes", "ensemble_bp", None),
        ("app.routes.clip_training_routes", "clip_training_bp", None),
    ]
    register_blueprints_from_list(flask_app, blueprint_defs)

    # Error handlers (404/500) – previously disabled for circular-import concerns
    # NOTE: This function is self-contained and only imports jsonify inside handlers.
    register_error_handlers(flask_app)

    # Global/static routes (e.g. serving processed artifacts) should be registered at app creation time,
    # not only when initialize_app() is called, otherwise some entrypoints/tests will miss them.
    register_global_routes(flask_app)
    
    # ERSIN WebSocket event handlers registration - ESKİ YÖNTEM DEVRE DIŞI
    # ERSIN Error handlers - geçici olarak kapalı (circular import)
    
    # ERSIN Startup görevleri
    with flask_app.app_context():
        try:
            db.create_all()
            logger.info("Veritabanı tabloları oluşturuldu/kontrol edildi")
        except Exception as e:
            logger.error(f"Startup görevleri hatası: {str(e)}", exc_info=True)
    
    # Geriye uyumluluk: default sadece Flask app döndür.
    # SocketIO instance'a ihtiyaç olan yerler return_socketio=True ile tuple alabilir.
    return (flask_app, minimal_socketio) if return_socketio else flask_app


# NOTE:
# `initialize_app()` calls `check_and_run_migrations()`. While defining the function later in the file
# is valid in Python (module is fully loaded first), some reviewers flag this as confusing.
# Keep a small wrapper here so the function name is defined before `initialize_app()`.
def check_and_run_migrations():
    return _check_and_run_migrations()

def initialize_app(app):
    """
    Uygulamayı başlatır ve gerekli temizlik işlemlerini yapar.
    Bu fonksiyon sadece ana süreçte çağrılmalıdır.
    
    Args:
        app: Flask uygulaması
    """
    with app.app_context():
        # Veritabanı başlatma (sadece yoksa oluştur, mevcut olanı silme)
        db_path = app.config.get('SQLALCHEMY_DATABASE_URI', '').replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            db_path = os.path.join(app.root_path, db_path)
        logger.info(f"Veritabanı yolu: {db_path}")
        
        # Sadece veritabanı yoksa oluştur, mevcut olanı silme
        if not os.path.exists(db_path):
            logger.info("Veritabanı bulunamadı, yeni veritabanı oluşturuluyor.")
            db.create_all()
            logger.info("Yeni veritabanı oluşturuldu.")
        else:
            logger.info("Mevcut veritabanı kullanılıyor.")
            db.create_all()
            # Mevcut veritabanı için migration kontrolü yap
            check_and_run_migrations()
        
        # Klasörlerin oluşturulması ve temizlenmesi
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
        
        # Upload klasörü temizlemeyi devre dışı bırak - Eğitim verisi güvenliği için
        # clean_folder(app.config['UPLOAD_FOLDER'])  # ← EĞİTİM VERİSİ GÜVENLİĞİ İÇİN KAPATILDI
        # clean_folder(app.config['PROCESSED_FOLDER'])  # Analiz sonuçlarını korumak için devre dışı
        
        # Eski analiz sonuçlarını temizle (7 günden eski olanları)
        cleanup_old_analysis_results(days_old=7)
        
        # Model versiyonlarını senkronize et (VT oluşturulduktan sonra)
        sync_model_versions_on_startup()
        
        # Worker crash recovery: "processing" durumunda olan ama uzun süredir ilerlemeyen analizleri kontrol et
        recover_stuck_analyses()
        
        # Analiz kuyruğu servisini başlat (sadece memory backend'te)
        # Redis backend'te queue processing ayrı worker prosesinde yapılır.
        try:
            from app.services import queue_service as _queue_service
            if not getattr(_queue_service, "is_redis_backend", lambda: False)():
                from app.services.queue_service import start_processor
                logger.info("Analiz kuyruğu servisi (memory backend) başlatılıyor...")
                start_processor()
                logger.info("Analiz kuyruğu servisi başlatıldı.")
            else:
                logger.info("Redis queue backend aktif: web prosesinde queue processor başlatılmıyor.")
        except Exception as e:
            logger.warning(f"Queue processor init atlandı: {e}")

    # Global routes are registered in create_app().

def clean_folder(folder_path):
    """
    Belirtilen klasörü temizler.
    
    Args:
        folder_path (str): Temizlenecek klasörün yolu.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Eğer 'logs' klasörüyse (veya 'logs' klasörünün içindeysek) silme
            if filename == 'logs' and os.path.isdir(file_path):
                logger.warning(f"'{file_path}' log klasörü atlanıyor, silinmeyecek.")
                continue # logs klasörünü silme, içini de boşaltma

            if os.path.isfile(file_path) or os.path.islink(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Dosya silinirken hata (atlanıyor): {file_path}, Hata: {e}", exc_info=True)
            elif os.path.isdir(file_path):
                try:
                    shutil.rmtree(file_path)
                except Exception as e:
                    logger.warning(f"Klasör silinirken hata (atlanıyor): {file_path}, Hata: {e}", exc_info=True)
    else:
        os.makedirs(folder_path, exist_ok=True)

def _check_and_run_migrations():
    """
    Veritabanı migration kontrolü yapar ve gerekli kolumları ekler.
    """
    from flask import current_app

    conn = None

    def _parse_bool_env(name: str) -> bool | None:
        if name not in os.environ:
            return None
        val = (os.environ.get(name) or "").strip().lower()
        if val in {"1", "true", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "no", "n", "off", ""}:
            return False
        return None

    strict_env = _parse_bool_env("WSANALIZ_MIGRATIONS_STRICT")
    strict = (not current_app.config.get("TESTING", False)) if strict_env is None else bool(strict_env)
    try:
        logger.info("🔄 Veritabanı migration kontrolü yapılıyor...")

        # Use SQLAlchemy's engine connection to avoid separate sqlite3 path handling/locking surprises.
        conn = db.engine.raw_connection()
        cursor = conn.cursor()

        # analyses tablosundaki kolonları kontrol et
        cursor.execute("PRAGMA table_info(analyses)")
        columns = [column[1] for column in cursor.fetchall()]

        migrations_needed = []
        errors: list[str] = []

        # websocket_session_id kolonu var mı?
        if 'websocket_session_id' not in columns:
            migrations_needed.append(('websocket_session_id', 'TEXT'))

        # is_cancelled kolonu var mı?
        if 'is_cancelled' not in columns:
            # SQLite has no native BOOLEAN; use INTEGER affinity.
            migrations_needed.append(('is_cancelled', 'INTEGER DEFAULT 0'))

        # Migration'ları uygula
        applied = 0
        for column_name, column_def in migrations_needed:
            try:
                sql = f"ALTER TABLE analyses ADD COLUMN {column_name} {column_def}"
                cursor.execute(sql)
                applied += 1
                logger.info(f"✅ Migration: {column_name} kolonu eklendi")
            except Exception as e:
                msg = str(e)
                # Duplicate column is safe to ignore. Any other error can leave schema inconsistent.
                if "duplicate column name" in msg.lower():
                    logger.info(f"ℹ️ Migration: {column_name} zaten var (duplicate), atlanıyor")
                else:
                    logger.error(f"❌ Migration hatası ({column_name}): {msg}", exc_info=True)
                    errors.append(f"{column_name}: {msg}")

        # commit sırasında hata olursa connection kapanmadan kalmasın
        try:
            conn.commit()
        except Exception as commit_err:
            logger.error(f"❌ Migration commit hatası: {commit_err}", exc_info=True)
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        if migrations_needed:
            logger.info(f"🎉 {applied}/{len(migrations_needed)} migration uygulandı!")
        else:
            logger.info("✅ Veritabanı şeması güncel, migration gerekmiyor")

        # Verify columns exist after migrations (db.create_all() won't add columns).
        cursor.execute("PRAGMA table_info(analyses)")
        final_cols = {column[1] for column in cursor.fetchall()}
        # These columns are used for cancellation/session tracking, but the app can still boot without them.
        # We treat them as optional to avoid bringing down production on an imperfect schema; functionality
        # that relies on them should degrade gracefully.
        optional_cols = {"websocket_session_id", "is_cancelled"}
        missing_optional = sorted(optional_cols - final_cols)
        if missing_optional:
            errors.append(f"missing optional columns after migration: {missing_optional}")

        if errors:
            err_msg = f"Database migrations incomplete/failed: {errors}"
            # Do not crash the app for optional-column issues; log loudly.
            logger.error(f"⚠️ {err_msg}")

    except Exception as e:
        logger.error(f"❌ Migration kontrolü hatası: {str(e)}", exc_info=True)
        # Only fail-fast for truly fatal migration issues (e.g., broken DB). Optional column drift should not
        # prevent the service from starting.
        if strict:
            raise
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

def recover_stuck_analyses():
    """
    Worker crash recovery: "processing" durumunda olan ama uzun süredir ilerlemeyen 
    analizleri kontrol edip "failed" yapar.
    
    Bu, worker process segfault veya crash olduğunda analizlerin takılı kalmasını önler.
    """
    try:
        from app.models.analysis import Analysis

        # In Redis queue architecture, a dedicated worker is responsible for processing.
        # If the worker heartbeat is fresh, do NOT auto-fail "processing" analyses on web startup;
        # this causes false-failures + SQLite lock contention.
        try:
            if (os.environ.get("WSANALIZ_QUEUE_BACKEND") or "").strip().lower() == "redis":
                import time as _time
                import redis  # type: ignore

                redis_url = (os.environ.get("WSANALIZ_REDIS_URL") or "redis://localhost:6379/0").strip()
                heartbeat_key = (os.environ.get("WSANALIZ_WORKER_HEARTBEAT_KEY") or "wsanaliz:worker:last_heartbeat").strip()
                r = redis.Redis.from_url(redis_url, decode_responses=True)
                hb = r.get(heartbeat_key)
                if hb:
                    try:
                        # Redis get() returns str when decode_responses=True
                        if isinstance(hb, str):
                            hb_ts = float(hb)
                            if (_time.time() - hb_ts) < 30:
                                logger.info("🔍 Worker crash recovery: Worker heartbeat taze (Redis). Recovery atlandı (false-fail + DB lock önleme).")
                                return
                    except Exception:
                        pass
        except Exception:
            # Best-effort; if Redis is unavailable, fall back to DB-only recovery.
            pass
        
        # "processing" durumunda olan analizleri bul
        stuck_analyses = Analysis.query.filter(
            Analysis.status == 'processing'
        ).all()
        
        if not stuck_analyses:
            logger.info("🔍 Worker crash recovery: Takılı analiz bulunamadı.")
            return
        
        recovered_count = 0
        timeout_minutes = 10  # 10 dakikadan fazla "processing" durumunda olan analizler takılı sayılır
        
        # Use UTC consistently; stored timestamps should be naive-UTC.
        now = datetime.utcnow()
        for analysis in stuck_analyses:
            # start_time can be NULL in existing DBs; fall back to created_at (or updated_at if present)
            start_time = analysis.start_time
            created_at = getattr(analysis, "created_at", None)
            updated_at = getattr(analysis, "updated_at", None)

            # Prefer start_time (if sane), then updated_at, then created_at.
            # Guard against mixed timezone origins / clock skew where timestamps can end up in the future.
            ref_time = start_time or updated_at or created_at
            if not ref_time:
                # Extremely rare/corrupt case: no timestamps at all -> fail to avoid indefinite stuck state.
                logger.warning(
                    f"⚠️ Worker crash recovery: Analiz #{analysis.id} 'processing' ama timestamp yok "
                    f"(start_time/created_at/updated_at NULL). 'failed' olarak işaretleniyor."
                )
                analysis.status = 'failed'
                analysis.error_message = "Analiz takılı kaldı (timestamp bulunamadı) ve otomatik olarak başarısız işaretlendi."
                analysis.end_time = now
                recovered_count += 1
                continue

            # If the chosen ref_time is in the future, try other timestamps that are <= now.
            # If ALL timestamps are in the future, do not auto-fail (likely clock skew); keep 'processing'.
            if ref_time > now:
                fallback = None
                if start_time and start_time <= now:
                    fallback = start_time
                elif updated_at and updated_at <= now:
                    fallback = updated_at
                elif created_at and created_at <= now:
                    fallback = created_at

                if fallback is None:
                    logger.warning(
                        "⚠️ Worker crash recovery: Analiz #%s 'processing' ama tüm timestamp'ler gelecekte "
                        "(start_time=%s, updated_at=%s, created_at=%s, now=%s). "
                        "Muhtemel timezone/clock skew; auto-fail uygulanmadı.",
                        analysis.id,
                        start_time,
                        updated_at,
                        created_at,
                        now,
                    )
                    continue

                ref_time = fallback

            elapsed = now - ref_time
            elapsed_minutes = elapsed.total_seconds() / 60

            if elapsed_minutes > timeout_minutes:
                logger.warning(
                    f"🔧 Worker crash recovery: Analiz #{analysis.id} "
                    f"{elapsed_minutes:.1f} dakikadır 'processing' durumunda, "
                    f"'failed' olarak işaretleniyor (muhtemelen worker crash)."
                )
                analysis.status = 'failed'
                analysis.error_message = (
                    f"Worker process crash nedeniyle analiz başarısız oldu. "
                    f"Analiz {elapsed_minutes:.1f} dakikadır işleniyordu."
                )
                analysis.end_time = now
                recovered_count += 1
        
        if recovered_count > 0:
            db.session.commit()
            logger.info(f"✅ Worker crash recovery: {recovered_count} takılı analiz 'failed' olarak işaretlendi.")
        else:
            logger.info("ℹ️ Worker crash recovery: Tüm 'processing' analizler aktif görünüyor.")
            
    except Exception as e:
        logger.error(f"❌ Worker crash recovery hatası: {e}", exc_info=True)
        try:
            db.session.rollback()
        except Exception:
            pass

def cleanup_old_analysis_results(days_old=7):
    """
    Belirli bir yaştan eski analiz sonuçlarını ve ilgili dosyaları temizler.
    
    Args:
        days_old: Kaç günden eski analizlerin temizleneceği (varsayılan: 7)
    """
    from flask import current_app as _current_app, has_app_context as _has_app_context

    def _resolve_app_obj():
        if _has_app_context():
            # Flask _get_current_object() exists but type checker doesn't recognize it
            return getattr(_current_app, '_get_current_object', lambda: _current_app)()  # type: ignore[attr-defined]
        try:
            from app import global_flask_app as _global_flask_app
            return _global_flask_app
        except Exception:
            return None

    app_obj = _resolve_app_obj()
    if app_obj is None:
        logger.warning("cleanup_old_analysis_results: Flask app bulunamadı (no app_context + global_flask_app None). Atlanıyor.")
        return

    try:
        with app_obj.app_context():
            from datetime import datetime, timedelta
            from app.models.analysis import Analysis
            from app.models.clip_training import CLIPTrainingSession  # CLIP training model import

            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            logger.info(f"Eski analiz sonuçları temizleniyor: {cutoff_date} tarihinden eski olanlar")

            # Eski analizleri bul (created_at yerine start_time kullan)
            old_analyses = Analysis.query.filter(Analysis.start_time < cutoff_date).all()

            if not old_analyses:
                logger.info("Temizlenecek eski analiz bulunamadı.")
                return

            logger.info(f"{len(old_analyses)} eski analiz bulundu, temizleniyor...")

            processed_root = app_obj.config['PROCESSED_FOLDER']

            # Her analiz için ilgili dosyaları temizle
            for analysis in old_analyses:
                try:
                    analysis_folder = os.path.join(processed_root, f"frames_{analysis.id}")
                    if os.path.exists(analysis_folder):
                        shutil.rmtree(analysis_folder)
                        logger.info(f"Analiz klasörü silindi: {analysis_folder}")

                    # İşlenmiş resim dosyasını sil (alan mevcutsa)
                    processed_image_rel = getattr(analysis, 'processed_image_path', None)
                    if processed_image_rel:
                        processed_file = os.path.join(processed_root, processed_image_rel)
                        if os.path.exists(processed_file):
                            os.unlink(processed_file)
                            logger.info(f"İşlenmiş resim silindi: {processed_file}")

                    # En yüksek riskli kare dosyasını sil
                    if analysis.highest_risk_frame:
                        risk_frame_file = os.path.join(processed_root, analysis.highest_risk_frame)
                        if os.path.exists(risk_frame_file):
                            os.unlink(risk_frame_file)
                            logger.info(f"En yüksek riskli kare silindi: {risk_frame_file}")

                    # Veritabanından analizi sil (cascade ile ilgili kayıtlar da silinir)
                    db.session.delete(analysis)

                except Exception as e:
                    logger.warning(f"Analiz {analysis.id} temizlenirken hata: {e}", exc_info=True)
                    continue

            # Değişiklikleri kaydet
            db.session.commit()
            logger.info(f"{len(old_analyses)} eski analiz başarıyla temizlendi.")

            # Artık kullanılmayan dosyaları da temizle
            cleanup_orphaned_files()

    except Exception as e:
        logger.error(f"Eski analiz sonuçları temizlenirken hata: {e}", exc_info=True)
        try:
            with app_obj.app_context():
                try:
                    db.session.rollback()
                except Exception:
                    pass
        except Exception:
            pass

def cleanup_orphaned_files():
    """
    Veritabanında kaydı olmayan yetim dosyaları temizler.
    """
    from flask import current_app as _current_app, has_app_context as _has_app_context

    def _resolve_app_obj():
        if _has_app_context():
            # Flask _get_current_object() exists but type checker doesn't recognize it
            return getattr(_current_app, '_get_current_object', lambda: _current_app)()  # type: ignore[attr-defined]
        try:
            from app import global_flask_app as _global_flask_app
            return _global_flask_app
        except Exception:
            return None

    app_obj = _resolve_app_obj()
    if app_obj is None:
        logger.warning("cleanup_orphaned_files: Flask app bulunamadı (no app_context + global_flask_app None). Atlanıyor.")
        return

    try:
        with app_obj.app_context():
            from app.models.analysis import Analysis

            processed_folder = app_obj.config['PROCESSED_FOLDER']

            if not os.path.exists(processed_folder):
                return

            # Processed klasöründeki tüm dosya ve klasörleri kontrol et
            for item in os.listdir(processed_folder):
                item_path = os.path.join(processed_folder, item)

                # Logs klasörünü atla
                if item == 'logs':
                    continue

                # frames_ ile başlayan klasörleri kontrol et
                if os.path.isdir(item_path) and item.startswith('frames_'):
                    analysis_id = item.replace('frames_', '')

                    # Bu analiz ID'si veritabanında var mı kontrol et
                    analysis_exists = Analysis.query.filter_by(id=analysis_id).first()

                    if not analysis_exists:
                        logger.warning(f"Yetim analiz klasörü bulundu, siliniyor: {item_path}")
                        try:
                            shutil.rmtree(item_path)
                        except Exception as e:
                            logger.warning(f"Yetim klasör silinirken hata: {e}", exc_info=True)

            logger.info("Yetim dosya temizliği tamamlandı.")

    except Exception as e:
        logger.error(f"Yetim dosya temizliği sırasında hata: {e}", exc_info=True)

def sync_model_versions_on_startup():
    """
    Uygulama başlangıcında model versiyonlarını senkronize eder.
    Dosya sistemindeki v1, v2 gibi versiyonları veritabanına kaydeder.
    """
    try:
        logger.info("🔄 Model versiyonları senkronize ediliyor...")
        
        # Age model versiyonlarını senkronize et
        sync_age_model_versions_startup()
        
        # CLIP model versiyonlarını senkronize et
        sync_clip_model_versions_startup()
        
        logger.info("[OK] Model versiyonları başarıyla senkronize edildi!")
        
    except Exception as e:
        logger.error(f"❌ Model senkronizasyon hatası: {e}", exc_info=True)

def sync_age_model_versions_startup():
    """Yaş modeli versiyonlarını startup'ta senkronize eder"""
    try:
        from app.models.content import ModelVersion
        import json
        
        # Dosya sistemindeki versiyonları kontrol et
        versions_dir = os.path.join('storage', 'models', 'age', 'custom_age_head', 'versions')
        
        if not os.path.exists(versions_dir):
            logger.warning("📊 Yaş modeli versiyonlar klasörü bulunamadı, atlanıyor...")
            return
        
        version_folders = [d for d in os.listdir(versions_dir) 
                          if os.path.isdir(os.path.join(versions_dir, d)) and d.startswith('v')]
        
        if not version_folders:
            logger.warning("📊 Yaş modeli versiyonu bulunamadı")
            return
            
        logger.info(f"📊 {len(version_folders)} yaş modeli versiyonu bulundu: {version_folders}")
        
        for version_folder in version_folders:
            version_path = os.path.join(versions_dir, version_folder)
            training_details_path = os.path.join(version_path, 'training_details.json')
            model_path = os.path.join(version_path, 'model.pth')
            
            # Bu versiyon veritabanında var mı kontrol et
            existing = ModelVersion.query.filter_by(
                model_type='age',
                version_name=version_folder
            ).first()
            
            if existing:
                logger.info(f"   ✓ {version_folder} zaten veritabanında mevcut")
                continue
            
            # Eğitim detaylarını oku
            if os.path.exists(training_details_path):
                with open(training_details_path, 'r') as f:
                    training_details = json.load(f)
                
                # Versiyon numarasını belirle
                if version_folder.startswith('v') and '_' in version_folder:
                    base_version = int(version_folder.split('_')[0][1:])
                    version_num = base_version + 1
                elif version_folder.startswith('v'):
                    base_version = int(version_folder[1:])
                    version_num = base_version + 1
                else:
                    version_num = 2
                
                # Yeni model versiyonu oluştur
                model_version = ModelVersion(
                    model_type='age',
                    version=version_num,
                    version_name=version_folder,
                    file_path=version_path,
                    weights_path=model_path,
                    metrics=training_details.get('metrics', {}),
                    training_samples=training_details.get('training_samples', 0),
                    validation_samples=training_details.get('validation_samples', 0),
                    epochs=len(training_details.get('history', {}).get('train_loss', [])),
                    is_active=False,
                    created_at=datetime.fromisoformat(training_details.get('training_date', datetime.now().isoformat())),
                    used_feedback_ids=[]
                )
                
                db.session.add(model_version)
                logger.info(f"   + {version_folder} veritabanına eklendi")
            else:
                logger.warning(f"   ⚠ {version_folder} için training_details.json bulunamadı")
        
        # En son versiyonu aktif yap - ama model state'i kontrol et
        if version_folders:
            # Model state dosyasını kontrol et
            try:
                # Model state modülünü fresh reload et (cache'den değil)
                import app.utils.model_state
                importlib.reload(app.utils.model_state)
                from app.utils.model_state import get_model_state
                
                current_model_state = get_model_state('age')  # Thread-safe getter
                current_age_version = current_model_state.get('active_version')
                logger.info(f"   📊 Model state age version: {current_age_version}")
                
                # Eğer model state version 0 (reset) ise, en son versiyonu aktif yapma
                if current_age_version == 0:
                    logger.info(f"   🔄 Model state version 0 (reset) - en son versiyon aktif yapılmıyor")
                    # Base model'i aktif yap (eğer varsa)
                    base_model = ModelVersion.query.filter_by(model_type='age', version=0).first()
                    if base_model:
                        ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                        base_model.is_active = True
                        logger.info(f"   🎯 Base model (v0) aktif olarak ayarlandı")
                elif current_age_version is not None:
                    # Model state'te belirli bir versiyon var - onu aktif tut
                    logger.info(f"   🔒 Model state version {current_age_version} korunuyor - startup sync atlanıyor")
                    target_model = ModelVersion.query.filter_by(model_type='age', version=current_age_version).first()
                    if target_model:
                        ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                        target_model.is_active = True
                        logger.info(f"   🎯 Version {current_age_version} ({target_model.version_name}) model state'e göre aktif tutuldu")
                    else:
                        logger.warning(f"   ⚠ Model state version {current_age_version} veritabanında bulunamadı, en son versiyon aktif yapılacak")
                        # Fallback to latest
                        latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
                        latest_model = ModelVersion.query.filter_by(model_type='age', version_name=latest_version).first()
                        if latest_model:
                            ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                            latest_model.is_active = True
                            logger.info(f"   🎯 {latest_version} aktif olarak ayarlandı (fallback)")
                else:
                    # Model state'te versiyon yok - normal durum, en son versiyonu aktif yap
                    latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
                    latest_model = ModelVersion.query.filter_by(
                        model_type='age',
                        version_name=latest_version
                    ).first()
                    
                    if latest_model:
                        ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                        latest_model.is_active = True
                        logger.info(f"   🎯 {latest_version} aktif olarak ayarlandı (varsayılan)")
            except Exception as e:
                logger.warning(f"   ⚠ Model state kontrol hatası: {e}", exc_info=True)
                # Fallback - en son versiyonu aktif yap
                latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
                latest_model = ModelVersion.query.filter_by(
                    model_type='age',
                    version_name=latest_version
                ).first()
                
                if latest_model:
                    ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                    latest_model.is_active = True
                    logger.info(f"   🎯 {latest_version} aktif olarak ayarlandı (fallback)")
        
        db.session.commit()
        
    except Exception as e:
        logger.error(f"❌ Yaş modeli senkronizasyon hatası: {e}", exc_info=True)

def sync_clip_model_versions_startup():
    """CLIP modeli versiyonlarını startup'ta senkronize eder"""
    try:
        from app.models.clip_training import CLIPTrainingSession
        import json
        
        # Base model kaydını kontrol et ve ekle
        base_model = CLIPTrainingSession.query.filter_by(
            version_name='base_openclip'
        ).first()
        
        if not base_model:
            # Base OpenCLIP model kaydını oluştur
            base_session = CLIPTrainingSession(
                version_name='base_openclip',
                feedback_count=0,
                training_start=datetime(2025, 1, 1),  # Sabit tarih
                training_end=datetime(2025, 1, 1),
                status='completed',
                model_path='storage/models/clip/ViT-H-14-378-quickgelu_dfn5b/active_model',
                is_active=True,  # Base model aktif olarak başlasın
                is_successful=True,
                created_at=datetime(2025, 1, 1)
            )
            
            # Base model parametrelerini ayarla
            base_session.set_training_params({
                'model_type': 'ViT-H-14-378-quickgelu',
                'pretrained': 'dfn5b',
                'description': 'Base OpenCLIP model - pre-trained'
            })
            
            base_session.set_performance_metrics({
                'description': 'Pre-trained OpenCLIP model',
                'model_size': 'ViT-H-14-378-quickgelu'
            })
            
            db.session.add(base_session)
            logger.info("   + Base OpenCLIP model kaydı oluşturuldu")
        else:
            logger.info("   ✓ Base OpenCLIP model zaten veritabanında mevcut")
        
        # Dosya sistemindeki versiyonları kontrol et
        versions_dir = os.path.join('storage', 'models', 'clip', 'versions')
        
        if not os.path.exists(versions_dir):
            logger.warning("🤖 CLIP versiyonlar klasörü bulunamadı, sadece base model kullanılacak")
            db.session.commit()
            return
        
        version_folders = [d for d in os.listdir(versions_dir) 
                          if os.path.isdir(os.path.join(versions_dir, d)) and d.startswith('v')]
        
        if not version_folders:
            logger.warning("🤖 Fine-tuned CLIP versiyonu bulunamadı, sadece base model kullanılacak")
            db.session.commit()
            return
            
        logger.info(f"🤖 {len(version_folders)} fine-tuned CLIP versiyonu bulundu: {version_folders}")
        
        for version_folder in version_folders:
            version_path = os.path.join(versions_dir, version_folder)
            metadata_path = os.path.join(version_path, 'metadata.json')
            model_path = os.path.join(version_path, 'pytorch_model.bin')
            
            # Bu versiyon veritabanında var mı kontrol et
            existing = CLIPTrainingSession.query.filter_by(
                version_name=version_folder
            ).first()
            
            if existing:
                logger.info(f"   ✓ {version_folder} zaten veritabanında mevcut")
                continue
            
            # Metadata dosyasını oku
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Yeni CLIP training session oluştur
                training_session = CLIPTrainingSession(
                    version_name=version_folder,
                    feedback_count=metadata.get('feedback_count', 0),
                    training_start=datetime.fromisoformat(metadata.get('training_start', datetime.now().isoformat())),
                    training_end=datetime.fromisoformat(metadata.get('training_end', datetime.now().isoformat())),
                    status='completed',
                    model_path=model_path,
                    is_active=False,
                    is_successful=True,
                    created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat()))
                )
                
                # Training parametrelerini ayarla
                if 'training_params' in metadata:
                    training_session.set_training_params(metadata['training_params'])
                
                # Performance metriklerini ayarla
                if 'performance_metrics' in metadata:
                    training_session.set_performance_metrics(metadata['performance_metrics'])
                
                db.session.add(training_session)
                logger.info(f"   + {version_folder} veritabanına eklendi")
            else:
                logger.warning(f"   ⚠ {version_folder} için metadata.json bulunamadı")
        
        # En son fine-tuned versiyonu aktif yap (varsa)
        if version_folders:
            latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
            latest_session = CLIPTrainingSession.query.filter_by(
                version_name=latest_version
            ).first()
            
            if latest_session:
                # Tüm versiyonları pasif yap (base model dahil)
                CLIPTrainingSession.query.update({'is_active': False})
                # En son fine-tuned versiyonu aktif yap
                latest_session.is_active = True
                logger.info(f"   🎯 {latest_version} aktif olarak ayarlandı (base model yerine)")
        
        db.session.commit()
        
    except Exception as e:
        logger.error(f"❌ CLIP senkronizasyon hatası: {e}", exc_info=True)

def register_global_routes(app):
    """
    Global route'ları kaydeder.
    
    Args:
        app: Flask uygulaması.
    """
    @app.route('/processed/<path:filename>')
    def serve_processed_file(filename):
        processed_folder = os.path.join(app.config['STORAGE_FOLDER'], 'processed')
        return send_from_directory(processed_folder, filename)

def register_error_handlers(flask_app):
    """Register error handlers"""
    logger.info(f"register_error_handlers called with: {type(flask_app)} - {flask_app}")
    @flask_app.errorhandler(404)
    def page_not_found(e):
        from flask import jsonify
        return jsonify({'error': 'Not found'}), 404

    @flask_app.errorhandler(500)
    def internal_server_error(e):
        from flask import jsonify
        return jsonify({'error': 'Internal server error'}), 500 