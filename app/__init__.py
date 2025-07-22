"""
WSANALIZ Flask Uygulaması - Ana Modül
"""
import logging
import os
import signal
import sys
import shutil
import importlib
from datetime import datetime, timedelta

from flask import Flask, request, send_from_directory, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import config

# 🎯 SocketIO'yu ayrı dosyadan import et (circular import önleme)
from app.socketio_instance import socketio
from app.json_encoder import CustomJSONEncoder

# Global minimal socketio reference (runtime'da set edilecek)
_current_running_socketio = None
global_flask_app = None  # Ana Flask app nesnesi, background thread'ler için

# Memory utils - optional import
try:
    from app.utils.memory_utils import initialize_memory_management
except ImportError:
    initialize_memory_management = None

# Global extensions
db = SQLAlchemy()
migrate = Migrate()

logger = logging.getLogger("wsanaliz.app_init")
logging.basicConfig(level=logging.INFO)

# ===============================
# 🎯 STANDARD FLASK-SOCKETIO PATTERN
# ===============================
# DİKKAT: SocketIO instance'ı SADECE burada, uygulama başlatılırken oluşturulur ve set edilir.
# Başka hiçbir yerde yeni SocketIO instance'ı yaratılmayacak veya set edilmeyecek!

def register_blueprints_from_list(app, blueprint_defs):
    """
    blueprint_defs: List of tuples (import_path, attr_name, alias)
    - import_path: Python import path as string (e.g. 'app.routes.main_routes')
    - attr_name: Attribute name in the module (e.g. 'main_bp' or 'bp')
    - alias: Optional alias for logging (e.g. 'file_bp'), can be None
    """
    import importlib
    logger = logging.getLogger("wsanaliz.app_init")
    blueprints_to_register = []
    for import_path, attr_name, alias in blueprint_defs:
        try:
            module = importlib.import_module(import_path)
            bp = getattr(module, attr_name)
            blueprints_to_register.append(bp)
            logger.info(f"Blueprint imported: {import_path}.{attr_name} as {alias or attr_name}")
        except ImportError as e:
            logger.warning(f"{import_path} import failed: {e}")
        except AttributeError as e:
            logger.warning(f"{import_path} has no attribute {attr_name}: {e}")
    for bp in blueprints_to_register:
        app.register_blueprint(bp)
        logger.info(f"Blueprint registered: {bp.name}")
    logger.info(f"Total blueprints registered: {len(blueprints_to_register)}")
    return blueprints_to_register

def create_app(config_name='default'):
    """
    Flask uygulaması fabrikası.
    Args:
        config_name (str): Kullanılacak konfigürasyon adı.
    Returns:
        Flask: Yapılandırılmış Flask uygulaması.
    """
    flask_app = Flask(__name__)
    flask_app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(flask_app)
    
    # ✅ MİNİMAL PATTERN: Direct SocketIO setup with optimized configuration
    from flask_socketio import SocketIO
    minimal_socketio = SocketIO(
        flask_app,
        cors_allowed_origins="*",
        ping_timeout=60,  # Stability için
        ping_interval=20,  # Daha sık ping ile stabilite
        logger=False,     # Verbose logging kapat
        engineio_logger=False
    )
    
    # Global instance'ı güncelleyelim - emit_analysis_progress için
    import app.socketio_instance
    app.socketio_instance.set_socketio(minimal_socketio)  # TEK NOKTA SET!
    
    # ✅ MİNİMAL PATTERN: Direct event handler registration
    
    @minimal_socketio.on('connect')
    def handle_connect():
        from flask import request
        from flask_socketio import emit
        print(f"🎉🎉🎉 MİNİMAL CONNECT! Session: {request.sid}")
        emit('connected', {'message': 'Minimal pattern bağlantısı başarılı!'})
        
    @minimal_socketio.on('disconnect')  
    def handle_disconnect():
        from flask import request
        print(f"❌❌❌ MİNİMAL DISCONNECT! Session: {request.sid}")
        
    @minimal_socketio.on('ping')
    def handle_ping(data):
        from flask import request
        from flask_socketio import emit
        print(f"🏓🏓🏓 MİNİMAL PING! Session: {request.sid}, Data: {data}")
        emit('pong', {'message': 'Minimal PONG!', 'data': data})

    @minimal_socketio.on('join_analysis')
    def handle_join_analysis(data):
        from flask import request
        from flask_socketio import emit, join_room
        print(f"🔍🔍🔍 MİNİMAL JOIN_ANALYSIS! Session: {request.sid}, Data: {data}")
        
        if data and 'analysis_id' in data:
            analysis_id = data['analysis_id']
            room = f"analysis_{analysis_id}"
            
            # Room'a katıl
            join_room(room)
            
            # DEBUG: Room membership kontrol et
            try:
                room_members = minimal_socketio.server.manager.get_participants(namespace='/', room=room)
                room_members_list = list(room_members)
                print(f"🔍🔍🔍 MİNİMAL JOIN: Room {room} members after join: {room_members_list}")
            except Exception as room_err:
                print(f"🔍 MİNİMAL JOIN: Room membership check failed: {room_err}")
            
            # Başarı mesajı gönder
            emit('joined_analysis', {
                'analysis_id': analysis_id,
                'room': room,
                'message': f'Analysis {analysis_id} room\'una katıldınız (minimal)',
                'source': 'minimal-handler'
            })
        else:
            print(f"❌ MİNİMAL JOIN_ANALYSIS: No analysis_id in data")
            
    logger.info("✅ Minimal pattern SocketIO handlers registered!")
    print("✅ Minimal pattern SocketIO handlers registered!")
    
    # Minimal SocketIO'yu app'e attach et ki emit_analysis_progress bulabilsin
    app.minimal_socketio = minimal_socketio
    
    # Global referans da tut
    import app as app_module
    app_module._current_minimal_socketio = minimal_socketio
    
    # Global modül-level referansı da set et
    global _current_running_socketio
    _current_running_socketio = minimal_socketio
    
    # Ana Flask app nesnesini global değişkene ata
    global global_flask_app
    global_flask_app = flask_app
    
    # JSON encoder'ı ayarla
    flask_app.json_encoder = CustomJSONEncoder
    
    # Initialize security middleware
    from app.middleware.security_middleware import SecurityMiddleware
    SecurityMiddleware(flask_app)
    
    # Initialize memory management for performance optimization
    try:
        if initialize_memory_management:
            initialize_memory_management()
            logger.info("Memory management initialized")
        else:
            logger.warning("Memory management not available (optional dependency)")
    except Exception as e:
        logger.warning(f"Memory management initialization failed: {e}", exc_info=True)
    
    # Register blueprints with error handling (refactored)
    blueprint_defs = [
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
    
    # WebSocket event handlers registration - ESKİ YÖNTEMs DEVRE DIŞI
    # try:
    #     from app.routes.websocket_routes import register_websocket_handlers
    #     register_websocket_handlers(socketio)
    #     logger.info("WebSocket event handlers registered successfully")
    # except Exception as e:
    #     logger.error(f"WebSocket handlers registration failed: {e}")
    
    # Error handlers - Geçici olarak disable edildi (circular import problemi)
    # register_error_handlers(flask_app)
    
    # Startup tasks
    with flask_app.app_context():
        try:
            db.create_all()
            logger.info("Veritabanı tabloları oluşturuldu/kontrol edildi")
            
            # Model versiyonlarını senkronize et
            sync_age_model_versions_startup()
            sync_clip_model_versions_startup()
            
        except Exception as e:
            logger.error(f"Startup görevleri hatası: {str(e)}", exc_info=True)
    
    return flask_app, minimal_socketio

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
        
        # Analiz kuyruğu servisini başlat
        from app.services.queue_service import start_processor
        logger.info("Analiz kuyruğu servisi başlatılıyor...")
        start_processor()
        logger.info("Analiz kuyruğu servisi başlatıldı.")

    # Global route'ları kaydet
    register_global_routes(app)

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

def cleanup_old_analysis_results(days_old=7):
    """
    Belirli bir yaştan eski analiz sonuçlarını ve ilgili dosyaları temizler.
    
    Args:
        days_old: Kaç günden eski analizlerin temizleneceği (varsayılan: 7)
    """
    try:
        from datetime import datetime, timedelta
        from app.models.analysis import Analysis, ContentDetection, AgeEstimation
        from app.models.file import File
        from app.models.clip_training import CLIPTrainingSession  # CLIP training model import
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        logger.info(f"Eski analiz sonuçları temizleniyor: {cutoff_date} tarihinden eski olanlar")
        
        # Eski analizleri bul (created_at yerine start_time kullan)
        old_analyses = Analysis.query.filter(Analysis.start_time < cutoff_date).all()
        
        if not old_analyses:
            logger.info("Temizlenecek eski analiz bulunamadı.")
            return
        
        logger.info(f"{len(old_analyses)} eski analiz bulundu, temizleniyor...")
        
        # Her analiz için ilgili dosyaları temizle
        for analysis in old_analyses:
            try:
                # Analiz klasörünü bul ve sil (app context içinde olduğumuz için current_app kullanabiliriz)
                from flask import current_app
                analysis_folder = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}")
                if os.path.exists(analysis_folder):
                    shutil.rmtree(analysis_folder)
                    logger.info(f"Analiz klasörü silindi: {analysis_folder}")
                
                # İşlenmiş resim dosyasını sil
                if analysis.processed_image_path:
                    processed_file = os.path.join(current_app.config['PROCESSED_FOLDER'], analysis.processed_image_path)
                    if os.path.exists(processed_file):
                        os.unlink(processed_file)
                        logger.info(f"İşlenmiş resim silindi: {processed_file}")
                
                # En yüksek riskli kare dosyasını sil
                if analysis.highest_risk_frame:
                    risk_frame_file = os.path.join(current_app.config['PROCESSED_FOLDER'], analysis.highest_risk_frame)
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
        if 'db' in locals():
            db.session.rollback()

def cleanup_orphaned_files():
    """
    Veritabanında kaydı olmayan yetim dosyaları temizler.
    """
    try:
        from app.models.analysis import Analysis
        
        processed_folder = current_app.config['PROCESSED_FOLDER']
        
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
        
        logger.info("✅ Model versiyonları başarıyla senkronize edildi!")
        
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