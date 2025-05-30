import os
import shutil
from datetime import datetime
from flask import Flask, send_from_directory, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_socketio import SocketIO
from config import config
import logging
import threading
import importlib
from app.middleware.security_middleware import SecurityMiddleware
from app.middleware import register_json_middleware
from app.services.analysis_service import AnalysisService
from app.utils.memory_utils import initialize_memory_management

# Global extensions
db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO()

# Thread-safe logging lock
_log_lock = threading.Lock()

def create_app(config_name='default'):
    """Flask uygulaması fabrikası"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    
    # JSON encoder'ı ayarla
    app.json_encoder = CustomJSONEncoder
    
    # Initialize security middleware
    from app.middleware.security_middleware import SecurityMiddleware
    SecurityMiddleware(app)
    
    # Initialize memory management for performance optimization
    try:
        initialize_memory_management()
        print("✅ Memory management initialized")
    except Exception as e:
        print(f"⚠️ Memory management initialization failed: {e}")
    
    # Register blueprints
    from app.routes.main_routes import main_bp
    from app.routes.file_routes import file_bp
    from app.routes.analysis_routes import analysis_bp
    from app.routes.feedback_routes import feedback_bp
    from app.routes.settings_routes import settings_bp
    from app.routes.model_management_routes import model_management_bp
    from app.routes.queue_routes import queue_bp
    from app.routes.performance_routes import performance_bp  # Performance routes
    
    app.register_blueprint(main_bp)
    app.register_blueprint(file_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(model_management_bp)
    app.register_blueprint(queue_bp)
    app.register_blueprint(performance_bp)  # Register performance routes
    
    # Socket.IO event handlers
    register_socketio_events(app)
    
    # Error handlers
    register_error_handlers(app)
    
    # Startup tasks
    with app.app_context():
        try:
            db.create_all()
            print("✅ Veritabanı tabloları oluşturuldu/kontrol edildi")
            
            # Model versiyonlarını senkronize et
            sync_age_model_versions_startup()
            sync_clip_model_versions_startup()
            
        except Exception as e:
            print(f"❌ Startup görevleri hatası: {str(e)}")
    
    return app

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
        print("Veritabanı yolu:", db_path)
        
        # Sadece veritabanı yoksa oluştur, mevcut olanı silme
        if not os.path.exists(db_path):
            print("Veritabanı bulunamadı, yeni veritabanı oluşturuluyor.")
            db.create_all()
            print("Yeni veritabanı oluşturuldu.")
        else:
            print("Mevcut veritabanı kullanılıyor.")
            # Sadece tabloları güncelle (yeni tablolar varsa ekle)
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
        print("Analiz kuyruğu servisi başlatılıyor...")
        start_processor()
        print("Analiz kuyruğu servisi başlatıldı.")

    # Global route'ları kaydet
    register_global_routes(app)

def clean_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Eğer 'logs' klasörüyse (veya 'logs' klasörünün içindeysek) silme
            if filename == 'logs' and os.path.isdir(file_path):
                print(f"'{file_path}' log klasörü atlanıyor, silinmeyecek.")
                continue # logs klasörünü silme, içini de boşaltma

            if os.path.isfile(file_path) or os.path.islink(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Dosya silinirken hata (atlanıyor): {file_path}, Hata: {e}")
            elif os.path.isdir(file_path):
                try:
                    shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Klasör silinirken hata (atlanıyor): {file_path}, Hata: {e}")
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
        print(f"Eski analiz sonuçları temizleniyor: {cutoff_date} tarihinden eski olanlar")
        
        # Eski analizleri bul (created_at yerine start_time kullan)
        old_analyses = Analysis.query.filter(Analysis.start_time < cutoff_date).all()
        
        if not old_analyses:
            print("Temizlenecek eski analiz bulunamadı.")
            return
        
        print(f"{len(old_analyses)} eski analiz bulundu, temizleniyor...")
        
        # Her analiz için ilgili dosyaları temizle
        for analysis in old_analyses:
            try:
                # Analiz klasörünü bul ve sil (app context içinde olduğumuz için current_app kullanabiliriz)
                from flask import current_app
                analysis_folder = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}")
                if os.path.exists(analysis_folder):
                    shutil.rmtree(analysis_folder)
                    print(f"Analiz klasörü silindi: {analysis_folder}")
                
                # İşlenmiş resim dosyasını sil
                if analysis.processed_image_path:
                    processed_file = os.path.join(current_app.config['PROCESSED_FOLDER'], analysis.processed_image_path)
                    if os.path.exists(processed_file):
                        os.unlink(processed_file)
                        print(f"İşlenmiş resim silindi: {processed_file}")
                
                # En yüksek riskli kare dosyasını sil
                if analysis.highest_risk_frame:
                    risk_frame_file = os.path.join(current_app.config['PROCESSED_FOLDER'], analysis.highest_risk_frame)
                    if os.path.exists(risk_frame_file):
                        os.unlink(risk_frame_file)
                        print(f"En yüksek riskli kare silindi: {risk_frame_file}")
                
                # Veritabanından analizi sil (cascade ile ilgili kayıtlar da silinir)
                db.session.delete(analysis)
                
            except Exception as e:
                print(f"Analiz {analysis.id} temizlenirken hata: {e}")
                continue
        
        # Değişiklikleri kaydet
        db.session.commit()
        print(f"{len(old_analyses)} eski analiz başarıyla temizlendi.")
        
        # Artık kullanılmayan dosyaları da temizle
        cleanup_orphaned_files()
        
    except Exception as e:
        print(f"Eski analiz sonuçları temizlenirken hata: {e}")
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
                    print(f"Yetim analiz klasörü bulundu, siliniyor: {item_path}")
                    try:
                        shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Yetim klasör silinirken hata: {e}")
        
        print("Yetim dosya temizliği tamamlandı.")
        
    except Exception as e:
        print(f"Yetim dosya temizliği sırasında hata: {e}")

def sync_model_versions_on_startup():
    """
    Uygulama başlangıcında model versiyonlarını senkronize eder.
    Dosya sistemindeki v1, v2 gibi versiyonları veritabanına kaydeder.
    """
    try:
        print("🔄 Model versiyonları senkronize ediliyor...")
        
        # Age model versiyonlarını senkronize et
        sync_age_model_versions_startup()
        
        # CLIP model versiyonlarını senkronize et
        sync_clip_model_versions_startup()
        
        print("✅ Model versiyonları başarıyla senkronize edildi!")
        
    except Exception as e:
        print(f"❌ Model senkronizasyon hatası: {e}")
        import traceback
        traceback.print_exc()

def sync_age_model_versions_startup():
    """Yaş modeli versiyonlarını startup'ta senkronize eder"""
    try:
        from app.models.content import ModelVersion
        import json
        
        # Dosya sistemindeki versiyonları kontrol et
        versions_dir = os.path.join('storage', 'models', 'age', 'custom_age_head', 'versions')
        
        if not os.path.exists(versions_dir):
            print("📊 Yaş modeli versiyonlar klasörü bulunamadı, atlanıyor...")
            return
        
        version_folders = [d for d in os.listdir(versions_dir) 
                          if os.path.isdir(os.path.join(versions_dir, d)) and d.startswith('v')]
        
        if not version_folders:
            print("📊 Yaş modeli versiyonu bulunamadı")
            return
            
        print(f"📊 {len(version_folders)} yaş modeli versiyonu bulundu: {version_folders}")
        
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
                print(f"   ✓ {version_folder} zaten veritabanında mevcut")
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
                print(f"   + {version_folder} veritabanına eklendi")
            else:
                print(f"   ⚠ {version_folder} için training_details.json bulunamadı")
        
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
                print(f"   📊 Model state age version: {current_age_version}")
                
                # Eğer model state version 0 (reset) ise, en son versiyonu aktif yapma
                if current_age_version == 0:
                    print(f"   🔄 Model state version 0 (reset) - en son versiyon aktif yapılmıyor")
                    # Base model'i aktif yap (eğer varsa)
                    base_model = ModelVersion.query.filter_by(model_type='age', version=0).first()
                    if base_model:
                        ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                        base_model.is_active = True
                        print(f"   🎯 Base model (v0) aktif olarak ayarlandı")
                elif current_age_version is not None:
                    # Model state'te belirli bir versiyon var - onu aktif tut
                    print(f"   🔒 Model state version {current_age_version} korunuyor - startup sync atlanıyor")
                    target_model = ModelVersion.query.filter_by(model_type='age', version=current_age_version).first()
                    if target_model:
                        ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                        target_model.is_active = True
                        print(f"   🎯 Version {current_age_version} ({target_model.version_name}) model state'e göre aktif tutuldu")
                    else:
                        print(f"   ⚠ Model state version {current_age_version} veritabanında bulunamadı, en son versiyon aktif yapılacak")
                        # Fallback to latest
                        latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
                        latest_model = ModelVersion.query.filter_by(model_type='age', version_name=latest_version).first()
                        if latest_model:
                            ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                            latest_model.is_active = True
                            print(f"   🎯 {latest_version} aktif olarak ayarlandı (fallback)")
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
                        print(f"   🎯 {latest_version} aktif olarak ayarlandı (varsayılan)")
            except Exception as e:
                print(f"   ⚠ Model state kontrol hatası: {e}")
                # Fallback - en son versiyonu aktif yap
                latest_version = max(version_folders, key=lambda x: os.path.getctime(os.path.join(versions_dir, x)))
                latest_model = ModelVersion.query.filter_by(
                    model_type='age',
                    version_name=latest_version
                ).first()
                
                if latest_model:
                    ModelVersion.query.filter_by(model_type='age', is_active=True).update({'is_active': False})
                    latest_model.is_active = True
                    print(f"   🎯 {latest_version} aktif olarak ayarlandı (fallback)")
        
        db.session.commit()
        
    except Exception as e:
        print(f"❌ Yaş modeli senkronizasyon hatası: {e}")

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
            print("   + Base OpenCLIP model kaydı oluşturuldu")
        else:
            print("   ✓ Base OpenCLIP model zaten veritabanında mevcut")
        
        # Dosya sistemindeki versiyonları kontrol et
        versions_dir = os.path.join('storage', 'models', 'clip', 'versions')
        
        if not os.path.exists(versions_dir):
            print("🤖 CLIP versiyonlar klasörü bulunamadı, sadece base model kullanılacak")
            db.session.commit()
            return
        
        version_folders = [d for d in os.listdir(versions_dir) 
                          if os.path.isdir(os.path.join(versions_dir, d)) and d.startswith('v')]
        
        if not version_folders:
            print("🤖 Fine-tuned CLIP versiyonu bulunamadı, sadece base model kullanılacak")
            db.session.commit()
            return
            
        print(f"🤖 {len(version_folders)} fine-tuned CLIP versiyonu bulundu: {version_folders}")
        
        for version_folder in version_folders:
            version_path = os.path.join(versions_dir, version_folder)
            metadata_path = os.path.join(version_path, 'metadata.json')
            model_path = os.path.join(version_path, 'pytorch_model.bin')
            
            # Bu versiyon veritabanında var mı kontrol et
            existing = CLIPTrainingSession.query.filter_by(
                version_name=version_folder
            ).first()
            
            if existing:
                print(f"   ✓ {version_folder} zaten veritabanında mevcut")
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
                print(f"   + {version_folder} veritabanına eklendi")
            else:
                print(f"   ⚠ {version_folder} için metadata.json bulunamadı")
        
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
                print(f"   🎯 {latest_version} aktif olarak ayarlandı (base model yerine)")
        
        db.session.commit()
        
    except Exception as e:
        print(f"❌ CLIP senkronizasyon hatası: {e}")

def register_global_routes(app):
    @app.route('/processed/<path:filename>')
    def serve_processed_file(filename):
        processed_folder = os.path.join(app.config['STORAGE_FOLDER'], 'processed')
        return send_from_directory(processed_folder, filename) 