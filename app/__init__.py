import os
import shutil
from flask import Flask, send_from_directory, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_socketio import SocketIO
from config import config
import logging
from logging.handlers import RotatingFileHandler

# Global extensions
db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO()

def create_app(config_name=None):
    """
    Flask uygulamasını oluşturur ve yapılandırır.
    
    Args:
        config_name: Kullanılacak konfigürasyon tipi ('development', 'production', 'testing')
        
    Returns:
        Flask uygulaması
    """
    app = Flask(__name__)
    
    # Konfigürasyonu yükle
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')
    
    app.config.from_object(config[config_name])
    
    # Loglama her zaman aktif olacak şekilde düzenlendi
    # Loglama için 'processed' klasörünün var olduğundan emin ol
    logs_folder = os.path.join(app.config['PROCESSED_FOLDER'], 'logs')
    os.makedirs(logs_folder, exist_ok=True)
    log_file_path = os.path.join(logs_folder, 'app.log')

    file_handler = RotatingFileHandler(log_file_path, maxBytes=1048576, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    # Önceki handler'ları temizle (tekrar eklemeyi önlemek için, özellikle debug modunda reloader ile)
    for handler in app.logger.handlers[:]:
        app.logger.removeHandler(handler)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('Uygulama başlatılıyor - Dosya loglama her zaman aktif')
    
    # CORS yapılandırması
    CORS(app)
    
    # Veritabanı başlatma
    db.init_app(app)
    
    # Migrasyon
    migrate.init_app(app, db)
    
    # Blueprintleri kaydet
    from app.routes.file_routes import bp as file_bp
    from app.routes.analysis_routes import bp as analysis_bp
    from app.routes.main_routes import bp as main_bp
    from app.routes.model_routes import bp as model_bp
    from app.routes.feedback_routes import bp as feedback_bp
    from app.routes.debug_routes import bp as debug_bp
    from app.routes.settings_routes import bp as settings_bp
    from app.routes.model_management_routes import model_management_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(file_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(debug_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(model_management_bp)
    
    # NumPy JSON serializer middleware'i kaydet
    from app.middleware import register_json_middleware
    register_json_middleware(app)
    
    # Servisler
    from app.services.analysis_service import AnalysisService
    analysis_service = AnalysisService()
    
    # SocketIO başlatma
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return {"error": "Resource not found"}, 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return {"error": "Internal server error"}, 500
    
    # A simple route to confirm the app is working
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
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
        
        # Sadece upload klasörünü temizle (processed klasörü analiz sonuçlarını içerdiği için korunur)
        clean_folder(app.config['UPLOAD_FOLDER'])
        # clean_folder(app.config['PROCESSED_FOLDER'])  # Analiz sonuçlarını korumak için devre dışı
        
        # Eski analiz sonuçlarını temizle (7 günden eski olanları)
        cleanup_old_analysis_results(days_old=7)
        
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

def register_global_routes(app):
    @app.route('/processed/<path:filename>')
    def serve_processed_file(filename):
        processed_folder = os.path.join(app.config['STORAGE_FOLDER'], 'processed')
        return send_from_directory(processed_folder, filename) 