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
    print("!!! TEST: create_app fonksiyonu ÇALIŞIYOR !!!")
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
    
    app.register_blueprint(main_bp)
    app.register_blueprint(file_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(debug_bp)
    app.register_blueprint(settings_bp)
    
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
        print("!!! TEST: initialize_app fonksiyonu ÇALIŞIYOR !!!")
        
        # Veritabanı temizliği
        db_path = app.config.get('SQLALCHEMY_DATABASE_URI', '').replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            db_path = os.path.join(app.root_path, db_path)
        print("Veritabanı yolu:", db_path)
        if db_path and os.path.exists(db_path):
            os.remove(db_path)
            print("Veritabanı silindi.")
        db.create_all()
        print("Veritabanı yeniden oluşturuldu.")
        
        # Klasörlerin oluşturulması ve temizlenmesi
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
        
        # Sadece upload ve processed klasörlerini temizle
        clean_folder(app.config['UPLOAD_FOLDER'])
        clean_folder(app.config['PROCESSED_FOLDER'])
        
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

def register_global_routes(app):
    @app.route('/processed/<path:filename>')
    def serve_processed_file(filename):
        processed_folder = os.path.join(app.config['STORAGE_FOLDER'], 'processed')
        return send_from_directory(processed_folder, filename) 