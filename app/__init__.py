import os
import shutil
from flask import Flask, send_from_directory, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_socketio import SocketIO
from config import config

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
    
    app.register_blueprint(main_bp)
    app.register_blueprint(file_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(debug_bp)
    
    # NumPy JSON serializer middleware'i kaydet
    from app.middleware import register_json_middleware
    register_json_middleware(app)
    
    # Servisler
    from app.services.analysis_service import AnalysisService
    analysis_service = AnalysisService(app)
    
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

    # Global route'ları kaydet
    register_global_routes(app)

def clean_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(folder_path, exist_ok=True)

def register_global_routes(app):
    @app.route('/processed/<path:filename>')
    def serve_processed_file(filename):
        processed_folder = os.path.join(app.config['STORAGE_FOLDER'], 'processed')
        return send_from_directory(processed_folder, filename) 