import os
from flask import Flask
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
    
    # Yükleme ve işleme klasörlerini oluştur
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    
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