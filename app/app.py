import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_cors import CORS
import logging
from config import Config
import threading

# Initialize extensions
db = SQLAlchemy()
socketio = SocketIO()

def create_app(config_class=Config):
    """Flask uygulamasını oluştur ve yapılandır."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Güçlü hata ayıklama ayarları
    if app.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # Debug modunda işlem durumunu izlemek için özel handler ekle
        def log_debug_info():
            from app.services.debug_service import log_active_analyses
            timer = threading.Timer(10.0, log_debug_info)
            timer.daemon = True
            timer.start()
            with app.app_context():
                log_active_analyses()
        
        # Periyodik debug logları için zamanlayıcı başlat
        debug_timer = threading.Timer(5.0, log_debug_info)
        debug_timer.daemon = True
        debug_timer.start()
    
    # Initialize extensions with app
    db.init_app(app)
    CORS(app)
    socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')
    
    # Register blueprints
    from app.routes.main_routes import bp as main_bp
    from app.routes.file_routes import bp as file_bp
    from app.routes.analysis_routes import bp as analysis_bp
    from app.routes.feedback_routes import bp as feedback_bp
    from app.routes.model_routes import bp as model_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(file_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(model_bp)
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    # Initialize database tables
    with app.app_context():
        db.create_all()
        
        # Populate database with default data if needed
        from app.models.model_version import ModelVersion
        from datetime import datetime
        
        # Default model sürümlerini oluştur (eğer yoksa)
        content_model = ModelVersion.query.filter_by(model_type='content').first()
        if not content_model:
            content_model = ModelVersion(
                model_type='content',
                version='1.0.0',
                created_at=datetime.utcnow(),
                metrics={'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85}
            )
            db.session.add(content_model)
            
        age_model = ModelVersion.query.filter_by(model_type='age').first()
        if not age_model:
            age_model = ModelVersion(
                model_type='age',
                version='1.0.0',
                created_at=datetime.utcnow(),
                metrics={'mae': 3.5, 'accuracy': 0.78}
            )
            db.session.add(age_model)
            
        db.session.commit()
    
    # Log startup information
    app.logger.info('Analiz modelleri başarıyla yüklendi')
    
    return app 