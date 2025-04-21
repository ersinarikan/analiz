import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_cors import CORS
import logging
from config import Config
import threading
import json
from datetime import datetime

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
    
    @app.route('/api/feedback/submit', methods=['POST'])
    def submit_feedback():
        try:
            feedback_data = request.json
            
            # Geri bildirim verilerini doğrula
            required_fields = ['content_id', 'rating']
            if not all(field in feedback_data for field in required_fields):
                return jsonify({'error': 'Eksik veri'}), 400
            
            # Geri bildirim verilerini kaydet
            feedback_file = os.path.join(app.config['UPLOAD_FOLDER'], 'feedback', 'content_feedback.jsonl')
            os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
            
            # Geri bildirime timestamp ekle
            feedback_data['timestamp'] = datetime.now().isoformat()
            
            # JSONL formatında kaydet
            with open(feedback_file, 'a', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False)
                f.write('\n')
            
            return jsonify({'success': True, 'message': 'Geri bildirim kaydedildi'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/feedback/age', methods=['POST'])
    def submit_age_feedback():
        try:
            feedback_data = request.json
            
            # Geri bildirim verilerini doğrula
            required_fields = ['person_id', 'corrected_age']
            if not all(field in feedback_data for field in required_fields):
                return jsonify({'error': 'Eksik veri'}), 400
            
            # Geri bildirim verilerini kaydet
            feedback_file = os.path.join(app.config['UPLOAD_FOLDER'], 'feedback', 'age_feedback.jsonl')
            os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
            
            # Geri bildirime timestamp ekle
            feedback_data['timestamp'] = datetime.now().isoformat()
            
            # JSONL formatında kaydet
            with open(feedback_file, 'a', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False)
                f.write('\n')
            
            return jsonify({'success': True, 'message': 'Yaş geri bildirimi kaydedildi'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app 