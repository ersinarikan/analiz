from flask import Blueprint, render_template
from flask import current_app as app

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """
    Ana sayfa - Dosya yükleme ve analiz arayüzü.
    """
    return render_template('index.html')

@main_bp.route('/metrics')
def metrics():
    """
    Model metrikleri sayfası.
    """
    return render_template('metrics.html')

@main_bp.route('/model-management')
def model_management():
    """
    Model yönetimi sayfası.
    """
    return render_template('model_management.html') 

@main_bp.route('/clip-monitoring')
def clip_monitoring():
    """
    CLIP Training Monitoring Dashboard.
    """
    return render_template('clip_monitoring.html') 