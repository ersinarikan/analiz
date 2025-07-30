from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)
"""
Ana uygulama blueprint'i.
- Ana sayfa, sağlık kontrolü ve temel endpointleri içerir.
"""

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