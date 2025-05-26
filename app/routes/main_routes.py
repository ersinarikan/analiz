from flask import Blueprint, render_template
from flask import current_app as app

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """
    Ana sayfa - Dosya yükleme ve analiz arayüzü.
    """
    return render_template('index.html')

@bp.route('/metrics')
def metrics():
    """
    Model metrikleri sayfası.
    """
    return render_template('metrics.html')

@bp.route('/model-management')
def model_management():
    """
    Model yönetimi sayfası.
    """
    return render_template('model_management.html') 