from flask import Blueprint, jsonify, current_app
import logging
from app.services.debug_service import test_numpy_serialization, debug_object
import numpy as np
import json
import os
import sys
import subprocess
import platform
import time
import psutil
import threading
from datetime import datetime, timedelta

# TensorFlow uyarılarını bastır
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
# Güncel TensorFlow 2.x logging API'si kullan
tf.get_logger().setLevel('ERROR')

from app.models.analysis import Analysis
from app.services.analysis_service import AnalysisService
from app.services.queue_service import get_queue_status

logger = logging.getLogger(__name__)
debug_bp = Blueprint('debug', __name__)
"""
Hata ayıklama için blueprint.
- Sistem ve model hata ayıklama endpointlerini içerir.
"""

@debug_bp.route('/test_serialization', methods=['GET'])
def test_serialization():
    """NumPy veri türlerinin JSON serileştirme testini çalıştırır."""
    success, result = test_numpy_serialization()
    
    return jsonify({
        'success': success,
        'result': result if success else None,
        'error': None if success else result
    })

@debug_bp.route('/test_content_detection', methods=['GET'])
def test_content_detection():
    """ContentDetection sınıfının serileştirme işlemini test eder."""
    from app.models.analysis import ContentDetection
    
    try:
        # Test nesnesi oluştur
        detection = ContentDetection(
            analysis_id=1,
            frame_path="test_frame.jpg"
        )
        
        # Rastgele skor değerleri ata
        detection.violence_score = 0.2
        detection.adult_content_score = 0.1
        detection.harassment_score = 0.3
        detection.weapon_score = 0.15
        detection.drug_score = 0.05
        
        # NumPy değerler içeren nesneler listesi oluştur
        detected_objects = [
            {
                'label': 'person',
                'confidence': np.float32(0.95),
                'box': [np.int32(10), np.int32(20), np.int32(30), np.int32(40)]
            },
            {
                'label': 'car',
                'confidence': np.float64(0.85),
                'box': np.array([50, 60, 70, 80])
            }
        ]
        
        # Debug nesnesini logla
        debug_object(detected_objects, "detected_objects_before")
        
        # Nesneyi ContentDetection sınıfına ata
        detection.detected_objects = detected_objects
        
        # Debug sonuç nesnesini logla
        debug_object(detection.get_detected_objects(), "detected_objects_after")
        
        # to_dict() metodunu kullanarak tüm nesneyi al
        result = detection.to_dict()
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        logger.error(f"ContentDetection testi sırasında hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500 

@debug_bp.route('/system-info', methods=['GET'])
def system_info():
    """
    Sistem bilgilerini döndürür.
    """
    try:
        import numpy as np
        import cv2
        import tensorflow as tf
        
        # Python ve kütüphane bilgileri
        python_info = {
            'version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'cv2_version': cv2.__version__,
            'tensorflow_version': tf.__version__
        }
        
        # İşletim sistemi ve donanım bilgileri
        system = {
            'os': platform.system(),
            'release': platform.release(),
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3)  # GB
        }
        
        # GPU bilgisi (NVIDIA için)
        gpu_info = {}
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', 
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_data = []
                    for line in result.stdout.strip().split('\n'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            gpu_data.append({
                                'name': parts[0].strip(),
                                'memory_total': float(parts[1].strip()),
                                'memory_used': float(parts[2].strip())
                            })
                    gpu_info['devices'] = gpu_data
            elif platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_names = [line.strip() for line in result.stdout.strip().split('\n')[1:] if line.strip()]
                    gpu_info['devices'] = [{'name': name} for name in gpu_names]
        except Exception as e:
            gpu_info['error'] = str(e)
        
        # Thread bilgileri
        current_threads = threading.enumerate()
        thread_info = [{'name': t.name, 'daemon': t.daemon} for t in current_threads]
        
        # Toplam bilgiler
        return jsonify({
            'python': python_info,
            'system': system,
            'gpu': gpu_info,
            'threads': thread_info,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Sistem bilgisi alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@debug_bp.route('/queue-status', methods=['GET'])
def queue_status():
    """
    Kuyruk durumunu döndürür.
    """
    from app.services.queue_service import get_queue_status
    
    try:
        status = get_queue_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Kuyruk durumu alınırken hata: {str(e)}")
        return jsonify({'error': f'Kuyruk durumu alınırken bir hata oluştu: {str(e)}'}), 500

@debug_bp.route('/uploaded-files-count', methods=['GET'])
def uploaded_files_count():
    """
    Yüklü dosya sayısını döndürür.
    """
    from app.models.file import File
    from app.models.analysis import Analysis
    
    try:
        # Son 24 saat içinde yüklenen dosyaları kontrol et
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Son 24 saat içinde yüklenen dosyaları al
        recent_files = File.query.filter(File.created_at >= cutoff_time).all()
        recent_file_ids = [f.id for f in recent_files]
        
        # Bu dosyalardan tamamlanmış analizlere sahip olanları bul
        if recent_file_ids:
            completed_analyses = Analysis.query.filter(
                Analysis.file_id.in_(recent_file_ids),
                Analysis.status == 'completed'
            ).all()
            completed_file_ids = set(analysis.file_id for analysis in completed_analyses)
            
            # Aktif analizlere sahip dosyaları bul (processing, queued)
            active_analyses = Analysis.query.filter(
                Analysis.file_id.in_(recent_file_ids),
                Analysis.status.in_(['processing', 'queued', 'pending'])
            ).all()
            active_file_ids = set(analysis.file_id for analysis in active_analyses)
            
            # Yüklü ama henüz analizi tamamlanmamış veya aktif analizi olan dosya sayısı
            uploaded_files_count = len(active_file_ids)
        else:
            uploaded_files_count = 0
            completed_file_ids = set()
        
        return jsonify({
            'uploaded_files_count': uploaded_files_count,
            'total_recent_files': len(recent_files),
            'completed_files': len(completed_file_ids),
            'cutoff_time': cutoff_time.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Yüklü dosya sayısı alınırken hata: {str(e)}")
        return jsonify({'error': f'Yüklü dosya sayısı alınırken bir hata oluştu: {str(e)}'}), 500

@debug_bp.route('/repair-stuck-analyses', methods=['POST'])
def repair_stuck_analyses_endpoint():
    """Takılmış analizleri düzeltir"""
    try:
        from app.services.debug_service import repair_stuck_analyses
        repair_stuck_analyses()
        
        return jsonify({
            'status': 'success',
            'message': 'Takılmış analizler kontrol edildi ve düzeltildi'
        }), 200
        
    except Exception as e:
        logger.error(f"Takılmış analizleri düzeltirken hata: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@debug_bp.route('/test-age-base-model', methods=['GET'])
def test_age_base_model():
    """Yaş base model kontrolünü test et"""
    try:
        from flask import current_app
        import os
        
        base_model_path = current_app.config['AGE_MODEL_BASE_PATH']
        buffalo_path = current_app.config['INSIGHTFACE_AGE_MODEL_BASE_PATH']
        
        custom_age_file = os.path.join(base_model_path, 'model.pth')
        buffalo_file = os.path.join(buffalo_path, 'w600k_r50.onnx')
        
        custom_age_exists = os.path.exists(custom_age_file)
        buffalo_exists = os.path.exists(buffalo_file)
        base_model_exists = custom_age_exists and buffalo_exists
        
        return jsonify({
            'status': 'success',
            'base_model_path': base_model_path,
            'buffalo_path': buffalo_path,
            'custom_age_file': custom_age_file,
            'buffalo_file': buffalo_file,
            'custom_age_exists': custom_age_exists,
            'buffalo_exists': buffalo_exists,
            'base_model_exists': base_model_exists
        }), 200
        
    except Exception as e:
        logger.error(f"Base model test hatası: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@debug_bp.route('/health')
def health_check():
    """Sağlık kontrolü endpoint'i"""
    return jsonify({
        'status': 'healthy',
        'message': 'Debug routes are working'
    })

bp = debug_bp
