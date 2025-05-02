from flask import Blueprint, jsonify, current_app
import logging
from app.services.debug_service import test_numpy_serialization, debug_object
import numpy as np
import json

logger = logging.getLogger(__name__)
bp = Blueprint('debug', __name__, url_prefix='/debug')

@bp.route('/test_serialization', methods=['GET'])
def test_serialization():
    """NumPy veri türlerinin JSON serileştirme testini çalıştırır."""
    success, result = test_numpy_serialization()
    
    return jsonify({
        'success': success,
        'result': result if success else None,
        'error': None if success else result
    })

@bp.route('/test_content_detection', methods=['GET'])
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