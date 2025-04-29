from flask import Blueprint, request, jsonify
from app.ai.age_estimator import AgeEstimator
from app.ai.content_analyzer import ContentAnalyzer
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
bp = Blueprint('test', __name__, url_prefix='/api/test')

@bp.route('/analyze', methods=['POST'])
def test_analyze():
    """Test endpoint for hybrid model analysis"""
    try:
        # Görüntüyü al
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        # Görüntüyü numpy dizisine dönüştür
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
            
        # Yaş tahmini yap
        age_estimator = AgeEstimator()
        age_results = age_estimator.analyze_image(image)
        
        # İçerik analizi yap
        content_analyzer = ContentAnalyzer()
        content_results, detected_objects = content_analyzer.analyze_image(image)
        
        return jsonify({
            'age_analysis': age_results,
            'content_analysis': content_results,
            'detected_objects': detected_objects
        })
        
    except Exception as e:
        logger.error(f"Test analiz hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500 