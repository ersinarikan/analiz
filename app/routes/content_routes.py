from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from app.ai.content_analyzer import ContentAnalyzer
import logging

logger = logging.getLogger(__name__)

# Blueprint oluştur
content_bp = Blueprint('content', __name__)

# Content Analyzer örneği
content_analyzer = ContentAnalyzer()

@content_bp.route('/analyze', methods=['POST'])
def analyze_content():
    """Görüntü içeriğini analiz et"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        # Görüntüyü oku
        file = request.files['image']
        image = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
            
        # İçerik analizi yap (analyze_image numpy array kabul eder)
        try:
            scores = content_analyzer.analyze_image(image)
            # Tuple'ı dict'e çevir
            categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug', 
                         'alcohol', 'gambling', 'hate_speech', 'self_harm', 'safe']
            results = {
                'success': True,
                'scores': dict(zip(categories, scores[:10])),
                'detected_objects': scores[10] if len(scores) > 10 else []
            }
            return jsonify(results)
        except Exception as analyze_err:
            return jsonify({
                'success': False,
                'error': str(analyze_err)
            }), 500
        
    except Exception as e:
        logger.error(f"Content analysis failed: {str(e)}")
        return jsonify({'error': str(e)}), 500 