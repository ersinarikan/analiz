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
            
        # İçerik analizi yap
        results = content_analyzer.analyze_content(image)
        
        if not results['success']:
            return jsonify({
                'error': results.get('error', 'Content analysis failed')
            }), 500
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Content analysis failed: {str(e)}")
        return jsonify({'error': str(e)}), 500 