import requests
import cv2
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_age_analysis():
    """Test yaş analizi API'sini"""
    try:
        # Test görüntüsünü yükle
        image_path = 'test_images/face.jpg'
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Görüntü yüklenemedi: {image_path}")
            return
        
        # Görüntüyü dosya olarak hazırla
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'image': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        # API'ye istek gönder
        response = requests.post(
            'http://localhost:5000/api/age/analyze',
            files=files
        )
        
        # Sonuçları kontrol et
        if response.status_code == 200:
            result = response.json()
            logger.info("API yanıtı:")
            logger.info(json.dumps(result, indent=2))
        else:
            logger.error(f"API hatası: {response.status_code}")
            logger.error(response.text)
            
    except Exception as e:
        logger.error(f"Test hatası: {str(e)}")

if __name__ == '__main__':
    test_age_analysis() 