import os
import sys
import cv2
import numpy as np
import logging
from flask import Flask

# Proje kök dizinini Python yoluna ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.ai.age_estimator import AgeEstimator
from app.ai.content_analyzer import ContentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_analyzers():
    """Yaş ve içerik analizini test et"""
    try:
        # Flask uygulaması oluştur
        app = Flask(__name__)
        app.config['MODELS_FOLDER'] = os.path.join(project_root, 'storage', 'models')
        
        # Test görüntüsü yükle
        image_path = os.path.join('storage', 'test_images', 'test1.jpg')
        if not os.path.exists(image_path):
            logger.error(f"Test görüntüsü bulunamadı: {image_path}")
            return
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Görüntü yüklenemedi")
            return
            
        with app.app_context():
            # Yaş tahmini
            logger.info("Yaş tahmini yapılıyor...")
            age_estimator = AgeEstimator()
            age_results = age_estimator.estimate_age(image)
            
            if age_results:
                logger.info("\nYaş Tahmini Sonuçları:")
                for i, result in enumerate(age_results, 1):
                    logger.info(f"\nYüz #{i}:")
                    logger.info(f"Tahmini Yaş: {result['age']}")
                    logger.info(f"Güvenilirlik: {result['confidence']:.2f}")
                    
                    if 'models' in result:
                        logger.info("\nModel Detayları:")
                        if result['models']['deepface']:
                            logger.info(f"DeepFace Tahmini: {result['models']['deepface']['age']}")
                            logger.info(f"DeepFace Güvenilirlik: {result['models']['deepface']['confidence']:.2f}")
                        if result['models']['clip']:
                            logger.info(f"CLIP Tahmini: {result['models']['clip']['age']}")
                            logger.info(f"CLIP Güvenilirlik: {result['models']['clip']['confidence']:.2f}")
            else:
                logger.warning("Yüz tespit edilemedi veya yaş tahmini yapılamadı")
                
            # İçerik analizi
            logger.info("\nİçerik analizi yapılıyor...")
            content_analyzer = ContentAnalyzer()
            content_results = content_analyzer.analyze_content(image)
            
            if content_results['success']:
                logger.info("\nİçerik Analizi Sonuçları:")
                for category, prediction in content_results['predictions'].items():
                    logger.info(f"\n{category.title()} Kategorisi:")
                    logger.info(f"Skor: {prediction['score']:.2f}")
                    logger.info(f"Tespit: {'Evet' if prediction['detected'] else 'Hayır'}")
                    logger.info(f"Güvenilirlik: {prediction['confidence']:.2f}")
                    
                    if 'models' in prediction:
                        logger.info("\nModel Detayları:")
                        if prediction['models']['cnn']:
                            logger.info(f"CNN Skoru: {prediction['models']['cnn']['score']:.2f}")
                            logger.info(f"CNN Güvenilirlik: {prediction['models']['cnn']['confidence']:.2f}")
                        if prediction['models']['clip']:
                            logger.info(f"CLIP Skoru: {prediction['models']['clip']['score']:.2f}")
                            logger.info(f"CLIP Güvenilirlik: {prediction['models']['clip']['confidence']:.2f}")
            else:
                logger.error(f"İçerik analizi hatası: {content_results.get('error', 'Bilinmeyen hata')}")
                
    except Exception as e:
        logger.error(f"Test sırasında hata: {str(e)}")

if __name__ == '__main__':
    test_analyzers() 