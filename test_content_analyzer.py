import cv2
import numpy as np
from app.ai.hybrid_content_analyzer import HybridContentAnalyzer
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_content_analysis():
    """İçerik analizi testi"""
    try:
        # Analizörü başlat
        analyzer = HybridContentAnalyzer()
        logger.info("İçerik analizörü başlatıldı")
        
        # Test görüntüsünü yükle
        image_path = "test_images/test_image.jpg"  # Test görüntüsünün yolu
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Görüntü yüklenemedi")
            return
            
        # İçerik analizi yap
        results = analyzer.analyze_content(image)
        logger.info("İçerik analizi sonuçları:")
        logger.info(f"Güvenilirlik: {results['confidence']:.2f}")
        logger.info(f"Güvenli mi: {results['is_safe']}")
        
        # Detaylı sonuçları göster
        content_analysis = results['content_analysis']
        if content_analysis['clip']:
            logger.info("\nCLIP Analizi:")
            logger.info(f"İçerik Tipi: {content_analysis['clip']['content_type']}")
            logger.info(f"Güvenilirlik: {content_analysis['clip']['confidence']:.2f}")
            
        if content_analysis['object_detection']:
            logger.info("\nNesne Tespiti:")
            for detection in content_analysis['object_detection']['detections']:
                logger.info(f"Nesne: {detection['class']}, Güvenilirlik: {detection['confidence']:.2f}")
                
        if content_analysis['scene']:
            logger.info("\nSahne Analizi:")
            for scene in content_analysis['scene']['scenes']:
                logger.info(f"Sahne: {scene['category']}, Güvenilirlik: {scene['confidence']:.2f}")
                
        if content_analysis['features']:
            logger.info("\nÖzellik Analizi:")
            for pred in content_analysis['features']['predictions']:
                logger.info(f"Sınıf: {pred['class']}, Güvenilirlik: {pred['confidence']:.2f}")
                
        # Geri bildirim ekle
        feedback_data = {
            'general_rating': 'safe_content',
            'violence_feedback': 'correct',
            'violence_correct_value': 0.8,
            'adult_feedback': 'false_positive',
            'harassment_feedback': 'missed',
            'weapon_feedback': 'correct',
            'weapon_correct_value': 0.9,
            'drug_feedback': 'correct',
            'drug_correct_value': 0.7,
            'feedback_comment': 'Genel olarak doğru tespitler yapıldı.'
        }
        
        success = analyzer.add_feedback(image, feedback_data)
        if success:
            logger.info("\nGeri bildirim başarıyla eklendi")
            
            # Eğitim istatistiklerini göster
            training_stats = analyzer.get_training_stats()
            logger.info("\nEğitim İstatistikleri:")
            logger.info(f"Eğitim Verisi Boyutları: {training_stats['training_data_sizes']}")
            logger.info(f"Minimum Örnek Sayısı: {training_stats['min_samples_for_training']}")
            
            # Önbellek istatistiklerini göster
            cache_stats = analyzer.get_cache_stats()
            logger.info("\nÖnbellek İstatistikleri:")
            logger.info(f"Önbellek Boyutu: {cache_stats['cache_size']}")
            logger.info(f"Önbellek İsabet Oranı: {cache_stats['hit_ratio']:.2f}")
            
    except Exception as e:
        logger.error(f"Test sırasında hata oluştu: {str(e)}")

if __name__ == "__main__":
    test_content_analysis() 