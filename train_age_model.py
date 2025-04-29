import os
import numpy as np
from app.ai.age_estimator import AgeEstimator
from app.ai.hybrid_analyzer import HybridAgeTrainer
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Eğitim verilerini hazırla
        trainer = HybridAgeTrainer()
        X_train, y_train = trainer.prepare_training_data()
        
        if X_train is None or y_train is None:
            logger.error("Eğitim verileri hazırlanamadı")
            return
            
        logger.info(f"Eğitim verileri hazırlandı: {len(X_train)} örnek")
        
        # Modeli eğit
        estimator = AgeEstimator()
        success = estimator.retrain_model(
            training_data=(X_train, y_train),
            epochs=10,
            batch_size=32
        )
        
        if success:
            logger.info("Model başarıyla eğitildi ve kaydedildi")
        else:
            logger.error("Model eğitimi başarısız oldu")
            
    except Exception as e:
        logger.error(f"Eğitim hatası: {str(e)}")

if __name__ == "__main__":
    main() 