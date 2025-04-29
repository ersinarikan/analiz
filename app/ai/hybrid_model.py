import os
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BaseHybridModel:
    """Temel hibrit model sınıfı"""
    
    def __init__(self, model_type: str):
        """Temel hibrit modeli başlat"""
        self.model_type = model_type
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models', model_type)
        
        # Model dizinini oluştur
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model ağırlıkları
        self.weights = {
            'clip': 0.4,
            'cnn': 0.3,
            'resnet': 0.2,
            'scene': 0.1
        }
        
        # Model durumları
        self.models = {}
        
    def load_models(self) -> bool:
        """Modelleri yükle - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Tahmin yap - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def train(self, training_data: Dict[str, Any]) -> bool:
        """Model eğitimi - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def save_models(self):
        """Modelleri kaydet - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def _is_correct(self, prediction: Any, actual: Any) -> bool:
        """Tahmin doğruluğunu kontrol et - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def _calculate_error(self, prediction: Any, actual: Any) -> float:
        """Tahmin hatasını hesapla - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def _weighted_average(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Ağırlıklı ortalama tahminleri hesapla - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError
        
    def _calculate_confidence(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Genel güven skorunu hesapla - Alt sınıflar tarafından uygulanmalı"""
        raise NotImplementedError 