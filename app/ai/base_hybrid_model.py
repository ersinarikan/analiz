import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseHybridModel(ABC):
    """Temel hybrid model sınıfı"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.models = {}
        self.performance_metrics = {}
        self.feedback_history = []
        self.model_weights = {}
        self.device = self._setup_device()
        self.model_dir = os.path.join('storage', 'models', model_name)
        
        # Model dizinini oluştur
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """GPU veya CPU aygıtını ayarla"""
        if torch.cuda.is_available():
            logger.info(f"GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda')
        else:
            logger.info("CPU kullanılıyor")
            return torch.device('cpu')
    
    @abstractmethod
    def load_models(self) -> bool:
        """Alt sınıflar tarafından implement edilecek"""
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Alt sınıflar tarafından implement edilecek"""
        pass
    
    @abstractmethod
    def train(self, training_data: Dict[str, Any]) -> bool:
        """Alt sınıflar tarafından implement edilecek"""
        pass
    
    def save_models(self) -> bool:
        """Modelleri kaydet"""
        try:
            for model_name, model_info in self.models.items():
                if model_info['model'] is not None:
                    model_path = os.path.join(self.model_dir, f"{model_name}.h5")
                    if model_info['trainable']:
                        model_info['model'].save(model_path)
                        logger.info(f"Model kaydedildi: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {str(e)}")
            return False
    
    def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """Geri bildirim ekle ve performans metriklerini güncelle"""
        try:
            self.feedback_history.append(feedback)
            self._update_performance_metrics(feedback)
        except Exception as e:
            logger.error(f"Geri bildirim ekleme hatası: {str(e)}")
    
    def _update_performance_metrics(self, feedback: Dict[str, Any]) -> None:
        """Performans metriklerini güncelle"""
        try:
            for model_name, model_info in self.models.items():
                if model_name in feedback['predictions']:
                    metrics = self.performance_metrics.setdefault(model_name, {
                        'accuracy': 0.0,
                        'error': 0.0,
                        'samples': 0
                    })
                    
                    # Metrikleri güncelle
                    metrics['samples'] += 1
                    n = metrics['samples']
                    
                    # Doğruluk ve hata hesapla
                    prediction = feedback['predictions'][model_name]
                    actual = feedback['actual']
                    
                    accuracy = 1.0 if self._is_correct(prediction, actual) else 0.0
                    error = self._calculate_error(prediction, actual)
                    
                    # Hareketli ortalama güncelle
                    metrics['accuracy'] = (metrics['accuracy'] * (n-1) + accuracy) / n
                    metrics['error'] = (metrics['error'] * (n-1) + error) / n
                    
                    # Model ağırlıklarını güncelle
                    self._update_model_weights()
                    
        except Exception as e:
            logger.error(f"Performans metrikleri güncelleme hatası: {str(e)}")
    
    @abstractmethod
    def _is_correct(self, prediction: Any, actual: Any) -> bool:
        """Alt sınıflar tarafından implement edilecek"""
        pass
    
    @abstractmethod
    def _calculate_error(self, prediction: Any, actual: Any) -> float:
        """Alt sınıflar tarafından implement edilecek"""
        pass
    
    def _update_model_weights(self) -> None:
        """Model ağırlıklarını performansa göre güncelle"""
        try:
            total_accuracy = 0.0
            weights = {}
            
            # Her model için ağırlık hesapla
            for model_name, metrics in self.performance_metrics.items():
                if metrics['samples'] > 0:
                    # Accuracy ve error'u kombine et
                    performance = metrics['accuracy'] / (1 + metrics['error'])
                    weights[model_name] = performance
                    total_accuracy += performance
            
            # Ağırlıkları normalize et
            if total_accuracy > 0:
                self.model_weights = {
                    model: weight/total_accuracy 
                    for model, weight in weights.items()
                }
            
        except Exception as e:
            logger.error(f"Model ağırlıkları güncelleme hatası: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Performans metriklerini döndür"""
        return {
            'metrics': self.performance_metrics,
            'weights': self.model_weights
        } 