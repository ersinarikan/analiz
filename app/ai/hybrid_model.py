#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Hibrit Model Base Sınıfı
=================================

Bu modül farklı AI modellerini kombine eden hibrit model mimarisinin
temel sınıfını içerir. Alt sınıflar bu base class'ı extend ederek
kendi hibrit modellerini oluşturabilir.
"""

import os
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BaseHybridModel:
    """
    Temel hibrit model sınıfı
    
    Bu sınıf birden fazla AI modelini kombine ederek daha güvenilir
    tahminler üretmek için kullanılır. Her model türü için ağırlık
    belirleyerek ağırlıklı ortalama sonuçları hesaplar.
    
    Desteklenen model türleri:
    - CLIP: Görsel-metin analizi
    - CNN: Konvolüsyonel sinir ağları  
    - ResNet: Derin residual ağlar
    - Scene: Sahne analizi modelleri
    """
    
    def __init__(self, model_type: str):
        """
        Temel hibrit modeli başlatır
        
        Args:
            model_type: Model türü ('content', 'age', vb.)
        """
        self.model_type = model_type
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models', model_type)
        
        # Model dizinini oluştur
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model ağırlıkları - toplam 1.0 olmalı
        self.weights = {
            'clip': 0.4,     # CLIP model ağırlığı
            'cnn': 0.3,      # CNN model ağırlığı
            'resnet': 0.2,   # ResNet model ağırlığı
            'scene': 0.1     # Scene model ağırlığı
        }
        
        # Yüklenen modellerin saklanacağı dictionary
        self.models = {}
        
    def load_models(self) -> bool:
        """
        Tüm modelleri yükler
        
        Bu metod alt sınıflar tarafından implement edilmelidir.
        Her model türü için gerekli modelleri yükler ve self.models
        dictionary'sine ekler.
        
        Returns:
            bool: Modeller başarıyla yüklendi mi?
        """
        raise NotImplementedError("Alt sınıflar load_models metodunu implement etmelidir")
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Görüntü üzerinde tahmin yapar
        
        Tüm modelleri kullanarak tahmin yapar ve ağırlıklı ortalama
        ile final sonucu hesaplar.
        
        Args:
            image: Analiz edilecek görüntü (numpy array)
            
        Returns:
            dict: Tahmin sonuçları ve güven skorları
        """
        raise NotImplementedError("Alt sınıflar predict metodunu implement etmelidir")
        
    def train(self, training_data: Dict[str, Any]) -> bool:
        """
        Modeli eğitim verisi ile eğitir
        
        Args:
            training_data: Eğitim verisi sözlüğü
            
        Returns:
            bool: Eğitim başarılı mı?
        """
        raise NotImplementedError("Alt sınıflar train metodunu implement etmelidir")
        
    def save_models(self):
        """
        Eğitilmiş modelleri diske kaydeder
        
        Tüm model ağırlıklarını ve konfigürasyonları
        model dizinine kaydeder.
        """
        raise NotImplementedError("Alt sınıflar save_models metodunu implement etmelidir")
        
    def _is_correct(self, prediction: Any, actual: Any) -> bool:
        """
        Tahmin doğruluğunu kontrol eder
        
        Args:
            prediction: Model tahmini
            actual: Gerçek değer
            
        Returns:
            bool: Tahmin doğru mu?
        """
        raise NotImplementedError("Alt sınıflar _is_correct metodunu implement etmelidir")
        
    def _calculate_error(self, prediction: Any, actual: Any) -> float:
        """
        Tahmin hatasını hesaplar
        
        Args:
            prediction: Model tahmini
            actual: Gerçek değer
            
        Returns:
            float: Hata değeri (0.0 - 1.0 arası)
        """
        raise NotImplementedError("Alt sınıflar _calculate_error metodunu implement etmelidir")
        
    def _weighted_average(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Ağırlıklı ortalama tahminleri hesaplar
        
        Args:
            predictions: Her model için tahmin sözlüğü
            
        Returns:
            dict: Ağırlıklı ortalama sonuçları
        """
        raise NotImplementedError("Alt sınıflar _weighted_average metodunu implement etmelidir")
        
    def _calculate_confidence(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """
        Genel güven skorunu hesaplar
        
        Modeller arası uyum ve individual güven skorlarını
        kombine ederek genel güven hesaplar.
        
        Args:
            predictions: Her model için tahmin sözlüğü
            
        Returns:
            float: Genel güven skoru (0.0 - 1.0 arası)
        """
        raise NotImplementedError("Alt sınıflar _calculate_confidence metodunu implement etmelidir") 