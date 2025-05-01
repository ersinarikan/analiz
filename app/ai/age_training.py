import os
import numpy as np
import cv2
import logging
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from app.ai.cnn_age_model import train_model, save_model

logger = logging.getLogger(__name__)

class AgeModelTrainer:
    def __init__(self, model_path=None):
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / 'storage' / 'models' / 'age' / 'age_model.h5'
        self.model_path = Path(model_path)
        self.training_data_path = self.model_path.parent / 'training_data'
        self.training_data_path.mkdir(exist_ok=True)
        
        # Eğitim verilerini saklamak için
        self.face_images = []
        self.age_labels = []
        self.confidence_scores = []
        
    def add_training_sample(self, face_image, predicted_age, confidence, feedback_age=None):
        """
        Eğitim örneği ekler.
        
        Args:
            face_image: Yüz görüntüsü
            predicted_age: InsightFace'in tahmin ettiği yaş
            confidence: InsightFace'in güven skoru
            feedback_age: Kullanıcı geri bildirimi (varsa)
        """
        try:
            # Görüntüyü ön işle
            img = cv2.resize(face_image, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            
            # Veriyi sakla
            self.face_images.append(img)
            
            # Eğer kullanıcı geri bildirimi varsa onu kullan, yoksa tahmin edilen yaşı kullan
            age = feedback_age if feedback_age is not None else predicted_age
            self.age_labels.append(age)
            self.confidence_scores.append(confidence)
            
            # Eğitim verisini kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_path = self.training_data_path / f"sample_{timestamp}"
            sample_path.mkdir(exist_ok=True)
            
            # Görüntüyü kaydet
            cv2.imwrite(str(sample_path / "face.jpg"), face_image)
            
            # Metadata'yı kaydet
            metadata = {
                "predicted_age": float(predicted_age),
                "confidence": float(confidence),
                "feedback_age": feedback_age,
                "timestamp": timestamp
            }
            with open(sample_path / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            logger.info(f"Eğitim örneği eklendi: {sample_path}")
            
            # Belirli sayıda örnek toplandığında otomatik eğitim başlat
            if len(self.face_images) >= 100:  # Her 100 örnekte bir eğit
                self.train_model()
                
        except Exception as e:
            logger.error(f"Eğitim örneği ekleme hatası: {str(e)}")
    
    def train_model(self):
        """Modeli mevcut verilerle eğitir"""
        try:
            if len(self.face_images) < 10:  # Minimum örnek sayısı
                logger.warning("Yetersiz eğitim verisi")
                return
            
            # Verileri numpy dizilerine dönüştür
            X = np.array(self.face_images)
            y = np.array(self.age_labels)
            
            # Verileri karıştır
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            # Eğitim ve doğrulama setlerine ayır
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # Mevcut modeli yükle
            model = tf.keras.models.load_model(str(self.model_path))
            
            # Modeli eğit
            history = train_model(
                model,
                (X_train, y_train),
                (X_val, y_val),
                epochs=10
            )
            
            # Modeli kaydet
            save_model(model, str(self.model_path))
            
            # Eğitim verilerini temizle
            self.face_images = []
            self.age_labels = []
            self.confidence_scores = []
            
            logger.info("Model başarıyla eğitildi ve kaydedildi")
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {str(e)}")
    
    def load_saved_samples(self):
        """Kaydedilmiş eğitim örneklerini yükler"""
        try:
            for sample_dir in self.training_data_path.glob("sample_*"):
                if not sample_dir.is_dir():
                    continue
                
                # Metadata'yı oku
                try:
                    with open(sample_dir / "metadata.json", "r") as f:
                        metadata = json.load(f)
                except:
                    continue
                
                # Görüntüyü oku
                face_image = cv2.imread(str(sample_dir / "face.jpg"))
                if face_image is None:
                    continue
                
                # Örneği ekle
                age = metadata.get("feedback_age", metadata.get("predicted_age"))
                confidence = metadata.get("confidence", 0.5)
                
                self.add_training_sample(
                    face_image,
                    metadata["predicted_age"],
                    confidence,
                    feedback_age=metadata.get("feedback_age")
                )
                
            logger.info(f"Toplam {len(self.face_images)} eğitim örneği yüklendi")
            
        except Exception as e:
            logger.error(f"Eğitim örneklerini yükleme hatası: {str(e)}") 