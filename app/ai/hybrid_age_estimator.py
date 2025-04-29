import numpy as np
import tensorflow as tf
import torch
import logging
from deepface import DeepFace
import clip
from PIL import Image
import cv2
from typing import Dict, Any, List, Optional
from .base_hybrid_model import BaseHybridModel
import os

logger = logging.getLogger(__name__)

class HybridAgeEstimator(BaseHybridModel):
    def __init__(self):
        super().__init__('age')
        self.models = {
            'deepface': {
                'model': None,
                'weight': 0.4,
                'trainable': False
            },
            'clip': {
                'model': None,
                'weight': 0.3,
                'trainable': False
            },
            'cnn': {
                'model': None,
                'weight': 0.3,
                'trainable': True
            }
        }
        
        # CLIP için yaş şablonları
        self.age_templates = [
            "This person appears to be a child under 12 years old.",
            "This person appears to be a teenager between 13-19 years old.",
            "This person appears to be a young adult in their twenties.",
            "This person appears to be an adult in their thirties.",
            "This person appears to be middle-aged in their forties or fifties.",
            "This person appears to be elderly, over 60 years old."
        ]
        
        # Modelleri yükle
        self.load_models()

    def load_models(self) -> bool:
        """Modelleri yükle"""
        try:
            # CLIP modelini yükle
            self.models['clip']['model'], self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP modeli yüklendi")
            
            # Custom CNN modelini yükle
            self.models['cnn']['model'] = self._create_cnn_model()
            
            # Eğer kaydedilmiş model varsa yükle
            model_path = f"{self.model_dir}/cnn.h5"
            if os.path.exists(model_path):
                self.models['cnn']['model'].load_weights(model_path)
                logger.info(f"CNN model ağırlıkları yüklendi: {model_path}")
            
            # DeepFace için model yükleme gerekmez (runtime'da yüklenir)
            logger.info("Tüm modeller başarıyla yüklendi")
            return True
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False

    def _create_cnn_model(self):
        """Custom CNN modelini oluştur"""
        try:
            # Model mimarisi
            inputs = tf.keras.Input(shape=(224, 224, 3))
            
            # Convolutional blocks
            x = self._conv_block(inputs, 32, 'block1')
            x = self._conv_block(x, 64, 'block2')
            x = self._conv_block(x, 128, 'block3')
            
            # Dense layers
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            outputs = tf.keras.layers.Dense(101, activation='softmax')(x)
            
            # Model oluştur
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Derle
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"CNN model oluşturma hatası: {str(e)}")
            return None
            
    def _conv_block(self, x, filters, name):
        """Konvolüsyon bloğu"""
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.Activation('relu', name=f'{name}_relu1')(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f'{name}_pool')(x)
        return x

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Yaş tahmini yap"""
        try:
            predictions = {}
            
            # DeepFace tahmini
            try:
                deepface_result = DeepFace.analyze(
                    img_path=image,
                    actions=['age'],
                    enforce_detection=False,
                    silent=True
                )
                predictions['deepface'] = {
                    'age': deepface_result[0]['age'] if isinstance(deepface_result, list) else deepface_result['age'],
                    'confidence': 0.8
                }
            except Exception as e:
                logger.error(f"DeepFace tahmin hatası: {str(e)}")
            
            # CLIP tahmini
            try:
                if self.models['clip']['model'] is not None:
                    clip_result = self._predict_with_clip(image)
                    if clip_result:
                        predictions['clip'] = clip_result
            except Exception as e:
                logger.error(f"CLIP tahmin hatası: {str(e)}")
            
            # CNN tahmini
            try:
                if self.models['cnn']['model'] is not None:
                    cnn_result = self._predict_with_cnn(image)
                    if cnn_result:
                        predictions['cnn'] = cnn_result
            except Exception as e:
                logger.error(f"CNN tahmin hatası: {str(e)}")
            
            # Ağırlıklı ortalama hesapla
            final_age = self._weighted_average(predictions)
            
            return {
                'age': final_age,
                'confidence': self._calculate_confidence(predictions),
                'model_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return None

    def _predict_with_clip(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """CLIP ile tahmin yap"""
        try:
            # Görüntüyü PIL formatına dönüştür
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # CLIP girişi hazırla
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            text_inputs = clip.tokenize(self.age_templates).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                image_features = self.models['clip']['model'].encode_image(image_input)
                text_features = self.models['clip']['model'].encode_text(text_inputs)
                
                # Normalize et
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Benzerlik skorları
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # En yüksek olasılıklı yaş aralığını seç
            probs = similarity[0].cpu().numpy()
            max_idx = np.argmax(probs)
            
            # Yaş aralıkları
            age_ranges = [6, 16, 25, 35, 45, 65]
            estimated_age = age_ranges[max_idx]
            
            return {
                'age': float(estimated_age),
                'confidence': float(probs[max_idx])
            }
            
        except Exception as e:
            logger.error(f"CLIP tahmin hatası: {str(e)}")
            return None

    def _predict_with_cnn(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """CNN ile tahmin yap"""
        try:
            # Görüntüyü hazırla
            processed_image = cv2.resize(image, (224, 224))
            processed_image = processed_image / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)
            
            # Tahmin yap
            predictions = self.models['cnn']['model'].predict(processed_image, verbose=0)
            age = np.argmax(predictions[0])
            confidence = float(predictions[0][age])
            
            return {
                'age': float(age),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"CNN tahmin hatası: {str(e)}")
            return None

    def _weighted_average(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Ağırlıklı ortalama yaş hesapla"""
        try:
            total_weight = 0
            weighted_sum = 0
            
            for model_name, pred in predictions.items():
                if pred is not None:
                    weight = self.models[model_name]['weight'] * pred['confidence']
                    weighted_sum += pred['age'] * weight
                    total_weight += weight
            
            return round(weighted_sum / total_weight) if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"Ağırlıklı ortalama hesaplama hatası: {str(e)}")
            return 0

    def _calculate_confidence(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Genel güven skorunu hesapla"""
        try:
            confidences = []
            for pred in predictions.values():
                if pred is not None and 'confidence' in pred:
                    confidences.append(pred['confidence'])
            
            return np.mean(confidences) if confidences else 0.5
            
        except Exception as e:
            logger.error(f"Güven skoru hesaplama hatası: {str(e)}")
            return 0.5

    def train(self, training_data: Dict[str, Any]) -> bool:
        """CNN modelini eğit"""
        try:
            if not self.models['cnn']['trainable']:
                return False
                
            X_train = training_data['images']
            y_train = training_data['ages']
            
            # Modeli eğit
            history = self.models['cnn']['model'].fit(
                X_train, y_train,
                epochs=5,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Modeli kaydet
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {str(e)}")
            return False

    def _is_correct(self, prediction: Any, actual: Any) -> bool:
        """Tahmin doğruluğunu kontrol et (±5 yaş tolerans)"""
        try:
            return abs(float(prediction['age']) - float(actual)) <= 5
        except:
            return False

    def _calculate_error(self, prediction: Any, actual: Any) -> float:
        """Tahmin hatasını hesapla (MAE)"""
        try:
            return abs(float(prediction['age']) - float(actual))
        except:
            return float('inf') 