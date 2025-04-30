import torch
import clip
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Union, Optional, Any
import cv2
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
import os
from .base_hybrid_model import BaseHybridModel
import tensorflow as tf

logger = logging.getLogger(__name__)

class HybridContentAnalyzer(BaseHybridModel):
    """İçerik analizi için hibrit model"""
    
    def __init__(self):
        super().__init__('content')
        
        self.categories = {
            'adult': 'yetişkin içerik',
            'violence': 'şiddet içeriği',
            'hate': 'nefret söylemi',
            'drugs': 'uyuşturucu içeriği',
            'safe': 'güvenli içerik'
        }
        
        self.clip_model = None
        self.cnn_model = None
        self.resnet_model = None
        self.scene_model = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self) -> bool:
        """Tüm modelleri yükle"""
        try:
            # CLIP modelini yükle
            try:
                self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.models['clip'] = {'model': self.clip_model, 'trainable': False}
                logger.info("CLIP modeli yüklendi")
            except Exception as e:
                logger.warning(f"CLIP model yükleme hatası: {str(e)}")
                self.models['clip'] = {'model': None, 'trainable': False}
            
            # ResNet modelini yükle
            try:
                resnet_path = os.path.join(self.model_dir, 'content_resnet.h5')
                if os.path.exists(resnet_path):
                    try:
                        self.resnet_model = tf.keras.models.load_model(resnet_path)
                        self.models['resnet'] = {'model': self.resnet_model, 'trainable': True}
                        logger.info("ResNet modeli yüklendi")
                    except Exception as e:
                        logger.warning(f"Kaydedilmiş ResNet modeli yüklenemedi, yeni model oluşturuluyor: {str(e)}")
                        self.resnet_model = self._create_resnet_model()
                        self.models['resnet'] = {'model': self.resnet_model, 'trainable': True}
                        logger.info("Yeni ResNet modeli oluşturuldu")
                else:
                    self.resnet_model = self._create_resnet_model()
                    self.models['resnet'] = {'model': self.resnet_model, 'trainable': True}
                    logger.info("Yeni ResNet modeli oluşturuldu")
            except Exception as e:
                logger.warning(f"ResNet model yükleme hatası: {str(e)}")
                self.models['resnet'] = {'model': None, 'trainable': True}
            
            # Scene modelini yükle
            try:
                scene_path = os.path.join(self.model_dir, 'content_scene.h5')
                if os.path.exists(scene_path):
                    try:
                        self.scene_model = tf.keras.models.load_model(scene_path)
                        self.models['scene'] = {'model': self.scene_model, 'trainable': True}
                        logger.info("Scene modeli yüklendi")
                    except Exception as e:
                        logger.warning(f"Kaydedilmiş Scene modeli yüklenemedi, yeni model oluşturuluyor: {str(e)}")
                        self.scene_model = self._create_scene_model()
                        self.models['scene'] = {'model': self.scene_model, 'trainable': True}
                        logger.info("Yeni Scene modeli oluşturuldu")
                else:
                    self.scene_model = self._create_scene_model()
                    self.models['scene'] = {'model': self.scene_model, 'trainable': True}
                    logger.info("Yeni Scene modeli oluşturuldu")
            except Exception as e:
                logger.warning(f"Scene model yükleme hatası: {str(e)}")
                self.models['scene'] = {'model': None, 'trainable': True}
            
            # YOLO modelini yükle
            try:
                yolo_path = os.path.join(self.model_dir, '..', 'detection', 'yolov8n.pt')
                yolo_path = os.path.abspath(yolo_path)
                if os.path.exists(yolo_path):
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(yolo_path)
                    logger.info(f"YOLO modeli yüklendi: {yolo_path}")
                else:
                    logger.error(f"YOLO model dosyası bulunamadı: {yolo_path}")
                    self.yolo_model = None
            except Exception as e:
                logger.error(f"YOLO model yükleme hatası: {str(e)}")
                self.yolo_model = None
            
            # Model ağırlıklarını başlat
            self.weights = {
                'resnet': 0.4,
                'scene': 0.3,
                'clip': 0.3,
                'object_detection': 0.3
            }
            
            # En az bir model yüklendiyse başarılı
            loaded_models = [name for name, info in self.models.items() if info['model'] is not None]
            if loaded_models:
                logger.info(f"Başarıyla yüklenen modeller: {', '.join(loaded_models)}")
                return True
            else:
                logger.error("Hiçbir model yüklenemedi")
                return False
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
            
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Verilen görüntü için tahmin yap.
        
        Args:
            image: Numpy array formatında görüntü
            
        Returns:
            Dict formatında tahmin sonuçları:
            {
                'predictions': {category_name: probability},
                'confidence': float,
                'error': str or None
            }
        """
        try:
            # Her modelden tahmin al
            predictions = {}
            
            # ResNet tahmini
            try:
                if self.models.get('resnet', {}).get('model') is not None:
                    resnet_preds = self._predict_with_resnet(image)
                    if resnet_preds is not None:
                        predictions['resnet'] = resnet_preds
            except Exception as e:
                logger.error(f"ResNet tahmin hatası: {str(e)}")
            
            # Scene tahmini
            try:
                if self.models.get('scene', {}).get('model') is not None:
                    scene_preds = self._predict_with_scene(image)
                    if scene_preds is not None:
                        predictions['scene'] = scene_preds
            except Exception as e:
                logger.error(f"Scene tahmin hatası: {str(e)}")
            
            # CLIP tahmini
            try:
                if self.models.get('clip', {}).get('model') is not None:
                    clip_preds = self._predict_with_clip(image)
                    if clip_preds is not None:
                        predictions['clip'] = clip_preds
            except Exception as e:
                logger.error(f"CLIP tahmin hatası: {str(e)}")
            
            # Eğer hiçbir model tahmin yapamadıysa
            if not predictions:
                return {
                    'predictions': {cat: 0.0 for cat in self.categories},
                    'confidence': 0.0,
                    'error': 'No models were able to make predictions'
                }
            
            # Ağırlıklı ortalama hesapla
            final_preds = {}
            total_weight = 0.0
            
            for model_name, preds in predictions.items():
                weight = self.weights.get(model_name, 0.0)
                total_weight += weight
                for category in self.categories:
                    if category not in final_preds:
                        final_preds[category] = 0.0
                    final_preds[category] += preds.get(category, 0.0) * weight
            
            # Normalize et
            if total_weight > 0:
                final_preds = {k: v/total_weight for k, v in final_preds.items()}
            else:
                final_preds = {cat: 0.0 for cat in self.categories}
            
            # Güven skorunu hesapla
            confidence = self._calculate_confidence_score(final_preds)
            
            return {
                'predictions': final_preds,
                'confidence': confidence,
                'error': None,
                'model_predictions': predictions  # Her modelin ayrı tahminlerini de ekle
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'predictions': {cat: 0.0 for cat in self.categories},
                'confidence': 0.0,
                'error': str(e)
            }
        
    def _predict_with_clip(self, image: np.ndarray) -> Dict[str, float]:
        """CLIP modeli ile tahmin yap"""
        try:
            # Görüntüyü hazırla
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Metin girdilerini hazırla
            text_inputs = torch.cat([clip.tokenize(f"an image about {cat}") for cat in self.categories.keys()]).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize et
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Benzerlik hesapla
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            return dict(zip(self.categories.keys(), similarity[0].cpu().numpy()))
            
        except Exception as e:
            logger.error(f"CLIP tahmin hatası: {str(e)}")
            return None
            
    def _predict_with_cnn(self, image: np.ndarray) -> Dict[str, float]:
        """CNN modeli ile tahmin yap"""
        try:
            # Görüntüyü hazırla
            image = cv2.resize(image, (224, 224))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Tahmin yap
            predictions = self.cnn_model.predict(image)[0]
            
            return dict(zip(self.categories.keys(), predictions))
            
        except Exception as e:
            logger.error(f"CNN tahmin hatası: {str(e)}")
            return {cat: 0.0 for cat in self.categories}
            
    def _weighted_average(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Ağırlıklı ortalama tahminleri hesapla"""
        final_predictions = {cat: 0.0 for cat in self.categories}
        
        for model, preds in predictions.items():
            weight = self.weights.get(model, 0.0)
            for cat, prob in preds.items():
                final_predictions[cat] += prob * weight
                
        # Normalize et
        total_weight = sum(self.weights[model] for model in predictions.keys())
        if total_weight > 0:
            final_predictions = {cat: prob/total_weight for cat, prob in final_predictions.items()}
            
        return final_predictions
        
    def _calculate_confidence(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Genel güven skorunu hesapla"""
        confidences = []
        
        for model, preds in predictions.items():
            # En yüksek ve ikinci en yüksek tahminler arasındaki fark
            sorted_preds = sorted(preds.values(), reverse=True)
            if len(sorted_preds) >= 2:
                conf = sorted_preds[0] - sorted_preds[1]
                confidences.append(conf * self.weights.get(model, 0.0))
                
        # Ağırlıklı ortalama güven skoru
        if confidences:
            total_weight = sum(self.weights[model] for model in predictions.keys())
            if total_weight > 0:
                return sum(confidences) / total_weight
                
        return 0.0
        
    def train(self, training_data: Dict[str, Any]) -> bool:
        """Model eğitimi"""
        try:
            # CNN modelini eğit
            if self.models.get('cnn'):
                history = self.cnn_model.fit(
                    training_data['images'],
                    training_data['labels'],
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2
                )
                self.save_models()
                return True
                
        except Exception as e:
            logger.error(f"Model eğitim hatası: {str(e)}")
            return False
            
    def save_models(self):
        """Modelleri kaydet"""
        try:
            if self.models.get('cnn'):
                self.cnn_model.save(os.path.join(self.model_dir, 'content_cnn.h5'))
                
            if self.models.get('resnet'):
                self.resnet_model.save(os.path.join(self.model_dir, 'content_resnet.h5'))
                
            if self.models.get('scene'):
                self.scene_model.save(os.path.join(self.model_dir, 'content_scene.h5'))
                
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {str(e)}")
            
    def _is_correct(self, prediction: Dict[str, float], actual: str) -> bool:
        """Tahmin doğruluğunu kontrol et"""
        predicted_category = max(prediction.items(), key=lambda x: x[1])[0]
        return predicted_category == actual
        
    def _calculate_error(self, prediction: Dict[str, float], actual: str) -> float:
        """Tahmin hatasını hesapla"""
        return 1.0 - prediction.get(actual, 0.0)

    def analyze_with_clip(self, image: np.ndarray) -> Optional[Dict]:
        """CLIP ile içerik analizi yap"""
        try:
            # Görüntüyü hazırla
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Metin şablonlarını tokenize et
            text_inputs = clip.tokenize(self.content_templates).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize et
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Benzerlik skorları
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            # Sonuçları döndür
            probs = similarity[0].cpu().numpy()
            max_idx = np.argmax(probs)
            
            return {
                'content_type': self.content_templates[max_idx],
                'confidence': float(probs[max_idx]),
                'all_probs': probs.tolist()
            }
            
        except Exception as e:
            logger.error(f"CLIP analiz hatası: {str(e)}")
            return None
            
    def analyze_with_yolo(self, image: np.ndarray) -> Optional[Dict]:
        """YOLO ile nesne tespiti yap"""
        try:
            # YOLO ile tahmin
            results = self.yolo_model(image)
            
            # Sonuçları işle
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append({
                        'class': self.yolo_model.names[cls],
                        'confidence': conf
                    })
                    
            return {
                'detections': detections,
                'confidence': float(np.mean([d['confidence'] for d in detections])) if detections else 0.0
            }
            
        except Exception as e:
            logger.error(f"YOLO analiz hatası: {str(e)}")
            return None
            
    def analyze_with_resnet(self, image: np.ndarray) -> Optional[Dict]:
        """ResNet ile görsel özellik analizi yap"""
        try:
            # Görüntüyü hazırla
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                features = self.resnet_model(image_input)
                probs = torch.nn.functional.softmax(features, dim=1)
                
            # En yüksek olasılıklı sınıfları al
            top_probs, top_classes = torch.topk(probs, 5)
            
            # Sınıf isimlerini ve olasılıkları eşleştir
            predictions = []
            for prob, cls_idx in zip(top_probs[0], top_classes[0]):
                predictions.append({
                    'class': self.resnet_classes[cls_idx],
                    'confidence': float(prob)
                })
                
            return {
                'predictions': predictions,
                'confidence': float(top_probs[0][0])
            }
            
        except Exception as e:
            logger.error(f"ResNet analiz hatası: {str(e)}")
            return None
            
    def analyze_with_scene(self, image: np.ndarray) -> Optional[Dict]:
        """Sahne sınıflandırması yap"""
        try:
            # Görüntüyü hazırla
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                features = self.scene_model(image_input)
                probs = torch.nn.functional.softmax(features, dim=1)
                
            # En yüksek olasılıklı sahneleri al
            top_probs, top_classes = torch.topk(probs, 5)
            
            # Sahne kategorilerini belirle
            scene_predictions = []
            for prob, cls_idx in zip(top_probs[0], top_classes[0]):
                scene_predictions.append({
                    'category': self.scene_categories[cls_idx],
                    'confidence': float(prob)
                })
                
            return {
                'scenes': scene_predictions,
                'confidence': float(top_probs[0][0])
            }
            
        except Exception as e:
            logger.error(f"Sahne sınıflandırma hatası: {str(e)}")
            return None
            
    def analyze_content(self, image: np.ndarray) -> Dict:
        """Hibrit içerik analizi yap"""
        try:
            # Önbellekte var mı kontrol et
            cached_result = self._check_cache(image)
            if cached_result:
                return cached_result
                
            # Her modelden tahmin al
            clip_result = self.analyze_with_clip(image)
            yolo_result = self.analyze_with_yolo(image)
            resnet_result = self.analyze_with_resnet(image)
            scene_result = self.analyze_with_scene(image)
            
            # Sonuçları birleştir
            combined_result = self.combine_predictions(
                clip_result=clip_result,
                yolo_result=yolo_result,
                resnet_result=resnet_result,
                scene_result=scene_result
            )
            
            # Sonucu önbelleğe al
            if combined_result:
                self._update_cache(image, combined_result)
                
            return combined_result
            
        except Exception as e:
            logger.error(f"İçerik analizi hatası: {str(e)}")
            return {
                'error': str(e),
                'confidence': 0.0
            }
            
    def combine_predictions(self, clip_result: Optional[Dict],
                          yolo_result: Optional[Dict],
                          resnet_result: Optional[Dict],
                          scene_result: Optional[Dict]) -> Dict:
        """Farklı modellerden gelen tahminleri birleştir"""
        results = []
        total_weight = 0
        
        # CLIP sonuçlarını ekle
        if clip_result:
            results.append({
                'content_type': clip_result['content_type'],
                'confidence': clip_result['confidence'],
                'weight': self.weights['clip']
            })
            total_weight += self.weights['clip']
            
        # YOLO sonuçlarını ekle
        if yolo_result:
            results.append({
                'detections': yolo_result['detections'],
                'confidence': yolo_result['confidence'],
                'weight': self.weights['object_detection']
            })
            total_weight += self.weights['object_detection']
            
        # ResNet sonuçlarını ekle
        if resnet_result:
            results.append({
                'predictions': resnet_result['predictions'],
                'confidence': resnet_result['confidence'],
                'weight': self.weights['features']
            })
            total_weight += self.weights['features']
            
        # Scene sonuçlarını ekle
        if scene_result:
            results.append({
                'scenes': scene_result['scenes'],
                'confidence': scene_result['confidence'],
                'weight': self.weights['scene']
            })
            total_weight += self.weights['scene']
            
        # Hiç sonuç yoksa None döndür
        if not results:
            return None
            
        # Ağırlıklı güvenilirlik skoru hesapla
        weighted_confidence = sum(r['confidence'] * r['weight'] for r in results) / total_weight
        
        return {
            'content_analysis': {
                'clip': clip_result,
                'object_detection': yolo_result,
                'features': resnet_result,
                'scene': scene_result
            },
            'confidence': weighted_confidence,
            'is_safe': weighted_confidence > 0.7  # Güvenilirlik eşiği
        }
        
    def update_weights(self, new_weights: Dict[str, float]):
        """Model ağırlıklarını güncelle"""
        if abs(sum(new_weights.values()) - 1.0) > 0.001:
            raise ValueError("Ağırlıkların toplamı 1 olmalıdır")
            
        self.weights.update(new_weights) 

    def add_feedback(self, image: np.ndarray, feedback_data: Dict):
        """Geri bildirim ekle ve model performansını güncelle"""
        try:
            # Analiz yap
            results = self.analyze_content(image)
            if not results:
                return False
                
            # Geri bildirim verilerini kaydet
            feedback = {
                'image': image,
                'predictions': results['content_analysis'],
                'feedback': feedback_data
            }
            self.feedback_data.append(feedback)
            
            # Eğitim verisi olarak ekle
            self._add_feedback_to_training_data(feedback)
            
            # Model performanslarını güncelle
            self._update_performance_metrics(feedback)
            
            # Model ağırlıklarını optimize et
            self._optimize_weights()
            
            return True
            
        except Exception as e:
            logger.error(f"Geri bildirim ekleme hatası: {str(e)}")
            return False
            
    def _add_feedback_to_training_data(self, feedback: Dict):
        """Geri bildirimi eğitim verisine dönüştür"""
        try:
            image = feedback['image']
            feedback_data = feedback['feedback']
            
            # CLIP verisi
            if 'general_rating' in feedback_data:
                self.training_data['clip'].append({
                    'image': image,
                    'label': feedback_data['general_rating']
                })
                
            # Kategori bazlı geri bildirimler
            for category in self.content_categories.keys():
                if f'{category}_feedback' in feedback_data:
                    feedback_type = feedback_data[f'{category}_feedback']
                    correct_value = feedback_data.get(f'{category}_correct_value')
                    
                    if feedback_type == 'correct' and correct_value is not None:
                        # Doğru tespit edilmiş
                        self.training_data['object_detection'].append({
                            'image': image,
                            'labels': [category],
                            'confidence': correct_value
                        })
                    elif feedback_type == 'missed':
                        # Tespit edilememiş
                        self.training_data['object_detection'].append({
                            'image': image,
                            'labels': [category],
                            'confidence': 0.0
                        })
                    elif feedback_type == 'false_positive':
                        # Yanlış tespit
                        self.training_data['object_detection'].append({
                            'image': image,
                            'labels': [],
                            'confidence': 0.0
                        })
                        
            # Yeterli veri varsa eğitimi başlat
            self._check_and_train()
            
        except Exception as e:
            logger.error(f"Eğitim verisi dönüştürme hatası: {str(e)}")
            
    def _update_performance_metrics(self, feedback: Dict):
        """Model performans metriklerini güncelle"""
        try:
            predictions = feedback['predictions']
            feedback_data = feedback['feedback']
            
            # CLIP performansı
            if predictions['clip'] and 'general_rating' in feedback_data:
                clip_pred = predictions['clip']['content_type']
                clip_acc = 1.0 if clip_pred == feedback_data['general_rating'] else 0.0
                self._update_model_metrics('clip', clip_acc)
                
            # Kategori bazlı performans
            for category in self.content_categories.keys():
                if f'{category}_feedback' in feedback_data:
                    feedback_type = feedback_data[f'{category}_feedback']
                    correct_value = feedback_data.get(f'{category}_correct_value')
                    
                    if feedback_type == 'correct' and correct_value is not None:
                        # Doğru tespit
                        self._update_model_metrics('object_detection', 1.0)
                    elif feedback_type == 'missed':
                        # Tespit edilememiş
                        self._update_model_metrics('object_detection', 0.0)
                    elif feedback_type == 'false_positive':
                        # Yanlış tespit
                        self._update_model_metrics('object_detection', 0.0)
                        
        except Exception as e:
            logger.error(f"Performans metrik güncelleme hatası: {str(e)}")
            
    def _update_model_metrics(self, model_name: str, accuracy: float):
        """Belirli bir modelin metriklerini güncelle"""
        metrics = self.performance_metrics[model_name]
        metrics['accuracy'] = (metrics['accuracy'] * metrics['samples'] + accuracy) / (metrics['samples'] + 1)
        metrics['samples'] += 1
        
    def _optimize_weights(self):
        """Model performanslarına göre ağırlıkları optimize et"""
        try:
            total_accuracy = sum(m['accuracy'] for m in self.performance_metrics.values())
            
            if total_accuracy == 0:
                return
                
            # Performansa dayalı yeni ağırlıklar
            new_weights = {
                model: metrics['accuracy'] / total_accuracy
                for model, metrics in self.performance_metrics.items()
            }
            
            # Ağırlıkları güncelle
            self.update_weights(new_weights)
            
        except Exception as e:
            logger.error(f"Ağırlık optimizasyon hatası: {str(e)}")
            
    def get_performance_metrics(self) -> Dict:
        """Güncel performans metriklerini döndür"""
        return {
            'models': self.performance_metrics,
            'total_samples': sum(m['samples'] for m in self.performance_metrics.values()),
            'average_accuracy': sum(m['accuracy'] for m in self.performance_metrics.values()) / len(self.performance_metrics)
        }

    def get_cache_stats(self) -> Dict:
        """Önbellek istatistiklerini döndür"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self._max_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }

    def add_training_data(self, image: np.ndarray, labels: Dict[str, Union[str, List[str]]]):
        """Eğitim verisi ekle"""
        try:
            # CLIP verisi
            if 'content_type' in labels:
                self.training_data['clip'].append({
                    'image': image,
                    'label': labels['content_type']
                })
                
            # YOLO verisi
            if 'objects' in labels:
                self.training_data['object_detection'].append({
                    'image': image,
                    'labels': labels['objects']
                })
                
            # Scene verisi
            if 'scene' in labels:
                self.training_data['scene'].append({
                    'image': image,
                    'label': labels['scene']
                })
                
            # ResNet verisi
            if 'objects' in labels:
                self.training_data['features'].append({
                    'image': image,
                    'labels': labels['objects']
                })
                
            # Yeterli veri varsa eğitimi başlat
            self._check_and_train()
            
        except Exception as e:
            logger.error(f"Eğitim verisi ekleme hatası: {str(e)}")
            
    def _check_and_train(self):
        """Yeterli veri varsa eğitimi başlat"""
        for model_name, data in self.training_data.items():
            if len(data) >= self.training_config['min_samples_for_training']:
                self.retrain_model(model_name)
                
    def retrain_model(self, model_name: str):
        """Belirli bir modeli yeniden eğit"""
        try:
            if model_name not in self.training_data:
                raise ValueError(f"Geçersiz model adı: {model_name}")
                
            training_data = self.training_data[model_name]
            if len(training_data) < self.training_config['min_samples_for_training']:
                logger.warning(f"Yeterli eğitim verisi yok: {model_name}")
                return
                
            logger.info(f"{model_name} modeli eğitimi başlıyor...")
            
            if model_name == 'clip':
                self._retrain_clip(training_data)
            elif model_name == 'object_detection':
                self._retrain_yolo(training_data)
            elif model_name == 'scene':
                self._retrain_scene(training_data)
            elif model_name == 'features':
                self._retrain_resnet(training_data)
                
            # Eğitim verilerini temizle
            self.training_data[model_name] = []
            
            logger.info(f"{model_name} modeli eğitimi tamamlandı")
            
        except Exception as e:
            logger.error(f"Model eğitimi hatası ({model_name}): {str(e)}")
            
    def _retrain_clip(self, training_data: List[Dict]):
        """CLIP modelini yeniden eğit"""
        try:
            # Eğitim verilerini hazırla
            images = []
            texts = []
            for data in training_data:
                images.append(data['image'])
                texts.append(data['label'])
                
            # Modeli eğit
            self.clip_model.train()
            optimizer = torch.optim.Adam(self.clip_model.parameters(), 
                                      lr=self.training_config['learning_rate'])
            
            for epoch in range(self.training_config['epochs']):
                total_loss = 0
                for i in range(0, len(images), self.training_config['batch_size']):
                    batch_images = images[i:i + self.training_config['batch_size']]
                    batch_texts = texts[i:i + self.training_config['batch_size']]
                    
                    # Batch'i hazırla
                    image_inputs = torch.stack([self.preprocess(Image.fromarray(img)) 
                                             for img in batch_images]).to(self.device)
                    text_inputs = clip.tokenize(batch_texts).to(self.device)
                    
                    # Forward pass
                    image_features = self.clip_model.encode_image(image_inputs)
                    text_features = self.clip_model.encode_text(text_inputs)
                    
                    # Loss hesapla
                    loss = self._compute_clip_loss(image_features, text_features)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                logger.info(f"CLIP Epoch {epoch+1}/{self.training_config['epochs']}, "
                          f"Loss: {total_loss/len(images):.4f}")
                
            self.clip_model.eval()
            
        except Exception as e:
            logger.error(f"CLIP eğitim hatası: {str(e)}")
            
    def _retrain_yolo(self, training_data: List[Dict]):
        """YOLO modelini yeniden eğit"""
        try:
            # Eğitim verilerini hazırla
            dataset = self._prepare_yolo_dataset(training_data)
            
            # Modeli eğit
            self.yolo_model.train(
                data=dataset,
                epochs=self.training_config['epochs'],
                batch=self.training_config['batch_size'],
                imgsz=640
            )
            
        except Exception as e:
            logger.error(f"YOLO eğitim hatası: {str(e)}")
            
    def _retrain_scene(self, training_data: List[Dict]):
        """Scene modelini yeniden eğit"""
        try:
            # Eğitim verilerini hazırla
            images = []
            labels = []
            for data in training_data:
                images.append(data['image'])
                labels.append(self.scene_categories.index(data['label']))
                
            # Modeli eğit
            self.scene_model.train()
            optimizer = torch.optim.Adam(self.scene_model.parameters(), 
                                      lr=self.training_config['learning_rate'])
            criterion = torch.nn.CrossEntropyLoss()
            
            for epoch in range(self.training_config['epochs']):
                total_loss = 0
                for i in range(0, len(images), self.training_config['batch_size']):
                    batch_images = images[i:i + self.training_config['batch_size']]
                    batch_labels = labels[i:i + self.training_config['batch_size']]
                    
                    # Batch'i hazırla
                    image_inputs = torch.stack([self.preprocess(Image.fromarray(img)) 
                                             for img in batch_images]).to(self.device)
                    label_inputs = torch.tensor(batch_labels).to(self.device)
                    
                    # Forward pass
                    outputs = self.scene_model(image_inputs)
                    loss = criterion(outputs, label_inputs)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                logger.info(f"Scene Epoch {epoch+1}/{self.training_config['epochs']}, "
                          f"Loss: {total_loss/len(images):.4f}")
                
            self.scene_model.eval()
            
        except Exception as e:
            logger.error(f"Scene eğitim hatası: {str(e)}")
            
    def _retrain_resnet(self, training_data: List[Dict]):
        """ResNet modelini yeniden eğit"""
        try:
            # Eğitim verilerini hazırla
            images = []
            labels = []
            for data in training_data:
                images.append(data['image'])
                # Çoklu etiket için one-hot encoding
                label_vector = torch.zeros(len(self.resnet_classes))
                for obj in data['labels']:
                    if obj in self.resnet_classes:
                        label_vector[self.resnet_classes.index(obj)] = 1
                labels.append(label_vector)
                
            # Modeli eğit
            self.resnet_model.train()
            optimizer = torch.optim.Adam(self.resnet_model.parameters(), 
                                      lr=self.training_config['learning_rate'])
            criterion = torch.nn.BCEWithLogitsLoss()
            
            for epoch in range(self.training_config['epochs']):
                total_loss = 0
                for i in range(0, len(images), self.training_config['batch_size']):
                    batch_images = images[i:i + self.training_config['batch_size']]
                    batch_labels = labels[i:i + self.training_config['batch_size']]
                    
                    # Batch'i hazırla
                    image_inputs = torch.stack([self.preprocess(Image.fromarray(img)) 
                                             for img in batch_images]).to(self.device)
                    label_inputs = torch.stack(batch_labels).to(self.device)
                    
                    # Forward pass
                    outputs = self.resnet_model(image_inputs)
                    loss = criterion(outputs, label_inputs)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                logger.info(f"ResNet Epoch {epoch+1}/{self.training_config['epochs']}, "
                          f"Loss: {total_loss/len(images):.4f}")
                
            self.resnet_model.eval()
            
        except Exception as e:
            logger.error(f"ResNet eğitim hatası: {str(e)}")
            
    def _compute_clip_loss(self, image_features, text_features):
        """CLIP loss hesapla"""
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Similarity matrix
        similarity = torch.matmul(image_features, text_features.T)
        
        # Labels (diagonal elements should be high)
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        # Cross entropy loss
        loss = torch.nn.CrossEntropyLoss()(similarity, labels)
        
        return loss
        
    def _prepare_yolo_dataset(self, training_data: List[Dict]) -> Dict:
        """YOLO eğitim veri setini hazırla"""
        # YOLO formatında veri seti oluştur
        dataset = {
            'train': [],
            'val': []
        }
        
        for data in training_data:
            # Görüntüyü kaydet
            img_path = f"temp/train_{len(dataset['train'])}.jpg"
            cv2.imwrite(img_path, data['image'])
            
            # Etiketleri YOLO formatına dönüştür
            annotations = []
            for obj in data['labels']:
                # Burada nesne konumlarını da eklemek gerekir
                # Şimdilik sadece sınıf bilgisi
                annotations.append({
                    'class': obj,
                    'bbox': [0, 0, 1, 1]  # Varsayılan bbox
                })
                
            dataset['train'].append({
                'image': img_path,
                'annotations': annotations
            })
            
        return dataset
        
    def get_training_stats(self) -> Dict:
        """Eğitim istatistiklerini döndür"""
        return {
            'training_data_sizes': {
                model: len(data) for model, data in self.training_data.items()
            },
            'min_samples_for_training': self.training_config['min_samples_for_training'],
            'training_config': self.training_config
        }

    def _predict_with_resnet(self, image: np.ndarray) -> Dict[str, float]:
        """ResNet modeli ile tahmin yap"""
        try:
            # Görüntüyü hazırla
            image = cv2.resize(image, (224, 224))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Tahmin yap
            predictions = self.resnet_model.predict(image)[0]
            
            return dict(zip(self.categories.keys(), predictions))
            
        except Exception as e:
            logger.error(f"ResNet tahmin hatası: {str(e)}")
            return {cat: 0.0 for cat in self.categories}
            
    def _predict_with_scene(self, image: np.ndarray) -> Dict[str, float]:
        """Scene modeli ile tahmin yap"""
        try:
            # Görüntüyü hazırla
            image = cv2.resize(image, (224, 224))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Tahmin yap
            predictions = self.scene_model.predict(image)[0]
            
            return dict(zip(self.categories.keys(), predictions))
            
        except Exception as e:
            logger.error(f"Scene tahmin hatası: {str(e)}")
            return {cat: 0.0 for cat in self.categories}

    def _calculate_confidence_score(self, predictions: Dict[str, float]) -> float:
        """
        Tahmin dağılımına göre güven skorunu hesapla.
        
        Args:
            predictions: Kategori-olasılık eşleşmelerini içeren sözlük
            
        Returns:
            0-1 arası güven skoru
        """
        if not predictions:
            return 0.0
            
        # En yüksek ve ikinci en yüksek tahminleri bul
        sorted_preds = sorted(predictions.values(), reverse=True)
        max_pred = sorted_preds[0]
        second_pred = sorted_preds[1] if len(sorted_preds) > 1 else 0.0
        
        # Güven skorunu hesapla:
        # 1. En yüksek tahmin değeri (0.7 ağırlık)
        # 2. En yüksek ile ikinci en yüksek arasındaki fark (0.3 ağırlık)
        confidence = 0.7 * max_pred + 0.3 * (max_pred - second_pred)
        
        return min(max(confidence, 0.0), 1.0)  # 0-1 arasına sınırla 

    def _create_resnet_model(self):
        """ResNet tabanlı model oluştur"""
        try:
            # Base modeli yükle
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            
            # Yeni katmanlar ekle
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            outputs = tf.keras.layers.Dense(len(self.categories), activation='softmax')(x)
            
            # Model oluştur
            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            
            # Base model katmanlarını dondur
            for layer in base_model.layers:
                layer.trainable = False
                
            # Modeli derle (weight_decay kaldırıldı)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"ResNet model oluşturma hatası: {str(e)}")
            return None
            
    def _create_scene_model(self):
        """Scene sınıflandırma modeli oluştur"""
        try:
            # Base modeli yükle
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            
            # Yeni katmanlar ekle
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(len(self.categories), activation='softmax')(x)
            
            # Model oluştur
            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            
            # Base model katmanlarını dondur
            for layer in base_model.layers:
                layer.trainable = False
                
            # Modeli derle (weight_decay kaldırıldı)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Scene model oluşturma hatası: {str(e)}")
            return None 