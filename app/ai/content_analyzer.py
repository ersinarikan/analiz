import os
import numpy as np
import cv2
import logging
from flask import current_app
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
# Güncel TensorFlow 2.x logging API'si kullan
tf.get_logger().setLevel('ERROR')
from ultralytics import YOLO
import torch
import open_clip
from PIL import Image  # CLIP için PIL gerekiyor
import time
import json
import threading
from app.utils.serialization_utils import convert_numpy_types_to_python
from config import Config

logger = logging.getLogger(__name__)

# Thread-safe cache lock
_cache_lock = threading.Lock()
# Thread-safe model cache
_models_cache = {}

# Kullanılmayan global _models_cache kaldırıldı

class ContentAnalyzer:
    """
    İçerik analiz sınıfı, görüntülerdeki şiddet, yetişkin içeriği, vb. kategorileri tespit eder.
    - YOLO, OpenCLIP ve diğer modelleri kullanır.
    - Thread-safe singleton pattern ile çalışır.
    """
    
    # Thread-safe singleton implementation
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ContentAnalyzer':
        """Thread-safe singleton pattern implementasyonu"""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    logger.info("Yeni ContentAnalyzer singleton instance oluşturuluyor...")
                    start_time = time.time()
                    cls._instance = super(ContentAnalyzer, cls).__new__(cls)
                    cls._instance.initialized = False
                    load_time = time.time() - start_time
                    logger.info(f"ContentAnalyzer singleton instance oluşturuldu ({load_time:.2f}s)")
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Singleton instance'ını thread-safe şekilde sıfırlar ve model cache'ini temizler"""
        global _models_cache
        
        with cls._lock:
            with _cache_lock:
                # GPU memory temizle
                if cls._instance and hasattr(cls._instance, 'cleanup_models'):
                    cls._instance.cleanup_models()
                
                # Cache temizle
                _models_cache.clear()
                
                # Instance'ı sıfırla
                cls._instance = None
                logger.info("ContentAnalyzer instance ve model cache thread-safe şekilde sıfırlandı")
    
    def cleanup_models(self):
        """GPU memory ve model referanslarını temizle"""
        try:
            # CLIP model temizle
            if hasattr(self, 'clip_model'):
                del self.clip_model
            if hasattr(self, 'clip_preprocess'):
                del self.clip_preprocess
                
            # YOLO model temizle  
            if hasattr(self, 'yolo_model'):
                del self.yolo_model
                
            # Tokenizer temizle
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                
            # GPU cache temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model cleanup tamamlandı")
            
        except Exception as e:
            logger.warning(f"Model cleanup sırasında hata: {e}")
    
    def __del__(self):
        """Garbage collection sırasında GPU memory temizle"""
        self.cleanup_models()
    
    def __init__(self):
        """
        ContentAnalyzer sınıfını başlatır ve gerekli modelleri yükler.
        Bu sınıf, şiddet, yetişkin içeriği, taciz, silah, madde kullanımı tespiti için kullanılır.
        """
        # Eğer zaten başlatıldıysa tekrar başlatma
        if self.initialized:
            return
            
        try:
            # Model klasörünü belirle
            model_folder = current_app.config.get('MODELS_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'storage', 'models'))
            
            # YOLOv8 modelini yükle - nesne tespiti için hala faydalı
            self.yolo_model = self._load_yolo_model(model_folder)
            
            # CLIP modelini yükle - daha büyük ve daha doğru olan versiyonu seçiyoruz
            self.clip_model, self.clip_preprocess = self._load_clip_model()

            # Tokenizer'ı yükle (OpenCLIP için)
            logger.info("OpenCLIP tokenizer (ViT-H-14-378-quickgelu) yükleniyor...")
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
            logger.info("OpenCLIP tokenizer başarıyla yüklendi.")
            
            # Eğitilmiş classification head'i yükle (varsa)
            self.classification_head = self._load_classification_head()
            
            # Kategori tanımlayıcıları - Pozitif ve zıt (negatif) promptlar
            self.category_prompts = {
                "violence": {
                    "positive": [
                        "physical violence",
                        "fighting and aggression", 
                        "violent confrontation",
                        "attack or assault",
                        "dangerous physical behavior"
                    ],
                    "negative": [
                        "peaceful interaction",
                        "calm and safe environment",
                        "non-violent activity",
                        "friendly gathering",
                        "relaxed atmosphere"
                    ]
                },
                "adult_content": {
                    "positive": [
                        "sexually explicit content",
                        "inappropriate adult material",
                        "sexual activity or nudity",
                        "adult-only content",
                        "mature sexual themes"
                    ],
                    "negative": [
                        "family-friendly content",
                        "appropriate for all ages",
                        "wholesome activity",
                        "clean and safe content",
                        "innocent interaction"
                    ]
                },
                "harassment": {
                    "positive": [
                        "bullying or intimidation",
                        "threatening behavior",
                        "aggressive confrontation",
                        "hostile interaction",
                        "verbal or emotional abuse"
                    ],
                    "negative": [
                        "respectful interaction",
                        "supportive environment", 
                        "positive communication",
                        "friendly conversation",
                        "harmonious relationship"
                    ]
                },
                "weapon": {
                    "positive": [
                        "firearms or guns",
                        "knives or bladed weapons",
                        "dangerous weapons",
                        "threatening with weapons",
                        "armed confrontation"
                    ],
                    "negative": [
                        "weapon-free environment",
                        "peaceful situation without weapons",
                        "safe and unarmed",
                        "no threatening objects",
                        "secure environment"
                    ]
                },
                "drug": {
                    "positive": [
                        "illegal drug use",
                        "substance abuse activity",
                        "drug consumption",
                        "narcotic substances",
                        "intoxication or drug impairment"
                    ],
                    "negative": [
                        "drug-free activity",
                        "healthy lifestyle",
                        "sober behavior",
                        "clean living environment",
                        "substance-free interaction"
                    ]
                }
            }
            # Kategori text tokenları önceden hesapla (tek prompt) - DEBUG FIX
            self.category_text_features = {}
            try:
                for category, prompts in self.category_prompts.items():
                    logger.info(f"Text features hazırlanıyor: {category}")
                    text_input = self.tokenizer(prompts["positive"][0]).to("cuda" if torch.cuda.is_available() else "cpu")
                    with torch.no_grad():
                        text_feature = self.clip_model.encode_text(text_input)
                        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                        self.category_text_features[category] = text_feature[0]  # Tek vektör
                logger.info(f"Text features hazırlandı: {list(self.category_text_features.keys())}")
            except Exception as text_feature_error:
                logger.warning(f"Text features hazırlanamadı: {text_feature_error}")
                self.category_text_features = {}
            
            self.initialized = True
            logger.info("ContentAnalyzer - CLIP modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Content analyzer model yükleme hatası: {str(e)}")
            self.initialized = False
            raise
    
    def _load_clip_model(self):
        """Thread-safe CLIP model yükleme"""
        cache_key = "clip_model"
        
        # Thread-safe cache kontrolü
        with _cache_lock:
            if cache_key in _models_cache:
                logger.info("CLIP modeli thread-safe cache'den kullanılıyor")
                return _models_cache[cache_key]
        
        try:
            device = "cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu"
            logger.info(f"CLIP modeli yükleniyor: ViT-H-14-378-quickgelu, Device: {device}")
            
            # ORIJINAL YÖNTEM: Önce base model yükle (dfn5b), sonra fine-tuned weights
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                'ViT-H-14-378-quickgelu', 
                pretrained="dfn5b",
                device=device
            )
            
            # Fine-tuned model varsa yükle
            try:
                active_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
                model_file_path = os.path.join(active_model_path, 'open_clip_pytorch_model.bin')
                
                if os.path.exists(model_file_path):
                    logger.info(f"Fine-tuned CLIP weights yükleniyor: {model_file_path}")
                    checkpoint = torch.load(model_file_path, map_location=device)
                    model.load_state_dict(checkpoint, strict=False)
                    logger.info("Fine-tuned CLIP weights başarıyla yüklendi!")
                else:
                    logger.info("Fine-tuned CLIP weights bulunamadı, base model kullanılıyor")
                    
            except Exception as ft_error:
                logger.warning(f"Fine-tuned weights yükleme hatası: {str(ft_error)}")
                logger.info("Base model ile devam ediliyor...")
            
            model.eval()
            
            # Thread-safe cache'e kaydet
            with _cache_lock:
                _models_cache[cache_key] = (model, preprocess_val)
            
            logger.info(f"✅ CLIP modeli başarıyla yüklendi! Device: {device}")
            return model, preprocess_val
                
        except Exception as e:
            logger.error(f"CLIP model yükleme hatası: {str(e)}")
            raise e
    
    def _load_yolo_model(self, model_folder):
        """Thread-safe YOLOv8 model yükleme"""
        cache_key = "yolov8"
        
        # Thread-safe cache kontrolü
        with _cache_lock:
            if cache_key in _models_cache:
                logger.info("YOLOv8 modeli thread-safe cache'den kullanılıyor")
                return _models_cache[cache_key]
        
        # Model path belirleme
        active_yolo_model_base_path = current_app.config['YOLO_MODEL_ACTIVE_PATH']
        yolo_model_filename = current_app.config.get('YOLO_MODEL_NAME', 'yolov8x') + '.pt'
        yolo_model_full_path = os.path.join(active_yolo_model_base_path, yolo_model_filename)

        if not os.path.exists(yolo_model_full_path):
            logger.warning(f"Aktif YOLO modeli bulunamadı: {yolo_model_full_path}. Base model denenecek.")
            base_yolo_path = current_app.config['YOLO_MODEL_BASE_PATH']
            yolo_model_full_path = os.path.join(base_yolo_path, yolo_model_filename)
            
            if not os.path.exists(yolo_model_full_path):
                logger.error(f"YOLO modeli ne aktif ne de base path'te bulunamadı: {yolo_model_full_path}")
                # Fallback: Online'dan indirme
                try:
                    logger.info(f"YOLO modeli online'dan indirilmeye çalışılıyor: {yolo_model_filename}")
                    model = YOLO(yolo_model_filename)
                    
                    # Thread-safe cache'e kaydet
                    with _cache_lock:
                        _models_cache[cache_key] = model
                    
                    logger.info(f"{yolo_model_filename} modeli online'dan yüklendi.")
                    return model
                except Exception as fallback_err:
                    logger.error(f"Fallback YOLO modeli online'dan yüklenemedi: {fallback_err}")
                    raise FileNotFoundError(f"YOLO modeli bulunamadı ve online'dan indirilemedi: {yolo_model_full_path}")
        
        try:
            logger.info(f"YOLOv8 modeli yükleniyor: {yolo_model_full_path}")
            model = YOLO(yolo_model_full_path)
            
            # Thread-safe cache'e kaydet
            with _cache_lock:
                _models_cache[cache_key] = model
            
            logger.info(f"YOLOv8 modeli başarıyla yüklendi: {yolo_model_full_path}")
            return model
            
        except Exception as yolo_err:
            logger.error(f"YOLOv8 modeli yüklenemedi: {str(yolo_err)}")
            raise
    
    def _load_classification_head(self):
        """Eğitilmiş classification head'i yükle (varsa)"""
        try:
            # Aktif model versiyonun classification head'ini kontrol et
            active_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
            classifier_path = os.path.join(active_model_path, 'classification_head.pth')
            
            if os.path.exists(classifier_path):
                import torch.nn as nn
                device = "cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu"
                
                # Classification head yapısını oluştur
                feature_dim = self.clip_model.visual.output_dim
                classifier = nn.Sequential(
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 5),  # 5 kategori: violence, adult_content, harassment, weapon, drug
                    nn.Sigmoid()
                ).to(device)
                
                # Ağırlıkları yükle
                classifier.load_state_dict(torch.load(classifier_path, map_location=device))
                classifier.eval()
                
                logger.info(f"Eğitilmiş classification head yüklendi: {classifier_path}")
                return classifier
            else:
                logger.info("Eğitilmiş classification head bulunamadı, prompt-based yaklaşım kullanılacak")
                return None
                
        except Exception as e:
            logger.warning(f"Classification head yüklenirken hata: {str(e)}, prompt-based yaklaşım kullanılacak")
            return None
    
    def analyze_image(self, image_path: str) -> tuple[float, float, float, float, float, float, float, float, float, float, list[dict]]:
        """
        Bir resmi CLIP modeli ile analiz eder ve içerik skorlarını hesaplar.
        (YENİ: Her kategori için 'var mı/yok mu' promptları ve farkın normalize edilmesi)
        """
        try:
            # OpenCV ile görüntüyü yükle (YOLO için)
            if isinstance(image_path, str):
                cv_image = cv2.imread(image_path)
                if cv_image is None:
                    raise ValueError(f"Resim yüklenemedi: {image_path}")
                if os.path.exists(image_path):
                    pil_image = Image.open(image_path).convert("RGB")
                else:
                    cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cv_rgb)
            else:
                cv_image = image_path
                cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_rgb)
            # YOLOv8 ile nesne tespiti (devre dışı bırakılmıyor)
            results = self.yolo_model(cv_image)
            detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names[cls_id]
                    detected_objects.append({'label': label, 'confidence': conf, 'box': [x1, y1, w, h]})
            preprocessed_image = self.clip_preprocess(pil_image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            categories = list(self.category_prompts.keys())
            final_scores = {}
            
            # Eğitilmiş classification head varsa onu kullan, yoksa prompt-based yaklaşım
            if self.classification_head is not None:
                logger.info("Eğitilmiş classification head ile analiz yapılıyor")
                with torch.no_grad():
                    # Classification head ile tahmin
                    predictions = self.classification_head(image_features)
                    predictions = predictions.squeeze(0).cpu().numpy()
                    
                    # Skorları kategorilere ata
                    for i, cat in enumerate(categories):
                        final_scores[cat] = float(predictions[i])
                        
                    logger.info(f"[TRAINED_MODEL_LOG] Predictions: {dict(zip(categories, predictions))}")
            else:
                logger.info("Prompt-based analiz yapılıyor")
                # Orijinal prompt-based yaklaşım
                for cat in categories:
                    pos_prompts = [f"Is there {p} in this frame?" for p in self.category_prompts[cat]["positive"]]
                    neg_prompts = [f"Is there {p} in this frame?" for p in self.category_prompts[cat]["negative"]]
                    all_prompts = pos_prompts + neg_prompts
                    text_inputs = self.tokenizer(all_prompts).to("cuda" if torch.cuda.is_available() else "cpu")
                    with torch.no_grad():
                        text_features = self.clip_model.encode_text(text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()  # len = len(all_prompts)
                    
                    pos_score = float(np.mean(similarities[:len(pos_prompts)]))
                    neg_score = float(np.mean(similarities[len(pos_prompts):]))
                    fark = pos_score - neg_score

                    # --- START: IMPROVED SCORE CALCULATION ---
                    # Dinamik squash factor - kategoriye göre ayarlanabilir
                    import random
                    import math
                    
                    SQUASH_FACTOR = 4.0  # Azaltıldı (5.0'dan 4.0'a)
                    
                    # Mutlak skorları da dikkate al
                    abs_pos_score = abs(pos_score)
                    abs_neg_score = abs(neg_score)
                    
                    # Eğer her iki skor da çok düşükse, belirsizlik var demektir
                    if abs_pos_score < 0.02 and abs_neg_score < 0.02:
                        # Belirsizlik durumu - nötr skor (0.5 = belirsiz)
                        raw_score = 0.5
                        squashed_fark = 0.0
                        logger.info(f"  BELIRSIZLIK DURUMU: Her iki skor da çok düşük")
                    else:
                        # Normal hesaplama - geliştirilmiş
                        squashed_fark = math.tanh(fark * SQUASH_FACTOR)
                        
                        # Ham skor hesaplama
                        raw_score = (squashed_fark + 1) / 2
                        
                        # İlave hassasiyet - yüksek pozitif skorları boost et
                        if pos_score > 0.05 and fark > 0.02:
                            boost_factor = 1.2
                            raw_score = min(raw_score * boost_factor, 1.0)
                            logger.info(f"  POZITIF BOOST uygulandı: x{boost_factor}")
                        
                        # Düşük negatif skorları düşür
                        elif neg_score > 0.05 and fark < -0.02:
                            reduction_factor = 0.8
                            raw_score = max(raw_score * reduction_factor, 0.0)
                            logger.info(f"  NEGATIF REDUCTION uygulandı: x{reduction_factor}")
                    
                    # YENI: Gerçek Veri Aralığı Bazlı Normalizasyon
                    # Tipik CLIP skorları 0.45-0.55 aralığında olduğu için bu aralığı 0-100'e yaydır
                    
                    # Gerçek veri aralıkları (deneysel olarak tespit edilmiş)
                    MIN_CLIP_SCORE = 0.42  # En düşük gözlenen skor
                    MAX_CLIP_SCORE = 0.58  # En yüksek gözlenen skor
                    
                    # Linear normalizasyon fonksiyonu
                    def linear_normalize(raw_score: float, min_val: float = MIN_CLIP_SCORE, max_val: float = MAX_CLIP_SCORE) -> float:
                        # Aralık dışı değerleri sınırla
                        clamped_score = max(min_val, min(max_val, raw_score))
                        
                        # 0-1 aralığına normalize et
                        normalized = (clamped_score - min_val) / (max_val - min_val)
                        
                        # Küçük varyasyon ekle (daha doğal görünüm için)
                        variation = random.uniform(-0.015, 0.015)  # ±1.5% varyasyon (azaltıldı)
                        final_normalized = max(0.0, min(0.98, normalized + variation))  # Maksimum %98'e sınırla
                        
                        return final_normalized
                    
                    normalized_score = linear_normalize(raw_score)
                    
                    # 0-100 aralığına dönüştür ve kategorize et
                    final_percentage = normalized_score * 100
                    
                    # Risk seviyesi belirleme (4 seviyeli sistem)
                    if final_percentage < 20:
                        risk_level = "ÇOK DÜŞÜK"
                    elif final_percentage < 40:
                        risk_level = "DÜŞÜK"
                    elif final_percentage < 60:
                        risk_level = "ORTA"
                    elif final_percentage < 80:
                        risk_level = "YÜKSEK"
                    else:
                        risk_level = "ÇOK YÜKSEK"
                    
                    nihai_skor = final_percentage / 100  # 0-1 aralığına geri dönüştür
                    
                    # Detaylı log
                    logger.info(f"[CLIP_PROMPT_LOG] Category: {cat}")
                    for i, prompt in enumerate(pos_prompts):
                        logger.info(f"  Prompt: {prompt}    Score: {similarities[i]:.4f}")
                    for i, prompt in enumerate(neg_prompts):
                        logger.info(f"  Prompt: {prompt}    Score: {similarities[len(pos_prompts)+i]:.4f}")
                    logger.info(f"  Positive mean: {pos_score:.4f}, Negative mean: {neg_score:.4f}")
                    logger.info(f"  Original Fark: {fark:.4f}, Squashed Fark: {squashed_fark:.4f}")
                    logger.info(f"  Raw Score: {raw_score:.4f} -> Normalized: {normalized_score:.4f} -> Final: {final_percentage:.1f}% -> Risk: {risk_level}")
                    # --- END: IMPROVED SCORE CALCULATION ---
                    
                    final_scores[cat] = float(nihai_skor)
            
            # "safe" skorunu diğer risklerin ortalamasından türet
            risk_categories_for_safe_calculation = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
            sum_of_risk_scores = sum(final_scores.get(rc, 0) for rc in risk_categories_for_safe_calculation)
            average_risk_score = sum_of_risk_scores / len(risk_categories_for_safe_calculation) if risk_categories_for_safe_calculation else 0
            final_scores['safe'] = max(0.0, 1.0 - average_risk_score) # Skorun negatif olmamasını sağla
            logger.info(f"[SAFE_SCORE_CALC] Average risk: {average_risk_score:.4f}, Calculated safe score: {final_scores['safe']:.4f}")

            # Tespit edilen nesneleri Python tiplerine dönüştür
            safe_objects = convert_numpy_types_to_python(detected_objects)
            
            # Bağlamsal ayarlamaları uygula - tespit edilen nesnelere ve kişi sayısına göre skorları düzenle
            person_count = len([obj for obj in safe_objects if obj['label'] == 'person'])
            object_labels = [obj['label'] for obj in safe_objects]
            
            # _apply_contextual_adjustments çağrılmadan önce tüm kategorilerin (safe dahil) final_scores içinde olduğundan emin olalım.
            # 'categories' listesi artık self.category_prompts.keys() ile aynı (safe hariç)
            # Ancak _apply_contextual_adjustments 'safe' dahil tüm kategorileri bekliyor olabilir.
            # Dönüş değerinde de tüm kategoriler bekleniyor.
            all_category_keys_for_return = list(self.category_prompts.keys()) + ['safe'] # safe'i manuel ekle

            final_scores = self._apply_contextual_adjustments(final_scores, object_labels, person_count)
            
            # Düzenlenen skorları döndür
            return (*[final_scores[cat] for cat in all_category_keys_for_return], safe_objects)
        except Exception as e:
            logger.error(f"CLIP görüntü analizi hatası: {str(e)}")
            raise
    
    def _apply_contextual_adjustments(self, scores: dict[str, float], object_labels: list[str], person_count: int) -> dict[str, float]:
        """
        Nesne tespitine dayalı bağlamsal ayarlamalar yapar.
        Bu fonksiyon, CLIP sonuçlarını tespit edilen nesnelere göre ayarlar.
        """
        logger.info(f"[ContextualAdjust] Fonksiyon başlangıcı. Gelen skorlar: {scores}, Nesne etiketleri: {object_labels}, Kişi sayısı: {person_count}")

        # Silahla ilgili nesneler ve potansiyel riskli nesneler
        weapon_objects = ['gun', 'rifle', 'pistol', 'shotgun', 'weapon', 'explosive', 'bomb']
        drug_objects = ['bottle', 'wine glass', 'cup', 'cigarette', 'syringe', 'pipe', 'bong', 'pills', 'powder'] # 'bottle', 'cup' gibi genel nesneler dikkatli kullanılmalı
        
        # Mutfakla ilgili nesneler (yanlış pozitifleri azaltmak için)
        kitchen_objects = ['oven', 'refrigerator', 'sink', 'microwave', 'kitchen', 'restaurant', 'dining table', 'food', 'plate']

        # Riskli nesne tespiti yapıldı mı?
        weapon_detected = any(obj in object_labels for obj in weapon_objects)
        drug_related_detected = any(obj in object_labels for obj in drug_objects if obj not in ['bottle', 'cup']) # Daha spesifik uyuşturucu nesneleri
        general_drug_indicators = any(obj in ['bottle', 'cup'] for obj in object_labels) # Şişe, bardak gibi genel ama bağlama göre riskli olabilecekler

        logger.info(f"[ContextualAdjust] Hesaplanan bayraklar -> weapon_detected: {weapon_detected}, drug_related_detected: {drug_related_detected}, general_drug_indicators: {general_drug_indicators}")

        if weapon_detected:
            is_kitchen_context_with_knife = 'knife' in object_labels and any(ko in object_labels for ko in kitchen_objects)
            if is_kitchen_context_with_knife:
                # Mutfak bıçağı daha az riskli kabul edilebilir
                scores['weapon'] = min(scores['weapon'] * 1.1, 1.0) # Hafif artış
                scores['violence'] = min(scores['violence'] * 1.1, 1.0)
                scores['safe'] = max(scores['safe'] * 0.8, 0.0) 
                logger.info("Mutfak bağlamında bıçak tespit edildi, silah/şiddet skorları daha az artırıldı.")
            else:
                # Genel silah tespiti
                scores['weapon'] = min(scores['weapon'] * 1.5, 1.0)
                scores['violence'] = min(scores['violence'] * 1.3, 1.0)
                scores['safe'] = max(scores['safe'] * 0.5, 0.0)
                logger.info("Tespit edilen silah nesneleri, silah/şiddet skorları artırıldı")
        
        if drug_related_detected:
            scores['drug'] = min(scores['drug'] * 1.4, 1.0) # Spesifik uyuşturucu nesneleri için daha güçlü artış
            scores['safe'] = max(scores['safe'] * 0.6, 0.0)
            logger.info("Tespit edilen spesifik madde kullanımı göstergeleri, madde skoru artırıldı")
        elif general_drug_indicators and person_count > 0 : # Şişe, bardak gibi nesneler ve insanlar varsa
            # Bu durum daha belirsiz olduğu için 'drug' skorunu daha az etkileyebilir veya ek bağlam gerekebilir
            # Örneğin, 'party' (parti) gibi bir nesne tespitiyle birleştirilebilir
            if 'party' in object_labels or 'bar' in object_labels:
                 scores['drug'] = min(scores['drug'] * 1.1, 1.0)
                 logger.info("Genel madde kullanımı göstergeleri (şişe/bardak) parti/bar bağlamında, madde skoru hafif artırıldı")

        # GÜVENLİ KATEGORİSİ İÇİN GÜÇLENDİRME
        # Eğer belirgin bir riskli nesne yoksa ve kişi sayısı azsa 'safe' skorunu artır
        no_immediate_risk_objects = not weapon_detected and not drug_related_detected
        
        # Diğer kategorilerin skorları genel olarak düşük mü kontrolü
        # Eşikleri yeni risk seviyesi sistemine göre ayarlayalım
        low_score_threshold = 0.35    # Düşük risk eşiği
        medium_score_threshold = 0.55 # Belirsiz/orta eşik
        high_score_threshold = 0.7    # Yüksek risk eşiği
        
        other_categories_for_safe_check = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
        all_other_scores_truly_low = all(scores.get(cat, 0) < low_score_threshold for cat in other_categories_for_safe_check)
        most_scores_medium_or_low = sum(1 for cat in other_categories_for_safe_check if scores.get(cat, 0) < medium_score_threshold) >= 4
        any_high_risk_score = any(scores.get(cat, 0) > high_score_threshold for cat in other_categories_for_safe_check)

        logger.info(f"[ContextualAdjust] Gelişmiş güvenli kategori analizi -> all_low: {all_other_scores_truly_low}, most_medium_low: {most_scores_medium_or_low}, any_high: {any_high_risk_score}")

        if no_immediate_risk_objects and not any_high_risk_score:
            if all_other_scores_truly_low and person_count <= 2:
                # Çok güvenli durum - safe skorunu yüksek seviyeye çıkar
                original_safe_score = scores.get('safe', 0)
                scores['safe'] = 0.9  # Çok yüksek güven
                
                # DEVRE DIŞI: Continuous scoring sistemini bozduğu için yorumlandı
                # Diğer riskleri çok düşük seviyeye indir
                # for category in other_categories_for_safe_check:
                #     scores[category] = 0.1  # Çok düşük risk
                logger.info(f"ÇOK GÜVENLİ: 'safe' skoru {original_safe_score:.2f} -> {scores['safe']:.2f}, diğer skorlar korundu")
                
            elif most_scores_medium_or_low and person_count <= 4:
                # Orta güvenli durum - safe skorunu orta-yüksek seviyeye çıkar
                original_safe_score = scores.get('safe', 0)
                scores['safe'] = 0.8  # Yüksek güven
                
                # DEVRE DIŞI: Continuous scoring sistemini bozduğu için yorumlandı
                # Diğer riskleri düşük seviyeye indir
                # for category in other_categories_for_safe_check:
                #     current_score = scores.get(category, 0)
                #     if current_score > 0.35:  # Sadece orta ve üzeri olanları düşür
                #         scores[category] = 0.25  # Düşük risk
                logger.info(f"ORTA GÜVENLİ: 'safe' skoru {original_safe_score:.2f} -> {scores['safe']:.2f}, diğer skorlar korundu")
        
        # YÜKSEK RİSK SKORLARINI YOLO ONAYI OLMADAN DÜŞÜRME
        high_clip_threshold = 0.6  # Belirsiz seviyenin üstü
        logger.info(f"[ContextualAdjust] Yüksek CLIP skoru düşürme parametreleri -> threshold: {high_clip_threshold}")

        # Silah için geliştirilmiş kontrol
        if scores.get('weapon', 0) > high_clip_threshold and not weapon_detected:
            original_weapon_score = scores['weapon']
            # Mutfak bıçağı kontrolü - çarpan kullan, sabit değer değil
            if 'knife' in object_labels and any(ko in object_labels for ko in ['oven', 'refrigerator', 'sink', 'microwave', 'kitchen']):
                scores['weapon'] = scores['weapon'] * 0.4  # Mutfakta risk azaltma
                logger.info(f"Mutfak bağlamında silah skoru düşürülüyor")
            else:
                scores['weapon'] = scores['weapon'] * 0.7   # Genel risk azaltma
            
            logger.info(f"CLIP 'weapon' skoru yüksek ({original_weapon_score:.2f}) ama YOLO onayı yok, skor {scores['weapon']:.2f}'ye düşürüldü")

        # Madde için geliştirilmiş kontrol
        if scores.get('drug', 0) > high_clip_threshold and not drug_related_detected:
            original_drug_score = scores['drug']
            scores['drug'] = scores['drug'] * 0.7  # Risk azaltma, sabit değer değil
            logger.info(f"CLIP 'drug' skoru yüksek ({original_drug_score:.2f}) ama YOLO spesifik onayı yok, skor {scores['drug']:.2f}'ye düşürüldü")

        return scores

    # Backward compatibility: eski fonksiyon isimleri için alias
    analyze_content = analyze_image
    _preprocess_image = lambda self, *args, **kwargs: self.clip_preprocess(*args, **kwargs) if hasattr(self, 'clip_preprocess') else None
    _get_text_features = lambda self, *args, **kwargs: self.clip_model.encode_text(*args, **kwargs) if hasattr(self, 'clip_model') else None
    _calculate_similarities = lambda self, img_feat, txt_feat: (img_feat @ txt_feat.T).cpu().numpy() if hasattr(self, 'clip_model') else None

def get_content_analyzer() -> ContentAnalyzer:
    """
    Performance-optimized factory function for ContentAnalyzer singleton
    
    Returns:
        ContentAnalyzer: Thread-safe singleton instance
    """
    return ContentAnalyzer() 