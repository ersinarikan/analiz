import os
import numpy as np
import cv2
import logging
from flask import current_app
import tensorflow as tf
from ultralytics import YOLO
import torch
import clip  # CLIP modelini import ediyoruz
from PIL import Image  # CLIP için PIL gerekiyor
import shutil
import time
import json
import math
from app.utils.serialization_utils import convert_numpy_types_to_python

logger = logging.getLogger(__name__)

# Singleton pattern için model repository
_models_cache = {}

class ContentAnalyzer:
    """İçerik analiz sınıfı, görüntülerdeki şiddet, yetişkin içeriği, vb. kategorileri tespit eder."""
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementasyonu - tek bir instance oluşturur"""
        if cls._instance is None:
            cls._instance = super(ContentAnalyzer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
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
            
            # Kategori tanımlayıcıları - Pozitif ve zıt (negatif) promptlar
            self.category_prompts = {
                "violence": {
                    "positive": [
                        "violence",
                        "aggression",
                        "brutality"
                    ],
                    "negative": [
                        "gentleness",
                        "calmness",
                        "tenderness"
                    ]
                },
                "adult_content": {
                    "positive": [
                        "adult content",
                        "explicit content",
                        "mature content"
                    ],
                    "negative": [
                        "innocence",
                        "purity",
                        "youthfulness"
                    ]
                },
                "harassment": {
                    "positive": [
                        "harassment",
                        "intimidation",
                        "coercion"
                    ],
                    "negative": [
                        "peaceful",
                        "serenity",
                        "tranquility"
                    ]
                },
                "weapon": {
                    "positive": [
                        "weapon",
                        "violence in the form of weapons",
                        "armament"
                    ],
                    "negative": [
                        "unarmed",
                        "nonviolence",
                        "peaceful situation"
                    ]
                },
                "drug": {
                    "positive": [
                        "drug",
                        "substance abuse",
                        "illegal drugs"
                    ],
                    "negative": [
                        "drug-free",
                        "sobriety",
                        "purity"
                    ]
                },
                "safe": {
                    "positive": [
                        "safe",
                        "security",
                        "secure environment"
                    ],
                    "negative": [
                        "dangerous",
                        "risk",
                        "harm"
                    ]
                }
            }
            # Kategori text tokenları önceden hesapla (tek prompt)
            self.category_text_features = {}
            for category, prompts in self.category_prompts.items():
                text_input = clip.tokenize(prompts["positive"][0]).to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    text_feature = self.clip_model.encode_text(text_input)
                    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                    self.category_text_features[category] = text_feature[0]  # Tek vektör
            
            self.initialized = True
            logger.info("ContentAnalyzer - CLIP modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Content analyzer model yükleme hatası: {str(e)}")
            self.initialized = False
            raise
    
    def _load_clip_model(self):
        """CLIP modelini yükler, önbellek kontrolü yapar"""
        cache_key = "clip_model"
        if cache_key in _models_cache:
            logger.info("CLIP modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
        try:
            logger.info("CLIP modeli yükleniyor (ViT-L/14@336px)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14@336px", device=device)
            logger.info("ViT-L/14@336px CLIP modeli başarıyla yüklendi")
            logger.info(f"CLIP modeli başarıyla yüklendi, çalışma ortamı: {device}")
            _models_cache[cache_key] = (model, preprocess)
            return model, preprocess
        except Exception as e:
            logger.error(f"CLIP modeli yüklenemedi: {str(e)}")
            raise
    
    def _load_yolo_model(self, model_folder):
        """YOLOv8 modelini yükler, cache kontrolü yapar"""
        cache_key = "yolov8"
        
        # Önbellekte varsa direkt döndür
        if cache_key in _models_cache:
            logger.info(f"YOLOv8 modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
            
        # Model klasörünü oluştur
        yolo_model_path = os.path.join(model_folder, 'detection', 'yolov8x.pt')
        if not os.path.exists(os.path.dirname(yolo_model_path)):
            os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
        
        try:
            # Model zaten var mı kontrolü
            if os.path.exists(yolo_model_path):
                logger.info(f"Mevcut YOLOv8x modeli yükleniyor: {yolo_model_path}")
                model = YOLO(yolo_model_path)
            else:
                # Model yoksa indir ve kaydet
                logger.info(f"YOLOv8x modeli indiriliyor: {yolo_model_path}")
                model = YOLO('yolov8x.pt')
                
                # İndirilen modeli kopyala
                if os.path.exists('yolov8x.pt'):
                    shutil.copy('yolov8x.pt', yolo_model_path)
                    logger.info(f"YOLOv8x modeli başarıyla kopyalandı: {yolo_model_path}")
            
            logger.info(f"YOLOv8x modeli başarıyla yüklendi")
            
            # Modeli önbelleğe ekle
            _models_cache[cache_key] = model
            return model
        except Exception as yolo_err:
            logger.error(f"YOLOv8x modeli yüklenemedi: {str(yolo_err)}")
            # Yeniden indirilmeye çalışılır
            try:
                logger.info("YOLOv8x yüklenemedi, fallback olarak YOLOv8n deneniyor...")
                model = YOLO('yolov8n.pt')
                logger.info(f"YOLOv8n modeli online kaynaktan fallback olarak yüklendi")
                _models_cache[cache_key] = model
                return model
            except Exception as e:
                logger.error(f"Fallback YOLOv8n modeli de online kaynaktan yüklenemedi: {str(e)}")
                raise
    
    def analyze_image(self, image_path):
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
            for cat in categories:
                pos_prompts = [f"Is there {p} in this frame?" for p in self.category_prompts[cat]["positive"]]
                neg_prompts = [f"Is there {p} in this frame?" for p in self.category_prompts[cat]["negative"]]
                all_prompts = pos_prompts + neg_prompts
                text_inputs = clip.tokenize(all_prompts).to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()  # len = len(all_prompts)
                pos_score = float(np.mean(similarities[:len(pos_prompts)]))
                neg_score = float(np.mean(similarities[len(pos_prompts):]))
                fark = pos_score - neg_score
                nihai_skor = (fark + 1) / 2  # 0-1 aralığına çek
                # Detaylı log
                logger.info(f"[CLIP_PROMPT_LOG] Category: {cat}")
                for i, prompt in enumerate(pos_prompts):
                    logger.info(f"  Prompt: {prompt}    Score: {similarities[i]:.4f}")
                for i, prompt in enumerate(neg_prompts):
                    logger.info(f"  Prompt: {prompt}    Score: {similarities[len(pos_prompts)+i]:.4f}")
                logger.info(f"  Positive mean: {pos_score:.4f}, Negative mean: {neg_score:.4f}, Final: {nihai_skor:.4f}")
                final_scores[cat] = float(nihai_skor)
            
            # Tespit edilen nesneleri Python tiplerine dönüştür
            safe_objects = convert_numpy_types_to_python(detected_objects)
            
            # Bağlamsal ayarlamaları uygula - tespit edilen nesnelere ve kişi sayısına göre skorları düzenle
            person_count = len([obj for obj in safe_objects if obj['label'] == 'person'])
            object_labels = [obj['label'] for obj in safe_objects]
            final_scores = self._apply_contextual_adjustments(final_scores, object_labels, person_count)
            
            # Düzenlenen skorları döndür
            return (*[final_scores[cat] for cat in categories], safe_objects)
        except Exception as e:
            logger.error(f"CLIP görüntü analizi hatası: {str(e)}")
            raise
    
    def _apply_contextual_adjustments(self, scores, object_labels, person_count):
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

        # Birden fazla kişi varsa kişilerarası etkileşim olasılığı
        if person_count >= 2:
            scores['harassment'] = min(scores['harassment'] * (1.0 + (person_count * 0.08)), 1.0) # Çarpanı biraz düşürdük
            scores['adult_content'] = min(scores['adult_content'] * (1.0 + (person_count * 0.04)), 1.0) # Çarpanı biraz düşürdük
            if person_count > 5: # Çok kalabalık durumlar için
                 scores['safe'] = max(scores['safe'] * 0.9, 0.0) # Kalabalıklar bazen daha az güvenli olabilir
            logger.info(f"{person_count} kişi tespit edildi, kişilerarası etkileşim skorları ayarlandı")
        
        # GÜVENLİ KATEGORİSİ İÇİN GÜÇLENDİRME
        # Eğer belirgin bir riskli nesne yoksa ve kişi sayısı azsa 'safe' skorunu artır
        no_immediate_risk_objects = not weapon_detected and not drug_related_detected
        
        # Diğer kategorilerin (safe hariç) skorları genel olarak düşük mü kontrolü
        # Bu kontrol, _apply_contextual_adjustments fonksiyonuna gelen orijinal 'scores' (scaled_scores) üzerinden yapılmalı
        # Henüz 'safe' skoru bu blok içinde yapay olarak artırılmadı veya diğerleri düşürülmedi.
        other_categories_for_safe_check = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
        all_other_scores_truly_low = all(scores.get(cat, 0) < 0.3 for cat in other_categories_for_safe_check)
        # Eşik değeri (0.3) ayarlanabilir bir parametre olabilir.

        logger.info(f"[ContextualAdjust] Güvenli kategori güçlendirme bayrakları -> no_immediate_risk_objects: {no_immediate_risk_objects}, all_other_scores_truly_low: {all_other_scores_truly_low}")

        if no_immediate_risk_objects and all_other_scores_truly_low and person_count <= 1:
            # Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve ortam sakin görünüyorsa
            original_safe_score = scores.get('safe', 0)
            scores['safe'] = min(original_safe_score * 1.5, 1.0)  # Safe skorunu belirgin şekilde artır (üst sınır 1.0)
            
            reduction_factor = 0.6 # Diğer skorlar için daha güçlü bir azaltma faktörü
            if original_safe_score > 0.7: # Eğer orijinal safe skoru zaten yüksekse, diğerlerini daha da fazla düşür
                reduction_factor = 0.4
            
            for category in other_categories_for_safe_check:
                scores[category] = max(scores.get(category, 0) * reduction_factor, 0.0)
            logger.info(f"Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve az kişi var, 'safe' skoru güçlendirildi ({original_safe_score:.2f} -> {scores['safe']:.2f}), diğerleri düşürüldü.")
        elif no_immediate_risk_objects and all_other_scores_truly_low and person_count > 1 and person_count <=3:
            # Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve 2-3 kişi var, safe yine de baskın olabilir
            original_safe_score = scores.get('safe', 0)
            scores['safe'] = min(original_safe_score * 1.3, 1.0) # Safe skorunu artır
            reduction_factor = 0.7
            if original_safe_score > 0.6:
                 reduction_factor = 0.5
            for category in other_categories_for_safe_check:
                scores[category] = max(scores.get(category, 0) * reduction_factor, 0.0)
            logger.info(f"Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve 2-3 kişi var, 'safe' skoru artırıldı ({original_safe_score:.2f} -> {scores['safe']:.2f}), diğerleri düşürüldü.")

        # CLIP SKORU YÜKSEK AMA YOLO ONAYI YOKSA DÜŞÜRME MANTIĞI
        # Bu blok, yukarıdaki 'safe' ayarlamalarından sonra çalışmalı ki,
        # 'safe' zaten diğer skorları düşürmüşse tekrar aşırı düşürmesin, ya da tam tersi.
        # Ancak mevcut durumda 'safe' ayarlaması sadece 'safe' olmayan kategorileri etkiliyor.
        # Bu düşürme, 'safe' dışındaki kategorilere odaklanmalı.

        high_clip_threshold = 0.7 # CLIP skorunun yüksek kabul edileceği eşik
        yolo_miss_reduction_factor = 0.5 # YOLO onayı olmadığında uygulanacak azaltma faktörü
        logger.info(f"[ContextualAdjust] Yüksek CLIP skoru düşürme kontrol parametreleri -> high_clip_threshold: {high_clip_threshold}, yolo_miss_reduction_factor: {yolo_miss_reduction_factor}")

        # Silah için
        if scores.get('weapon', 0) > high_clip_threshold and not weapon_detected:
            original_weapon_score = scores['weapon']
            scores['weapon'] *= yolo_miss_reduction_factor
            logger.info(f"CLIP 'weapon' skoru yüksek ({original_weapon_score:.2f}) ama YOLO onayı yok, skor {scores['weapon']:.2f}'ye düşürüldü.")

        # Madde için
        # 'general_drug_indicators' burada kafa karıştırabilir, çünkü bunlar zaten zayıf göstergeler.
        # Bu yüzden sadece 'drug_related_detected' (spesifik uyuşturucu nesneleri) kontrolüne odaklanalım.
        if scores.get('drug', 0) > high_clip_threshold and not drug_related_detected:
            original_drug_score = scores['drug']
            scores['drug'] *= yolo_miss_reduction_factor
            logger.info(f"CLIP 'drug' skoru yüksek ({original_drug_score:.2f}) ama YOLO spesifik onayı yok, skor {scores['drug']:.2f}'ye düşürüldü.")

        return scores

# Bu fonksiyonu analysis_service.py tarafından import edilebilmesi için ekliyoruz.
def get_content_analyzer():
    """ContentAnalyzer sınıfından bir örnek (singleton) döndürür."""
    return ContentAnalyzer() 