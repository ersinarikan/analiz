import os
import numpy as np
import cv2
import logging
from flask import current_app
import tensorflow as tf
from ultralytics import YOLO
import torch
import clip
import open_clip
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
                }
            }
            # Kategori text tokenları önceden hesapla (tek prompt)
            self.category_text_features = {}
            for category, prompts in self.category_prompts.items():
                text_input = self.tokenizer(prompts["positive"][0]).to("cuda" if torch.cuda.is_available() else "cpu")
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
        """CLIP modelini yükler, önbellek kontrolü yapar ve merkezi yoldan yükler."""
        cache_key = "clip_model"
        if cache_key in _models_cache:
            logger.info("CLIP modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
        
        try:
            device = "cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu"
            
            # Merkezi model yolunu ve model dosya adını al
            active_clip_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
            # Varsayılan olarak base_model'deki .bin dosyasını kullanacağız.
            # Fine-tuning sonrası bu active_model klasörü güncellenecek.
            # Eğer active_model boşsa veya .bin dosyası yoksa, base_model'e fallback yapabiliriz.
            # Şimdilik active_model içinde open_clip_pytorch_model.bin olduğunu varsayıyoruz.
            
            # OpenCLIP için model adı ve ön eğitimli ağırlıkların yolu
            # model_name hala gerekli çünkü config ve tokenizer dosyaları bununla eşleşebilir.
            clip_model_name_from_config = current_app.config['OPENCLIP_MODEL_NAME'].split('_')[0] # örn: ViT-H-14-378-quickgelu
            
            # Kontrol: active_model klasörü var mı ve içinde .bin dosyası var mı?
            # Eğer yoksa, base_model'den yüklemeyi deneyebilir veya hata verebilir.
            # Şimdilik, active_model içinde olduğunu varsayarak devam edelim.
            # Gerçek model dosyasının adı standart olmayabilir, bu yüzden onu da config'e eklemek iyi olabilir.
            # Şimdilik 'open_clip_pytorch_model.bin' olduğunu varsayıyoruz.
            pretrained_weights_path = os.path.join(active_clip_model_path, 'open_clip_pytorch_model.bin')

            if not os.path.exists(pretrained_weights_path):
                logger.error(f"CLIP model ağırlık dosyası bulunamadı: {pretrained_weights_path}")
                # Fallback: base_model'i dene
                base_clip_model_path = current_app.config['OPENCLIP_MODEL_BASE_PATH']
                pretrained_weights_path = os.path.join(base_clip_model_path, 'open_clip_pytorch_model.bin')
                if not os.path.exists(pretrained_weights_path):
                    logger.error(f"Fallback CLIP model ağırlık dosyası da bulunamadı: {pretrained_weights_path}")
                    raise FileNotFoundError(f"CLIP model ağırlık dosyası ne aktif ne de base path'te bulunamadı: {pretrained_weights_path}")
                logger.info(f"Aktif CLIP modeli bulunamadı, base modelden yüklenecek: {pretrained_weights_path}")

            logger.info(f"CLIP modeli yükleniyor (Model: {clip_model_name_from_config}, Ağırlıklar: {pretrained_weights_path})")
            
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                model_name=clip_model_name_from_config, # örn: "ViT-H-14-378-quickgelu"
                pretrained=pretrained_weights_path,     # örn: ".../active_model/open_clip_pytorch_model.bin"
                device=device,
                jit=False # JIT ile sorunlar yaşanabiliyor, False olarak ayarlayalım
            )
            _models_cache[cache_key] = (model, preprocess_val)
            logger.info(f"{clip_model_name_from_config} CLIP modeli (Ağırlıklar: {pretrained_weights_path}) {device} üzerinde başarıyla yüklendi ve önbelleğe alındı.")
            return model, preprocess_val
            
        except Exception as e:
            logger.error(f"CLIP modeli yüklenemedi: {str(e)}", exc_info=True)
            raise
    
    def _load_yolo_model(self, model_folder):
        """YOLOv8 modelini merkezi aktif yoldan yükler, cache kontrolü yapar"""
        cache_key = "yolov8"
        
        if cache_key in _models_cache:
            logger.info(f"YOLOv8 modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
            
        active_yolo_model_base_path = current_app.config['YOLO_MODEL_ACTIVE_PATH']
        # YOLO model adı config'den alınabilir veya sabit olabilir.
        # Örn: YOLO_MODEL_NAME = 'yolov8x'
        yolo_model_filename = current_app.config.get('YOLO_MODEL_NAME', 'yolov8x') + '.pt' #örn: yolov8x.pt
        yolo_model_full_path = os.path.join(active_yolo_model_base_path, yolo_model_filename)

        if not os.path.exists(yolo_model_full_path):
            logger.warning(f"Aktif YOLO modeli bulunamadı: {yolo_model_full_path}. Base model denenecek.")
            base_yolo_path = current_app.config['YOLO_MODEL_BASE_PATH']
            yolo_model_full_path = os.path.join(base_yolo_path, yolo_model_filename)
            if not os.path.exists(yolo_model_full_path):
                logger.error(f"YOLO modeli ne aktif ne de base path'te bulunamadı: {yolo_model_full_path}")
                # Acil durum: Online'dan indirmeyi dene (eski davranış)
                try:
                    logger.info(f"YOLO modeli yerel olarak bulunamadı. {yolo_model_filename} online'dan indirilmeye çalışılıyor...")
                    model = YOLO(yolo_model_filename) # yolov8x.pt veya yolov8n.pt
                    # İndirilen modeli base_path'e kaydetmeyi düşünebiliriz.
                    # shutil.copy(yolo_model_filename, os.path.join(current_app.config['YOLO_MODEL_BASE_PATH'], yolo_model_filename))
                    logger.info(f"{yolo_model_filename} modeli online kaynaktan fallback olarak yüklendi.")
                    _models_cache[cache_key] = model
                    return model
                except Exception as fallback_err:
                    logger.error(f"Fallback {yolo_model_filename} modeli de online kaynaktan yüklenemedi: {fallback_err}", exc_info=True)
                    raise FileNotFoundError(f"YOLO modeli bulunamadı: {yolo_model_full_path} ve online'dan da indirilemedi.")
            logger.info(f"Aktif YOLO modeli bulunamadı, base modelden yüklenecek: {yolo_model_full_path}")
        
        try:
            logger.info(f"YOLOv8 modeli yükleniyor: {yolo_model_full_path}")
            model = YOLO(yolo_model_full_path)
            logger.info(f"YOLOv8 modeli ({yolo_model_full_path}) başarıyla yüklendi")
            _models_cache[cache_key] = model
            return model
        except Exception as yolo_err:
            logger.error(f"YOLOv8 modeli yüklenemedi ({yolo_model_full_path}): {str(yolo_err)}", exc_info=True)
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

                # --- START: MODIFIED SCORE CALCULATION ---
                SQUASH_FACTOR = 3.5 
                squashed_fark = math.tanh(fark * SQUASH_FACTOR)
                nihai_skor = (squashed_fark + 1) / 2  # Normalize to 0-1 range
                
                # Detaylı log
                logger.info(f"[CLIP_PROMPT_LOG] Category: {cat}")
                for i, prompt in enumerate(pos_prompts):
                    logger.info(f"  Prompt: {prompt}    Score: {similarities[i]:.4f}")
                for i, prompt in enumerate(neg_prompts):
                    logger.info(f"  Prompt: {prompt}    Score: {similarities[len(pos_prompts)+i]:.4f}")
                logger.info(f"  Positive mean: {pos_score:.4f}, Negative mean: {neg_score:.4f}")
                logger.info(f"  Original Fark: {fark:.4f}, Squashed Fark (tanh(fark*{SQUASH_FACTOR})): {squashed_fark:.4f}, Final Score: {nihai_skor:.4f}")
                # --- END: MODIFIED SCORE CALCULATION ---
                
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

        # Kişilerarası etkileşim ayarlaması kaldırıldı - çok yüksek skorlara neden oluyordu
        
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