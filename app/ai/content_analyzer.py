import os
import numpy as np
import cv2
import logging
from flask import current_app
import tensorflow as tf
from ultralytics import YOLO
import torch
import shutil
import time
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
            
            # YOLOv8 modelini yükle
            self.yolo_model = self._load_yolo_model(model_folder)
            
            # Kategori sınıflandırıcılarını yükle
            self.violence_model = self._load_tf_model('violence', model_folder)
            self.adult_model = self._load_tf_model('adult', model_folder)
            self.harassment_model = self._load_tf_model('harassment', model_folder)
            self.weapon_model = self._load_tf_model('weapon', model_folder)
            self.drug_model = self._load_tf_model('drug', model_folder)
            
            self.initialized = True
            logger.info("ContentAnalyzer modelleri başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Content analyzer model yükleme hatası: {str(e)}")
            self.initialized = False
            raise
    
    def _load_yolo_model(self, model_folder):
        """YOLOv8 modelini yükler, cache kontrolü yapar"""
        cache_key = "yolov8"
        
        # Önbellekte varsa direkt döndür
        if cache_key in _models_cache:
            logger.info(f"YOLOv8 modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
            
        # Model klasörünü oluştur
        yolo_model_path = os.path.join(model_folder, 'detection', 'yolov8n.pt')
        if not os.path.exists(os.path.dirname(yolo_model_path)):
            os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
        
        try:
            # Model zaten var mı kontrolü
            if os.path.exists(yolo_model_path):
                logger.info(f"Mevcut YOLOv8 modeli yükleniyor: {yolo_model_path}")
                model = YOLO(yolo_model_path)
            else:
                # Model yoksa indir ve kaydet
                logger.info(f"YOLOv8 modeli indiriliyor: {yolo_model_path}")
                model = YOLO('yolov8n.pt')
                
                # İndirilen modeli kopyala
                if os.path.exists('yolov8n.pt'):
                    shutil.copy('yolov8n.pt', yolo_model_path)
                    logger.info(f"YOLOv8 modeli başarıyla kopyalandı: {yolo_model_path}")
            
            logger.info(f"YOLOv8 modeli başarıyla yüklendi")
            
            # Modeli önbelleğe ekle
            _models_cache[cache_key] = model
            return model
        except Exception as yolo_err:
            logger.error(f"YOLOv8 modeli yüklenemedi: {str(yolo_err)}")
            # Yeniden indirilmeye çalışılır
            try:
                model = YOLO('yolov8n.pt')
                logger.info(f"YOLOv8 modeli online kaynaktan yüklendi")
                _models_cache[cache_key] = model
                return model
            except Exception as e:
                logger.error(f"YOLOv8 modeli online kaynaktan da yüklenemedi: {str(e)}")
                raise
    
    def _load_tf_model(self, model_name, model_folder):
        """TensorFlow modelini yükler, önbellek kontrolü yapar"""
        cache_key = f"tf_{model_name}"
        
        # Önbellekte varsa direkt döndür
        if cache_key in _models_cache:
            logger.info(f"{model_name} modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
        
        # Model dizini tanımla
        model_path = os.path.join(model_folder, model_name)
        model_file = os.path.join(model_path, 'model.h5')
        
        try:
            # Klasörü oluştur
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
                
            # Eğer model dosyası varsa yüklemeyi dene
            if os.path.exists(model_file):
                logger.info(f"TensorFlow modeli yükleniyor: {model_file}")
                try:
                    # Custom nesneleri None olarak ayarla (eski modeller için)
                    model = tf.keras.models.load_model(model_file, compile=False)
                    logger.info(f"{model_name} modeli başarıyla yüklendi")
                    _models_cache[cache_key] = model
                    return model
                except Exception as load_err:
                    logger.error(f"Model yüklenirken hata: {str(load_err)}")
                    
                    # Eğer yüklenemezse, varsayılan MobileNetV2 modeline dön
                    logger.warning(f"{model_name} modeli standart ağırlıklar ile başlatılıyor")
                    model = self._create_base_model()
                    _models_cache[cache_key] = model
                    return model
            else:
                # Model yoksa yeni bir taban model oluştur
                logger.info(f"{model_name} modeli bulunamadı, standart ağırlıklar ile başlatılıyor")
                model = self._create_base_model()
                _models_cache[cache_key] = model
                return model
                
        except Exception as e:
            logger.error(f"{model_name} modeli yüklenemedi: {str(e)}")
            # Hata durumunda fallback olarak boş bir model döndür
            logger.warning(f"{model_name} için standart model döndürülüyor")
            
            try:
                model = self._create_base_model()
                _models_cache[cache_key] = model
                return model
            except Exception as base_err:
                logger.error(f"Standart model de oluşturulamadı: {str(base_err)}")
                raise
    
    def _create_base_model(self):
        """Temel bir MobileNetV2 modeli oluşturur (eğitilmemiş)"""
        # MobileNetV2 tabanlı model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Temel modeli dondur
        base_model.trainable = False
        
        # Üst katmanları ekle
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Model oluştur
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = global_avg_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = prediction_layer(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_image(self, image_path):
        """
        Bir resmi analiz eder ve içerik skorlarını hesaplar.
        
        Args:
            image_path: Analiz edilecek resmin dosya yolu
            
        Returns:
            tuple: (şiddet skoru, yetişkin içerik skoru, taciz skoru, silah skoru, madde kullanımı skoru, güvenli skoru, tespit edilen nesneler)
        """
        try:
            # Görüntüyü yükle
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Resim yüklenemedi: {image_path}")
            else:
                image = image_path  # Zaten numpy array
            
            # YOLOv8 ile nesne tespiti
            results = self.yolo_model(image)
            
            # Sonuçları işle
            detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Kutu koordinatları
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Yuvarlama ve Python int türüne dönüştürme
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    
                    # Güven skoru ve sınıf ID'si
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Etiket
                    label = self.yolo_model.names[cls_id]
                    
                    # Nesne bilgisini ekle
                    detected_objects.append({
                        'label': label,
                        'confidence': conf,
                        'box': [x1, y1, w, h]
                    })
            
            # Her kategori için skorları hesapla
            violence_score = self._analyze_with_model(image, self.violence_model)
            adult_content_score = self._analyze_with_model(image, self.adult_model)
            harassment_score = self._analyze_with_model(image, self.harassment_model)
            weapon_score = self._analyze_with_model(image, self.weapon_model)
            drug_score = self._analyze_with_model(image, self.drug_model)
            
            # Nesne tespitine dayalı skor iyileştirmeleri
            person_count = sum(1 for obj in detected_objects if obj['label'].lower() == 'person')
            weapon_objects = []
            drug_objects = []
            adult_objects = []
            
            # Tespit edilen nesneleri kategorilere ayır
            for obj in detected_objects:
                label = obj['label'].lower()
                conf = obj['confidence']
                
                # Silah kategorisi
                if label in ['gun', 'knife', 'rifle', 'pistol', 'shotgun', 'weapon']:
                    weapon_objects.append((label, conf))
                
                # Madde kullanımı ile ilişkili nesneler
                if label in ['bottle', 'wine glass', 'cup', 'syringe']:
                    drug_objects.append((label, conf))
                    
                # Yetişkin içeriği ile ilişkili nesneler
                if label in ['person'] and person_count >= 2:
                    adult_objects.append((label, conf))
            
            # Bağlama dayalı skor ayarlamaları (daha dinamik ve gerçekçi)
            
            # Silah skoru - silah nesneleri varsa
            if weapon_objects:
                # En yüksek güvenilirlikli silah nesnesini bul
                max_weapon_conf = max([conf for _, conf in weapon_objects])
                # Skor artışı - yüksek güvenilirlik = büyük artış
                weapon_boost = max_weapon_conf * 0.8
                weapon_score = max(weapon_score, weapon_boost)
                # Ayrıca şiddet skorunu da artır - silah genelde şiddetle ilişkilidir
                violence_boost = max_weapon_conf * 0.6
                violence_score = max(violence_score, violence_boost)
                logger.info(f"Silah nesnesi tespit edildi, weapon_score={weapon_score:.2f}, violence_score={violence_score:.2f}")
            
            # Kişiler arası etkileşim - 2+ kişi varsa
            if person_count >= 2:
                # Kişi sayısı arttıkça, taciz/yetişkin içeriği olasılığı artar
                interaction_factor = min(0.2 + (person_count * 0.05), 0.4)  # max 0.4
                harassment_score = max(harassment_score, harassment_score + interaction_factor)
                adult_content_score = max(adult_content_score, adult_content_score + (interaction_factor * 0.5))
                logger.info(f"{person_count} kişi tespit edildi, harassment_score={harassment_score:.2f}")
            
            # Madde kullanımı ile ilişkili nesneler
            if drug_objects:
                # Şüpheli nesne sayısı ve güvenilirliği önemli
                drug_object_boost = min(0.2 + (len(drug_objects) * 0.1), 0.5) # max 0.5
                drug_score = max(drug_score, drug_score + drug_object_boost)
                logger.info(f"Madde kullanımı ile ilişkili nesneler tespit edildi, drug_score={drug_score:.2f}")
                
            # Son skorları 0-1 aralığında sınırla
            violence_score = min(max(violence_score, 0.0), 1.0)
            adult_content_score = min(max(adult_content_score, 0.0), 1.0)
            harassment_score = min(max(harassment_score, 0.0), 1.0)
            weapon_score = min(max(weapon_score, 0.0), 1.0)
            drug_score = min(max(drug_score, 0.0), 1.0)
            
            # Güvenli skorunu hesapla - diğer kategorilerin tersi olarak
            # (Diğer tüm kategorilerin puanlarının maksimumu çıkarılır)
            risk_scores = [violence_score, adult_content_score, harassment_score, weapon_score, drug_score]
            max_risk_score = max(risk_scores)
            safe_score = max(0.0, 1.0 - max_risk_score)
            
            # Tüm puanları %100 toplam olacak şekilde normalize et
            normalized_scores = self._normalize_scores({
                'violence': violence_score,
                'adult_content': adult_content_score,
                'harassment': harassment_score,
                'weapon': weapon_score,
                'drug': drug_score,
                'safe': safe_score
            })
            
            # Normalize edilmiş skorları al
            violence_score = normalized_scores['violence']
            adult_content_score = normalized_scores['adult_content']
            harassment_score = normalized_scores['harassment']
            weapon_score = normalized_scores['weapon']
            drug_score = normalized_scores['drug']
            safe_score = normalized_scores['safe']
            
            # NumPy türlerini Python türlerine dönüştür - yeni utils modülünü kullan
            safe_objects = convert_numpy_types_to_python(detected_objects)
            
            # Tüm değerleri Python standart türlerine dönüştürerek döndür
            return (float(violence_score), float(adult_content_score), float(harassment_score), 
                   float(weapon_score), float(drug_score), float(safe_score), safe_objects)
        except Exception as e:
            logger.error(f"Görüntü analizi hatası: {str(e)}")
            raise
            
    def _analyze_with_model(self, image, model):
        """
        Verilen modeli kullanarak görüntüyü analiz eder.
        
        Args:
            image: Analiz edilecek görüntü
            model: Kullanılacak model
            
        Returns:
            float: Analiz skoru (0-1 arası)
        """
        try:
            # Görüntüyü ön işle
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalleştir
            img = np.expand_dims(img, axis=0)  # Batch boyutu ekle
            
            # Tahmini yap
            prediction = model.predict(img, verbose=0)
            score = float(prediction[0][0])
            
            return score
        except Exception as e:
            logger.error(f"Model tahmin hatası: {str(e)}")
            # Hata durumunda geri dönüş olarak imgenin özelliklerine dayalı bir skor belirle
            return self._calculate_image_features_score(image)
            
    def _calculate_image_features_score(self, image):
        """
        Görüntü özelliklerine dayalı bir skor hesaplar (model çalışmadığında).
        Bu fonksiyon, renk dağılımı, kenarlar ve dokular gibi çeşitli görüntü 
        özelliklerini analiz eder ve bu özelliklere dayalı bir risk skoru tahmin eder.
        """
        try:
            # Görüntü özelliklerini analiz et
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            
            # Renk analizleri
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            hue_var = np.std(hsv[:, :, 0]) / 180.0
            
            # Kenar tespiti (Yüksek kenar yoğunluğu şiddet veya silah içerebilir)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (image.shape[0] * image.shape[1] * 255)
            
            # Doku analizi
            texture_score = 0.0
            try:
                # Gri tonlamalı görüntü için GLCM (Gray Level Co-occurrence Matrix) hesapla
                from skimage.feature import graycomatrix, graycoprops
                glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
                contrast_texture = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                texture_score = min((contrast_texture + dissimilarity) / 50.0, 0.3)
            except Exception as tex_err:
                logger.warning(f"Doku analizi hesaplanamadı: {str(tex_err)}")
            
            # Farklı içerik kategorileri için farklı skorlar hesapla
            
            # Şiddet skoru: Kenar yoğunluğu ve kontrasttan etkilenir
            violence_score = 0.1 + (edge_density * 0.5) + (contrast * 0.2)
            violence_score = min(max(violence_score, 0.1), 0.5)
            
            # Yetişkin içeriği: Cilt tonu ve doygunluk seviyelerinden etkilenir
            has_skin_tones = self._detect_skin_tones(image)
            adult_score = 0.05 + (saturation * 0.3) + (0.2 if has_skin_tones else 0.0)
            adult_score = min(max(adult_score, 0.05), 0.3)
            
            # Taciz: Kompozisyon düzensizliğinden etkilenir
            harassment_score = 0.1 + (hue_var * 0.2) + (texture_score * 0.5)
            harassment_score = min(max(harassment_score, 0.1), 0.3)
            
            # Silah: Kenar yoğunluğundan etkilenir
            weapon_score = 0.05 + (edge_density * 0.4) + (contrast * 0.1)
            weapon_score = min(max(weapon_score, 0.05), 0.3)
            
            # Madde kullanımı: Renk dağılımı ve dokulardan etkilenir
            drug_score = 0.05 + (saturation * 0.2) + (texture_score * 0.3)
            drug_score = min(max(drug_score, 0.05), 0.25)
            
            logger.info(f"Görüntü özellikleri analizi: Şiddet={violence_score:.2f}, Yetişkin={adult_score:.2f}, "
                       f"Taciz={harassment_score:.2f}, Silah={weapon_score:.2f}, Madde={drug_score:.2f}")
            
            # İstenilen kategori için skorları döndür (bu durumda genel skor)
            return violence_score  # Şiddet puanını örnek olarak döndür
            
        except Exception as e:
            logger.error(f"Görüntü özellikleri hesaplama hatası: {str(e)}")
            # Güvenli bir varsayılan değer döndür
            return 0.2
    
    def _detect_skin_tones(self, image):
        """Görüntüde cilt tonları olup olmadığını tespit eder (basit yaklaşım)"""
        try:
            # HSV renk uzayına dönüştür
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Cilt tonu aralığı (HSV'de)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            # Cilt tonu maskesi oluştur
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Cilt pikseli oranı hesapla
            skin_ratio = np.sum(mask) / (image.shape[0] * image.shape[1] * 255)
            
            # Belirli bir eşiğin üzerinde cilt tonu varsa true döndür
            return skin_ratio > 0.15  # %15'den fazla cilt tonu içeriyorsa
        
        except Exception as e:
            logger.error(f"Cilt tonu tespiti hatası: {str(e)}")
            return False 
    
    def _normalize_scores(self, scores):
        """
        Kategorik skorları normalize eder, böylece toplam %100 olur.
        
        Args:
            scores: Kategori skorlarını içeren sözlük
            
        Returns:
            dict: Normalize edilmiş skorları içeren sözlük
        """
        try:
            # Skorların toplamını hesapla
            total = sum(scores.values())
            
            # Eğer toplam 0'sa, güvenli skoru 1 yapıp diğer skorları 0 olarak belirle
            if total <= 0.001:  # Çok küçük değerler için
                normalized_scores = {key: 0.0 for key in scores}
                normalized_scores['safe'] = 1.0
                return normalized_scores
            
            # Her skoru, toplama bölerek normalize et (%100 olacak şekilde)
            normalized_scores = {key: (value / total) for key, value in scores.items()}
            
            # Log çıktısı
            logger.info(f"Skorlar normalize edildi. Toplam: {sum(normalized_scores.values()):.2f}")
            for key, value in normalized_scores.items():
                logger.info(f"  {key}: {value:.4f} (%{value * 100:.1f})")
                
            return normalized_scores
        except Exception as e:
            logger.error(f"Skor normalizasyonu hatası: {str(e)}")
            # Hata durumunda orijinal skorları döndür
            return scores 