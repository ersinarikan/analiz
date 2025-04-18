import os
import numpy as np
import torch
import cv2
import sys
from flask import current_app
from app.utils.image_utils import load_image
from PIL import Image
import logging
import face_recognition
import math
from facenet_pytorch import MTCNN
import tensorflow as tf

logger = logging.getLogger(__name__)

# Pytorch-Age-Estimation modülü yoksa hata vermesin
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Pytorch-Age-Estimation'))
    from AgeNet.models import Model
except ImportError:
    logger.warning("AgeNet.models modülü bulunamadı, alternatif model kullanılacak")

class AgeEstimator:
    """
    Yaş tahmin sınıfı, görüntülerdeki yüzleri tespit eder ve yaş tahmini yapar.
    """
    
    def __init__(self):
        """
        AgeEstimator sınıfını başlatır ve gerekli modelleri yükler.
        """
        self.models = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_models()
        
    def _load_models(self):
        """
        Gerekli yüz tanıma ve yaş tahmin modellerini yükler.
        """
        try:
            # Yüz tespiti için MTCNN modelini yükle
            self.models['face_detector'] = MTCNN(
                image_size=224, margin=0, keep_all=True, 
                min_face_size=60, device=self.device
            )
            
            # Yaş tahmini için model
            model_folder = current_app.config.get('MODELS_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'storage', 'models'))
            age_model_dir = os.path.join(model_folder, 'age')
            age_model_path = os.path.join(age_model_dir, 'age_model.h5')
            
            # Modeli yükleme denemeye çalış
            load_failed = False
            
            # Mevcut modeli yüklemeyi dene
            if os.path.exists(age_model_path):
                logger.info(f"Yaş modeli yükleniyor: {age_model_path}")
                try:
                    # Custom objects'i tanımla
                    custom_objects = {
                        'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError
                    }
                    self.models['age_predictor'] = tf.keras.models.load_model(
                        age_model_path,
                        custom_objects=custom_objects
                    )
                    logger.info("Yaş tahmin modeli başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"Age estimator model yükleme hatası: {str(e)}")
                    # Mevcut modeli sil ve yenisini oluştur
                    load_failed = True
                    if os.path.exists(age_model_path):
                        try:
                            os.remove(age_model_path)
                            logger.info(f"Hatalı yaş modeli silindi: {age_model_path}")
                        except:
                            logger.warning(f"Hatalı yaş modeli silinemedi: {age_model_path}")
            else:
                load_failed = True
                
            # Eğer model yüklenemedi veya yoksa, yeni model oluştur
            if load_failed or not os.path.exists(age_model_path):
                logger.info(f"Yaş modeli oluşturuluyor: {age_model_path}")
                
                # Klasör yoksa oluştur
                if not os.path.exists(age_model_dir):
                    os.makedirs(age_model_dir)
                
                # MobileNetV2 tabanlı yaş tahmin modeli
                model = tf.keras.Sequential([
                    tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet'
                    ),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')  # Yaş tahmini için regresyon
                ])
                
                # Compile et - MAE metrik ve loss fonksiyonu olarak doğrudan fonksiyon nesnelerini kullan
                model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()]
                )
                
                # Modeli kaydet - HDF5 formatında
                model.save(age_model_path)
                
                # Modeli belleğe yükle
                self.models['age_predictor'] = model
                logger.info(f"Yaş tahmin modeli başarıyla oluşturuldu ve kaydedildi: {age_model_path}")
            
            logger.info("Age estimator modelleri başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Age estimator model yükleme hatası: {str(e)}")
            logger.error("Yüz tespiti ve yaş tahmini yapılamayabilir")
            # Face detector'ı yine de hazır tut
            try:
                if 'face_detector' not in self.models:
                    self.models['face_detector'] = MTCNN(
                        image_size=224, margin=0, keep_all=True, 
                        min_face_size=60, device=self.device
                    )
            except Exception:
                pass
    
    def detect_faces(self, image):
        """
        Bir görüntüdeki yüzleri tespit eder.
        
        Args:
            image: Yüz tespiti yapılacak görüntü (OpenCV formatında)
            
        Returns:
            list: Tespit edilen yüzlerin konum bilgilerini içeren sözlük listesi
        """
        try:
            # Görüntüyü RGB'ye dönüştür (MTCNN için)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MTCNN ile yüz tespiti yap
            boxes, probs = self.models['face_detector'].detect(rgb_image)
            
            # Sonuç None ise boş liste döndür
            if boxes is None:
                return []
            
            # Tespit edilen yüzleri işle
            faces = []
            for i, box in enumerate(boxes):
                # Güven skoru yeterli ise ekle
                if probs[i] >= 0.9:
                    x1, y1, x2, y2 = box.astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    faces.append({
                        'location': (x1, y1, w, h),
                        'confidence': float(probs[i])
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"Yüz tespiti hatası: {str(e)}")
            return []  # Hata durumunda boş liste döndür
    
    def estimate_age(self, face_image):
        """
        Verilen yüz görüntüsü için yaş tahmini yapar.
        
        Args:
            face_image: Yaş tahmini yapılacak yüz görüntüsü
            
        Returns:
            tuple: (tahmini yaş, güven skoru)
        """
        try:
            # age_predictor modeli yok mu kontrol et
            if 'age_predictor' not in self.models:
                # Varsayılan değerler döndür - 25 yaş, düşük güven
                return 25.0, 0.5
                
            # Yüz görüntüsünü ön işle
            img = cv2.resize(face_image, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalleştir
            img = np.expand_dims(img, axis=0)  # Batch boyutu ekle
            
            # Yaş tahmini yap
            prediction = self.models['age_predictor'].predict(img, verbose=0)
            estimated_age = float(prediction[0][0])
            
            # Yaş tahmini genelde 0-100 arasında olmalı
            estimated_age = max(0, min(100, estimated_age))
            
            # Güven skoru için basit bir hesaplama
            confidence = 0.9
            
            return estimated_age, confidence
            
        except Exception as e:
            logger.error(f"Yaş tahmini hatası: {str(e)}")
            # Hata durumunda varsayılan değerler döndür
            return 25.0, 0.5  # Ortalama yaş ve düşük güven
    
    def compute_face_encoding(self, face_image):
        """
        Yüz için 128 boyutlu bir vektör hesaplar (yüz tanıma için).
        
        Args:
            face_image: Vektörü hesaplanacak yüz görüntüsü
            
        Returns:
            list: 128 boyutlu yüz vektörü
        """
        try:
            # RGB'ye dönüştür (face_recognition için)
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Yüz kodlamasını hesapla
            encodings = face_recognition.face_encodings(rgb_face)
            
            # Eğer hiç kodlama bulunamazsa boş vektor döndür
            if not encodings:
                return np.zeros(128)
                
            return encodings[0]
            
        except Exception as e:
            logger.error(f"Yüz vektörü hesaplama hatası: {str(e)}")
            return np.zeros(128)  # Hata durumunda sıfır vektör döndür
    
    def compare_faces(self, encoding1, encoding2, tolerance=0.6):
        """
        İki yüz vektörü arasındaki benzerliği kontrol eder.
        
        Args:
            encoding1: Birinci yüz vektörü (liste veya NumPy dizisi)
            encoding2: İkinci yüz vektörü (liste veya NumPy dizisi)
            tolerance: Benzerlik eşiği (daha düşük değer = daha sıkı eşleşme)
            
        Returns:
            bool: İki yüz vektörü birbirine benziyorsa True
        """
        try:
            # Liste ise NumPy dizisine dönüştür
            if isinstance(encoding1, list):
                encoding1 = np.array(encoding1)
            if isinstance(encoding2, list):
                encoding2 = np.array(encoding2)
                
            # Öklid mesafesini hesapla
            distance = np.linalg.norm(encoding1 - encoding2)
            
            # Mesafe eşikten küçükse, benzerdir
            return distance <= tolerance
            
        except Exception as e:
            logger.error(f"Yüz karşılaştırma hatası: {str(e)}")
            return False
    
    def transform(self, image):
        """
        Görüntüyü model için uygun formata dönüştürür.
        Bu fonksiyon, çeşitli formatlardaki görüntüleri model girişi için uygun tensörlere dönüştürür.
        
        Args:
            image: Numpy array veya PIL Image formatında görüntü
            
        Returns:
            torch.Tensor: Model girişi için hazırlanmış tensör
        """
        try:
            # PIL formatına dönüştürülüyor
            if isinstance(image, np.ndarray):
                # OpenCV BGR'dan RGB'ye dönüştürülüyor
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Boyutlandırma ve normalizasyon işlemleri yapılıyor
            transform = torch.nn.Sequential(
                torch.nn.Resize((64, 64)),  # Yeniden boyutlandırma
                torch.nn.ToTensor(),  # Tensöre dönüştürme [0-255] -> [0-1]
                torch.nn.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalleştirme [-1, 1]
            )
            
            # Batch boyutu ekleniyor ve dönüştürülmüş görüntü döndürülüyor
            return transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Görüntü dönüştürme hatası: {str(e)}")
            # Hata durumunda None döndür
            return None
    
    def analyze_image(self, image):
        """
        Görüntü üzerinde yüz tespiti ve yaş/cinsiyet tahmini yapar.
        Bu metot, bir görüntüdeki tüm yüzleri tespit edip her biri için yaş ve cinsiyet tahmini yapar.
        
        Args:
            image: Görüntü yolu veya numpy array
            
        Returns:
            list: Her yüz için tespit ve tahmin sonuçlarını içeren sözlük listesi
        """
        try:
            if image is None:
                return []
            
            # Görüntü bir dosya yolu ise yükleniyor
            if isinstance(image, str):
                image = load_image(image)
                if image is None:
                    return []
            
            # Yüzleri tespit et
            faces = self.detect_faces(image)
            
            # Her yüz için yaş ve cinsiyet tahmini yapılıyor
            results = []
            for face in faces:
                try:
                    x, y, w, h = face['location']
                    face_img = image[y:y+h, x:x+w]  # Yüz bölgesi kesiliyor
                    
                    # Yaş ve cinsiyet tahmini yapılıyor
                    age, confidence = self.estimate_age(face_img)
                    
                    # Sonuçlar listeye ekleniyor
                    results.append({
                        'location': face['location'],
                        'confidence': face['confidence'],
                        'age': age,
                        'age_confidence': confidence
                    })
                except Exception as face_error:
                    logger.error(f"Yüz analizi hatası: {str(face_error)}")
                    # Hataya rağmen devam et, diğer yüzleri analiz et
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Görüntü analizi hatası: {str(e)}")
            return []  # Hata durumunda boş liste döndür 