import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_hair_color(face_image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
    """
    Yüzün üst kısmından saç rengini çıkarır.
    
    Args:
        face_image: Tam frame
        bbox: [x, y, w, h] bounding box
        
    Returns:
        (r, g, b): Ortalama saç rengi veya None
    """
    try:
        x, y, w, h = bbox
        # Yüzün üst kısmını al (saç bölgesi)
        hair_region_height = int(h * 0.2)
        hair_region = face_image[max(0, y-hair_region_height):y, max(0, x):min(face_image.shape[1], x+w)]
        
        if hair_region.size == 0:
            return None
        
        # Görüntüyü HSV'ye dönüştür
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        # Renk maskeleri (saç renkleri için)
        # Siyah
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        # Kahverengi
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 255, 200])
        # Sarı/Sarışın
        lower_blonde = np.array([20, 50, 150])
        upper_blonde = np.array([30, 255, 255])
        
        # Maskeleri uygula
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask_blonde = cv2.inRange(hsv, lower_blonde, upper_blonde)
        
        # En fazla piksele sahip rengi seç
        black_count = np.sum(mask_black > 0)
        brown_count = np.sum(mask_brown > 0)
        blonde_count = np.sum(mask_blonde > 0)
        
        # Basitçe ortalama rengi döndür
        avg_color = np.mean(hair_region, axis=(0, 1))
        return tuple(avg_color)
        
    except Exception as e:
        logger.error(f"Saç rengi çıkarma hatası: {str(e)}")
        return None

def extract_skin_tone(face_image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
    """
    Yüz bölgesinden ortalama cilt tonunu çıkarır.
    
    Args:
        face_image: Tam frame
        bbox: [x, y, w, h] bounding box
        
    Returns:
        (r, g, b): Ortalama cilt tonu veya None
    """
    try:
        x, y, w, h = bbox
        # Yüzün orta kısmını al (burun ve yanaklar)
        center_x, center_y = x + w//2, y + h//2
        face_center_size = min(w, h) // 4
        
        face_center = face_image[
            max(0, center_y - face_center_size):min(face_image.shape[0], center_y + face_center_size),
            max(0, center_x - face_center_size):min(face_image.shape[1], center_x + face_center_size)
        ]
        
        if face_center.size == 0:
            return None
            
        # Ortalama rengi hesapla
        avg_color = np.mean(face_center, axis=(0, 1))
        return tuple(avg_color)
        
    except Exception as e:
        logger.error(f"Cilt tonu çıkarma hatası: {str(e)}")
        return None

def extract_face_landmarks(face_obj: object) -> np.ndarray | None:
    """
    InsightFace yüz nesnesinden landmark'ları çıkarır.
    
    Args:
        face_obj: InsightFace yüz nesnesi
        
    Returns:
        landmarks: Landmark noktaları (array) veya None
    """
    try:
        if hasattr(face_obj, 'landmark') and face_obj.landmark is not None:
            # InsightFace landmark'larını numpy array'e dönüştür
            return face_obj.landmark.astype(np.float32)
        return None
    except Exception as e:
        logger.error(f"Landmark çıkarma hatası: {str(e)}")
        return None

def extract_face_features(face_obj: object) -> dict:
    """
    Yüz nesnesinden temel özellikleri çıkarır.
    Args:
        face_obj: Yüz nesnesi (InsightFace veya dict).
    Returns:
        dict: Özellikler (bbox, kps, age, gender, embedding, vs.).
    """
    features = {
        'embedding': face_obj.embedding if hasattr(face_obj, 'embedding') else None,
        'gender': face_obj.gender if hasattr(face_obj, 'gender') else None,
        'landmarks': extract_face_landmarks(face_obj),
        'hair_color': extract_hair_color(image, bbox),
        'skin_tone': extract_skin_tone(image, bbox)
    }
    return features 