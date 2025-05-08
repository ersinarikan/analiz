import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AppearanceFeatures:
    """
    Kişilerin görünüm özelliklerini (kıyafet renkleri, vücut oranları vb.) analiz eden ve 
    DeepSORT takipçisine ek bilgi sağlayan yardımcı sınıf.
    """
    
    def __init__(self):
        # Renk aralıkları (HSV)
        self.color_ranges = {
            'kirmizi': ([0, 100, 100], [10, 255, 255]),
            'turuncu': ([10, 100, 100], [25, 255, 255]),
            'sari': ([25, 100, 100], [35, 255, 255]),
            'yesil': ([35, 100, 100], [85, 255, 255]),
            'mavi': ([85, 100, 100], [130, 255, 255]),
            'mor': ([130, 100, 100], [170, 255, 255]),
            'pembe': ([170, 100, 100], [180, 255, 255]),
            'siyah': ([0, 0, 0], [180, 30, 30]),
            'beyaz': ([0, 0, 200], [180, 30, 255]),
            'gri': ([0, 0, 70], [180, 30, 200])
        }
    
    def extract_upper_body_region(self, image, face_bbox):
        """
        Yüz etrafındaki üst vücut bölgesini çıkarır
        
        Args:
            image: Orijinal görüntü
            face_bbox: Yüz sınırlayıcı kutusu (x, y, w, h)
            
        Returns:
            Üst vücut bölgesi görüntüsü veya None
        """
        try:
            x, y, w, h = face_bbox
            
            # Yüzün altındaki bölgeyi kıyafet olarak varsay
            # Genişlik: yüz genişliğinin 3 katı
            # Yükseklik: yüz yüksekliğinin 4 katı
            body_width = w * 3
            body_height = h * 4
            
            # Merkezi yüzün merkezine hizalı olsun
            body_x = max(0, x + w//2 - body_width//2)
            body_y = min(y + h, image.shape[0]) # Yüzün hemen altından başla
            
            # Görüntü sınırlarını kontrol et
            body_width = min(body_width, image.shape[1] - body_x)
            body_height = min(body_height, image.shape[0] - body_y)
            
            if body_width <= 0 or body_height <= 0:
                return None
                
            # Üst vücut bölgesini çıkar
            upper_body = image[body_y:body_y+body_height, body_x:body_x+body_width]
            
            if upper_body.size == 0:
                return None
                
            return upper_body, (body_x, body_y, body_width, body_height)
            
        except Exception as e:
            logger.error(f"Üst vücut bölgesi çıkarma hatası: {str(e)}")
            return None
    
    def compute_color_histogram(self, image_region):
        """
        Görüntü bölgesinin renk histogramını hesaplar
        
        Args:
            image_region: Görüntü bölgesi (BGR formatında)
            
        Returns:
            Renk histogramı (özellik vektörü) veya None
        """
        try:
            if image_region is None or image_region.size == 0:
                return None
                
            # BGR'den HSV'ye dönüştür
            hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
            
            # Renk histogramı hesapla (H: 30 bin, S: 32 bin, V: 32 bin)
            hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize et
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
            
            # Histogramları birleştir
            histogram = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            
            return histogram
            
        except Exception as e:
            logger.error(f"Renk histogramı hesaplama hatası: {str(e)}")
            return None
    
    def identify_dominant_colors(self, image_region, top_n=2):
        """
        Görüntü bölgesinin baskın renklerini belirler
        
        Args:
            image_region: Görüntü bölgesi (BGR formatında)
            top_n: Kaç baskın renk döndürüleceği
            
        Returns:
            Baskın renkler listesi veya None
        """
        try:
            if image_region is None or image_region.size == 0:
                return None
                
            # BGR'den HSV'ye dönüştür
            hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
            
            # Her renk için maske oluştur ve piksel sayısını hesapla
            color_counts = {}
            
            for color_name, (lower, upper) in self.color_ranges.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                
                # Renk aralığı maskesi
                mask = cv2.inRange(hsv, lower, upper)
                count = cv2.countNonZero(mask)
                
                # Toplam piksel sayısına göre normalize et
                color_counts[color_name] = count / (image_region.shape[0] * image_region.shape[1])
            
            # En çok piksel içeren top_n rengi bul
            dominant_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Baskın renk belirleme hatası: {str(e)}")
            return None
    
    def calculate_body_ratio_features(self, full_body_bbox):
        """
        Vücut oranlarından özellikler çıkarır
        
        Args:
            full_body_bbox: Tam vücut sınırlayıcı kutusu (x, y, w, h)
            
        Returns:
            Vücut oran özellikleri veya None
        """
        try:
            x, y, w, h = full_body_bbox
            
            # Basit oran özelliği: genişlik/yükseklik
            aspect_ratio = w / h if h > 0 else 0
            
            # Normalleştirilmiş özellikler
            norm_width = w / 100  # Örnek normalizasyon
            norm_height = h / 200  # Örnek normalizasyon
            
            return {
                'aspect_ratio': aspect_ratio,
                'norm_width': norm_width,
                'norm_height': norm_height
            }
            
        except Exception as e:
            logger.error(f"Vücut oranı hesaplama hatası: {str(e)}")
            return None
    
    def compare_color_histograms(self, hist1, hist2):
        """
        İki renk histogramını karşılaştırır
        
        Args:
            hist1: Birinci histogram
            hist2: İkinci histogram
            
        Returns:
            Benzerlik skoru (0-1 arasında, 1 en yüksek benzerlik)
        """
        try:
            if hist1 is None or hist2 is None:
                return 0.0
                
            # Histogram benzerliğini hesapla (Bhattacharyya mesafesi)
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            
            # Mesafeyi benzerlik skoruna dönüştür (1 - mesafe)
            similarity = 1.0 - score
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Histogram karşılaştırma hatası: {str(e)}")
            return 0.0
    
    def compare_dominant_colors(self, colors1, colors2):
        """
        İki baskın renk setini karşılaştırır
        
        Args:
            colors1: Birinci baskın renkler listesi
            colors2: İkinci baskın renkler listesi
            
        Returns:
            Benzerlik skoru (0-1 arasında, 1 en yüksek benzerlik)
        """
        try:
            if colors1 is None or colors2 is None:
                return 0.0
                
            # Renk isimlerini ve oranlarını ayrı listeler olarak al
            color_names1 = [c[0] for c in colors1]
            color_names2 = [c[0] for c in colors2]
            
            # Ortak renkleri bul
            common_colors = set(color_names1).intersection(set(color_names2))
            
            # Ortak renk sayısını toplam renk sayısına böl
            similarity = len(common_colors) / max(len(color_names1), len(color_names2))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Baskın renk karşılaştırma hatası: {str(e)}")
            return 0.0
    
    def combine_appearance_similarity(self, hist_similarity, color_similarity, ratio_similarity=None):
        """
        Farklı benzerlik skorlarını birleştirerek genel görünüm benzerlik skoru hesaplar
        
        Args:
            hist_similarity: Histogram benzerlik skoru
            color_similarity: Baskın renk benzerlik skoru
            ratio_similarity: Vücut oranı benzerlik skoru (opsiyonel)
            
        Returns:
            Genel benzerlik skoru (0-1 arasında)
        """
        # Ağırlıklar (önem sırası)
        hist_weight = 0.6
        color_weight = 0.3
        ratio_weight = 0.1
        
        # Temel ağırlıklı ortalama
        total_similarity = hist_similarity * hist_weight + color_similarity * color_weight
        total_weight = hist_weight + color_weight
        
        # Vücut oranı benzerliği varsa ekle
        if ratio_similarity is not None:
            total_similarity += ratio_similarity * ratio_weight
            total_weight += ratio_weight
        
        # Normalize et
        if total_weight > 0:
            total_similarity /= total_weight
            
        return total_similarity 