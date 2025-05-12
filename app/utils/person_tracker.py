import numpy as np
import logging
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

def cosine_similarity(v1, v2):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar (1 = aynı, 0 = farklı)"""
    return 1 - cosine(v1, v2)

def color_similarity(color1, color2):
    """İki renk arasındaki benzerliği hesaplar"""
    return 1 - np.linalg.norm(np.array(color1) - np.array(color2)) / 441.7  # 255*sqrt(3) maks uzaklık

def landmark_distance(lm1, lm2):
    """İki yüz landmark seti arasındaki normalize edilmiş mesafeyi hesaplar"""
    if len(lm1) != len(lm2):
        return 1.0  # Farklı boyutlar, büyük mesafe
    return np.mean(np.linalg.norm(np.array(lm1) - np.array(lm2), axis=1))

class PersonTracker:
    """
    Kişileri takip etmek ve ID değişimlerini (ID switches) tespit etmek için 
    birden fazla özelliği birleştiren sınıf.
    """
    def __init__(self, track_id, initial_gender, initial_embedding):
        self.track_id = track_id
        self.gender = initial_gender
        self.face_embeddings = [initial_embedding] 
        self.last_seen_frame = 0
        self.face_landmarks = []
        self.hair_color = None
        self.skin_tone = None
        self.avg_embedding = initial_embedding
        self.reliability_score = 0.5  # Başlangıç güvenilirlik skoru
        
    def update(self, new_embedding, new_gender, frame_idx, face_landmarks=None, 
              hair_color=None, skin_tone=None):
        """
        Kişinin özelliklerini günceller ve mevcut tespitle tutarlılık kontrolü yapar.
        
        Returns:
            bool: Güncellemenin kabul edilip edilmediği
        """
        # Cinsiyet tutarlılığı kontrolü
        gender_match = (new_gender == self.gender)
        
        # Embedding mesafesi kontrolü (eşik değeri 0.3'ten 0.4'e yükseltildi, yani daha toleranslı)
        embedding_distance = cosine(new_embedding, self.avg_embedding)
        embedding_match = embedding_distance < 0.4  # Eşik değeri güncellendi
        
        # Saç rengi ve cilt tonu kontrolü (eğer varsa)
        hair_match = True
        if hair_color is not None and self.hair_color is not None:
            hair_match = color_similarity(hair_color, self.hair_color) > 0.7
            
        skin_match = True
        if skin_tone is not None and self.skin_tone is not None:
            skin_match = color_similarity(skin_tone, self.skin_tone) > 0.8
        
        # Yüz özellikleri farkı kontrolü
        face_match = True
        if face_landmarks is not None and len(self.face_landmarks) > 0:
            landmarks_distance = landmark_distance(face_landmarks, self.face_landmarks[-1])
            face_match = landmarks_distance < 0.2
        
        # Güvenilirlik skorunu güncelle
        weights = {'gender': 0.3, 'embedding': 0.4, 'hair': 0.1, 'skin': 0.1, 'face': 0.1}
        new_reliability_raw = ( # Ham güvenilirlik skoru
            weights['gender'] * gender_match +
            weights['embedding'] * (1.0 - min(1.0, embedding_distance / 0.4)) + # Bölücü de güncellendi
            weights['hair'] * hair_match +
            weights['skin'] * skin_match +
            weights['face'] * face_match
        )
        
        # Eğer ham güvenilirlik skoru düşükse, bu kişi farklı olabilir
        # (Eşik değeri 0.6'dan 0.5'e düşürüldü)
        if new_reliability_raw < 0.5:  # Eşik değeri güncellendi
            logger.warning(f"Olası ID switch: Track ID {self.track_id}, Ham Güvenilirlik: {new_reliability_raw:.2f}")
            return False  # ID Switch olabilir, güncellemeyi reddet
        
        # Güncelleme yap
        self.face_embeddings.append(new_embedding)
        if len(self.face_embeddings) > 10:  # Sadece son 10 embedding'i tut
            self.face_embeddings.pop(0)
            
        # Ortalama embedding'i güncelle
        self.avg_embedding = np.mean(self.face_embeddings, axis=0)
        self.last_seen_frame = frame_idx
        
        if face_landmarks is not None:
            self.face_landmarks.append(face_landmarks)
            if len(self.face_landmarks) > 5:  # Son 5 landmark set'ini tut
                self.face_landmarks.pop(0)
                
        if hair_color is not None:
            if self.hair_color is None:
                self.hair_color = np.array(hair_color) if isinstance(hair_color, (list, tuple)) else hair_color
            else:
                current_hair_color = np.array(self.hair_color) if isinstance(self.hair_color, (list, tuple)) else self.hair_color
                new_hair_color_arr = np.array(hair_color) if isinstance(hair_color, (list, tuple)) else hair_color
                self.hair_color = 0.9 * current_hair_color + 0.1 * new_hair_color_arr # Yavaşça güncelle
                
        if skin_tone is not None:
            if self.skin_tone is None:
                self.skin_tone = np.array(skin_tone) if isinstance(skin_tone, (list, tuple)) else skin_tone
            else:
                current_skin_tone = np.array(self.skin_tone) if isinstance(self.skin_tone, (list, tuple)) else self.skin_tone
                new_skin_tone_arr = np.array(skin_tone) if isinstance(skin_tone, (list, tuple)) else skin_tone
                self.skin_tone = 0.9 * current_skin_tone + 0.1 * new_skin_tone_arr # Yavaşça güncelle
                
        # Güvenilirlik skorunu güncelle (biraz ağırlıklı ortalama)
        self.reliability_score = 0.8 * self.reliability_score + 0.2 * new_reliability_raw
        
        return True


class PersonTrackerManager:
    """
    Birden fazla kişiyi takip eden ve ID değişimlerini önleyen yönetici sınıf.
    DeepSORT'tan gelen takipleri filtreler ve güvenilirlik skorlarını yönetir.
    """
    def __init__(self, reliability_threshold=0.5):
        self.person_trackers = {}  # track_id -> PersonTracker
        self.reliability_threshold = reliability_threshold
        self.max_frames_missing = 30  # Bu kadar frame'dir görünmeyen track'leri sil
        self.current_frame = 0
        
    def update(self, tracks, face_data, frame_idx):
        """
        DeepSORT takiplerini işler ve güvenilir takipleri döndürür
        
        Args:
            tracks: DeepSORT'tan gelen takip listesi
            face_data: Her yüz için {embedding, gender, landmarks, hair_color, skin_tone} içeren sözlük 
            frame_idx: Mevcut frame indeksi
            
        Returns:
            list: Güvenilir takipler listesi
        """
        self.current_frame = frame_idx
        
        # Sonuçları saklayacak liste
        reliable_tracks = []
        
        # Mevcut frame'de görülen track ID'leri
        seen_track_ids = set()
        
        # Her takibi işle
        for i, (track_obj, data) in enumerate(zip(tracks, face_data)): # track_obj ve i eklendi
            if not hasattr(track_obj, 'is_confirmed') or not track_obj.is_confirmed():
                logger.debug(f"Track {i} onaylanmamış, atlanıyor.")
                continue  # Onaylanmamış track'leri atla
                
            track_id = str(track_obj.track_id)
            seen_track_ids.add(track_id)
            
            # Yüz özellikleri
            embedding = data.get('embedding')
            gender = data.get('gender')
            landmarks = data.get('landmarks')
            hair_color = data.get('hair_color')
            skin_tone = data.get('skin_tone')

            if embedding is None or gender is None:
                logger.warning(f"Track ID {track_id} için embedding veya gender eksik, bu track atlanıyor.")
                continue
            
            # Eğer bu track daha önce görülmemişse, yeni bir tracker oluştur
            if track_id not in self.person_trackers:
                logger.info(f"Yeni kişi takibi başlatılıyor: ID {track_id}")
                self.person_trackers[track_id] = PersonTracker(track_id, gender, embedding)
                # Yeni oluşturulan tracker'ın güvenilirliğini doğrudan kontrol et
                if self.person_trackers[track_id].reliability_score >= self.reliability_threshold:
                    reliable_tracks.append(track_obj)
                else:
                    logger.info(f"Yeni takip ID {track_id} başlangıç güvenilirliği ({self.person_trackers[track_id].reliability_score:.2f}) düşük, listeye eklenmedi.")

                continue
            
            # Mevcut tracker'ı güncelle
            person_tracker_instance = self.person_trackers[track_id] # Değişken adı düzeltildi
            is_update_accepted = person_tracker_instance.update( # Değişken adı düzeltildi
                embedding, gender, frame_idx,
                face_landmarks=landmarks,
                hair_color=hair_color,
                skin_tone=skin_tone
            )
            
            # Eğer güncelleme kabul edildiyse ve genel güvenilirlik yeterliyse, track'i listeye ekle
            if is_update_accepted and person_tracker_instance.reliability_score >= self.reliability_threshold: # Değişken adı düzeltildi
                reliable_tracks.append(track_obj)
            else:
                logger.warning(f"ID {track_id} için güncelleme reddedildi veya güvenilirlik düşük ({person_tracker_instance.reliability_score:.2f}), takip bu frame için filtrelendi") # Değişken adı düzeltildi
        
        # Uzun süre görünmeyen track'leri temizle
        self._cleanup_old_trackers(seen_track_ids)
        
        return reliable_tracks
    
    def _cleanup_old_trackers(self, seen_track_ids):
        """Uzun süre görünmeyen takipleri temizler"""
        to_delete = []
        for track_id, tracker_instance in self.person_trackers.items(): # Değişken adı düzeltildi
            # Bu frame'de görünmeyen track'leri işaretle
            if track_id not in seen_track_ids:
                frames_missing = self.current_frame - tracker_instance.last_seen_frame # Değişken adı düzeltildi
                if frames_missing > self.max_frames_missing:
                    to_delete.append(track_id)
        
        # Eski takipleri sil
        for track_id in to_delete:
            logger.info(f"Eski takip siliniyor: ID {track_id}, son görülme: {self.person_trackers[track_id].last_seen_frame}")
            del self.person_trackers[track_id]
    
    def get_reliability_score(self, track_id):
        """Belirli bir track ID'si için güvenilirlik skorunu döndürür"""
        if track_id in self.person_trackers:
            return self.person_trackers[track_id].reliability_score
        return 0.0 