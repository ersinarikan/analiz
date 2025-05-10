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
        
        # Embedding mesafesi kontrolü
        embedding_distance = cosine(new_embedding, self.avg_embedding)
        embedding_match = embedding_distance < 0.3  # Eşik değeri
        
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
        new_reliability = (
            weights['gender'] * gender_match +
            weights['embedding'] * (1.0 - min(1.0, embedding_distance / 0.3)) +
            weights['hair'] * hair_match +
            weights['skin'] * skin_match +
            weights['face'] * face_match
        )
        
        # Eğer güvenilirlik skoru düşükse, bu kişi farklı olabilir
        if new_reliability < 0.6:
            logger.warning(f"Olası ID switch: Track ID {self.track_id}, Güvenilirlik: {new_reliability:.2f}")
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
                self.hair_color = hair_color
            else:
                self.hair_color = 0.9 * self.hair_color + 0.1 * hair_color  # Yavaşça güncelle
                
        if skin_tone is not None:
            if self.skin_tone is None:
                self.skin_tone = skin_tone
            else:
                self.skin_tone = 0.9 * self.skin_tone + 0.1 * skin_tone  # Yavaşça güncelle
                
        # Güvenilirlik skorunu güncelle (biraz ağırlıklı ortalama)
        self.reliability_score = 0.8 * self.reliability_score + 0.2 * new_reliability
        
        return True


class PersonTrackerManager:
    """
    Birden fazla kişiyi takip eden ve ID değişimlerini önleyen yönetici sınıf.
    DeepSORT'tan gelen takipleri filtreler ve güvenilirlik skorlarını yönetir.
    """
    def __init__(self, reliability_threshold=0.6):
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
        for track, data in zip(tracks, face_data):
            if not track.is_confirmed():
                continue  # Onaylanmamış track'leri atla
                
            track_id = str(track.track_id)
            seen_track_ids.add(track_id)
            
            # Yüz özellikleri
            embedding = data.get('embedding')
            gender = data.get('gender')
            landmarks = data.get('landmarks')
            hair_color = data.get('hair_color')
            skin_tone = data.get('skin_tone')
            
            # Eğer bu track daha önce görülmemişse, yeni bir tracker oluştur
            if track_id not in self.person_trackers:
                logger.info(f"Yeni kişi takibi başlatılıyor: ID {track_id}")
                self.person_trackers[track_id] = PersonTracker(track_id, gender, embedding)
                reliable_tracks.append(track)
                continue
            
            # Mevcut tracker'ı güncelle
            tracker = self.person_trackers[track_id]
            is_reliable = tracker.update(
                embedding, gender, frame_idx,
                face_landmarks=landmarks,
                hair_color=hair_color,
                skin_tone=skin_tone
            )
            
            # Eğer güvenilirse, track'i listeye ekle
            if is_reliable:
                reliable_tracks.append(track)
            else:
                logger.warning(f"ID {track_id} için güvenilirlik düşük, takip bu frame için filtrelendi")
        
        # Uzun süre görünmeyen track'leri temizle
        self._cleanup_old_trackers(seen_track_ids)
        
        return reliable_tracks
    
    def _cleanup_old_trackers(self, seen_track_ids):
        """Uzun süre görünmeyen takipleri temizler"""
        to_delete = []
        for track_id, tracker in self.person_trackers.items():
            # Bu frame'de görünmeyen track'leri işaretle
            if track_id not in seen_track_ids:
                frames_missing = self.current_frame - tracker.last_seen_frame
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