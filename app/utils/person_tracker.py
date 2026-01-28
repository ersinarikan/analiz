import numpy as np 
import logging 
from scipy .spatial .distance import cosine 

logger =logging .getLogger (__name__ )

def cosine_similarity (v1 ,v2 ):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar (1 = aynı, 0 = farklı)"""
    return 1 -cosine (v1 ,v2 )

def color_similarity (color1 ,color2 ):
    """İki renk arasındaki benzerliği hesaplar"""
    c1 =np .array (color1 )
    c2 =np .array (color2 )
    return 1.0 -np .linalg .norm (c1 -c2 )/np .sqrt (3 *(255 **2 ))

def landmark_distance (lm1 ,lm2 ):
    """İki yüz landmark seti arasındaki normalize edilmiş mesafeyi hesaplar"""
    if len (lm1 )!=len (lm2 ):
        return 1.0 # ERSIN Farklı boyutlar, büyük mesafe
    return np .mean (np .linalg .norm (np .array (lm1 )-np .array (lm2 ),axis =1 ))

class PersonTracker :
    """
    Kişileri takip etmek ve ID değişimlerini (ID switches) tespit etmek için 
    birden fazla özelliği birleştiren sınıf.
    """
    def __init__ (self ,track_id ,initial_gender ,initial_embedding ,id_change_threshold ,embedding_distance_threshold ):
        self .track_id =track_id 
        self .gender =initial_gender 
        self .face_embeddings =[initial_embedding ]
        self .last_seen_frame =0 
        self .face_landmarks =[]
        self .hair_color =None 
        self .skin_tone =None 
        self .avg_embedding =initial_embedding 
        self .reliability_score =0.5 # ERSIN Başlangıç güvenilirlik skoru
        self .max_landmark_penalty_distance =0.5 # ERSIN Landmark eşleşmesi için maksimum ceza mesafesi
        self .id_change_threshold =id_change_threshold # ERSIN Config'den gelecek
        self .embedding_distance_threshold =embedding_distance_threshold # ERSIN Config'den gelecek

    def update (self ,new_embedding ,new_gender ,frame_idx ,face_landmarks =None ,
    hair_color =None ,skin_tone =None ):
        """
        Kişinin özelliklerini günceller ve mevcut tespitle tutarlılık kontrolü yapar.
        
        Returns:
            bool: Güncellemenin kabul edilip edilmediği
        """
        is_first_update =len (self .face_embeddings )==1 and self .last_seen_frame ==0 

        # ERSIN Cinsiyet tutarlılığı kontrolü
        gender_match_score =1.0 if new_gender ==self .gender else 0.0 

        # ERSIN Embedding mesafesi kontrolü
        embedding_distance =cosine (new_embedding ,self .avg_embedding )
        # ERSIN embedding_match_score: self.embedding_distance_threshold'da 0, 0.0'da 1 olacak şekilde lineer
        # ERSIN Eğer embedding_distance_threshold 0 ise (olmamalı ama), bölme hatasını engelle
        denominator_embedding =self .embedding_distance_threshold if self .embedding_distance_threshold >0 else 0.4 
        embedding_match_score =max (0.0 ,1.0 -(embedding_distance /denominator_embedding ))

        if is_first_update :
            logger .info (f"[PersonTracker][{self .track_id }] İlk güncelleme - embedding_distance={embedding_distance :.4f}, embedding_match_score={embedding_match_score :.4f}, gender_match={gender_match_score :.2f}")

            # ERSIN Saç rengi ve cilt tonu benzerlik skorları
        hair_match_score =0.5 # ERSIN Varsayılan (bilgi yoksa nötr)
        if hair_color is not None and self .hair_color is not None :
            hair_match_score =color_similarity (hair_color ,self .hair_color )
        elif hair_color is not None and self .hair_color is None :# ERSIN İlk defa saç rengi bilgisi geldi
            hair_match_score =0.7 # ERSIN Yeni bilgi için hafif pozitif

        skin_match_score =0.5 
        if skin_tone is not None and self .skin_tone is not None :
            skin_match_score =color_similarity (skin_tone ,self .skin_tone )
        elif skin_tone is not None and self .skin_tone is None :# ERSIN İlk defa cilt tonu bilgisi geldi
            skin_match_score =0.7 # ERSIN Yeni bilgi için hafif pozitif

            # ERSIN Yüz landmark eşleşme skoru
        face_match_score =0.5 # ERSIN Varsayılan (bilgi yoksa veya karşılaştırılamıyorsa nötr)
        new_landmarks_present =face_landmarks is not None and len (face_landmarks )>0 
        stored_landmarks_present =len (self .face_landmarks )>0 

        if new_landmarks_present and stored_landmarks_present :
            distance =landmark_distance (face_landmarks ,self .face_landmarks [-1 ])
            # ERSIN landmark_distance 1.0 döndürürse (nokta sayısı farklı), skor 0 olur.
            face_match_score =max (0.0 ,1.0 -(distance /self .max_landmark_penalty_distance ))
        elif new_landmarks_present and not stored_landmarks_present :
            face_match_score =0.6 # ERSIN Yeni landmark var, saklanan yok: hafif pozitif
        elif not new_landmarks_present and stored_landmarks_present :
            face_match_score =0.3 # ERSIN Yeni landmark yok, saklanan var: cezalandır
            # ERSIN Her iki landmark da yoksa, varsayılan 0.5 kalır.

            # ERSIN Güvenilirlik skorunu güncelle
            # ERSIN Ağırlıklar aynı kalabilir veya landmark için önemi artırılabilir.
        weights ={'gender':0.25 ,'embedding':0.4 ,'hair':0.1 ,'skin':0.1 ,'face':0.15 }

        new_reliability_raw =(
        weights ['gender']*gender_match_score +
        weights ['embedding']*embedding_match_score +
        weights ['hair']*hair_match_score +
        weights ['skin']*skin_match_score +
        weights ['face']*face_match_score 
        )

        # ERSIN Eğer ham güvenilirlik skoru düşükse, bu kişi farklı olabilir
        if new_reliability_raw <self .id_change_threshold :# ERSIN Dinamik eşik kullan
            logger .warning (f"[PersonTracker][{self .track_id }] Olası ID switch (eşik {self .id_change_threshold }): Ham Güvenilirlik: {new_reliability_raw :.2f} (Detaylar: G:{gender_match_score :.2f} E:{embedding_match_score :.2f} H:{hair_match_score :.2f} S:{skin_match_score :.2f} F:{face_match_score :.2f})")
            return False # ERSIN ID Switch olabilir, güncellemeyi reddet

            # ERSIN Güncelleme yap
        self .face_embeddings .append (new_embedding )
        if len (self .face_embeddings )>10 :# ERSIN Sadece son 10 embedding'i tut
            self .face_embeddings .pop (0 )

            # ERSIN Ortalama embedding'i güncelle
        self .avg_embedding =np .mean (self .face_embeddings ,axis =0 )
        self .last_seen_frame =frame_idx 

        if face_landmarks is not None :
            self .face_landmarks .append (face_landmarks )
            if len (self .face_landmarks )>5 :# ERSIN Son 5 landmark set'ini tut
                self .face_landmarks .pop (0 )

        if hair_color is not None :
            if self .hair_color is None :
                self .hair_color =np .array (hair_color )if isinstance (hair_color ,(list ,tuple ))else hair_color 
            else :
            # ERSIN Renklerin numpy array olduğundan emin ol
                current_hair_color_arr =np .array (self .hair_color )
                new_hair_color_arr =np .array (hair_color )
                self .hair_color =0.9 *current_hair_color_arr +0.1 *new_hair_color_arr # ERSIN Yavaşça güncelle

        if skin_tone is not None :
            if self .skin_tone is None :
                self .skin_tone =np .array (skin_tone )if isinstance (skin_tone ,(list ,tuple ))else skin_tone 
            else :
                current_skin_tone_arr =np .array (self .skin_tone )
                new_skin_tone_arr =np .array (skin_tone )
                self .skin_tone =0.9 *current_skin_tone_arr +0.1 *new_skin_tone_arr # ERSIN Yavaşça güncelle

                # ERSIN Güvenilirlik skorunu güncelle (biraz ağırlıklı ortalama)
        old_reliability =self .reliability_score 
        self .reliability_score =0.8 *self .reliability_score +0.2 *new_reliability_raw 

        if is_first_update :
            logger .info (f"[PersonTracker][{self .track_id }] İlk güncelleme tamamlandı - reliability_score: {old_reliability :.2f} -> {self .reliability_score :.2f} (new_raw={new_reliability_raw :.2f})")

        return True 


class PersonTrackerManager :
    """
    Birden fazla kişiyi takip eden ve ID değişimlerini önleyen yönetici sınıf.
    DeepSORT'tan gelen takipleri filtreler ve güvenilirlik skorlarını yönetir.
    """
    def __init__ (self ,reliability_threshold ,max_frames_missing ,id_change_threshold ,embedding_distance_threshold ):
        self .person_trackers ={}# ERSIN track_id -> PersonTracker
        self .reliability_threshold =reliability_threshold # ERSIN Parametreden al
        self .max_frames_missing =max_frames_missing # ERSIN Parametreden al
        self .id_change_threshold =id_change_threshold # ERSIN PersonTracker'a iletilecek
        self .embedding_distance_threshold =embedding_distance_threshold # ERSIN PersonTracker'a iletilecek
        self .warmup_frames =3 # ERSIN İlk birkaç frame için is_confirmed kontrolünü atla
        self .current_frame =0 

    def update (self ,tracks ,face_data ,frame_idx ):
        """
        DeepSORT takiplerini işler ve güvenilir takipleri döndürür
        
        Args:
            tracks: DeepSORT'tan gelen takip listesi
            face_data: Her yüz için {embedding, gender, landmarks, hair_color, skin_tone} içeren sözlük 
            frame_idx: Mevcut frame indeksi
            
        Returns:
            list: Güvenilir takipler listesi
        """
        self .current_frame =frame_idx 

        # ERSIN Sonuçları saklayacak liste
        reliable_tracks =[]

        # ERSIN Mevcut frame'de görülen track ID'leri
        seen_track_ids =set ()

        # ERSIN Her takibi işle
        logger .info (f"[PersonTrackerManager] Frame {frame_idx }: {len (tracks )} track, {len (face_data )} face_data işleniyor")
        for i ,(track_obj ,data )in enumerate (zip (tracks ,face_data )):# ERSIN track_obj ve i eklendi
        # ERSIN İlk birkaç frame için is_confirmed kontrolünü atla (DeepSORT n_init=2 için)
            skip_confirmed_check =frame_idx <self .warmup_frames 
            if not skip_confirmed_check and (not hasattr (track_obj ,'is_confirmed')or not track_obj .is_confirmed ()):
                logger .info (f"[PersonTrackerManager] Track {i } (ID: {getattr (track_obj ,'track_id','unknown')}) onaylanmamış, atlanıyor.")
                continue # ERSIN Onaylanmamış track'leri atla
            elif skip_confirmed_check and (not hasattr (track_obj ,'is_confirmed')or not track_obj .is_confirmed ()):
                logger .info (f"[PersonTrackerManager] Frame {frame_idx } warmup aşamasında - Track {i } (ID: {getattr (track_obj ,'track_id','unknown')}) onaylanmamış ama işleniyor (warmup)")

            track_id =str (track_obj .track_id )
            seen_track_ids .add (track_id )

            # ERSIN Yüz özellikleri
            embedding =data .get ('embedding')
            gender =data .get ('gender')
            landmarks =data .get ('landmarks')
            hair_color =data .get ('hair_color')
            skin_tone =data .get ('skin_tone')

            if embedding is None or gender is None :
                logger .warning (f"[PersonTrackerManager] Track ID {track_id } için embedding veya gender eksik (embedding={embedding is not None }, gender={gender }), bu track atlanıyor.")
                continue 

                # ERSIN Eğer bu track daha önce görülmemişse, yeni bir tracker oluştur
            is_new_track =track_id not in self .person_trackers 
            if is_new_track :
                logger .info (f"[PersonTrackerManager] Yeni kişi takibi başlatılıyor: ID {track_id }, gender={gender }, embedding_shape={embedding .shape if embedding is not None else None }")
                logger .info (f"[PersonTrackerManager] Threshold değerleri: reliability={self .reliability_threshold }, id_change={self .id_change_threshold }, embedding_dist={self .embedding_distance_threshold }")
                self .person_trackers [track_id ]=PersonTracker (track_id ,gender ,embedding ,self .id_change_threshold ,self .embedding_distance_threshold )
                logger .info (f"[PersonTrackerManager] Yeni tracker oluşturuldu: ID {track_id }, başlangıç reliability_score={self .person_trackers [track_id ].reliability_score :.2f}")

                # ERSIN Mevcut tracker'ı güncelle (veya yeni oluşturulanı)
            person_tracker_instance =self .person_trackers [track_id ]
            is_update_accepted =person_tracker_instance .update (
            embedding ,gender ,frame_idx ,
            face_landmarks =landmarks ,
            hair_color =hair_color ,
            skin_tone =skin_tone 
            )

            # ERSIN Eğer güncelleme kabul edildiyse ve genel güvenilirlik yeterliyse, track'i listeye ekle
            if is_update_accepted and person_tracker_instance .reliability_score >=self .reliability_threshold :
                reliable_tracks .append (track_obj )
                if is_new_track :
                    logger .info (f"[PersonTrackerManager] ✅ Yeni takip ID {track_id } güvenilir olarak eklendi (reliability_score={person_tracker_instance .reliability_score :.2f} >= {self .reliability_threshold })")
            else :
                if not is_update_accepted :
                    logger .warning (f"[PersonTrackerManager] ❌ ID {track_id } için güncelleme reddedildi (olası ID switch), takip bu frame için filtrelendi")
                else :# ERSIN reliability_score < self.reliability_threshold
                    logger .warning (f"[PersonTrackerManager] ❌ ID {track_id } için güvenilirlik düşük ({person_tracker_instance .reliability_score :.2f} < {self .reliability_threshold }), takip bu frame için filtrelendi")

                    # ERSIN Uzun süre görünmeyen track'leri temizle
        self ._cleanup_old_trackers (seen_track_ids )

        logger .info (f"[PersonTrackerManager] Frame {frame_idx }: {len (reliable_tracks )}/{len (tracks )} track güvenilir olarak döndürüldü")
        return reliable_tracks 

    def _cleanup_old_trackers (self ,seen_track_ids ):
        """Uzun süre görünmeyen takipleri temizler"""
        to_delete =[]
        for track_id ,tracker_instance in self .person_trackers .items ():# ERSIN Değişken adı düzeltildi
        # ERSIN Bu frame'de görünmeyen track'leri işaretle
            if track_id not in seen_track_ids :
                frames_missing =self .current_frame -tracker_instance .last_seen_frame # ERSIN Değişken adı düzeltildi
                if frames_missing >self .max_frames_missing :
                    to_delete .append (track_id )

                    # ERSIN Eski takipleri sil
        for track_id in to_delete :
            logger .info (f"Eski takip siliniyor: ID {track_id }, son görülme: {self .person_trackers [track_id ].last_seen_frame }, kayıp frame sayısı: {self .current_frame -self .person_trackers [track_id ].last_seen_frame } > {self .max_frames_missing }")
            del self .person_trackers [track_id ]

    def get_reliability_score (self ,track_id ):
        """Belirli bir track ID'si için güvenilirlik skorunu döndürür"""
        if track_id in self .person_trackers :
            return self .person_trackers [track_id ].reliability_score 
        return 0.0 