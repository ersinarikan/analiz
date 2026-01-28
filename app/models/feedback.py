from datetime import datetime 
from typing import Any
from app import db 
from sqlalchemy .dialects .postgresql import JSON 
from sqlalchemy import Column ,Integer ,String ,Float ,DateTime ,Boolean ,Text 

class Feedback (db .Model ):
    """
    Kullanıcı geri bildirimi modeli.
    - Analiz sonuçlarına verilen yaş, içerik, doğruluk gibi geri bildirimleri tutar.
    """
    __tablename__ ='feedback'

    id =db .Column (db .Integer ,primary_key =True )
    created_at =db .Column (db .DateTime ,default =datetime .now )

    # ERSIN Görüntü ve Yüz Bilgileri
    frame_path =db .Column (db .String (1024 ),nullable =True )
    face_bbox =db .Column (db .String (255 ),nullable =True )# ERSIN "x1,y1,x2,y2" veya JSON
    embedding =db .Column (db .Text ,nullable =True )# ERSIN Artık virgül ile ayrılmış float string olarak saklanacak

    # ERSIN İçerik ve Analiz ID'leri
    content_id =db .Column (db .String (36 ),nullable =True )
    analysis_id =db .Column (db .String (36 ),nullable =True )

    # ERSIN Kişi ID'si (yaş geri bildirimi için)
    person_id =db .Column (db .String (36 ),nullable =True ,index =True )

    # ERSIN Yaş Geri Bildirimleri
    corrected_age =db .Column (db .Integer ,nullable =True )# ERSIN Kullanıcının girdiği yaş
    pseudo_label_original_age =db .Column (db .Float ,nullable =True )# ERSIN BuffaloL'nin sözde etiket yaşı
    pseudo_label_clip_confidence =db .Column (db .Float ,nullable =True )# ERSIN BuffaloL sözde etiketinin CLIP güveni
    is_age_range_correct =db .Column (db .Boolean ,nullable =True )# ERSIN default=False kaldırıldı

    # ERSIN Geri Bildirim Türü ve Kaynağı
    feedback_type =db .Column (db .String (50 ),nullable =True ,index =True )# ERSIN örn: 'age', 'content', 'general'
    feedback_source =db .Column (db .String (50 ),nullable =True ,default ='MANUAL_USER',index =True )
    # ERSIN örn: 'MANUAL_USER_AGE_CORRECTION', 'PSEUDO_BUFFALO_HIGH_CONF'

    # ERSIN Genel Derecelendirme ve Yorum
    rating =db .Column (db .Integer ,nullable =True )
    comment =db .Column (db .Text ,nullable =True )

    # ERSIN Kategori Bazlı Geri Bildirimler (JSON)
    category_feedback =db .Column (db .JSON ,nullable =True )# ERSIN PostgreSQL için JSON, diğer DB'ler için db.Text veya db.String
    category_correct_values =db .Column (db .JSON ,nullable =True )# ERSIN PostgreSQL için JSON, diğer DB'ler için db.Text veya db.String

    # ERSIN Eğitim Durumu Alanları
    training_status =db .Column (db .String (50 ),nullable =True ,index =True )# ERSIN 'used_in_training', 'archived', vb.
    used_in_model_version =db .Column (db .String (100 ),nullable =True )# ERSIN Hangi model versiyonunda kullanıldı
    training_used_at =db .Column (db .DateTime ,nullable =True )# ERSIN Ne zaman eğitimde kullanıldı
    is_archived =db .Column (db .Boolean ,default =False ,index =True )# ERSIN Arşivlenmiş mi
    archive_reason =db .Column (db .String (100 ),nullable =True )# ERSIN Arşivleme nedeni

    # ERSIN Ensemble Kullanım Takibi
    used_in_ensemble =db .Column (db .Boolean ,default =False ,index =True )# ERSIN Ensemble'da kullanılmış mı
    ensemble_usage_count =db .Column (db .Integer ,default =0 )# ERSIN Kaç kez ensemble'da kullanıldı
    last_used_at =db .Column (db .DateTime ,nullable =True )# ERSIN Son kullanım zamanı
    ensemble_model_versions =db .Column (db .JSON ,nullable =True )# ERSIN Hangi ensemble versiyonlarında kullanıldı

    def __repr__ (self ):
        return f"<Feedback(id={self .id }, type='{self .feedback_type }', source='{self .feedback_source }')>"

    def to_dict (self )->dict [str ,Any ]:
        data ={
        'id':self .id ,
        'created_at':self .created_at .isoformat ()if self .created_at else None ,
        'frame_path':self .frame_path ,
        'face_bbox':self .face_bbox ,
        # ERSIN embedding'i to_dict'e eklemek genellikle iyi bir fikir değil, büyük olabilir.
        'content_id':self .content_id ,
        'analysis_id':self .analysis_id ,
        'person_id':self .person_id ,
        'corrected_age':self .corrected_age ,
        'pseudo_label_original_age':self .pseudo_label_original_age ,
        'pseudo_label_clip_confidence':self .pseudo_label_clip_confidence ,
        'is_age_range_correct':self .is_age_range_correct ,
        'feedback_type':self .feedback_type ,
        'feedback_source':self .feedback_source ,
        'rating':self .rating ,
        'comment':self .comment ,
        'category_feedback':self .category_feedback ,
        'category_correct_values':self .category_correct_values ,
        'training_status':self .training_status ,
        'used_in_model_version':self .used_in_model_version ,
        'training_used_at':self .training_used_at .isoformat ()if self .training_used_at else None ,
        'is_archived':self .is_archived ,
        'archive_reason':self .archive_reason ,
        'used_in_ensemble':self .used_in_ensemble ,
        'ensemble_usage_count':self .ensemble_usage_count ,
        'last_used_at':self .last_used_at .isoformat ()if self .last_used_at else None ,
        'ensemble_model_versions':self .ensemble_model_versions 
        }
        return data 