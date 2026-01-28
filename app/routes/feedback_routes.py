from flask import Blueprint ,request ,jsonify 
from app .models .feedback import Feedback 
from app .models .analysis import AgeEstimation 
from app import db 
import logging 
from app .utils .path_utils import to_rel_path 

feedback_bp =Blueprint ('feedback',__name__ ,url_prefix ='/api/feedback')
"""
Geri bildirim işlemleri için blueprint.
- Kullanıcı geri bildirimi gönderme ve yönetme endpointlerini içerir.
"""
logger =logging .getLogger (__name__ )

@feedback_bp .route ('/submit',methods =['POST'])
def submit_feedback ():
    """
    İçerik analizi için geri bildirim gönderir (her kategori için ayrı kayıt).
    Post Body:
        {
            "content_id": "uuid",
            "analysis_id": "uuid",
            "category": "violence|weapon|...",
            "feedback": "over_estimated|accurate|...",
            "frame_path": "string"
        }
    """
    try :
        data =request .json 
        if data is None :
            return jsonify ({'error':'Request body is required'}),400 
            # ERSIN Zorunlu alanlar
        required_fields =['content_id','analysis_id','category','feedback','frame_path']
        for field in required_fields :
            if field not in data or not data [field ]:
                return jsonify ({'error':f'{field } alanı gereklidir ve boş olamaz.'}),400 
        feedback =Feedback (
        content_id =data ['content_id'],
        analysis_id =data ['analysis_id'],
        frame_path =to_rel_path (data .get ('frame_path')),
        feedback_type ='content',
        feedback_source ='MANUAL_USER_CONTENT_CORRECTION',
        category_feedback ={data ['category']:data ['feedback']},
        # ERSIN Eski toplu category_feedback yerine sadece ilgili kategori ve feedback
        )
        db .session .add (feedback )
        db .session .commit ()
        logger .info (f"İçerik geri bildirimi kaydedildi, ID: {feedback .id }, kategori: {data ['category']}, analiz ID: {data ['analysis_id']}")
        return jsonify ({
        'success':True ,
        'feedback_id':feedback .id ,
        'message':'Geri bildirim başarıyla kaydedildi'
        }),201 
    except Exception as e :
        db .session .rollback ()
        logger .error (f"Geri bildirim kaydedilirken hata: {str (e )}")
        return jsonify ({'error':f'Geri bildirim kaydedilemedi: {str (e )}'}),500 

@feedback_bp .route ('/age',methods =['POST'])
def submit_age_feedback ():
    """
    Yaş tahmini için geri bildirim gönderir.
    
    Bu fonksiyon, kullanıcıdan gelen yaş tahmini geri bildirimlerini işler ve veritabanına kaydeder.
    Yaş geri bildirimleri, yaş tahmin modelinin eğitimi için kullanılabilir.
    
    Post Body:
        {
            "person_id": "uuid", 
            "corrected_age": int,
            "is_age_range_correct": bool,
            "analysis_id": "uuid",
            "frame_path": "string"
        }
    """
    try :
        data =request .json 
        if data is None or not isinstance (data ,dict ):
            return jsonify ({'error':'Request body is required'}),400 

            # ERSIN Gerekli alanları kontrol et
        required_fields =['person_id','corrected_age','analysis_id','frame_path']
        for field in required_fields :
            if field not in data or data [field ]is None :
                return jsonify ({'error':f'Geçersiz istek formatı. {field } alanı gereklidir ve boş olamaz.'}),400 

        person_id =data ['person_id']
        corrected_age =data ['corrected_age']
        is_age_range_correct =data .get ('is_age_range_correct',False )
        analysis_id =data ['analysis_id']
        frame_path =to_rel_path (data ['frame_path'])

        # ERSIN Embedding'i AgeEstimation tablosundan bul
        embedding_str =None 
        try :
        # ERSIN confidence_score sütunu ile sıralama
            age_est =AgeEstimation .query .filter_by (analysis_id =analysis_id ,person_id =person_id ).order_by (AgeEstimation .confidence_score .desc ()).first ()
            if age_est and hasattr (age_est ,'embedding')and age_est .embedding :
                emb =age_est .embedding 
                if isinstance (emb ,str ):
                    embedding_str =emb 
                elif hasattr (emb ,'tolist'):
                    embedding_str =",".join (str (float (x ))for x in emb .tolist ())
                elif isinstance (emb ,(list ,tuple )):
                    embedding_str =",".join (str (float (x ))for x in emb )
                else :
                    embedding_str =str (emb )
        except Exception as emb_err :
            logger .warning (f"Embedding alınırken hata: {str (emb_err )}")

        feedback =Feedback (
        person_id =person_id ,
        analysis_id =analysis_id ,
        corrected_age =corrected_age ,
        is_age_range_correct =is_age_range_correct ,
        feedback_type ='age',
        feedback_source ='MANUAL_USER_AGE_CORRECTION',
        frame_path =frame_path ,
        embedding =embedding_str 
        )

        db .session .add (feedback )
        db .session .commit ()

        logger .info (f"Yaş geri bildirimi kaydedildi, ID: {feedback .id }, kişi ID: {person_id }, analiz ID: {analysis_id }, düzeltilmiş yaş: {corrected_age }")

        return jsonify ({
        'success':True ,
        'feedback_id':feedback .id ,
        'message':'Yaş geri bildirimi başarıyla kaydedildi'
        }),201 

    except Exception as e :
        db .session .rollback ()
        logger .error (f"Yaş geri bildirimi kaydedilirken hata: {str (e )}")
        return jsonify ({'error':f'Yaş geri bildirimi kaydedilemedi: {str (e )}'}),500 

@feedback_bp .route ('/content/<content_id>',methods =['GET'])
def get_content_feedback (content_id ):
    """
    Belirli bir içerik için geri bildirimleri getirir.
    
    Args:
        content_id: İçerik ID'si
        
    Returns:
        JSON: Geri bildirim listesi
    """
    try :
    # ERSIN İçerik için tüm geri bildirimleri bul
        feedbacks =Feedback .query .filter_by (content_id =content_id ).all ()

        return jsonify ([feedback .to_dict ()for feedback in feedbacks ]),200 

    except Exception as e :
        logger .error (f"Geri bildirim getirme hatası: {str (e )}")
        return jsonify ({'error':f'Geri bildirimler getirilirken bir hata oluştu: {str (e )}'}),500 

@feedback_bp .route ('/content/recent',methods =['GET'])
def get_recent_content_feedback ():
    """
    Son içerik analizi geri bildirimlerini ve kategori dağılımını döndürür.
    """
    try :
    # ERSIN Son 10 geri bildirimi çek
        feedbacks =Feedback .query .filter_by (feedback_type ='content').order_by (Feedback .created_at .desc ()).limit (10 ).all ()
        feedback_list =[f .to_dict ()for f in feedbacks ]

        # ERSIN Kategori dağılımı
        from collections import Counter 
        category_counter =Counter ()
        for f in feedbacks :
            if f .category_feedback :
                for k ,v in f .category_feedback .items ():
                    category_counter [k +':'+str (v )]+=1 

        return jsonify ({
        'recent_feedbacks':feedback_list ,
        'category_distribution':dict (category_counter )
        }),200 
    except Exception as e :
        return jsonify ({'error':f'Geri bildirimler getirilirken hata: {str (e )}'}),500 

bp =feedback_bp 