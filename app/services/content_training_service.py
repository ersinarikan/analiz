import os 
import json 
import logging 
from typing import Dict ,List ,Optional ,Any 
from datetime import datetime ,timedelta 
from flask import current_app 

from app import db 
from app .models .feedback import Feedback 
from app .models .clip_training import CLIPTrainingSession 
from app .services .clip_training_service import ClipTrainingService 

logger =logging .getLogger ('app.content_training_service')

class ContentTrainingService :
    """
    İçerik modeli eğitimi için yardımcı servis
    - Training data analizi
    - Feedback quality kontrolü
    - Training session yönetimi
    - Performance tracking
    """

    def __init__ (self ):
        self .clip_training_service =ClipTrainingService ()
        logger .info ("ContentTrainingService initialized")

    def analyze_training_readiness (self )->Dict [str ,Any ]:
        """Training için hazırlık durumunu analiz et"""
        logger .info ("Training hazırlık analizi başlıyor...")

        try :
            analysis ={
            'feedback_analysis':self ._analyze_feedback_data (),
            'data_quality':self ._check_data_quality (),
            'training_recommendation':{},
            'ready_for_training':False 
            }

            # ERSIN Training önerisi oluştur
            analysis ['training_recommendation']=self ._generate_training_recommendation (analysis )
            analysis ['ready_for_training']=analysis ['training_recommendation']['recommended']

            logger .info (f"✅ Training analizi tamamlandı. Hazır: {analysis ['ready_for_training']}")
            return analysis 

        except Exception as e :
            logger .error (f"Training analizi hatası: {e }")
            return {'error':str (e ),'ready_for_training':False }

    def _analyze_feedback_data (self )->Dict [str ,Any ]:
        """Feedback verilerini analiz et"""
        try :
        # ERSIN Toplam feedback sayısı
            total_feedback =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content'
            ).count ()

            # ERSIN Geçerli feedback (frame_path olan)
            valid_feedback =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content',
            Feedback .frame_path .isnot (None )
            ).all ()

            # ERSIN Kategori dağılımı
            category_stats ={
            'violence':{'high':0 ,'low':0 ,'none':0 },
            'adult_content':{'high':0 ,'low':0 ,'none':0 },
            'harassment':{'high':0 ,'low':0 ,'none':0 },
            'weapon':{'high':0 ,'low':0 ,'none':0 },
            'drug':{'high':0 ,'low':0 ,'none':0 }
            }

            comment_count =0 
            rating_count =0 

            for feedback in valid_feedback :
            # ERSIN Yorum sayısı
                if feedback .comment and feedback .comment .strip ():
                    comment_count +=1 

                    # ERSIN Rating sayısı
                if feedback .rating is not None :
                    rating_count +=1 

                    # ERSIN Kategori analizi
                if feedback .category_feedback :
                    try :
                        categories =feedback .category_feedback 
                        if isinstance (categories ,str ):
                            categories =json .loads (categories )

                        for category ,level in categories .items ():
                            if category in category_stats :
                                if level in category_stats [category ]:
                                    category_stats [category ][level ]+=1 
                                else :
                                    category_stats [category ]['none']+=1 
                    except Exception as e :
                        logger .warning (f"Kategori analiz hatası: {e }")

                        # ERSIN Son 30 gün içindeki feedback
            recent_date =datetime .now ()-timedelta (days =30 )
            recent_feedback =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content',
            Feedback .created_at >=recent_date 
            ).count ()

            return {
            'total_feedback':total_feedback ,
            'valid_feedback':len (valid_feedback ),
            'comment_count':comment_count ,
            'rating_count':rating_count ,
            'recent_feedback_30d':recent_feedback ,
            'category_distribution':category_stats ,
            'data_coverage':{
            'has_comments':comment_count >0 ,
            'has_ratings':rating_count >0 ,
            'has_categories':any (
            sum (cat_data .values ())>0 
            for cat_data in category_stats .values ()
            )
            }
            }

        except Exception as e :
            logger .error (f"Feedback analizi hatası: {e }")
            return {'error':str (e )}

    def _check_data_quality (self )->Dict [str ,Any ]:
        """Veri kalitesini kontrol et"""
        try :
        # ERSIN Geçerli feedback'leri al
            valid_feedbacks =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content',
            Feedback .frame_path .isnot (None )
            ).all ()

            quality_metrics ={
            'total_samples':len (valid_feedbacks ),
            'missing_files':0 ,
            'corrupted_files':0 ,
            'valid_files':0 ,
            'missing_categories':0 ,
            'missing_comments':0 ,
            'quality_score':0.0 
            }

            if not valid_feedbacks :
                return quality_metrics 

            for feedback in valid_feedbacks :
            # ERSIN Dosya kontrolü
                if feedback .frame_path :
                    file_path =os .path .join (current_app .config ['STORAGE_FOLDER'],feedback .frame_path )

                    if not os .path .exists (file_path ):
                        quality_metrics ['missing_files']+=1 
                    else :
                        try :
                        # ERSIN Dosya boyutu kontrolü
                            file_size =os .path .getsize (file_path )
                            if file_size <1024 :# ERSIN 1KB'den küçükse şüpheli
                                quality_metrics ['corrupted_files']+=1 
                            else :
                                quality_metrics ['valid_files']+=1 
                        except :
                            quality_metrics ['corrupted_files']+=1 

                            # ERSIN Kategori kontrolü
                if not feedback .category_feedback :
                    quality_metrics ['missing_categories']+=1 

                    # ERSIN Yorum kontrolü
                if not feedback .comment or not feedback .comment .strip ():
                    quality_metrics ['missing_comments']+=1 

                    # ERSIN Kalite skoru hesapla (0-1 arası)
            total =quality_metrics ['total_samples']
            if total >0 :
                valid_ratio =quality_metrics ['valid_files']/total 
                category_ratio =(total -quality_metrics ['missing_categories'])/total 
                comment_ratio =(total -quality_metrics ['missing_comments'])/total 

                # ERSIN Ağırlıklı ortalama
                quality_metrics ['quality_score']=(
                valid_ratio *0.5 +# ERSIN Dosya geçerliliği %50
                category_ratio *0.3 +# ERSIN Kategori bilgisi %30
                comment_ratio *0.2 # ERSIN Yorum bilgisi %20
                )

            return quality_metrics 

        except Exception as e :
            logger .error (f"Veri kalitesi kontrolü hatası: {e }")
            return {'error':str (e )}

    def _generate_training_recommendation (self ,analysis :Dict [str ,Any ])->Dict [str ,Any ]:
        """Training önerisi oluştur"""
        try :
            feedback_data =analysis .get ('feedback_analysis',{})
            quality_data =analysis .get ('data_quality',{})

            # ERSIN Type checker için recommendation'ı doğru tiplerle initialize et
            recommendation :Dict [str ,Any ]={
            'recommended':False ,# ERSIN bool
            'confidence':0.0 ,# ERSIN float
            'reasons':[],# ERSIN list[str]
            'warnings':[],# ERSIN list[str]
            'suggested_params':{}# ERSIN dict[str, Any]
            }

            valid_samples =feedback_data .get ('valid_feedback',0 )
            quality_score =quality_data .get ('quality_score',0.0 )

            # ERSIN Minimum sample kontrolü
            # ERSIN Type checker için warnings ve reasons list'lerini güvenli şekilde kullan
            warnings_list =recommendation .get ('warnings',[])
            reasons_list =recommendation .get ('reasons',[])
            if isinstance (warnings_list ,list )and isinstance (reasons_list ,list ):
                if valid_samples <10 :
                    warnings_list .append (f"Yetersiz örnek sayısı: {valid_samples } (minimum 10)")
                    recommendation ['confidence']=float (recommendation .get ('confidence',0.0 ))-0.3 
                elif valid_samples <50 :
                    warnings_list .append (f"Az örnek sayısı: {valid_samples } (önerilen 50+)")
                    recommendation ['confidence']=float (recommendation .get ('confidence',0.0 ))-0.1 
                else :
                    reasons_list .append (f"Yeterli örnek sayısı: {valid_samples }")
                    recommendation ['confidence']=float (recommendation .get ('confidence',0.0 ))+0.3 
                recommendation ['warnings']=warnings_list 
                recommendation ['reasons']=reasons_list 

                # ERSIN Kalite kontrolü
            if quality_score <0.5 :
                recommendation ['warnings'].append (f"Düşük veri kalitesi: {quality_score :.2f} (minimum 0.5)")
                recommendation ['confidence']-=0.2 
            elif quality_score <0.7 :
                recommendation ['warnings'].append (f"Orta veri kalitesi: {quality_score :.2f} (önerilen 0.7+)")
                recommendation ['confidence']-=0.1 
            else :
                recommendation ['reasons'].append (f"İyi veri kalitesi: {quality_score :.2f}")
                recommendation ['confidence']+=0.2 

                # ERSIN Kategori dağılımı kontrolü
            category_dist =feedback_data .get ('category_distribution',{})
            balanced_categories =0 
            for _category ,counts in category_dist .items ():
                total_cat =sum (counts .values ())
                if total_cat >=5 :# ERSIN En az 5 örnek
                    balanced_categories +=1 

            if balanced_categories >=3 :
                recommendation ['reasons'].append (f"Dengeli kategori dağılımı: {balanced_categories }/5")
                recommendation ['confidence']+=0.2 
            else :
                recommendation ['warnings'].append (f"Dengesiz kategori dağılımı: {balanced_categories }/5")
                recommendation ['confidence']-=0.1 

                # ERSIN Son aktivite kontrolü
            recent_feedback =feedback_data .get ('recent_feedback_30d',0 )
            if recent_feedback >=5 :
                recommendation ['reasons'].append (f"Son 30 günde aktif feedback: {recent_feedback }")
                recommendation ['confidence']+=0.1 
            else :
                recommendation ['warnings'].append (f"Son 30 günde az feedback: {recent_feedback }")

                # ERSIN Confidence normalize et
            recommendation ['confidence']=max (0.0 ,min (1.0 ,recommendation ['confidence']+0.5 ))

            # ERSIN Karar ver
            recommendation ['recommended']=(
            valid_samples >=10 and 
            quality_score >=0.5 and 
            recommendation ['confidence']>=0.6 
            )

            # ERSIN Training parametreleri öner
            if recommendation ['recommended']:
                recommendation ['suggested_params']=self ._suggest_training_params (valid_samples ,quality_score )

            return recommendation 

        except Exception as e :
            logger .error (f"Training önerisi oluşturma hatası: {e }")
            return {'recommended':False ,'error':str (e )}

    def _suggest_training_params (self ,sample_count :int ,quality_score :float )->Dict [str ,Any ]:
        """Training parametreleri öner"""
        # ERSIN Base parametreler
        params ={
        'epochs':10 ,
        'batch_size':16 ,
        'learning_rate':1e-4 ,
        'patience':3 
        }

        # ERSIN Örnek sayısına göre ayarla
        if sample_count <30 :
            params ['epochs']=15 # ERSIN Daha fazla epoch
            params ['batch_size']=8 # ERSIN Küçük batch
            params ['learning_rate']=5e-5 # ERSIN Düşük LR
        elif sample_count >100 :
            params ['epochs']=8 # ERSIN Daha az epoch
            params ['batch_size']=32 # ERSIN Büyük batch
            params ['learning_rate']=2e-4 # ERSIN Yüksek LR

            # ERSIN Kaliteye göre ayarla
        if quality_score <0.7 :
            params ['patience']=5 # ERSIN Daha sabırlı
            params ['learning_rate']*=0.5 # ERSIN Daha dikkatli

        return params 

    def prepare_training_session (self ,training_params :Optional [Dict [str ,Any ]]=None )->Dict [str ,Any ]:
        """Training session hazırla"""
        logger .info ("Training session hazırlanıyor...")

        try :
        # ERSIN Hazırlık analizi
            readiness =self .analyze_training_readiness ()

            if not readiness .get ('ready_for_training',False ):
                return {
                'success':False ,
                'error':'Sistem training için hazır değil',
                'analysis':readiness 
                }

                # ERSIN Training parametrelerini hazırla
            if training_params is None :
                training_params =readiness ['training_recommendation']['suggested_params']

                # ERSIN Training data hazırla
            training_data =self .clip_training_service .prepare_training_data ()

            if not training_data :
                return {
                'success':False ,
                'error':'Training data hazırlanamadı'
                }

            return {
            'success':True ,
            'training_data':training_data ,
            'training_params':training_params ,
            'readiness_analysis':readiness 
            }

        except Exception as e :
            logger .error (f"Training session hazırlama hatası: {e }")
            return {'success':False ,'error':str (e )}

    def execute_training (self ,training_params :Optional [Dict [str ,Any ]]=None )->Dict [str ,Any ]:
        """Training'i çalıştır"""
        logger .info ("CLIP model training başlatılıyor...")

        try :
        # ERSIN Session hazırla
            session_prep =self .prepare_training_session (training_params )

            if not session_prep ['success']:
                return session_prep 

                # ERSIN Training çalıştır
            training_result =self .clip_training_service .train_model (
            session_prep ['training_data'],
            session_prep ['training_params']
            )

            if training_result ['success']:
            # ERSIN Başarılı training sonrası aktif modeli güncelle
                self ._update_active_model (training_result )

                logger .info ("✅ CLIP model training başarıyla tamamlandı")

                return {
                'success':True ,
                'training_session_id':training_result ['training_session_id'],
                'model_path':training_result ['model_path'],
                'performance':{
                'final_train_loss':training_result ['final_train_loss'],
                'final_val_loss':training_result ['final_val_loss'],
                'epochs_completed':training_result ['epochs_completed']
                },
                'training_data_stats':session_prep ['training_data']
                }
            else :
                return {
                'success':False ,
                'error':f"Training hatası: {training_result .get ('error','Bilinmeyen hata')}"
                }

        except Exception as e :
            logger .error (f"Training execution hatası: {e }")
            return {'success':False ,'error':str (e )}

    def _update_active_model (self ,training_result :Dict [str ,Any ]):
        """Aktif modeli güncelle"""
        try :
        # ERSIN Aktif model path'ini güncelle
            model_path =training_result ['model_path']
            active_model_dir =current_app .config ['OPENCLIP_MODEL_ACTIVE_PATH']

            # ERSIN Aktif model klasörünü oluştur
            os .makedirs (active_model_dir ,exist_ok =True )

            # ERSIN Model dosyasını kopyala
            import shutil 
            active_model_path =os .path .join (active_model_dir ,'open_clip_pytorch_model.bin')
            shutil .copy2 (model_path ,active_model_path )

            logger .info (f"✅ Aktif model güncellendi: {active_model_path }")

        except Exception as e :
            logger .error (f"Aktif model güncelleme hatası: {e }")

    def get_training_history (self ,limit :int =10 )->List [Dict [str ,Any ]]:
        """Training geçmişini getir"""
        try :
            sessions =db .session .query (CLIPTrainingSession ).order_by (
            CLIPTrainingSession .created_at .desc ()
            ).limit (limit ).all ()

            history =[]
            for session in sessions :
                session_data ={
                'id':session .id ,
                'version_name':session .version_name ,
                'feedback_count':session .feedback_count ,
                'status':session .status ,
                'is_active':session .is_active ,
                'is_successful':session .is_successful ,
                'created_at':session .created_at .isoformat ()if session .created_at else None ,
                'training_start':session .training_start .isoformat ()if session .training_start else None ,
                'training_end':session .training_end .isoformat ()if session .training_end else None 
                }

                # ERSIN Performance metrics parse et
                if session .performance_metrics :
                    try :
                        metrics =json .loads (session .performance_metrics )
                        session_data ['performance_metrics']=metrics 
                    except :
                        pass 

                history .append (session_data )

            return history 

        except Exception as e :
            logger .error (f"Training history getirme hatası: {e }")
            return []