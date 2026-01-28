import os 
import json 
import logging 
import torch 
import torch .nn as nn 
# ERSIN optim şu an kullanılmıyor 
from torch .utils .data import Dataset ,DataLoader 
import open_clip 
import PIL .Image as Image 
from datetime import datetime 
from typing import List ,Dict ,Tuple ,Optional ,Any 
from flask import current_app 
from sklearn .model_selection import train_test_split 

from app import db 
from app .models .feedback import Feedback 
from app .models .clip_training import CLIPTrainingSession 

logger =logging .getLogger ('app.clip_training_service')

class ContentDataset (Dataset [Any ]):
    """CLIP fine-tuning için veri seti"""

    def __init__ (self ,image_paths :List [str ],captions :List [str ],labels :List [Dict [str ,Any ]],preprocess ):
        self .image_paths =image_paths 
        self .captions =captions 
        self .labels =labels 
        self .preprocess =preprocess 

    def __len__ (self ):
        return len (self .image_paths )

    def __getitem__ (self ,idx ):
    # ERSIN Görüntüyü yükle
        try :
            image =Image .open (self .image_paths [idx ]).convert ('RGB')
            image =self .preprocess (image )
        except Exception as e :
            logger .warning (f"Görüntü yüklenemedi {self .image_paths [idx ]}: {e }")
            # ERSIN Boş görüntü oluştur
            image =torch .zeros (3 ,224 ,224 )

        caption =self .captions [idx ]
        labels =self .labels [idx ]

        return image ,caption ,labels 

class ClipTrainingService :
    """
    OpenCLIP modeli için fine-tuning servisi
    - Feedback verilerinden training data hazırlar
    - Contrastive learning ile model eğitir
    - Classification head ekler
    - Model versiyonlarını yönetir
    """

    def __init__ (self ):
        self .device =torch .device ("cuda"if torch .cuda .is_available ()and current_app .config .get ('USE_GPU',True )else "cpu")
        self .model =None 
        self .tokenizer =None 
        self .preprocess =None 
        self .classification_head =None 
        self .best_model_state :dict [str ,Any ]|None =None # ERSIN Initialize best_model_state

        logger .info (f"ClipTrainingService initialized on device: {self .device }")

    def load_base_model (self ):
        """Base OpenCLIP modelini yükle"""
        try :
            logger .info ("Base OpenCLIP modeli yükleniyor...")

            # ERSIN Base model yükle
            model ,_ ,preprocess =open_clip .create_model_and_transforms (
            'ViT-H-14-378-quickgelu',
            pretrained ='dfn5b',
            device =self .device 
            )

            # ERSIN Tokenizer yükle
            tokenizer =open_clip .get_tokenizer ('ViT-H-14-378-quickgelu')

            self .model =model 
            self .tokenizer =tokenizer 
            self .preprocess =preprocess 

            logger .info ("✅ Base OpenCLIP modeli başarıyla yüklendi")
            return True 

        except Exception as e :
            logger .error (f"Base model yükleme hatası: {e }")
            return False 

    def prepare_training_data (self ,min_samples :int =10 )->Optional [Dict [str ,Any ]]:
        """Feedback verilerinden training data hazırla"""
        logger .info (f"CLIP training verisi hazırlanıyor (min: {min_samples } örnek)...")

        try :
        # ERSIN Content feedback'lerini al
            feedbacks =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content',
            Feedback .frame_path .isnot (None )
            ).all ()

            if len (feedbacks )<min_samples :
                logger .warning (f"Yetersiz feedback: {len (feedbacks )} < {min_samples }")
                return None 

                # ERSIN Training data listeler
            image_paths =[]
            positive_captions =[]
            negative_captions =[]
            labels =[]

            for feedback in feedbacks :
            # ERSIN Frame path'i tam yola çevir
                if feedback .frame_path :
                    frame_path =os .path .join (current_app .config ['STORAGE_FOLDER'],feedback .frame_path )

                    if os .path .exists (frame_path ):
                        image_paths .append (frame_path )

                        # ERSIN Kullanıcı yorumundan pozitif caption oluştur
                        positive_caption =self ._create_positive_caption (feedback )
                        positive_captions .append (positive_caption )

                        # ERSIN Ters caption oluştur (contrastive learning için)
                        negative_caption =self ._create_negative_caption (feedback )
                        negative_captions .append (negative_caption )

                        # ERSIN Label bilgileri
                        label_info =self ._extract_labels (feedback )
                        labels .append (label_info )

            if len (image_paths )<min_samples :
                logger .warning (f"Geçerli görüntü sayısı yetersiz: {len (image_paths )} < {min_samples }")
                return None 

                # ERSIN Training ve validation'a ayır
            train_indices ,val_indices =train_test_split (
            range (len (image_paths )),
            test_size =0.2 ,
            random_state =42 
            )

            training_data ={
            'train_images':[image_paths [i ]for i in train_indices ],
            'train_positive_captions':[positive_captions [i ]for i in train_indices ],
            'train_negative_captions':[negative_captions [i ]for i in train_indices ],
            'train_labels':[labels [i ]for i in train_indices ],

            'val_images':[image_paths [i ]for i in val_indices ],
            'val_positive_captions':[positive_captions [i ]for i in val_indices ],
            'val_negative_captions':[negative_captions [i ]for i in val_indices ],
            'val_labels':[labels [i ]for i in val_indices ],

            'total_samples':len (image_paths ),
            'train_samples':len (train_indices ),
            'val_samples':len (val_indices )
            }

            logger .info (f"✅ Training data hazırlandı: {training_data ['total_samples']} örnek")
            logger .info (f"   Train: {training_data ['train_samples']}, Val: {training_data ['val_samples']}")

            return training_data 

        except Exception as e :
            logger .error (f"Training data hazırlama hatası: {e }")
            return None 

    def _create_positive_caption (self ,feedback :Feedback )->str :
        """Feedback'ten pozitif caption oluştur"""
        try :
        # ERSIN Kullanıcı yorumu varsa onu kullan
            if feedback .comment and feedback .comment .strip ():
                return feedback .comment .strip ()

                # ERSIN Category feedback'ten caption oluştur
            if feedback .category_feedback :
                category_data =feedback .category_feedback 
                if isinstance (category_data ,str ):
                    category_data =json .loads (category_data )

                return self ._generate_caption_from_categories (category_data ,positive =True )

                # ERSIN Default safe caption
            return "safe appropriate content"

        except Exception as e :
            logger .warning (f"Pozitif caption oluşturma hatası: {e }")
            return "appropriate content"

    def _create_negative_caption (self ,feedback :Feedback )->str :
        """Contrastive learning için negatif caption oluştur"""
        try :
            if feedback .category_feedback :
                category_data =feedback .category_feedback 
                if isinstance (category_data ,str ):
                    category_data =json .loads (category_data )

                return self ._generate_caption_from_categories (category_data ,positive =False )

                # ERSIN Default negative caption
            return "inappropriate violent adult content"

        except Exception as e :
            logger .warning (f"Negatif caption oluşturma hatası: {e }")
            return "inappropriate content"

    def _generate_caption_from_categories (self ,categories :Dict [str ,Any ],positive :bool =True )->str :
        """Kategori feedback'ten caption oluştur"""
        safe_terms =[]
        unsafe_terms =[]

        category_mappings ={
        'violence':('peaceful non-violent content','violent aggressive content'),
        'adult_content':('family-friendly appropriate content','adult explicit content'),
        'harassment':('respectful positive interaction','harassment bullying content'),
        'weapon':('safe environment without weapons','dangerous weapons present'),
        'drug':('drug-free healthy environment','drug substance abuse content')
        }

        for category ,level in categories .items ():
            if category in category_mappings :
                safe_desc ,unsafe_desc =category_mappings [category ]

                if level =='low':
                    safe_terms .append (safe_desc )
                elif level =='high':
                    unsafe_terms .append (unsafe_desc )

        if positive :
        # ERSIN Pozitif için safe terms kullan, yoksa genel safe
            return ', '.join (safe_terms )if safe_terms else "safe appropriate content"
        else :
        # ERSIN Negatif için unsafe terms kullan, yoksa genel unsafe
            return ', '.join (unsafe_terms )if unsafe_terms else "inappropriate harmful content"

    def _extract_labels (self ,feedback :Feedback )->Dict [str ,Any ]:
        """Feedback'ten label bilgilerini çıkart (frontend ile tam uyumlu)"""
        labels ={
        'violence':0.5 ,
        'adult_content':0.5 ,
        'harassment':0.5 ,
        'weapon':0.5 ,
        'drug':0.5 ,
        'safe':1.0 # ERSIN Default safe
        }
        try :
            if feedback .category_feedback :
                category_data =feedback .category_feedback 
                if isinstance (category_data ,str ):
                    category_data =json .loads (category_data )
                for category ,value in category_data .items ():
                    if category in labels :
                    # ERSIN Modern frontend değerleri
                        if value =='accurate':
                            labels [category ]=0.5 
                        elif value =='false_positive':
                            labels [category ]=0.0 
                            labels ['safe']=0.0 
                        elif value =='false_negative':
                            labels [category ]=1.0 
                            labels ['safe']=0.0 
                        elif value =='over_estimated':
                            labels [category ]=0.3 
                            labels ['safe']=0.0 
                        elif value =='under_estimated':
                            labels [category ]=0.8 
                            labels ['safe']=0.0 
                            # ERSIN Eski backend değerleri (geriye dönük uyumluluk)
                        elif value =='high':
                            labels [category ]=1.0 
                            labels ['safe']=0.0 
                        elif value =='low':
                            labels [category ]=0.0 
        except Exception as e :
            logger .warning (f"Label extraction hatası: {e }")
        return labels 

    def create_classification_head (self ,num_classes :int =6 )->nn .Module :
        """CLIP için classification head oluştur"""
        if self .model is None :
            raise ValueError ("Model yüklenmemiş. Önce load_base_model() çağırın.")
            # ERSIN CLIP'in text encoder çıkış boyutu (ViT-H-14 için 1024)
            # ERSIN Use getattr to safely access nested attributes
        text_attr =getattr (self .model ,'text',None )
        if text_attr is None :
            raise RuntimeError ("CLIP model text encoder not available")
        text_projection =getattr (text_attr ,'text_projection',None )
        if text_projection is None :
            raise RuntimeError ("CLIP model text_projection not available")
        out_features_attr =getattr (text_projection ,'out_features',None )
        clip_dim =out_features_attr if out_features_attr is not None else 1024 

        classification_head =nn .Sequential (
        nn .Linear (clip_dim ,512 ),
        nn .ReLU (),
        nn .Dropout (0.3 ),
        nn .Linear (512 ,256 ),
        nn .ReLU (),
        nn .Dropout (0.2 ),
        nn .Linear (256 ,num_classes ),
        nn .Sigmoid ()# ERSIN Multi-label classification için
        ).to (self .device )

        return classification_head 

    def train_model (self ,training_data :Dict [str ,Any ],training_params :Dict [str ,Any ])->Dict [str ,Any ]:
        """CLIP modelini fine-tune et"""
        logger .info ("CLIP fine-tuning başlıyor...")

        try :
        # ERSIN Base model yükle
            if not self .load_base_model ():
                raise Exception ("Base model yüklenemedi")

                # ERSIN Classification head oluştur
            self .classification_head =self .create_classification_head ()

            # ERSIN Data loaders oluştur
            train_dataset =ContentDataset (
            training_data ['train_images'],
            training_data ['train_positive_captions'],
            training_data ['train_labels'],
            self .preprocess 
            )

            val_dataset =ContentDataset (
            training_data ['val_images'],
            training_data ['val_positive_captions'],
            training_data ['val_labels'],
            self .preprocess 
            )

            train_loader =DataLoader (
            train_dataset ,
            batch_size =training_params .get ('batch_size',16 ),
            shuffle =True ,
            num_workers =0 # ERSIN Windows için 0
            )

            val_loader =DataLoader (
            val_dataset ,
            batch_size =training_params .get ('batch_size',16 ),
            shuffle =False ,
            num_workers =0 
            )

            # ERSIN Optimizer - sadece classification head'i eğit (CLIP frozen)
            if self .classification_head is None :
                raise ValueError ("Classification head yüklenmemiş. Önce create_classification_head() çağırın.")
                # ERSIN Import Adam from the correct module for type checking
            from torch .optim .adam import Adam 
            optimizer =Adam (
            self .classification_head .parameters (),
            lr =training_params .get ('learning_rate',1e-4 )
            )

            # ERSIN Loss function
            criterion =nn .BCELoss ()

            # ERSIN Training loop
            history =self ._training_loop (
            train_loader ,val_loader ,optimizer ,criterion ,training_params 
            )

            # ERSIN Model kaydet
            model_path =self ._save_trained_model (training_params )

            # ERSIN Training session kaydet
            training_session =self ._save_training_session (training_data ,training_params ,history ,model_path )

            result ={
            'success':True ,
            'training_session_id':training_session .id ,
            'model_path':model_path ,
            'history':history ,
            'final_train_loss':history ['train_loss'][-1 ],
            'final_val_loss':history ['val_loss'][-1 ],
            'epochs_completed':len (history ['train_loss'])
            }

            logger .info ("✅ CLIP fine-tuning tamamlandı")
            return result 

        except Exception as e :
            logger .error (f"CLIP training hatası: {e }")
            return {'success':False ,'error':str (e )}

    def _training_loop (self ,train_loader ,val_loader ,optimizer ,criterion ,params :Dict [str ,Any ])->Dict [str ,Any ]:
        """Training loop implementation"""
        if self .model is None or self .classification_head is None :
            raise ValueError ("Model veya classification head yüklenmemiş.")
        history ={'train_loss':[],'val_loss':[],'val_accuracy':[]}
        epochs =params .get ('epochs',10 )

        # ERSIN CLIP'i frozen tut
        parameters_method =getattr (self .model ,'parameters',None )
        if parameters_method is not None and callable (parameters_method ):
            params_iter =parameters_method ()
            # ERSIN Type checker doesn't know params_iter is iterable, convert to list
            from typing import cast ,Any 
            try :
            # ERSIN params_iter is an iterator, convert to list for type safety
                params_iter_any =cast (Any ,params_iter )
                if hasattr (params_iter_any ,'__iter__'):
                    params_list =list (params_iter_any )
                else :
                    params_list =[]
            except (TypeError ,AttributeError ):
                params_list =[]
            for param in params_list :
                if hasattr (param ,'requires_grad'):
                    param .requires_grad =False 
        else :
            raise RuntimeError ("Model parameters method not available")

        best_val_loss =float ('inf')
        patience =params .get ('patience',3 )
        patience_counter =0 

        for epoch in range (epochs ):
        # ERSIN Training phase
            train_method =getattr (self .classification_head ,'train',None )
            if train_method is not None and callable (train_method ):
                train_method ()
            else :
                raise RuntimeError ("Classification head train method not available")
            train_loss =0.0 

            for batch_idx ,(images ,_ ,labels )in enumerate (train_loader ):
                images =images .to (self .device )

                # ERSIN Label'ları tensor'e çevir
                batch_labels =[]
                for label_dict in labels :
                    label_tensor =torch .tensor ([
                    label_dict ['violence'],
                    label_dict ['adult_content'],
                    label_dict ['harassment'],
                    label_dict ['weapon'],
                    label_dict ['drug'],
                    label_dict ['safe']
                    ],dtype =torch .float32 )
                    batch_labels .append (label_tensor )

                batch_labels =torch .stack (batch_labels ).to (self .device )

                optimizer .zero_grad ()

                # ERSIN CLIP image features (frozen)
                with torch .no_grad ():
                    encode_image_method =getattr (self .model ,'encode_image',None )
                    if encode_image_method is not None and callable (encode_image_method ):
                        image_features =encode_image_method (images )
                    else :
                        raise RuntimeError ("CLIP model encode_image method not available")
                        # ERSIN Type checker doesn't know image_features has norm method, cast to torch.Tensor
                    from typing import cast 
                    image_features_tensor =cast (torch .Tensor ,image_features )
                    norm_result =image_features_tensor .norm (dim =-1 ,keepdim =True )
                    image_features =image_features_tensor /norm_result 

                    # ERSIN Classification head
                    # ERSIN classification_head is nn.Module, call it directly
                if self .classification_head is not None :
                    predictions =self .classification_head (image_features )
                else :
                    raise RuntimeError ("Classification head not available")

                    # ERSIN Loss hesapla
                loss =criterion (predictions ,batch_labels )
                loss .backward ()
                optimizer .step ()

                train_loss +=loss .item ()

                if batch_idx %10 ==0 :
                    logger .info (f"Epoch {epoch +1 }/{epochs }, Batch {batch_idx }/{len (train_loader )}, Loss: {loss .item ():.4f}")

            avg_train_loss =train_loss /len (train_loader )

            # ERSIN Validation phase
            val_loss ,val_accuracy =self ._validate (val_loader ,criterion )

            # ERSIN History güncelle
            history ['train_loss'].append (avg_train_loss )
            history ['val_loss'].append (val_loss )
            history ['val_accuracy'].append (val_accuracy )

            logger .info (f"Epoch {epoch +1 }/{epochs }: Train Loss: {avg_train_loss :.4f}, Val Loss: {val_loss :.4f}, Val Acc: {val_accuracy :.4f}")

            # ERSIN Early stopping
            if val_loss <best_val_loss :
                best_val_loss =val_loss 
                patience_counter =0 
                # ERSIN En iyi modeli kaydet
                state_dict_method =getattr (self .classification_head ,'state_dict',None )
                if state_dict_method is not None and callable (state_dict_method ):
                    state_dict_result =state_dict_method ()
                    # ERSIN Type checker doesn't know state_dict has copy method, cast to dict
                    from typing import cast 
                    state_dict_dict =cast (dict [str ,Any ],state_dict_result )
                    self .best_model_state =state_dict_dict .copy ()
                else :
                    raise RuntimeError ("Classification head state_dict method not available")
            else :
                patience_counter +=1 

            if patience_counter >=patience :
                logger .info (f"Early stopping at epoch {epoch +1 }")
                break 

                # ERSIN En iyi modeli yükle
        if hasattr (self ,'best_model_state'):
            load_state_dict_method =getattr (self .classification_head ,'load_state_dict',None )
            if load_state_dict_method is not None and callable (load_state_dict_method ):
                load_state_dict_method (self .best_model_state )
            else :
                raise RuntimeError ("Classification head load_state_dict method not available")

        return history 

    def _validate (self ,val_loader ,criterion )->Tuple [float ,float ]:
        """Validation phase"""
        if self .classification_head is None :
            raise ValueError ("Classification head yüklenmemiş.")
        self .classification_head .eval ()
        val_loss =0.0 
        correct_predictions =0 
        total_predictions =0 

        with torch .no_grad ():
            for images ,_ ,labels in val_loader :
                images =images .to (self .device )

                # ERSIN Label'ları tensor'e çevir
                batch_labels =[]
                for label_dict in labels :
                    label_tensor =torch .tensor ([
                    label_dict ['violence'],
                    label_dict ['adult_content'],
                    label_dict ['harassment'],
                    label_dict ['weapon'],
                    label_dict ['drug'],
                    label_dict ['safe']
                    ],dtype =torch .float32 )
                    batch_labels .append (label_tensor )

                batch_labels =torch .stack (batch_labels ).to (self .device )

                # ERSIN CLIP image features
                if self .model is None :
                    raise ValueError ("Model yüklenmemiş.")
                encode_image_method =getattr (self .model ,'encode_image',None )
                if encode_image_method is not None and callable (encode_image_method ):
                    image_features =encode_image_method (images )
                else :
                    raise RuntimeError ("CLIP model encode_image method not available")
                    # ERSIN Type checker doesn't know image_features has norm method, cast to torch.Tensor
                from typing import cast 
                image_features_tensor =cast (torch .Tensor ,image_features )
                norm_result =image_features_tensor .norm (dim =-1 ,keepdim =True )
                image_features =image_features_tensor /norm_result 

                # ERSIN Predictions
                if self .classification_head is None :
                    raise ValueError ("Classification head yüklenmemiş.")
                predictions =self .classification_head (image_features )

                # ERSIN Loss
                loss =criterion (predictions ,batch_labels )
                val_loss +=loss .item ()

                # ERSIN Accuracy (threshold 0.5)
                pred_binary =(predictions >0.5 ).float ()
                correct_predictions +=(pred_binary ==batch_labels ).all (dim =1 ).sum ().item ()
                total_predictions +=batch_labels .size (0 )

        avg_val_loss =val_loss /len (val_loader )
        val_accuracy =correct_predictions /total_predictions if total_predictions >0 else 0.0 

        return avg_val_loss ,val_accuracy 

    def _save_trained_model (self ,training_params :Dict [str ,Any ])->str :
        """Eğitilmiş modeli kaydet"""
        try :
        # ERSIN Model klasörü oluştur
            timestamp =datetime .now ().strftime ('%Y%m%d_%H%M%S')
            version_name =f"clip_finetuned_v{timestamp }"

            model_dir =os .path .join (
            current_app .config ['OPENCLIP_MODEL_VERSIONS_PATH'],
            version_name 
            )
            os .makedirs (model_dir ,exist_ok =True )

            # ERSIN CLIP base model + classification head kaydet
            # ERSIN Use getattr to safely access state_dict methods
            model_state_dict_method =getattr (self .model ,'state_dict',None )
            head_state_dict_method =getattr (self .classification_head ,'state_dict',None )
            if model_state_dict_method is None or not callable (model_state_dict_method ):
                raise RuntimeError ("Model state_dict method not available")
            if head_state_dict_method is None or not callable (head_state_dict_method ):
                raise RuntimeError ("Classification head state_dict method not available")
            model_data ={
            'clip_state_dict':model_state_dict_method (),
            'classification_head_state_dict':head_state_dict_method (),
            'model_config':{
            'clip_model_name':'ViT-H-14-378-quickgelu',
            'pretrained':'dfn5b',
            'num_classes':6 
            },
            'training_params':training_params ,
            'timestamp':timestamp ,
            'version_name':version_name 
            }

            model_path =os .path .join (model_dir ,'open_clip_pytorch_model.bin')
            torch .save (model_data ,model_path )

            logger .info (f"✅ Model kaydedildi: {model_path }")
            return model_path 

        except Exception as e :
            logger .error (f"Model kaydetme hatası: {e }")
            raise e 

    def _save_training_session (self ,training_data :Dict [str ,Any ],training_params :Dict [str ,Any ],history :Dict [str ,Any ],model_path :str )->CLIPTrainingSession :
        """Training session'ı veritabanına kaydet"""
        try :
            timestamp =datetime .now ().strftime ('%Y%m%d_%H%M%S')
            version_name =f"clip_finetuned_v{timestamp }"

            # ERSIN Tüm mevcut session'ları pasif yap
            db .session .query (CLIPTrainingSession ).update ({'is_active':False })

            # ERSIN Yeni session oluştur
            training_session =CLIPTrainingSession (
            version_name =version_name ,
            feedback_count =training_data ['total_samples'],
            training_start =datetime .now (),
            training_end =datetime .now (),
            status ='completed',
            training_params =json .dumps (training_params ),
            performance_metrics =json .dumps ({
            'final_train_loss':history ['train_loss'][-1 ],
            'final_val_loss':history ['val_loss'][-1 ],
            'final_val_accuracy':history ['val_accuracy'][-1 ],
            'epochs_completed':len (history ['train_loss']),
            'train_samples':training_data ['train_samples'],
            'val_samples':training_data ['val_samples']
            }),
            model_path =model_path ,
            is_active =True ,
            is_successful =True 
            )

            db .session .add (training_session )
            db .session .commit ()

            logger .info (f"✅ Training session kaydedildi: {training_session .id }")
            return training_session 

        except Exception as e :
            db .session .rollback ()
            logger .error (f"Training session kaydetme hatası: {e }")
            raise e 

    def get_training_statistics (self )->Dict [str ,Any ]:
        """Training istatistiklerini döndür"""
        try :
        # ERSIN Toplam feedback sayısı
            total_feedback =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content'
            ).count ()

            # ERSIN Geçerli feedback sayısı (frame_path olan)
            valid_feedback =db .session .query (Feedback ).filter (
            Feedback .feedback_type =='content',
            Feedback .frame_path .isnot (None )
            ).count ()

            # ERSIN Training sessions
            total_sessions =db .session .query (CLIPTrainingSession ).count ()
            # ERSIN SQLAlchemy boolean columns - use is_() method
            from sqlalchemy .sql import ColumnElement 
            from typing import cast 
            is_successful_col =cast (ColumnElement [Any ],CLIPTrainingSession .is_successful )
            is_active_col =cast (ColumnElement [Any ],CLIPTrainingSession .is_active )
            # ERSIN column_in şu an kullanılmıyor 
            # ERSIN SQLAlchemy boolean column comparison works at runtime, cast to Any for type checker
            from typing import Any as AnyType 
            successful_sessions =db .session .query (CLIPTrainingSession ).filter (
            cast (AnyType ,is_successful_col )==True 
            ).count ()

            # ERSIN Aktif session
            active_session =db .session .query (CLIPTrainingSession ).filter (
            cast (AnyType ,is_active_col )==True 
            ).first ()

            stats ={
            'total_content_feedback':total_feedback ,
            'valid_content_feedback':valid_feedback ,
            'total_training_sessions':total_sessions ,
            'successful_training_sessions':successful_sessions ,
            'active_session':{
            'id':active_session .id if active_session else None ,
            'version_name':active_session .version_name if active_session else None ,
            'created_at':active_session .created_at .isoformat ()if active_session and active_session .created_at else None 
            }if active_session else None ,
            'ready_for_training':valid_feedback >=10 
            }

            return stats 

        except Exception as e :
            logger .error (f"Training statistics hatası: {e }")
            return {'error':str (e )}