#!/usr/bin/env python3
"""
NSFW Model İndirme ve ONNX Dönüştürme Scripti

Bu script:
1. HuggingFace'den Marqo/nsfw-image-detection-384 modelini indirir
2. PyTorch modelini ONNX formatına dönüştürür (12x daha hızlı inference için)
3. Model dosyasını storage/models/nsfw/ konumuna kaydeder
4. Model metadata'sını JSON olarak kaydeder
"""

import os 
import sys 
import json 
import logging 
from pathlib import Path 

# ERSIN Proje root dizinini ekle
project_root =Path (__file__ ).parent .parent 
sys .path .insert (0 ,str (project_root ))

logging .basicConfig (
level =logging .INFO ,
format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger =logging .getLogger (__name__ )

def download_and_convert_nsfw_model ():
    """NSFW modelini indir ve ONNX formatına dönüştür"""
    try :
        import torch 
        from transformers import AutoImageProcessor ,AutoModelForImageClassification 
        import onnxruntime as ort 
        from PIL import Image 
        import numpy as np 
    except ImportError as e :
        logger .error (f"Gerekli kütüphaneler yüklü değil: {e }")
        logger .error ("Lütfen şu paketleri yükleyin: torch, transformers, onnxruntime, pillow, numpy")
        return False 

        # ERSIN Model yolları
        # ERSIN Falconsai modeli daha başarılı (%100 vs %5.4) ve daha hızlı (11.6ms vs 131ms)
    model_name ="Falconsai/nsfw_image_detection"# ERSIN Aciklama.
    models_dir =project_root /"storage"/"models"/"nsfw"
    models_dir .mkdir (parents =True ,exist_ok =True )

    onnx_model_path =models_dir /"nsfw-detector-224.onnx"# ERSIN Aciklama.
    metadata_path =models_dir /"metadata.json"

    # ERSIN Model zaten varsa kontrol et
    if onnx_model_path .exists ()and metadata_path .exists ():
        logger .info (f"Model zaten mevcut: {onnx_model_path }")
        try :
            with open (metadata_path ,'r')as f :
                metadata =json .load (f )
            logger .info (f"Model metadata: {metadata }")
            return True 
        except Exception as e :
            logger .warning (f"Metadata okunamadı, yeniden indiriliyor: {e }")

    logger .info (f"NSFW modeli indiriliyor: {model_name }")

    # ERSIN 1. Model ve processor'ı yükle
    try :
        processor =AutoImageProcessor .from_pretrained (model_name )
        model =AutoModelForImageClassification .from_pretrained (model_name )
        model .eval ()# ERSIN Inference modu
        logger .info ("✅ Model ve processor başarıyla yüklendi")
    except Exception as e :
        logger .error (f"Model yükleme hatası: {e }")
        return False 

        # ERSIN 2. Input size'ı processor'dan al (Falconsai 224x224, Marqo 384x384)
    if hasattr (processor ,'size'):
        if isinstance (processor .size ,dict ):
            input_size =processor .size .get ('height',processor .size .get ('shortest_edge',224 ))
        elif isinstance (processor .size ,(list ,tuple )):
            input_size =processor .size [0 ]if len (processor .size )>0 else 224 
        else :
            input_size =int (processor .size )
    else :
        input_size =224 # ERSIN Default

    logger .info (f"Model input size: {input_size }x{input_size }")

    # ERSIN 3. Model metadata'sını kaydet
    metadata ={
    "model_name":model_name ,
    "input_size":input_size ,
    "normalization":{
    "mean":processor .image_mean if hasattr (processor ,'image_mean')else [0.485 ,0.456 ,0.406 ],
    "std":processor .image_std if hasattr (processor ,'image_std')else [0.229 ,0.224 ,0.225 ]
    },
    "num_classes":model .config .num_labels if hasattr (model .config ,'num_labels')else 2 ,
    "model_type":"binary_classification",
    "output_format":"probability"
    }

    # ERSIN 4. ONNX formatına dönüştür
    logger .info ("Model ONNX formatına dönüştürülüyor...")
    try :
    # ERSIN Örnek input, doğru size ile
    # ERSIN torch.randn creates tensor with random values from normal distribution
        dummy_input =torch .randn (size =(1 ,3 ,input_size ,input_size ))
        logger .info (f"ONNX export için dummy input shape: {dummy_input .shape }")

        # ERSIN ONNX export
        torch .onnx .export (
        model ,
        dummy_input ,
        str (onnx_model_path ),
        input_names =['input'],
        output_names =['output'],
        dynamic_axes ={
        'input':{0 :'batch_size'},
        'output':{0 :'batch_size'}
        },
        opset_version =14 ,
        do_constant_folding =True ,
        export_params =True ,
        verbose =False 
        )
        logger .info (f"✅ Model ONNX formatına dönüştürüldü: {onnx_model_path }")
    except Exception as e :
        logger .error (f"ONNX dönüştürme hatası: {e }")
        return False 

        # ERSIN 5. ONNX modelini test et
    logger .info ("ONNX modeli test ediliyor...")
    try :
        ort_session =ort .InferenceSession (str (onnx_model_path ))

        # ERSIN Test input'u doğru size ile hazırla
        test_input =np .random .randn (1 ,3 ,input_size ,input_size ).astype (np .float32 )
        outputs =ort_session .run (None ,{'input':test_input })
        # ERSIN outputs is a list, first element is the output tensor
        if outputs and len (outputs )>0 :
            first_output =outputs [0 ]
            # ERSIN Check if first_output is a numpy array with shape attribute
            if isinstance (first_output ,np .ndarray )and hasattr (first_output ,'shape'):
                logger .info (f"✅ ONNX model test başarılı. Output shape: {first_output .shape }")
            else :
                logger .warning ("ONNX model test: output shape bilgisi alınamadı")
        else :
            logger .warning ("ONNX model test: output boş")
    except Exception as e :
        logger .error (f"ONNX model test hatası: {e }")
        return False 

        # ERSIN 5. Metadata'yı kaydet
    try :
        with open (metadata_path ,'w')as f :
            json .dump (metadata ,f ,indent =2 )
        logger .info (f"✅ Metadata kaydedildi: {metadata_path }")
    except Exception as e :
        logger .error (f"Metadata kaydetme hatası: {e }")
        return False 

    logger .info ("="*60 )
    logger .info ("✅ NSFW modeli başarıyla indirildi ve dönüştürüldü!")
    logger .info (f"   Model: {onnx_model_path }")
    logger .info (f"   Metadata: {metadata_path }")
    logger .info ("="*60 )

    return True 

if __name__ =="__main__":
    success =download_and_convert_nsfw_model ()
    sys .exit (0 if success else 1 )
