#!/usr/bin/env python
"""
Yapay zeka modellerini indirme betiği.
Bu betik, uygulamanın ihtiyaç duyduğu ön eğitimli modelleri indirir ve uygun dizinlere yerleştirir.
"""

import os 
import sys 
import argparse 
import shutil 
import requests 
import zipfile 
import tarfile 
import gdown 
from tqdm import tqdm 

# ERSIN Proje kök dizinini al
script_dir =os .path .dirname (os .path .abspath (__file__ ))
project_root =os .path .dirname (script_dir )

# ERSIN Model dizinleri
MODEL_DIR =os .path .join (project_root ,'app','static','models')
STORAGE_MODEL_DIR =os .path .join (project_root ,'storage','models')
DLIB_MODELS_DIR =os .path .join (MODEL_DIR ,'dlib_models')

# ERSIN İndirilecek modeller
MODELS ={
'clip':{
'name':'CLIP Model',
'url':'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
'dest_dir':os .path .join (MODEL_DIR ,'content_model'),
'filename':'ViT-B-32.pt'
},
'yolo':{
'name':'YOLOv8 Model',
'url':'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
'dest_dir':os .path .join (MODEL_DIR ,'yolo_model'),
'filename':'best.pt'
},
'dlib_face_recognition':{
'name':'dlib Face Recognition Model',
'url':'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
'dest_dir':DLIB_MODELS_DIR ,
'filename':'dlib_face_recognition_resnet_model_v1.dat'
},
'dlib_shape_predictor':{
'name':'dlib Shape Predictor Model',
'url':'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
'dest_dir':DLIB_MODELS_DIR ,
'filename':'shape_predictor_68_face_landmarks.dat'
},
'age_model':{
'name':'Age Estimation Model',
'gdrive_id':'1YCox_4kJ-BYeXq27uUbasu--yz28zUMV',# ERSIN Aciklama.
'dest_dir':os .path .join (MODEL_DIR ,'age_model'),
'filename':'age_model.pt'
}
}

def download_file (url ,dest_path ,desc =None ):
    """URL'den dosya indir ve ilerlemeyi göster."""
    response =requests .get (url ,stream =True )
    total_size =int (response .headers .get ('content-length',0 ))
    block_size =1024 # ERSIN 1 Kibibyte

    with open (dest_path ,'wb')as f :
        with tqdm (total =total_size ,unit ='iB',unit_scale =True ,desc =desc )as bar :
            for data in response .iter_content (block_size ):
                bar .update (len (data ))
                f .write (data )

def extract_archive (archive_path ,dest_dir ,archive_type ='auto'):
    """Arşiv dosyasını çıkar."""
    if archive_type =='auto':
        if archive_path .endswith ('.zip'):
            archive_type ='zip'
        elif archive_path .endswith ('.tar.gz')or archive_path .endswith ('.tgz'):
            archive_type ='tar'
        elif archive_path .endswith ('.bz2'):
            archive_type ='bz2'

    if archive_type =='zip':
        with zipfile .ZipFile (archive_path ,'r')as zip_ref :
            zip_ref .extractall (dest_dir )
    elif archive_type =='tar':
        with tarfile .open (archive_path ,'r:gz')as tar_ref :
            tar_ref .extractall (dest_dir )
    elif archive_type =='bz2':
        import bz2 
        with open (archive_path ,'rb')as src ,open (archive_path [:-4 ],'wb')as dst :
            dst .write (bz2 .decompress (src .read ()))
        return archive_path [:-4 ]

    return dest_dir 

def download_from_gdrive (file_id ,dest_path ,desc =None ):
    """Google Drive'dan dosya indir."""
    try :
        gdown .download (id =file_id ,output =dest_path ,quiet =False )
        return True 
    except Exception as e :
        print (f"Google Drive'dan indirme hatası: {e }")
        return False 

def main ():
    parser =argparse .ArgumentParser (description ='Yapay zeka modellerini indir')
    parser .add_argument ('--models',nargs ='+',choices =list (MODELS .keys ())+['all'],default =['all'],
    help ='İndirilecek modeller (varsayılan: all)')
    parser .add_argument ('--force',action ='store_true',
    help ='Mevcut modelleri yeniden indir')

    args =parser .parse_args ()

    # ERSIN 'all' seçeneği varsa tüm modelleri seç
    if 'all'in args .models :
        models_to_download =list (MODELS .keys ())
    else :
        models_to_download =args .models 

        # ERSIN Dizinleri oluştur
    for model_info in MODELS .values ():
        os .makedirs (model_info ['dest_dir'],exist_ok =True )

        # ERSIN Storage model dizini için de aynısını yap
    for model_key in ['content_model','age_model']:
        os .makedirs (os .path .join (STORAGE_MODEL_DIR ,model_key ),exist_ok =True )

        # ERSIN Modelleri indir
    for model_key in models_to_download :
        model_info =MODELS [model_key ]
        dest_file =os .path .join (model_info ['dest_dir'],model_info ['filename'])

        # ERSIN Eğer dosya zaten varsa ve force bayrağı set edilmemişse atla
        if os .path .exists (dest_file )and not args .force :
            print (f"{model_info ['name']} zaten mevcut, atlanıyor.")
            continue 

        print (f"{model_info ['name']} indiriliyor...")

        # ERSIN Google Drive'dan indirme
        if 'gdrive_id'in model_info :
            success =download_from_gdrive (model_info ['gdrive_id'],dest_file ,desc =model_info ['name'])
            if not success :
                print (f"{model_info ['name']} indirilirken hata oluştu.")
                continue 
        else :
        # ERSIN Normal URL'den indirme
            download_file (model_info ['url'],dest_file ,desc =model_info ['name'])

            # ERSIN Arşivleri çıkar
        if dest_file .endswith (('.zip','.tar.gz','.tgz','.bz2')):
            print (f"{model_info ['name']} arşivi çıkarılıyor...")
            extracted_path =extract_archive (dest_file ,model_info ['dest_dir'])

            # ERSIN .bz2 dosyaları için orijinal arşivi sil
            if dest_file .endswith ('.bz2'):
                os .remove (dest_file )

                # ERSIN Storage model klasörüne kopyala, bazı modeller için
        if model_key in ['content_model','age_model']:
            storage_dest =os .path .join (STORAGE_MODEL_DIR ,model_key )
            print (f"{model_info ['name']} storage dizinine kopyalanıyor...")

            # ERSIN Eğer bir dosya ise direkt kopyala
            if os .path .isfile (dest_file ):
                shutil .copy2 (dest_file ,os .path .join (storage_dest ,os .path .basename (dest_file )))
                # ERSIN Eğer bir klasör ise içindeki her şeyi kopyala
            elif os .path .isdir (dest_file ):
                for item in os .listdir (dest_file ):
                    s =os .path .join (dest_file ,item )
                    d =os .path .join (storage_dest ,item )
                    if os .path .isfile (s ):
                        shutil .copy2 (s ,d )

        print (f"{model_info ['name']} başarıyla indirildi.")

    print ("Tüm modeller başarıyla indirildi!")

if __name__ =="__main__":
    main ()