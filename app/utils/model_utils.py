import os 
import torch 
import logging 
import json 

def load_torch_model (model_path ,model_class ,config_keys =None ,device ='cpu',default_config =None ):
    """
    Genel amaçlı PyTorch model yükleyici.
    - model_path: .pth dosya yolu
    - model_class: Model sınıfı (ör. CustomAgeHead)
    - config_keys: Konfigürasyon anahtarları (örn: ['input_dim', 'hidden_dims', 'output_dim'])
    - device: 'cpu' veya 'cuda'
    - default_config: Varsayılan konfigürasyon dict'i
    """
    logger =logging .getLogger (__name__ )
    if not os .path .exists (model_path ):
        logger .error (f"Model dosyası bulunamadı: {model_path }")
        raise FileNotFoundError (f"Model dosyası bulunamadı: {model_path }")
    checkpoint =torch .load (model_path ,map_location =device ,weights_only =True )
    if config_keys and 'model_config'in checkpoint :
        model_config ={k :checkpoint ['model_config'][k ]for k in config_keys if k in checkpoint ['model_config']}
    elif default_config :
        model_config =default_config 
    else :
        model_config ={}
    model =model_class (**model_config )if model_config else model_class ()
    if 'model_state_dict'in checkpoint :
        model .load_state_dict (checkpoint ['model_state_dict'])
    else :
        model .load_state_dict (checkpoint )
    model .to (device )
    model .eval ()
    logger .info (f"Model başarıyla yüklendi: {model_path } -> {device }")
    return model 

def save_torch_model (model ,version_dir ,config_dict =None ,extra_metadata =None ,filename ='model.pth'):
    """
    Genel amaçlı PyTorch model kaydedici.
    - model: PyTorch model nesnesi
    - version_dir: Klasör yolu (oluşturulur)
    - config_dict: Model konfigürasyonu (örn: input_dim, hidden_dims, output_dim)
    - extra_metadata: Ek metadata dict'i (örn: eğitim metrikleri)
    - filename: Kaydedilecek dosya adı (varsayılan: model.pth)
    """
    logger =logging .getLogger (__name__ )
    os .makedirs (version_dir ,exist_ok =True )
    model_path =os .path .join (version_dir ,filename )
    save_dict ={'model_state_dict':model .state_dict ()}
    if config_dict :
        save_dict ['model_config']=config_dict 
    torch .save (save_dict ,model_path )
    logger .info (f"Model ağırlıkları kaydedildi: {model_path }")
    if extra_metadata :
        metadata_path =os .path .join (version_dir ,'metadata.json')
        with open (metadata_path ,'w')as f :
            json .dump (extra_metadata ,f ,indent =4 ,default =str )
        logger .info (f"Model metadata kaydedildi: {metadata_path }")
    return model_path 