import os
import sys
import shutil
import open_clip
import logging

# Proje kök dizinini Python arama yoluna ekle
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config # config.py kök dizinde olduğu için doğrudan import

# Basit bir logger ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_openclip_ViT_H_14_378_dfn5b():
    model_name_full = Config.OPENCLIP_MODEL_NAME # "ViT-H-14-378-quickgelu_dfn5b"
    model_type_from_config = Config.OPENCLIP_MODEL_TYPE # "clip"
    
    # model_name_for_lib = "ViT-H-14-378-quickgelu"
    # pretrained_tag_for_lib = "dfn5b"
    # Bu bilgileri Config.OPENCLIP_MODEL_NAME'den ayrıştıralım
    try:
        model_name_for_lib, pretrained_tag_for_lib = model_name_full.split('_')
    except ValueError:
        logger.error(f"Config.OPENCLIP_MODEL_NAME ('{model_name_full}') formatı beklenmiyor. 'modeladi_pretrainedtag' formatında olmalı.")
        return

    logger.info(f"OpenCLIP {model_name_for_lib} (pretrained: {pretrained_tag_for_lib}) modeli indirilmeye başlanıyor...")
    
    # Hedef base_model klasör yolu
    # storage/models/clip/ViT-H-14-378-quickgelu_dfn5b/base_model/
    target_base_model_dir = os.path.join(Config.MODELS_FOLDER, model_type_from_config, model_name_full, 'base_model')
    os.makedirs(target_base_model_dir, exist_ok=True)
    target_model_file_path = os.path.join(target_base_model_dir, "open_clip_pytorch_model.bin") # Veya .pt uzantılı olabilir

    try:
        # Modeli indirerek cache'lenmesini sağla (dosya yolu için değil, sadece indirme amaçlı)
        # OpenCLIP kütüphanesi indirme işlemini ve cache'lemeyi kendisi yönetir.
        # İndirme genellikle ~/.cache/clip/ veya ~/.cache/huggingface/hub/ altına yapılır.
        # ÖNEMLİ: create_model_and_transforms doğrudan bir dosya yolu döndürmez, modeli yükler.
        # Cache'deki dosyanın tam adını ve yerini bulmamız gerekecek.
        
        logger.info("Modelin cache'e indirilmesi için open_clip.create_model_and_transforms çağrılıyor...")
        # Modeli CPU'da indirip yükleyelim, sonra sadece dosyayı kopyalayacağız.
        temp_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name_for_lib,
            pretrained=pretrained_tag_for_lib,
            device="cpu",
            cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "clip") # Cache dizinini belirtelim
        )
        logger.info("Model başarıyla cache'e indirildi/yüklendi.")
        del temp_model # Bellekten serbest bırakalım

        # Cache'deki model dosyasını bul
        # Bu kısım OpenCLIP'in cache yapısına göre uyarlanmalı.
        # Genellikle model_name ve pretrained tag ile bir dosya adı oluşur.
        # Örnek cache yolu: ~/.cache/clip/ViT-H-14-dfn5b.pt (veya benzeri)
        # VEYA huggingface hub cache'i için daha karmaşık bir yol olabilir.
        
        # huggingface_hub cache'i daha olası: ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<hash>/pytorch_model.bin
        # Biz dfn5b için apple/DFN5B-CLIP-ViT-H-14-378 kullanmıştık.
        # Bunun tam cache yolunu manuel bulmak zor, bu yüzden şimdilik varsayımsal bir cache yolu kullanacağız
        # ve kullanıcıdan bu dosyayı manuel olarak base_model'e koymasını isteyebiliriz.
        # VEYA, model.config.json gibi bir dosyadan pretrained_model_name_or_path alınabilir.
        
        # Şimdilik en güvenli yol, kullanıcının daha önce yaptığı gibi indirme scriptini çalıştırıp
        # cache'deki dosyayı manuel olarak base_model altına "open_clip_pytorch_model.bin" adıyla koymasıdır.
        # Bu script sadece bu işlemin yapılacağını varsayacak ve o dosyanın varlığını kontrol edecek.

        # Model dosyasının cache'deki varsayılan yerini tahmin etmek yerine,
        # doğrudan HuggingFace Hub'dan indirilmiş dosyaların yerini kullanıcının belirttiği varsayalım.
        # VEYA modelin kendisinden bu bilgiyi çekmeye çalışalım (eğer mümkünse)

        # OpenCLIP kütüphanesinin modeli nereye kaydettiğini bulmak için daha sağlam bir yol gerekli.
        # Şimdilik, bu scriptin amacı sadece MODELİN İNDİRİLMESİNİ TETİKLEMEK olsun.
        # Kopyalama işlemini, cache'deki dosyanın yerini netleştirdikten sonra ekleyelim.
        # YA DA, eğer create_model_and_transforms'e `pretrained` olarak doğrudan bir URL veya bilinen bir HF repo adı verilirse,
        # `hf_hub_download` kullanılabilir.
        
        # dfn5b için pretrained = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
        if pretrained_tag_for_lib.lower() == "dfn5b":
            hf_repo_id = "apple/DFN5B-CLIP-ViT-H-14-378"
            # Bu repoda genellikle pytorch_model.bin olur.
            # config.json içinde "_name_or_path": "ViT-H-14-378-quickgelu" gibi bilgiler de olur.
            # Gerçek dosya adı snapshot hash'ine bağlıdır.
            # En iyisi kullanıcıya bu dosyayı manuel kopyalatmak veya hf_hub_download kullanmak.
            
            # huggingface_hub.hf_hub_download kullanarak dosyayı doğrudan hedef yola indirelim.
            from huggingface_hub import hf_hub_download
            logger.info(f"HuggingFace Hub'dan {hf_repo_id} (pytorch_model.bin) doğrudan hedef yola indiriliyor...")
            
            # DFN5B modeli için beklenen dosya adları genellikle şunlardır:
            # open_clip_pytorch_model.bin, pytorch_model.bin, open_clip_config.json, config.json, tokenizer.json vb.
            # Biz sadece ana model ağırlığını (genellikle .bin veya .pt) hedefliyoruz.
            # `DFN5B-CLIP-ViT-H-14-378` reposunda ana dosya `pytorch_model.bin` olarak görünüyor.
            downloaded_file = hf_hub_download(
                repo_id=hf_repo_id, 
                filename="pytorch_model.bin", 
                local_dir=target_base_model_dir,
                local_dir_use_symlinks=False, # Windows'ta symlink sorun yaratabilir, doğrudan kopyala
                # force_download=True # Her zaman indirmesi için (opsiyonel)
            )
            logger.info(f"Dosya başarıyla '{downloaded_file}' olarak indirildi.")
            # İndirilen dosyanın adını bizim standart adımıza (open_clip_pytorch_model.bin) çevirelim.
            if os.path.basename(downloaded_file) != "open_clip_pytorch_model.bin":
                final_target_path = os.path.join(target_base_model_dir, "open_clip_pytorch_model.bin")
                shutil.move(downloaded_file, final_target_path)
                logger.info(f"Dosya '{final_target_path}' olarak yeniden adlandırıldı/taşındı.")
            else:
                final_target_path = downloaded_file

            # Diğer gerekli dosyalar (config.json, tokenizer.json) da aynı şekilde indirilebilir.
            # Şimdilik sadece model ağırlığını indiriyoruz.
            logger.info(f"OpenCLIP modeli ({model_name_for_lib} - {pretrained_tag_for_lib}) başarıyla '{target_base_model_dir}' altına indirildi.")
            print(f"OpenCLIP modeli '{target_base_model_dir}' altına indirildi.")
            print("İçerik:", os.listdir(target_base_model_dir))

        else:
            logger.warning(f"'{pretrained_tag_for_lib}' için otomatik indirme ve kopyalama mantığı henüz tanımlanmadı.")
            logger.warning("Lütfen modeli manuel olarak indirip cache'den kopyalayın veya bu script'i güncelleyin.")
            print(f"Lütfen {model_name_for_lib} - {pretrained_tag_for_lib} modelini manuel olarak indirip")
            print(f"'{target_base_model_dir}' altına 'open_clip_pytorch_model.bin' adıyla kopyalayın.")

    except Exception as e:
        logger.error(f"OpenCLIP model indirme/kopyalama hatası: {str(e)}")
        print(f"Hata: {str(e)}")

if __name__ == '__main__':
    # Sadece belirli bir modeli indirmek için bir fonksiyon çağıralım.
    # Daha sonra farklı OpenCLIP modelleri için ayrı fonksiyonlar eklenebilir.
    download_openclip_ViT_H_14_378_dfn5b()
    logger.info("İndirme scripti tamamlandı. Lütfen logları ve cache klasörlerini kontrol edin.") 