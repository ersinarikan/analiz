import os
import cv2
from tqdm import tqdm
import logging
import config
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if config.Config.USE_GPU:
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']

def process_images(image_dir, output_dir, img_limit):
    import insightface
    import numpy as np # np.array için
    try:
        # INSIGHTFACE_AGE_MODEL_BASE_PATH -> .../storage/models/age/buffalo_l/base_model
        # Gerçek model dosyası: .../base_model/models/buffalo_l/w600k_r50.onnx
        # (InsightFace kütüphanesi modelleri bir 'models' alt klasöründe arayabilir)
        # Ancak get_model doğrudan .onnx yolunu da alabilir.
        # Emin olmak için config'deki path'e göre tam yolu oluşturalım.
        base_path = config.Config.INSIGHTFACE_AGE_MODEL_BASE_PATH
        # InsightFace'in kendi iç yapısına göre modeller 'models/model_adı/' altında olur.
        # buffalo_l için bu 'models/buffalo_l/' şeklindedir.
        recognition_model_path = os.path.join(base_path, "models", "buffalo_l", "w600k_r50.onnx")
        
        if not os.path.exists(recognition_model_path):
            # Eğer beklenen yerde değilse, base_path'in doğrudan buffalo_l modeli olduğunu varsayalım
            # ve içinde w600k_r50.onnx olduğunu umalım (bu genellikle olmaz).
            # Daha olası bir durum, base_path'in ZATEN .insightface/models/buffalo_l gibi bir yeri göstermesi
            # VEYA bizim indirme scriptimizin dosyaları doğrudan base_model altına kopyalıyor.
            # İndirme scriptimiz dosyaları storage/models/age/buffalo_l/base_model/ altına kopyalıyor,
            # InsightFace'in beklediği 'models/buffalo_l' alt klasör yapısı OLMADAN.
            # BU YÜZDEN DOĞRUDAN base_path altındaki w600k_r50.onnx'i denemeliyiz.
            logger.debug(f"Standart iç içe model yolu ({recognition_model_path}) bulunamadı.")
            recognition_model_path = os.path.join(base_path, "w600k_r50.onnx") # İndirme yapımıza göre bu olmalı.
            logger.debug(f"Doğrudan base_path altında model yolu deneniyor: {recognition_model_path}")

        if not os.path.exists(recognition_model_path):
            logger.error(f"Tanıma modeli ONNX dosyası bulunamadı: {recognition_model_path} (ve alternatif yollar denendi)")
            return

        rec_model = insightface.model_zoo.get_model(recognition_model_path, providers=providers)
        if rec_model is None:
            logger.error(f"Tanıma modeli ({recognition_model_path}) insightface.model_zoo.get_model ile yüklenemedi.")
            return
        
        # rec_model.prepare() diye bir şey var mı kontrol et, genellikle gerekmez.
        # input_size, rec_model.input_shape üzerinden öğrenilebilir, örn: (112, 112)
        logger.info(f"InsightFace tanıma modeli ({recognition_model_path}) doğrudan yüklendi.")
        # logger.info(f"Kullanılan provider: {rec_model.providers}")
        # logger.info(f"Model input shape: {rec_model.input_shape}") # input_shape özelliği olmayabilir

    except Exception as e:
        logger.error(f"InsightFace modeli yüklenirken hata oluştu: {e}", exc_info=True)
        return

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"{len(image_files)} adet resim dosyası bulundu.")

    if img_limit > 0:
        image_files = image_files[:img_limit]
        logger.info(f"İşlem {img_limit} resim ile sınırlandırıldı.")

    embeddings = []
    ages_list = []
    
    processed_count = 0
    skipped_age_filter = 0
    skipped_no_face = 0
    skipped_load_error = 0

    for image_path in tqdm(image_files, desc="Resimler işleniyor"):
        try:
            filename = os.path.basename(image_path)
            age_str = filename.split('_')[0]
            age = int(age_str)

            if not (0 <= age <= 100):
                skipped_age_filter += 1
                continue

            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Resim yüklenemedi: {image_path}")
                skipped_load_error += 1
                continue

            # Resmi modelin beklediği boyuta getirmemiz gerekebilir.
            # buffalo_l (ArcFace) genellikle 112x112 BGR resim bekler.
            # UTKFace resimleri zaten kırpılmış ama boyutları farklı olabilir.
            # get_feat metodu bunu kendi içinde yapıyor olabilir, kontrol etmek lazım.
            # Şimdilik doğrudan veriyoruz.
            logger.debug(f"[{filename}] rec_model.get_feat(img) çağrılıyor.")
            embedding_array = rec_model.get_feat(img)
            
            if embedding_array is not None:
                embedding = embedding_array.flatten() # Genellikle (1, 512) döner, flatten() (512,) yapar.
                logger.debug(f"[{filename}] Embedding (doğrudan model ile) başarıyla çıkarıldı. Şekil: {embedding.shape}")
            else:
                logger.warning(f"[{filename}] rec_model.get_feat(img) ile embedding çıkarılamadı (None döndü).")
                skipped_no_face +=1
                continue
            
            processed_count += 1
            embeddings.append(embedding)
            ages_list.append(age)

        except ValueError:
            skipped_age_filter +=1
            continue
        except Exception as e:
            logger.error(f"Resim işlenirken beklenmedik hata ({image_path}): {e}", exc_info=True)
            skipped_no_face += 1
            continue
    
    logger.info(f"Başarıyla işlenen resim sayısı: {processed_count}")
    logger.info(f"Yaş filtresi nedeniyle atlanan resim sayısı: {skipped_age_filter}")
    logger.info(f"Embedding çıkarılamayan/yüz bulunamayan resim sayısı: {skipped_no_face}")
    logger.info(f"Yüklenemeyen resim sayısı: {skipped_load_error}")

    if not embeddings:
        logger.warning("Kaydedilecek hiç embedding bulunamadı.")
        return

    output_file_path = os.path.join(output_dir, "utkface_embeddings.npz")
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(output_file_path, embeddings=np.array(embeddings), ages=np.array(ages_list))
    logger.info(f"Embeddingler ve yaşlar başarıyla kaydedildi: {output_file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UTKFace veri setinden yüz embeddinglerini çıkarır.")
    parser.add_argument("--image_dir", type=str, required=True, help="UTKFace resimlerinin bulunduğu dizin.")
    parser.add_argument("--output_dir", type=str, required=True, help="Embeddinglerin kaydedileceği .npz dosyasının çıktı dizini.")
    parser.add_argument("--img_limit", type=int, default=0, help="İşlenecek maksimum resim sayısı (0 = hepsi).")

    args = parser.parse_args()

    process_images(args.image_dir, args.output_dir, args.img_limit) 