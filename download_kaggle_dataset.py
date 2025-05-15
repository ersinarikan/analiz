import os
from kaggle.api.kaggle_api_extended import KaggleApi

# İndirme hedef klasörü
download_path = "storage/datasets/nsfw_2k_raw" # Mevcut klasörü kullanmaya devam edebiliriz
dataset_slug = "excellentconvolution/nsfw-dataset" # Yeni veri seti slug'ı: nsfw-dataset

print(f"Kaggle veri seti {dataset_slug} şuraya indirilecek: {download_path}")

try:
    # Klasörün var olduğundan emin ol
    os.makedirs(download_path, exist_ok=True)
    print(f"'{download_path}' klasörü oluşturuldu veya zaten mevcut.")

    # Kaggle API'sini başlat
    api = KaggleApi()
    api.authenticate() # Bu, C:\Users\ersin\.kaggle\kaggle.json dosyasını veya ortam değişkenlerini kullanır.
    print("Kaggle API ile kimlik doğrulama başarılı.")

    # Veri setini indir ve aç
    print(f"Veri seti indiriliyor: {dataset_slug}...")
    print(f"Lütfen unutmayın: Bu veri setinin Kaggle sayfasında (https://www.kaggle.com/datasets/{dataset_slug}) kullanım koşullarını kabul etmeniz gerekebilir.")
    api.dataset_download_files(dataset_slug, path=download_path, unzip=True, force=True)
    
    print(f"Veri seti başarıyla '{download_path}' klasörüne indirildi ve açıldı.")

    # İndirilen dosyaları listele (isteğe bağlı)
    print(f"İndirilen dosyalar '{download_path}' içinde:")
    for dirname, _, filenames in os.walk(download_path):
        for filename in filenames:
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                print(os.path.join(dirname, filename))

except Exception as e:
    print(f"Veri seti indirilirken bir hata oluştu: {e}")
    import traceback
    traceback.print_exc()

print("İndirme betiği tamamlandı.") 