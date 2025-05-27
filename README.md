# WSANALIZ Projesi

## Proje Genel Bakış
WSANALIZ projesi, görsel ve video içeriklerini otomatik olarak analiz ederek çeşitli kategorilerde risk değerlendirmesi yapan bir sistemdir. Proje, yapay zeka modelleri kullanarak şiddet, taciz, yetişkin içeriği, silah kullanımı ve madde kullanımı gibi kategorilerde içerik analizi yapabilmektedir. Ayrıca, yaş tahmini özelliği ile görüntülerdeki kişilerin yaklaşık yaşını belirleyebilmektedir.

## Proje Mimarisi
Proje, aşağıdaki ana bileşenlerden oluşmaktadır:

1. **Web Arayüzü**: Kullanıcı etkileşimleri için Flask tabanlı web uygulaması
2. **İçerik Analiz Motoru**: Görsel ve video içeriklerini analiz eden yapay zeka modelleri
3. **Veritabanı Katmanı**: Analiz sonuçlarını ve kullanıcı geri bildirimlerini depolayan veritabanı
4. **Dosya İşleme Servisi**: Yüklenen dosyaların işlenmesi ve depolanması
5. **Model Servisi**: Yapay zeka modellerinin yönetimini sağlayan servis
6. **Analiz Servisi**: İçerik analizini yöneten ve sonuçları derleyen servis
7. **Model Versiyonu Yönetimi**: Model versiyonlarının takibi, aktivasyonu ve temizlenmesi

## Model Yönetimi Sistemi

### Yaş Tahmin Modeli Yönetimi

WSANALIZ projesi, yaş tahmini için gelişmiş bir model yönetimi sistemi kullanır:

#### Model Yapısı:
- **Base Model**: UTKFace dataset ile eğitilmiş temel model (`storage/models/age/custom_age_head/base_model/`)
- **Active Model**: Şu anda kullanılan aktif model (`storage/models/age/custom_age_head/active_model/`)
- **Versioned Models**: Eğitilen yeni model versiyonları (`storage/models/age/custom_age_head/versions/`)
- **Buffalo Model**: Yedek yüz tanıma modeli (`storage/models/age/buffalo_l/`)

#### Dual Model Sistemi:
Sistem hem Custom Age Head modeli hem de Buffalo yüz tanıma modelini paralel olarak kullanır:
- **Custom Age Head**: Yaş tahmini için özel eğitilmiş model
- **Buffalo**: Yüz tespiti ve embedding çıkarımı için InsightFace modeli

### Model Yönetimi Web Arayüzü

Web arayüzünde "Model Yönetimi" butonu ile şu özellikler kullanılabilir:

#### Yaş Tahmin Modeli:
- **Aktif Versiyon Görüntüleme**: Şu anda aktif olan model versiyonu
- **Model Versiyonları**: Tüm eğitilmiş versiyonların listesi (v1, v2, v3, vb.)
- **Versiyon Aktivasyonu**: Herhangi bir versiyonu aktif hale getirme
- **Model Sıfırlama**: Modeli base model (UTKFace eğitimli) haline döndürme
- **En Son Versiyon Silme**: En yeni versiyonu silme (güvenlik kontrolü ile)
- **Yeni Eğitim Başlatma**: Geri bildirimler ile yeni model eğitimi

#### İçerik Analiz Modeli:
- **CLIP Model Bilgileri**: Kullanılan CLIP modeli versiyonu
- **Kategori Sayıları**: Şiddet, taciz, yetişkin içeriği vb. kategoriler
- **Model Durumu**: Aktif/pasif durum bilgisi

### Sistem Yeniden Başlatma Mekanizması

**Önemli**: Model değişiklikleri (aktivasyon/sıfırlama) sistem yeniden başlatılmasını gerektirir.

#### Yeniden Başlatma Nedenleri:
1. **Model Aktivasyonu**: Yeni bir yaş modeli versiyonu aktif edildiğinde
2. **Model Sıfırlama**: Yaş modeli base modele sıfırlandığında
3. **Yeni Model Yükleme**: Sistem belleğindeki modellerin yenilenmesi için

#### Yeniden Başlatma Süreci:
1. Kullanıcı model değişikliği yapar (aktivasyon/sıfırlama)
2. Sistem uyarı verir: "Model değişikliği sistem yeniden başlatılmasını gerektirir"
3. İşlem onaylanırsa:
   - Backend API çağrısı yapılır
   - Yükleyici ekranı gösterilir
   - Sistem yeni subprocess ile yeniden başlatılır
   - Eski süreç kapatılır
   - WebSocket bağlantısı kesilir ve yeniden kurulur
   - Yükleyici ekranı otomatik kapanır

#### Güvenlik Önlemleri:
- Dosya yüklü veya analiz devam ederken model işlemleri engellenir
- Aktif model silinemez (önce başka versiyon aktif edilmeli)
- En az bir model versiyonu sistemde kalmalıdır
- Base model (UTKFace eğitimli) hiçbir zaman silinmez

## Gereksinim-Modül İlişkileri

### 1. Dosya Yükleme ve İşleme
**İlgili Dosyalar:**
- `app/services/file_service.py`: Dosya yükleme, MIME tipi belirleme, küçük resim oluşturma işlemlerini yönetir
  - `save_uploaded_file()`: Yüklenen dosyaları güvenli bir şekilde kaydeder
  - `get_file_info()`: Dosya boyutu ve MIME tipi gibi bilgileri sağlar
  - `create_thumbnail()`: Resim ve video dosyaları için küçük resimler oluşturur

**Gereksinim Karşılama:**
Bu modül, sistemin farklı dosya formatlarını (resim, video) kabul etme ve güvenli bir şekilde işleme gereksinimini karşılar. Dosya formatı doğrulaması, güvenli dosya isimlendirme ve uygun depolama yöntemleri ile güvenlik sağlanır.

### 2. İçerik Analizi
**İlgili Dosyalar:**
- `app/ai/content_analyzer.py`: Şiddet, yetişkin içeriği, taciz, silah ve madde kullanımı tespiti yapan yapay zeka modellerini barındırır
- `app/ai/age_estimator.py`: Resimlerdeki kişilerin yüzlerini tespit eden ve yaş tahmini yapan yapay zeka modelini içerir
- `app/services/analysis_service.py`: Analiz işlemlerini koordine eden servis
  - `analyze_image()`: Resim dosyalarını analiz eder
  - `analyze_video()`: Video dosyalarını analiz eder, karelere ayırır ve her kare için analiz gerçekleştirir
  - `calculate_overall_scores()`: Analiz sonuçlarını derleyerek genel risk skorları hesaplar

**Gereksinim Karşılama:**
Bu modüller, sistemin farklı kategorilerde içerik analizi yapabilme gereksinimini karşılar. Derin öğrenme modelleri kullanarak şiddet, yetişkin içeriği, taciz, silah ve madde kullanımı gibi öğeleri tespit eder ve her kategori için risk skoru üretir.

### 3. Gerçek Zamanlı İzleme
**İlgili Dosyalar:**
- `app/services/analysis_service.py`: 
  - `analyze_video()`: Video dosyalarını karelere ayırarak gerçek zamanlı analiz yapabilme kabiliyeti sağlar
  - `update_progress()`: Analiz ilerleme durumunu izleme ve raporlama yapar

**Gereksinim Karşılama:**
Bu fonksiyonlar, uzun videoların kare kare işlenmesini ve belirli zaman dilimlerinde risk içeren bölümlerin tespit edilmesini sağlar. Video içindeki kişilerin takibi ve sürekli analizi de gerçekleştirilir.

### 4. Yaş Tahmini
**İlgili Dosyalar:**
- `app/ai/age_estimator.py`: Yüz tespiti ve yaş tahmini yapan yapay zeka modelini içerir
  - `detect_faces()`: Görüntüdeki yüzleri tespit eder
  - `estimate_age()`: Tespit edilen yüzler için yaş tahmini yapar
  - `compute_face_encoding()`: Yüz vektörü hesaplayarak kişi takibi sağlar
- `app/services/age_training_service.py`: Yaş modeli eğitimi ve versiyon yönetimi
  - `prepare_training_data()`: Geri bildirimlerden eğitim verisi hazırlar
  - `train_model()`: PyTorch ile yaş modeli eğitir
  - `save_model_version()`: Eğitilen modeli versiyonlar
  - `activate_model_version()`: Model versiyonunu aktif eder
  - `reset_to_base_model()`: Base modele sıfırlar

**Gereksinim Karşılama:**
Bu modül, görüntülerdeki kişilerin yaşını tahmin etme gereksinimini karşılar. MTCNN yüz tespit algoritması ve özel eğitilmiş yaş tahmin modeli kullanarak yüksek doğrulukta yaş tahminleri sağlar.

### 5. Model Yönetimi
**İlgili Dosyalar:**
- `app/services/model_service.py`: Model performans istatistikleri, model sürüm yönetimi ve model sıfırlama işlevleri sağlar
  - `get_model_stats()`: Model performans istatistiklerini döndürür
  - `reset_model()`: Modeli orijinal eğitimli haline sıfırlar
  - `prepare_training_data()`: Eğitim için gerekli verileri hazırlar
  - `activate_model_version()`: Model versiyonunu aktif eder
  - `delete_latest_version()`: En son versiyonu siler
- `app/routes/model_routes.py`: Model yönetimi API endpoint'leri
  - `/api/model/activate/<version_id>`: Model versiyonu aktivasyonu (sistem yeniden başlatma ile)
  - `/api/model/reset/<model_type>`: Model sıfırlama (sistem yeniden başlatma ile)
  - `/api/model/delete-latest/<model_type>`: En son versiyon silme
  - `/api/model/versions/<model_type>`: Model versiyonları listeleme

**Gereksinim Karşılama:**
Bu modül, sistemin yapay zeka modellerinin yönetimi ve performans izleme gereksinimlerini karşılar. Modellerin sürüm kontrolü, performans metriklerinin takibi ve gerektiğinde modellerin güncellenmesi gibi işlevleri sağlar.

## Kritik Dosyalar ve Sınıflar

### 1. `app/ai/content_analyzer.py`
İçerik analiz motoru, şiddet, yetişkin içeriği, taciz, silah ve madde kullanımı tespiti için derin öğrenme modellerini barındırır.

**Önemli Fonksiyonlar:**
- `analyze_image()`: Bir görüntüyü analiz eder ve kategori bazlı risk skorlarını döndürür
- `detect_objects()`: Görüntüdeki nesneleri tespit eder

### 2. `app/ai/age_estimator.py`
Yaş tahmin motoru, yüz tespiti ve yaş/cinsiyet tahmini yapan yapay zeka modellerini içerir.

**Önemli Fonksiyonlar:**
- `detect_faces()`: Görüntüdeki yüzleri tespit eder
- `estimate_age()`: Bir yüz görüntüsünden yaş ve cinsiyet tahmini yapar
- `analyze_image()`: Bir görüntüdeki tüm yüzleri tespit eder ve yaş/cinsiyet tahmini yapar

### 3. `app/services/analysis_service.py`
Analiz servis sınıfı, içerik analiz süreçlerini yönetir.

**Önemli Fonksiyonlar:**
- `start_analysis()`: Bir dosya için analiz işlemini başlatır
- `analyze_image()`: Resim analizi yapar
- `analyze_video()`: Video analizi yapar
- `calculate_overall_scores()`: Kategorilere göre genel risk skorlarını hesaplar
- `get_analysis_results()`: Analiz sonuçlarını formatlanmış şekilde döndürür

### 4. `app/services/model_service.py`
Model servis sınıfı, yapay zeka modellerinin yönetimini sağlar.

**Önemli Fonksiyonlar:**
- `get_model_stats()`: Model performans istatistiklerini döndürür
- `get_available_models()`: Sistemde kullanılabilir modelleri listeler
- `reset_model()`: Bir modeli orijinal eğitimli haline sıfırlar
- `prepare_training_data()`: Eğitim için gerekli verileri hazırlar

### 5. `app/services/file_service.py`
Dosya servis sınıfı, dosya işleme ve depolama işlemlerini yönetir.

**Önemli Fonksiyonlar:**
- `save_uploaded_file()`: Yüklenen dosyayı güvenli bir şekilde kaydeder
- `get_file_info()`: Dosya hakkında temel bilgileri döndürür
- `create_thumbnail()`: Dosya için küçük resim oluşturur

### 6. `app/services/age_training_service.py`
Yaş modeli eğitimi ve versiyon yönetimi sınıfı.

**Önemli Fonksiyonlar:**
- `prepare_training_data()`: Geri bildirimlerden eğitim verisi hazırlar
- `train_model()`: PyTorch kullanarak yaş modeli eğitir
- `save_model_version()`: Eğitilen modeli yeni versiyon olarak kaydeder
- `activate_model_version()`: Belirli bir versiyonu aktif eder
- `reset_to_base_model()`: Base modele (UTKFace eğitimli) sıfırlar
- `cleanup_training_data()`: Kullanılan eğitim verilerini temizler

## Kurulum ve Çalıştırma

1.  **Sanal Ortam Oluşturma (Önerilir):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **Gerekli Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Yapay Zeka Modellerini İndirin:**
    Proje için gerekli olan önceden eğitilmiş yapay zeka modellerini indirmek için aşağıdaki script'leri çalıştırın:
    ```bash
    # InsightFace modeli için
    python download_insightface_model.py
    
    # OpenCLIP modeli için
    python download_openclip.py
    ```
    Modellerin `storage/models` klasörüne doğru şekilde indiğinden emin olun.

4.  **İlk Model Kurulumu (Yaş Tahmini için):**
    ```bash
    # UTKFace dataset ile base model oluşturma
    python sync_model_versions.py
    ```

5.  **Yapılandırma Dosyasını Hazırlayın (Gerekirse):**
    Proje kök dizininde `.env.example` dosyası bulunmaktadır. Bu dosyayı kopyalayarak `.env` adında yeni bir dosya oluşturun ve kendi yerel ayarlarınıza göre düzenleyebilirsiniz.

6.  **Uygulamayı Başlatın:**
    ```bash
    python app.py
    ```
    Uygulama varsayılan olarak `http://localhost:5000` adresinde çalışmaya başlayacaktır.

## Model Versiyonu Yönetimi Komutları

### Komut Satırı Araçları

#### 1. En Son Model Versiyonunu Silme
```bash
# En son yaş modeli versiyonunu sil
python delete_latest_model_version.py --model-type age

# Dry run (sadece ne yapılacağını göster, silme)
python delete_latest_model_version.py --model-type age --dry-run

# En son içerik modeli versiyonunu sil
python delete_latest_model_version.py --model-type content
```

#### 2. Model Versiyonlarını Senkronize Etme
```bash
# Tüm model versiyonlarını kontrol et ve eksikleri tamamla
python sync_model_versions.py

# Belirli model tipini senkronize et
python sync_model_versions.py --model-type age
```

### API Endpoint'leri

#### 1. Model Versiyonu Yönetimi
```bash
# Model versiyonlarını listele
curl http://localhost:5000/api/model/versions/age

# Model versiyonu aktif et (sistem yeniden başlar)
curl -X POST http://localhost:5000/api/model/activate/3

# En son versiyonu sil
curl -X DELETE http://localhost:5000/api/model/delete-latest/age

# Model sıfırla (sistem yeniden başlar)
curl -X POST http://localhost:5000/api/model/reset/age
```

#### 2. Model İstatistikleri
```bash
# Yaş modeli metrikleri
curl http://localhost:5000/api/model/metrics/age

# İçerik modeli metrikleri
curl http://localhost:5000/api/model/metrics/content

# Tüm model metrikleri
curl http://localhost:5000/api/model/metrics/all
```

#### 3. Model Eğitimi

##### Yaş Modeli Eğitimi
```bash
# Varsayılan parametrelerle eğitim
python train_v1_model.py

# Özel parametrelerle eğitim
python train_v1_model.py --epochs 20 --batch-size 64 --learning-rate 0.001

# Sadece veri istatistiklerini görmek için
python train_v1_model.py --dry-run
```

##### OpenCLIP İçerik Modeli Eğitimi
```bash
# Varsayılan parametrelerle eğitim
python train_content_model.py

# Özel parametrelerle eğitim
python train_content_model.py --epochs 15 --batch-size 32 --learning-rate 0.0005

# Sadece veri istatistiklerini görmek için
python train_content_model.py --dry-run

# Minimum örnek sayısını değiştirme
python train_content_model.py --min-samples 100
```

**İçerik Modeli Eğitim Özellikleri:**
- OpenCLIP base modelini kullanır (ViT-H-14-378-quickgelu)
- Classification head eklenerek fine-tuning yapılır
- Kategoriler: şiddet, yetişkin içerik, taciz, silah, uyuşturucu
- Kullanıcı geri bildirimleri training data olarak kullanılır
- Eğitim sonrası model versiyonu otomatik olarak kaydedilir

### Güvenlik Kontrolleri

#### Model Silme Güvenliği:
- En az bir model versiyonu sistemde kalmalıdır
- Aktif model silinemez (önce başka versiyon aktif edilmeli)
- Base model (UTKFace eğitimli) hiçbir zaman silinmez
- Silme işlemi geri alınamaz uyarısı verilir

#### Sistem Durumu Kontrolleri:
- Dosya yüklü veya analiz devam ederken model işlemleri engellenir
- Model değişiklikleri sistem yeniden başlatılmasını gerektirir
- WebSocket bağlantı durumu izlenir

## Kullanılan Ana Teknolojiler ve Kütüphaneler

*   **Backend:** Python, Flask, Flask-SQLAlchemy, Flask-Migrate, Flask-SocketIO, Flask-CORS.
*   **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js.
*   **Yapay Zeka & Görüntü İşleme:**
    *   PyTorch (Yaş modeli eğitimi için)
    *   TensorFlow / Keras (Model eğitimi ve kullanımı)
    *   ONNX / ONNXRuntime (Cross-platform model çalıştırma)
    *   OpenCV (Görüntü işleme)
    *   Dlib (Yüz tespiti ve landmark tespiti)
    *   InsightFace (Gelişmiş yüz analizi - Buffalo model)
    *   OpenCLIP (CLIP-based içerik analizi)
    *   YOLO (Gerçek zamanlı nesne tespiti)
    *   Scikit-learn (Makine öğrenmesi metrikleri)
    *   NumPy, Pandas (Veri manipülasyonu)
*   **Veritabanı:** SQLite (Geliştirme için varsayılan).
*   **Diğer:** Requests, Pillow (Görüntü işleme), python-dotenv.

## Geliştirme Rehberi

### Yeni Bir Analiz Kategorisi Ekleme
1. `app/ai/content_analyzer.py` dosyasında yeni kategori için tespit fonksiyonu ekleyin
2. `app/models/analysis.py` dosyasında ilgili veritabanı alanlarını güncelleyin
3. `app/services/analysis_service.py` dosyasında analiz metodunu güncelleyin
4. Kullanıcı arayüzünde yeni kategoriyi görüntülemek için gerekli değişiklikleri yapın

### Yeni Bir Model Ekleme
1. `app/ai` klasöründe yeni model için Python dosyası oluşturun
2. `app/services/model_service.py` dosyasına yeni model için yönetim fonksiyonları ekleyin
3. `app/services/analysis_service.py` dosyasında yeni modeli kullanacak analiz fonksiyonlarını güncelleyin
4. Model versiyonu yönetimi için gerekli tabloları ve endpoint'leri ekleyin

### Yaş Modeli Eğitimi Geliştirme
1. `app/services/age_training_service.py` dosyasında eğitim parametrelerini ayarlayın
2. Geri bildirim veri toplama mekanizmasını geliştirin
3. Model performans metriklerini iyileştirin
4. Eğitim veri temizleme politikalarını güncelleyin

## Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### 1. Model Yükleme Hatası
```
Error: Could not load age estimation model
```
**Çözüm**: 
- Base model dosyasının varlığını kontrol edin: `storage/models/age/custom_age_head/base_model/model.pth`
- InsightFace modellerinin indirildiğinden emin olun: `python download_insightface_model.py`

#### 2. Model Aktivasyon Hatası
```
Error: Model version activation failed
```
**Çözüm**:
- Model dosyasının bozuk olmadığını kontrol edin
- Sistem yeniden başlatma izinlerini kontrol edin
- Aktif model sembolik linkini manuel olarak düzeltin

#### 3. Video İşleme Hatası
**Çözüm**: FFmpeg bağımlılığının doğru şekilde yüklendiğinden emin olun

#### 4. Bellek Yetersizliği
**Çözüm**: Büyük videoları işlerken bellek limitlerinizi kontrol edin ve gerekirse ayarlayın

#### 5. Sistem Yeniden Başlatma Sorunu
```
Error: System restart failed after model change
```
**Çözüm**:
- `app.py` dosyasının mevcut olduğunu kontrol edin
- Python interpreter izinlerini kontrol edin
- Manuel olarak uygulamayı yeniden başlatın

### Log Dosyaları
- Uygulama logları: Konsol çıktısında
- Model eğitimi logları: `storage/processed/logs/` klasöründe
- Hata logları: Flask development server çıktısında

## Performans Optimizasyonları

### Model Yükleme Optimizasyonu
- Model önbellekleme sistemi kullanılır
- Lazy loading ile ihtiyaç halinde model yüklenir
- GPU kullanımı desteklenir (mevcut ise)

### Bellek Yönetimi
- Büyük video dosyaları chunk'lar halinde işlenir
- Kullanılmayan modeller bellekten temizlenir
- Garbage collection optimize edilmiştir

### Veritabanı Optimizasyonu
- Index'ler performans için optimize edilmiştir
- Query'ler batch işlem için optimize edilmiştir
- Cleanup politikaları eski verileri temizler

## Güvenlik Özellikleri

### Dosya Güvenliği
- Dosya tipi doğrulaması
- Güvenli dosya isimlendirme
- Dosya boyutu limitleri
- Virus tarama desteği (opsiyonel)

### Model Güvenliği
- Model dosyası integrity kontrolü
- Authorized model activation
- Safe model fallback mechanisms
- Encrypted model storage (opsiyonel)

### API Güvenliği
- Request rate limiting
- Input validation
- Error message sanitization
- CORS policy enforcement

## Lisans ve Katkıda Bulunma

Bu proje açık kaynak olarak geliştirilmektedir. Katkıda bulunmak için:

1. Repository'yi fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

### Kod Standartları
- PEP 8 Python stil rehberini takip edin
- Fonksiyonlar için docstring kullanın
- Unit testler yazın
- Type hints kullanın (Python 3.6+)

## İletişim ve Destek

Proje ile ilgili sorularınız için:
- GitHub Issues kullanın
- Dokümantasyonu kontrol edin
- Log dosyalarını inceleyin

---

**Not**: Bu proje sürekli geliştirilmekte olup, yeni özellikler ve iyileştirmeler düzenli olarak eklenmektedir.