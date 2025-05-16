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

**Gereksinim Karşılama:**
Bu modül, görüntülerdeki kişilerin yaşını tahmin etme gereksinimini karşılar. MTCNN yüz tespit algoritması ve özel eğitilmiş yaş tahmin modeli kullanarak yüksek doğrulukta yaş tahminleri sağlar.

### 5. Model Yönetimi
**İlgili Dosyalar:**
- `app/services/model_service.py`: Model performans istatistikleri, model sürüm yönetimi ve model sıfırlama işlevleri sağlar
  - `get_model_stats()`: Model performans istatistiklerini döndürür
  - `reset_model()`: Modeli orijinal eğitimli haline sıfırlar
  - `prepare_training_data()`: Eğitim için gerekli verileri hazırlar

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
    python download_insightface_model.py
    # Not: Eğer ek model indirme script'leriniz varsa (örn: download_yolo_model.py, download_kaggle_dataset.py)
    # onları da burada listeyebilirsiniz veya kullanıcıya ilgili modellere göre çalıştırması gerektiğini belirtebilirsiniz.
    ```
    Modellerin genellikle `storage/models` klasörü veya alt klasörlerine doğru şekilde indiğinden emin olun. Her modelin beklenen konumu script içinde belirtilmiş olabilir.

4.  **Yapılandırma Dosyasını Hazırlayın (Gerekirse):**
    Proje kök dizininde `.env.example` dosyası bulunmaktadır. Bu dosyayı kopyalayarak `.env` adında yeni bir dosya oluşturun ve kendi yerel ayarlarınıza göre (örneğin, veritabanı yolu, API anahtarları, gizli anahtarlar vb.) düzenleyebilirsiniz. Uygulama, `.env` dosyası bulunamazsa `config.py` dosyasındaki varsayılan geliştirme ayarlarını kullanacaktır.

5.  **Uygulamayı Başlatın:**
    ```bash
    python app.py
    ```
    Uygulama varsayılan olarak `http://0.0.0.0:5000` adresinde çalışmaya başlayacaktır. 
    **Not:** Mevcut `initialize_app` fonksiyonu, geliştirme modunda her uygulama başlangıcında veritabanını (`wsanaliz_dev.db`) silip yeniden oluşturmaktadır. Kalıcı veri tutmak istiyorsanız bu davranışı `app/__init__.py` içerisinden düzenlemeniz gerekebilir.

## Kullanılan Ana Teknolojiler ve Kütüphaneler

*   **Backend:** Python, Flask, Flask-SQLAlchemy, Flask-Migrate (Veritabanı şema yönetimi için), Flask-SocketIO (Gerçek zamanlı iletişim için), Flask-CORS.
*   **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js (Grafik gösterimleri için).
*   **Yapay Zeka & Görüntü İşleme:**
    *   TensorFlow / Keras (Model eğitimi ve kullanımı)
    *   ONNX / ONNXRuntime (Farklı framework'lerde eğitilmiş modelleri çalıştırmak için)
    *   OpenCV (Temel görüntü işleme görevleri)
    *   Dlib (Yüz tespiti ve landmark tespiti gibi görevler için)
    *   InsightFace (Gelişmiş yüz analizi ve tanıma modelleri)
    *   YOLO (Gerçek zamanlı nesne tespiti için)
    *   Scikit-learn (Makine öğrenmesi görevleri ve metrikler için)
    *   NumPy, Pandas (Veri manipülasyonu ve analizi)
*   **Veritabanı:** SQLite (Geliştirme için varsayılan).
*   **Diğer:** Requests (HTTP istekleri için), Pillow (Görüntü işleme).

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

## Sorun Giderme

### Yaygın Hatalar ve Çözümleri
1. **Model Yükleme Hatası**: Yapay zeka modelleri için gerekli ön eğitimli dosyaların varlığını kontrol edin
2. **Video İşleme Hatası**: FFmpeg bağımlılığının doğru şekilde yüklendiğinden emin olun
3. **Bellek Yetersizliği**: Büyük videoları işlerken bellek limitlerinizi kontrol edin ve gerekirse ayarlayın

## Gelecek Özellikler
1. Gerçek zamanlı video akışı analizi
2. Sesli içerik analizi (konuşma ve ses tanıma)
3. Daha detaylı yüz analizi (ifade tanıma)
4. Çoklu dil desteği
5. Özelleştirilebilir risk eşikleri