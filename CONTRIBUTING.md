# WSANALIZ Projesine Katkıda Bulunma Rehberi

WSANALIZ projesine katkıda bulunmak istediğiniz için teşekkür ederiz! Bu rehber, projeye nasıl katkıda bulunabileceğiniz konusunda bilgi sağlar.

## Katkı Türleri

### 1. Hata Bildirimi (Bug Reports)
- GitHub Issues'da yeni bir issue açın
- Hatayı detaylı şekilde açıklayın
- Hata ile ilgili ekran görüntüsü ekleyin
- Sistem bilgilerinizi (OS, Python version) belirtin

### 2. Özellik İstekleri (Feature Requests)
- Yeni özellik önerilerinizi GitHub Issues'da paylaşın
- Özelliğin neden gerekli olduğunu açıklayın
- Varsa örnek kullanım senaryoları ekleyin

### 3. Kod Katkıları (Code Contributions)
- Fork yapın ve yeni bir branch oluşturun
- Kodlama standartlarımıza uyun
- Test yazın
- Pull Request gönderin

## Geliştirme Ortamı Kurulumu

### Ön Gereksinimler
- Python 3.8+
- Git
- CUDA (GPU kullanımı için opsiyonel)

### Kurulum Adımları

1. **Repository'yi fork edin ve clone yapın:**
```bash
git clone https://github.com/yourusername/analiz.git
cd analiz
```

2. **Virtual environment oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Environment dosyasını ayarlayın:**
```bash
cp env.example .env
# .env dosyasını düzenleyin
```

5. **Uygulamayı çalıştırın:**
```bash
python app.py
```

## Kodlama Standartları

### Python Kod Stili
- PEP 8 standartlarına uyun
- Fonksiyon ve sınıf isimleri açıklayıcı olmalı
- Docstring'ler ekleyin
- Type hints kullanın

```python
def analyze_image(image_path: str) -> Dict[str, float]:
    """
    Görüntüyü analiz eder ve kategori skorlarını döndürür.
    
    Args:
        image_path: Analiz edilecek görüntünün yolu
        
    Returns:
        Kategori skorları içeren dictionary
    """
    pass
```

### Dosya Organizasyonu
- `app/` - Ana uygulama kodu
- `app/ai/` - AI model kodları
- `app/services/` - İş mantığı servisleri
- `app/routes/` - Flask route'ları
- `app/models/` - Veritabanı modelleri
- `tests/` - Test dosyaları

## Test Yazma

### Test Türleri
1. **Unit Tests** - Tekil fonksiyonları test eder
2. **Integration Tests** - Bileşenler arası etkileşimi test eder
3. **End-to-End Tests** - Tam kullanıcı senaryolarını test eder

### Test Çalıştırma
```bash
python -m pytest tests/
```

### Test Örneği
```python
import pytest
from app.services.analysis_service import AnalysisService

def test_analyze_image():
    service = AnalysisService()
    result = service.analyze_image("test_image.jpg")
    
    assert "violence" in result
    assert 0 <= result["violence"] <= 1
```

## Pull Request Süreci

### 1. Branch Oluşturma
```bash
git checkout -b feature/yeni-ozellik
# veya
git checkout -b bugfix/hata-duzeltmesi
```

### 2. Değişiklikleri Commit Etme
```bash
git add .
git commit -m "feat: yeni özellik açıklaması"
```

### Commit Mesajı Formatı
- `feat:` - Yeni özellik
- `fix:` - Hata düzeltmesi
- `docs:` - Dokümantasyon değişikliği
- `style:` - Kod formatı değişikliği
- `refactor:` - Kod yeniden düzenleme
- `test:` - Test ekleme/düzeltme

### 3. Pull Request Gönderme
- Açıklayıcı başlık yazın
- Değişiklikleri detaylandırın
- İlgili issue'ları belirtin
- Test sonuçlarını ekleyin

## AI Model Geliştirme

### Model Ekleme Süreci
1. Model dosyasını `app/ai/` klasörüne ekleyin
2. Servis katmanında (`app/services/`) integration yapın
3. API endpoint'i oluşturun (`app/routes/`)
4. Test yazın
5. Dokümantasyon güncelleyin

### Model Performans Kriterleri
- Doğruluk (Accuracy) > %85
- İşlem süresi < 2 saniye (tek görüntü)
- Bellek kullanımı < 2GB

## Veritabanı Değişiklikleri

### Migration Oluşturma
```bash
flask db migrate -m "migration açıklaması"
flask db upgrade
```

### Model Değişiklikleri
- Mevcut tabloları koruyun
- Backward compatibility sağlayın
- Migration script'lerini test edin

## Dokümantasyon

### Gerekli Dokümantasyon
- Kod içi docstring'ler
- API dokümantasyonu
- README güncellemeleri
- CHANGELOG kayıtları

### Dokümantasyon Formatı
- Markdown kullanın
- Kod örnekleri ekleyin
- Görsel açıklamalar (gerektiğinde)

## İletişim

### Discussions
- Genel sorular için GitHub Discussions kullanın
- Teknik tartışmalar için Issues açın

### Code Review
- Tüm PR'lar review edilir
- En az 1 reviewer onayı gerekir
- CI/CD testlerinin geçmesi zorunludur

## Lisans

Katkıda bulunarak, kodunuzun MIT lisansı altında dağıtılmasını kabul etmiş olursunuz.

## Teşekkürler

Her türlü katkı değerlidir! Sorunuz varsa çekinmeden iletişime geçin. 