# WSANALIZ - Web Tabanlı Yapay Zeka Analiz Sistemi

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.3-green)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Proprietary-yellow)](LICENSE)

## 📋 Proje Genel Bakış

WSANALIZ, görüntü ve video içeriklerinde yapay zeka destekli analiz yapan kapsamlı bir web uygulamasıdır. Sistem, içerik güvenliği analizi ve yaş tahmini özelliklerini gelişmiş makine öğrenmesi modelleri ile sunar.

### ✨ Ana Özellikler

- 🔍 **İçerik Analizi**: Şiddet, yetişkin içerik, taciz, silah, uyuşturucu tespiti
- 👥 **Yaş Tahmini**: Yapay zeka destekli yüz tanıma ve yaş tahmini
- 🎥 **Video İşleme**: Kare kare analiz ve risk skorlaması
- 🔄 **Model Eğitimi**: Kullanıcı geri bildirimleri ile model iyileştirme
- 📊 **Gerçek Zamanlı İzleme**: WebSocket ile canlı analiz takibi
- 🌐 **Web Arayüzü**: Kullanıcı dostu, responsive tasarım

### 🏗️ Sistem Mimarisi

```
WSANALIZ/
├── Frontend (Bootstrap + JS)
├── Backend (Flask)
├── AI Models (TensorFlow + PyTorch)
├── Database (SQLite/PostgreSQL)
└── File Storage
```

## 🔧 Teknoloji Stack'i

### Backend Framework
- **Flask 2.2.3** - Web framework
- **SQLAlchemy** - ORM ve veritabanı yönetimi
- **Flask-SocketIO** - Gerçek zamanlı iletişim
- **Gunicorn** - Production WSGI server

### Yapay Zeka Modelleri
- **TensorFlow 2.16.1** - İçerik analizi modelleri
- **PyTorch 2.2.2** - Yaş tahmini modelleri
- **OpenCLIP** - Görsel-metin analizi
- **YOLO (Ultralytics)** - Nesne tespiti
- **InsightFace** - Yüz tanıma

### Görüntü İşleme
- **OpenCV** - Görüntü/video işleme
- **Pillow** - Resim manipülasyonu
- **MoviePy** - Video dönüştürme

## 🚀 Kurulum ve Çalıştırma

### Sistem Gereksinimleri
- Python 3.8+
- 8GB+ RAM (model yükleme için)
- CUDA uyumlu GPU (opsiyonel, performans için)

### 1. Projeyi İndirme
```bash
git clone https://github.com/yourusername/wsanaliz.git
cd wsanaliz
```

### 2. Virtual Environment Kurulumu
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

### 4. Çevre Değişkenlerini Ayarlama
`.env` dosyası oluşturun:
```env
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
DATABASE_URL=sqlite:///wsanaliz.db
```

### 5. Veritabanını Başlatma
```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### 6. Uygulamayı Başlatma

#### Development Modu
```bash
python app.py
```

#### Production Modu
```bash
# Gunicorn ile
gunicorn --bind 0.0.0.0:5000 wsgi:app

# Veya systemd servisi olarak
sudo systemctl start wsanaliz
```

## 📊 Kullanım Kılavuzu

### İçerik Analizi
1. Ana sayfadan dosya yükleyin (resim/video)
2. Analiz parametrelerini ayarlayın
3. "Analizi Başlat" butonuna tıklayın
4. Sonuçları gerçek zamanlı olarak izleyin

### Model Yönetimi
1. "Model Yönetimi" butonuna tıklayın
2. Model versiyonlarını görüntüleyin
3. Yeni model eğitimi başlatın
4. Model performansını izleyin

### Geri Bildirim Sistemi
1. Analiz sonuçlarında "Geri Bildirim" sekmesini açın
2. Yanlış tespitleri düzeltin
3. Doğru değerleri girin
4. Geri bildirimi gönderin

## 🤖 Model Detayları

### İçerik Analizi Modeli
- **Base Model**: OpenCLIP ViT-H-14
- **Kategoriler**: 6 kategori (şiddet, yetişkin, taciz, silah, uyuşturucu, güvenli)
- **Güven Skorları**: CLIP similarity ile hesaplanır
- **Eğitim**: Kullanıcı geri bildirimleri ile fine-tuning

### Yaş Tahmini Modeli
- **Yüz Tespiti**: MTCNN algorithm
- **Base Model**: UTKFace dataset ile eğitilmiş
- **Çıktı**: Yaş (0-100), güven skoru
- **Eğitim**: Custom PyTorch head + geri bildirimler

## 🔒 Güvenlik

### Dosya Güvenliği
- MIME type doğrulaması
- Dosya boyutu limitleri
- Güvenli dosya isimlendirme
- Virus tarama entegrasyonu (opsiyonel)

### Veri Koruma
- Şifreli veritabanı bağlantıları
- Session güvenliği
- CORS koruması
- Input sanitization

## 📈 Performans

### Optimizasyonlar
- Model caching
- Asenkron işleme
- GPU acceleration
- Batch processing

### Benchmark'lar
- Resim analizi: ~2-5 saniye
- Video analizi: ~30fps işleme hızı
- Yaş tahmini: ~1 saniye/yüz
- Model eğitimi: ~5-10 dakika

## 🔧 Yapılandırma

### config.py Ayarları
```python
# Production ayarları
DEBUG = False
SECRET_KEY = os.environ.get('SECRET_KEY')
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

# Dosya limitleri
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'jpg', 'png'}

# Model ayarları
MAX_CONCURRENT_ANALYSES = 3
ANALYSIS_TIMEOUT = 1800  # 30 dakika
```

## 🐛 Sorun Giderme

### Yaygın Sorunlar

**Model yüklenmiyor:**
```bash
# GPU memory kontrolü
nvidia-smi

# Model dosyalarını kontrol edin
ls storage/models/
```

**WebSocket bağlantı sorunu:**
```javascript
// Browser console'da
socket.connected  // true olmalı
```

**Yavaş analiz:**
```bash
# CPU/Memory kullanımını kontrol edin
htop

# Log dosyalarını inceleyin
tail -f storage/processed/logs/app.log
```

## 📚 API Referansı

### Ana Endpoints
- `POST /api/files/upload` - Dosya yükleme
- `POST /api/analysis/start` - Analiz başlatma
- `GET /api/analysis/results/{id}` - Sonuç alma
- `POST /api/feedback/submit` - Geri bildirim gönderme

### Model Management
- `GET /api/model/stats` - Model istatistikleri
- `POST /api/model/train` - Model eğitimi
- `POST /api/model/reset` - Model sıfırlama

Detaylı API dokümantasyonu için `/api/docs` sayfasını ziyaret edin.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje özel lisans altındadır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Geliştirici Ekibi

- **Lead Developer**: [Adınız]
- **AI/ML Engineer**: [Adınız]
- **Frontend Developer**: [Adınız]

## 🙏 Teşekkürler

- OpenAI CLIP modeli için
- UTKFace dataset sağlayıcıları
- InsightFace kütüphanesi
- Flask ve Python topluluğu

## 📞 İletişim

- **Email**: info@wsanaliz.com
- **Website**: https://wsanaliz.com
- **Support**: https://support.wsanaliz.com

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!