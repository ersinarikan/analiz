#!/bin/bash
# WSANALIZ Production Deployment Script
# ===================================
# Bu script production ortamında WSANALIZ sistemini başlatır

set -e  # Hata durumunda script'i durdur

echo "🚀 WSANALIZ Production Deployment Başlatılıyor..."

# Çevre değişkenlerini kontrol et
if [ -z "$SECRET_KEY" ]; then
    echo "❌ HATA: SECRET_KEY çevre değişkeni tanımlanmamış!"
    echo "   export SECRET_KEY='your-secure-secret-key-here'"
    exit 1
fi

# Virtual environment kontrolü
if [ ! -d "venv" ]; then
    echo "❌ HATA: Virtual environment bulunamadı!"
    echo "   python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Virtual environment'ı aktifleştir
echo "📦 Virtual environment aktifleştiriliyor..."
source venv/bin/activate

# Model dosyalarını kontrol et
echo "🤖 Model dosyalarını kontrol ediliyor..."
if [ ! -d "storage/models" ]; then
    echo "⚠️  Model klasörü bulunamadı, oluşturuluyor..."
    mkdir -p storage/models
fi

# Veritabanını güncelle
echo "🗄️  Veritabanı güncelleniyor..."
export FLASK_APP=wsgi.py
flask db upgrade

# Log klasörünü oluştur
echo "📝 Log klasörü hazırlanıyor..."
mkdir -p storage/processed/logs

# Production environment ayarla
export FLASK_ENV=production
export FLASK_DEBUG=False

echo "✅ Sistem hazır!"
echo "🌐 Production sunucusu başlatılıyor..."
echo "   http://localhost:5000"
echo "⏹️  Durdurmak için: Ctrl+C"

# Gunicorn ile production server başlat
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --worker-class eventlet \
    --worker-connections 1000 \
    --timeout 120 \
    --keepalive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    --log-level warning \
    wsgi:app
