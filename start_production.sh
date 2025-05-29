#!/bin/bash
# WSANALIZ Simple Start Script
# ============================

echo "🚀 WSANALIZ Başlatılıyor..."

# Virtual environment'ı aktifleştir
if [ -d "venv" ]; then
    echo "📦 Virtual environment aktifleştiriliyor..."
    source venv/bin/activate
fi

# Gerekli klasörleri oluştur
mkdir -p storage/uploads storage/processed storage/models

echo "✅ Sistem hazır!"
echo "🌐 Sunucu başlatılıyor: http://localhost:5000"
echo "⏹️  Durdurmak için: Ctrl+C"

# Flask development server ile başlat
python app.py
