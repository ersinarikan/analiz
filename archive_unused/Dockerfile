# Python 3.9 slim imajını kullan
FROM python:3.9-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgstreamer-plugins-base1.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Storage dizinlerini oluştur
RUN mkdir -p storage/uploads storage/processed storage/models logs

# Çevre değişkenlerini ayarla
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Port 5000'i expose et
EXPOSE 5000

# Uygulama kullanıcısı oluştur ve değiştir
RUN useradd --create-home --shell /bin/bash wsanaliz
RUN chown -R wsanaliz:wsanaliz /app
USER wsanaliz

# Health check ekle
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Uygulamayı başlat
CMD ["python", "app.py"] 