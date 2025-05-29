# WSANALIZ Production Dockerfile
# =============================
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim-bullseye as builder

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libboost-all-dev \
    libdlib-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim-bullseye

# Sistem kullanıcısı oluştur
RUN useradd --create-home --shell /bin/bash wsanaliz

# Runtime bağımlılıkları
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libdlib19 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libgtk2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini builder'dan kopyala
COPY --from=builder /root/.local /home/wsanaliz/.local

# Uygulama dizinini oluştur
WORKDIR /app

# Uygulama dosyalarını kopyala
COPY --chown=wsanaliz:wsanaliz . .

# Gerekli klasörleri oluştur
RUN mkdir -p storage/uploads storage/processed storage/models \
    && chown -R wsanaliz:wsanaliz storage

# Kullanıcıya geç
USER wsanaliz

# PATH'e kullanıcı Python paketlerini ekle
ENV PATH="/home/wsanaliz/.local/bin:${PATH}"

# Production ortam değişkenleri
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV PYTHONPATH=/app

# Port açığa çıkar
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Uygulama başlat
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "--timeout", "120", "wsgi:app"] 