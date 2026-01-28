#!/bin/bash
# WSANALIZ Temiz Başlangıç Scripti
# 1) Çalışan servisleri durdurur
# 2) storage/processed ve storage/uploads klasörlerini temizler
# 3) Veritabanını siler (wsanaliz_dev.db + -shm, -wal)
# 4) Servisleri yeniden başlatır

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[*]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_stop()  { echo -e "${GREEN}[−]${NC} $1"; }  # durduruldu (hata değil)

# --- 1) Servisleri durdur ---
log_info "Çalışan servisler durduruluyor..."

USE_SYSTEMD=0
if systemctl list-unit-files wsanaliz-web.service 2>/dev/null | grep -q wsanaliz-web; then
    USE_SYSTEMD=1
fi

if [ "$USE_SYSTEMD" = "1" ]; then
    # ERSIN Systemd ile yönetiliyorsa systemctl stop (graceful, doğru yöntem)
    log_info "Systemd servisleri durduruluyor (systemctl stop)..."
    sudo systemctl stop wsanaliz-web wsanaliz-worker 2>/dev/null || true
    sleep 2
    log_stop "wsanaliz-web ve wsanaliz-worker systemctl ile durduruldu."
else
    # ERSIN Manuel başlatılmışsa pkill (systemd yok)
    if pkill -f "gunicorn.*wsgi:app" 2>/dev/null; then
        log_stop "Gunicorn durduruldu (pkill)."
        sleep 1
    fi
    if pkill -f "app.services.queue_worker" 2>/dev/null; then
        log_stop "Queue worker durduruldu (pkill)."
        sleep 1
    fi
fi

if [ -f "$ROOT/wsanaliz.pid" ]; then
    PID=$(cat "$ROOT/wsanaliz.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        log_stop "PID dosyasındaki $PID durduruldu."
    fi
    rm -f "$ROOT/wsanaliz.pid"
fi

for _ in 1 2; do
    if pgrep -f "gunicorn.*wsgi" >/dev/null 2>&1 || pgrep -f "queue_worker" >/dev/null 2>&1; then
        sleep 1
    else
        break
    fi
done

# --- 2) processed ve uploads temizle ---
log_info "storage/processed ve storage/uploads temizleniyor..."

PROCESSED="$ROOT/storage/processed"
UPLOADS="$ROOT/storage/uploads"

for dir in "$PROCESSED" "$UPLOADS"; do
    if [ -d "$dir" ]; then
        rm -rf "${dir:?}"/*
        log_info "  $dir içeriği silindi."
    else
        mkdir -p "$dir"
        log_info "  $dir oluşturuldu."
    fi
done

# --- 3) Veritabanını sil ---
log_info "Veritabanı siliniyor..."

DB_PATH="$ROOT/wsanaliz_dev.db"
for f in "$DB_PATH" "${DB_PATH}-shm" "${DB_PATH}-wal"; do
    if [ -e "$f" ]; then
        rm -f "$f"
        log_info "  $f silindi."
    fi
done

# --- 4) Servisleri başlat ---
log_info "Servisler başlatılıyor..."

if [ "$USE_SYSTEMD" = "1" ]; then
    log_info "Systemd ile başlatılıyor (sudo şifresi gerekebilir)..."
    if ! sudo systemctl start wsanaliz-web wsanaliz-worker 2>/dev/null; then
        log_warn "systemctl start başarısız veya yetki yok. Manuel: sudo systemctl start wsanaliz-web wsanaliz-worker"
    fi
    sleep 3
    if systemctl is-active --quiet wsanaliz-web 2>/dev/null; then
        log_info "wsanaliz-web çalışıyor."
    else
        log_warn "wsanaliz-web aktif değil. Kontrol: sudo systemctl status wsanaliz-web"
    fi
else
    VENV="${VENV:-$ROOT/venv}"
    if [ ! -d "$VENV" ]; then
        log_warn "Venv bulunamadı: $VENV"
        exit 1
    fi

    export PATH="$VENV/bin:$PATH"
    export FLASK_ENV="${FLASK_ENV:-production}"
    export FLASK_DEBUG="${FLASK_DEBUG:-0}"
    export PYTHONUNBUFFERED=1
    export WSANALIZ_QUEUE_BACKEND="${WSANALIZ_QUEUE_BACKEND:-redis}"
    export WSANALIZ_REDIS_URL="${WSANALIZ_REDIS_URL:-redis://localhost:6379/0}"

    mkdir -p "$ROOT/logs"

    log_info "Gunicorn başlatılıyor (port 5000)..."
    gunicorn --bind 0.0.0.0:5000 \
        --workers 1 \
        --worker-class eventlet \
        --worker-connections 1000 \
        --timeout 300 \
        --keep-alive 5 \
        --log-level info \
        --access-logfile - \
        --error-logfile - \
        --daemon \
        --pid "$ROOT/wsanaliz_gunicorn.pid" \
        wsgi:app

    sleep 2
    log_info "Queue worker başlatılıyor..."
    nohup "$VENV/bin/python" -m app.services.queue_worker >> "$ROOT/logs/worker.log" 2>&1 &
    echo $! > "$ROOT/wsanaliz_worker.pid"
fi

log_info "Temiz başlangıç tamamlandı."
log_info "  Web: http://0.0.0.0:5000"
log_info "  VT ve storage sıfırlandı; tablolar uygulama açılışında oluşturulacak."

# ERSIN Bağlantı kontrolü
sleep 2
if command -v ss >/dev/null 2>&1 && ss -tlnp 2>/dev/null | grep -q ':5000 '; then
    log_info "Port 5000 dinleniyor — bağlanabilirsiniz."
elif command -v netstat >/dev/null 2>&1 && netstat -tlnp 2>/dev/null | grep -q ':5000 '; then
    log_info "Port 5000 dinleniyor — bağlanabilirsiniz."
else
    log_warn "Port 5000 dinlenmiyor olabilir. Bağlanamıyorsanız:"
    log_warn "  systemd: sudo systemctl status wsanaliz-web && journalctl -u wsanaliz-web -n 30"
    log_warn "  manuel:  $ROOT/logs/worker.log ve gunicorn çıktısını kontrol edin."
fi
