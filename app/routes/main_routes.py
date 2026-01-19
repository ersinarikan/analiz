from flask import Blueprint, render_template, jsonify, current_app

import os
import time

main_bp = Blueprint('main', __name__)
"""
Ana uygulama blueprint'i.
- Ana sayfa, sağlık kontrolü ve temel endpointleri içerir.
"""

@main_bp.route('/')
def index():
    """
    Ana sayfa - Dosya yükleme ve analiz arayüzü.
    """
    return render_template('index.html')

@main_bp.route('/metrics')
def metrics():
    """
    Model metrikleri sayfası.
    """
    return render_template('metrics.html') 


@main_bp.route('/api/health', methods=['GET'])
def api_health():
    """
    Basit health endpoint'i (prod monitoring için).

    - Web prosesinin ayakta olduğunu doğrular.
    - DB ve Redis kontrollerini best-effort yapar (hata durumunda 200 dönüp 'degraded' işaretler).
    """
    checks: dict[str, object] = {}
    status = "ok"

    # DB check (best-effort)
    try:
        from sqlalchemy import text
        from app import db

        db.session.execute(text("SELECT 1"))
        checks["db"] = {"ok": True}
    except Exception as e:
        status = "degraded"
        checks["db"] = {"ok": False, "error": str(e)}
        try:
            from app import db  # noqa: F401
            db.session.rollback()
        except Exception:
            pass

    # Redis check (best-effort)
    try:
        import redis

        redis_url = (
            os.environ.get("WSANALIZ_REDIS_URL")
            or os.environ.get("SOCKETIO_MESSAGE_QUEUE")
            or "redis://localhost:6379/0"
        )
        r = redis.Redis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
        pong = r.ping()
        checks["redis"] = {"ok": bool(pong), "url": redis_url}
    except Exception as e:
        status = "degraded"
        checks["redis"] = {"ok": False, "error": str(e)}

    return jsonify(
        {
            "status": status,
            "ts": time.time(),
            "checks": checks,
            "queue_backend": os.environ.get("WSANALIZ_QUEUE_BACKEND"),
            "version": current_app.config.get("APP_VERSION", None),
        }
    ), 200