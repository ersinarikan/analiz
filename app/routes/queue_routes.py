"""Queue management routes for analysis processing"""

from flask import Blueprint, jsonify, request
import logging
import os
import shutil
from app.services.queue_service import get_queue_status, get_queue_stats, clear_queue

logger = logging.getLogger(__name__)

queue_bp = Blueprint('queue', __name__, url_prefix='/api/queue')
"""
Analiz kuyruğu için blueprint.
- Analiz işlemlerinin kuyruk yönetimi endpointlerini içerir.
"""

@queue_bp.route('/status', methods=['GET'])
def get_queue_status_route():
    """Get current queue status"""
    try:
        status = get_queue_status()
        return jsonify({
            'status': 'success',
            'data': status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@queue_bp.route('/stats', methods=['GET'])
def get_queue_stats_route():
    """Get queue statistics"""
    try:
        stats = get_queue_stats()
        return jsonify({
            'status': 'success',
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 

@queue_bp.route('/stop', methods=['POST'])
def stop_queue_route():
    """Stop all analyses and clear queue and uploads"""
    try:
        # Kuyruğu temizle
        cleared_count = clear_queue()
        logger.info(f"Kuyruktan {cleared_count} analiz temizlendi")
        
        # Upload klasörünü temizle
        upload_path = os.path.join('storage', 'uploads')
        if os.path.exists(upload_path):
            for filename in os.listdir(upload_path):
                file_path = os.path.join(upload_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Upload dosyası silinemedi {file_path}: {e}")
            logger.info(f"Upload klasörü temizlendi: {upload_path}")
        
        # Processed klasörünü temizle (isteğe bağlı)
        processed_path = os.path.join('storage', 'processed')
        if os.path.exists(processed_path):
            for filename in os.listdir(processed_path):
                file_path = os.path.join(processed_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Processed dosyası silinemedi {file_path}: {e}")
            logger.info(f"Processed klasörü temizlendi: {processed_path}")
        
        return jsonify({
            'status': 'success',
            'message': f'Kuyruk temizlendi ({cleared_count} analiz), upload ve processed klasörleri temizlendi'
        }), 200
        
    except Exception as e:
        logger.error(f"Kuyruk durdurma hatası: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500