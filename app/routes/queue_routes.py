"""Queue management routes for analysis processing"""

from flask import Blueprint, jsonify, request
from app.services.queue_service import get_queue_status, get_queue_stats
import logging

logger = logging.getLogger(__name__)

queue_bp = Blueprint('queue', __name__, url_prefix='/api/queue')

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