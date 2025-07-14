"""Performance monitoring and optimization routes"""

from flask import Blueprint, jsonify, request
from app.utils.memory_utils import (
    get_memory_usage, cleanup_memory, get_memory_recommendations,
    memory_manager, GPUMemoryManager
)
from app.utils.model_state import get_cache_stats, clear_model_cache
from app.services.db_service import get_database_stats, clear_query_cache
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

performance_bp = Blueprint('performance', __name__)
"""
Performans ölçümü için blueprint.
- Sistem ve model performansını izlemeye yönelik endpointleri içerir.
"""

@performance_bp.route('/memory', methods=['GET'])
def get_memory_stats():
    """Get current memory usage statistics"""
    try:
        memory_stats = get_memory_usage()
        return jsonify({
            'status': 'success',
            'data': memory_stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/memory/cleanup', methods=['POST'])
def trigger_memory_cleanup():
    """Trigger manual memory cleanup"""
    try:
        force = request.json.get('force', False) if request.is_json else False
        
        start_time = time.time()
        cleanup_memory(force=force)
        cleanup_time = time.time() - start_time
        
        # Get memory stats after cleanup
        memory_stats = get_memory_usage()
        
        return jsonify({
            'status': 'success',
            'message': 'Memory cleanup completed',
            'cleanup_time': f"{cleanup_time:.2f}s",
            'memory_after_cleanup': memory_stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/memory/recommendations', methods=['GET'])
def get_memory_recommendations_endpoint():
    """Get memory optimization recommendations"""
    try:
        recommendations = get_memory_recommendations()
        return jsonify({
            'status': 'success',
            'data': recommendations,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting memory recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/gpu', methods=['GET'])
def get_gpu_stats():
    """Get GPU memory and usage statistics"""
    try:
        gpu_info = GPUMemoryManager.get_gpu_memory_info()
        return jsonify({
            'status': 'success',
            'data': gpu_info,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/gpu/cleanup', methods=['POST'])
def trigger_gpu_cleanup():
    """Trigger GPU cache cleanup"""
    try:
        GPUMemoryManager.clear_gpu_cache()
        gpu_info = GPUMemoryManager.get_gpu_memory_info()
        
        return jsonify({
            'status': 'success',
            'message': 'GPU cache cleared',
            'gpu_info_after_cleanup': gpu_info,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error during GPU cleanup: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/models', methods=['GET'])
def get_model_cache_stats():
    """Get model cache statistics"""
    try:
        cache_stats = get_cache_stats()
        return jsonify({
            'status': 'success',
            'data': cache_stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model cache stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/models/clear', methods=['POST'])
def clear_model_cache_endpoint():
    """Clear model cache"""
    try:
        model_type = request.json.get('model_type') if request.is_json else None
        
        clear_model_cache(model_type)
        
        # Get updated stats
        cache_stats = get_cache_stats()
        
        return jsonify({
            'status': 'success',
            'message': f'Model cache cleared{"" if not model_type else f" for {model_type}"}',
            'cache_stats_after': cache_stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing model cache: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/database', methods=['GET'])
def get_database_performance():
    """Get database performance statistics"""
    try:
        db_stats = get_database_stats()
        return jsonify({
            'status': 'success',
            'data': db_stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/database/clear-cache', methods=['POST'])
def clear_database_cache():
    """Clear database query cache"""
    try:
        clear_query_cache()
        db_stats = get_database_stats()
        
        return jsonify({
            'status': 'success',
            'message': 'Database query cache cleared',
            'database_stats_after': db_stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing database cache: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/overview', methods=['GET'])
def get_performance_overview():
    """Get comprehensive performance overview"""
    try:
        # Collect all performance data
        memory_stats = get_memory_usage()
        gpu_info = GPUMemoryManager.get_gpu_memory_info()
        cache_stats = get_cache_stats()
        db_stats = get_database_stats()
        recommendations = get_memory_recommendations()
        
        overview = {
            'memory': memory_stats,
            'gpu': gpu_info,
            'model_cache': cache_stats,
            'database': db_stats,
            'recommendations': recommendations,
            'system_health': _calculate_system_health(memory_stats, gpu_info)
        }
        
        return jsonify({
            'status': 'success',
            'data': overview,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting performance overview: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@performance_bp.route('/optimize', methods=['POST'])
def optimize_system():
    """Perform comprehensive system optimization"""
    try:
        optimization_steps = []
        start_time = time.time()
        
        # Step 1: Memory cleanup
        try:
            cleanup_memory(force=True)
            optimization_steps.append({'step': 'memory_cleanup', 'status': 'success'})
        except Exception as e:
            optimization_steps.append({'step': 'memory_cleanup', 'status': 'error', 'error': str(e)})
        
        # Step 2: GPU cleanup
        try:
            GPUMemoryManager.clear_gpu_cache()
            optimization_steps.append({'step': 'gpu_cleanup', 'status': 'success'})
        except Exception as e:
            optimization_steps.append({'step': 'gpu_cleanup', 'status': 'error', 'error': str(e)})
        
        # Step 3: Clear query cache
        try:
            clear_query_cache()
            optimization_steps.append({'step': 'database_cache_clear', 'status': 'success'})
        except Exception as e:
            optimization_steps.append({'step': 'database_cache_clear', 'status': 'error', 'error': str(e)})
        
        # Step 4: Clear model cache if memory usage is high
        memory_stats = get_memory_usage()
        if memory_stats.get('system', {}).get('percentage', 0) > 80:
            try:
                clear_model_cache()
                optimization_steps.append({'step': 'model_cache_clear', 'status': 'success'})
            except Exception as e:
                optimization_steps.append({'step': 'model_cache_clear', 'status': 'error', 'error': str(e)})
        
        total_time = time.time() - start_time
        
        # Get final stats
        final_memory = get_memory_usage()
        final_gpu = GPUMemoryManager.get_gpu_memory_info()
        
        return jsonify({
            'status': 'success',
            'message': 'System optimization completed',
            'optimization_steps': optimization_steps,
            'optimization_time': f"{total_time:.2f}s",
            'final_stats': {
                'memory': final_memory,
                'gpu': final_gpu
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error during system optimization: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def _calculate_system_health(memory_stats, gpu_info):
    """Calculate overall system health score"""
    try:
        health_score = 100
        issues = []
        
        # Memory health
        memory_usage = memory_stats.get('system', {}).get('percentage', 0)
        if memory_usage > 90:
            health_score -= 30
            issues.append('Critical memory usage')
        elif memory_usage > 80:
            health_score -= 15
            issues.append('High memory usage')
        elif memory_usage > 70:
            health_score -= 5
            issues.append('Moderate memory usage')
        
        # GPU health (if available)
        if isinstance(gpu_info, dict) and 'status' not in gpu_info:
            for gpu_id, gpu_data in gpu_info.items():
                if isinstance(gpu_data, dict):
                    allocated = gpu_data.get('allocated_mb', 0)
                    if allocated > 10000:  # 10GB
                        health_score -= 20
                        issues.append(f'High GPU memory usage on {gpu_id}')
                    elif allocated > 8000:  # 8GB
                        health_score -= 10
                        issues.append(f'Moderate GPU memory usage on {gpu_id}')
        
        # Determine health status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 80:
            status = 'good'
        elif health_score >= 70:
            status = 'fair'
        elif health_score >= 60:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'score': health_score,
            'status': status,
            'issues': issues
        }
        
    except Exception as e:
        logger.error(f"Error calculating system health: {e}")
        return {
            'score': 0,
            'status': 'unknown',
            'issues': [f'Health calculation error: {str(e)}']
        } 