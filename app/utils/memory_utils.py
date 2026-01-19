"""Memory management utilities for performance optimization"""

import gc
import psutil
import torch
import logging
import time
import threading
from typing import Dict, Any
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

# Memory monitoring
_memory_stats = {}
_stats_lock = threading.Lock()
_last_cleanup = datetime.now()

def get_memory_usage() -> Dict[str, Any]:
    """
    Sistem memory kullanımını döndürür
    
    Returns:
        dict: Memory usage statistics
    """
    try:
        # System memory
        memory = psutil.virtual_memory()
        
        stats = {
            'system': {
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'used': memory.used / (1024**3),  # GB
                'percentage': memory.percent
            },
            'python_process': {
                'rss': psutil.Process().memory_info().rss / (1024**2),  # MB
                'vms': psutil.Process().memory_info().vms / (1024**2),  # MB
            }
        }
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_stats = {}
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_stats(device=i)
                gpu_stats[f'gpu_{i}'] = {
                    'allocated': gpu_memory.get('allocated_bytes.all.current', 0) / (1024**2),  # MB
                    'reserved': gpu_memory.get('reserved_bytes.all.current', 0) / (1024**2),  # MB
                    'max_allocated': gpu_memory.get('allocated_bytes.all.peak', 0) / (1024**2),  # MB
                }
            stats['gpu'] = gpu_stats
        else:
            stats['gpu'] = {'status': 'not_available'}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {'error': str(e)}

def cleanup_memory(force=False):
    """
    Memory cleanup işlemi
    
    Args:
        force (bool): Zorla cleanup yap
    """
    global _last_cleanup
    
    # Minimum 30 saniye ara ile cleanup
    if not force and (datetime.now() - _last_cleanup).total_seconds() < 30:
        return
    
    try:
        logger.info("Memory cleanup başlatılıyor...")
        
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection: {collected} objects collected")
        
        # GPU memory cleanup (if available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache temizlendi")
        
        # Clear model cache if memory usage is critically high
        memory_stats = get_memory_usage()
        if memory_stats.get('system', {}).get('percentage', 0) > 92:
            logger.warning("Critical memory usage detected, clearing model cache")
            from app.utils.model_state import clear_model_cache
            clear_model_cache()
        elif memory_stats.get('system', {}).get('percentage', 0) > 88:
            # Sadece GPU cache temizle, model instance'ları korur
            logger.warning("High memory usage detected, clearing GPU cache only")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear database query cache only on critical memory
            from app.services.db_service import clear_query_cache
            clear_query_cache()
        
        _last_cleanup = datetime.now()
        logger.info("Memory cleanup tamamlandı")
        
    except Exception as e:
        logger.error(f"Memory cleanup error: {e}")

@contextmanager
def memory_monitor(operation_name: str, cleanup_threshold_mb: int = 1000):
    """
    Memory monitoring context manager
    
    Args:
        operation_name (str): İşlem adı
        cleanup_threshold_mb (int): Cleanup threshold (MB)
    """
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        logger.debug(f"Memory monitor başlatıldı: {operation_name}")
        yield
        
    finally:
        end_memory = get_memory_usage()
        end_time = time.time()
        
        # Calculate memory difference
        start_rss = start_memory.get('python_process', {}).get('rss', 0)
        end_rss = end_memory.get('python_process', {}).get('rss', 0)
        memory_diff = end_rss - start_rss
        
        execution_time = end_time - start_time
        
        logger.info(f"Memory monitor - {operation_name}: "
                   f"Time: {execution_time:.2f}s, "
                   f"Memory diff: {memory_diff:.1f}MB")
        
        # Auto cleanup if memory usage is high
        if memory_diff > cleanup_threshold_mb:
            logger.warning(f"High memory usage detected in {operation_name}, triggering cleanup")
            cleanup_memory(force=True)

def memory_profiler(func):
    """
    Function memory profiling decorator
    """
    def wrapper(*args, **kwargs):
        with memory_monitor(func.__name__):
            return func(*args, **kwargs)
    return wrapper

class MemoryManager:
    """Memory management class for tracking and optimizing memory usage"""
    
    def __init__(self, max_memory_percent: int = 80, cleanup_interval: int = 300):
        """
        Initialize memory manager
        
        Args:
            max_memory_percent (int): Maximum memory usage percentage
            cleanup_interval (int): Cleanup interval in seconds
        """
        self.max_memory_percent = max_memory_percent
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = datetime.now()
        self.monitoring_enabled = True
        
    def start_monitoring(self):
        """Start background memory monitoring"""
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    memory_stats = get_memory_usage()
                    memory_percent = memory_stats.get('system', {}).get('percentage', 0)
                    
                    if memory_percent > self.max_memory_percent:
                        logger.warning(f"High memory usage: {memory_percent:.1f}%")
                        cleanup_memory(force=True)
                    
                    time.sleep(self.cleanup_interval)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(60)  # Wait before retry
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_enabled = False
        logger.info("Memory monitoring stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return get_memory_usage()

# GPU Memory Management
class GPUMemoryManager:
    """GPU memory management for model operations"""
    
    @staticmethod
    def optimize_gpu_memory():
        """Optimize GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable memory pool
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
            
            logger.info("GPU memory optimized")
            
        except Exception as e:
            logger.warning(f"GPU memory optimization failed: {e}")
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {'status': 'cuda_not_available'}
        
        info = {}
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved(device) / (1024**2)   # MB
            
            info[f'gpu_{i}'] = {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'device_name': torch.cuda.get_device_name(i)
            }
        
        return info

# Global memory manager instance
memory_manager = MemoryManager()

def initialize_memory_management() -> None:
    """
    Bellek yönetimi için gerekli ayarları başlatır.
    Returns:
        None
    """
    try:
        # Optimize GPU memory
        GPUMemoryManager.optimize_gpu_memory()
        
        # Start memory monitoring
        memory_manager.start_monitoring()
        
        logger.info("Memory management initialized")
        
    except Exception as e:
        logger.error(f"Memory management initialization failed: {e}")

def get_memory_recommendations() -> Dict[str, Any]:
    """
    Memory kullanımı önerilerini döndürür
    
    Returns:
        dict: Memory optimization recommendations
    """
    try:
        memory_stats = get_memory_usage()
        recommendations = []
        
        # System memory recommendations
        system_usage = memory_stats.get('system', {}).get('percentage', 0)
        if system_usage > 90:
            recommendations.append({
                'type': 'critical',
                'message': 'Critical memory usage! Consider restarting the application.',
                'action': 'restart_required'
            })
        elif system_usage > 80:
            recommendations.append({
                'type': 'warning',
                'message': 'High memory usage. Consider clearing caches.',
                'action': 'clear_cache'
            })
        
        # GPU memory recommendations
        gpu_stats = memory_stats.get('gpu', {})
        if isinstance(gpu_stats, dict) and gpu_stats.get('status') != 'not_available':
            for gpu_id, gpu_info in gpu_stats.items():
                if isinstance(gpu_info, dict):
                    allocated = gpu_info.get('allocated', 0)
                    if allocated > 8000:  # 8GB
                        recommendations.append({
                            'type': 'warning',
                            'message': f'High GPU memory usage on {gpu_id}: {allocated:.0f}MB',
                            'action': 'clear_gpu_cache'
                        })
        
        return {
            'memory_stats': memory_stats,
            'recommendations': recommendations,
            'last_cleanup': _last_cleanup.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating memory recommendations: {e}")
        return {'error': str(e)} 