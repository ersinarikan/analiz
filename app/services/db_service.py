"""Database service for optimized query operations"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Type
from contextlib import contextmanager
from functools import wraps

from flask import current_app
from sqlalchemy.orm import joinedload, selectinload, subqueryload
from sqlalchemy import text, and_, or_
from app import db

logger = logging.getLogger(__name__)

# Query cache for performance optimization
_query_cache = {}
_cache_lock = threading.Lock()
_cache_ttl = 300  # 5 minutes

def cache_query(ttl_seconds=300):
    """
    Query result caching decorator
    
    Args:
        ttl_seconds (int): Cache TTL in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            with _cache_lock:
                # Check cache
                if cache_key in _query_cache:
                    cached_result, cached_time = _query_cache[cache_key]
                    if (datetime.now() - cached_time).total_seconds() < ttl_seconds:
                        logger.debug(f"Query cache hit: {cache_key}")
                        return cached_result
                    else:
                        # Remove expired cache
                        del _query_cache[cache_key]
            
            # Execute query
            start_time = time.time()
            result = func(*args, **kwargs)
            query_time = time.time() - start_time
            
            # Cache result
            with _cache_lock:
                _query_cache[cache_key] = (result, datetime.now())
            
            logger.debug(f"Query executed and cached: {cache_key} ({query_time:.3f}s)")
            return result
        return wrapper
    return decorator

def clear_query_cache():
    """Clear all query cache"""
    with _cache_lock:
        _query_cache.clear()
        logger.info("Query cache cleared")

@contextmanager
def safe_database_session():
    """
    Thread-safe database session context manager with automatic rollback
    """
    session = db.session
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def query_db(model, **kwargs):
    """
    Veritabanında belirtilen modeli filtreleyerek arar.
    
    Args:
        model: Sorgulanacak model sınıfı
        **kwargs: Sorgu filtreleri
        
    Returns:
        model ya da None: Eşleşen model örneği veya bulunamazsa None
    """
    try:
        return model.query.filter_by(**kwargs).first()
    except Exception as e:
        current_app.logger.error(f"Veritabanı sorgu hatası: {str(e)}")
        return None
        
def query_all_db(model, **kwargs):
    """
    Veritabanında belirtilen modeli filtreleyerek arar ve tüm sonuçları döndürür.
    
    Args:
        model: Sorgulanacak model sınıfı
        **kwargs: Sorgu filtreleri
        
    Returns:
        list: Eşleşen model örneklerinin listesi
    """
    try:
        return model.query.filter_by(**kwargs).all()
    except Exception as e:
        current_app.logger.error(f"Veritabanı sorgu hatası: {str(e)}")
        return []

# Performance-optimized query functions

@cache_query(ttl_seconds=180)
def get_analysis_with_relations(analysis_id: str, include_content_detections=True, include_age_estimations=True):
    """
    Analysis'i ilişkili verilerle beraber tek sorguda getirir (Eager Loading)
    
    Args:
        analysis_id (str): Analysis ID
        include_content_detections (bool): Content detections dahil edilsin mi
        include_age_estimations (bool): Age estimations dahil edilsin mi
        
    Returns:
        Analysis: Analysis nesnesi veya None
    """
    try:
        from app.models.analysis import Analysis
        
        query = db.session.query(Analysis)
        
        # Eager loading with joinedload for better performance
        if include_content_detections:
            query = query.options(joinedload(Analysis.content_detections))
        
        if include_age_estimations:
            query = query.options(joinedload(Analysis.age_estimations))
        
        # File relationship
        query = query.options(joinedload(Analysis.file))
        
        result = query.filter(Analysis.id == analysis_id).first()
        
        logger.debug(f"Analysis {analysis_id} loaded with eager loading")
        return result
        
    except Exception as e:
        logger.error(f"Error loading analysis with relations: {e}")
        return None

@cache_query(ttl_seconds=300)
def get_analyses_by_file_optimized(file_id: str, limit: int = 10):
    """
    File'a ait analizleri optimize edilmiş şekilde getirir
    
    Args:
        file_id (str): File ID
        limit (int): Maksimum sonuç sayısı
        
    Returns:
        List[Analysis]: Analysis listesi
    """
    try:
        from app.models.analysis import Analysis
        
        analyses = db.session.query(Analysis)\
            .filter(Analysis.file_id == file_id)\
            .options(
                selectinload(Analysis.content_detections),
                selectinload(Analysis.age_estimations)
            )\
            .order_by(Analysis.created_at.desc())\
            .limit(limit)\
            .all()
            
        logger.debug(f"Loaded {len(analyses)} analyses for file {file_id}")
        return analyses
        
    except Exception as e:
        logger.error(f"Error loading analyses for file {file_id}: {e}")
        return []

@cache_query(ttl_seconds=600)
def get_feedbacks_for_training_optimized(model_type='age', limit=1000):
    """
    Eğitim için feedbacks'leri optimize edilmiş şekilde getirir
    
    Args:
        model_type (str): Model tipi ('age', 'content')
        limit (int): Maksimum sonuç sayısı
        
    Returns:
        List[Feedback]: Feedback listesi
    """
    try:
        from app.models.feedback import Feedback
        
        query = db.session.query(Feedback)
        
        if model_type == 'age':
            query = query.filter(
                or_(
                    Feedback.feedback_type == 'age',
                    Feedback.feedback_type == 'age_pseudo'
                )
            ).filter(
                Feedback.embedding.isnot(None)
            ).filter(
                or_(
                    Feedback.training_status.is_(None),
                    Feedback.training_status != 'used_in_training'
                )
            )
        
        feedbacks = query.order_by(Feedback.created_at.desc())\
            .limit(limit)\
            .all()
            
        logger.debug(f"Loaded {len(feedbacks)} feedbacks for {model_type} training")
        return feedbacks
        
    except Exception as e:
        logger.error(f"Error loading feedbacks for training: {e}")
        return []

def get_paginated_analyses(page=1, per_page=20, status_filter=None, file_type_filter=None):
    """
    Pagination ile analizleri getirir - memory efficient
    
    Args:
        page (int): Sayfa numarası
        per_page (int): Sayfa başına kayıt sayısı
        status_filter (str): Status filtresi
        file_type_filter (str): File type filtresi
        
    Returns:
        dict: {'items': [], 'total': int, 'page': int, 'per_page': int}
    """
    try:
        from app.models.analysis import Analysis
        from app.models.file import File
        
        query = db.session.query(Analysis)\
            .join(File)\
            .options(
                joinedload(Analysis.file),
                selectinload(Analysis.content_detections),
                selectinload(Analysis.age_estimations)
            )
        
        # Filters
        if status_filter:
            query = query.filter(Analysis.status == status_filter)
        
        if file_type_filter:
            query = query.filter(File.file_type == file_type_filter)
        
        # Count total for pagination
        total = query.count()
        
        # Apply pagination
        items = query.order_by(Analysis.created_at.desc())\
            .offset((page - 1) * per_page)\
            .limit(per_page)\
            .all()
        
        logger.debug(f"Paginated query: page {page}, {len(items)}/{total} results")
        
        return {
            'items': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'has_next': page * per_page < total,
            'has_prev': page > 1
        }
        
    except Exception as e:
        logger.error(f"Error in paginated analyses query: {e}")
        return {'items': [], 'total': 0, 'page': page, 'per_page': per_page}

def bulk_update_training_status(feedback_ids: List[int], status: str, model_version: str = None):
    """
    Bulk update training status - efficient batch operation
    
    Args:
        feedback_ids (List[int]): Feedback ID listesi
        status (str): Yeni status
        model_version (str): Model versiyonu
    """
    try:
        from app.models.feedback import Feedback
        
        update_data = {
            'training_status': status,
            'training_used_at': datetime.now()
        }
        
        if model_version:
            update_data['used_in_model_version'] = model_version
        
        db.session.query(Feedback)\
            .filter(Feedback.id.in_(feedback_ids))\
            .update(update_data, synchronize_session=False)
        
        db.session.commit()
        
        logger.info(f"Bulk updated {len(feedback_ids)} feedback statuses to {status}")
        
    except Exception as e:
        logger.error(f"Error in bulk update training status: {e}")
        db.session.rollback()
        raise

def get_database_stats():
    """
    Database performans istatistiklerini getirir
    
    Returns:
        dict: Database stats
    """
    try:
        stats = {
            'query_cache_size': len(_query_cache),
            'query_cache_keys': list(_query_cache.keys()) if len(_query_cache) < 10 else f"{len(_query_cache)} keys",
            'database_status': 'connected'
        }
        
        # Table counts
        from app.models.analysis import Analysis
        from app.models.feedback import Feedback
        from app.models.file import File
        
        stats['table_counts'] = {
            'analyses': db.session.query(Analysis).count(),
            'feedbacks': db.session.query(Feedback).count(),
            'files': db.session.query(File).count()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {'error': str(e)}

# Performance monitoring
def log_slow_queries(threshold_seconds=1.0):
    """
    Yavaş sorguları logla
    
    Args:
        threshold_seconds (float): Threshold for slow query logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                logger.warning(f"Slow query detected: {func.__name__} took {execution_time:.3f}s")
            
            return result
        return wrapper
    return decorator

def save_to_db(model_instance):
    """
    Veritabanı modelini kaydeder.
    
    Args:
        model_instance: Kaydedilecek model nesnesi
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
        
    Raises:
        Exception: Veritabanı işlemi sırasında oluşan hatalar
    """
    try:
        db.session.add(model_instance)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Veritabanına kaydetme hatası: {str(e)}")
        raise

def update_db(model_instance, **kwargs):
    """
    Veritabanı model örneğini günceller.
    
    Args:
        model_instance: Güncellenecek model örneği
        **kwargs: Güncellenecek alanlar ve değerleri
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
    """
    try:
        for key, value in kwargs.items():
            setattr(model_instance, key, value)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Veritabanı güncelleme hatası: {str(e)}")
        return False
        
def delete_from_db(model_instance):
    """
    Veritabanından model örneğini siler.
    
    Args:
        model_instance: Silinecek model örneği
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
    """
    try:
        db.session.delete(model_instance)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Veritabanından silme hatası: {str(e)}")
        return False 