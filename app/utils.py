#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Yardımcı Fonksiyonlar
==============================

Bu modül uygulamada kullanılan genel amaçlı yardımcı fonksiyonları içerir.
Veri tipi dönüşümleri ve JSON serileştirme işlemleri için kullanılır.
"""

import numpy as np

def numpy_to_python(obj):
    """
    NumPy veri tiplerini Python native tiplerine dönüştürür
    
    Args:
        obj: Dönüştürülecek NumPy nesnesi
        
    Returns:
        Python native tipi (int, float, list)
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj 