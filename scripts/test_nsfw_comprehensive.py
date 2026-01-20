#!/usr/bin/env python3
"""
NSFW Entegrasyonu Kapsamlı Test Scripti

Test senaryoları:
1. NSFW enabled - adult_content CLIP döngüsünden çıkarılmalı
2. NSFW disabled - adult_content CLIP döngüsünde olmalı
3. NSFW inference testi
4. Fallback testi (model yoksa)
"""

import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_scenario_1_nsfw_enabled():
    """Senaryo 1: NSFW enabled - adult_content CLIP döngüsünden çıkarılmalı"""
    logger.info("\n" + "=" * 60)
    logger.info("SENARYO 1: NSFW ENABLED")
    logger.info("=" * 60)
    
    from flask import Flask
    from app import create_app
    import numpy as np
    
    app = create_app()
    # NSFW'yi aktifleştir
    app.config['NSFW_ENABLED'] = True
    
    with app.app_context():
        from app.ai.content_analyzer import ContentAnalyzer
        ContentAnalyzer.reset_instance()  # Singleton'ı sıfırla
        analyzer = ContentAnalyzer()
        
        # Test görüntüsü
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # analyze_image çalıştır
        result = analyzer.analyze_image(test_image)
        violence, adult_content, harassment, weapon, drug, safe, objects = result
        
        logger.info(f"✅ Test tamamlandı")
        logger.info(f"   adult_content: {adult_content:.4f}")
        logger.info(f"   NSFW model yüklü: {analyzer._nsfw_model is not None}")
        
        # Kontrol: adult_content skoru NSFW'den gelmeli
        if analyzer._nsfw_model is not None:
            nsfw_score = analyzer._analyze_with_nsfw_model(test_image)
            if abs(adult_content - nsfw_score) < 0.01:
                logger.info(f"✅ adult_content skoru NSFW'den geliyor ({adult_content:.4f} ≈ {nsfw_score:.4f})")
                return True
            else:
                logger.error(f"❌ adult_content skoru NSFW'den gelmiyor ({adult_content:.4f} ≠ {nsfw_score:.4f})")
                return False
        else:
            logger.warning("⚠️ NSFW modeli yüklenemedi")
            return False

def test_scenario_2_nsfw_disabled():
    """Senaryo 2: NSFW disabled - adult_content CLIP döngüsünde olmalı"""
    logger.info("\n" + "=" * 60)
    logger.info("SENARYO 2: NSFW DISABLED")
    logger.info("=" * 60)
    
    from flask import Flask
    from app import create_app
    import numpy as np
    
    app = create_app()
    # NSFW'yi devre dışı bırak
    app.config['NSFW_ENABLED'] = False
    
    with app.app_context():
        from app.ai.content_analyzer import ContentAnalyzer
        ContentAnalyzer.reset_instance()  # Singleton'ı sıfırla
        analyzer = ContentAnalyzer()
        
        # Test görüntüsü
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # analyze_image çalıştır
        result = analyzer.analyze_image(test_image)
        violence, adult_content, harassment, weapon, drug, safe, objects = result
        
        logger.info(f"✅ Test tamamlandı")
        logger.info(f"   adult_content: {adult_content:.4f}")
        logger.info(f"   NSFW model yüklü: {analyzer._nsfw_model is not None}")
        
        # Kontrol: NSFW modeli yüklenmemeli
        if analyzer._nsfw_model is None:
            logger.info("✅ NSFW modeli yüklenmedi (NSFW_ENABLED=False)")
            return True
        else:
            logger.warning("⚠️ NSFW modeli yüklendi ama NSFW_ENABLED=False")
            return False

def test_scenario_3_fallback():
    """Senaryo 3: NSFW model yoksa fallback çalışmalı"""
    logger.info("\n" + "=" * 60)
    logger.info("SENARYO 3: FALLBACK (Model yoksa)")
    logger.info("=" * 60)
    
    from flask import Flask
    from app import create_app
    import numpy as np
    
    app = create_app()
    app.config['NSFW_ENABLED'] = True
    # Geçersiz model path
    original_path = app.config.get('NSFW_MODEL_PATH')
    app.config['NSFW_MODEL_PATH'] = '/nonexistent/path/model.onnx'
    
    with app.app_context():
        from app.ai.content_analyzer import ContentAnalyzer
        ContentAnalyzer.reset_instance()
        analyzer = ContentAnalyzer()
        
        # Test görüntüsü
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            # analyze_image çalıştır (fallback çalışmalı)
            result = analyzer.analyze_image(test_image)
            violence, adult_content, harassment, weapon, drug, safe, objects = result
            
            logger.info(f"✅ Fallback testi tamamlandı")
            logger.info(f"   adult_content: {adult_content:.4f}")
            logger.info(f"   NSFW model yüklü: {analyzer._nsfw_model is not None}")
            
            # Kontrol: adult_content 0.0 olmalı (NSFW model yok, CLIP'e sorulmuyor)
            if adult_content == 0.0:
                logger.info("✅ Fallback çalıştı (adult_content=0.0)")
                return True
            else:
                logger.warning(f"⚠️ Fallback çalışmadı (adult_content={adult_content:.4f})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fallback testi başarısız: {e}")
            return False
        finally:
            # Path'i geri yükle
            app.config['NSFW_MODEL_PATH'] = original_path

def main():
    """Tüm test senaryolarını çalıştır"""
    logger.info("NSFW Entegrasyonu Kapsamlı Test Başlatılıyor...")
    
    results = []
    
    # Senaryo 1: NSFW enabled
    try:
        results.append(("NSFW Enabled", test_scenario_1_nsfw_enabled()))
    except Exception as e:
        logger.error(f"Senaryo 1 hatası: {e}", exc_info=True)
        results.append(("NSFW Enabled", False))
    
    # Senaryo 2: NSFW disabled
    try:
        results.append(("NSFW Disabled", test_scenario_2_nsfw_disabled()))
    except Exception as e:
        logger.error(f"Senaryo 2 hatası: {e}", exc_info=True)
        results.append(("NSFW Disabled", False))
    
    # Senaryo 3: Fallback
    try:
        results.append(("Fallback", test_scenario_3_fallback()))
    except Exception as e:
        logger.error(f"Senaryo 3 hatası: {e}", exc_info=True)
        results.append(("Fallback", False))
    
    # Sonuçları özetle
    logger.info("\n" + "=" * 60)
    logger.info("TEST SONUÇLARI ÖZETİ")
    logger.info("=" * 60)
    
    for name, result in results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        logger.info(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ TÜM TESTLER BAŞARILI!")
    else:
        logger.error("❌ BAZI TESTLER BAŞARISIZ!")
    logger.info("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
