#!/usr/bin/env python3
"""
NSFW Model İndirme ve ONNX Dönüştürme Scripti

Bu script:
1. HuggingFace'den Marqo/nsfw-image-detection-384 modelini indirir
2. PyTorch modelini ONNX formatına dönüştürür (12x daha hızlı inference için)
3. Model dosyasını storage/models/nsfw/ konumuna kaydeder
4. Model metadata'sını JSON olarak kaydeder
"""

import os
import sys
import json
import logging
from pathlib import Path

# Proje root dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_and_convert_nsfw_model():
    """NSFW modelini indir ve ONNX formatına dönüştür"""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import onnxruntime as ort
        from PIL import Image
        import numpy as np
    except ImportError as e:
        logger.error(f"Gerekli kütüphaneler yüklü değil: {e}")
        logger.error("Lütfen şu paketleri yükleyin: torch, transformers, onnxruntime, pillow, numpy")
        return False
    
    # Model yolları
    model_name = "Marqo/nsfw-image-detection-384"
    models_dir = project_root / "storage" / "models" / "nsfw"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_model_path = models_dir / "nsfw-detector-384.onnx"
    metadata_path = models_dir / "metadata.json"
    
    # Model zaten varsa kontrol et
    if onnx_model_path.exists() and metadata_path.exists():
        logger.info(f"Model zaten mevcut: {onnx_model_path}")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata: {metadata}")
            return True
        except Exception as e:
            logger.warning(f"Metadata okunamadı, yeniden indiriliyor: {e}")
    
    logger.info(f"NSFW modeli indiriliyor: {model_name}")
    
    # 1. Model ve processor'ı yükle
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()  # Inference modu
        logger.info("✅ Model ve processor başarıyla yüklendi")
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        return False
    
    # 2. Model metadata'sını kaydet
    metadata = {
        "model_name": model_name,
        "input_size": 384,
        "normalization": {
            "mean": processor.image_mean if hasattr(processor, 'image_mean') else [0.485, 0.456, 0.406],
            "std": processor.image_std if hasattr(processor, 'image_std') else [0.229, 0.224, 0.225]
        },
        "num_classes": model.config.num_labels if hasattr(model.config, 'num_labels') else 2,
        "model_type": "binary_classification",
        "output_format": "probability"
    }
    
    # 3. ONNX formatına dönüştür
    logger.info("Model ONNX formatına dönüştürülüyor...")
    try:
        # Örnek input (384x384 RGB image)
        dummy_input = torch.randn(1, 3, 384, 384)
        
        # ONNX export
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_model_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True
        )
        logger.info(f"✅ Model ONNX formatına dönüştürüldü: {onnx_model_path}")
    except Exception as e:
        logger.error(f"ONNX dönüştürme hatası: {e}")
        return False
    
    # 4. ONNX modelini test et
    logger.info("ONNX modeli test ediliyor...")
    try:
        ort_session = ort.InferenceSession(str(onnx_model_path))
        test_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
        outputs = ort_session.run(None, {'input': test_input})
        logger.info(f"✅ ONNX model test başarılı. Output shape: {outputs[0].shape}")
    except Exception as e:
        logger.error(f"ONNX model test hatası: {e}")
        return False
    
    # 5. Metadata'yı kaydet
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata kaydedildi: {metadata_path}")
    except Exception as e:
        logger.error(f"Metadata kaydetme hatası: {e}")
        return False
    
    logger.info("=" * 60)
    logger.info("✅ NSFW modeli başarıyla indirildi ve dönüştürüldü!")
    logger.info(f"   Model: {onnx_model_path}")
    logger.info(f"   Metadata: {metadata_path}")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = download_and_convert_nsfw_model()
    sys.exit(0 if success else 1)
