# NSFW Model Entegrasyonu - Performans Analizi ve Öneriler

## 📊 Mevcut Sistem Analizi

### Mevcut Adult Content Detection
- **Yöntem:** CLIP-based prompt matching
- **Model:** OpenCLIP ViT-H-14-378-quickgelu (2.3GB)
- **Inference Zamanı:** ~50-100ms/frame (GPU)
- **Bellek:** ~4-6GB GPU (CLIP + YOLO)
- **Doğruluk:** ~85-90% (prompt-based, genel amaçlı)

### Mevcut İşlem Akışı
```
Frame → YOLO (opsiyonel) → CLIP encode → Prompt matching → Score calculation
```

## 🎯 NSFW Model Seçenekleri

### Seçenek 1: Marqo/nsfw-image-detection-384 (ÖNERİLEN)
- **Model Tipi:** ViT-tiny (384x384)
- **Boyut:** ~18-20x daha küçük (CLIP'e göre)
- **Doğruluk:** %98.56
- **Inference:** ~10-20ms/frame (GPU, ONNX)
- **Bellek:** ~200-300MB ek GPU
- **Format:** ONNX (önerilen) veya PyTorch

### Seçenek 2: Falconsai/nsfw_image_detection
- **Model Tipi:** ResNet-based
- **Boyut:** Orta
- **Doğruluk:** ~95%
- **Inference:** ~30-50ms/frame

### Seçenek 3: vit-base-nsfw-detector
- **Model Tipi:** ViT-base
- **Boyut:** Büyük
- **Doğruluk:** ~97%
- **Inference:** ~80-120ms/frame

## ⚡ Performans Etkisi Analizi

### Senaryo 1: Her Frame'de NSFW Çalıştırma (KÖTÜ)
```
Mevcut: 50-100ms/frame
+ NSFW: +10-20ms/frame
Toplam: 60-120ms/frame
Yavaşlama: %20-40
```

### Senaryo 2: Conditional NSFW (ÖNERİLEN) ✅
```
CLIP adult_content > 0.3 → NSFW çalıştır
CLIP adult_content ≤ 0.3 → NSFW atla

Tahmini yavaşlama:
- %70-80 frame'de NSFW atlanır (CLIP skoru düşük)
- Sadece %20-30 frame'de NSFW çalışır
- Ortalama yavaşlama: %4-8 (çok düşük!)
```

### Senaryo 3: Lazy Loading + Conditional (EN İYİ) ✅✅
```
1. Model sadece gerektiğinde yüklenir (ilk yüksek skorlu frame'de)
2. Conditional execution (CLIP > threshold)
3. Model memory'de tutulur (sonraki frame'ler için)

Yavaşlama: %4-8 (conditional) + ilk yükleme 1-2 saniye (tek seferlik)
```

## 🏗️ Önerilen Entegrasyon Mimarisi

### 1. Lazy Loading Pattern
```python
class ContentAnalyzer:
    _nsfw_model = None
    _nsfw_model_loaded = False
    
    def _load_nsfw_model(self):
        """NSFW modelini sadece gerektiğinde yükle"""
        if self._nsfw_model_loaded:
            return self._nsfw_model
        
        # ONNX Runtime ile yükle (daha hızlı)
        import onnxruntime as ort
        model_path = "storage/models/nsfw/nsfw-detector-384.onnx"
        self._nsfw_model = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._nsfw_model_loaded = True
        return self._nsfw_model
```

### 2. Conditional Execution
```python
def analyze_image(self, image_path, ...):
    # Önce CLIP ile normal analiz
    violence_score, adult_content_score, ... = self._analyze_with_clip(...)
    
    # Sadece CLIP skoru yüksekse NSFW çalıştır
    if adult_content_score > 0.3:  # Threshold ayarlanabilir
        nsfw_score = self._analyze_with_nsfw_model(image_path)
        # NSFW skorunu CLIP skoru ile birleştir (weighted average)
        adult_content_score = 0.7 * adult_content_score + 0.3 * nsfw_score
    else:
        # Düşük skorlu frame'lerde NSFW atla (performans kazancı)
        pass
    
    return violence_score, adult_content_score, ...
```

### 3. Batch Processing (Video için)
```python
# Video analizinde: Yüksek skorlu frame'leri topla, batch'te NSFW çalıştır
high_risk_frames = [frame for frame in frames if clip_scores[frame] > 0.3]
if high_risk_frames:
    nsfw_scores = self._batch_nsfw_inference(high_risk_frames)  # Batch = daha hızlı
```

## 📈 Performans Metrikleri (Tahmini)

### Tek Frame Analizi
| Senaryo | Inference Zamanı | GPU Bellek | Yavaşlama |
|---------|------------------|------------|-----------|
| Mevcut (CLIP only) | 50-100ms | 4-6GB | Baseline |
| + NSFW (her frame) | 60-120ms | 4.2-6.3GB | +20-40% |
| + NSFW (conditional) | 52-110ms | 4.1-6.2GB | +4-8% ✅ |
| + NSFW (lazy+conditional) | 52-110ms | 4.1-6.2GB | +4-8% ✅ |

### Video Analizi (100 frame)
| Senaryo | Toplam Süre | Yavaşlama |
|---------|-------------|-----------|
| Mevcut | 5-10 saniye | Baseline |
| + NSFW (her frame) | 6-12 saniye | +20-40% |
| + NSFW (conditional) | 5.2-10.8 saniye | +4-8% ✅ |

## 🎛️ Yapılandırılabilir Parametreler

```python
# config.py veya environment variables
NSFW_ENABLED = True
NSFW_CLIP_THRESHOLD = 0.3  # CLIP skoru bu değerin üstündeyse NSFW çalıştır
NSFW_WEIGHT = 0.3  # NSFW skorunun final skora katkısı (0.3 = %30)
NSFW_MODEL_PATH = "storage/models/nsfw/nsfw-detector-384.onnx"
NSFW_USE_ONNX = True  # ONNX kullan (daha hızlı)
NSFW_BATCH_SIZE = 8  # Video için batch processing
```

## 🔧 Uygulama Adımları

### 1. Model İndirme ve Dönüştürme
```bash
# HuggingFace'den model indir
# PyTorch → ONNX dönüştür (daha hızlı inference için)
python scripts/convert_nsfw_to_onnx.py
```

### 2. ContentAnalyzer'a Entegrasyon
- Lazy loading pattern ekle
- Conditional execution logic ekle
- Weighted score combination

### 3. Test ve Benchmark
- Performans testleri (inference zamanı)
- Doğruluk testleri (CLIP vs NSFW vs Combined)
- Bellek kullanımı testleri

## ⚠️ Dikkat Edilmesi Gerekenler

1. **GPU Bellek:** NSFW model ek GPU bellek kullanır (~200-300MB)
2. **Model Yükleme:** İlk yükleme 1-2 saniye sürebilir (lazy loading ile minimize)
3. **Threshold Tuning:** CLIP threshold (0.3) test edilerek optimize edilmeli
4. **False Positives:** NSFW modelleri bazen yanlış pozitif verebilir, CLIP ile birleştirme önemli

## 📊 Sonuç ve Öneri

**ÖNERİLEN YAKLAŞIM:**
1. ✅ **Marqo/nsfw-image-detection-384** modeli (hafif, hızlı, doğru)
2. ✅ **ONNX format** (PyTorch'tan 12x daha hızlı)
3. ✅ **Lazy loading** (sadece gerektiğinde yükle)
4. ✅ **Conditional execution** (CLIP > 0.3 threshold)
5. ✅ **Weighted combination** (CLIP %70 + NSFW %30)

**BEKLENEN YAVAŞLAMA:** %4-8 (minimal, kabul edilebilir)

**BEKLENEN DOĞRULUK ARTIŞI:** %85-90 → %92-95 (CLIP + NSFW kombinasyonu)
