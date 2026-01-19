#!/usr/bin/env python3
"""
OpenCLIP İçerik Modeli Eğitim Script'i

Bu script feedback verilerini kullanarak OpenCLIP modelini fine-tune eder.

Kullanım:
    python train_content_model.py                          # Varsayılan parametrelerle
    python train_content_model.py --epochs 15 --batch-size 32 --learning-rate 0.0005
    python train_content_model.py --dry-run               # Sadece analiz
    python train_content_model.py --min-samples 50        # Minimum örnek sayısı
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Proje root'unu sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.content_training_service import ContentTrainingService
from app.services.clip_training_service import ClipTrainingService

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'clip_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger('train_content_model')

def print_banner():
    """Hoş geldin banner'ı yazdır"""
    print("=" * 70)
    print("🤖 OpenCLIP İçerik Modeli Eğitim Script'i")
    print("📊 Feedback verilerinden CLIP fine-tuning")
    print("=" * 70)
    print()

def print_analysis_results(analysis):
    """Analiz sonuçlarını yazdır"""
    print("📊 VERİ ANALİZİ SONUÇLARI")
    print("-" * 50)
    
    feedback_data = analysis.get('feedback_analysis', {})
    quality_data = analysis.get('data_quality', {})
    recommendation = analysis.get('training_recommendation', {})
    
    # Feedback istatistikleri
    print(f"📋 Toplam Feedback: {feedback_data.get('total_feedback', 0)}")
    print(f"✅ Geçerli Feedback: {feedback_data.get('valid_feedback', 0)}")
    print(f"💬 Yorum Sayısı: {feedback_data.get('comment_count', 0)}")
    print(f"⭐ Rating Sayısı: {feedback_data.get('rating_count', 0)}")
    print(f"📅 Son 30 Gün: {feedback_data.get('recent_feedback_30d', 0)}")
    print()
    
    # Kategori dağılımı
    print("📊 KATEGORİ DAĞILIMI:")
    categories = feedback_data.get('category_distribution', {})
    for category, counts in categories.items():
        total = sum(counts.values())
        high_count = counts.get('high', 0)
        low_count = counts.get('low', 0)
        print(f"  {category:12}: Toplam={total:3d}, Yüksek={high_count:2d}, Düşük={low_count:2d}")
    print()
    
    # Veri kalitesi
    print("🔍 VERİ KALİTESİ:")
    print(f"  Toplam Örnek: {quality_data.get('total_samples', 0)}")
    print(f"  Geçerli Dosya: {quality_data.get('valid_files', 0)}")
    print(f"  Eksik Dosya: {quality_data.get('missing_files', 0)}")
    print(f"  Bozuk Dosya: {quality_data.get('corrupted_files', 0)}")
    print(f"  Kalite Skoru: {quality_data.get('quality_score', 0.0):.2f}/1.00")
    print()
    
    # Öneriler
    print("💡 EĞİTİM ÖNERİSİ:")
    recommended = recommendation.get('recommended', False)
    confidence = recommendation.get('confidence', 0.0)
    
    print(f"  Önerilen: {'✅ EVET' if recommended else '❌ HAYIR'}")
    print(f"  Güven: {confidence:.2f}/1.00")
    
    reasons = recommendation.get('reasons', [])
    warnings = recommendation.get('warnings', [])
    
    if reasons:
        print("  ✅ Pozitif Faktörler:")
        for reason in reasons:
            print(f"    • {reason}")
    
    if warnings:
        print("  ⚠️  Uyarılar:")
        for warning in warnings:
            print(f"    • {warning}")
    
    print()

def print_training_params(params):
    """Training parametrelerini yazdır"""
    print("⚙️  EĞİTİM PARAMETRELERİ:")
    print(f"  Epochs: {params.get('epochs', 10)}")
    print(f"  Batch Size: {params.get('batch_size', 16)}")
    print(f"  Learning Rate: {params.get('learning_rate', 1e-4)}")
    print(f"  Patience: {params.get('patience', 3)}")
    print()

def print_training_results(result):
    """Training sonuçlarını yazdır"""
    print("🎯 EĞİTİM SONUÇLARI")
    print("-" * 50)
    
    if result['success']:
        print("✅ Eğitim başarıyla tamamlandı!")
        print(f"📝 Training Session ID: {result['training_session_id']}")
        print(f"💾 Model Yolu: {result['model_path']}")
        
        performance = result.get('performance', {})
        print(f"📉 Final Train Loss: {performance.get('final_train_loss', 0.0):.4f}")
        print(f"📊 Final Val Loss: {performance.get('final_val_loss', 0.0):.4f}")
        print(f"🔄 Tamamlanan Epoch: {performance.get('epochs_completed', 0)}")
        
        data_stats = result.get('training_data_stats', {})
        print(f"📊 Train Örnekleri: {data_stats.get('train_samples', 0)}")
        print(f"🔍 Val Örnekleri: {data_stats.get('val_samples', 0)}")
        
    else:
        print("❌ Eğitim başarısız!")
        print(f"🚫 Hata: {result.get('error', 'Bilinmeyen hata')}")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description='OpenCLIP İçerik Modeli Eğitim Script\'i',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python train_content_model.py                           # Varsayılan parametreler
  python train_content_model.py --dry-run                # Sadece analiz
  python train_content_model.py --epochs 15 --batch-size 32
  python train_content_model.py --min-samples 50         # Minimum örnek sayısı
        """
    )
    
    # Arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Eğitim epoch sayısı (varsayılan: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch boyutu (varsayılan: 16)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Öğrenme oranı (varsayılan: 1e-4)')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (varsayılan: 3)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum örnek sayısı (varsayılan: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Sadece analiz yap, eğitim yapma')
    parser.add_argument('--force', action='store_true',
                       help='Uyarıları görmezden gel ve zorla eğit')
    parser.add_argument('--verbose', action='store_true',
                       help='Detaylı log çıktısı')
    
    args = parser.parse_args()
    
    # Verbose mode
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    try:
        # Flask uygulama bağlamı oluştur
        app, socketio_direct = create_app(return_socketio=True)  # (app, socketio)
        
        with app.app_context():
            logger.info("🚀 CLIP training script başlatılıyor...")
            
            # Servisleri oluştur
            content_service = ContentTrainingService()
            clip_service = ClipTrainingService()
            
            # 1. Analiz aşaması
            print("🔍 VERİ ANALİZİ YAPILIYOR...")
            analysis = content_service.analyze_training_readiness()
            
            print_analysis_results(analysis)
            
            # Dry run kontrolü
            if args.dry_run:
                print("🏁 Dry run tamamlandı. Eğitim yapılmadı.")
                return
            
            # Eğitim önerisi kontrolü
            recommended = analysis.get('training_recommendation', {}).get('recommended', False)
            
            if not recommended and not args.force:
                print("⚠️  Sistem eğitim için önerilmiyor!")
                print("💡 --force flag'i ile zorla eğitebilirsiniz.")
                print("📊 Önce daha fazla feedback toplayın.")
                return
            
            # 2. Training parametreleri hazırla
            training_params = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'patience': args.patience
            }
            
            print_training_params(training_params)
            
            # 3. Training data hazırla
            print("📦 TRAINING VERİSİ HAZIRLANIYOR...")
            training_data = clip_service.prepare_training_data(min_samples=args.min_samples)
            
            if not training_data:
                print("❌ Training verisi hazırlanamadı!")
                print(f"🔢 En az {args.min_samples} geçerli örnek gerekli.")
                return
            
            print(f"✅ Training verisi hazır:")
            print(f"  📊 Toplam: {training_data['total_samples']} örnek")
            print(f"  🎯 Train: {training_data['train_samples']} örnek")
            print(f"  🔍 Val: {training_data['val_samples']} örnek")
            print()
            
            # 4. Training başlat
            print("🚀 MODEL EĞİTİMİ BAŞLATILIYOR...")
            print("⏳ Bu işlem birkaç dakika sürebilir...")
            print()
            
            result = clip_service.train_model(training_data, training_params)
            
            # 5. Sonuçları yazdır
            print_training_results(result)
            
            if result['success']:
                print("🎉 CLIP modeli başarıyla eğitildi ve aktif edildi!")
                print("🔄 Artık içerik analizinde yeni model kullanılacak.")
            else:
                print("💥 Eğitim sırasında hata oluştu.")
                
                # Hata detayları
                error_msg = result.get('error', 'Bilinmeyen hata')
                print(f"🚫 Hata Detayı: {error_msg}")
                
                if 'CUDA' in error_msg:
                    print("💡 GPU bellek sorunu olabilir. Batch size'ı düşürün.")
                elif 'memory' in error_msg.lower():
                    print("💡 Bellek sorunu. Batch size'ı düşürün veya daha az örnek kullanın.")
    
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu.")
        
    except Exception as e:
        logger.error(f"Script hatası: {e}")
        print(f"\n💥 Beklenmeyen hata: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    finally:
        print("\n" + "=" * 70)
        print("🏁 Script tamamlandı.")
        print("📝 Detaylı loglar için .log dosyasını kontrol edin.")
        print("=" * 70)

if __name__ == '__main__':
    main() 