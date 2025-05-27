#!/usr/bin/env python3
"""
OpenCLIP Content Model Training Script

Bu script, kullanıcı geri bildirimlerini kullanarak OpenCLIP modelini
içerik analizi için fine-tune eder.

Kullanım:
    python train_content_model.py --epochs 20 --batch-size 32 --learning-rate 0.001
    python train_content_model.py --dry-run  # Sadece veri istatistiklerini göster
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Proje kök dizinini Python yoluna ekle
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import create_app
from app.services.content_training_service import ContentTrainingService

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'content_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='OpenCLIP Content Model Training')
    
    # Eğitim parametreleri
    parser.add_argument('--epochs', type=int, default=10,
                      help='Eğitim epoch sayısı (varsayılan: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch boyutu (varsayılan: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Öğrenme oranı (varsayılan: 0.001)')
    parser.add_argument('--min-samples', type=int, default=50,
                      help='Minimum eğitim örnek sayısı (varsayılan: 50)')
    
    # Kontrol parametreleri
    parser.add_argument('--dry-run', action='store_true',
                      help='Sadece veri istatistiklerini göster, eğitim yapma')
    parser.add_argument('--force', action='store_true',
                      help='Yetersiz veri uyarılarını görmezden gel')
    
    args = parser.parse_args()
    
    logger.info("OpenCLIP Content Model Training başlatılıyor...")
    logger.info(f"Parametreler: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"learning_rate={args.learning_rate}, min_samples={args.min_samples}")
    
    try:
        # Flask app bağlamı oluştur
        app = create_app()
        
        with app.app_context():
            # Training service başlat
            trainer = ContentTrainingService()
            
            # Veriyi hazırla
            logger.info("Eğitim verisi hazırlanıyor...")
            training_data = trainer.prepare_training_data(min_samples=args.min_samples)
            
            if training_data is None:
                logger.error(f"Yeterli eğitim verisi bulunamadı. En az {args.min_samples} örnek gerekli.")
                if not args.force:
                    sys.exit(1)
                else:
                    logger.warning("--force kullanıldı, eğitime devam ediliyor...")
            
            # Veri istatistikleri
            total_samples = training_data['total_samples']
            category_counts = training_data['category_counts']
            
            logger.info(f"Toplam eğitim örneği: {total_samples}")
            logger.info("Kategori dağılımı:")
            for category, count in category_counts.items():
                logger.info(f"  {category}: {count} örnek")
            
            if args.dry_run:
                logger.info("Dry-run modu: Sadece veri istatistikleri gösterildi")
                return
            
            # Eğitim parametrelerini hazırla
            training_params = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'min_samples': args.min_samples
            }
            
            # Onay al
            print(f"\n{'='*60}")
            print("EĞİTİM ÖNCESİ ONAY")
            print(f"{'='*60}")
            print(f"Toplam eğitim örneği: {total_samples}")
            print(f"Epoch sayısı: {args.epochs}")
            print(f"Batch boyutu: {args.batch_size}")
            print(f"Öğrenme oranı: {args.learning_rate}")
            print(f"{'='*60}")
            
            if not args.force:
                response = input("Eğitime başlamak istiyor musunuz? (E/h): ")
                if response.lower() not in ['e', 'evet', 'yes', 'y']:
                    logger.info("Eğitim iptal edildi.")
                    return
            
            # Model eğitimi başlat
            logger.info("Model eğitimi başlatılıyor...")
            start_time = datetime.now()
            
            try:
                training_result = trainer.train_model(training_data, training_params)
                
                # Model versiyonunu kaydet
                model_version = trainer.save_model_version(
                    training_result['model'], 
                    training_result
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Sonuçları göster
                print(f"\n{'='*60}")
                print("EĞİTİM TAMAMLANDI!")
                print(f"{'='*60}")
                print(f"Model versiyonu: {model_version.version_name}")
                print(f"Eğitim süresi: {duration:.2f} saniye")
                print(f"Eğitim örnekleri: {training_result['training_samples']}")
                print(f"Doğrulama örnekleri: {training_result['validation_samples']}")
                
                print(f"\nPerformans Metrikleri:")
                metrics = training_result['metrics']
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  En iyi validation loss: {metrics['best_val_loss']:.4f}")
                
                print(f"\nModel dosyaları:")
                print(f"  Versiyon dizini: {model_version.file_path}")
                print(f"  Classification head: {model_version.weights_path}")
                
                print(f"\n{'='*60}")
                print("Model başarıyla eğitildi ve kaydedildi!")
                print("Model yönetimi panelinden bu versiyonu aktifleştirebilirsiniz.")
                print(f"{'='*60}")
                
                logger.info(f"Eğitim başarıyla tamamlandı. Versiyon: {model_version.version_name}")
                
            except Exception as training_error:
                logger.error(f"Eğitim sırasında hata: {str(training_error)}")
                raise
                
    except KeyboardInterrupt:
        logger.info("Eğitim kullanıcı tarafından iptal edildi.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 