#!/usr/bin/env python3
"""
Feedback Data Durumu Kontrolü
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.age_training_service import AgeTrainingService
import numpy as np

def main():
    print("📊 GERİ BİLDİRİM VERİSİ DURUMU")
    print("=" * 40)
    
    # Flask app context
    app = create_app()
    with app.app_context():
        
        service = AgeTrainingService()
        
        # Geri bildirimleri yükle
        feedback_data = service.prepare_training_data(min_samples=1)
        
        if feedback_data is None or len(feedback_data['embeddings']) == 0:
            print("❌ Hiç geri bildirim verisi bulunamadı")
            return
        
        total_count = len(feedback_data['embeddings'])
        manual_count = feedback_data['sources'].count('manual')
        pseudo_count = feedback_data['sources'].count('pseudo')
        ages = feedback_data['ages']
        
        print(f"📊 Toplam feedback: {total_count}")
        print(f"📊 Manual feedback: {manual_count}")
        print(f"📊 Pseudo feedback: {pseudo_count}")
        print(f"📊 Yaş aralığı: {np.min(ages):.1f} - {np.max(ages):.1f}")
        print(f"📊 Ortalama yaş: {np.mean(ages):.1f}")
        
        # Yaş dağılımı
        age_distribution = {}
        for age in ages:
            decade = (int(age) // 10) * 10
            age_distribution[decade] = age_distribution.get(decade, 0) + 1
        
        print("\n📊 Yaş dağılımı (10'lu gruplar):")
        for decade in sorted(age_distribution.keys()):
            print(f"   {decade}-{decade+9} yaş: {age_distribution[decade]} örnek")
        
        # Güven skorları
        confidence_scores = feedback_data['confidence_scores']
        print(f"\n📊 Güven skorları:")
        print(f"   Ortalama: {np.mean(confidence_scores):.3f}")
        print(f"   Minimum: {np.min(confidence_scores):.3f}")
        print(f"   Maksimum: {np.max(confidence_scores):.3f}")
        
        print(f"\n✅ Feedback verisi kullanıma hazır!")
        print(f"💡 UTKFace ({1979} örnek) + Feedback ({total_count} örnek) = {1979 + total_count} toplam örnek")

if __name__ == "__main__":
    main() 