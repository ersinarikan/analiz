#!/usr/bin/env python3
"""
Feedback veri kontrolü ve eğitim uygunluk analizi
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.feedback import Feedback
from app.models.analysis import Analysis
from datetime import datetime, timedelta
import json

def check_feedback_data():
    """Geri bildirim verilerini kontrol et"""
    
    app = create_app()
    
    with app.app_context():
        print("🔍 Geri Bildirim Veri Analizi")
        print("=" * 50)
        
        # Tüm feedback'leri al
        all_feedbacks = Feedback.query.all()
        print(f"📊 Toplam feedback sayısı: {len(all_feedbacks)}")
        
        if len(all_feedbacks) == 0:
            print("❌ Hiç geri bildirim verisi bulunamadı!")
            return
        
        # Son 30 gün içindeki feedback'ler
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_feedbacks = [f for f in all_feedbacks if f.created_at and f.created_at >= recent_cutoff]
        print(f"📅 Son 30 gün içindeki feedback: {len(recent_feedbacks)}")
        
        print("\n" + "="*50)
        print("🎯 YAŞ TAHMİNİ (AGE) FEEDBACK ANALİZİ")
        print("="*50)
        
        # Yaş feedback'lerini kontrol et
        age_feedbacks = [f for f in all_feedbacks if f.corrected_age is not None]
        age_manual = [f for f in age_feedbacks if f.feedback_source == 'MANUAL_USER']
        age_pseudo = [f for f in age_feedbacks if f.feedback_source != 'MANUAL_USER']
        
        print(f"📈 Toplam yaş feedback: {len(age_feedbacks)}")
        print(f"✋ Manuel yaş feedback: {len(age_manual)}")
        print(f"🤖 Pseudo yaş feedback: {len(age_pseudo)}")
        
        if len(age_feedbacks) > 0:
            ages = [f.corrected_age for f in age_feedbacks if f.corrected_age]
            avg_age = sum(ages) / len(ages) if ages else 0
            min_age = min(ages) if ages else 0
            max_age = max(ages) if ages else 0
            
            print(f"📊 Yaş dağılımı:")
            print(f"   - Ortalama: {avg_age:.1f} yaş")
            print(f"   - Min: {min_age} yaş")
            print(f"   - Max: {max_age} yaş")
            
            # Yaş grupları
            age_groups = {"0-9": 0, "10-19": 0, "20-29": 0, "30-39": 0, "40-49": 0, "50-59": 0, "60+": 0}
            for age in ages:
                if age < 10:
                    age_groups["0-9"] += 1
                elif age < 20:
                    age_groups["10-19"] += 1
                elif age < 30:
                    age_groups["20-29"] += 1
                elif age < 40:
                    age_groups["30-39"] += 1
                elif age < 50:
                    age_groups["40-49"] += 1
                elif age < 60:
                    age_groups["50-59"] += 1
                else:
                    age_groups["60+"] += 1
            
            print(f"   - Yaş grupları:")
            for group, count in age_groups.items():
                print(f"     {group}: {count} kişi")
        
        # Eğitim uygunluğu
        print(f"\n✅ Yaş modeli eğitim değerlendirmesi:")
        if len(age_feedbacks) >= 10:
            print(f"   ✓ Yeterli veri var ({len(age_feedbacks)} ≥ 10)")
            if len(age_manual) >= 5:
                print(f"   ✓ Manuel feedback yeterli ({len(age_manual)} ≥ 5)")
            else:
                print(f"   ⚠️ Manuel feedback az ({len(age_manual)} < 5)")
                
            age_training_ready = True
        else:
            print(f"   ❌ Yetersiz veri ({len(age_feedbacks)} < 10)")
            age_training_ready = False
        
        print("\n" + "="*50)
        print("🎨 İÇERİK ANALİZİ (CLIP) FEEDBACK ANALİZİ")
        print("="*50)
        
        # İçerik kategorileri feedback'lerini kontrol et (YENİ YAPI)
        content_categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
        content_feedbacks = []
        
        # MANUAL_USER_CONTENT_CORRECTION feedback'lerini al
        content_feedbacks = [f for f in all_feedbacks if f.feedback_source == 'MANUAL_USER_CONTENT_CORRECTION' 
                           and f.category_feedback is not None]
        
        print(f"📈 Toplam içerik feedback: {len(content_feedbacks)}")
        
        # Kategori bazlı analiz (YENİ YAPI)
        category_stats = {}
        for category in content_categories:
            positive_count = 0
            negative_count = 0
            
            for feedback in content_feedbacks:
                if feedback.category_feedback and category in feedback.category_feedback:
                    value = feedback.category_feedback[category]
                    
                    # Boş string veya None değilse sayalım
                    if value and value.strip():
                        if value in ['true_positive', 'flagged', 'positive']:
                            positive_count += 1
                        elif value in ['false_positive', 'safe', 'negative']:
                            negative_count += 1
            
            category_stats[category] = {
                'positive': positive_count,
                'negative': negative_count,
                'total': positive_count + negative_count
            }
            
            print(f"🏷️ {category.title()}:")
            print(f"   - Pozitif: {positive_count}")
            print(f"   - Negatif: {negative_count}")
            print(f"   - Toplam: {positive_count + negative_count}")
        
        # İçerik eğitim uygunluğu
        total_content_feedbacks = sum(stats['total'] for stats in category_stats.values())
        min_per_category = min(stats['total'] for stats in category_stats.values()) if category_stats else 0
        
        print(f"\n✅ İçerik modeli eğitim değerlendirmesi:")
        if total_content_feedbacks >= 50:
            print(f"   ✓ Toplam feedback yeterli ({total_content_feedbacks} ≥ 50)")
            if min_per_category >= 5:
                print(f"   ✓ Her kategori için yeterli veri (min: {min_per_category} ≥ 5)")
                content_training_ready = True
            else:
                print(f"   ⚠️ Bazı kategorilerde yetersiz veri (min: {min_per_category} < 5)")
                content_training_ready = False
        else:
            print(f"   ❌ Toplam feedback yetersiz ({total_content_feedbacks} < 50)")
            content_training_ready = False
        
        print("\n" + "="*60)
        print("🎯 EĞİTİM ÖNERİLERİ")
        print("="*60)
        
        if age_training_ready:
            print("✅ YAŞ MODELİ: Eğitime hazır!")
        else:
            print("❌ YAŞ MODELİ: Daha fazla feedback gerekli")
            print(f"   - En az {10 - len(age_feedbacks)} yaş feedback'i daha ekleyin")
        
        if content_training_ready:
            print("✅ İÇERİK MODELİ: Eğitime hazır!")
        else:
            print("❌ İÇERİK MODELİ: Daha fazla feedback gerekli")
            print(f"   - En az {50 - total_content_feedbacks} içerik feedback'i daha ekleyin")
            for category, stats in category_stats.items():
                if stats['total'] < 5:
                    print(f"   - {category.title()}: {5 - stats['total']} feedback daha")
        
        # Son feedback örnekleri
        print(f"\n📋 Son 5 feedback örneği:")
        recent_5 = sorted(all_feedbacks, key=lambda x: x.created_at or datetime.min, reverse=True)[:5]
        
        for i, feedback in enumerate(recent_5, 1):
            feedback_type = "Yaş" if feedback.corrected_age else "İçerik"
            person_id = getattr(feedback, 'person_id', 'N/A')
            created = feedback.created_at.strftime('%Y-%m-%d %H:%M') if feedback.created_at else 'N/A'
            print(f"   {i}. {feedback_type} | Person: {person_id} | {created}")

if __name__ == "__main__":
    check_feedback_data() 