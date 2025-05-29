#!/usr/bin/env python3

import os
import sys
import numpy as np
from datetime import datetime

# Flask app'i import etmek için gerekli
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_feedbacks():
    """Test için örnek feedback verileri oluşturur"""
    
    # Flask app context'ini başlat
    from app import create_app, db
    from app.models.feedback import Feedback
    
    app = create_app()
    with app.app_context():
        
        print("Test feedback verileri oluşturuluyor...")
        
        # Önce mevcut yaş feedback'lerini kontrol et
        existing_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo')
        ).count()
        
        print(f"Mevcut yaş feedback sayısı: {existing_feedbacks}")
        
        # Örnek embedding (512 boyutlu)
        def generate_random_embedding():
            embedding = np.random.randn(512)
            return ",".join(str(float(x)) for x in embedding)
        
        # Manuel feedback örnekleri oluştur
        manual_feedbacks = []
        for i in range(5):
            feedback = Feedback(
                feedback_type='age',
                feedback_source='MANUAL_USER',
                corrected_age=20 + (i * 10),  # 20, 30, 40, 50, 60 yaş
                embedding=generate_random_embedding(),
                frame_path=f'test/manual_frame_{i+1}.jpg',
                person_id=f'test_manual_person_{i+1}',
                created_at=datetime.now()
            )
            manual_feedbacks.append(feedback)
        
        # Pseudo feedback örnekleri oluştur  
        pseudo_feedbacks = []
        for i in range(7):
            feedback = Feedback(
                feedback_type='age_pseudo',
                feedback_source='PSEUDO_BUFFALO_HIGH_CONF',
                pseudo_label_original_age=25 + (i * 5),  # 25, 30, 35, 40, 45, 50, 55 yaş
                pseudo_label_clip_confidence=0.9 + (i * 0.01),  # Yüksek güven
                embedding=generate_random_embedding(),
                frame_path=f'test/pseudo_frame_{i+1}.jpg',
                person_id=f'test_pseudo_person_{i+1}',
                created_at=datetime.now()
            )
            pseudo_feedbacks.append(feedback)
        
        # Veritabanına ekle
        try:
            for feedback in manual_feedbacks + pseudo_feedbacks:
                db.session.add(feedback)
            
            db.session.commit()
            
            print(f"✅ {len(manual_feedbacks)} manuel feedback oluşturuldu")
            print(f"✅ {len(pseudo_feedbacks)} pseudo feedback oluşturuldu")
            print(f"📊 Toplam yaş feedback: {len(manual_feedbacks) + len(pseudo_feedbacks) + existing_feedbacks}")
            
            # Doğrulama
            new_total = Feedback.query.filter(
                (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo')
            ).count()
            
            print(f"🔍 Doğrulama: Veritabanında toplam {new_total} yaş feedback bulundu")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    create_sample_feedbacks() 