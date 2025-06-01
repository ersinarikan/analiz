#!/usr/bin/env python3

from app import app, db
from app.models.feedback import Feedback

def check_feedbacks():
    with app.app_context():
        # Tüm feedback'leri al
        all_feedbacks = Feedback.query.all()
        print(f"Toplam feedback kayıtları: {len(all_feedbacks)}")
        
        # Yaş feedback'lerini al
        age_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo')
        ).all()
        print(f"Yaş feedback'leri: {len(age_feedbacks)}")
        
        # Manuel yaş feedback'leri
        manual_age_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo'),
            Feedback.feedback_source == 'MANUAL_USER'
        ).all()
        print(f"Manuel yaş feedback'leri: {len(manual_age_feedbacks)}")
        
        # Pseudo yaş feedback'leri
        pseudo_age_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo'),
            Feedback.feedback_source == 'PSEUDO_BUFFALO_HIGH_CONF'
        ).all()
        print(f"Pseudo yaş feedback'leri: {len(pseudo_age_feedbacks)}")
        
        # Embedding'li yaş feedback'leri
        age_with_embedding = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | (Feedback.feedback_type == 'age_pseudo'),
            Feedback.embedding.isnot(None)
        ).all()
        print(f"Embedding'li yaş feedback'leri: {len(age_with_embedding)}")
        
        # Örnek feedback'leri göster
        if age_feedbacks:
            print("\nÖrnek yaş feedback'leri:")
            for feedback in age_feedbacks[:5]:
                print(f"  ID: {feedback.id}")
                print(f"  Tip: {feedback.feedback_type}")
                print(f"  Kaynak: {feedback.feedback_source}")
                print(f"  Person ID: {feedback.person_id}")
                print(f"  Corrected Age: {feedback.corrected_age}")
                print(f"  Pseudo Age: {feedback.pseudo_label_original_age}")
                print(f"  Embedding var mı: {feedback.embedding is not None}")
                print("  ---")
        
        # Tüm feedback tiplerini say
        print("\nTüm feedback tipleri:")
        from sqlalchemy import func
        feedback_types = db.session.query(
            Feedback.feedback_type, 
            func.count(Feedback.id)
        ).group_by(Feedback.feedback_type).all()
        
        for feedback_type, count in feedback_types:
            print(f"  {feedback_type}: {count}")
            
        # Feedback source'ları say
        print("\nFeedback kaynakları:")
        feedback_sources = db.session.query(
            Feedback.feedback_source, 
            func.count(Feedback.id)
        ).group_by(Feedback.feedback_source).all()
        
        for source, count in feedback_sources:
            print(f"  {source}: {count}")

if __name__ == "__main__":
    check_feedbacks() 