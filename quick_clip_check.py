#!/usr/bin/env python3
"""
CLIP Feedback Kontrol Scripti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.feedback import Feedback

def main():
    app = create_app()
    
    with app.app_context():
        print("🔍 CLIP Feedback Kontrolü")
        print("=" * 40)
        
        # 1. MANUAL_USER_CONTENT_CORRECTION source
        content_feedbacks = Feedback.query.filter(
            Feedback.feedback_source == 'MANUAL_USER_CONTENT_CORRECTION'
        ).all()
        print(f"MANUAL_USER_CONTENT_CORRECTION: {len(content_feedbacks)} feedback")
        
        # 2. category_feedback field'ı olan
        category_feedbacks = Feedback.query.filter(
            Feedback.category_feedback.isnot(None)
        ).all()
        print(f"category_feedback field: {len(category_feedbacks)} feedback")
        
        # 3. Ayrı field'ları olan (eski yöntem)
        violence_feedbacks = Feedback.query.filter(
            Feedback.violence_feedback.isnot(None)
        ).all() if hasattr(Feedback, 'violence_feedback') else []
        print(f"violence_feedback field: {len(violence_feedbacks)} feedback")
        
        # 4. Tüm feedback türleri
        all_feedbacks = Feedback.query.all()
        feedback_types = {}
        feedback_sources = {}
        
        for f in all_feedbacks:
            # Türler
            if f.feedback_type:
                feedback_types[f.feedback_type] = feedback_types.get(f.feedback_type, 0) + 1
            
            # Kaynaklar
            if f.feedback_source:
                feedback_sources[f.feedback_source] = feedback_sources.get(f.feedback_source, 0) + 1
        
        print(f"\nToplam feedback: {len(all_feedbacks)}")
        print(f"Feedback türleri: {feedback_types}")
        print(f"Feedback kaynakları: {feedback_sources}")
        
        # 5. category_feedback örneği
        if category_feedbacks:
            sample = category_feedbacks[0]
            print(f"\nÖrnek category_feedback:")
            print(f"ID: {sample.id}")
            print(f"category_feedback: {sample.category_feedback}")
            print(f"content_id: {sample.content_id}")
            print(f"feedback_source: {sample.feedback_source}")

if __name__ == "__main__":
    main() 