from app import create_app, db
from app.models.feedback import Feedback

app = create_app()
with app.app_context():
    count = Feedback.query.filter_by(feedback_type='content').count()
    print(f"İçerik analizi için elle girilmiş feedback sayısı: {count}")
    if count > 0:
        feedbacks = Feedback.query.filter_by(feedback_type='content').order_by(Feedback.created_at.desc()).limit(5).all()
        for f in feedbacks:
            print(f"- Tarih: {f.created_at}, Yorum: {f.comment}, Kategoriler: {f.category_feedback}")