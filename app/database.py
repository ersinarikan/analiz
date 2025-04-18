from app import db

def init_db():
    """Veritabanını başlatır ve tabloları oluşturur."""
    db.create_all()

def reset_db():
    """Veritabanını sıfırlar."""
    db.drop_all()
    db.create_all()

def get_engine():
    """SQLAlchemy engine'i döndürür."""
    return db.engine 