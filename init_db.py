from app import db, create_app

def init_database():
    """Veritabanını sıfırdan oluşturur"""
    print("Veritabanı tabloları oluşturuluyor...")
    app = create_app()
    with app.app_context():
        # Tüm tabloları sil ve yeniden oluştur
        db.drop_all()
        db.create_all()
        print("Veritabanı başarıyla oluşturuldu!")

if __name__ == "__main__":
    init_database() 