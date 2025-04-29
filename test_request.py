import requests
import json

def test_analyze():
    # Test görüntüsünün yolu
    image_path = 'test_images/test_image.jpg'
    
    # API endpoint'i
    url = 'http://localhost:5000/api/test/analyze'
    
    # Dosyayı hazırla
    files = {
        'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        # POST isteği yap
        response = requests.post(url, files=files)
        
        # Yanıtı kontrol et
        if response.status_code == 200:
            # JSON yanıtı formatla ve yazdır
            result = json.dumps(response.json(), indent=2, ensure_ascii=False)
            print("Analiz sonuçları:")
            print(result)
        else:
            print(f"Hata: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"İstek hatası: {str(e)}")

if __name__ == "__main__":
    test_analyze() 