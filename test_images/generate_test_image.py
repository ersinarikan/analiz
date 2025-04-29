import cv2
import numpy as np

def generate_test_face():
    """Test için basit bir yüz görüntüsü oluştur"""
    # 224x224 boyutunda boş bir görüntü oluştur
    image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Yüz şekli çiz (basit bir oval)
    center = (112, 112)
    axes = (60, 80)
    cv2.ellipse(image, center, axes, 0, 0, 360, (200, 200, 200), -1)
    
    # Gözleri çiz
    left_eye = (92, 92)
    right_eye = (132, 92)
    cv2.circle(image, left_eye, 8, (100, 100, 100), -1)
    cv2.circle(image, right_eye, 8, (100, 100, 100), -1)
    
    # Ağzı çiz
    mouth_start = (92, 142)
    mouth_end = (132, 142)
    cv2.line(image, mouth_start, mouth_end, (100, 100, 100), 2)
    
    # Görüntüyü kaydet
    cv2.imwrite('face.jpg', image)
    print("Test görüntüsü oluşturuldu: face.jpg")

if __name__ == '__main__':
    generate_test_face() 