import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class Scene:
    """Video sahnesi için veri sınıfı"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    keyframe: np.ndarray  # Sahneyi temsil eden kare

class SceneDetector:
    """Video sahne tespiti için sınıf"""
    
    def __init__(self, threshold: float = 30.0, min_scene_length: int = 15):
        """
        Args:
            threshold: Sahne değişimi için hassasiyet eşiği (0-100)
            min_scene_length: Minimum sahne uzunluğu (kare sayısı)
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        logger.info(f"SceneDetector başlatıldı: threshold={threshold}, min_scene_length={min_scene_length}")

    def detect_scenes(self, video_path: str) -> List[Scene]:
        """Video'daki sahneleri tespit eder"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Video açılamadı: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            scenes = []
            current_scene_start = 0
            prev_frame = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Kareler arası farkı hesapla
                    diff = self._calculate_frame_difference(prev_frame, frame)

                    # Sahne değişimi kontrolü
                    if diff > self.threshold and frame_count - current_scene_start >= self.min_scene_length:
                        # Yeni sahne oluştur
                        scene = Scene(
                            start_frame=current_scene_start,
                            end_frame=frame_count,
                            start_time=current_scene_start / fps,
                            end_time=frame_count / fps,
                            keyframe=prev_frame.copy()
                        )
                        scenes.append(scene)
                        current_scene_start = frame_count
                        logger.debug(f"Sahne değişimi tespit edildi: frame={frame_count}, diff={diff:.2f}")

                prev_frame = frame.copy()
                frame_count += 1

            # Son sahneyi ekle
            if frame_count - current_scene_start >= self.min_scene_length:
                scene = Scene(
                    start_frame=current_scene_start,
                    end_frame=frame_count,
                    start_time=current_scene_start / fps,
                    end_time=frame_count / fps,
                    keyframe=prev_frame.copy() if prev_frame is not None else None
                )
                scenes.append(scene)

            cap.release()
            logger.info(f"Video analizi tamamlandı: {len(scenes)} sahne tespit edildi")
            return scenes

        except Exception as e:
            logger.error(f"Sahne tespiti hatası: {str(e)}")
            raise

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """İki kare arasındaki farkı hesaplar"""
        try:
            # Gri tonlamaya dönüştür
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Histogram farkı
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

            # Yapısal benzerlik (SSIM)
            score, _ = self._calculate_ssim(gray1, gray2)
            
            # Fark skorunu normalize et (0-100 arası)
            normalized_diff = (hist_diff * (1 - score)) * 100
            return normalized_diff

        except Exception as e:
            logger.error(f"Kare farkı hesaplama hatası: {str(e)}")
            return 0.0

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Yapısal benzerlik indeksini hesaplar"""
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map), ssim_map 