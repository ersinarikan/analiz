import os
import time
from datetime import datetime
import threading
import logging
import numpy as np
import traceback
from flask import current_app
from app import db
# Import the real analyzers
from app.ai.content_analyzer import ContentAnalyzer
from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
from app.models.analysis import Analysis, ContentDetection, AgeEstimation
from app.models.file import File
from app.utils.video_utils import extract_frames
from app.utils.image_utils import load_image
from config import Config
import json
import uuid
from app.models.content import Content, ContentType, AnalysisResult, ContentCategory
from app.services.file_service import get_file_info, create_thumbnail
from app.services.db_service import save_to_db, query_db
from app.services.model_service import load_model, run_image_analysis, run_video_analysis
from app.utils.content_utils import detect_content_type
from app.json_encoder import json_dumps_numpy, NumPyJSONEncoder
from deep_sort_realtime.deepsort_tracker import DeepSort
import clip
import torch
from PIL import Image
import cv2
from sqlalchemy import desc

logger = logging.getLogger(__name__)

clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)

# NumPy değerlerini Python standart türlerine dönüştürmek için yardımcı fonksiyon
def ensure_serializable(obj):
    """NumPy türleri dahil tüm verileri JSON serileştirilebilir hale getirir."""
    if obj is None:
        return None
    elif isinstance(obj, (str, bool, int, float)):
        return obj  # zaten Python tipi
    elif isinstance(obj, np.ndarray):
        return ensure_serializable(obj.tolist())
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        # Genel NumPy skaler tipi için item() metodunu kullan
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}
    try:
        # Son çare: metin temsiline dönüştür
        return str(obj)
    except:
        logger.error(f"Serileştirilemeyen veri tipi: {type(obj)}")
        return None

# Mock analizör sınıflarını kaldırıyoruz, gerçek analizörleri kullanacağız

class AnalysisService:
    """
    İçerik analiz servis sınıfı, yüklenen dosyaların analiz işlemlerini yönetir.
    Bu sınıf, farklı kategorilerde (şiddet, taciz, yetişkin içeriği, vb.) 
    içerik analizi gerçekleştirmek için gerekli tüm metotları içerir.
    """
    
    def __init__(self, app=None):
        """
        Servis sınıfını başlatır ve gerekli modelleri yükler.
        
        Args:
            app: Flask uygulama nesnesi (opsiyonel)
        """
        self.models = {}
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """
        Flask uygulamasına servis sınıfını entegre eder ve gerekli modelleri yükler.
        Flask contextinde çalışarak modellerin doğru şekilde yüklenmesini sağlar.
        
        Args:
            app: Flask uygulama nesnesi
        """
        # Analiz modellerini yükle
        with app.app_context():
            self._load_models()
    
    def _load_models(self):
        """
        Gerekli analiz modellerini yükler. Her kategori için ayrı model kullanılır.
        Şiddet, taciz, yetişkin içeriği, silah ve madde kullanımı için
        ayrı modeller yüklenir ve hatalar loglama sistemi ile kaydedilir.
        """
        try:
            # Şiddet içeriği için model
            self.models['violence'] = load_model('violence_detection')
            
            # Taciz, hakaret içeriği için model
            self.models['harassment'] = load_model('harassment_detection')
            
            # Yetişkin içeriği için model
            self.models['adult'] = load_model('adult_content_detection')
            
            # Silah kullanımı için model
            self.models['weapon'] = load_model('weapon_detection')
            
            # Madde kullanımı için model
            self.models['substance'] = load_model('substance_detection')
            
            current_app.logger.info("Analiz modelleri başarıyla yüklendi")
        except Exception as e:
            current_app.logger.error(f"Model yükleme hatası: {str(e)}")
    
    def start_analysis(self, file_id, frames_per_second=None, include_age_analysis=False):
        """
        Verilen dosya ID'si için analiz işlemini başlatır.
        
        Args:
            file_id: Analiz edilecek dosyanın veritabanı ID'si
            frames_per_second: Video analizi için saniyede işlenecek kare sayısı
            include_age_analysis: Yaş analizi yapılsın mı?
            
        Returns:
            Analysis: Oluşturulan analiz nesnesi veya None
        """
        try:
            # Dosyayı veritabanından al
            file = File.query.get(file_id)
            if not file:
                logger.error(f"Dosya bulunamadı: {file_id}")
                return None
                
            # Yeni bir analiz oluştur
            analysis = Analysis(
                file_id=file_id,
                frames_per_second=frames_per_second,
                include_age_analysis=include_age_analysis
            )
            
            # Başlangıç durumunu ayarla
            analysis.status = 'pending'
            analysis.status_message = 'Analiz başlatılıyor, dosya hazırlanıyor...'
            analysis.progress = 5
            
            db.session.add(analysis)
            db.session.commit()
            
            logger.info(f"Analiz oluşturuldu: #{analysis.id} - Dosya: {file.original_filename}, Durum: pending")
            
            # Socket.io üzerinden bildirim gönder
            try:
                from app import socketio
                socketio.emit('analysis_started', {
                    'analysis_id': analysis.id,
                    'file_id': file_id,
                    'file_name': file.original_filename,
                    'file_type': file.file_type,
                    'status': 'pending'
                })
            except Exception as socket_err:
                logger.warning(f"Socket.io analiz başlangıç bildirimi hatası: {str(socket_err)}")
            
            # Analizi kuyruğa ekle (thread yerine kuyruk kullan)
            from app.services.queue_service import add_to_queue
            add_to_queue(analysis.id)
            
            logger.info(f"Analiz kuyruğa eklendi: #{analysis.id}")
            
            return analysis
                
        except Exception as e:
            logger.error(f"Analiz başlatma hatası: {str(e)}")
            db.session.rollback()
            return None
    
    def _process_analysis_legacy(self, analysis_id):
        """
        [LEGACY/UNUSED] Bu metod artık kullanılmıyor, queue_service tarafından yönetiliyor.
        Eski analiz işlemini gerçekleştirirdi (Celery task yerine doğrudan çalışır)
        
        Args:
            analysis_id: Analiz edilecek analizin ID'si
        """
        import traceback
        import time
        from flask import current_app
        
        start_time = time.time()
        logger.info(f"[LEGACY] Analiz işlemi başlatılıyor: #{analysis_id}")
        
        # Flask uygulama bağlamını al
        from app import create_app
        app = create_app()
        
        # Thread içinde app_context kullanarak veritabanı işlemlerini yap
        with app.app_context():
            try:
                from app import db
                from app.models.analysis import Analysis
                
                # Analiz nesnesini al
                analysis = Analysis.query.get(analysis_id)
                if not analysis:
                    logger.error(f"Thread içinde analiz bulunamadı: {analysis_id}")
                    return
                
                # Dosya bilgilerini logla
                try:
                    file = analysis.file
                    if not file:
                        # File ilişkisi bulunamadı, manuel olarak dosya bilgisini al
                        file = File.query.get(analysis.file_id)
                        if not file:
                            logger.error(f"Dosya bulunamadı: {analysis.file_id} (Analiz #{analysis_id})")
                            raise ValueError(f"Analiz için dosya bulunamadı: #{analysis_id}")
                    
                    logger.info(f"Analiz #{analysis_id} başlıyor - Dosya: {file.original_filename}, Tip: {file.file_type}, "
                               f"Boyut: {file.file_size/1024/1024:.2f} MB")
                except Exception as file_error:
                    logger.error(f"Dosya bilgisi hatası: {str(file_error)}")
                    # WebSocket üzerinden hata bildirimi gönder
                    try:
                        from app import socketio
                        socketio.emit('analysis_failed', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id,
                            'error': f"Dosya bilgisi alınamadı: {str(file_error)}"
                        })
                    except Exception:
                        pass
                    
                    # Analizi başarısız olarak işaretle
                    try:
                        analysis.status = 'failed'
                        analysis.status_message = f"Dosya bilgisi alınamadı: {str(file_error)}"
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                    
                    return
                
                # Analizi başlat
                try:
                    analysis.status = 'processing'
                    analysis.progress = 10
                    analysis.status_message = 'Analiz başlatıldı, dosya hazırlanıyor...'
                    db.session.commit()
                    logger.info(f"Analiz #{analysis_id} işleniyor - Durum: processing, İlerleme: %10")
                    
                    # Frontend'e analiz başlangıç bilgisini socket ile gönder
                    try:
                        from app import socketio
                        socketio.emit('analysis_started', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id,
                            'file_name': file.original_filename,
                            'file_type': file.file_type
                        })
                    except Exception as socket_err:
                        logger.warning(f"Socket.io analiz başlangıç bildirimi hatası: {str(socket_err)}")
                    
                except Exception as commit_error:
                    logger.error(f"Analiz başlatma hatası: {str(commit_error)}")
                    db.session.rollback()
                    return
                
                # Analizi gerçekleştir
                try:
                    success, message = analyze_file(analysis_id)
                    
                    # Analiz tamamlandı, geçen süreyi hesapla
                    elapsed_time = time.time() - start_time
                    logger.info(f"Analiz #{analysis_id} tamamlandı - Sonuç: {'Başarılı' if success else 'Başarısız'}, "
                               f"Süre: {elapsed_time:.2f} saniye, Mesaj: {message}")
                    
                    # WebSocket üzerinden bildirim gönder (tamamlandı/başarısız)
                    from app import socketio
                    if success:
                        socketio.emit('analysis_completed', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id,
                            'elapsed_time': elapsed_time,
                            'message': message
                        })
                    else:
                        socketio.emit('analysis_failed', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id,
                            'elapsed_time': elapsed_time,
                            'error': message
                        })
                except Exception as analysis_error:
                    elapsed_time = time.time() - start_time
                    error_message = f"Analiz işleme hatası: {str(analysis_error)}"
                    logger.error(f"{error_message}\n{traceback.format_exc()}")
                    logger.error(f"Analiz #{analysis_id} başarısız oldu - Süre: {elapsed_time:.2f} saniye")
                    
                    # Analizi başarısız olarak işaretle
                    try:
                        analysis = Analysis.query.get(analysis_id)
                        if analysis:
                            analysis.status = 'failed'
                            analysis.status_message = error_message[:250]  # 250 karakter sınırı
                            db.session.commit()
                            logger.info(f"Analiz #{analysis_id} 'failed' olarak işaretlendi")
                    except Exception as update_error:
                        logger.error(f"Başarısız analizi işaretlerken hata: {str(update_error)}")
                        db.session.rollback()
                    
                    # WebSocket üzerinden hata bildirimi gönder
                    try:
                        from app import socketio
                        socketio.emit('analysis_failed', {
                            'analysis_id': analysis_id,
                            'file_id': analysis.file_id if analysis else None,
                            'elapsed_time': elapsed_time,
                            'error': error_message
                        })
                    except Exception as ws_error:
                        logger.error(f"WebSocket hata bildirimi hatası: {str(ws_error)}")
                        
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.critical(f"_process_analysis kritik hatası: {str(e)}\n{traceback.format_exc()}")
                logger.critical(f"Analiz #{analysis_id} kritik hata - Süre: {elapsed_time:.2f} saniye")
    
    def cancel_analysis(self, analysis_id):
        """
        Devam eden bir analizi iptal eder.
        
        Args:
            analysis_id: İptal edilecek analizin ID'si
            
        Returns:
            bool: İptal başarılı mı?
        """
        try:
            analysis = Analysis.query.get(analysis_id)
            if not analysis:
                return False
                
            # Analiz durumunu iptal edildi olarak işaretle
            analysis.status = 'cancelled'
            analysis.status_message = 'Analiz kullanıcı tarafından iptal edildi'
            analysis.updated_at = datetime.now()
            db.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Analiz iptal hatası: {str(e)}")
            return False
    
    def retry_analysis(self, analysis_id):
        """
        Başarısız bir analizi tekrar dener.
        
        Args:
            analysis_id: Tekrar denenecek analizin ID'si
            
        Returns:
            Analysis: Yeni analiz nesnesi veya None
        """
        try:
            # Önceki analizi al
            prev_analysis = Analysis.query.get(analysis_id)
            if not prev_analysis:
                return None
                
            # Aynı parametrelerle yeni analiz oluştur
            new_analysis = Analysis(
                file_id=prev_analysis.file_id,
                frames_per_second=prev_analysis.frames_per_second,
                include_age_analysis=prev_analysis.include_age_analysis
            )
            
            db.session.add(new_analysis)
            db.session.commit()
            
            # Analizi kuyruğa ekle (thread yerine)
            from app.services.queue_service import add_to_queue
            add_to_queue(new_analysis.id)
            
            logger.info(f"Tekrar analiz kuyruğa eklendi: #{new_analysis.id}")
            
            return new_analysis
            
        except Exception as e:
            logger.error(f"Analiz tekrar deneme hatası: {str(e)}")
            db.session.rollback()
            return None
    
    def start_analysis_original(self, file_path, original_filename, user_id=None):
        """
        Verilen dosya için analiz işlemini başlatır ve sonuçları veritabanına kaydeder.
        Bu metot analiz sürecinin giriş noktasıdır ve tüm analiz işlemini koordine eder.
        
        Args:
            file_path: Analiz edilecek dosyanın tam yolu
            original_filename: Orijinal dosya adı
            user_id: İçeriği yükleyen kullanıcı ID'si (opsiyonel)
            
        Returns:
            dict: Analiz sonuçlarını içeren veri yapısı
            
        Raises:
            Exception: Dosya analizi sırasında oluşan hatalar
        """
        try:
            # Dosya bilgilerini al
            file_info = get_file_info(file_path)
            if not file_info:
                raise Exception("Dosya bilgileri alınamadı")
            
            # Dosya türünü belirle
            content_type = detect_content_type(file_info['mime_type'])
            
            # Analiz işlemini gerçekleştir
            analysis_results = self._analyze_file(file_path, content_type, file_info['mime_type'])
            
            # Küçük resim oluştur
            thumbnail_data = create_thumbnail(file_path, file_info['mime_type'])
            
            # Benzersiz içerik ID'si oluştur
            content_id = str(uuid.uuid4())
            
            # Sonuçları modele dönüştür
            content = Content(
                id=content_id,
                filename=original_filename,
                file_path=file_path,
                file_size=file_info['size'],
                mime_type=file_info['mime_type'],
                content_type=content_type,
                user_id=user_id,
                upload_date=datetime.now(),
                thumbnail=thumbnail_data
            )
            
            # Analiz sonuçlarını ekle
            for category, result in analysis_results.items():
                content.add_analysis_result(
                    category=category,
                    score=result['score'],
                    details=result.get('details', None)
                )
            
            # Veritabanına kaydet
            save_to_db(content)
            
            return {
                'content_id': content_id,
                'analysis_results': analysis_results,
                'content_type': content_type.value
            }
            
        except Exception as e:
            current_app.logger.error(f"İçerik analiz hatası: {str(e)}")
            raise
    
    def _analyze_file(self, file_path, content_type, mime_type):
        """
        Dosya türüne göre uygun analiz yöntemini çağırır.
        Resim ve video dosyaları için farklı analiz yöntemleri kullanılır.
        
        Args:
            file_path: Analiz edilecek dosyanın tam yolu
            content_type: İçerik türü (IMAGE, VIDEO, vb.)
            mime_type: Dosyanın MIME tipi
            
        Returns:
            dict: Kategorilere göre analiz sonuçları
        """
        if content_type == ContentType.IMAGE:
            return self._analyze_content(file_path, run_image_analysis)
        elif content_type == ContentType.VIDEO:
            return self._analyze_content(file_path, run_video_analysis)
        else:
            raise ValueError(f"Desteklenmeyen içerik türü: {content_type}")
    
    def _analyze_content(self, file_path, analysis_function):
        """
        Belirtilen analiz fonksiyonunu kullanarak içeriği tüm kategoriler için analiz eder.
        Her analiz kategorisi için ayrı model çalıştırılır ve sonuçlar birleştirilir.
        
        Args:
            file_path: Analiz edilecek dosyanın tam yolu
            analysis_function: Kullanılacak analiz fonksiyonu
            
        Returns:
            dict: Kategorilere göre analiz sonuçları
        """
        results = {}
        
        # Her kategori için analiz gerçekleştir
        for category, model in self.models.items():
            try:
                category_result = analysis_function(model, file_path)
                results[category] = category_result
            except Exception as e:
                current_app.logger.error(f"{category} analizi sırasında hata: {str(e)}")
                # Hata durumunda varsayılan sonuçlar
                results[category] = {
                    'score': 0.0,
                    'details': {"error": str(e)}
                }
        
        return results
    
    def get_analysis_result(self, content_id):
        """
        Belirli bir içerik ID'si için analiz sonuçlarını veritabanından getirir.
        Bu metot, önceden yapılmış analizlerin sonuçlarını görüntülemek için kullanılır.
        
        Args:
            content_id: İçerik benzersiz tanımlayıcısı
            
        Returns:
            dict: İçerik ve analiz sonuçlarını içeren veri yapısı veya None
        """
        try:
            # Veritabanından içeriği sorgula
            content = query_db(Content, id=content_id)
            
            if not content:
                return None
            
            # Yanıt formatını oluştur
            response = {
                'content_id': content.id,
                'filename': content.filename,
                'file_size': content.file_size,
                'mime_type': content.mime_type,
                'content_type': content.content_type.value,
                'upload_date': content.upload_date.isoformat(),
                'analysis_results': {}
            }
            
            # Analiz sonuçlarını ekle
            for result in content.analysis_results:
                response['analysis_results'][result.category] = {
                    'score': result.score,
                    'details': result.details
                }
            
            return response
            
        except Exception as e:
            current_app.logger.error(f"Analiz sonucu getirme hatası: {str(e)}")
            return None

def analyze_file(analysis_id):
    """
    Dosya analizi gerçekleştirir.
    Bu fonksiyon analiz işleminin başlangıç noktasıdır ve verilen ID'ye göre analizi başlatır.
    
    Args:
        analysis_id: Analiz edilecek dosyanın ID'si
        
    Returns:
        tuple: (başarı durumu, mesaj)
    """
    analysis = Analysis.query.get(analysis_id)
    
    if not analysis:
        current_app.logger.error(f"Analiz bulunamadı: {analysis_id}")
        return False, "Analiz bulunamadı"
    
    try:
        # Analizi başlat
        analysis.start_analysis()
        
        # Dosyayı al
        file = analysis.file
        
        # Dosya türüne göre uygun analiz metodunu çağır
        if file.file_type == 'image':
            success, message = analyze_image(analysis)
        elif file.file_type == 'video':
            success, message = analyze_video(analysis)
        else:
            analysis.fail_analysis("Desteklenmeyen dosya türü")
            return False, "Desteklenmeyen dosya türü"
        
        if success:
            # Analiz sonuçlarını hesapla
            calculate_overall_scores(analysis)
            analysis.complete_analysis()
            return True, "Analiz başarıyla tamamlandı"
        else:
            analysis.fail_analysis(message)
            return False, message
    
    except Exception as e:
        error_message = f"Analiz hatası: {str(e)}"
        current_app.logger.error(error_message)
        analysis.fail_analysis(error_message)
        return False, error_message


def analyze_image(analysis):
    """
    Bir resmi analiz eder.
    Bu fonksiyon resim dosyaları için içerik analizi yapar ve sonuçları veritabanına kaydeder.
    Şiddet, yetişkin içeriği, taciz, silah, madde kullanımı ve güvenli analizi yapar.
    
    Args:
        analysis: Analiz nesnesi
        
    Returns:
        tuple: (başarı durumu, mesaj)
    """
    file = analysis.file
    
    try:
        # Resmi yükle
        image = load_image(file.file_path)
        if image is None:
            return False, "Resim yüklenemedi"
        
        # İçerik analizi yap
        content_analyzer = ContentAnalyzer()
        violence_score, adult_content_score, harassment_score, weapon_score, drug_score, safe_score, detected_objects = content_analyzer.analyze_image(file.file_path)
        
        # Analiz sonuçlarını veritabanına kaydet
        detection = ContentDetection(
            analysis_id=analysis.id,
            frame_path=file.file_path
        )
        
        # NumPy türlerini Python türlerine dönüştürdüğümüzden emin olalım
        detection.violence_score = float(violence_score)
        detection.adult_content_score = float(adult_content_score)
        detection.harassment_score = float(harassment_score)
        detection.weapon_score = float(weapon_score)
        detection.drug_score = float(drug_score)
        detection.safe_score = float(safe_score)
        
        # JSON uyumlu nesneyi kaydet
        try:
            detection.set_detected_objects(detected_objects)
        except Exception as e:
            logger.error(f"set_detected_objects hatası: {str(e)}")
            logger.error(f"Hata izi: {traceback.format_exc()}")
            detection._detected_objects = "[]"  # Boş liste olarak ayarla
        
        db.session.add(detection)
        
        # Eğer yaş analizi isteniyorsa, yüzleri tespit et ve yaşları tahmin et
        if analysis.include_age_analysis:
            age_estimator = InsightFaceAgeEstimator()
            faces = age_estimator.model.get(image)
            persons = {}
            for i, face in enumerate(faces):
                logger.warning(f"Yüz {i} - face objesi: {face.__dict__ if hasattr(face, '__dict__') else face}")
                if face.age is None:
                    logger.warning(f"Yüz {i} için yaş None, atlanıyor. Face: {face}")
                    continue
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                
                # Bounding box'ı kontrol et
                try:
                    # 1. Negatif koordinatları düzelt
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    
                    # 2. Görüntünün dışına taşan koordinatları sınırlandır
                    img_height, img_width = image.shape[:2]
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    # 3. Geçersiz bounding box kontrolü
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"Geçersiz bounding box boyutları (x2 <= x1 veya y2 <= y1): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        continue
                    
                    # 4. Yüz boyutu çok küçük mü?
                    w = x2 - x1
                    h = y2 - y1
                    MIN_FACE_SIZE = 20  # 30x30 piksel altındaki yüzleri eleme
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        logger.warning(f"Yüz çok küçük, atlanıyor: w={w}, h={h}, min={MIN_FACE_SIZE}")
                        continue
                    
                    # 5. Aspect ratio kontrolü - çok uzun veya geniş yüzleri eleme
                    aspect_ratio = w / h if h > 0 else 0
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        logger.warning(f"Anormal yüz oranı (aspect ratio), atlanıyor: w={w}, h={h}, ratio={aspect_ratio:.2f}")
                        continue
                    
                    # Yüz alanı hesabı için boyutları güncelle
                    bbox = [x1, y1, w, h]
                    
                    logger.info(f"Geçerli yüz bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}, ratio={aspect_ratio:.2f}")
                    
                except (ValueError, TypeError) as bbox_err:
                    logger.warning(f"Bounding box dönüşüm hatası: {str(bbox_err)}")
                    continue
                
                if x1 >= 0 and y1 >= 0 and w > 0 and h > 0 and x1+w <= image.shape[1] and y1+h <= image.shape[0]:
                    person_id = f"{analysis.id}_person_{i}"
                    
                    # Yüz görüntüsünü çıkar
                    face_roi = image[y1:y2, x1:x2]
                    
                    # Yaş tahmini ve güven skoru hesapla
                    estimated_age, confidence = age_estimator.estimate_age(face_roi)
                    
                    # DEBUG: CLIP modeli ile hesaplanan güven skorunu ve önceki InsightFace güven skorunu karşılaştır
                    logger.info(f"DEBUG - Güven Skorları Karşılaştırma - Kare #{i}: InsightFace={face.confidence}, CLIP={confidence}")
                    
                    if estimated_age is None:
                        logger.warning(f"Kare #{i}: Yaş tahmini yapılamadı (track {person_id})")
                        continue
                                
                    age = float(estimated_age)
                    
                    logger.info(f"Kare #{i}: Yaş: {age:.1f}, Güven: {confidence:.2f} (track {person_id})")
                    
                    # Takipteki kişi için en iyi kareyi kaydet (yüksek güven skoru varsa)
                    if person_id not in persons or confidence > persons[person_id]['confidence']:
                        persons[person_id] = {
                            'confidence': confidence,
                            'frame_path': file.file_path,
                            'timestamp': None,
                            'bbox': (x1, y1, w, h),
                            'age': age
                        }
                    
                    # AgeEstimation kaydını oluştur veya güncelle
                    try:
                        age_est = AgeEstimation.query.filter_by(
                            analysis_id=analysis.id,
                            person_id=person_id
                        ).first()
                        
                        if not age_est:
                            age_est = AgeEstimation(
                                analysis_id=analysis.id,
                                person_id=person_id,
                                frame_path=file.file_path,
                                estimated_age=age,
                                confidence_score=confidence
                            )
                        else:
                            # Sadece daha yüksek güven skorlu sonuçları güncelle
                            if confidence > age_est.confidence_score:
                                age_est.frame_path = file.file_path
                                age_est.estimated_age = age
                                age_est.confidence_score = confidence
                            
                        db.session.add(age_est)
                        
                        # Overlay oluştur
                        out_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}", "overlays")
                        os.makedirs(out_dir, exist_ok=True)
                        out_name = f"{person_id}_{os.path.basename(file.file_path)}"
                        out_path = os.path.join(out_dir, out_name)
                        
                        try:
                            # Görüntüyü kopyala ve overlay ekle
                            image_with_overlay = image.copy()
                            x2, y2 = x1 + w, y1 + h
                            
                            # Sınırları kontrol et
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(image.shape[1], x2)
                            y2 = min(image.shape[0], y2)
                            
                            # Çerçeve çiz
                            cv2.rectangle(image_with_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Metin için arka plan oluştur
                            text = f"ID: {person_id.split('_')[-1]}  YAS: {int(age)}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            text_y = y1 - 10 if y1 > 20 else y1 + h + 25
                            
                            # Metin arka planı için koordinatları hesapla
                            text_bg_x1 = x1
                            text_bg_y1 = text_y - text_size[1] - 5
                            text_bg_x2 = x1 + text_size[0] + 10
                            text_bg_y2 = text_y + 5
                            
                            # Arka plan çiz
                            cv2.rectangle(image_with_overlay, 
                                        (text_bg_x1, text_bg_y1),
                                        (text_bg_x2, text_bg_y2),
                                        (0, 0, 0),
                                        -1)
                            
                            # Metni çiz
                            cv2.putText(image_with_overlay, text, (x1 + 5, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Overlay'i kaydet
                            success = cv2.imwrite(out_path, image_with_overlay)
                            if not success:
                                logger.error(f"Overlay kaydedilemedi: {out_path}")
                                logger.error(f"[OVERLAY HATASI] Dosya yazma hatası - Koordinatlar: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")
                                continue
                                
                            # Göreceli yolu hesapla ve kaydet
                            rel_path = os.path.relpath(out_path, current_app.config['STORAGE_FOLDER']).replace('\\', '/')
                            age_est.processed_image_path = rel_path
                            db.session.add(age_est)
                            logger.info(f"[DEBUG] Overlay başarıyla oluşturuldu: person_id={person_id}, frame={file.file_path}, path={rel_path}")
                            
                        except Exception as overlay_err:
                            logger.error(f"Overlay oluşturma hatası (person_id={person_id}): {str(overlay_err)}")
                            logger.error(f"[DETAY] Overlay hatası detayları: Kare boyutu={image.shape if image is not None else 'None'}")
                            logger.error(f"[DETAY] Overlay bölgesi: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")
                            logger.error(f"[DETAY] Overlay hata traceback: {traceback.format_exc()}")
                            continue
                            
                    except Exception as db_err:
                        logger.error(f"Veritabanı işlemi hatası (person_id={person_id}): {str(db_err)}")
                        continue
            
            db.session.commit()
        
        # Değişiklikleri veritabanına kaydet
        db.session.commit()
        analysis.update_progress(100)  # İlerleme durumunu %100 olarak güncelle
        
        return True, "Resim analizi tamamlandı"
    
    except Exception as e:
        db.session.rollback()  # Hata durumunda değişiklikleri geri al
        return False, f"Resim analizi hatası: {str(e)}"


def analyze_video(analysis):
    """
    Bir videoyu analiz eder.
    Bu fonksiyon video dosyaları için içerik analizi yapar, video karelerini çıkararak 
    her kare için şiddet, yetişkin içeriği, taciz, silah, madde kullanımı ve güvenli analizi yapar.
    Ayrıca istenirse yüz tespiti ve yaş tahmini de gerçekleştirir.
    
    Args:
        analysis: Analiz nesnesi
        
    Returns:
        tuple: (başarı durumu, mesaj)
    """
    # Tüm logları bir dosyaya yaz - Bu şekilde tam analiz loglarını görebiliriz
    log_file_path = os.path.join(current_app.config['PROCESSED_FOLDER'], f"analysis_{analysis.id}_full.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"[DETAY LOG] Detaylı analiz logu başlatıldı: {log_file_path}")
    
    file = analysis.file
    frames_per_second = analysis.frames_per_second or 1  # Varsayılan saniyede 1 kare
    
    try:
        # Video karelerini çıkar ve geçici klasöre kaydet
        frames_folder = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}")
        os.makedirs(frames_folder, exist_ok=True)
        
        logger.info(f"Video analizi başlıyor: Analiz #{analysis.id}, Dosya: {file.original_filename}, FPS: {frames_per_second}")
        frame_paths, total_frames, duration = extract_frames(file.file_path, frames_folder, frames_per_second)
        
        if not frame_paths:
            logger.error(f"Video kareleri çıkarılamadı: Analiz #{analysis.id}")
            return False, "Video kareleri çıkarılamadı"
        
        logger.info(f"Kare çıkarma tamamlandı: {len(frame_paths)} kare, klasör: {frames_folder}")
        for fp, ts in frame_paths:
            logger.info(f"Kare çıkarıldı: {fp}, exists={os.path.exists(fp)}")
        
        logger.info(f"Video analizi: Toplam {len(frame_paths)} kare çıkarıldı, Video süresi: {duration:.2f} saniye")
        
        # İlerleme veritabanına kaydedilirken mesaj güncelle
        analysis.status_message = f"Video analizi: {len(frame_paths)} kare işlenecek"
        db.session.commit()
        
        # İçerik analizörü hazırla
        content_analyzer = ContentAnalyzer()
        logger.info(f"ContentAnalyzer başarıyla yüklendi: Analiz #{analysis.id}")
        
        # Yaş tahmini için estimator hazırla
        if analysis.include_age_analysis:
            age_estimator = InsightFaceAgeEstimator(det_size=(800, 800))
            # Benzeşme eşik değerini 0.5'ten 0.3'e düşürerek farklı yüzlerin ayrı kişiler olarak tespit edilmesini sağla
            tracker = DeepSort(max_age=45, n_init=2, nms_max_overlap=1.0, embedder=None, max_cosine_distance=0.3)
            logger.info(f"DeepSORT tracker başlatıldı: Analiz #{analysis.id}, Parametreler:")
            logger.info(f"  - max_age=45 (30 yerine): Yüzlerin görünmediği karelerde 45 kare boyunca takip edilmesi")
            logger.info(f"  - n_init=2 (3 yerine): Yüzlerin track ID alması için min. 2 karede görünmesi yeterli")
            logger.info(f"  - det_size=(800, 800): Daha detaylı yüz tespiti için büyük tespit boyutu")
            logger.info(f"  - max_cosine_distance=0.3 (0.6 yerine): Farklı yüzleri daha kesin şekilde ayrı kişiler olarak tespit etmek için daha düşük eşik değeri")
        else:
            age_estimator = None
            tracker = None
        
        # Kişi takibi için yardımcı sözlük (aynı kişinin farklı karelerde izlenmesi için)
        person_tracker = {}
        
        # İşlenecek toplam kare sayısı ve diğer istatistikleri logla
        detected_faces_count = 0
        high_risk_frames_count = 0
        risk_threshold = 0.7  # Yüksek risk eşiği
        
        person_best_frames = {}  # {person_id: {confidence, frame_path, timestamp, bbox, age}}
        
        for i, (frame_path, timestamp) in enumerate(frame_paths):
            detections = []
            # Önceki için durum mesajını ve ilerlemeyi güncelle
            progress = (i / len(frame_paths)) * 100
            status_message = f"Kare {i+1}/{len(frame_paths)} analiz ediliyor ({progress:.1f}%)"
            
            if i % 5 == 0 or i == len(frame_paths) - 1:  # Her 5 karede bir veya son kare
                analysis.status_message = status_message
                analysis.update_progress(progress)
                
                # Her 10 karede bir ilerleme durumunu logla
                if i % 10 == 0 or i == len(frame_paths) - 1:
                    logger.info(f"Video analizi ilerliyor: Analiz #{analysis.id}, Kare {i+1}/{len(frame_paths)}, "
                               f"Zaman {timestamp:.2f}s, İlerleme: %{progress:.1f}")
            
            # Kareyi yükle
            image = cv2.imread(frame_path)
            if image is None:
                logger.error(f"[DEBUG] Video analizi - Kare okunamadı: {frame_path}")
                continue
            
            # İçerik analizi yap
            violence_score, adult_content_score, harassment_score, weapon_score, drug_score, safe_score, detected_objects = content_analyzer.analyze_image(frame_path)
            
            # Her skor ve nesneyi ayrı ayrı dönüştür
            violence_score = float(violence_score)
            adult_content_score = float(adult_content_score)
            harassment_score = float(harassment_score)
            weapon_score = float(weapon_score)
            drug_score = float(drug_score)
            safe_score = float(safe_score)
            
            # Yüksek risk skorlu kareleri say
            max_risk_score = max(violence_score, adult_content_score, harassment_score, weapon_score, drug_score)
            if max_risk_score >= risk_threshold:
                high_risk_frames_count += 1
                score_str = f"Şiddet: {violence_score:.2f}, Yetişkin: {adult_content_score:.2f}, Taciz: {harassment_score:.2f}, Silah: {weapon_score:.2f}, Madde: {drug_score:.2f}, Güvenli: {safe_score:.2f}"
                logger.info(f"Yüksek riskli kare tespit edildi: Analiz #{analysis.id}, Kare {i+1}, Zaman {timestamp:.2f}s, Skorlar: {score_str}")
            
            # Nesneleri manuel olarak güvenli hale getir
            safe_objects = ensure_serializable(detected_objects)
            
            try:
                # Test et - Serileştirilebilir mi?
                json_text = json.dumps(safe_objects)
                if not json_text:
                    # Boş dize ise, None veya geçersiz JSON olabilir
                    logger.warning(f"Serileştirilen JSON geçersiz veya boş: {safe_objects}")
                    safe_objects = []
            except Exception as ser_err:
                # Serileştirilemiyorsa, boş liste kullan
                logger.error(f"Nesneler serileştirilemedi: {str(ser_err)}")
                logger.error(f"Nesne tipi: {type(safe_objects)}")
                logger.error(f"Nesne içeriği (kısmi): {str(safe_objects)[:500]}")
                safe_objects = []
                
            # Analiz sonuçlarını veritabanına kaydet
            detection = ContentDetection(
                analysis_id=analysis.id,
                frame_path=frame_path,
                frame_timestamp=timestamp
            )
            
            detection.violence_score = float(violence_score)
            detection.adult_content_score = float(adult_content_score)
            detection.harassment_score = float(harassment_score)
            detection.weapon_score = float(weapon_score)
            detection.drug_score = float(drug_score)
            detection.safe_score = float(safe_score)
            detection.set_detected_objects(safe_objects)
            
            # Nesnenin serileştirilebilir olup olmadığını kontrol et
            try:
                detection_dict = detection.to_dict()
                json.dumps(detection_dict)
            except Exception as json_err:
                logger.error(f"ContentDetection to_dict serileştirilemedi: {str(json_err)}")
                # Sorun detected_objects'de ise onu temizle
                detection._detected_objects = '[]'
            
            db.session.add(detection)
            
            # Yaş analizi yapılacaksa yüz tespiti ve yaş tahmini yap
            if age_estimator and tracker:
                try:
                    faces = age_estimator.model.get(image)
                    logger.info(f"Kare: {frame_path}, {len(faces) if faces else 0} yüz bulundu.")
                    
                    if not faces or len(faces) == 0:
                        logger.warning(f"Karede hiç yüz tespit edilemedi: {frame_path}, overlay oluşturulmayacak.")
                        continue
                        
                    detections = []
                    for idx, face in enumerate(faces):
                        try:
                            # Yüz özelliklerini kontrol et
                            if not hasattr(face, 'age') or not hasattr(face, 'confidence') or not hasattr(face, 'bbox'):
                                logger.warning(f"Yüz {idx} için gerekli özellikler eksik: {face}")
                                continue
                                
                            age = face.age
                            confidence = face.confidence
                            
                            # DEBUG: InightFace modelinin kendi güven skorunu logla
                            logger.info(f"DEBUG - InightFace Ham Değerler: Yüz #{idx}, Yaş={age}, InsightFace Güven Skoru={confidence}")
                            
                            # Eğer confidence None ise 0.5 olarak ayarla
                            if confidence is None:
                                logger.info(f"DEBUG - InightFace Güven Skoru NONE döndürdü, varsayılan 0.5 kullanılıyor - Yüz #{idx}")
                                confidence = 0.5
                            
                            # Yaş ve güven skorunu kontrol et
                            if not isinstance(age, (int, float)) or not isinstance(confidence, (int, float)):
                                logger.warning(f"Geçersiz yaş veya güven skoru: age={age}, confidence={confidence}")
                                continue
                                
                            if age < 1 or age > 100 or confidence < 0.1:
                                logger.warning(f"Geçersiz yaş aralığı veya düşük güven: age={age}, confidence={confidence}")
                                continue
                                
                            # Bounding box'ı kontrol et
                            try:
                                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                                
                                # 1. Negatif koordinatları düzelt
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                
                                # 2. Görüntünün dışına taşan koordinatları sınırlandır
                                img_height, img_width = image.shape[:2]
                                x2 = min(img_width, x2)
                                y2 = min(img_height, y2)
                                
                                # 3. Geçersiz bounding box kontrolü
                                if x2 <= x1 or y2 <= y1:
                                    logger.warning(f"Geçersiz bounding box boyutları (x2 <= x1 veya y2 <= y1): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                    continue
                                
                                # 4. Yüz boyutu çok küçük mü?
                                w = x2 - x1
                                h = y2 - y1
                                MIN_FACE_SIZE = 20  # 30x30 piksel altındaki yüzleri eleme
                                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                                    logger.warning(f"Yüz çok küçük, atlanıyor: w={w}, h={h}, min={MIN_FACE_SIZE}")
                                    continue
                                
                                # 5. Aspect ratio kontrolü - çok uzun veya geniş yüzleri eleme
                                aspect_ratio = w / h if h > 0 else 0
                                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                                    logger.warning(f"Anormal yüz oranı (aspect ratio), atlanıyor: w={w}, h={h}, ratio={aspect_ratio:.2f}")
                                    continue
                                
                                # Yüz alanı hesabı için boyutları güncelle
                                bbox = [x1, y1, w, h]
                                
                                logger.info(f"Geçerli yüz bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}, ratio={aspect_ratio:.2f}")
                                
                            except (ValueError, TypeError) as bbox_err:
                                logger.warning(f"Bounding box dönüşüm hatası: {str(bbox_err)}")
                                continue
                                
                            # Embedding kontrolü
                            if not hasattr(face, 'embedding') or face.embedding is None:
                                logger.warning(f"Yüz {idx} için embedding bulunamadı")
                                continue
                                
                            embedding = face.embedding
                            
                            detections.append({
                                'bbox': bbox,
                                'embedding': embedding,
                                'face': face
                            })
                            
                            # Geçerli bir yüz tespiti yapıldığında sayacı artır
                            detected_faces_count += 1
                            logger.info(f"Kare: {frame_path}, Yüz {idx} geçerli, toplam tespit: {detected_faces_count}")
                            
                        except Exception as face_err:
                            logger.error(f"Yüz {idx} işlenirken hata: {str(face_err)}")
                            continue
                            
                    if not detections:
                        logger.warning(f"İşlenebilir yüz bulunamadı: {frame_path}")
                        continue
                        
                    # DeepSORT ile takip
                    try:
                        tracks = tracker.update_tracks(
                            [(d['bbox'], 1.0, "face") for d in detections],
                            embeds=[d['embedding'] for d in detections],
                            frame=image
                        )
                        
                        for det, track in zip(detections, tracks):
                            if not track.is_confirmed():
                                continue
                                
                            track_id = f"{analysis.id}_person_{track.track_id}"
                            face = det['face']
                            x1, y1, w, h = det['bbox']
                            age = face.age
                            
                            # Yaş tahmini ve güven skoru hesapla
                            x2, y2 = x1 + w, y1 + h  # Eksik satır: x2 ve y2 koordinatlarını hesapla
                            face_roi = image[y1:y2, x1:x2]
                            if face_roi.size == 0:
                                logger.warning(f"Kare #{i}: Boş yüz bölgesi (track {track_id})")
                                continue
                                
                            # Yeni güncellendi - doğrudan estimate_age fonksiyonunu kullan
                            estimated_age, confidence = age_estimator.estimate_age(face_roi)
                            
                            # DEBUG: CLIP modeli ile hesaplanan güven skorunu ve önceki InsightFace güven skorunu karşılaştır
                            logger.info(f"DEBUG - Güven Skorları Karşılaştırma - Kare #{i}: InsightFace={face.confidence}, CLIP={confidence}")
                            
                            if estimated_age is None:
                                logger.warning(f"Kare #{i}: Yaş tahmini yapılamadı (track {track_id})")
                                continue
                                
                            age = float(estimated_age)
                            
                            logger.info(f"Kare #{i}: Yaş: {age:.1f}, Güven: {confidence:.2f} (track {track_id})")
                            
                            # Takipteki kişi için en iyi kareyi kaydet (yüksek güven skoru varsa)
                            if track_id not in person_best_frames or confidence > person_best_frames[track_id]['confidence']:
                                person_best_frames[track_id] = {
                                    'confidence': confidence,
                                    'frame_path': frame_path, 
                                    'timestamp': timestamp,
                                    'bbox': (x1, y1, w, h),
                                    'age': age
                                }
                            
                            # AgeEstimation kaydını oluştur veya güncelle
                            try:
                                age_est = AgeEstimation.query.filter_by(
                                    analysis_id=analysis.id,
                                    person_id=track_id
                                ).first()
                                
                                if not age_est:
                                    age_est = AgeEstimation(
                                        analysis_id=analysis.id,
                                        person_id=track_id,
                                        frame_path=frame_path,
                                        estimated_age=age,
                                        confidence_score=confidence
                                    )
                                else:
                                    # Sadece daha yüksek güven skorlu sonuçları güncelle
                                    if confidence > age_est.confidence_score:
                                        age_est.frame_path = frame_path
                                        age_est.estimated_age = age
                                        age_est.confidence_score = confidence
                            
                                db.session.add(age_est)
                                
                                # Overlay oluştur
                                out_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}", "overlays")
                                os.makedirs(out_dir, exist_ok=True)
                                out_name = f"{track_id}_{os.path.basename(frame_path)}"
                                out_path = os.path.join(out_dir, out_name)
                                
                                try:
                                    # Görüntüyü kopyala ve overlay ekle
                                    image_with_overlay = image.copy()
                                    x2, y2 = x1 + w, y1 + h
                                    
                                    # Sınırları kontrol et
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(image.shape[1], x2)
                                    y2 = min(image.shape[0], y2)
                                    
                                    # Çerçeve çiz
                                    cv2.rectangle(image_with_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Metin için arka plan oluştur
                                    text = f"ID: {track_id.split('_')[-1]}  YAS: {int(age)}"
                                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                    text_y = y1 - 10 if y1 > 20 else y1 + h + 25
                                    
                                    # Metin arka planı için koordinatları hesapla
                                    text_bg_x1 = x1
                                    text_bg_y1 = text_y - text_size[1] - 5
                                    text_bg_x2 = x1 + text_size[0] + 10
                                    text_bg_y2 = text_y + 5
                                    
                                    # Arka plan çiz
                                    cv2.rectangle(image_with_overlay, 
                                                (text_bg_x1, text_bg_y1),
                                                (text_bg_x2, text_bg_y2),
                                                (0, 0, 0),
                                                -1)
                                    
                                    # Metni çiz
                                    cv2.putText(image_with_overlay, text, (x1 + 5, text_y), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    # Overlay'i kaydet
                                    success = cv2.imwrite(out_path, image_with_overlay)
                                    if not success:
                                        logger.error(f"Overlay kaydedilemedi: {out_path}")
                                        logger.error(f"[OVERLAY HATASI] Dosya yazma hatası - Koordinatlar: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")
                                        continue
                                        
                                    # Göreceli yolu hesapla ve kaydet
                                    rel_path = os.path.relpath(out_path, current_app.config['STORAGE_FOLDER']).replace('\\', '/')
                                    age_est.processed_image_path = rel_path
                                    db.session.add(age_est)
                                    logger.info(f"[DEBUG] Overlay başarıyla oluşturuldu: person_id={track_id}, frame={frame_path}, path={rel_path}")
                                    
                                except Exception as overlay_err:
                                    logger.error(f"Overlay oluşturma hatası (person_id={track_id}): {str(overlay_err)}")
                                    logger.error(f"[DETAY] Overlay hatası detayları: Kare boyutu={image.shape if image is not None else 'None'}")
                                    logger.error(f"[DETAY] Overlay bölgesi: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")
                                    logger.error(f"[DETAY] Overlay hata traceback: {traceback.format_exc()}")
                                    continue
                                    
                            except Exception as db_err:
                                logger.error(f"Veritabanı işlemi hatası (person_id={track_id}): {str(db_err)}")
                                continue
                            
                    except Exception as track_err:
                        logger.error(f"DeepSORT takip hatası: {str(track_err)}")
                        continue
                        
                except Exception as age_err:
                    logger.error(f"Yaş analizi hatası: {str(age_err)}")
                    continue
            
            # Belirli aralıklarla işlemleri veritabanına kaydet (her 10 karede bir)
            if i % 10 == 0:
                db.session.commit()
                
                # Socket.io ile anlık ilerleme bilgisi gönder
                try:
                    from app import socketio
                    # Kategori skorlarını da ekleyelim
                    scores = {
                        'violence': violence_score,
                        'adult_content': adult_content_score,
                        'harassment': harassment_score,
                        'weapon': weapon_score, 
                        'drug': drug_score,
                        'safe': safe_score
                    }
                    
                    # Skorlar toplamı 1 olacak şekilde normalize et (UI için)
                    total = sum(scores.values())
                    if total > 0:
                        normalized_scores = {k: v/total for k, v in scores.items()}
                    else:
                        normalized_scores = scores
                        
                    socketio.emit('analysis_progress', {
                        'analysis_id': analysis.id,
                        'file_id': analysis.file_id,
                        'current_frame': i + 1,
                        'total_frames': len(frame_paths),
                        'progress': progress,
                        'timestamp': timestamp,
                        'detected_faces': detected_faces_count,
                        'high_risk_frames': high_risk_frames_count,
                        'status': status_message,
                        'scores': normalized_scores  # Normalize edilmiş skorları ekle
                    })
                except Exception as socket_err:
                    logger.warning(f"Socket.io ilerleme bildirimi hatası: {str(socket_err)}")
        
        # Tüm değişiklikleri veritabanına kaydet
        db.session.commit()
        
        # Analiz tamamlandı, istatistikleri logla
        unique_persons = len(person_best_frames) if person_best_frames else 0
        logger.info(f"Video analizi tamamlandı: Analiz #{analysis.id}, Dosya: {file.original_filename}")
        logger.info(f"  - Toplam {len(frame_paths)} kare analiz edildi")
        logger.info(f"  - {detected_faces_count} yüz tespiti, {unique_persons} benzersiz kişi")
        logger.info(f"  - {high_risk_frames_count} yüksek riskli kare tespit edildi")
        
        # Döngü bittikten sonra, her kişi için en iyi kareyi işleyip kaydet (önce kopyala, sonra overlay)
        overlay_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}", "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        for person_id, info in person_best_frames.items():
            frame_path = info['frame_path']
            x1, y1, w, h = info['bbox']
            age = info['age']
            confidence = info['confidence']
            try:
                # Orijinal kareyi oku
                src_path = frame_path  # path birleştirme yok, doğrudan kullan
                logger.info(f"[DEBUG] Overlay için kare okunuyor: {src_path}, exists={os.path.exists(src_path)}")
                image = cv2.imread(src_path)
                if image is None:
                    logger.error(f"[DEBUG] Overlay için kare okunamadı (person_id={person_id}): {src_path}")
                    continue
                out_name = f"{person_id}.jpg"
                out_path = os.path.join(overlay_dir, out_name)
                # Önce orijinal kareyi kopyala
                cv2.imwrite(out_path, image)
                logger.info(f"[DEBUG] Overlay dosyası kaydedildi - path={out_path}, exists={os.path.exists(out_path)}")
                # Sonra overlay işlemi yap
                try:
                    x2, y2 = x1 + w, y1 + h
                    image_with_overlay = image.copy()
                    cv2.rectangle(image_with_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"ID: {person_id.split('_')[-1]}  YAS: {int(age)}"
                    text_y = y1 - 10 if y1 > 20 else y1 + h + 25
                    cv2.putText(image_with_overlay, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imwrite(out_path, image_with_overlay)
                    logger.info(f"[DEBUG] Overlay oluşturuluyor - path={out_path}, exists={os.path.exists(out_path)}")
                    rel_path = os.path.relpath(out_path, current_app.config['STORAGE_FOLDER']).replace('\\', '/')
                    logger.info(f"[DEBUG] Overlay DB path - rel_path={rel_path}")
                    # AgeEstimation kaydını güncelle (en yüksek confidence'lı olanı bul)
                    best_est = AgeEstimation.query.filter_by(analysis_id=analysis.id, person_id=person_id).order_by(desc('confidence_score')).first()
                    if best_est:
                        logger.info(f"[DEBUG] Overlay güncelleme: person_id={person_id}, rel_path={rel_path}, DB'deki eski path={best_est.processed_image_path}")
                        best_est.processed_image_path = rel_path
                        db.session.add(best_est)
                        db.session.commit()
                        logger.info(f"[DEBUG] Overlay DB'ye yazıldı - person_id={person_id}, rel_path={rel_path}, DB'deki yeni path={best_est.processed_image_path}")
                        # DB'den tekrar oku ve logla
                        kontrol_est = AgeEstimation.query.filter_by(id=best_est.id).first()
                        logger.info(f"[DEBUG] DB'den tekrar okundu: id={best_est.id}, processed_image_path={kontrol_est.processed_image_path}")
                    else:
                        logger.info(f"[DEBUG] Overlay yeni kayıt: person_id={person_id}, rel_path={rel_path}")
                        new_est = AgeEstimation(
                            analysis_id=analysis.id,
                            person_id=person_id,
                            frame_path=frame_path,
                            estimated_age=age,
                            confidence_score=confidence,
                        )
                        new_est.set_face_location(x1, y1, w, h)
                        new_est.processed_image_path = rel_path
                        db.session.add(new_est)
                        db.session.commit()
                        logger.info(f"[DEBUG] Overlay yeni kayıt DB'ye yazıldı - person_id={person_id}, rel_path={rel_path}, DB'deki path={new_est.processed_image_path}")
                        # DB'den tekrar oku ve logla
                        kontrol_est = AgeEstimation.query.filter_by(id=new_est.id).first()
                        logger.info(f"[DEBUG] DB'den tekrar okundu: id={new_est.id}, processed_image_path={kontrol_est.processed_image_path}")
                except Exception as overlay_err:
                    logger.warning(f"[DEBUG] Overlay oluşturulamadı (person_id={person_id}, kare={src_path})")
                    logger.error(f"[DETAY] Overlay hatası detayları: Kare boyutu={image.shape if image is not None else 'None'}")
                    logger.error(f"[DETAY] Overlay bölgesi: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")
                    logger.error(f"[DETAY] Overlay hata traceback: {traceback.format_exc()}")
                    # Overlay hatası olursa orijinal kare klasörde kalır
            except Exception as copy_err:
                logger.error(f"[DEBUG] Kare kopyalama/overlay işlemi hatası (person_id={person_id}): {str(copy_err)}")
                continue
        db.session.commit()
        
        if not person_best_frames:
            logger.warning(f"[DEBUG] Analiz sonunda hiç geçerli yüz (yaş/confidence) bulunamadı, overlay klasörü ve dosyası oluşmayacak: {overlay_dir}")
        
        # Eklenen log handler'ını kaldır 
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path:
                logger.removeHandler(handler)
                handler.close()
                logger.info(f"[DETAY LOG] Detaylı analiz logu tamamlandı: {log_file_path}")
                break
        
        return True, "Video analizi tamamlandı"
    
    except Exception as e:
        logger.error(f"[DEBUG] Video analizi başarısız oldu: Analiz #{analysis.id}, Hata: {str(e)}")
        # Hata durumunda da log handler'ını kaldır
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path:
                logger.removeHandler(handler)
                handler.close()
                break
        db.session.rollback()  # Hata durumunda değişiklikleri geri al
        return False, f"Video analizi hatası: {str(e)}"


def calculate_overall_scores(analysis):
    """
    Bir analiz için genel skorları hesaplar.
    Tüm tespit edilen kareler için ortalama skorları hesaplar ve 
    en yüksek risk içeren kareyi belirler.
    
    Args:
        analysis: Skorları hesaplanacak analiz nesnesi
    """
    try:
        # Tüm içerik tespitlerini veritabanından al
        detections = ContentDetection.query.filter_by(analysis_id=analysis.id).all()
        
        if not detections:
            return
        
        # Her kategori için tüm skorları topla
        violence_scores = [d.violence_score for d in detections]
        adult_content_scores = [d.adult_content_score for d in detections]
        harassment_scores = [d.harassment_score for d in detections]
        weapon_scores = [d.weapon_score for d in detections]
        drug_scores = [d.drug_score for d in detections]
        safe_scores = [d.safe_score for d in detections]
        
        # Genel skorları ortalama alarak hesapla
        analysis.overall_violence_score = sum(violence_scores) / len(violence_scores) if violence_scores else 0
        analysis.overall_adult_content_score = sum(adult_content_scores) / len(adult_content_scores) if adult_content_scores else 0
        analysis.overall_harassment_score = sum(harassment_scores) / len(harassment_scores) if harassment_scores else 0
        analysis.overall_weapon_score = sum(weapon_scores) / len(weapon_scores) if weapon_scores else 0
        analysis.overall_drug_score = sum(drug_scores) / len(drug_scores) if drug_scores else 0
        analysis.overall_safe_score = sum(safe_scores) / len(safe_scores) if safe_scores else 0
        
        # Skorların toplamının %100 olmasını sağla
        total_score = (analysis.overall_violence_score + analysis.overall_adult_content_score + 
                      analysis.overall_harassment_score + analysis.overall_weapon_score + 
                      analysis.overall_drug_score + analysis.overall_safe_score)
        
        if total_score > 0:
            # Her skoru normalize et (toplamı 1.0 olacak şekilde)
            analysis.overall_violence_score /= total_score
            analysis.overall_adult_content_score /= total_score
            analysis.overall_harassment_score /= total_score
            analysis.overall_weapon_score /= total_score
            analysis.overall_drug_score /= total_score
            analysis.overall_safe_score /= total_score
            
            # Toplam skorun 1.0 olduğunu doğrula
            logger.info(f"Normalize edilmiş toplam skor: {analysis.overall_violence_score + analysis.overall_adult_content_score + analysis.overall_harassment_score + analysis.overall_weapon_score + analysis.overall_drug_score + analysis.overall_safe_score}")
        
        # En yüksek riskli kareyi bul (en yüksek riskli kare "safe" hariç)
        highest_risk_score = 0
        highest_risk_category = None
        highest_risk_detection = None
        
        for detection in detections:
            scores = {
                'violence': detection.violence_score,
                'adult_content': detection.adult_content_score,
                'harassment': detection.harassment_score,
                'weapon': detection.weapon_score,
                'drug': detection.drug_score
            }
            
            # En yüksek skora sahip kategoriyi bul
            max_category = max(scores, key=scores.get)
            max_score = scores[max_category]
            
            # Şimdiye kadarki en yüksek skordan büyükse güncelle
            if max_score > highest_risk_score:
                highest_risk_score = max_score
                highest_risk_category = max_category
                highest_risk_detection = detection
        
        # En riskli kare bilgilerini analiz nesnesine kaydet
        if highest_risk_detection:
            analysis.highest_risk_frame = highest_risk_detection.frame_path
            analysis.highest_risk_frame_timestamp = highest_risk_detection.frame_timestamp
            analysis.highest_risk_score = highest_risk_score
            analysis.highest_risk_category = highest_risk_category
        
        # Değişiklikleri veritabanına kaydet
        db.session.commit()
    
    except Exception as e:
        current_app.logger.error(f"[DEBUG] Genel skor hesaplama hatası: {str(e)}")
        db.session.rollback()  # Hata durumunda değişiklikleri geri al


def get_analysis_results(analysis_id):
    """
    Bir analizin tüm sonuçlarını getirir.
    Bu fonksiyon, analiz sonuçlarını kapsamlı bir şekilde raporlamak için 
    kullanılır ve tüm tespit ve tahminleri içerir.
    
    Args:
        analysis_id: Sonuçları getirilecek analizin ID'si
        
    Returns:
        dict: Analiz sonuçlarını içeren sözlük
    """
    analysis = Analysis.query.get(analysis_id)
    
    if not analysis:
        return {'error': 'Analiz bulunamadı'}
    
    # Analiz henüz tamamlanmamış ise durumu bilgisini döndür
    if analysis.status != 'completed':
        return {
            'status': analysis.status,
            'progress': analysis.progress,
            'message': 'Analiz henüz tamamlanmadı'
        }
    
    # Ana analiz sonuçları
    result = analysis.to_dict()
    
    # İçerik tespitlerini sonuçlara ekle
    content_detections = ContentDetection.query.filter_by(analysis_id=analysis_id).all()
    result['content_detections'] = [cd.to_dict() for cd in content_detections]
    
    # Eğer yaş analizi yapıldıysa, yaş tahminlerini sonuçlara ekle
    if analysis.include_age_analysis:
        age_estimations = AgeEstimation.query.filter_by(analysis_id=analysis_id).all()
        
        # Kişi bazlı yaş tahminlerini grupla (aynı kişinin farklı tahminleri)
        persons = {}
        for estimation in age_estimations:
            person_id = estimation.person_id
            if person_id not in persons:
                persons[person_id] = []
            persons[person_id].append(estimation.to_dict())
        
        # Her kişi için en güvenilir yaş tahminini bul
        best_estimations = []
        for person_id, estimations in persons.items():
            # Güvenilirlik skoru en yüksek tahmini seç
            best_estimation = max(estimations, key=lambda e: e['confidence_score'])
            best_estimations.append(best_estimation)
        
        result['age_estimations'] = best_estimations
    
    logger.info(f"[DEBUG] API yanıtı - processed_image_path={best_estimation.get('processed_image_path', None)} (person_id={person_id})")
    
    return result 