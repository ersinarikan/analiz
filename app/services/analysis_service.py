import os
import logging
import traceback
from datetime import datetime
import json
import cv2
import threading
import time
import concurrent.futures
from queue import Queue

from flask import current_app
from app import db
from app.ai.content_analyzer import ContentAnalyzer
from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
from app.utils.model_state import get_content_analyzer
from app.utils.model_state import get_age_estimator as model_state_get_age_estimator
from app.models.analysis import Analysis, ContentDetection, AgeEstimation
from app.models.feedback import Feedback
from app.models.file import File
from app.utils.image_utils import load_image
from app.routes.settings_routes import FACTORY_DEFAULTS
from app.json_encoder import NumPyJSONEncoder
from app.services.db_service import safe_database_session
from deep_sort_realtime.deepsort_tracker import DeepSort
from app.utils.person_tracker import PersonTrackerManager
from app.utils.face_utils import extract_face_features
from app.utils.path_utils import to_rel_path

logger = logging.getLogger(__name__)

# Thread-safe session management
_session_lock = threading.Lock()

# 🚀 ASYNC AGE ESTIMATION: ThreadPoolExecutor for non-blocking age estimation
_age_estimation_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, 
    thread_name_prefix="AgeEstimation"
)

def _async_age_estimation(age_estimator, image, face, face_idx, analysis_id, person_id):
    """
    Age estimation işlemini background thread'de yapar - main thread bloklanmaz!
    Bu sayede CLIP hesaplamaları sırasında WebSocket bağlantısı cevap verebilir.
    """
    try:
        logger.info(f"[ASYNC_AGE] Thread başlatıldı: Yüz #{face_idx} (person_id={person_id})")
        start_time = time.time()
        
        # Age estimation işlemini yap (bu kısım 11-12 saniye sürebilir)
        estimated_age, confidence, pseudo_data = age_estimator.estimate_age(image, face)
        
        elapsed_time = time.time() - start_time
        logger.info(f"[ASYNC_AGE] Thread tamamlandı: Yüz #{face_idx}, Süre: {elapsed_time:.2f}s, Yaş={estimated_age}, Güven={confidence}")
        
        return {
            'face_idx': face_idx,
            'person_id': person_id,
            'estimated_age': estimated_age,
            'confidence': confidence,
            'pseudo_data': pseudo_data,
            'processing_time': elapsed_time
        }
        
    except Exception as e:
        logger.error(f"[ASYNC_AGE] Thread hatası - Yüz #{face_idx}: {str(e)}")
        return {
            'face_idx': face_idx,
            'person_id': person_id,
            'estimated_age': None,
            'confidence': None,
            'pseudo_data': None,
            'error': str(e)
        }

class AnalysisService:
    """
    Analiz işlemlerini yöneten ana servis sınıfı.
    - Yüz tespiti, yaş/cinsiyet tahmini, içerik analizi gibi işlemleri yönetir.
    - Kuyruk ve arka plan işleyişini koordine eder.
    """
    
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
            
            # File bilgilerini önceden çek
            file_info = {
                'original_filename': file.original_filename,
                'file_type': file.file_type,
                'filename': file.filename,
                'file_path': file.file_path
            }
                
            # Yeni bir analiz oluştur
            analysis = Analysis(
                file_id=file_id,
                frames_per_second=frames_per_second,
                include_age_analysis=include_age_analysis
            )
            
            # Başlangıç durumunu ayarla
            analysis.status = 'pending'
            
            db.session.add(analysis)
            db.session.commit()
            
            logger.info(f"Analiz oluşturuldu: #{analysis.id} - Dosya: {file_info['original_filename']}, Durum: pending")
            
            # WebSocket üzerinden analiz başlangıç bildirimi gönder
            try:
                from app.routes.websocket_routes import emit_analysis_started
                emit_analysis_started(analysis.id, f"Analiz başlatıldı: {file_info['original_filename']}", file_id)
                logger.info(f"Analiz başlatıldı - WebSocket bildirimi gönderildi: #{analysis.id}, File: {file_id}")
            except Exception as socket_err:
                logger.warning(f"WebSocket bildirim hatası: {str(socket_err)}")
            
            # Analizi kuyruğa ekle
            from app.services.queue_service import add_to_queue
            add_to_queue(analysis.id)
            
            logger.info(f"Analiz kuyruğa eklendi: #{analysis.id}")
            
            return analysis
                
        except Exception as e:
            logger.error(f"Analiz başlatma hatası: {str(e)}", exc_info=True)
            db.session.rollback()
            return None
    
    def cancel_analysis(self, analysis_id):
        """
        Devam eden bir analizi iptal eder.
        
        Args:
            analysis_id: İptal edilecek analizin ID'si
            
        Returns:
            bool: İptal başarılı mı?
        """
        try:
            with safe_database_session() as session:
                analysis = Analysis.query.get(analysis_id)
                if not analysis:
                    return False
                    
                # Analiz durumunu iptal edildi olarak işaretle
                analysis.status = 'cancelled'
                analysis.updated_at = datetime.now()
                session.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Analiz iptal hatası: {str(e)}", exc_info=True)
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
            
            # Analizi kuyruğa ekle
            from app.services.queue_service import add_to_queue
            add_to_queue(new_analysis.id)
            
            logger.info(f"Tekrar analiz kuyruğa eklendi: #{new_analysis.id}")
            
            return new_analysis
            
        except Exception as e:
            logger.error(f"Analiz tekrar deneme hatası: {str(e)}", exc_info=True)
            db.session.rollback()
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
    # Flask app context kontrolü ve yönetimi
    app = None
    needs_context = False
    
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            app = current_app._get_current_object()
        else:
            needs_context = True
    except (RuntimeError, ImportError):
        needs_context = True
    
    def _perform_analysis():
        """Actual analysis logic"""
        try:
            analysis = Analysis.query.get(analysis_id)
            
            if not analysis:
                logger.error(f"Analiz bulunamadı: {analysis_id}")
                return False, "Analiz bulunamadı"
            
            # Analiz başlıyor
            analysis.start_analysis()
            db.session.commit()
            
            logger.info(f"[SVC_LOG][START_ANALYSIS] Analiz BAŞLATILDI. Analiz ID: {analysis.id}, Dosya ID: {analysis.file_id}")  # YENİ LOG
            
            # WebSocket ile analiz başlatma bildirimi gönder
            try:
                from app.routes.websocket_routes import emit_analysis_progress
                emit_analysis_progress(analysis.id, 0, "Analiz başlatıldı, dosya hazırlanıyor...", analysis.file_id)
            except Exception as ws_err:
                logger.warning(f"WebSocket started progress event hatası: {str(ws_err)}")
            
            # Dosyayı al
            file = analysis.file
            
            # Dosya türüne göre uygun analiz metodunu çağır
            if file.file_type == 'image':
                success, message = analyze_image(analysis)
            elif file.file_type == 'video':
                success, message = analyze_video(analysis)
            else:
                analysis.fail_analysis("Desteklenmeyen dosya türü")
                db.session.commit()
                return False, "Desteklenmeyen dosya türü"
            
            if success:
                # Analiz sonuçlarını hesapla
                calculate_overall_scores(analysis)
                analysis.complete_analysis()
                db.session.commit()
                
                # WebSocket ile analiz tamamlanma bildirimi gönder
                try:
                    from app.routes.websocket_routes import emit_analysis_completed
                    emit_analysis_completed(analysis.id, "Analiz başarıyla tamamlandı", analysis.file_id)
                except Exception as ws_err:
                    logger.warning(f"WebSocket completion event hatası: {str(ws_err)}")
                
                return True, "Analiz başarıyla tamamlandı"
            else:
                analysis.fail_analysis(message)
                db.session.commit()
                
                # WebSocket ile analiz başarısızlık bildirimi gönder
                try:
                    from app.routes.websocket_routes import emit_analysis_completed
                    emit_analysis_completed(analysis.id, f"Analiz başarısız: {message}", analysis.file_id)
                except Exception as ws_err:
                    logger.warning(f"WebSocket failed event hatası: {str(ws_err)}")
                
                return False, message
                
        except Exception as e:
            error_message = f"Analiz hatası: {str(e)}"
            logger.error(error_message, exc_info=True)
            try:
                analysis = Analysis.query.get(analysis_id)
                if analysis:
                    analysis.fail_analysis(error_message)
                    db.session.commit()
            except Exception as db_err:
                logger.error(f"Error updating analysis status: {str(db_err)}", exc_info=True)
            return False, error_message
    
    # Context management
    if needs_context:
        # Import burada yapıyoruz circular import'ı önlemek için
        try:
            from flask import current_app
            app_ctx = current_app._get_current_object()
            with app_ctx.app_context():
                return _perform_analysis()
        except Exception as e:
            logger.error(f"App context oluşturma hatası: {str(e)}", exc_info=True)
            return False, f"App context hatası: {str(e)}"
    else:
        # Zaten app context var
        return _perform_analysis()


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
    logger.info(f"[DEBUG_ANALYZE] analyze_image ÇAĞRILDI. Analiz ID: {analysis.id}, Dosya: {file.original_filename}")
    logger.info(f"[SVC_LOG][ANALYZE_IMAGE] Resim analizi BAŞLADI. Analiz ID: {analysis.id}, Dosya: {file.original_filename}") # YENİ LOG

    try:
        # İlk progress güncellemesi
        analysis.update_progress(10, "Resim yükleniyor...")
        db.session.commit()
        
        # Resmi yükle
        image = load_image(file.file_path)
        if image is None:
            logger.error(f"[SVC_LOG][ANALYZE_IMAGE] Resim YÜKLENEMEDİ. Analiz ID: {analysis.id}, Dosya Yolu: {file.file_path}") # YENİ LOG
            return False, "Resim yüklenemedi"
        
        logger.info(f"[SVC_LOG][ANALYZE_IMAGE] Resim başarıyla yüklendi. Analiz ID: {analysis.id}") # YENİ LOG
        
        # İçerik analizi
        analysis.update_progress(25, "İçerik analizi yapılıyor...")
        
        # Görüntüyü ContentAnalyzer ile analiz et
        content_analyzer = ContentAnalyzer()
        logger.info(f"[SVC_LOG][ANALYZE_IMAGE] ContentAnalyzer çağrılmadan önce. Analiz ID: {analysis.id}") # YENİ LOG
        violence_score, adult_content_score, harassment_score, weapon_score, drug_score, safe_score, detected_objects = content_analyzer.analyze_image(file.file_path)
        logger.info(f"[SVC_LOG][ANALYZE_IMAGE] ContentAnalyzer çağrıldı. Analiz ID: {analysis.id}. Adult Score: {adult_content_score}") # YENİ LOG
        
        # İçerik analizi tamamlandı
        analysis.update_progress(50, "İçerik analizi sonuçları kaydediliyor...")
        db.session.commit()
        
        # Analiz sonuçlarını veritabanına kaydet
        detection = ContentDetection(
            analysis_id=analysis.id,
            frame_path=file.file_path,
            frame_timestamp=None,
            frame_index=None
        )
        
        # NumPy türlerini Python türlerine dönüştürdüğümüzden emin olalım
        detection.violence_score = float(violence_score)
        detection.adult_content_score = float(adult_content_score)
        detection.harassment_score = float(harassment_score)
        detection.weapon_score = float(weapon_score)
        detection.drug_score = float(drug_score)
        detection.safe_score = float(safe_score)
        logger.info(f"[SVC_LOG][ANALYZE_IMAGE] ContentDetection skorları atandı. Analiz ID: {analysis.id}. Adult Score: {detection.adult_content_score}") # YENİ LOG
        
        # JSON uyumlu nesneyi kaydet
        try:
            detection.set_detected_objects(detected_objects)
        except Exception as e:
            logger.error(f"[SVC_LOG][ANALYZE_IMAGE] set_detected_objects hatası: {str(e)}. Analiz ID: {analysis.id}") # YENİ LOG
            logger.error(f"Hata izi: {traceback.format_exc()}")
            detection._detected_objects = "[]"  # Boş liste olarak ayarla
        
        db.session.add(detection)
        
        # Eğer yaş analizi isteniyorsa, yüzleri tespit et ve yaşları tahmin et
        if analysis.include_age_analysis:
            # Yaş analizi başlatılıyor
            analysis.update_progress(60, "Yüz tespiti ve yaş analizi yapılıyor...")
            db.session.commit()
            
            from app.utils.model_state import get_age_estimator
            age_estimator = get_age_estimator()
            faces = age_estimator.model.get(image)
            persons = {}
            
            total_faces = len(faces)
            for i, face in enumerate(faces):
                logger.warning(f"Yüz {i} - face objesi: {face.__dict__ if hasattr(face, '__dict__') else face}")
                if face.age is None:
                    logger.warning(f"Yüz {i} için yaş None, atlanıyor. Face: {face}")
                    continue
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                w = x2 - x1
                h = y2 - y1
                if x1 >= 0 and y1 >= 0 and w > 0 and h > 0 and x1+w <= image.shape[1] and y1+h <= image.shape[0]:
                    person_id = f"{analysis.id}_person_{i}"
                    logger.info(f"[SVC_LOG] Yüz #{i} (person_id={person_id}) için ASYNC yaş tahmini başlatılıyor. BBox: [{x1},{y1},{w},{h}]")
                    
                    # 🚀 ASYNC AGE ESTIMATION: Background thread'de yap - main thread bloklanmasın!
                    future = _age_estimation_executor.submit(
                        _async_age_estimation, 
                        age_estimator, image, face, i, analysis.id, person_id
                    )
                    
                    # Short timeout - age estimation uzun sürerse bile main thread devam etsin
                    try:
                        result = future.result(timeout=1.0)  # 1 saniye bekle, sonra devam et
                        estimated_age = result['estimated_age']
                        confidence = result['confidence'] 
                        pseudo_data = result['pseudo_data']
                        logger.info(f"[SVC_LOG] Yüz #{i} SYNC sonuç alındı: Yaş={estimated_age}, Güven={confidence}")
                    except concurrent.futures.TimeoutError:
                        # Age estimation henüz bitmedi, ama main thread devam etsin
                        logger.info(f"[SVC_LOG] Yüz #{i} için age estimation background'da devam ediyor...")
                        estimated_age = None
                        confidence = None
                        pseudo_data = None

                    if estimated_age is None or confidence is None:
                        logger.info(f"[SVC_LOG] Yüz #{i} için yaş/güven henüz hazır değil, background işlem devam ediyor.")
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
                        
                        # Convert absolute file.file_path to relative path for storage
                        relative_frame_path = to_rel_path(file.file_path)
                        if os.path.isabs(file.file_path):
                            try:
                                relative_frame_path = os.path.relpath(file.file_path, current_app.config['STORAGE_FOLDER']).replace('\\\\', '/')
                            except ValueError as ve:
                                logger.error(f"Error creating relative path for {file.file_path} relative to {current_app.config['STORAGE_FOLDER']}: {ve}. Falling back to original path.")
                        else:
                            # Ensure consistent path separators even if already relative
                            relative_frame_path = file.file_path.replace('\\\\', '/')

                        if not age_est:
                            embedding = face.embedding if hasattr(face, 'embedding') and face.embedding is not None else None
                            if embedding is not None:
                                if hasattr(embedding, 'tolist'):
                                    embedding_str = ",".join(str(float(x)) for x in embedding.tolist())
                                elif isinstance(embedding, (list, tuple)):
                                    embedding_str = ",".join(str(float(x)) for x in embedding)
                                else:
                                    embedding_str = str(embedding)
                            else:
                                embedding_str = None
                            age_est = AgeEstimation(
                                analysis_id=analysis.id,
                                person_id=person_id,
                                frame_path=relative_frame_path, # Use relative path
                                estimated_age=age,
                                confidence_score=confidence,
                                embedding=embedding_str
                            )
                            logger.info(f"[SVC_LOG] Yeni AgeEstimation kaydı oluşturuldu: {person_id}")
                        else:
                            if confidence > age_est.confidence_score:
                                age_est.frame_path = relative_frame_path # Use relative path
                                age_est.estimated_age = age
                                age_est.confidence_score = confidence
                                age_est.embedding = embedding_str # embedding güncelle
                                logger.info(f"[SVC_LOG] AgeEstimation kaydı güncellendi (daha iyi güven): {person_id}, Yeni Güven: {confidence:.4f}")
                        
                        db.session.add(age_est)
                        
                        # Sözde etiket verisi varsa Feedback tablosuna kaydet
                        if pseudo_data:
                            try:
                                logger.info(f"[SVC_LOG] Sözde etiket verisi kaydediliyor. Person ID: {person_id}")
                                embedding = None
                                if pseudo_data.get("embedding") is not None:
                                    emb = pseudo_data.get("embedding")
                                    if isinstance(emb, (list, tuple)):
                                        embedding = ",".join(str(float(x)) for x in emb)
                                    elif hasattr(emb, 'tolist'):
                                        embedding = ",".join(str(float(x)) for x in emb.tolist())
                                    else:
                                        embedding = str(emb)
                                feedback_entry = Feedback(
                                    frame_path=to_rel_path(file.file_path), # Resim yolu (relatif)
                                    face_bbox=pseudo_data.get("face_bbox"),
                                    embedding=embedding,
                                    pseudo_label_original_age=pseudo_data.get("pseudo_label_original_age"),
                                    pseudo_label_clip_confidence=pseudo_data.get("pseudo_label_clip_confidence"),
                                    feedback_source=pseudo_data.get("feedback_source", "PSEUDO_BUFFALO_HIGH_CONF"),
                                    feedback_type="age_pseudo", # Ya da pseudo_data.get("feedback_type")
                                    content_id=analysis.file_id, # content_id'yi analysis'ten al
                                    analysis_id=analysis.id,
                                    person_id=person_id 
                                )
                                db.session.add(feedback_entry)
                                logger.info(f"[SVC_LOG] Sözde etiket için Feedback kaydı eklendi: {feedback_entry.id}")
                            except Exception as fb_err:
                                logger.error(f"[SVC_LOG] Sözde etiket Feedback kaydı oluşturulurken hata: {str(fb_err)}")
                                # Bu hata ana akışı durdurmamalı, sadece loglanmalı.
                        
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
                            text = f"ID: {person_id.split('_')[-1]}  YAS: {round(age)}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            text_y = y1 - 10 if y1 > 20 else y1 + h + 25
                            
                            # Metin arka planı için koordinatları hesapla
                            text_bg_x1 = x1
                            text_bg_y1 = text_y - text_size[1] - 5
                            text_bg_x2, text_bg_y2 = x1 + text_size[0] + 10, text_y + 5
                            
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
                                continue
                                
                            # Göreceli yolu hesapla ve kaydet
                            rel_path = to_rel_path(out_path)
                            rel_path = normalize_rel_storage_path(rel_path)
                            age_est.processed_image_path = rel_path
                            db.session.add(age_est)

                        except Exception as overlay_err:
                            logger.error(f"Overlay oluşturma hatası (person_id={person_id}): {str(overlay_err)}", exc_info=True)
                            continue
                            
                    except Exception as db_err:
                        logger.error(f"[SVC_LOG] DB hatası (person_id={person_id}): {str(db_err)}", exc_info=True)
                        continue
            
            db.session.commit()
            
            # Yaş analizi tamamlandı - GERÇEK tamamlanma kontrolü
            logger.info(f"[SVC_LOG] Yaş analizi döngüsü tamamlandı, {total_faces} yüz için işlem yapıldı. Final kontrol başlıyor...")
            
            # Son DB commit'ten sonra kısa bir bekleme - tüm CLIP hesaplamalarının tamamlandığından emin olmak için
            time.sleep(0.5)  # 500ms bekleme
            
            # Final veritabanı durumunu kontrol et
            final_age_estimations = db.session.query(AgeEstimation).filter_by(analysis_id=analysis.id).all()
            logger.info(f"[SVC_LOG] Final kontrol: {len(final_age_estimations)} AgeEstimation kaydı veritabanında mevcut")
            
            analysis.update_progress(90, "Yaş analizi tamamlandı, sonuçlar kaydediliyor...")
            db.session.commit()
        else:
            # Yaş analizi yapılmadıysa direkt sona yakın progress
            analysis.update_progress(85, "Analiz sonuçları kaydediliyor...")
            db.session.commit()
        
        # Tüm değişiklikleri veritabanına kaydet
        db.session.commit()
        
        # Final sync bekleme - tüm asenkron işlemlerin tamamlanması için
        time.sleep(0.2)  # 200ms ek bekleme
        
        # Son progress güncellemesi
        analysis.update_progress(95, "Analiz sonuçlandırılıyor...")
        db.session.commit()
        
        # Final bekleme ve kontrol
        time.sleep(0.3)  # 300ms son bekleme
        
        analysis.update_progress(100, "Analiz tamamlandı")
        logger.info(f"[SVC_LOG][ANALYZE_IMAGE] Resim analizi BAŞARIYLA TAMAMLANDI. Analiz ID: {analysis.id}")
        
        return True, "Resim analizi tamamlandı"
    
    except Exception as e:
        db.session.rollback()  # Hata durumunda değişiklikleri geri al
        logger.error(f"[SVC_LOG][ANALYZE_IMAGE] Resim analizi HATASI: {str(e)}. Analiz ID: {analysis.id}", exc_info=True)
        logger.error(f"Detaylı Hata İzi (analyze_image): {traceback.format_exc()}")
        return False, f"Resim analizi hatası: {str(e)}"


def analyze_video(analysis):
    """
    Video analizini gerçekleştirir.
    Her kareyi analiz eder ve tüm içerik tespitlerini veritabanına yazar.
    Yaş analizi yapılıyorsa DeepSORT ile kişileri takip eder ve her kişi için yaş tahminleri kaydeder.
    
    Args:
        analysis: Analiz nesnesi (Analysis model)
        
    Returns:
        Tuple[bool, str]: (başarı, mesaj)
    """
    try:
        file = File.query.get(analysis.file_id)
        if not file:
            logger.error(f"Analiz için dosya bulunamadı: #{analysis.id}")
            return False, "Dosya bulunamadı"
        
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        if not os.path.exists(file_path):
            logger.error(f"Video dosyası bulunamadı: {file_path}")
            return False, "Video dosyası bulunamadı"
        
        # Video yakalama nesnesi oluştur
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Video dosyası açılamadı: {file_path}")
            return False, "Video dosyası açılamadı"
        
        # Video FPS, kare sayısı, süre hesapla
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        frames_per_second_config = analysis.frames_per_second # Bu değer kullanıcıdan geliyor mu yoksa configden mi alınmalı?
                                                            # Şimdilik analysis objesinden alınıyor.
        if not frames_per_second_config or frames_per_second_config <= 0:
            frames_per_second_config = fps  # Eğer belirtilmemişse, videonun kendi FPS'ini kullan
        
        # Kaç kare atlayacağımızı hesapla (her saniye için kaç kare analiz edilecek)
        frame_skip = max(1, int(fps / frames_per_second_config))
        
        # Kare indekslerini oluştur (istenen FPS'e göre)
        frame_indices = range(0, frame_count, frame_skip)
        
        # Video'dan işlenecek kareleri oku ve kaydet (ilk 30 kare için)
        frame_paths = []
        frames_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i_frame, frame_idx in enumerate(frame_indices):
            if i_frame >= 30: # Sadece ilk 30 kareyi önceden kaydet
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            # Timestamp'i tutarlı formatta kaydet (her zaman .XX formatı)
            timestamp_str = f"{timestamp:.2f}" if timestamp % 1 != 0 else f"{timestamp:.0f}.00"
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}_{timestamp_str}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # İçerik analizi için model yükle
        try:
            content_analyzer_instance = get_content_analyzer() # get_ ile alınıyor
            logger.info(f"İçerik analiz modeli yüklendi: Analiz #{analysis.id}")
        except Exception as model_err:
            logger.error(f"İçerik analiz modeli yüklenemedi: {str(model_err)}")
            return False, f"Model yükleme hatası: {str(model_err)}"
        
        # Yaş analizi için gerekli modelleri ve ayarları yükle
        age_estimator = None
        tracker = None
        person_tracker_manager = None

        if analysis.include_age_analysis:
            try:
                from app.utils.model_state import get_age_estimator
                age_estimator = get_age_estimator() # model_state'den alınıyor
                logger.info(f"Yaş tahmin modeli yüklendi: Analiz #{analysis.id}")
                
                # Config'den takip parametrelerini oku
                max_lost_frames_config = current_app.config.get('MAX_LOST_FRAMES', FACTORY_DEFAULTS['MAX_LOST_FRAMES'])
                tracking_reliability_thresh_config = current_app.config.get('TRACKING_RELIABILITY_THRESHOLD', FACTORY_DEFAULTS['TRACKING_RELIABILITY_THRESHOLD'])
                id_change_thresh_config = current_app.config.get('ID_CHANGE_THRESHOLD', FACTORY_DEFAULTS['ID_CHANGE_THRESHOLD'])
                embedding_dist_thresh_config = current_app.config.get('EMBEDDING_DISTANCE_THRESHOLD', FACTORY_DEFAULTS['EMBEDDING_DISTANCE_THRESHOLD'])

                logger.info(f"DeepSORT başlatılıyor: max_age={max_lost_frames_config}, n_init=2, Analiz #{analysis.id}")
                tracker = DeepSort(max_age=max_lost_frames_config, n_init=2, nms_max_overlap=1.0, embedder=None) # embedder=None (InsightFace kullanacak)
                
                person_tracker_manager = PersonTrackerManager(
                    reliability_threshold=tracking_reliability_thresh_config,
                    max_frames_missing=max_lost_frames_config,
                    id_change_threshold=id_change_thresh_config,
                    embedding_distance_threshold=embedding_dist_thresh_config
                )
                logger.info(f"PersonTrackerManager başlatıldı (reliability_threshold={tracking_reliability_thresh_config}, max_frames_missing={max_lost_frames_config}, id_change_threshold={id_change_thresh_config}, embedding_distance_threshold={embedding_dist_thresh_config}): Analiz #{analysis.id}")
            except Exception as age_err:
                logger.error(f"Yaş tahmin modelleri veya takipçi yüklenemedi: {str(age_err)}", exc_info=True)
                logger.warning(f"Yaş analizi devre dışı bırakıldı: Analiz #{analysis.id}")
                analysis.include_age_analysis = False # Yaş analizi yapılamıyorsa kapat
                db.session.commit()
        
        # İlerleme bilgisi
        total_frames_to_process = len(frame_indices)
        high_risk_frames_count = 0
        detected_faces_count = 0
        
        # Tüm kareleri işle
        # person_best_frames = {} # REMOVED
        track_genders = {}
        processed_persons_with_data = set() # Keep this to know which persons to process later
        
        # Video'yu baştan sonra kadar işle
        for i, frame_idx in enumerate(frame_indices):
            try: # ADDED MAIN TRY FOR FRAME PROCESSING
                progress = min(100, int((i / total_frames_to_process) * 100))
                analysis.update_progress(progress)  # Bu metot zaten commit yapıyor
                timestamp = frame_idx / fps
                status_message = f"Kare #{i+1}/{total_frames_to_process} işleniyor ({timestamp:.1f}s)"
                
                # Kareyi oku
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, image = cap.read()
                if not ret:
                    logger.warning(f"Kare okunamadı: #{frame_idx}, işlem sonlandırıldı")
                    break
                
                # Kareyi kaydet
                if i >= 30:  # İlk 30 kare zaten kaydedilmişti
                    # Timestamp'i tutarlı formatta kaydet
                    timestamp_str = f"{timestamp:.2f}" if timestamp % 1 != 0 else f"{timestamp:.0f}.00"
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}_{timestamp_str}.jpg")
                    cv2.imwrite(frame_path, image)
                    frame_paths.append(frame_path)
                else:
                    frame_path = frame_paths[i]
                
                # İçerik analizi yap
                try:
                    # Her kategori için skorlar
                    violence_score, adult_content_score, harassment_score, weapon_score, drug_score, safe_score, safe_objects = content_analyzer_instance.analyze_image(
                        image
                    )
                    
                    # Eğer herhangi bir kategoride yüksek risk varsa, yüksek riskli kare sayısını artır
                    if max(violence_score, adult_content_score, harassment_score, weapon_score, drug_score) > 0.7:
                        high_risk_frames_count += 1
                except Exception as e_content_analysis: # ADDED EXCEPTION HANDLING
                    logger.error(f"Kare #{i} ({frame_path}) içerik analizi hatası: {str(e_content_analysis)}")
                    violence_score, adult_content_score, harassment_score, weapon_score, drug_score, safe_score = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 # Default to safe
                    safe_objects = []
                
                # ContentDetection nesnesini oluştur ve veritabanına ekle
                detection = ContentDetection(
                    analysis_id=analysis.id,
                    frame_path=frame_path,
                        frame_timestamp=timestamp,
                        frame_index=frame_idx
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
                        logger.info(f"[SVC_LOG][VID] Kare #{i} ({timestamp:.2f}s): {len(faces) if faces else 0} yüz tespit edildi.")
                        
                        if not faces or len(faces) == 0:
                            logger.warning(f"[SVC_LOG][VID] Karede hiç yüz tespit edilemedi: {frame_path}, overlay oluşturulmayacak.")
                            continue
                            
                        detections = []
                        face_features_list = []  # Yüz özelliklerini saklayacak liste
                        
                        for idx, face in enumerate(faces):
                            try:
                                # Yüz özelliklerini kontrol et
                                if not hasattr(face, 'age') or not hasattr(face, 'confidence') or not hasattr(face, 'bbox'):
                                    logger.warning(f"Yüz {idx} için gerekli özellikler eksik: {face}")
                                    continue
                                age = face.age
                                confidence = face.confidence
                                if confidence is None:
                                    confidence = 0.5
                                if not isinstance(age, (int, float)) or not isinstance(confidence, (int, float)):
                                    logger.warning(f"Geçersiz yaş veya güven skoru: age={age}, confidence={confidence}")
                                    continue
                                if age < 1 or age > 100 or confidence < 0.1:
                                    logger.warning(f"Geçersiz yaş aralığı veya düşük güven: age={age}, confidence={confidence}")
                                    continue
                                # Bounding box'ı kontrol et
                                try:
                                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                                    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                                        logger.warning(f"Geçersiz bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                        continue
                                except (ValueError, TypeError) as bbox_err:
                                    logger.warning(f"Bounding box dönüşüm hatası: {str(bbox_err)}")
                                    continue
                                w = x2 - x1
                                h = y2 - y1
                                bbox = [x1, y1, w, h]
                                # Embedding kontrolü
                                embedding = face.embedding if hasattr(face, 'embedding') and face.embedding is not None else None
                                if embedding is not None:
                                    if hasattr(embedding, 'tolist'):
                                        embedding_vector = embedding.tolist()
                                        embedding_str = ",".join(str(float(x)) for x in embedding_vector)
                                    elif isinstance(embedding, (list, tuple)):
                                        embedding_vector = list(embedding)
                                        embedding_str = ",".join(str(float(x)) for x in embedding_vector)
                                    else:
                                        # Tek bir float veya yanlış tip
                                        embedding_vector = [float(embedding)]
                                        embedding_str = str(float(embedding))
                                else:
                                    embedding_vector = None
                                    embedding_str = None
                                # Yüz özelliklerini çıkar
                                face_features = extract_face_features(image, face, bbox)
                                face_features_list.append(face_features)
                                detections.append({
                                    'bbox': bbox,
                                    'embedding_vector': embedding_vector,  # float vektör (DeepSORT için)
                                    'embedding_str': embedding_str,        # string (veritabanı için)
                                    'face': face
                                })
                                logger.info(f"Kare: {frame_path}, Yüz {idx}: age={age}, confidence={confidence}")
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
                                embeds=[d['embedding_vector'] for d in detections],  # float vektörler!
                                frame=image
                            )
                            logger.info(f"[SVC_LOG][VID] Kare #{i}: DeepSORT {len(tracks)} track döndürdü.")
                                
                            # PersonTrackerManager ile güvenilir takipleri filtrele
                            reliable_tracks = person_tracker_manager.update(tracks, face_features_list, i)
                            logger.info(f"[SVC_LOG][VID] Kare #{i}: PersonTrackerManager {len(reliable_tracks)} güvenilir track döndürdü.")
                                
                            processed_track_ids = set() # Aynı karede birden fazla kez loglamayı önle
                            
                            active_detections_in_frame = []
                            for det_idx, (det_data, track_obj) in enumerate(zip(detections, tracks)):
                                    # Sadece güvenilir takipleri ekle
                                    if track_obj in reliable_tracks:
                                        active_detections_in_frame.append({'det': det_data, 'track': track_obj})

                            for item in active_detections_in_frame:
                                det = item['det']
                                track = item['track']

                                if not track.is_confirmed() or track.track_id in processed_track_ids:
                                    continue
                                processed_track_ids.add(track.track_id)
                                    
                                    # Bu kısımda gender_match kontrolü yerine PersonTrackerManager'ın güvenilirlik kontrolünü kullanıyoruz
                                    # Artık mevcut gender_match bloğunu kullanmak yerine, güvenilir takipleri işliyoruz
                                
                                track_id_str = f"{analysis.id}_person_{track.track_id}"
                                face_obj = det['face'] # Bu InsightFace face nesnesi

                                x1, y1, w, h = det['bbox']
                                logger.info(f"[SVC_LOG][VID] Kare #{i}: Track ID={track.track_id} (person_id={track_id_str}) için ASYNC yaş tahmini başlatılıyor. BBox: [{x1},{y1},{w},{h}]")
                                embedding_str = det['embedding_str']  # string (veritabanı için)
                                
                                # 🚀 ASYNC AGE ESTIMATION: Background thread'de yap - video frame processing bloklanmasın!
                                future = _age_estimation_executor.submit(
                                    _async_age_estimation, 
                                    age_estimator, image, face_obj, i, analysis.id, track_id_str
                                )
                                
                                # Video için daha kısa timeout - frame processing hızlı devam etmeli
                                try:
                                    result = future.result(timeout=0.5)  # 500ms bekle, sonra devam et
                                    estimated_age = result['estimated_age']
                                    confidence = result['confidence']
                                    pseudo_data = result['pseudo_data']
                                    logger.info(f"[SVC_LOG][VID] Kare #{i}: Track ID={track.track_id} SYNC sonuç: Yaş={estimated_age}, Güven={confidence}")
                                except concurrent.futures.TimeoutError:
                                    # Age estimation background'da devam ediyor, frame processing devam etsin
                                    logger.info(f"[SVC_LOG][VID] Kare #{i}: Track ID={track.track_id} age estimation background'da devam ediyor...")
                                    estimated_age = None
                                    confidence = None
                                    pseudo_data = None

                                if estimated_age is None or confidence is None:
                                    logger.info(f"[SVC_LOG][VID] Kare #{i}: Track ID={track.track_id} için yaş/güven henüz hazır değil, sonraki frame'e geç.")
                                    continue
                                
                                age = float(estimated_age)

                                # AgeEstimation record creation/update logic:
                                try:
                                    age_est = AgeEstimation.query.filter_by(analysis_id=analysis.id, person_id=track_id_str).first()
                                    db_bbox_to_store = [x1, y1, w, h]
                                    if not age_est:
                                        age_est = AgeEstimation(
                                            analysis_id=analysis.id,
                                            person_id=track_id_str,
                                            frame_path=frame_path,
                                            frame_timestamp=timestamp,  # Timestamp eklendi
                                            estimated_age=age,
                                            confidence_score=confidence,
                                            frame_index=frame_idx,
                                            face_bbox=json.dumps(db_bbox_to_store),
                                            embedding=embedding_str
                                        )
                                        logger.info(f"[SVC_LOG][VID] Yeni AgeEstimation: {track_id_str}, Kare: {frame_idx}, BBox: {db_bbox_to_store}")
                                    else:
                                        if confidence > age_est.confidence_score:
                                            age_est.frame_path = frame_path
                                            age_est.frame_timestamp = timestamp  # Timestamp eklendi
                                            age_est.estimated_age = age
                                            age_est.confidence_score = confidence
                                            age_est.frame_index = frame_idx
                                            age_est.face_bbox = json.dumps(db_bbox_to_store)
                                            age_est.embedding = embedding_str
                                            logger.info(f"[SVC_LOG][VID] AgeEstimation Güncelleme: {track_id_str}, Yeni Güven: {confidence:.4f}, Kare: {frame_idx}")
                                    db.session.add(age_est)
                                    processed_persons_with_data.add(track_id_str)
                                    
                                    # Sözde etiket verisi varsa Feedback tablosuna kaydet
                                    if pseudo_data:
                                        try:
                                            logger.info(f"[SVC_LOG][VID] Sözde etiket verisi kaydediliyor. Person ID: {track_id_str}, Kare Path: {frame_path}")
                                            embedding_fb = pseudo_data.get("embedding")
                                            if embedding_fb is not None:
                                                if hasattr(embedding_fb, 'tolist'):
                                                    embedding_fb_str = ",".join(str(float(x)) for x in embedding_fb.tolist())
                                                elif isinstance(embedding_fb, (list, tuple)):
                                                    embedding_fb_str = ",".join(str(float(x)) for x in embedding_fb)
                                                else:
                                                    embedding_fb_str = str(embedding_fb)
                                            else:
                                                embedding_fb_str = None
                                            feedback_entry = Feedback(
                                                frame_path=to_rel_path(file.file_path), 
                                                face_bbox=pseudo_data.get("face_bbox"),
                                                embedding=embedding_fb_str,
                                                pseudo_label_original_age=pseudo_data.get("pseudo_label_original_age"),
                                                pseudo_label_clip_confidence=pseudo_data.get("pseudo_label_clip_confidence"),
                                                feedback_source=pseudo_data.get("feedback_source", "PSEUDO_BUFFALO_HIGH_CONF"),
                                                feedback_type="age_pseudo",
                                                content_id=analysis.file_id,  # DÜZELTİLDİ: Artık file_id kullanılıyor
                                                analysis_id=analysis.id,
                                                person_id=track_id_str 
                                            )
                                            db.session.add(feedback_entry)
                                            logger.info(f"[SVC_LOG][VID] Sözde etiket için Feedback kaydı eklendi: {feedback_entry.id} (Person: {track_id_str})")
                                        except Exception as fb_err:
                                            logger.error(f"[SVC_LOG][VID] Sözde etiket Feedback kaydı oluşturulurken hata (Person: {track_id_str}): {str(fb_err)}")

                                except Exception as db_err:
                                    logger.error(f"[SVC_LOG][VID] DB hatası (track_id={track_id_str}, kare={i}): {str(db_err)}")
                                    continue
                                
                        except Exception as track_err:
                            logger.error(f"DeepSORT takip hatası: {str(track_err)}")
                            continue
                            
                    except Exception as age_err:
                        logger.error(f"Yaş analizi hatası: {str(age_err)}")
                        continue
                
                # Progress güncellemesini her 3 karede bir yap (daha responsive)
                if i % 3 == 0:
                    db.session.commit()
                    
                    # Progress bilgisi - her 3 karede bir
                    try:
                        # İlerleme durumunu logla
                        logger.info(f"Analiz #{analysis.id}: Kare {i+1}/{len(frame_indices)} ({progress:.1f}%) - Risk: {high_risk_frames_count} kare")
                    except Exception as progress_err:
                        logger.warning(f"İlerleme bildirimi hatası: {str(progress_err)}")
                
                # Büyük işlemleri her 10 karede bir kaydet
                elif i % 10 == 0:
                    db.session.commit()
            
            except Exception as frame_err: # ALIGNED WITH THE NEW MAIN TRY BLOCK
                logger.error(f"Kare #{i} ({frame_path}) analiz hatası: {str(frame_err)}")
                continue
        
        # Tüm değişiklikleri veritabanına kaydet
        db.session.commit()
        
        logger.info(f"Video analizi DB commit sonrası. Analiz ID: {analysis.id}, Include Age: {analysis.include_age_analysis}, Processed Persons Count: {len(processed_persons_with_data) if processed_persons_with_data else 'None'}")

        # Analiz tamamlandı, istatistikleri logla
        unique_persons_query = db.session.query(AgeEstimation.person_id).filter(AgeEstimation.analysis_id == analysis.id).distinct().count()
        logger.info(f"Video analizi tamamlandı: Analiz #{analysis.id}, Dosya: {file.original_filename}")
        logger.info(f"  - Toplam {len(frame_paths)} kare analiz edildi ({total_frames_to_process} hedeflenmişti)")
        logger.info(f"  - {detected_faces_count} yüz tespiti, {unique_persons_query} benzersiz kişi")
        logger.info(f"  - {high_risk_frames_count} yüksek riskli kare tespit edildi")
        
        # NEW OVERLAY GENERATION LOGIC
        if analysis.include_age_analysis and processed_persons_with_data:
            logger.info(f"Analiz #{analysis.id} için final overlayler oluşturuluyor. İşlenecek kişi sayısı: {len(processed_persons_with_data)}")
            base_overlay_dir = os.path.join(current_app.config['PROCESSED_FOLDER'], f"frames_{analysis.id}", 'overlays')
            os.makedirs(base_overlay_dir, exist_ok=True)

            for person_id_str in processed_persons_with_data:
                logger.info(f"Overlay oluşturma döngüsü: Kişi ID {person_id_str} işleniyor.")
                try:
                    best_est = db.session.query(AgeEstimation).filter_by(
                        analysis_id=analysis.id,
                        person_id=person_id_str
                    ).order_by(AgeEstimation.confidence_score.desc(), AgeEstimation.id.desc()).first()
                    
                    logger.info(f"Kişi {person_id_str} için best_est sorgulandı. Sonuç: {{'Bulundu' if best_est else 'Bulunamadı'}}")

                    if not best_est:
                        logger.warning(f"Kişi {person_id_str} için final AgeEstimation kaydı bulunamadı (best_est None), overlay atlanıyor.")
                        continue
                    
                    # Kaynak kare yolunu best_est.frame_path'ten al (bu geçici tam yol olabilir)
                    source_frame_for_overlay_path = best_est.frame_path
                    logger.info(f"Kişi {person_id_str} için kaynak kare yolu (best_est.frame_path): {source_frame_for_overlay_path}, best_est.estimated_age: {best_est.estimated_age}, best_est.confidence_score: {best_est.confidence_score}")

                    if not source_frame_for_overlay_path or not os.path.exists(source_frame_for_overlay_path):
                        logger.error(f"Overlay için kaynak kare {source_frame_for_overlay_path} bulunamadı/geçersiz (Kişi: {person_id_str}). Disk kontrolü: {{'Var' if source_frame_for_overlay_path and os.path.exists(source_frame_for_overlay_path) else 'Yok veya Path Hatalı'}}")
                        continue
                    
                    image_source_for_overlay = cv2.imread(source_frame_for_overlay_path)
                    if image_source_for_overlay is None:
                        logger.error(f"Overlay için kare okunamadı (Kişi: {person_id_str}): {source_frame_for_overlay_path}")
                        continue

                    age_to_display = round(best_est.estimated_age)  # JavaScript Math.round ile aynı davranış
                    logger.info(f"DEBUG - Kişi {person_id_str}: best_est.estimated_age={best_est.estimated_age}, round()={age_to_display}")
                    bbox_json_str = best_est.face_bbox
                    if not bbox_json_str:
                        logger.warning(f"Kişi {person_id_str} için BBox yok, overlay atlanıyor (Kayıt ID: {best_est.id}).")
                        continue
                    
                    try:
                        x1_bbox, y1_bbox, w_bbox, h_bbox = json.loads(bbox_json_str)
                    except (TypeError, ValueError) as json_parse_err:
                        logger.error(f"Kişi {person_id_str} BBox parse edilemedi ({bbox_json_str}): {json_parse_err}")
                        continue
                    
                    # Overlay çizimi (yaş ve kutu)
                    image_with_overlay = image_source_for_overlay.copy()
                    # person_id_str'den ID numarasını çıkar
                    person_number = person_id_str.split('_person_')[-1] if '_person_' in person_id_str else person_id_str
                    label = f"ID: {person_number}  YAS: {age_to_display}"
                    cv2.rectangle(image_with_overlay, (x1_bbox, y1_bbox), (x1_bbox + w_bbox, y1_bbox + h_bbox), (0, 255, 0), 2)
                    
                    # Metin için arka plan oluştur (görüntü analizindeki gibi)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_y = y1_bbox - 10 if y1_bbox > 20 else y1_bbox + h_bbox + 25
                    
                    # Metin arka planı için koordinatları hesapla
                    text_bg_x1 = x1_bbox
                    text_bg_y1 = text_y - text_size[1] - 5
                    text_bg_x2, text_bg_y2 = x1_bbox + text_size[0] + 10, text_y + 5
                    
                    # Arka plan çiz
                    cv2.rectangle(image_with_overlay, 
                                (text_bg_x1, text_bg_y1),
                                (text_bg_x2, text_bg_y2),
                                (0, 0, 0),
                                -1)
                    
                    # Metni çiz
                    cv2.putText(image_with_overlay, label, (x1_bbox + 5, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Benzersiz ve anlamlı bir dosya adı oluştur (orijinal kare adını içerebilir)
                    original_frame_basename = os.path.basename(source_frame_for_overlay_path) # ör: frame_000123.jpg
                    overlay_filename = f"{person_id_str}_overlay_{original_frame_basename}"
                    final_overlay_path_on_disk = os.path.join(base_overlay_dir, overlay_filename)
                    
                    # Overlay'li resmi diske kaydet
                    save_success = cv2.imwrite(final_overlay_path_on_disk, image_with_overlay)
                    if not save_success:
                        logger.error(f"Overlay dosyası diske kaydedilemedi: {final_overlay_path_on_disk}")
                        continue
                    
                    logger.info(f"Overlay başarıyla diske kaydedildi: {final_overlay_path_on_disk}")

                    # GÖRECELİ YOLU OLUŞTUR VE VERİTABANINA KAYDET
                    #STORAGE_FOLDER (örn: /.../WSANALIZ/storage) PROCESSED_FOLDER (örn: /.../WSANALIZ/storage/processed)
                    # base_overlay_dir (örn: /.../WSANALIZ/storage/processed/frames_ANALYSISID/overlays)
                    # final_overlay_path_on_disk (örn: /.../WSANALIZ/storage/processed/frames_ANALYSISID/overlays/FILENAME.jpg)
                    # Hedef: processed/frames_ANALYSISID/overlays/FILENAME.jpg
                    try:
                        relative_overlay_path_for_db = to_rel_path(final_overlay_path_on_disk)
                        relative_overlay_path_for_db = normalize_rel_storage_path(relative_overlay_path_for_db)
                    except ValueError as ve:
                        logger.error(f"Göreli yol oluşturulurken hata (final_overlay_path_on_disk='{final_overlay_path_on_disk}', STORAGE_FOLDER='{current_app.config['STORAGE_FOLDER']}'): {ve}")
                        # Fallback to a simpler relative path construction if relpath fails due to different drives on Windows etc.
                        # This assumes PROCESSED_FOLDER is a subfolder of STORAGE_FOLDER or correctly configured.
                        path_parts = final_overlay_path_on_disk.split(os.sep)
                        try:
                            storage_index = path_parts.index('storage')
                            relative_overlay_path_for_db = os.path.join(*path_parts[storage_index+1:]).replace('\\', '/')
                            logger.info(f"Fallback göreceli yol oluşturuldu: {relative_overlay_path_for_db}")
                        except ValueError:
                            logger.error(f"'storage' fallback göreceli yol için path içinde bulunamadı: {final_overlay_path_on_disk}")
                            relative_overlay_path_for_db = os.path.join('processed', f"frames_{analysis.id}", 'overlays', overlay_filename).replace('\\', '/') # Son çare
                            logger.warning(f"Son çare göreceli yol kullanıldı: {relative_overlay_path_for_db}")

                    if best_est:
                        best_est.processed_image_path = relative_overlay_path_for_db
                        logger.info(f"Kişi {person_id_str} için AgeEstimation.processed_image_path güncellendi: {relative_overlay_path_for_db}")
                    
                except Exception as e:
                    logger.error(f"Kişi {person_id_str} için overlay oluşturma/kaydetme hatası: {str(e)} - Traceback: {traceback.format_exc()}")
                    continue
            
            try:
                db.session.commit()
                logger.info(f"Analiz #{analysis.id} için tüm AgeEstimation.processed_image_path güncellemeleri commit edildi.")
            except Exception as commit_err:
                logger.error(f"AgeEstimation.processed_image_path güncellemeleri commit edilirken hata: {str(commit_err)}")
                db.session.rollback()
        # END NEW OVERLAY GENERATION LOGIC
        
        # Genel skorları hesapla (içerik analizi için)
        try:
            # Tüm içerik tespitlerini veritabanından al
            detections = ContentDetection.query.filter_by(analysis_id=analysis.id).all()
            
            if not detections:
                logger.warning(f"ContentDetection kaydı bulunamadı: Analiz #{analysis.id}")
                db.session.commit()
                return
            
            logger.info(f"Calculate_overall_scores: Analiz #{analysis.id} için {len(detections)} ContentDetection kaydı bulundu")
            
            categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug', 'safe']
            category_scores_sum = {cat: 0 for cat in categories}
            category_counts = {cat: 0 for cat in categories} # Her kategoride skoru olan kare sayısı
            
            category_specific_highest_risks = {
                cat: {'score': -1, 'frame_path': None, 'timestamp': None, 'detection_id': None} for cat in categories
            }

            for detection in detections:
                detection_scores = {
                    'violence': detection.violence_score,
                    'adult_content': detection.adult_content_score,
                    'harassment': detection.harassment_score,
                    'weapon': detection.weapon_score,
                    'drug': detection.drug_score,
                    'safe': detection.safe_score
                }
                
                for category in categories:
                    score = detection_scores.get(category)
                    if score is not None:
                        category_scores_sum[category] += score
                        category_counts[category] += 1
                        
                        if score > category_specific_highest_risks[category]['score']:
                            category_specific_highest_risks[category]['score'] = score
                            category_specific_highest_risks[category]['frame_path'] = detection.frame_path
                            category_specific_highest_risks[category]['timestamp'] = detection.frame_timestamp
                            category_specific_highest_risks[category]['detection_id'] = detection.id

            # Genel skorları basit aritmetik ortalama alarak hesapla
            avg_scores = {}
            avg_scores['violence'] = category_scores_sum['violence'] / category_counts['violence'] if category_counts['violence'] > 0 else 0
            avg_scores['adult_content'] = category_scores_sum['adult_content'] / category_counts['adult_content'] if category_counts['adult_content'] > 0 else 0
            avg_scores['harassment'] = category_scores_sum['harassment'] / category_counts['harassment'] if category_counts['harassment'] > 0 else 0
            avg_scores['weapon'] = category_scores_sum['weapon'] / category_counts['weapon'] if category_counts['weapon'] > 0 else 0
            avg_scores['drug'] = category_scores_sum['drug'] / category_counts['drug'] if category_counts['drug'] > 0 else 0
            # avg_scores['safe'] = category_scores_sum['safe'] / category_counts['safe'] if category_counts['safe'] > 0 else 0 # Eski safe hesaplaması
            
            logger.info(f"Analiz #{analysis.id} - Ham Ortalama Skorlar (safe hariç): {json.dumps({k: f'{v:.4f}' for k, v in avg_scores.items() if k != 'safe'})}")

            # --- YENİ: Güç Dönüşümü ile Skorları Ayrıştırma ---
            power_value = 1.5  # Bu değer ayarlanabilir (örneğin 1.5, 2, 2.5). Değer arttıkça ayrışma artar.
            
            enhanced_scores = {}
            risk_categories_for_safe_calc = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']

            for category in risk_categories_for_safe_calc: # Sadece risk kategorileri için güç dönüşümü
                avg_score_cat = avg_scores.get(category, 0) # .get() ile güvenli erişim
                enhanced_scores[category] = avg_score_cat ** power_value
            
            # Şimdi "safe" skorunu diğerlerinin geliştirilmiş ortalamasından türet
            sum_of_enhanced_risk_scores = sum(enhanced_scores.get(rc, 0) for rc in risk_categories_for_safe_calc)
            average_enhanced_risk_score = sum_of_enhanced_risk_scores / len(risk_categories_for_safe_calc) if risk_categories_for_safe_calc else 0
            enhanced_scores['safe'] = max(0.0, 1.0 - average_enhanced_risk_score) # Skorun negatif olmamasını sağla
            
            logger.info(f"Analiz #{analysis.id} - Güç Dönüşümü Sonrası Skorlar (p={power_value}): {json.dumps({k: f'{v:.4f}' for k, v in enhanced_scores.items()})}")
            logger.info(f"[SAFE_OVERALL_CALC] Average ENHANCED risk for overall: {average_enhanced_risk_score:.4f}, Calculated overall safe score: {enhanced_scores['safe']:.4f}")

            # Genel skorları güncelle (geliştirilmiş skorlarla)
            analysis.overall_violence_score = enhanced_scores['violence']
            analysis.overall_adult_content_score = enhanced_scores['adult_content']
            analysis.overall_harassment_score = enhanced_scores['harassment']
            analysis.overall_weapon_score = enhanced_scores['weapon']
            analysis.overall_drug_score = enhanced_scores['drug']
            analysis.overall_safe_score = enhanced_scores['safe']
            
            logger.info(f"Analiz #{analysis.id} - Geliştirilmiş Ortalama Skorlar: Violence={analysis.overall_violence_score:.4f}, Adult={analysis.overall_adult_content_score:.4f}, Harassment={analysis.overall_harassment_score:.4f}, Weapon={analysis.overall_weapon_score:.4f}, Drug={analysis.overall_drug_score:.4f}, Safe={analysis.overall_safe_score:.4f}")

            # Kategori bazlı en yüksek risk bilgilerini JSON olarak kaydetmek için (Analysis modelinde alan olmalı)
            # Şimdilik loglayalım ve dinamik attribute olarak ekleyelim. DB'ye yazmak için model değişikliği gerekebilir.
            analysis.category_specific_highest_risks_data = json.dumps(category_specific_highest_risks, cls=NumPyJSONEncoder) # NumPyJSONEncoder eklendi
            logger.info(f"Analiz #{analysis.id} - Kategori Bazlı En Yüksek Riskler: {analysis.category_specific_highest_risks_data}")

            # Mevcut en yüksek risk alanlarını (safe hariç genel en yüksek) yine de dolduralım, ama bu yeni mantığa göre olacak.
            # Tüm kategoriler (safe hariç) arasında en yüksek olanı bulalım.
            overall_highest_risk_score = -1
            overall_highest_risk_category = None
            overall_highest_risk_frame_path = None
            overall_highest_risk_timestamp = None

            for cat in categories:
                if cat == 'safe': # 'safe' kategorisini genel en yüksek risk için dahil etme
                    continue
                if category_specific_highest_risks[cat]['score'] > overall_highest_risk_score:
                    overall_highest_risk_score = category_specific_highest_risks[cat]['score']
                    overall_highest_risk_category = cat
                    overall_highest_risk_frame_path = category_specific_highest_risks[cat]['frame_path']
                    overall_highest_risk_timestamp = category_specific_highest_risks[cat]['timestamp']
            
            if overall_highest_risk_category:
                analysis.highest_risk_frame = overall_highest_risk_frame_path
                analysis.highest_risk_frame_timestamp = overall_highest_risk_timestamp
                analysis.highest_risk_score = overall_highest_risk_score
                analysis.highest_risk_category = overall_highest_risk_category
                logger.info(f"Analiz #{analysis.id} - Genel En Yüksek Risk ('safe' hariç): {overall_highest_risk_category} skoru {overall_highest_risk_score:.4f}, kare: {overall_highest_risk_frame_path}")
            else:
                # Eğer safe dışında hiçbir kategoride risk bulunamazsa (çok nadir olmalı)
                analysis.highest_risk_score = category_specific_highest_risks['safe']['score']
                analysis.highest_risk_category = 'safe'
                analysis.highest_risk_frame = category_specific_highest_risks['safe']['frame_path']
                analysis.highest_risk_frame_timestamp = category_specific_highest_risks['safe']['timestamp']
                logger.info(f"Analiz #{analysis.id} - 'safe' dışında risk bulunamadı. En yüksek 'safe' skoru: {analysis.highest_risk_score:.4f}")

            db.session.commit()
            
        except Exception as e:
            current_app.logger.error(f"Genel skor hesaplama hatası: {str(e)}")
            logger.error(f"Hata detayı: {traceback.format_exc()}")
            db.session.rollback()

        logger.info(f"Video analizi başarıyla tamamlandı: Analiz #{analysis.id}")
        return True, "Video analizi başarıyla tamamlandı"

    except Exception as e: # analyze_video için ana try bloğunun (satır 809'daki) except kısmı
        error_message = f"Video analizi sırasında genel hata: Analiz #{analysis.id}, Hata: {str(e)}"
        logger.error(error_message, exc_info=True)
        logger.error(traceback.format_exc())
        db.session.rollback() 
        return False, f"Video analizi hatası: {str(e)}"


def calculate_overall_scores(analysis):
    """
    Bir analiz için genel skorları hesaplar.
    Her kategori için tüm karelerdeki skorların basit aritmetik ortalamasını alır
    ve her kategori için en yüksek risk içeren kareyi belirler.
    
    Args:
        analysis: Skorları hesaplanacak analiz nesnesi
    """
    try:
        # Tüm içerik tespitlerini veritabanından al
        detections = ContentDetection.query.filter_by(analysis_id=analysis.id).all()
        
        if not detections:
            logger.warning(f"ContentDetection kaydı bulunamadı: Analiz #{analysis.id}")
            db.session.commit()
            return
        
        logger.info(f"Calculate_overall_scores: Analiz #{analysis.id} için {len(detections)} ContentDetection kaydı bulundu")
        
        categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug', 'safe']
        category_scores_sum = {cat: 0 for cat in categories}
        category_counts = {cat: 0 for cat in categories} # Her kategoride skoru olan kare sayısı
        
        category_specific_highest_risks = {
            cat: {'score': -1, 'frame_path': None, 'timestamp': None, 'detection_id': None} for cat in categories
        }

        for detection in detections:
            detection_scores = {
                'violence': detection.violence_score,
                'adult_content': detection.adult_content_score,
                'harassment': detection.harassment_score,
                'weapon': detection.weapon_score,
                'drug': detection.drug_score,
                'safe': detection.safe_score
            }
            
            for category in categories:
                score = detection_scores.get(category)
                if score is not None:
                    category_scores_sum[category] += score
                    category_counts[category] += 1
                    
                    if score > category_specific_highest_risks[category]['score']:
                        category_specific_highest_risks[category]['score'] = score
                        category_specific_highest_risks[category]['frame_path'] = detection.frame_path
                        category_specific_highest_risks[category]['timestamp'] = detection.frame_timestamp
                        category_specific_highest_risks[category]['detection_id'] = detection.id

        # Genel skorları basit aritmetik ortalama alarak hesapla
        avg_scores = {}
        avg_scores['violence'] = category_scores_sum['violence'] / category_counts['violence'] if category_counts['violence'] > 0 else 0
        avg_scores['adult_content'] = category_scores_sum['adult_content'] / category_counts['adult_content'] if category_counts['adult_content'] > 0 else 0
        avg_scores['harassment'] = category_scores_sum['harassment'] / category_counts['harassment'] if category_counts['harassment'] > 0 else 0
        avg_scores['weapon'] = category_scores_sum['weapon'] / category_counts['weapon'] if category_counts['weapon'] > 0 else 0
        avg_scores['drug'] = category_scores_sum['drug'] / category_counts['drug'] if category_counts['drug'] > 0 else 0
        # avg_scores['safe'] = category_scores_sum['safe'] / category_counts['safe'] if category_counts['safe'] > 0 else 0 # Eski safe hesaplaması
            
        logger.info(f"Analiz #{analysis.id} - Ham Ortalama Skorlar (safe hariç): {json.dumps({k: f'{v:.4f}' for k, v in avg_scores.items() if k != 'safe'})}")

        # --- YENİ: Güç Dönüşümü ile Skorları Ayrıştırma ---
        power_value = 1.5  # Bu değer ayarlanabilir (örneğin 1.5, 2, 2.5). Değer arttıkça ayrışma artar.
            
        enhanced_scores = {}
        risk_categories_for_safe_calc = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']

        for category in risk_categories_for_safe_calc: # Sadece risk kategorileri için güç dönüşümü
            avg_score_cat = avg_scores.get(category, 0) # .get() ile güvenli erişim
            enhanced_scores[category] = avg_score_cat ** power_value
        
        # Şimdi "safe" skorunu diğerlerinin geliştirilmiş ortalamasından türet
        sum_of_enhanced_risk_scores = sum(enhanced_scores.get(rc, 0) for rc in risk_categories_for_safe_calc)
        average_enhanced_risk_score = sum_of_enhanced_risk_scores / len(risk_categories_for_safe_calc) if risk_categories_for_safe_calc else 0
        enhanced_scores['safe'] = max(0.0, 1.0 - average_enhanced_risk_score) # Skorun negatif olmamasını sağla
            
        logger.info(f"Analiz #{analysis.id} - Güç Dönüşümü Sonrası Skorlar (p={power_value}): {json.dumps({k: f'{v:.4f}' for k, v in enhanced_scores.items()})}")
        logger.info(f"[SAFE_OVERALL_CALC] Average ENHANCED risk for overall: {average_enhanced_risk_score:.4f}, Calculated overall safe score: {enhanced_scores['safe']:.4f}")

        # Genel skorları güncelle (geliştirilmiş skorlarla)
        analysis.overall_violence_score = enhanced_scores['violence']
        analysis.overall_adult_content_score = enhanced_scores['adult_content']
        analysis.overall_harassment_score = enhanced_scores['harassment']
        analysis.overall_weapon_score = enhanced_scores['weapon']
        analysis.overall_drug_score = enhanced_scores['drug']
        analysis.overall_safe_score = enhanced_scores['safe']
            
        logger.info(f"Analiz #{analysis.id} - Geliştirilmiş Ortalama Skorlar: Violence={analysis.overall_violence_score:.4f}, Adult={analysis.overall_adult_content_score:.4f}, Harassment={analysis.overall_harassment_score:.4f}, Weapon={analysis.overall_weapon_score:.4f}, Drug={analysis.overall_drug_score:.4f}, Safe={analysis.overall_safe_score:.4f}")

        # Kategori bazlı en yüksek risk bilgilerini JSON olarak kaydetmek için (Analysis modelinde alan olmalı)
        # Şimdilik loglayalım ve dinamik attribute olarak ekleyelim. DB'ye yazmak için model değişikliği gerekebilir.
        analysis.category_specific_highest_risks_data = json.dumps(category_specific_highest_risks, cls=NumPyJSONEncoder)
        logger.info(f"Analiz #{analysis.id} - Kategori Bazlı En Yüksek Riskler: {analysis.category_specific_highest_risks_data}")

        # Mevcut en yüksek risk alanlarını (safe hariç genel en yüksek) yine de dolduralım, ama bu yeni mantığa göre olacak.
        # Tüm kategoriler (safe hariç) arasında en yüksek olanı bulalım.
        overall_highest_risk_score = -1
        overall_highest_risk_category = None
        overall_highest_risk_frame_path = None
        overall_highest_risk_timestamp = None

        for cat in categories:
            if cat == 'safe': 
                continue
            if category_specific_highest_risks[cat]['score'] > overall_highest_risk_score:
                overall_highest_risk_score = category_specific_highest_risks[cat]['score']
                overall_highest_risk_category = cat
                overall_highest_risk_frame_path = category_specific_highest_risks[cat]['frame_path']
                overall_highest_risk_timestamp = category_specific_highest_risks[cat]['timestamp']
        
        if overall_highest_risk_category:
            analysis.highest_risk_frame = overall_highest_risk_frame_path
            analysis.highest_risk_frame_timestamp = overall_highest_risk_timestamp
            analysis.highest_risk_score = overall_highest_risk_score
            analysis.highest_risk_category = overall_highest_risk_category
            logger.info(f"Analiz #{analysis.id} - Genel En Yüksek Risk ('safe' hariç): {overall_highest_risk_category} skoru {overall_highest_risk_score:.4f}, kare: {overall_highest_risk_frame_path}")
        else:
            analysis.highest_risk_score = category_specific_highest_risks['safe']['score']
            analysis.highest_risk_category = 'safe'
            analysis.highest_risk_frame = category_specific_highest_risks['safe']['frame_path']
            analysis.highest_risk_frame_timestamp = category_specific_highest_risks['safe']['timestamp']
            logger.info(f"Analiz #{analysis.id} - 'safe' dışında risk bulunamadı. En yüksek 'safe' skoru: {analysis.highest_risk_score:.4f}")

        db.session.commit()
        
    except Exception as e:
        current_app.logger.error(f"Genel skor hesaplama hatası: {str(e)}")
        logger.error(f"Hata detayı: {traceback.format_exc()}")
        db.session.rollback()

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
    logger.info(f"[SVC_LOG][ENTRY] get_analysis_results fonksiyonu çağrıldı. analysis_id: {analysis_id}") # YENİ GİRİŞ LOGU
    analysis = Analysis.query.get(analysis_id)

    if not analysis:
        return {'error': 'Analiz bulunamadı'}
    
    if analysis.status != 'completed':
        return {
            'status': analysis.status,
            'message': 'Analiz henüz tamamlanmadı - WebSocket üzerinden progress takip edin'
        }
    
    result = analysis.to_dict()
    
    content_detections = ContentDetection.query.filter_by(analysis_id=analysis_id).all()
    result['content_detections'] = [cd.to_dict() for cd in content_detections]
    
    if analysis.include_age_analysis:
        age_estimations = AgeEstimation.query.filter_by(analysis_id=analysis_id).all()
        logger.info(f"[SVC_LOG][RESULTS] get_analysis_results: DB'den {len(age_estimations)} AgeEstimation kaydı çekildi.")
        persons = {}
        for estimation in age_estimations:
            person_id = estimation.person_id
            if person_id not in persons:
                persons[person_id] = []
            persons[person_id].append(estimation.to_dict())
        
        logger.info(f"[SVC_LOG][RESULTS] get_analysis_results: {len(persons)} kişiye göre gruplandı.")
        best_estimations = []
        for person_id, estimations in persons.items():
            if not estimations:  # Boş liste kontrolü
                logger.warning(f"[SVC_LOG][RESULTS] Kişi {person_id} için tahmin listesi boş, atlanıyor.")
                continue
                
            best_estimation = max(estimations, key=lambda e: e.get('confidence_score', 0) if e else 0)
            if not best_estimation:  # None kontrolü
                logger.warning(f"[SVC_LOG][RESULTS] Kişi {person_id} için en iyi tahmin None, atlanıyor.")
                continue
                
            logger.info(f"[SVC_LOG][RESULTS] get_analysis_results: Kişi {person_id} için en iyi tahmin seçildi (Güven: {best_estimation.get('confidence_score', 0):.4f}).")
            logger.info(f"DEBUG - Frontend'e gönderilecek yaş: person_id={person_id}, estimated_age={best_estimation.get('estimated_age', 'N/A')}, all_estimations_for_person={[(e.get('estimated_age', 'N/A') if e else 'None', e.get('confidence_score', 0) if e else 0) for e in estimations]}")
            logger.info(f"DEBUG - best_estimation tüm alanları: {best_estimation}")
            best_estimations.append(best_estimation)
        result['age_estimations'] = best_estimations
        logger.info(f"[SVC_LOG][RESULTS] get_analysis_results: API yanıtına {len(best_estimations)} en iyi tahmin eklendi.")

    # ---- YENİ LOGLAR ----
    logger.info(f"[SVC_LOG][DEBUG] get_analysis_results - json.dumps öncesi.")
    if 'category_specific_highest_risks_data' in result:
        logger.info(f"[SVC_LOG][DEBUG] result['category_specific_highest_risks_data'] var. Türü: {type(result['category_specific_highest_risks_data'])}")
        logger.info(f"[SVC_LOG][DEBUG] result['category_specific_highest_risks_data'] içeriği: {result['category_specific_highest_risks_data']}")
    else:
        logger.info(f"[SVC_LOG][DEBUG] result['category_specific_highest_risks_data'] YOK.")
    # ---- YENİ LOGLAR SONU ----

    try:
        # Orijinal log satırını try-except içine alalım
        final_result_json = json.dumps(result, indent=2, cls=NumPyJSONEncoder)
        logger.info(f"[SVC_LOG][RESULTS] get_analysis_results sonu - Dönecek Result: {final_result_json}")
    except Exception as e_dumps:
        logger.error(f"[SVC_LOG][ERROR] get_analysis_results - json.dumps sırasında HATA: {str(e_dumps)}", exc_info=True)
        logger.error(f"[SVC_LOG][ERROR] Hata anındaki result sözlüğü (ilk 1000 karakter): {str(result)[:1000]}") # Hata anındaki result'ı logla (çok uzunsa kırp)
        # Hata durumunda da bir şeyler döndürmek gerekebilir, yoksa frontend askıda kalabilir.
        # Şimdilik orijinal davranışı koruyup, sadece logluyoruz.
        # Sorun buysa, buraya bir `return {'error': 'Sonuçlar serileştirilemedi'}` eklenebilir.
    return result

# Model yükleme için yardımcı fonksiyonlar
def get_content_analyzer():
    """İçerik analizi için ContentAnalyzer nesnesi döndürür"""
    return ContentAnalyzer()

# Deprecated get_age_estimator function removed - use app.utils.model_state.get_age_estimator() instead

# --- PATH NORMALİZASYON HELPER ---
def normalize_rel_storage_path(rel_path: str) -> str:
    """
    Göreli depolama yolunu normalize eder.
    Args:
        rel_path (str): Göreli yol.
    Returns:
        str: Normalize edilmiş yol.
    """
    rel_path = os.path.normpath(rel_path).replace("\\", "/")
    # Başındaki ../ veya ./ gibi ifadeleri tamamen temizle
    while rel_path.startswith("../") or rel_path.startswith("./"):
        rel_path = rel_path[3:] if rel_path.startswith("../") else rel_path[2:]
    # Sadece 'storage/...' ile başlayan kısmı al
    idx = rel_path.find("storage/")
    if idx != -1:
        rel_path = rel_path[idx:]
    return rel_path
