import logging
from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models.file import File
from app.models.analysis import Analysis, ContentDetection, AgeEstimation
from app.models.feedback import Feedback
from app.services.analysis_service import AnalysisService, get_analysis_results
from app.json_encoder import json_dumps_numpy
from app.utils.security import (
    validate_request_params, validate_json_input,
    SecurityError, sanitize_html_input
)

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/analysis')
"""
Analiz işlemleri için blueprint.
- Analiz başlatma, sonuç alma ve analizle ilgili endpointleri içerir.
"""

@analysis_bp.route('/start', methods=['POST'])
def start_analysis():
    """
    Güvenli analiz başlatma endpoint'i. Tüm girişleri doğrular.
    """
    try:
        # ERSIN JSON input validation
        if not request.is_json:
            return jsonify({'error': 'Content-Type application/json gereklidir'}), 400
        
        try:
            data = validate_json_input(request.json)
        except SecurityError as e:
            return jsonify({'error': f'JSON doğrulama hatası: {str(e)}'}), 400
        
        # ERSIN Parametre doğrulama
        try:
            params = validate_request_params(
                data,
                {
                    'file_id': {
                        'type': 'int',
                        'min': 1,
                        'required': True
                    },
                    'frames_per_second': {
                        'type': 'float',
                        'min': 0.1,
                        'max': 60.0,
                        'required': False
                    },
                    'include_age_analysis': {
                        'type': 'bool',
                        'default': False
                    }
                }
            )
        except SecurityError as e:
            return jsonify({'error': f'Parameter doğrulama hatası: {str(e)}'}), 400
        
        file_id = params['file_id']
        
        # ERSIN Dosyanın varlığını kontrol et
        file = File.query.get(file_id)
        if not file:
            return jsonify({'error': 'Dosya bulunamadı'}), 404
            
        # ERSIN Analiz parametrelerini güvenli şekilde al
        frames_per_second = params.get('frames_per_second')
        include_age_analysis = params.get('include_age_analysis', False)
        websocket_session_id = request.headers.get('X-WebSocket-Session-ID')  # Header'dan WebSocket session ID'yi al
        
        # ERSIN AnalysisService ile analizi başlat
        analysis_service = AnalysisService()
        analysis = analysis_service.start_analysis(file_id, frames_per_second, include_age_analysis, websocket_session_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz başlatılamadı'}), 500
        
        return jsonify({
            'message': 'Analiz başarıyla başlatıldı',
            'analysis': analysis.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Analiz başlatılırken hata: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Analiz başlatılırken bir hata oluştu'}), 500

@analysis_bp.route('/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """
    Güvenli analiz bilgisi getirme endpoint'i.
    """
    try:
        # ERSIN ID validation (negatif veya sıfır sayı geçersiz; UUID/string kabul)
        if isinstance(analysis_id, str) and analysis_id.isdigit() and int(analysis_id) <= 0:
            return jsonify({'error': 'Geçersiz analiz ID'}), 400
        
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
            
        return jsonify(analysis.to_dict()), 200
        
    except Exception as e:
        logger.error(f"Analiz bilgisi alınırken hata: {str(e)}")
        return jsonify({'error': 'Analiz bilgisi alınırken bir hata oluştu'}), 500

@analysis_bp.route('/file/<int:file_id>', methods=['GET'])
def get_file_analyses(file_id):
    """
    Güvenli dosya analizleri getirme endpoint'i.
    """
    try:
        # ERSIN ID validation
        if file_id <= 0:
            return jsonify({'error': 'Geçersiz dosya ID'}), 400
        
        # ERSIN Dosyanın varlığını kontrol et
        file = File.query.get(file_id)
        if not file:
            return jsonify({'error': 'Dosya bulunamadı'}), 404
        
        # ERSIN Dosyaya ait tüm analizleri bul
        analyses = Analysis.query.filter_by(file_id=file_id).all()
        
        return jsonify([a.to_dict() for a in analyses]), 200
        
    except Exception as e:
        logger.error(f"Dosya analizleri alınırken hata: {str(e)}")
        return jsonify({'error': 'Dosya analizleri alınırken bir hata oluştu'}), 500

@analysis_bp.route('/<analysis_id>/results', methods=['GET'])
def get_results(analysis_id):
    """
    Güvenli analiz sonuçları getirme endpoint'i.
    """
    try:
        # ERSIN ID validation (negatif/sıfır ID geçersiz; UUID/string kabul)
        if isinstance(analysis_id, str) and analysis_id.isdigit() and int(analysis_id) <= 0:
            return jsonify({'error': 'Geçersiz analiz ID'}), 400
        
        results = get_analysis_results(analysis_id)
        
        if 'error' in results:
            return jsonify(results), 404
            
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Analiz sonuçları alınırken hata: {str(e)}")
        return jsonify({'error': 'Analiz sonuçları alınırken bir hata oluştu'}), 500

@analysis_bp.route('/<analysis_id>/feedback', methods=['POST'])
def submit_feedback(analysis_id):
    """
    Güvenli feedback gönderme endpoint'i.
    """
    try:
        # ID validation (numeric negatives or zero are invalid; UUID/string ids allowed)
        if isinstance(analysis_id, str) and analysis_id.isdigit() and int(analysis_id) <= 0:
            return jsonify({'error': 'Geçersiz analiz ID'}), 400
        
        # ERSIN Analizin varlığını kontrol et
        analysis = Analysis.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
        
        # ERSIN JSON input validation
        if not request.is_json:
            return jsonify({'error': 'Content-Type application/json gereklidir'}), 400
        
        try:
            data = validate_json_input(request.json)
        except SecurityError as e:
            return jsonify({'error': f'JSON doğrulama hatası: {str(e)}'}), 400
        
        # ERSIN Parametre doğrulama
        try:
            params = validate_request_params(
                data,
                {
                    'rating': {
                        'type': 'int',
                        'min': 1,
                        'max': 5,
                        'required': True
                    },
                    'comment': {
                        'type': 'str',
                        'max_length': 1000,
                        'required': False,
                        'default': ''
                    },
                    'false_positives': {
                        'required': False
                    },
                    'false_negatives': {
                        'required': False
                    }
                }
            )
        except SecurityError as e:
            return jsonify({'error': f'Parameter doğrulama hatası: {str(e)}'}), 400
        
        # ERSIN Yorum alanını sanitize et
        comment = sanitize_html_input(params.get('comment', ''))
        
        # ERSIN Feedback oluştur
        feedback = Feedback(
            analysis_id=analysis_id,
            rating=params['rating'],
            comment=comment,
            false_positives=params.get('false_positives', []),
            false_negatives=params.get('false_negatives', [])
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        return jsonify({
            'message': 'Geribildirim başarıyla kaydedildi',
            'feedback': feedback.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Feedback kaydedilirken hata: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Geribildirim kaydedilirken bir hata oluştu'}), 500

@analysis_bp.route('/<analysis_id>/feedback', methods=['GET'])
def get_feedback(analysis_id):
    """
    Belirtilen analiz ID'si için tüm geribildirimleri getirir.
    
    Args:
        analysis_id: Analiz ID'si
        
    Returns:
        JSON: Geribildirim listesi veya hata mesajı
    """
    # ERSIN Analizin varlığını kontrol et
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # ERSIN Analiz için tüm geribildirimleri bul
    feedbacks = Feedback.query.filter_by(analysis_id=analysis_id).all()
    
    return jsonify([f.to_dict() for f in feedbacks]), 200

@analysis_bp.route('/<analysis_id>/cancel', methods=['POST'])
def cancel_analysis(analysis_id):
    """
    Devam eden bir analizi iptal eder.
    
    Args:
        analysis_id: İptal edilecek analiz ID'si
        
    Returns:
        JSON: Başarı mesajı veya hata mesajı
    """
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
            
        # ERSIN Analiz zaten tamamlanmış veya iptal edilmişse
        if analysis.status in ['completed', 'failed', 'cancelled']:
            return jsonify({'error': f'Bu analiz zaten {analysis.status} durumunda'}), 400
            
        # ERSIN Analiz servisini çağırarak iptal et
        analysis_service = AnalysisService()
        success = analysis_service.cancel_analysis(analysis_id)
        
        if not success:
            return jsonify({'error': 'Analiz iptal edilemedi'}), 500
            
        return jsonify({'message': 'Analiz başarıyla iptal edildi'}), 200
        
    except Exception as e:
        logger.error(f"Analiz iptal edilirken hata: {str(e)}")
        return jsonify({'error': f'Analiz iptal edilirken bir hata oluştu: {str(e)}'}), 500

@analysis_bp.route('/<analysis_id>/retry', methods=['POST'])
def retry_analysis(analysis_id):
    """
    Başarısız olan bir analizi tekrar dener.
    
    Args:
        analysis_id: Tekrar denenecek analiz ID'si
        
    Returns:
        JSON: Başarı mesajı veya hata mesajı
    """
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
            
        # ERSIN Analiz başarısız değilse tekrar denemek anlamlı değil
        if analysis.status != 'failed':
            return jsonify({'error': f'Sadece başarısız analizler tekrar denenebilir. Mevcut durum: {analysis.status}'}), 400
            
        # ERSIN Analiz servisini çağırarak tekrar başlat
        analysis_service = AnalysisService()
        new_analysis = analysis_service.retry_analysis(analysis_id)
        
        if not new_analysis:
            return jsonify({'error': 'Analiz tekrar başlatılamadı'}), 500
            
        return jsonify({
            'message': 'Analiz tekrar başlatıldı',
            'analysis_id': new_analysis.id,
            'status': new_analysis.status
        }), 200
        
    except Exception as e:
        logger.error(f"Analiz tekrar denenirken hata: {str(e)}")
        return jsonify({'error': f'Analiz tekrar denenirken bir hata oluştu: {str(e)}'}), 500

@analysis_bp.route('/<analysis_id>/status', methods=['GET'])
def get_analysis_status(analysis_id):
    """
    Analiz durumunu ve ilerleme bilgisini getirir.
    
    Args:
        analysis_id: Durumu kontrol edilecek analiz ID'si
        
    Returns:
        JSON: Analiz durum bilgileri veya hata mesajı
    """
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
            
        status_info = {
            'analysis_id': analysis.id,
            'status': analysis.status,
            'start_time': analysis.start_time.isoformat() if analysis.start_time else None,
            'end_time': analysis.end_time.isoformat() if analysis.end_time else None,
            'note': 'Progress ve durum bilgileri WebSocket üzerinden gerçek zamanlı gönderilir'
        }
        
        return jsonify(status_info), 200
        
    except Exception as e:
        logger.error(f"Analiz durumu alınırken hata: {str(e)}")
        return jsonify({'error': f'Analiz durumu alınırken bir hata oluştu: {str(e)}'}), 500

@analysis_bp.route('/<analysis_id>/detailed-results', methods=['GET'])
def get_detailed_results(analysis_id):
    """
    Tamamlanmış bir analizin detaylı sonuçlarını getirir.
    Kare-kare içerik tespitleri, yaş tahminleri vb. detaylı bilgileri içerir.
    
    Args:
        analysis_id: Sonuçları getirilecek analiz ID'si
        
    Returns:
        JSON: Analiz sonuçları veya hata mesajı
    """
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
            
        # ERSIN Analiz henüz tamamlanmamışsa, processing durumundaysa partial sonuçlar dön
        if analysis.status not in ['completed', 'processing']:
            return jsonify({'error': f'Analiz henüz başlamadı veya başarısız oldu. Mevcut durum: {analysis.status}'}), 400
            
        # ERSIN Analiz sonuçlarını getir
        content_detections = [cd.to_dict() for cd in analysis.content_detections]
        
        # ERSIN Age estimations'ı alıp processed_image_path'leri düzelt
        age_estimations = []
        if analysis.include_age_analysis:
            for ae in analysis.age_estimations:
                ae_dict = ae.to_dict()
                # ERSIN Eğer processed_image_path None ise, frame_path'den türet
                if not ae_dict.get('processed_image_path') and ae_dict.get('frame_path'):
                    frame_path = ae_dict['frame_path'].replace('\\', '/')
                    if '/storage/' in frame_path:
                        ae_dict['processed_image_path'] = frame_path.split('/storage/')[-1]
                    else:
                        ae_dict['processed_image_path'] = frame_path
                age_estimations.append(ae_dict)
        
        # ERSIN En yüksek riskli kare yolunu uygun formata getir
        highest_risk_frame = analysis.highest_risk_frame
        if highest_risk_frame:
            # ERSIN Dosya adını düzgün biçimde çıkar - frame_XXX.jpg formatındaki dosya adını al
            frame_filename = highest_risk_frame.split('/')[-1].split('\\')[-1]
            
            # ERSIN \"frame_\" ön ekini tekrar eklemeyelim
            if frame_filename.startswith("frame_"):
                frame_url = frame_filename
            else:
                frame_url = f"frame_{frame_filename}"
                
            # ERSIN Processed image path'i de set et (highest_risk için)
            processed_image_path = highest_risk_frame.replace('\\', '/')
            if '/storage/' in processed_image_path:
                path_after_storage = processed_image_path.split('/storage/')[-1]
                # Resim analizi için uploads klasörü kontrolü
                if path_after_storage.startswith('uploads/'):
                    # Resim dosyası için uploads klasöründen serve et
                    processed_image_path = path_after_storage  # uploads/dosya.jpg
                else:
                    # Video kareleri için processed klasöründen serve et  
                    processed_image_path = path_after_storage  # processed/frames_xxx/...
            else:
                processed_image_path = highest_risk_frame
        else:
            frame_url = None
            processed_image_path = None
        
        results = {
            'analysis_id': analysis.id,
            'file_id': analysis.file_id,
            'file_name': analysis.file.original_filename if analysis.file else None,
            'file_type': analysis.file.file_type if analysis.file else None,
            'file_path': analysis.file.file_path if analysis.file else None,
            'file_filename': analysis.file.filename if analysis.file else None,  # Orijinal video için
            'overall_scores': {
                'violence': analysis.overall_violence_score,
                'adult_content': analysis.overall_adult_content_score,
                'harassment': analysis.overall_harassment_score,
                'weapon': analysis.overall_weapon_score,
                'drug': analysis.overall_drug_score,
                'safe': analysis.overall_safe_score
            },
            'highest_risk': {
                'frame': frame_url,
                'frame_dir': f"frames_{analysis.id}",
                'analysis_id': analysis.id,
                'timestamp': analysis.highest_risk_frame_timestamp,
                'score': analysis.highest_risk_score,
                'category': analysis.highest_risk_category,
                'processed_image_path': processed_image_path
            },
            'highest_risk_frame_details': {
                'frame_path': analysis.highest_risk_frame
            } if analysis.highest_risk_frame else None,
            'content_detections': content_detections,
            'age_estimations': age_estimations,
            'category_specific_highest_risks_data': analysis.category_specific_highest_risks_data
        }
        
        # ERSIN --- LOG EKLEME ---
        # ERSIN content_detections içindeki frame_path ve processed_image_path logla
        for idx, det in enumerate(content_detections):
            logger.info(f"[BACKEND][content_detections][{idx}] frame_path: {det.get('frame_path')}, processed_image_path: {det.get('processed_image_path')}")
        # ERSIN age_estimations içindeki processed_image_path logla
        for idx, age in enumerate(age_estimations):
            logger.info(f"[BACKEND][age_estimations][{idx}] processed_image_path: {age.get('processed_image_path')}")
        # ERSIN highest_risk frame logla
        logger.info(f"[BACKEND][highest_risk] frame: {results['highest_risk'].get('frame')}")
        logger.info(f"[BACKEND][highest_risk] processed_image_path: {results['highest_risk'].get('processed_image_path')}")
        logger.info(f"[BACKEND][highest_risk] RAW analysis.highest_risk_frame: {analysis.highest_risk_frame}")
        # ERSIN --- LOG EKLEME SONU ---

        # ERSIN NumPy veri tipleri için özel encoder kullanmadan doğrudan dict döndür
        # ERSIN Flask 3 ve json modülü ile dict döndürmek yeterli
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Detaylı analiz sonuçları alınırken hata: {str(e)}")
        return jsonify({'error': f'Detaylı analiz sonuçları alınırken bir hata oluştu: {str(e)}'}), 500 


@analysis_bp.route('/pending-feedback', methods=['GET'])
def get_pending_feedback_analyses():
    """
    Feedback verilmemiş analiz sonuçlarını listeler.
    Otomatik yönlendirme için kullanılır.
    """
    try:
        # ERSIN Tamamlanmış analizleri al
        completed_analyses = Analysis.query.filter_by(status='completed').all()
        
        pending_analyses = []
        for analysis in completed_analyses:
            # ERSIN Bu analiz için feedback verilmiş mi?
            has_feedback = Feedback.query.filter_by(analysis_id=analysis.id).first() is not None
            
            if not has_feedback:
                pending_analyses.append({
                    'analysis_id': analysis.id,
                    'file_id': analysis.file_id,
                    'filename': analysis.file.filename if analysis.file else 'Unknown',
                    'created_at': analysis.created_at.isoformat() if analysis.created_at else None
                })
        
        return jsonify({
            'success': True,
            'pending_analyses': pending_analyses,
            'count': len(pending_analyses)
        }), 200
        
    except Exception as e:
        logger.error(f"Bekleyen feedback analizleri alınırken hata: {str(e)}")
        return jsonify({'error': f'Bekleyen analizler alınırken bir hata oluştu: {str(e)}'}), 500


@analysis_bp.route('/recent', methods=['GET'])
def get_recent_analyses():
    """
    🔄 Son tamamlanan analizleri getirir (page refresh sonrası restore için)
    """
    try:
        # ERSIN Son 10 completed analizi al (en yeni üstte)
        recent_analyses = Analysis.query.filter_by(
            status='completed'
        ).order_by(Analysis.end_time.desc()).limit(10).all()
        
        result = []
        for analysis in recent_analyses:
            # ERSIN Basic analiz bilgileri
            analysis_data = {
                'analysis_id': analysis.id,
                'file_id': analysis.file_id,
                'file_name': analysis.file.original_filename if analysis.file else 'Unknown',
                'status': analysis.status,
                'include_age_analysis': analysis.include_age_analysis,
                'completed_at': analysis.end_time.isoformat() if analysis.end_time else None,
                'overall_scores': {}
            }
            
            # ERSIN Overall scores'ları field'lardan oluştur
            analysis_data['overall_scores'] = {
                'violence': analysis.overall_violence_score or 0,
                'adult_content': analysis.overall_adult_content_score or 0,
                'harassment': analysis.overall_harassment_score or 0,
                'weapon': analysis.overall_weapon_score or 0,
                'drug': analysis.overall_drug_score or 0,
                'safe': analysis.overall_safe_score or 0
            }
            
            result.append(analysis_data)
        
        logger.info(f"📊 Recent analyses: {len(result)} completed analiz döndürüldü")
        return jsonify({
            'success': True,
            'recent_analyses': result,
            'count': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Recent analyses alınırken hata: {str(e)}")
        return jsonify({'error': f'Recent analyses alınırken bir hata oluştu: {str(e)}'}), 500


@analysis_bp.route('/clear-all', methods=['DELETE'])
def clear_all_analyses():
    """
    Tüm analiz sonuçlarını temizler (veritabanından siler)
    """
    try:
        from app.models.analysis import Analysis
        
        # ERSIN Tüm analizleri sil
        deleted_count = Analysis.query.delete()
        db.session.commit()
        
        logger.info(f"🗑️ {deleted_count} analiz sonucu veritabanından temizlendi")
        
        return jsonify({
            'success': True,
            'message': f'{deleted_count} analiz sonucu başarıyla temizlendi',
            'deleted_count': deleted_count
        }), 200
        
    except Exception as e:
        logger.error(f"Analiz sonuçları temizlenirken hata: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Analiz sonuçları temizlenirken bir hata oluştu: {str(e)}'}), 500


bp = analysis_bp 