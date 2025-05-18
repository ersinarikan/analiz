import logging
from flask import Blueprint, request, jsonify, current_app, Response
from app import db, socketio
from app.models.file import File
from app.models.analysis import Analysis, ContentDetection, AgeEstimation, AnalysisFeedback
from app.models.feedback import Feedback
from app.services.analysis_service import AnalysisService, get_analysis_results
from app.json_encoder import json_dumps_numpy

logger = logging.getLogger(__name__)

bp = Blueprint('analysis', __name__, url_prefix='/api/analysis')

@bp.route('/start', methods=['POST'])
def start_analysis():
    """
    Yeni bir analiz başlatır. Belirtilen dosya ID'si için analiz işlemini kuyruğa ekler.
    
    Gerekli parametreler:
    - file_id: Analiz edilecek dosyanın ID'si
    - Opsiyonel parametreler: frames_per_second, include_age_analysis
    
    Returns:
        JSON: Oluşturulan analiz bilgileri veya hata mesajı
    """
    try:
        data = request.json
        
        # Gerekli alanları kontrol et
        if not data or 'file_id' not in data:
            return jsonify({'error': 'file_id alanı gereklidir'}), 400
            
        file_id = data['file_id']
        
        # Dosyanın varlığını kontrol et
        file = File.query.get(file_id)
        if not file:
            return jsonify({'error': f'ID: {file_id} ile dosya bulunamadı'}), 404
            
        # İsteğe bağlı parametreleri al
        frames_per_second = data.get('frames_per_second')
        include_age_analysis = data.get('include_age_analysis', False)
        
        # AnalysisService ile analizi başlat
        analysis_service = AnalysisService()
        analysis = analysis_service.start_analysis(file_id, frames_per_second, include_age_analysis)
        
        if not analysis:
            return jsonify({'error': 'Analiz başlatılamadı'}), 500
        
        # Analizi başarılı durum güncellemesi
        analysis.status = 'pending'
        analysis.status_message = 'Analiz başlatıldı'
        db.session.commit()
        
        # Oluşturulan analiz bilgilerini döndür
        return jsonify({
            'message': 'Analiz başarıyla başlatıldı',
            'analysis': analysis.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Analiz başlatılırken hata: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Analiz başlatılırken bir hata oluştu: {str(e)}'}), 500

@bp.route('/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """
    Belirtilen ID'ye sahip analizin bilgilerini getirir.
    
    Args:
        analysis_id: Analiz ID'si
        
    Returns:
        JSON: Analiz bilgileri veya hata mesajı
    """
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
            
        return jsonify(analysis.to_dict()), 200
        
    except Exception as e:
        logger.error(f"Analiz bilgisi alınırken hata: {str(e)}")
        return jsonify({'error': f'Analiz bilgisi alınırken bir hata oluştu: {str(e)}'}), 500


@bp.route('/file/<file_id>', methods=['GET'])
def get_file_analyses(file_id):
    """
    Belirtilen dosya ID'sine ait tüm analizleri getirir.
    
    Args:
        file_id: Dosya ID'si
        
    Returns:
        JSON: Analiz listesi veya hata mesajı
    """
    # Dosyanın varlığını kontrol et
    file = File.query.get_or_404(file_id)
    
    # Dosyaya ait tüm analizleri bul
    analyses = Analysis.query.filter_by(file_id=file_id).all()
    
    return jsonify([a.to_dict() for a in analyses]), 200


@bp.route('/<analysis_id>/results', methods=['GET'])
def get_results(analysis_id):
    print("--- GET_RESULTS ENDPOINT CALLED ---")
    """
    Belirtilen analiz ID'si için detaylı sonuçları getirir.
    
    Args:
        analysis_id: Analiz ID'si
        
    Returns:
        JSON: Analiz sonuçları veya hata mesajı
    """
    results = get_analysis_results(analysis_id)
    
    # YENİ LOGLAR
    logger.info(f"[ROUTE_LOG] get_results: get_analysis_results fonksiyonundan dönen results: {type(results)}")
    if isinstance(results, dict):
        logger.info(f"[ROUTE_LOG] results sözlüğünün anahtarları: {list(results.keys())}")
        # category_specific_highest_risks_data alanının varlığını ve türünü kontrol et
        if 'category_specific_highest_risks_data' in results:
            logger.info(f"[ROUTE_LOG] results['category_specific_highest_risks_data'] TÜRÜ: {type(results['category_specific_highest_risks_data'])}")
            logger.info(f"[ROUTE_LOG] results['category_specific_highest_risks_data'] İÇERİĞİ (ilk 200 karakter): {str(results['category_specific_highest_risks_data'])[:200]}")
        else:
            logger.info("[ROUTE_LOG] results içerisinde 'category_specific_highest_risks_data' anahtarı BULUNAMADI.")
    else:
        logger.warning(f"[ROUTE_LOG] get_analysis_results beklenen gibi sözlük dönmedi, dönen değer (ilk 200 karakter): {str(results)[:200]}")
    # YENİ LOGLAR SONU

    if 'error' in results:
        return jsonify(results), 404
        
    return jsonify(results), 200


@bp.route('/<analysis_id>/feedback', methods=['POST'])
def submit_feedback(analysis_id):
    """
    Bir analiz sonucu için kullanıcı geribildirimini kaydeder.
    
    Args:
        analysis_id: Geribildirim yapılacak analiz ID'si
        
    İstek gövdesinde:
        - rating: Puanlama (1-5 arası)
        - comment: Yorumlar/açıklamalar
        - false_positives: Yanlış pozitif tespitler (kategori listesi)
        - false_negatives: Yanlış negatif tespitler (kategori listesi)
    
    Returns:
        JSON: Başarı mesajı veya hata mesajı
    """
    # Analizin varlığını kontrol et
    analysis = Analysis.query.get_or_404(analysis_id)
    
    data = request.json
    
    # Veri doğrulama
    if not data or 'rating' not in data:
        return jsonify({'error': 'Puanlama (rating) alanı gereklidir'}), 400
        
    # Feedback oluştur
    feedback = Feedback(
        analysis_id=analysis_id,
        rating=data['rating'],
        comment=data.get('comment', ''),
        false_positives=data.get('false_positives', []),
        false_negatives=data.get('false_negatives', [])
    )
    
    db.session.add(feedback)
    db.session.commit()
    
    # Socket.io ile bildirimleri gönder (gerçek zamanlı)
    socketio.emit('new_feedback', feedback.to_dict())
    
    return jsonify({
        'message': 'Geribildirim başarıyla kaydedildi',
        'feedback': feedback.to_dict()
    }), 201


@bp.route('/<analysis_id>/feedback', methods=['GET'])
def get_feedback(analysis_id):
    """
    Belirtilen analiz ID'si için tüm geribildirimleri getirir.
    
    Args:
        analysis_id: Analiz ID'si
        
    Returns:
        JSON: Geribildirim listesi veya hata mesajı
    """
    # Analizin varlığını kontrol et
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Analiz için tüm geribildirimleri bul
    feedbacks = Feedback.query.filter_by(analysis_id=analysis_id).all()
    
    return jsonify([f.to_dict() for f in feedbacks]), 200


@bp.route('/<analysis_id>/cancel', methods=['POST'])
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
            
        # Analiz zaten tamamlanmış veya iptal edilmişse
        if analysis.status in ['completed', 'failed', 'cancelled']:
            return jsonify({'error': f'Bu analiz zaten {analysis.status} durumunda'}), 400
            
        # Analiz servisini çağırarak iptal et
        analysis_service = AnalysisService()
        success = analysis_service.cancel_analysis(analysis_id)
        
        if not success:
            return jsonify({'error': 'Analiz iptal edilemedi'}), 500
            
        return jsonify({'message': 'Analiz başarıyla iptal edildi'}), 200
        
    except Exception as e:
        logger.error(f"Analiz iptal edilirken hata: {str(e)}")
        return jsonify({'error': f'Analiz iptal edilirken bir hata oluştu: {str(e)}'}), 500


@bp.route('/<analysis_id>/retry', methods=['POST'])
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
            
        # Analiz başarısız değilse tekrar denemek anlamlı değil
        if analysis.status != 'failed':
            return jsonify({'error': f'Sadece başarısız analizler tekrar denenebilir. Mevcut durum: {analysis.status}'}), 400
            
        # Analiz servisini çağırarak tekrar başlat
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


@bp.route('/<analysis_id>/status', methods=['GET'])
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
            'progress': analysis.progress,
            'message': analysis.status_message,
            'start_time': analysis.start_time.isoformat() if analysis.start_time else None,
            'end_time': analysis.end_time.isoformat() if analysis.end_time else None
        }
        
        return jsonify(status_info), 200
        
    except Exception as e:
        logger.error(f"Analiz durumu alınırken hata: {str(e)}")
        return jsonify({'error': f'Analiz durumu alınırken bir hata oluştu: {str(e)}'}), 500


@bp.route('/<analysis_id>/detailed-results', methods=['GET'])
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
            
        # Analiz henüz tamamlanmamışsa
        if analysis.status != 'completed':
            return jsonify({'error': f'Analiz henüz tamamlanmadı. Mevcut durum: {analysis.status}'}), 400
            
        # Analiz sonuçlarını getir
        content_detections = [cd.to_dict() for cd in analysis.content_detections]
        age_estimations = [ae.to_dict() for ae in analysis.age_estimations] if analysis.include_age_analysis else []
        
        # En yüksek riskli kare yolunu uygun formata getir
        highest_risk_frame = analysis.highest_risk_frame
        if highest_risk_frame:
            # Dosya adını düzgün biçimde çıkar - frame_XXX.jpg formatındaki dosya adını al
            frame_filename = highest_risk_frame.split('/')[-1].split('\\')[-1]
            
            # "frame_" ön ekini tekrar eklemeyelim
            if frame_filename.startswith("frame_"):
                frame_url = frame_filename
            else:
                frame_url = f"frame_{frame_filename}"
        else:
            frame_url = None
        
        results = {
            'analysis_id': analysis.id,
            'file_id': analysis.file_id,
            'file_name': analysis.file.original_filename if analysis.file else None,
            'file_type': analysis.file.file_type if analysis.file else None,
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
                'category': analysis.highest_risk_category
            },
            'content_detections': content_detections,
            'age_estimations': age_estimations,
            'category_specific_highest_risks_data': analysis.category_specific_highest_risks_data
        }
        
        # NumPy veri tipleri ile başa çıkabilmek için özel JSON dönüştürücü kullan
        json_str = json_dumps_numpy(results)
        return Response(json_str, mimetype='application/json'), 200
        
    except Exception as e:
        logger.error(f"Detaylı analiz sonuçları alınırken hata: {str(e)}")
        return jsonify({'error': f'Detaylı analiz sonuçları alınırken bir hata oluştu: {str(e)}'}), 500 