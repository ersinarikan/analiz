import os
import uuid
import mimetypes
from flask import Blueprint, request, jsonify, current_app, send_from_directory, g, send_file
from werkzeug.utils import secure_filename
import logging

from app import db
from app.models.file import File
from app.services.file_service import is_allowed_file, save_uploaded_file
from app.services.analysis_service import AnalysisService
from app.utils.security import (
    validate_file_upload, validate_path, validate_request_params,
    FileSecurityError, PathSecurityError, SecurityError,
    sanitize_html_input
)

logger = logging.getLogger(__name__)

bp = Blueprint('files', __name__, url_prefix='/api/files')

# Kabul edilen dosya uzantıları (güvenlik kontrolü ile)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Dosya uzantısının geçerli olup olmadığını kontrol eder
def allowed_file(filename):
    """
    Yüklenen dosya uzantısının kabul edilen türlerden olup olmadığını kontrol eder.
    
    Args:
        filename: Kontrol edilecek dosya adı
        
    Returns:
        bool: Dosya uzantısı kabul ediliyorsa True, aksi halde False
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/', methods=['POST'])
def upload_file():
    """
    Güvenli dosya yükleme endpoint'i.
    """
    try:
        # Input validation
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya bulunamadı'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        # Validate file using security module
        try:
            file_info = validate_file_upload(file, ALLOWED_EXTENSIONS)
        except FileSecurityError as e:
            return jsonify({'error': f'Dosya güvenlik kontrolü başarısız: {str(e)}'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'Dosya çok büyük (max {MAX_FILE_SIZE // (1024*1024)}MB)'}), 400
        
        if file_size == 0:
            return jsonify({'error': 'Boş dosya yüklenemez'}), 400
        
        # Generate unique secure filename
        unique_filename = str(uuid.uuid4()) + '_' + file_info['safe_filename']
        
        # Validate upload path
        try:
            upload_folder = current_app.config['UPLOAD_FOLDER']
            safe_path = validate_path(
                os.path.join(upload_folder, unique_filename),
                upload_folder
            )
        except PathSecurityError as e:
            return jsonify({'error': f'Path güvenlik hatası: {str(e)}'}), 400
        
        # Save file securely
        try:
            file.save(safe_path)
        except Exception as e:
            logger.error(f"Dosya kaydetme hatası: {str(e)}")
            return jsonify({'error': 'Dosya kaydedilemedi'}), 500
        
        # Double-check MIME type after saving
        actual_mime, _ = mimetypes.guess_type(safe_path)
        if actual_mime and actual_mime != file_info['detected_mime']:
            logger.warning(f"MIME type mismatch: detected={file_info['detected_mime']}, actual={actual_mime}")
        
        # Create database record
        file_record = File(
            filename=unique_filename,
            original_filename=sanitize_html_input(file.filename),  # XSS protection
            file_path=safe_path,
            file_size=file_size,
            mime_type=file_info['detected_mime'],
            user_id=g.user.id if hasattr(g, 'user') else None
        )
        
        db.session.add(file_record)
        db.session.commit()
        
        # Validate auto_analyze parameter
        try:
            form_params = validate_request_params(
                dict(request.form),
                {
                    'auto_analyze': {
                        'type': 'bool',
                        'default': False
                    }
                }
            )
        except SecurityError as e:
            return jsonify({'error': f'Parameter hatası: {str(e)}'}), 400
        
        # Start analysis if requested
        if form_params.get('auto_analyze', False):
            analysis_service = AnalysisService()
            analysis = analysis_service.start_analysis(file_record.id)
            
            if analysis:
                return jsonify({
                    'message': 'Dosya başarıyla yüklendi ve analiz başlatıldı',
                    'file_id': file_record.id,
                    'analysis_id': analysis.id
                }), 201
        
        return jsonify({
            'message': 'Dosya başarıyla yüklendi',
            'file_id': file_record.id
        }), 201
        
    except Exception as e:
        logger.error(f"Dosya yüklenirken beklenmeyen hata: {str(e)}")
        return jsonify({'error': 'Dosya yüklenirken bir hata oluştu'}), 500

@bp.route('/', methods=['GET'])
def get_files():
    """
    Güvenli dosya listesi endpoint'i. Parametreleri doğrular ve güvenli sorgu yapar.
    """
    try:
        # Validate request parameters
        params = validate_request_params(
            dict(request.args),
            {
                'page': {
                    'type': 'int',
                    'min': 1,
                    'max': 1000,
                    'default': 1
                },
                'per_page': {
                    'type': 'int',
                    'min': 1,
                    'max': 100,
                    'default': 10
                },
                'file_type': {
                    'type': 'str',
                    'allowed': ['image', 'video'],
                    'required': False
                },
                'user_id': {
                    'type': 'int',
                    'min': 1,
                    'required': False
                }
            }
        )
        
        query = File.query
        
        # Apply filters safely
        if params.get('file_type'):
            query = query.filter(File.file_type == params['file_type'])
        if params.get('user_id'):
            query = query.filter(File.user_id == params['user_id'])
        
        query = query.order_by(File.created_at.desc())
        
        paginated_files = query.paginate(
            page=params['page'], 
            per_page=params['per_page'], 
            error_out=False
        )
        
        result = {
            'files': [file.to_dict() for file in paginated_files.items],
            'total': paginated_files.total,
            'pages': paginated_files.pages,
            'current_page': params['page']
        }
        
        return jsonify(result), 200
        
    except SecurityError as e:
        return jsonify({'error': f'Parameter hatası: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Dosya listesi alınırken hata: {str(e)}")
        return jsonify({'error': 'Dosya listesi alınırken bir hata oluştu'}), 500

@bp.route('/<int:file_id>', methods=['GET'])
def get_file(file_id):
    """
    Belirtilen ID'ye sahip dosyanın bilgilerini getirir.
    
    Args:
        file_id: Dosya ID'si
        
    Returns:
        JSON: Dosya bilgileri veya hata mesajı
    """
    try:
        file = File.query.get(file_id)
        
        if not file:
            return jsonify({'error': 'Dosya bulunamadı'}), 404
            
        return jsonify(file.to_dict()), 200
        
    except Exception as e:
        logger.error(f"Dosya bilgisi alınırken hata oluştu: {str(e)}")
        return jsonify({'error': f'Dosya bilgisi alınırken bir hata oluştu: {str(e)}'}), 500

@bp.route('/<int:file_id>/download', methods=['GET'])
def download_file(file_id):
    """
    Güvenli dosya indirme endpoint'i. Path traversal saldırılarına karşı korumalı.
    """
    try:
        # Validate file_id parameter
        if not isinstance(file_id, int) or file_id <= 0:
            return jsonify({'error': 'Geçersiz dosya ID'}), 400
        
        file = File.query.get(file_id)
        
        if not file:
            return jsonify({'error': 'Dosya bulunamadı'}), 404
        
        # Validate file path against directory traversal
        try:
            upload_folder = current_app.config['UPLOAD_FOLDER']
            safe_file_path = validate_path(file.file_path, upload_folder)
        except PathSecurityError as e:
            logger.error(f"Path security error for file {file_id}: {str(e)}")
            return jsonify({'error': 'Dosya erişim hatası'}), 403
        
        # Check if file actually exists
        if not os.path.exists(safe_file_path):
            logger.error(f"File not found on disk: {safe_file_path}")
            return jsonify({'error': 'Dosya sistemde bulunamadı'}), 404
        
        # Get directory and filename safely
        directory = os.path.dirname(safe_file_path)
        filename = os.path.basename(safe_file_path)
        
        # Ensure directory is still within upload folder after manipulation
        try:
            validate_path(directory, upload_folder)
        except PathSecurityError:
            return jsonify({'error': 'Güvenlik hatası: Geçersiz dosya yolu'}), 403
        
        return send_from_directory(
            directory,
            filename,
            as_attachment=True,
            download_name=sanitize_html_input(file.original_filename)
        )
        
    except Exception as e:
        logger.error(f"Dosya indirme hatası: {str(e)}")
        return jsonify({'error': 'Dosya indirilemedi'}), 500

@bp.route('/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """
    Belirtilen ID'ye sahip dosyayı sistemden ve veritabanından siler.
    
    Args:
        file_id: Silinecek dosyanın ID'si
        
    Returns:
        JSON: Başarı mesajı veya hata mesajı
    """
    try:
        file = File.query.get(file_id)
        
        if not file:
            return jsonify({'error': 'Dosya bulunamadı'}), 404
            
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        db.session.delete(file)
        db.session.commit()
        
        return jsonify({'message': 'Dosya başarıyla silindi'}), 200
        
    except Exception as e:
        logger.error(f"Dosya silinirken hata oluştu: {str(e)}")
        return jsonify({'error': f'Dosya silinirken bir hata oluştu: {str(e)}'}), 500

@bp.route('/frames/<int:frame_id>/<path:filename>', methods=['GET'])
def serve_frame_file(frame_id, filename):
    """
    Belirli bir frame klasöründeki işlenmiş görseli sunar.
    
    Args:
        frame_id: Frame dizini numarası (frames_X'deki X)
        filename: Görsel dosyasının adı (frame_XXXX_XX.XX.jpg formatında)
        
    Returns:
        File: İstenen görsel
    """
    try:
        frame_dir = f"frames_{frame_id}"
        frame_path = os.path.join(current_app.config['PROCESSED_FOLDER'], frame_dir)
        logger.info(f"Frame görselini servis etmeye çalışıyorum: {frame_dir}/{filename}")
        
        # Eğer dosya için tam yol verilmişse, temizle
        clean_filename = filename.split('/')[-1].split('\\')[-1]
        
        # Klasik dizinde ara
        if os.path.exists(frame_path) and os.path.exists(os.path.join(frame_path, clean_filename)):
            logger.info(f"Frame görsel bulundu: {frame_path}/{clean_filename}")
            return send_from_directory(frame_path, clean_filename, as_attachment=False)
        
        # Tüm frames_X dizinlerini kontrol et (backup)
        processed_folder = current_app.config['PROCESSED_FOLDER']
        all_frame_dirs = [d for d in os.listdir(processed_folder) if os.path.isdir(os.path.join(processed_folder, d)) and d.startswith('frames_')]
        
        for dir_name in all_frame_dirs:
            alt_path = os.path.join(processed_folder, dir_name, clean_filename)
            if os.path.exists(alt_path):
                logger.info(f"Frame görsel alternatif dizinde bulundu: {dir_name}/{clean_filename}")
                return send_from_directory(os.path.join(processed_folder, dir_name), clean_filename, as_attachment=False)
        
        # Doğrudan processed klasöründe ara
        direct_file = os.path.join(processed_folder, clean_filename)
        if os.path.exists(direct_file):
            logger.info(f"Frame görsel ana dizinde bulundu: {clean_filename}")
            return send_from_directory(processed_folder, clean_filename, as_attachment=False)
        
        # Hata detayları
        logger.error(f"Frame görsel bulunamadı: {frame_dir}/{clean_filename}")
        logger.error(f"Aranan dizinler: {frame_path} ve {all_frame_dirs}")
        return jsonify({'error': f'Frame görsel bulunamadı: {clean_filename}'}), 404
            
    except Exception as e:
        logger.error(f"Frame görsel servis hatası: {str(e)}")
        return jsonify({'error': f'Dosya servis hatası: {str(e)}'}), 500

@bp.route('/processed/<path:filename>', methods=['GET'])
def serve_processed_image(filename):
    """
    İşlenmiş bir görseli sunar.
    
    Args:
        filename: Görsel dosyasının adı
        
    Returns:
        File: İstenen görsel
    """
    try:
        # İşlenmiş klasöründe doğrudan ara
        if os.path.exists(os.path.join(current_app.config['PROCESSED_FOLDER'], filename)):
            return send_from_directory(current_app.config['PROCESSED_FOLDER'], filename, as_attachment=False)
        
        # Tüm frames_X dizinlerini kontrol et
        processed_folder = current_app.config['PROCESSED_FOLDER']
        frame_dirs = [d for d in os.listdir(processed_folder) if d.startswith('frames_')]
        
        for frame_dir in frame_dirs:
            frame_path = os.path.join(processed_folder, frame_dir, filename)
            if os.path.exists(frame_path):
                return send_from_directory(os.path.join(processed_folder, frame_dir), filename, as_attachment=False)
        
        return jsonify({'error': f'İşlenmiş görsel bulunamadı: {filename}'}), 404
        
    except Exception as e:
        logger.error(f"İşlenmiş görsel sunulurken hata: {str(e)}")
        return jsonify({'error': f'Dosya sunum hatası: {str(e)}'}), 500

# İşlenmiş dosyaları servis etmek için genel bir route
@bp.route('/storage/processed/<path:filename>', methods=['GET'])
def serve_storage_processed_file(filename):
    """
    İşlenmiş dosyaları storage/processed/ klasöründen sunar.
    
    Args:
        filename: Dosya adı veya alt dizin/dosya yolu
        
    Returns:
        File: İstenen dosya
    """
    try:
        # storage/processed/ altındaki tüm dizinleri kontrol et
        processed_folder = os.path.join(current_app.config['STORAGE_FOLDER'], 'processed')
        
        # Tüm frames_X dizinlerini bul
        frame_dirs = [d for d in os.listdir(processed_folder) if d.startswith('frames_')]
        
        # Her dizini kontrol et
        for frame_dir in frame_dirs:
            frame_path = os.path.join(processed_folder, frame_dir, filename)
            if os.path.exists(frame_path):
                return send_from_directory(
                    os.path.join(processed_folder, frame_dir),
                    filename,
                    as_attachment=False
                )
                
        # Ana processed klasöründe ara
        if os.path.exists(os.path.join(processed_folder, filename)):
            return send_from_directory(
                processed_folder,
                filename,
                as_attachment=False
            )
                
        return jsonify({'error': f'Dosya bulunamadı: {filename}'}), 404
        
    except Exception as e:
        logger.error(f"İşlenmiş dosya sunulurken hata oluştu: {str(e)}")
        return jsonify({'error': f'Dosya bulunamadı: {str(e)}'}), 404

# Video karelerini doğrudan sunmak için yeni rota
@bp.route('/frames/<analysis_id>/<path:frame_file>', methods=['GET'])
def serve_analysis_frame(analysis_id, frame_file):
    """
    Analiz ID'sine göre işlenmiş bir video karesini sunar.
    
    Args:
        analysis_id: Analiz ID'si (UUID)
        frame_file: Kare dosyasının adı (örn. frame_000123_12.34.jpg)
        
    Returns:
        Dosya içeriği veya hata mesajı
    """
    try:
        # Önce doğrudan işlenmiş klasöründe ara
        processed_folder = current_app.config['PROCESSED_FOLDER']
        
        # Temizlenmiş dosya adı
        clean_frame = frame_file.split('/')[-1].split('\\')[-1]
        
        # Öncelikle UUID bazlı klasörde ara (en muhtemel konum)
        primary_path = os.path.join(processed_folder, f"frames_{analysis_id}", clean_frame)
        
        if os.path.exists(primary_path):
            current_app.logger.debug(f"Kare dosyası bulundu: {primary_path}")
            
            # MIME tipini belirle
            mime_type = 'image/jpeg'  # Varsayılan olarak JPEG
            if clean_frame.lower().endswith('.png'):
                mime_type = 'image/png'
            elif clean_frame.lower().endswith('.gif'):
                mime_type = 'image/gif'
            
            return send_file(primary_path, mimetype=mime_type)
        
        # Fallback: Ana klasörde ara
        fallback_path = os.path.join(processed_folder, clean_frame)
        if os.path.exists(fallback_path):
            current_app.logger.debug(f"Kare dosyası fallback konumunda bulundu: {fallback_path}")
            mime_type = 'image/jpeg'
            if clean_frame.lower().endswith('.png'):
                mime_type = 'image/png'
            elif clean_frame.lower().endswith('.gif'):
                mime_type = 'image/gif'
            return send_file(fallback_path, mimetype=mime_type)
        
        # Bulunamadı
        current_app.logger.error(f"Kare dosyası bulunamadı: {clean_frame}")
        return jsonify({'error': 'Dosya bulunamadı'}), 404
        
    except Exception as e:
        current_app.logger.error(f"Kare dosyası görüntülenirken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/files/processed/<path:frame_file>', methods=['GET'])
def get_processed_frame(frame_file):
    """
    İşlenmiş bir video karesini doğrudan dosyasından getirir.
    
    Args:
        frame_file: Kare dosyasının adı (örn. frame_000123_12.34.jpg)
        
    Returns:
        Dosya içeriği veya hata mesajı
    """
    try:
        # Dosya yolunu oluştur
        processed_folder = current_app.config['PROCESSED_FOLDER']
        
        # Temizlenmiş dosya adı
        clean_frame = frame_file.split('/')[-1].split('\\')[-1]
        
        # Ana dosya yolu
        file_path = os.path.join(processed_folder, clean_frame)
        
        # Alternatif yollar
        possible_paths = [
            file_path,  # Ana klasörde
            # Tüm frames_X klasörlerini kontrol et
            *[os.path.join(processed_folder, d, clean_frame) 
              for d in os.listdir(processed_folder) 
              if os.path.isdir(os.path.join(processed_folder, d)) and d.startswith('frames_')]
        ]
        
        # Log bilgisi
        current_app.logger.info(f"İşlenmiş kare dosyası aranıyor: {clean_frame}")
        
        # Olası yolları kontrol et
        for path in possible_paths:
            if os.path.exists(path):
                current_app.logger.info(f"İşlenmiş kare dosyası bulundu: {path}")
                
                # MIME tipini belirle
                mime_type = 'image/jpeg'  # Varsayılan olarak JPEG
                if clean_frame.lower().endswith('.png'):
                    mime_type = 'image/png'
                elif clean_frame.lower().endswith('.gif'):
                    mime_type = 'image/gif'
                
                return send_file(path, mimetype=mime_type)
        
        # Dosya bulunamadı
        current_app.logger.error(f"İşlenmiş kare dosyası bulunamadı: {clean_frame}")
        return jsonify({'error': 'Dosya bulunamadı'}), 404
        
    except Exception as e:
        current_app.logger.error(f"İşlenmiş kare dosyası servis edilirken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500 