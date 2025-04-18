import os
import uuid
from flask import Blueprint, request, jsonify, current_app, send_from_directory, g
from werkzeug.utils import secure_filename
import magic
import logging

from app import db
from app.models.file import File
from app.services.file_service import is_allowed_file, save_uploaded_file
from app.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

bp = Blueprint('files', __name__, url_prefix='/api/files')

# Kabul edilen dosya türleri
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

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
    Dosya yükleme endpoint'i. Medya dosyasını sisteme kaydeder ve veritabanına kaydını oluşturur.
    
    Returns:
        JSON: Yüklenen dosya bilgileri veya hata mesajı
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Geçersiz dosya türü. İzin verilen türler: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + '_' + filename
        
        file_path = save_uploaded_file(file, unique_filename)
        
        if not file_path:
            return jsonify({'error': 'Dosya kaydedilemedi'}), 500
        
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)
        
        if not (mime_type.startswith('image/') or mime_type.startswith('video/')):
            os.remove(file_path)
            return jsonify({'error': 'Sadece resim ve video dosyaları desteklenmektedir'}), 400
        
        file_size = os.path.getsize(file_path)
        file_record = File(
            filename=unique_filename,
            original_filename=filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            user_id=g.user.id if hasattr(g, 'user') else None
        )
        
        db.session.add(file_record)
        db.session.commit()
        
        auto_analyze = request.form.get('auto_analyze', 'false').lower() == 'true'
        if auto_analyze:
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
        logger.error(f"Dosya yüklenirken hata oluştu: {str(e)}")
        return jsonify({'error': f'Dosya yüklenirken bir hata oluştu: {str(e)}'}), 500

@bp.route('/', methods=['GET'])
def get_files():
    """
    Sistemdeki tüm dosyaları veya belirli kullanıcıya ait dosyaları listeler.
    Sayfalama ve filtreleme özellikleri sunar.
    
    Returns:
        JSON: Dosya listesi veya hata mesajı
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        file_type = request.args.get('file_type')
        user_id = request.args.get('user_id')
        
        query = File.query
        
        if file_type:
            query = query.filter(File.file_type == file_type)
        if user_id:
            query = query.filter(File.user_id == user_id)
        
        query = query.order_by(File.created_at.desc())
        
        paginated_files = query.paginate(page=page, per_page=per_page, error_out=False)
        
        result = {
            'files': [file.to_dict() for file in paginated_files.items],
            'total': paginated_files.total,
            'pages': paginated_files.pages,
            'current_page': page
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Dosya listesi alınırken hata oluştu: {str(e)}")
        return jsonify({'error': f'Dosya listesi alınırken bir hata oluştu: {str(e)}'}), 500

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
    Belirtilen ID'ye sahip dosyayı indirme endpoint'i.
    
    Args:
        file_id: İndirilecek dosyanın ID'si
        
    Returns:
        File: İndirilecek dosya veya hata mesajı
    """
    try:
        file = File.query.get(file_id)
        
        if not file:
            return jsonify({'error': 'Dosya bulunamadı'}), 404
            
        return send_from_directory(
            current_app.config['UPLOAD_FOLDER'],
            file.filename,
            as_attachment=True,
            download_name=file.original_filename
        )
        
    except Exception as e:
        logger.error(f"Dosya indirilirken hata oluştu: {str(e)}")
        return jsonify({'error': f'Dosya indirilirken bir hata oluştu: {str(e)}'}), 500

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
        import os
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