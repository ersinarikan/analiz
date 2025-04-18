"""
Analiz işlemlerini gerçekleştiren fonksiyonlar.
Bu modül daha önce Celery görevleri içeriyordu, şimdi doğrudan çağrılabilir fonksiyonlar içeriyor.
"""
from flask import current_app
from app import socketio
from app.services.analysis_service import analyze_file

def start_analysis_task(analysis_id):
    """Bir dosyanın analizini asenkron olarak gerçekleştirir."""
    try:
        # WebSocket üzerinden ilerleme bildirimi gönder
        socketio.emit('analysis_started', {
            'analysis_id': analysis_id
        })
        
        # Analizi gerçekleştir
        success, message = analyze_file(analysis_id)
        
        # WebSocket üzerinden sonuç bildirimi gönder
        if success:
            socketio.emit('analysis_completed', {
                'analysis_id': analysis_id
            })
            return {'status': 'completed', 'analysis_id': analysis_id, 'message': message}
        else:
            socketio.emit('analysis_failed', {
                'analysis_id': analysis_id,
                'error': message
            })
            return {'status': 'failed', 'analysis_id': analysis_id, 'error': message}
    
    except Exception as e:
        # Hata durumunu WebSocket üzerinden bildir
        error_message = f"Analiz görevi hatası: {str(e)}"
        current_app.logger.error(error_message)
        
        socketio.emit('analysis_failed', {
            'analysis_id': analysis_id,
            'error': error_message
        })
        
        return {'status': 'failed', 'analysis_id': analysis_id, 'error': error_message}


def batch_analysis_task(analysis_ids):
    """Birden fazla dosyanın analizini sırayla gerçekleştirir."""
    results = []
    total = len(analysis_ids)
    
    for i, analysis_id in enumerate(analysis_ids):
        try:
            # Her bir dosyayı sırayla analiz et
            success, message = analyze_file(analysis_id)
            
            results.append({
                'analysis_id': analysis_id,
                'status': 'completed' if success else 'failed',
                'message': message
            })
            
            # WebSocket üzerinden ilerleme bildirimi gönder
            socketio.emit('batch_analysis_progress', {
                'current': i + 1,
                'total': total,
                'progress': ((i + 1) / total) * 100,
                'results': results
            })
        
        except Exception as e:
            error_message = f"Toplu analiz hatası (ID {analysis_id}): {str(e)}"
            current_app.logger.error(error_message)
            
            results.append({
                'analysis_id': analysis_id,
                'status': 'failed',
                'error': error_message
            })
    
    # WebSocket üzerinden tamamlandı bildirimi gönder
    socketio.emit('batch_analysis_completed', {
        'results': results
    })
    
    return {
        'status': 'completed',
        'results': results
    } 