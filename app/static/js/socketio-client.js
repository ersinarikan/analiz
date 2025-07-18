// SocketIO bağlantısı ve event listeners
let socketioClient = null;

function initializeSocketIO() {
    console.log("SocketIO bağlantısı başlatılıyor...");
    socketioClient = io({
        autoConnect: true,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 10,
        timeout: 60000,
        pingInterval: 25000,
        pingTimeout: 60000
    });
    
    socketioClient.on('connect', function() {
        console.log('SocketIO bağlantısı başarılı - ID:', socketioClient.id);
    });
    
    socketioClient.on('disconnect', function(reason) {
        console.log('SocketIO bağlantısı kesildi - Sebep:', reason);
    });
    
    // DEBUG: Tüm event'leri yakala
    socketioClient.onAny((eventName, ...args) => {
        console.log('🔵 [SocketIO] Event received:', eventName, args);
        console.log('🔵 [SocketIO] Connection ID at event time:', socketioClient.id);
        console.log('🔵 [SocketIO] Connected status:', socketioClient.connected);
    });
    
    // Connection error handling
    socketioClient.on('connect_error', function(error) {
        console.error('❌ SocketIO connection error:', error);
    });
    
    socketioClient.on('reconnect', function(attemptNumber) {
        console.log('🔄 SocketIO reconnected after', attemptNumber, 'attempts');
    });
    
    socketioClient.on('reconnect_error', function(error) {
        console.error('❌ SocketIO reconnection error:', error);
    });
    
    // Analiz başlama event'i
    socketioClient.on('analysis_started', function(data) {
        console.log('Analysis started:', data);
        if (typeof updateFileStatus === 'function') {
            updateFileStatus(data.file_id, data.status, data.progress);
        }
        if (typeof fileStatuses !== 'undefined') {
            fileStatuses.set(data.file_id, data.status);
        }
        
        // File'a analysisId ekle
        if (typeof uploadedFiles !== 'undefined') {
            const fileIndex = uploadedFiles.findIndex(f => f.id === data.file_id);
            if (fileIndex !== -1) {
                uploadedFiles[fileIndex].analysisId = data.analysis_id;
            }
        }
        if (typeof fileAnalysisMap !== 'undefined') {
            fileAnalysisMap.set(data.file_id, data.analysis_id);
        }
        
        if (typeof updateGlobalProgress === 'function') {
            updateGlobalProgress();
        }
    });
    
    // Analiz progress event'i
    socketioClient.on('analysis_progress', function(data) {
        console.log('Analysis progress:', data);
        if (typeof updateFileStatus === 'function') {
            updateFileStatus(data.file_id, data.status, data.progress);
        }
        if (typeof fileStatuses !== 'undefined') {
            fileStatuses.set(data.file_id, data.status);
        }
        if (typeof updateGlobalProgress === 'function') {
            updateGlobalProgress();
        }
    });
    
    // Analiz tamamlanma event'i
    socketioClient.on('analysis_completed', function(data) {
        console.log('Analysis completed:', data);
        if (typeof updateFileStatus === 'function') {
            updateFileStatus(data.file_id, 'completed', 100);
        }
        if (typeof fileStatuses !== 'undefined') {
            fileStatuses.set(data.file_id, 'completed');
        }
        
        // Sonuçları getir - polling yerine anında
        setTimeout(() => {
            if (typeof getAnalysisResults === 'function') {
                getAnalysisResults(data.file_id, data.analysis_id);
            }
        }, 500); // 500ms küçük delay - backend'in tamamen bitmesi için
        
        if (typeof updateGlobalProgress === 'function') {
            updateGlobalProgress();
        }
    });
    
    // Analiz başarısız event'i
    socketioClient.on('analysis_failed', function(data) {
        console.log('Analysis failed:', data);
        if (typeof updateFileStatus === 'function') {
            updateFileStatus(data.file_id, 'failed', 0);
        }
        if (typeof fileStatuses !== 'undefined') {
            fileStatuses.set(data.file_id, 'failed');
        }
        if (typeof showError === 'function' && typeof fileNameFromId === 'function') {
            showError(`${fileNameFromId(data.file_id)} dosyası için analiz başarısız oldu: ${data.message}`);
        }
        if (typeof updateGlobalProgress === 'function') {
            updateGlobalProgress();
        }
    });
}

// DOM yüklendiğinde SocketIO'yu başlat
document.addEventListener('DOMContentLoaded', function() {
    initializeSocketIO();
}); 