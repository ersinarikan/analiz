// SocketIO bağlantısı ve event listeners
let socketioClient = null;

function initializeSocketIO() {
    console.log("SocketIO bağlantısı başlatılıyor...");
    socketioClient = io();
    
    socketioClient.on('connect', function() {
        console.log('SocketIO bağlantısı başarılı');
    });
    
    socketioClient.on('disconnect', function() {
        console.log('SocketIO bağlantısı kesildi');
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