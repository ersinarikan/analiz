/**
 * Progress Bar Debug Script
 * Ana sayfadaki progress bar sorunlarını debug etmek için kullanılır
 */

// Debug modunu aktifleştir
window.progressDebugMode = true;

// Original fonksiyonları saklayalım
window.originalUpdateFileStatus = window.updateFileStatus;
window.originalUpdateGlobalProgress = window.updateGlobalProgress;
window.originalGetCompletedAnalysesCount = window.getCompletedAnalysesCount;

// Debug wrappers
window.updateFileStatus = function(fileId, status, progress, error = null) {
    console.log('🐛 [DEBUG] updateFileStatus çağrıldı:', {
        fileId: fileId,
        status: status,
        progress: progress,
        error: error,
        timestamp: new Date().toISOString()
    });
    
    // fileStatuses map durumunu logla
    console.log('🐛 [DEBUG] fileStatuses öncesi:', new Map(fileStatuses));
    
    // Original fonksiyonu çağır
    const result = window.originalUpdateFileStatus(fileId, status, progress, error);
    
    // fileStatuses map durumunu tekrar logla
    console.log('🐛 [DEBUG] fileStatuses sonrası:', new Map(fileStatuses));
    
    // Progress bar elementini kontrol et
    const globalProgressBar = document.getElementById('globalProgressBar');
    const analysisStatus = document.getElementById('analysisStatus');
    
    console.log('🐛 [DEBUG] Progress bar elemenleri:', {
        globalProgressBar: {
            exists: !!globalProgressBar,
            width: globalProgressBar?.style.width,
            textContent: globalProgressBar?.textContent,
            ariaValueNow: globalProgressBar?.getAttribute('aria-valuenow')
        },
        analysisStatus: {
            exists: !!analysisStatus,
            textContent: analysisStatus?.textContent
        }
    });
    
    return result;
};

window.updateGlobalProgress = function(current, total) {
    console.log('🐛 [DEBUG] updateGlobalProgress çağrıldı:', {
        current: current,
        total: total,
        timestamp: new Date().toISOString()
    });
    
    // Completed count hesapla
    const completedCount = window.originalGetCompletedAnalysesCount();
    const totalFiles = fileStatuses.size;
    
    console.log('🐛 [DEBUG] Progress hesaplaması:', {
        completedCount: completedCount,
        totalFiles: totalFiles,
        fileStatuses: new Map(fileStatuses),
        calculatedCurrent: current || completedCount,
        calculatedTotal: total || totalFiles
    });
    
    // Original fonksiyonu çağır
    const result = window.originalUpdateGlobalProgress(current, total);
    
    // Sonucu kontrol et
    const globalProgressBar = document.getElementById('globalProgressBar');
    console.log('🐛 [DEBUG] updateGlobalProgress sonrası progress bar:', {
        width: globalProgressBar?.style.width,
        textContent: globalProgressBar?.textContent,
        ariaValueNow: globalProgressBar?.getAttribute('aria-valuenow')
    });
    
    return result;
};

window.getCompletedAnalysesCount = function() {
    const result = window.originalGetCompletedAnalysesCount();
    console.log('🐛 [DEBUG] getCompletedAnalysesCount:', {
        result: result,
        fileStatuses: new Map(fileStatuses),
        completedStatuses: Array.from(fileStatuses.values()).filter(s => s === 'completed')
    });
    return result;
};

// WebSocket event debug'ları
if (window.socketioClient && window.socketioClient.socket) {
    const socket = window.socketioClient.socket;
    
    // Analysis progress listener debug
    socket.on('analysis_progress_debug', (data) => {
        console.log('🐛 [DEBUG] WebSocket analysis_progress event:', {
            data: data,
            timestamp: new Date().toISOString(),
            fileAnalysisMap: window.fileAnalysisMap ? new Map(window.fileAnalysisMap) : 'not found'
        });
    });
    
    // Analysis completed listener debug
    socket.on('analysis_completed_debug', (data) => {
        console.log('🐛 [DEBUG] WebSocket analysis_completed event:', {
            data: data,
            timestamp: new Date().toISOString(),
            fileAnalysisMap: window.fileAnalysisMap ? new Map(window.fileAnalysisMap) : 'not found'
        });
    });
}

// Manual progress test fonksiyonu
window.testProgressBar = function() {
    console.log('🐛 [DEBUG] Manual progress bar test başlatılıyor...');
    
    // Test için fake file status'ları ekle
    fileStatuses.set('test-file-1', 'processing');
    fileStatuses.set('test-file-2', 'queued');
    fileStatuses.set('test-file-3', 'completed');
    
    console.log('🐛 [DEBUG] Test fileStatuses eklendi:', new Map(fileStatuses));
    
    // Progress bar'ı güncelle
    updateGlobalProgress();
    
    setTimeout(() => {
        fileStatuses.set('test-file-2', 'completed');
        updateGlobalProgress();
        console.log('🐛 [DEBUG] İkinci dosya completed olarak işaretlendi');
    }, 2000);
    
    setTimeout(() => {
        fileStatuses.set('test-file-1', 'completed');
        updateGlobalProgress();
        console.log('🐛 [DEBUG] Üçüncü dosya completed olarak işaretlendi');
        
        // Test dosyalarını temizle
        setTimeout(() => {
            fileStatuses.delete('test-file-1');
            fileStatuses.delete('test-file-2');
            fileStatuses.delete('test-file-3');
            updateGlobalProgress();
            console.log('🐛 [DEBUG] Test dosyaları temizlendi');
        }, 2000);
    }, 4000);
};

// Queue status debug
window.debugQueueStatus = function() {
    fetch('/api/queue/status')
    .then(response => response.json())
    .then(data => {
        console.log('🐛 [DEBUG] Queue status:', data);
    })
    .catch(error => {
        console.log('🐛 [DEBUG] Queue status error:', error);
    });
};

console.log('🐛 [DEBUG] Progress debug script loaded! Kullanılabilir fonksiyonlar:');
console.log('   - window.testProgressBar(): Manuel progress bar testi');
console.log('   - window.debugQueueStatus(): Queue durumu kontrol');
console.log('   - Tüm updateFileStatus ve updateGlobalProgress çağrıları loglanıyor'); 