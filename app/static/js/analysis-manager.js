/**
 * WSANALIZ - Analysis Manager Module
 * 
 * Bu modül analiz süreçlerini, kuyruk yönetimini ve progress tracking'i yönetir.
 * main.js'ten extract edilmiştir.
 */

import { 
    uploadedFiles,
    fileStatuses,
    fileAnalysisMap,
    cancelledAnalyses,
    API_URL,
    showToast,
    showError,
    fileNameFromId
} from './globals.js';

import { updateFileStatus } from './file-manager.js';
import { emitSocketEvent, isSocketConnected } from './websocket-manager.js';

// =====================================
// ANALYSIS MANAGEMENT
// =====================================

let queueStatusChecker = null;
const QUEUE_CHECK_INTERVAL = 2000; // 2 saniye

/**
 * Tüm yüklenen dosyalar için analiz başlatır
 * @param {number} framesPerSecond - Video için FPS
 * @param {boolean} includeAgeAnalysis - Yaş analizi dahil edilsin mi
 */
export function startAnalysisForAllFiles(framesPerSecond, includeAgeAnalysis) {
    const settingsSaveLoader = document.getElementById('settingsSaveLoader');
    console.log('[DEBUG] startAnalysisForAllFiles: settingsSaveLoader element:', settingsSaveLoader);
    
    // Loading spinner göster
    if (settingsSaveLoader) {
        settingsSaveLoader.style.display = 'inline-block';
        settingsSaveLoader.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analiz başlatılıyor...';
        console.log('[DEBUG] startAnalysisForAllFiles: Loading spinner GÖSTERILDI');
    } else {
        console.error('[DEBUG] startAnalysisForAllFiles: settingsSaveLoader element BULUNAMADI!');
    }
    
    // Analiz Et ve Analiz Başlat butonlarını "Analizi Durdur" moduna çevir
    changeButtonsToStopMode();
    
    // Her dosya için analiz başlat
    uploadedFiles.forEach(file => {
        if (file.serverFileId) {
            // File status'ını güncelle
            updateFileStatus(file.id, 'Sırada', 0);
            
            // Analizi başlat
            startAnalysis(file.id, file.serverFileId, framesPerSecond, includeAgeAnalysis);
        }
    });
    
    // Queue status checker'ı başlat
    startQueueStatusChecker();
}

/**
 * Tek bir dosya için analiz başlatır
 * @param {string} fileId - Client file ID
 * @param {number} serverFileId - Server file ID
 * @param {number} framesPerSecond - Video için FPS
 * @param {boolean} includeAgeAnalysis - Yaş analizi dahil edilsin mi
 */
export function startAnalysis(fileId, serverFileId, framesPerSecond, includeAgeAnalysis) {
    const analysisParams = {
        file_id: serverFileId,
        frames_per_second: framesPerSecond || 1,
        include_age_analysis: includeAgeAnalysis || false
    };
    
    console.log("Analiz başlatılıyor:", analysisParams);
    
    // Temporary mapping oluştur (analysis ID gelmeden önce)
    const tempMappingKey = `temp_${serverFileId}`;
    window.fileIdToCardId = window.fileIdToCardId || {};
    window.fileIdToCardId[tempMappingKey] = fileId;
    console.log(`[DEBUG] Immediate temporary mapping: ${tempMappingKey} → ${fileId}`);
    
    fetch(`${API_URL}/analysis/start`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(analysisParams)
    })
    .then(response => response.json())
    .then(data => {
        if (data.analysis) {
            console.log("Analysis started", data);
            
            // Temporary mapping'i temizle
            delete window.fileIdToCardId[tempMappingKey];
            console.log(`[DEBUG] Temp mapping temizlendi: ${tempMappingKey}`);
            
            // Real mapping oluştur
            const analysisId = data.analysis.id;
            fileAnalysisMap.set(analysisId, fileId);
            console.log(`[DEBUG] fileAnalysisMap güncellendi: ${analysisId} ${fileId}`, fileAnalysisMap);
            
            // DOM'da analysis-id attribute'unu set et
            const fileCard = document.getElementById(fileId);
            if (fileCard) {
                fileCard.setAttribute('data-analysis-id', analysisId);
                console.log(`[DEBUG] DOM fileCard analysis-id set edildi: ${fileId} ${analysisId}`);
                
                const fileCardElement = fileCard.querySelector('.file-card');
                if (fileCardElement) {
                    fileCardElement.setAttribute('data-analysis-id', analysisId);
                    console.log(`[DEBUG] file-card data-analysis-id güncellendi:`, fileCardElement);
                }
            }
            
            // WebSocket analysis room'una katıl
            joinAnalysisRoom(analysisId, fileId);
            
        } else {
            throw new Error(data.error || 'Analiz başlatılamadı');
        }
    })
    .catch(error => {
        console.error('Analysis start error:', error);
        updateFileStatus(fileId, 'Hata', 0, null, error.message);
        showError(`Analiz başlatma hatası: ${error.message}`);
    });
}

/**
 * WebSocket analysis room'una katılır
 * @param {string} analysisId - Analysis ID
 * @param {string} fileId - File ID
 */
function joinAnalysisRoom(analysisId, fileId) {
    console.log(`[DEBUG] WebSocket join kontrolleri:`, {
        analysisId,
        fileId,
        isConnected: isSocketConnected()
    });
    
    if (isSocketConnected()) {
        console.log(`🚀 WebSocket analysis room'una katılıyor: ${analysisId}`);
        emitSocketEvent('join_analysis', analysisId);
        console.log(`[WebSocket] Analiz odasına katılındı: analysis_${analysisId}`);
        
        // Alert timeout ayarla (48 saniye)
        const alertTimeout = setTimeout(() => {
            console.log(`[DEBUG] 🔥 Alert timeout set for file: ${fileId}`, Date.now());
        }, 48000);
        
        // Timeout'u global bir yerde sakla (gerekirse iptal etmek için)
        if (!window.analysisAlertTimeouts) {
            window.analysisAlertTimeouts = {};
        }
        window.analysisAlertTimeouts[fileId] = alertTimeout;
    } else {
        console.warn('⚠️ WebSocket bağlı değil, analysis room\'una katılamadı');
    }
}

/**
 * Analizi durdur fonksiyonu
 */
export function stopAnalysis() {
    console.log('[DEBUG] stopAnalysis çağrıldı');
    
    // 🚀 "Analizi Durdur" butonuna basıldıysa kullanıcı zaten onaylamış demektir
    // Notification engelleme sistemi confirm'u engelleyebilir, bu durumda da devam et
    let userConfirmed = false;
    try {
        userConfirmed = confirm('Tüm analizler durdurulacak ve kuyruk temizlenecek. Emin misiniz?');
    } catch(e) {
        console.log('[DEBUG] stopAnalysis: Confirm dialog hatası/engellendi, devam ediliyor...');
        userConfirmed = true; // Dialog engellenirse otomatik onay
    }
    
    // Eğer confirm false dönerse ve notification engelleme sisteminden kaynaklanmıyorsa
    if (!userConfirmed && window.confirm !== undefined) {
        console.log('[DEBUG] stopAnalysis: Kullanıcı işlemi iptal etti');
        return;
    }
    
    console.log('[DEBUG] stopAnalysis: İşlem onaylandı, API çağrısı yapılıyor...');
    
    // Loading spinner'ı gizle
    const settingsSaveLoader = document.getElementById('settingsSaveLoader');
    if (settingsSaveLoader) {
        settingsSaveLoader.style.display = 'none';
        settingsSaveLoader.innerHTML = '';
        console.log('[DEBUG] stopAnalysis: Loading spinner gizlendi');
    }
    
    // API'ye durdurma isteği gönder
    fetch('/api/queue/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log('[DEBUG] stopAnalysis: Response status:', response.status);
        console.log('[DEBUG] stopAnalysis: Response ok:', response.ok);
        return response.json();
    })
    .then(data => {
        console.log('[DEBUG] stopAnalysis API response:', data);
        console.log('[DEBUG] stopAnalysis: showToast çağrılıyor...');
        showToast('Başarılı', 'Analizler durduruldu ve kuyruk temizlendi.', 'success');
        console.log('[DEBUG] stopAnalysis: showToast çağrıldı');
        
        // Tüm dosya durumlarını iptal edildi olarak işaretle
        for (const [fileId, status] of fileStatuses.entries()) {
            if (status !== "completed" && status !== "failed") {
                updateFileStatus(fileId, "cancelled", 0, null, null);
            }
        }
        
        // Analyse button'ları eski haline döndür
        resetAnalyzeButton();
        
        // Queue checker'ı durdur
        stopQueueStatusChecker();
        
        // Sayfa yeniden yükleme (temizlik için)
        setTimeout(() => {
            console.log('[DEBUG] stopAnalysis: Sayfa yeniden yükleniyor...');
            window.location.reload();
        }, 2000);
    })
    .catch(error => {
        console.error('[DEBUG] stopAnalysis error:', error);
        showError('Analizi durdururken bir hata oluştu.');
        
        // Hata durumunda da button'ları reset et
        resetAnalyzeButton();
    });
}

/**
 * Analyze butonlarını "Durdur" moduna çevirir
 */
function changeButtonsToStopMode() {
    // "Analiz Et" butonunu "Analizi Durdur" olarak değiştir
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-stop me-1"></i> Analizi Durdur';
        analyzeBtn.className = 'btn btn-danger';
        
        // 🔧 TÜM EVENT LISTENER'LARI TEMİZLE (modal açan handler'lar dahil)
        const newAnalyzeBtn = analyzeBtn.cloneNode(true);
        analyzeBtn.parentNode.replaceChild(newAnalyzeBtn, analyzeBtn);
        
        // Sadece stopAnalysis handler'ını ekle
        newAnalyzeBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[DEBUG] Analizi Durdur butonu tıklandı!');
            stopAnalysis();
        };
        console.log('[DEBUG] Analiz Et butonu -> Analizi Durdur olarak değiştirildi (event listeners temizlendi)');
    }
    
    // Ana sayfadaki "Analiz Başlat" butonunu da değiştir
    const startAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
    if (startAnalysisMainBtn) {
        startAnalysisMainBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Analizi Durdur';
        startAnalysisMainBtn.className = 'btn btn-danger btn-lg me-3';
        
        // 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newStartAnalysisMainBtn = startAnalysisMainBtn.cloneNode(true);
        startAnalysisMainBtn.parentNode.replaceChild(newStartAnalysisMainBtn, startAnalysisMainBtn);
        
        // Sadece stopAnalysis handler'ını ekle
        newStartAnalysisMainBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[DEBUG] Ana sayfa Analizi Durdur butonu tıklandı!');
            stopAnalysis();
        };
        console.log('[DEBUG] Analiz Başlat butonu -> Analizi Durdur olarak değiştirildi (event listeners temizlendi)');
    }
}

/**
 * Analyze butonlarını orijinal haline döndürür
 */
export function resetAnalyzeButton() {
    // "Analizi Durdur" butonunu "Analiz Et" olarak değiştir
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-play me-1"></i> Analiz Et';
        analyzeBtn.className = 'btn btn-primary';
        
        // 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newAnalyzeBtn = analyzeBtn.cloneNode(true);
        analyzeBtn.parentNode.replaceChild(newAnalyzeBtn, analyzeBtn);
        
        // Yeni referansı al ve orijinal event listener'ı ekle
        const newAnalyzeBtnRef = document.getElementById('analyzeBtn');
        newAnalyzeBtnRef.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (uploadedFiles.length > 0) {
                // Analiz parametreleri modalını aç (ANLIK AYARLAR İÇİN YENİ MODAL)
                const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
                modal.show();
            }
        };
    }
    
    // Ana sayfadaki butonu da değiştir
    const startAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
    if (startAnalysisMainBtn) {
        startAnalysisMainBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Dosya Ekle ve Analiz Et';
        startAnalysisMainBtn.className = 'btn btn-primary btn-lg me-3';
        
        // 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newStartAnalysisMainBtn = startAnalysisMainBtn.cloneNode(true);
        startAnalysisMainBtn.parentNode.replaceChild(newStartAnalysisMainBtn, startAnalysisMainBtn);
        
        // Yeni referansı al ve orijinal event listener'ı ekle
        const newStartAnalysisMainBtnRef = document.getElementById('startAnalysisMainBtn');
        newStartAnalysisMainBtnRef.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation(); 
            document.getElementById('uploadFileBtn').click(); 
        };
    }
}

// =====================================
// QUEUE STATUS MANAGEMENT
// =====================================

/**
 * Queue status checker'ı başlatır
 */
export function startQueueStatusChecker() {
    if (queueStatusChecker) {
        clearInterval(queueStatusChecker);
    }
    
    queueStatusChecker = setInterval(() => {
        checkQueueStatus();
    }, QUEUE_CHECK_INTERVAL);
    
    console.log('Queue status checker başlatıldı');
}

/**
 * Queue status checker'ı durdurur
 */
export function stopQueueStatusChecker() {
    if (queueStatusChecker) {
        clearInterval(queueStatusChecker);
        queueStatusChecker = null;
        console.log('Queue status checker durduruldu');
    }
}

/**
 * Queue status'ını kontrol eder
 */
function checkQueueStatus() {
    fetch(`${API_URL}/queue/status`)
    .then(response => response.json())
    .then(data => {
        updateQueueStatus(data);
    })
    .catch(error => {
        console.error('Queue status check error:', error);
    });
}

/**
 * Queue status'ını günceller
 * @param {Object} response - API response
 */
function updateQueueStatus(response) {
    // Queue size'ı göster
    const queueSizeElement = document.getElementById('queueSize');
    if (queueSizeElement && response.queue_size !== undefined) {
        queueSizeElement.textContent = response.queue_size;
    }
    
    // Processing status'ını göster
    const processingStatusElement = document.getElementById('processingStatus');
    if (processingStatusElement) {
        processingStatusElement.textContent = response.is_processing ? 'Evet' : 'Hayır';
    }
    
    // Eğer kuyruk boşsa ve işlem yoksa, checker'ı durdur
    if (response.queue_size === 0 && !response.is_processing) {
        console.log('Kuyruk boş ve işlem yok, status checker durduruluyor');
        stopQueueStatusChecker();
        
        // Tüm analizler tamamlandığını kontrol et
        checkAllAnalysesCompleted();
    }
}

/**
 * Tüm analizlerin tamamlanıp tamamlanmadığını kontrol eder
 */
function checkAllAnalysesCompleted() {
    const completedCount = getCompletedAnalysesCount();
    const totalCount = uploadedFiles.length;
    
    if (completedCount === totalCount && totalCount > 0) {
        console.log('🎉 Tüm analizler tamamlandı!');
        showToast('Başarılı', 'Tüm analizler tamamlandı!', 'success');
        
        // Butonları reset et
        resetAnalyzeButton();
        
        // Loading spinner'ı gizle
        const settingsSaveLoader = document.getElementById('settingsSaveLoader');
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'none';
        }
    }
}

/**
 * Tamamlanan analiz sayısını döndürür
 */
function getCompletedAnalysesCount() {
    let completedCount = 0;
    for (const [fileId, status] of fileStatuses.entries()) {
        if (status === 'completed' || status === 'failed') {
            completedCount++;
        }
    }
    return completedCount;
}

// =====================================
// ANALYSIS PROGRESS HANDLING
// =====================================

/**
 * Analysis progress event'ini işler
 * @param {Object} data - Progress data
 */
export function handleAnalysisProgress(data) {
    const analysisId = data.analysis_id;
    const progress = data.progress || 0;
    const message = data.message || '';
    
    // Analysis ID'den file ID'yi bul
    const fileId = fileAnalysisMap.get(analysisId);
    if (fileId) {
        // Processing status tespit et
        if (data.status === 'processing' || progress > 0) {
            console.log(`[DEBUG] updateFileStatus - Processing status tespit edildi, progress: ${progress} , mesaj: ${message}`);
            
            // Loading spinner'ı gizle (processing başladığında)
            const settingsSaveLoader = document.getElementById('settingsSaveLoader');
            if (settingsSaveLoader && settingsSaveLoader.style.display !== 'none') {
                settingsSaveLoader.style.display = 'none';
                console.log('[DEBUG] updateFileStatus: Processing status ile Loading spinner GİZLENDİ');
            }
            
            updateFileStatus(fileId, 'processing', progress, message);
        }
    }
}

/**
 * Analysis completed event'ini işler
 * @param {Object} data - Completion data
 */
export function handleAnalysisCompleted(data) {
    const analysisId = data.analysis_id;
    const message = data.message || 'Analiz tamamlandı';
    const success = data.success !== false;
    
    // Analysis ID'den file ID'yi bul
    const fileId = fileAnalysisMap.get(analysisId);
    if (fileId) {
        const status = success ? 'completed' : 'failed';
        const progress = success ? 100 : 0;
        
        updateFileStatus(fileId, status, progress, message);
        
        // Alert timeout'u temizle
        if (window.analysisAlertTimeouts && window.analysisAlertTimeouts[fileId]) {
            clearTimeout(window.analysisAlertTimeouts[fileId]);
            delete window.analysisAlertTimeouts[fileId];
        }
        
        console.log(`Analysis ${success ? 'tamamlandı' : 'başarısız'}: ${fileNameFromId(fileId)}`);
    }
}

/**
 * Analysis manager fonksiyonlarını window'a expose et
 */
export function exposeAnalysisManagerToWindow() {
    window.analysisManager = {
        startAnalysisForAllFiles,
        startAnalysis,
        stopAnalysis,
        resetAnalyzeButton,
        handleAnalysisProgress,
        handleAnalysisCompleted,
        checkAllAnalysesCompleted: checkAllAnalysesCompleted
    };
}

// Initialize window exposure
exposeAnalysisManagerToWindow(); 