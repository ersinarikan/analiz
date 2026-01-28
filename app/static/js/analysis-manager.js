/* ERSIN Aciklama. */

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

// ERSIN =====================================
// ERSIN UTILITY FUNCTIONS
// ERSIN =====================================

/* ERSIN Aciklama. */
function getCategoryNameTurkish(category) {
    const names = {
        'violence': 'Şiddet',
        'adult_content': 'Yetişkin İçeriği',
        'harassment': 'Taciz',
        'weapon': 'Silah',
        'drug': 'Madde Kullanımı',
        'safe': 'Güvenli'
    };
    return names[category] || category;
}

// ERSIN =====================================
// ERSIN ANALYSIS MANAGEMENT
// ERSIN =====================================

let queueStatusChecker = null;
const QUEUE_CHECK_INTERVAL = 10000;  // ERSIN 10 saniye - Rate limiting önlemi

/* ERSIN Aciklama. */
export function startAnalysisForAllFiles(framesPerSecond, includeAgeAnalysis) {
    const settingsSaveLoader = document.getElementById('settingsSaveLoader');
    console.log('[DEBUG] startAnalysisForAllFiles: settingsSaveLoader element:', settingsSaveLoader);
    
    // ERSIN Loading spinner göster
    if (settingsSaveLoader) {
        settingsSaveLoader.style.display = 'inline-block';
        settingsSaveLoader.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analiz başlatılıyor...';
        console.log('[DEBUG] startAnalysisForAllFiles: Loading spinner GÖSTERILDI');
    } else {
        console.error('[DEBUG] startAnalysisForAllFiles: settingsSaveLoader element BULUNAMADI!');
    }
    
    // ERSIN Analiz Et ve Analiz Başlat butonlarını "Analizi Durdur" moduna çevir
    changeButtonsToStopMode();
    
    // ERSIN Her dosya için analiz başlat
    uploadedFiles.forEach(file => {
        if (file.serverFileId) {
            // ERSIN File status'ını güncelle
            updateFileStatus(file.id, 'Sırada', 0);
            
            // ERSIN Analizi başlat
            startAnalysis(file.id, file.serverFileId, framesPerSecond, includeAgeAnalysis);
        }
    });
    
    // ERSIN Queue status checker'ı başlat
    startQueueStatusChecker();
}

/* ERSIN Aciklama. */
export function startAnalysis(fileId, serverFileId, framesPerSecond, includeAgeAnalysis) {
    const analysisParams = {
        file_id: serverFileId,
        frames_per_second: framesPerSecond || 1,
        include_age_analysis: includeAgeAnalysis || false
    };
    
    console.log("🔍 ANALIZ BAŞLATILIYOR:", analysisParams);
    console.log("🔍 include_age_analysis değeri:", analysisParams.include_age_analysis);
    console.log("🔍 includeAgeAnalysis parameter değeri:", includeAgeAnalysis);
    
    // ERSIN Temporary mapping oluştur (analysis ID gelmeden önce)
    const tempMappingKey = `temp_${serverFileId}`;
    window.fileIdToCardId = window.fileIdToCardId || {};
    window.fileIdToCardId[tempMappingKey] = fileId;
    console.log(`[DEBUG] Immediate temporary mapping: ${tempMappingKey} → ${fileId}`);
    
    fetch(`${API_URL}/analysis/start`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-WebSocket-Session-ID': window.socketioClient?.socket?.id || null
        },
        body: JSON.stringify(analysisParams)
    })
    .then(response => response.json())
    .then(data => {
        if (data.analysis) {
            console.log("Analysis started", data);
            
            // ERSIN Temporary mapping'i temizle
            delete window.fileIdToCardId[tempMappingKey];
            console.log(`[DEBUG] Temp mapping temizlendi: ${tempMappingKey}`);
            
            // ERSIN Real mapping oluştur
            const analysisId = data.analysis.id;
            fileAnalysisMap.set(analysisId, fileId);
            console.log(`[DEBUG] fileAnalysisMap güncellendi: ${analysisId} ${fileId}`, fileAnalysisMap);
            
            // ERSIN DOM'da analysis-id attribute'unu set et
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
            
            // ERSIN WebSocket analysis room'una katıl
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

/* ERSIN Aciklama. */
function joinAnalysisRoom(analysisId, fileId) {
    console.log(`[DEBUG] WebSocket join kontrolleri:`, {
        analysisId,
        fileId,
        isConnected: isSocketConnected(),
        socketClient: !!window.socketioClient,
        socketConnected: window.socketioClient?.connected
    });
    
    if (isSocketConnected()) {
        console.log(`🚀 WebSocket analysis room'una katılıyor: ${analysisId}`);
        emitSocketEvent('join_analysis', analysisId);
        console.log(`[WebSocket] Analiz odasına katılındı: analysis_${analysisId}`);
        
        // ERSIN Alert timeout ayarla (48 saniye)
        const alertTimeout = setTimeout(() => {
            console.log(`[DEBUG] 🔥 Alert timeout set for file: ${fileId}`, Date.now());
        }, 48000);
        
        // ERSIN Timeout'u global bir yerde sakla (gerekirse iptal etmek için)
        if (!window.analysisAlertTimeouts) {
            window.analysisAlertTimeouts = {};
        }
        window.analysisAlertTimeouts[fileId] = alertTimeout;
    } else {
        console.warn('⚠️ WebSocket bağlı değil, analysis room\'una katılamadı');
    }
}

/* ERSIN Aciklama. */
export function stopAnalysis() {
    console.log('[DEBUG] stopAnalysis çağrıldı - Force Stop modunda');
    
    // ERSIN Kullanıcı onayı
    let userConfirmed = false;
    try {
        userConfirmed = confirm('🚨 ZORLA DURDURMA 🚨\n\n• Tüm aktif analizler zorla durdurulacak\n• Veritabanından silinecek\n• Dosyalar temizlenecek\n• Uygulama restart edilecek\n\nEmin misiniz?');
    } catch(e) {
        console.log('[DEBUG] stopAnalysis: Confirm dialog hatası/engellendi');
        userConfirmed = false;  // ERSIN Force stop için kesinlikle onay gerekli
    }
    
    if (!userConfirmed) {
        console.log('[DEBUG] stopAnalysis: Kullanıcı işlemi iptal etti');
        return;
    }
    
    console.log('[DEBUG] stopAnalysis: Force Stop onaylandı, loading başlatılıyor...');
    
    // ERSIN Loading overlay göster
    showFullPageLoading();
    
    // ERSIN Force stop bildirim göster
    showToast('Zorla Durdurma', 'Aktif analizler zorla durduruluyor...', 'warning');
    
    // ERSIN API'ye force-stop isteği gönder
    fetch('/api/queue/force-stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log('[DEBUG] forceStopAnalysis: Response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('[DEBUG] forceStopAnalysis API response:', data);
        
        if (data.force_stopped) {
            // ERSIN Başarılı force stop
            showToast('Zorla Durduruldu', data.message || 'Tüm analizler zorla durduruldu, sistem restart ediliyor...', 'success');
            
            // ERSIN UI'yi temizle
            for (const [fileId, status] of fileStatuses.entries()) {
                updateFileStatus(fileId, "cancelled", 0, null, null);
            }
            resetAnalyzeButton();
            stopQueueStatusChecker();
            
            // ERSIN Loading mesajını güncelle
            updateLoadingMessage('Uygulama restart ediliyor...', 'Thread\'ler durduruluyor, lütfen bekleyin...');
            
            // ERSIN 8 saniye bekle sonra sayfa yenile (thread cleanup + restart)
            setTimeout(() => {
                console.log('[DEBUG] forceStopAnalysis: Sayfa yeniden yükleniyor (restart bekleniyor)...');
                updateLoadingMessage('Sayfa yeniden yükleniyor...', 'Sistem restart tamamlanıyor.');
                window.location.reload();
            }, 8000);
        } else {
            hideFullPageLoading();
            throw new Error(data.message || 'Force stop başarısız');
        }
    })
    .catch(error => {
        console.error('[DEBUG] forceStopAnalysis error:', error);
        hideFullPageLoading();
        showError('Zorla durdurma hatası: ' + error.message);
        resetAnalyzeButton();
    });
}

/* ERSIN Aciklama. */
function showFullPageLoading() {
    // ERSIN Mevcut loading overlay'i kaldır
    hideFullPageLoading();
    
    const loadingHTML = `
        <div id="fullPageLoading" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 9999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 18px;
        ">
            <div style="text-align: center;">
                <div class="spinner-border text-warning" role="status" style="width: 3rem; height: 3rem; margin-bottom: 20px;">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <h4 id="loadingTitle">Zorla Durdurma İşlemi</h4>
                <p id="loadingMessage">Aktif analizler durduruluyor, lütfen bekleyin...</p>
                <small style="opacity: 0.7;">Bu işlem birkaç saniye sürebilir</small>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', loadingHTML);
    document.body.style.overflow = 'hidden';  // ERSIN Scroll'u engelle
}

/* ERSIN Aciklama. */
function hideFullPageLoading() {
    const loadingElement = document.getElementById('fullPageLoading');
    if (loadingElement) {
        loadingElement.remove();
        document.body.style.overflow = '';  // ERSIN Scroll'u geri getir
    }
}

/* ERSIN Aciklama. */
function updateLoadingMessage(title, message) {
    const titleElement = document.getElementById('loadingTitle');
    const messageElement = document.getElementById('loadingMessage');
    
    if (titleElement) titleElement.textContent = title;
    if (messageElement) messageElement.textContent = message;
}

/* ERSIN Aciklama. */
export function forceStopAnalysis() {
    console.log('[DEBUG] forceStopAnalysis çağrıldı');
    
    // ERSIN Kullanıcı onayı - Bu daha ciddi bir işlem
    let userConfirmed = false;
    try {
        userConfirmed = confirm('🚨 ZORLA DURDURMA 🚨\n\n• Tüm aktif analizler zorla durdurulacak\n• Veritabanından silinecek\n• Dosyalar temizlenecek\n• Uygulama restart edilecek\n\nBu işlem geri alınamaz! Emin misiniz?');
    } catch(e) {
        console.log('[DEBUG] forceStopAnalysis: Confirm dialog hatası/engellendi');
        userConfirmed = false;  // ERSIN Force stop için kesinlikle onay gerekli
    }
    
    if (!userConfirmed) {
        console.log('[DEBUG] forceStopAnalysis: Kullanıcı işlemi iptal etti');
        return;
    }
    
    console.log('[DEBUG] forceStopAnalysis: Zorla durdurma onaylandı, API çağrısı yapılıyor...');
    
    // ERSIN Force stop bildirim göster
    showToast('Zorla Durdurma', 'Aktif analizler zorla durduruluyor...', 'warning');
    
    // ERSIN API'ye force-stop isteği gönder
    fetch('/api/queue/force-stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log('[DEBUG] forceStopAnalysis: Response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('[DEBUG] forceStopAnalysis API response:', data);
        
        if (data.force_stopped) {
            // ERSIN Başarılı force stop
            showToast('Zorla Durduruldu', data.message || 'Tüm analizler zorla durduruldu, sistem restart ediliyor...', 'success');
            
            // ERSIN UI'yi temizle
            for (const [fileId, status] of fileStatuses.entries()) {
                updateFileStatus(fileId, "cancelled", 0, null, null);
            }
            resetAnalyzeButton();
            stopQueueStatusChecker();
            
            // ERSIN 3 saniye bekle sonra sayfa yenile (uygulama restart olacak)
            setTimeout(() => {
                console.log('[DEBUG] forceStopAnalysis: Sayfa yeniden yükleniyor (restart bekleniyor)...');
                window.location.reload();
            }, 3000);
        } else {
            throw new Error(data.message || 'Force stop başarısız');
        }
    })
    .catch(error => {
        console.error('[DEBUG] forceStopAnalysis error:', error);
        showError('Zorla durdurma hatası: ' + error.message);
        resetAnalyzeButton();
    });
}

/* ERSIN Aciklama. */
function changeButtonsToStopMode() {
    // ERSIN "Analiz Et" butonunu direkt "Force Stop" butonu olarak değiştir
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-power-off me-1"></i> Analizi Durdur';
        analyzeBtn.className = 'btn btn-danger';
        
        // ERSIN 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newAnalyzeBtn = analyzeBtn.cloneNode(true);
        analyzeBtn.parentNode.replaceChild(newAnalyzeBtn, analyzeBtn);
        
        // ERSIN Sadece stopAnalysis handler'ını ekle (artık force stop)
        newAnalyzeBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[DEBUG] Force Stop butonu tıklandı!');
            stopAnalysis();
        };
        console.log('[DEBUG] Analiz Et butonu -> Force Stop butonu olarak değiştirildi');
    }
    
    // ERSIN Ana sayfadaki "Analiz Başlat" butonunu da force stop butonu olarak değiştir
    const startAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
    if (startAnalysisMainBtn) {
        startAnalysisMainBtn.innerHTML = '<i class="fas fa-power-off me-2"></i>Analizi Durdur';
        startAnalysisMainBtn.className = 'btn btn-danger btn-lg me-3';
        
        // ERSIN 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newStartAnalysisMainBtn = startAnalysisMainBtn.cloneNode(true);
        startAnalysisMainBtn.parentNode.replaceChild(newStartAnalysisMainBtn, startAnalysisMainBtn);
        
        // ERSIN Sadece stopAnalysis handler'ını ekle (artık force stop)
        newStartAnalysisMainBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[DEBUG] Ana sayfa Force Stop butonu tıklandı!');
            stopAnalysis();
        };
        console.log('[DEBUG] Ana sayfa Analiz Başlat butonu -> Force Stop butonu olarak değiştirildi');
    }
}

/* ERSIN Aciklama. */
export function resetAnalyzeButton() {
    // ERSIN "Analizi Durdur" butonunu "Analiz Et" olarak değiştir
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-play me-1"></i> Analiz Et';
        analyzeBtn.className = 'btn btn-primary';
        
        // ERSIN 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newAnalyzeBtn = analyzeBtn.cloneNode(true);
        analyzeBtn.parentNode.replaceChild(newAnalyzeBtn, analyzeBtn);
        
        // ERSIN Yeni referansı al ve orijinal event listener'ı ekle
        const newAnalyzeBtnRef = document.getElementById('analyzeBtn');
        newAnalyzeBtnRef.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (uploadedFiles.length > 0) {
                // ERSIN Analiz parametreleri modalını aç (ANLIK AYARLAR İÇİN YENİ MODAL)
                const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
                modal.show();
            }
        };
    }
    
    // ERSIN Ana sayfadaki butonu da değiştir
    const startAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
    if (startAnalysisMainBtn) {
        startAnalysisMainBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Dosya Ekle ve Analiz Et';
        startAnalysisMainBtn.className = 'btn btn-primary btn-lg me-3';
        
        // ERSIN 🔧 TÜM EVENT LISTENER'LARI TEMİZLE
        const newStartAnalysisMainBtn = startAnalysisMainBtn.cloneNode(true);
        startAnalysisMainBtn.parentNode.replaceChild(newStartAnalysisMainBtn, startAnalysisMainBtn);
        
        // ERSIN Yeni referansı al ve orijinal event listener'ı ekle
        const newStartAnalysisMainBtnRef = document.getElementById('startAnalysisMainBtn');
        newStartAnalysisMainBtnRef.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation(); 
            document.getElementById('uploadFileBtn').click(); 
        };
    }
}

// ERSIN =====================================
// ERSIN QUEUE STATUS MANAGEMENT
// ERSIN =====================================

/* ERSIN Aciklama. */
export function startQueueStatusChecker() {
    // ERSIN Önceki checker'ı temizle
    if (queueStatusChecker) {
        clearInterval(queueStatusChecker);
        queueStatusChecker = null;
    }
    
    // ERSIN Global duplicate önlemekion
    if (window.queueStatusActive) {
        console.log('⚠️ Queue status checker zaten aktif - duplikasyon önlendi');
        return;
    }
    
    window.queueStatusActive = true;
    
    queueStatusChecker = setInterval(() => {
        checkQueueStatus();
    }, QUEUE_CHECK_INTERVAL);
    
    console.log(`🔄 Queue status checker başlatıldı (${QUEUE_CHECK_INTERVAL}ms interval)`);
}

/* ERSIN Aciklama. */
export function stopQueueStatusChecker() {
    if (queueStatusChecker) {
        clearInterval(queueStatusChecker);
        queueStatusChecker = null;
    }
    
    // ERSIN Global flag'i temizle
    window.queueStatusActive = false;
    
    console.log('🛑 Queue status checker durduruldu');
}

/* ERSIN Aciklama. */
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

/* ERSIN Aciklama. */
function updateQueueStatus(response) {
    console.log('🔄 İlk yükleme - Queue status:', response);
    
    // ERSIN Overall progress bar sistemini güncelle
    updateOverallProgress(response);
    
    // ERSIN Queue display'i güncelle
    updateQueueDisplay(response);
    
    // ERSIN Buton state'ini güncelle
    updateButtonStateBasedOnQueue(response.queue_size, response.is_processing);
    
    if (response.queue_size === 0 && !response.is_processing) {
        console.log('Kuyruk boş ve işlem yok, status checker durduruluyor');
        stopQueueStatusChecker();
        
        // ERSIN Tüm analizler tamamlandığını kontrol et
        checkAllAnalysesCompleted();
    }
}

/* ERSIN Aciklama. */
function updateOverallProgress(queueData) {
    const overallProgressBar = document.getElementById('overall-progress-bar');
    const overallProgressText = document.getElementById('overall-progress-text');
    const overallProgressContainer = document.getElementById('overall-progress-container');
    
    if (!overallProgressBar || !overallProgressText) {
        console.warn('⚠️ Overall progress elements bulunamadı');
        return;
    }
    
    // ERSIN Not: uploadedFiles içine "recent/stored analyses restore" için fake kayıtlar da eklenebiliyor.
    // ERSIN Genel ilerleme sayacı sadece bu oturumda gerçekten upload edilmiş dosyaları göstermeli.
    // ERSIN Bu yüzden serverFileId'si olanları "aktif upload" kabul ediyoruz.
    const activeFiles = uploadedFiles.filter(f => f && f.serverFileId !== undefined && f.serverFileId !== null);
    const activeFileIds = new Set(activeFiles.map(f => f.id));
    const totalFiles = activeFiles.length;
    const completedFiles = getCompletedAnalysesCount(activeFileIds);
    const queueSize = queueData.queue_size || 0;
    const isProcessing = queueData.is_processing || false;
    
    // ERSIN Progress hesaplama
    let progressPercent = 0;
    if (totalFiles > 0) {
        progressPercent = Math.round((completedFiles / totalFiles) * 100);
    }
    
    // ERSIN 🎯 OVERALL PROGRESS BAR GÖRÜNÜRLÜK KONTROLÜ
    // ERSIN Sadece 2+ dosya varsa göster
    if (totalFiles >= 2) {
        if (overallProgressContainer) {
            overallProgressContainer.style.display = 'block';
        }
        
        // ERSIN Progress bar güncelle
        overallProgressBar.style.width = `${progressPercent}%`;
        overallProgressBar.setAttribute('aria-valuenow', progressPercent);
        
        // ERSIN Text güncelle
        overallProgressText.textContent = `${completedFiles}/${totalFiles} dosya`;
        
        // ERSIN Renk ve animasyon durumları
        if (completedFiles === totalFiles) {
            overallProgressText.textContent = `✅ ${completedFiles}/${totalFiles} dosya tamamlandı`;
            overallProgressBar.className = 'progress-bar bg-success';
        } else if (isProcessing || queueSize > 0) {
            overallProgressText.textContent = `⏳ ${completedFiles}/${totalFiles} dosya (${progressPercent}%)`;
            overallProgressBar.className = 'progress-bar bg-info progress-bar-striped progress-bar-animated';
        } else {
            overallProgressText.textContent = `📊 ${completedFiles}/${totalFiles} dosya (${progressPercent}%)`;
            overallProgressBar.className = 'progress-bar bg-info';
        }
        
        console.log(`✅ Overall Progress Bar: ${completedFiles}/${totalFiles} (${progressPercent}%) - Queue: ${queueSize}, Processing: ${isProcessing}`);
    } else {
        // ERSIN 1 dosya veya hiç dosya yoksa gizle
        if (overallProgressContainer) {
            overallProgressContainer.style.display = 'none';
        }
        console.log(`📝 Overall Progress Bar gizli (${totalFiles} dosya - minimum 2 gerekli)`);
    }
}

/* ERSIN Aciklama. */
function updateQueueDisplay(queueData) {
    const queueStatus = document.getElementById('queueStatus');
    if (!queueStatus) return;
    
    const queueSize = queueData.queue_size || 0;
    const isProcessing = queueData.is_processing || false;
    
    if (queueSize > 0 || isProcessing) {
        queueStatus.style.display = 'inline-flex';
        if (isProcessing) {
            queueStatus.innerHTML = `<i class="fas fa-cog fa-spin"></i> İşleniyor... (${queueSize} bekliyor)`;
        } else {
            queueStatus.innerHTML = `<i class="fas fa-hourglass-half"></i> Kuyruk: ${queueSize} dosya bekliyor`;
        }
    } else {
        queueStatus.style.display = 'none';
    }
}

/* ERSIN Aciklama. */
function updateButtonStateBasedOnQueue(queueSize, isProcessing) {
    // ERSIN Local olarak herhangi bir dosya halen işleniyor mu?
    const hasActiveLocalProcessing = Array.from(fileStatuses.values()).some(
        status => status === 'processing' || status === 'queued' || status === 'Sırada'
    );
    
    const hasActiveQueue = queueSize > 0 || isProcessing || hasActiveLocalProcessing;
    
    // ERSIN Mevcut buton durumunu kontrol et
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (!analyzeBtn) return;
    
    const isCurrentlyStopMode = analyzeBtn.innerHTML.includes('Analizi Durdur');
    
    console.log(`🔄 Button state check: queueSize=${queueSize}, isProcessing=${isProcessing}, hasActiveLocalProcessing=${hasActiveLocalProcessing}, hasActiveQueue=${hasActiveQueue}, isCurrentlyStopMode=${isCurrentlyStopMode}`);
    
    // ERSIN Queue aktifse ve buton henüz "Durdur" modunda değilse
    if (hasActiveQueue && !isCurrentlyStopMode) {
        console.log('📍 Queue aktif - butonu "Durdur" moduna çeviriliyor');
        changeButtonsToStopMode();
    }
    // ERSIN Hiçbir analiz yoksa "Analiz Et" moduna dön
    else if (!hasActiveQueue && isCurrentlyStopMode) {
        console.log('📍 Queue boş - butonu "Analiz Et" moduna çeviriliyor');
        resetAnalyzeButton();
    }
}

/* ERSIN Aciklama. */
function checkAllAnalysesCompleted() {
    const activeFiles = uploadedFiles.filter(f => f && f.serverFileId !== undefined && f.serverFileId !== null);
    const activeFileIds = new Set(activeFiles.map(f => f.id));
    const completedCount = getCompletedAnalysesCount(activeFileIds);
    const totalCount = activeFiles.length;
    
    if (completedCount === totalCount && totalCount > 0) {
        console.log('🎉 Tüm analizler tamamlandı!');
        showToast('Başarılı', 'Tüm analizler tamamlandı!', 'success');
        
        // ERSIN Butonları reset et
        resetAnalyzeButton();
        
        // ERSIN Loading spinner'ı gizle
        const settingsSaveLoader = document.getElementById('settingsSaveLoader');
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'none';
        }
        
        // ERSIN 🎯 Overall progress bar'ı final state'e güncelle
        updateOverallProgress({ queue_size: 0, is_processing: false });
    }
}

/* ERSIN Aciklama. */
function getCompletedAnalysesCount(activeFileIds = null) {
    let completedCount = 0;
    for (const [fileId, status] of fileStatuses.entries()) {
        if (activeFileIds && !activeFileIds.has(fileId)) {
            continue;
        }
        if (status === 'completed' || status === 'failed') {
            completedCount++;
        }
    }
    return completedCount;
}

// ERSIN =====================================
// ERSIN ANALYSIS PROGRESS HANDLING
// ERSIN =====================================

/* ERSIN Aciklama. */
export function handleAnalysisProgress(data) {
    const analysisId = data.analysis_id;
    const progress = data.progress || 0;
    const message = data.message || '';
    
    // ERSIN Analysis ID'den file ID'yi bul
    const fileId = fileAnalysisMap.get(analysisId);
    if (fileId) {
        // ERSIN Processing status tespit et
        if (data.status === 'processing' || progress > 0) {
            console.log(`[DEBUG] updateFileStatus - Processing status tespit edildi, progress: ${progress} , mesaj: ${message}`);
            
            // ERSIN Loading spinner'ı gizle (processing başladığında)
            const settingsSaveLoader = document.getElementById('settingsSaveLoader');
            if (settingsSaveLoader && settingsSaveLoader.style.display !== 'none') {
                settingsSaveLoader.style.display = 'none';
                console.log('[DEBUG] updateFileStatus: Processing status ile Loading spinner GİZLENDİ');
            }
            
            updateFileStatus(fileId, 'processing', progress, message);
        }
    }
}

/* ERSIN Aciklama. */
export function handleAnalysisCompleted(data) {
    const analysisId = data.analysis_id;
    const message = data.message || 'Analiz tamamlandı';
    const success = data.success !== false;
    
    // ERSIN Analysis ID'den file ID'yi bul
    const fileId = fileAnalysisMap.get(analysisId);
    if (fileId) {
        const status = success ? 'completed' : 'failed';
        const progress = success ? 100 : 0;
        
        updateFileStatus(fileId, status, progress, message);
        
        // ERSIN Alert timeout'u temizle
        if (window.analysisAlertTimeouts && window.analysisAlertTimeouts[fileId]) {
            clearTimeout(window.analysisAlertTimeouts[fileId]);
            delete window.analysisAlertTimeouts[fileId];
        }
        
        // ERSIN 🎯 OTOMATİK SONUÇ GÖSTERİMİ (yedek main.js'teki gibi)
        if (success) {
            try {
                console.log(`🎉 Analiz tamamlandı, sonuçlar getiriliyor: ${fileNameFromId(fileId)}`);
                getAnalysisResults(fileId, analysisId);
                
                // ERSIN 💾 localStorage'a ekle (persistent storage için)
                if (window.addAnalysisToLocalStorage) {
                    window.addAnalysisToLocalStorage(fileId, analysisId, fileNameFromId(fileId));
                }
                
            } catch (error) {
                console.error('Sonuçlar alınırken hata:', error);
                showToast('Uyarı', 'Analiz tamamlandı ama sonuçlar alınırken hata oluştu. Sayfayı yenileyin.', 'warning');
            }
        }
        
        console.log(`Analysis ${success ? 'tamamlandı' : 'başarısız'}: ${fileNameFromId(fileId)}`);
    }
}

// ERSIN =====================================
// ERSIN ANALYSIS RESULTS DISPLAY (from backup main.js)
// ERSIN =====================================

/* ERSIN Aciklama. */
export function getAnalysisResults(fileId, analysisId, isPartial = false) {
    console.log(`Analiz sonuçları alınıyor: fileId=${fileId}, analysisId=${analysisId}, partial=${isPartial}`);
    
    if (!analysisId) {
        console.error(`Analiz ID bulunamadı, fileId=${fileId}`);
        if (!isPartial) {
            showToast('Hata', `Analiz ID'si bulunamadı. Bu beklenmeyen bir durum.`, 'error');
        }
        return;
    }
    
    // ERSIN Yükleme göstergesi ekle
    const resultsList = document.getElementById('resultsList');
    if (resultsList && !isPartial) {
        const existingLoading = document.getElementById(`loading-${fileId}`);
        if (!existingLoading) {
            const loadingEl = document.createElement('div');
            loadingEl.id = `loading-${fileId}`;
            loadingEl.className = 'text-center my-3';
            loadingEl.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Yükleniyor...</span></div><p class="mt-2">Sonuçlar yükleniyor...</p>';
            resultsList.appendChild(loadingEl);
        }
    }
    
    // ERSIN 🎯 RATE LIMITING İÇİN RETRY MECHANISM
    const fetchWithRetry = async (url, retries = 3, delay = 2000) => {
        for (let i = 0; i < retries; i++) {
            try {
                const response = await fetch(url);
                if (response.status === 429) {
                    if (i < retries - 1) {
                        console.log(`⚠️ Rate limit (429) - ${delay}ms bekleyip yeniden deneniyor... (${i + 1}/${retries})`);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        delay *= 2;  // ERSIN Exponential backoff
                        continue;
                    }
                }
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            } catch (error) {
                if (i === retries - 1) throw error;
                console.log(`🔄 Fetch hatası, tekrar deneniyor: ${error.message}`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    };
    
    fetchWithRetry(`/api/analysis/${analysisId}/detailed-results`)
    .then(data => {
        // ERSIN 404 durumunda null dönebilir
        if (data === null) {
            console.log(`ℹ️ Analiz sonuçları alınamadı (404) - analiz muhtemelen temizlendi: ${analysisId}`);
            return;  // ERSIN Sessizce çık, hata gösterme
        }
        console.log(`Analiz sonuçları alındı (${analysisId}):`, data);
        
        // ERSIN String ise tekrar parse et
        if (typeof data === 'string') {
            console.log('JSON string detected, parsing again...');
            data = JSON.parse(data);
        }

        // ERSIN Backend failed/pending/cancelled için artık 200 + error payload dönebiliyor.
        // ERSIN Bu durumda UI'ı hata durumuna çek ve sonuç render etmeye çalışma.
        if (data && data.error && data.status && data.status !== 'completed' && !isPartial) {
            console.warn(`Analiz tamamlanmadı (${analysisId}) status=${data.status}:`, data);
            const loadingEl = document.getElementById(`loading-${fileId}`);
            if (loadingEl) loadingEl.remove();

            updateFileStatus(fileId, data.status === 'failed' ? 'failed' : 'queued', 0, data.error_message || data.error);
            showToast('Uyarı', `${fileNameFromId(fileId)}: ${data.error_message || data.error}`, 'warning');
            return;
        }
        
        // ERSIN Yükleme göstergesini kaldır
        const loadingEl = document.getElementById(`loading-${fileId}`);
        if (loadingEl) loadingEl.remove();
        
        if (!data) {
            throw new Error("Analiz sonuç verisi boş");
        }
        
        // ERSIN Sonuçları göster
        try {
            displayAnalysisResults(fileId, data);
        } catch (displayError) {
            console.error("Sonuçları gösterirken hata oluştu:", displayError);
            showToast('Hata', `Sonuçlar alındı fakat gösterilirken hata oluştu: ${displayError.message}`, 'error');
        }
        
        // ERSIN Sonuçlar bölümünü görünür yap
        document.getElementById('resultsSection').style.display = 'block';
        
        // ERSIN Buton durumunu reset et
        resetAnalyzeButton();
    })
    .catch(error => {
        console.error(`Analiz sonuçları alınırken hata (${analysisId}):`, error);
        
        const loadingEl = document.getElementById(`loading-${fileId}`);
        if (loadingEl) loadingEl.remove();
        
        showToast('Hata', `${fileNameFromId(fileId)} dosyası için sonuçlar alınırken hata oluştu: ${error.message}`, 'error');
        updateFileStatus(fileId, "error", 0, error.message);
    });
}

/* ERSIN Aciklama. */
function displayAnalysisResults(fileId, results) {
    console.log(`Analiz sonuçları gösteriliyor: fileId=${fileId}`, results);
    
    // ERSIN Video filename'i global olarak sakla (kategori bazlı timestamp'lar için)
    window.currentVideoFilename = results.file_filename || null;
    
    // ERSIN Sonuçlar bölümünü görünür yap
    document.getElementById('resultsSection').style.display = 'block';
    
    // ERSIN Dosya bilgisini al
    const file = uploadedFiles.find(f => f.id === fileId);
    
    if (!file) {
        console.error(`Sonuçları göstermek için dosya bulunamadı: fileId=${fileId}`);
        return;
    }
    
    // ERSIN Sonuç kartı template'ini klonla
    const template = document.getElementById('resultCardTemplate');
    if (!template) {
        console.error('resultCardTemplate bulunamadı!');
        return;
    }
    
    const resultCard = template.content.cloneNode(true);
    
    // ERSIN Benzersiz ID'ler için rastgele suffix
    const uniqueSuffix = Math.random().toString(36).substr(2, 9);
    
    // ERSIN Tab ID'lerini benzersiz yap
    const tabs = resultCard.querySelectorAll('[id$="-tab"]');
    tabs.forEach(tab => {
        const originalId = tab.id;
        const newId = `${originalId}-${uniqueSuffix}`;
        tab.id = newId;
        
        const targetId = tab.getAttribute('data-bs-target');
        if (targetId) {
            const newTargetId = `${targetId}-${uniqueSuffix}`;
            tab.setAttribute('data-bs-target', newTargetId);
            
            const targetPane = resultCard.querySelector(targetId);
            if (targetPane) {
                targetPane.id = newTargetId.substring(1);
            }
        }
    });
    
    // ERSIN 18 yaş altı kontrolü
    let hasUnder18 = false;
    if (results.age_estimations && Array.isArray(results.age_estimations) && results.age_estimations.length > 0) {
        hasUnder18 = results.age_estimations.some(item => {
            const estimatedAge = item.estimated_age || 0;
            return estimatedAge < 18;
        });
    }
    
    // ERSIN Dosya adını ayarla
    const fileNameElement = resultCard.querySelector('.result-filename');
    if (fileNameElement) {
        fileNameElement.textContent = file.name;
        
        // ERSIN 18 yaş altı uyarısı
        if (hasUnder18) {
            const warningBadge = document.createElement('span');
            warningBadge.className = 'badge bg-danger ms-2';
            warningBadge.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i> 18 yaş altı birey tespit edildi!';
            fileNameElement.appendChild(warningBadge);
        }
    }
    
    // ERSIN 18 yaş altı genel uyarısı
    if (hasUnder18) {
        const cardHeader = resultCard.querySelector('.card-header');
        if (cardHeader) {
            const warningAlert = document.createElement('div');
            warningAlert.className = 'alert alert-danger mb-3 mt-0 py-2';
            warningAlert.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i><strong>DİKKAT:</strong> Bu içerikte 18 yaşından küçük birey tespiti yapılmıştır!';
            cardHeader.parentNode.insertBefore(warningAlert, cardHeader);
        }
        
        const cardElement = resultCard.querySelector('.card');
        if (cardElement) {
            cardElement.classList.add('bg-danger-subtle', 'border-danger');
        }
    }
    
    // ERSIN Risk skorlarını göster
    const riskScoresContainer = resultCard.querySelector('.risk-scores-container');
    if (riskScoresContainer && results.overall_scores && typeof results.overall_scores === 'object' && Object.keys(results.overall_scores).length > 0) {
        console.log(`Risk skorları gösteriliyor (${file.name}):`, results.overall_scores);
        
        const infoText = document.createElement('div');
        infoText.className = 'alert alert-info mb-3';
        infoText.innerHTML = '<small><i class="fas fa-info-circle me-1"></i> Bu skorlar içeriğin tamamı için hesaplanan ortalama risk değerlerini gösterir.</small>';
        riskScoresContainer.appendChild(infoText);
        
        // ERSIN Risk skorları için progress barlar
        Object.entries(results.overall_scores).forEach(([category, score]) => {
            const scorePercentage = Math.round(score * 100);
            let badgeClass = 'bg-success';
            
            // ERSIN Safe kategorisi için ters mantık: yüksek değer = iyi (mavi), düşük değer = kötü (kırmızı)
            if (category === 'safe') {
                if (scorePercentage < 30) badgeClass = 'bg-danger';  // ERSIN Çok düşük güvenlik = Kırmızı
                else if (scorePercentage < 60) badgeClass = 'bg-warning';  // ERSIN Orta güvenlik = Sarı
                else badgeClass = 'bg-info';  // ERSIN Yüksek güvenlik = Mavi
            } else {
                // ERSIN Diğer kategoriler için normal mantık: yüksek değer = kötü (kırmızı)
                if (scorePercentage > 70) badgeClass = 'bg-danger';
                else if (scorePercentage > 40) badgeClass = 'bg-warning';
                else badgeClass = 'bg-success';
            }
            
            const scoreElement = document.createElement('div');
            scoreElement.className = 'mb-2';
            scoreElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="fw-bold">${getCategoryNameTurkish(category)}</span>
                    <span class="badge ${badgeClass}">${scorePercentage}%</span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar ${badgeClass}" style="width: ${scorePercentage}%"></div>
                </div>
            `;
            riskScoresContainer.appendChild(scoreElement);
        });
    }
    
    // ERSIN 🎯 EN YÜKSEK RİSKLİ KARE'yi main card'da göster
    displayMainHighestRiskFrame(resultCard, results, file);
    
    // ERSIN === CONSOLE DEBUG ===
    console.log('🔍 AGE ESTIMATIONS DEBUG:');
    console.log('results.age_estimations:', results.age_estimations);
    console.log('results.age_analysis:', results.age_analysis);
    console.log('results.include_age_analysis:', results.include_age_analysis);
    console.log('Full results object keys:', Object.keys(results));
    
    // ERSIN Yaş tahminlerini göster (yedek main.js'ten - detaylı versiyon)
    // ERSIN include_age_analysis true ise ama age_estimations boşsa, bilgilendirme mesajı göster
    if (results.include_age_analysis && (!results.age_estimations || results.age_estimations.length === 0) && (!results.age_analysis || results.age_analysis.length === 0)) {
        console.warn('⚠️ Yaş tahmini istenmişti ama sonuç bulunamadı - muhtemelen yüz tespit edilmedi');
        const detailsTab = resultCard.querySelector('.tab-content .tab-pane:nth-child(2)') || resultCard.querySelector('#details');
        if (detailsTab) {
            detailsTab.innerHTML += '<div class="alert alert-warning mt-3"><i class="fas fa-exclamation-triangle me-2"></i>Yaş tahmini istenmişti ancak bu dosyada tespit edilen yüz bulunamadı.</div>';
        }
    } else if ((results.age_estimations && results.age_estimations.length > 0) || 
        (results.age_analysis && results.age_analysis.length > 0)) {
        const detailsTab = resultCard.querySelector('.tab-content .tab-pane:nth-child(2)') || resultCard.querySelector('#details');
        if (detailsTab) {
            try {
                // ERSIN Backend'in döndüğü veri yapısına göre uygun değişkeni seç
                const ageData = results.age_estimations || results.age_analysis || [];
                console.log('Yaş tahmini işlenen veriler:', ageData.length, 'kayıt bulundu');

                // ERSIN En yüksek confidence'lı kaydı seç
                const faces = {};
                ageData.forEach(item => {
                    const faceId = item.person_id || item.face_id || 'unknown';
                    const confidence = item.confidence_score || item.confidence || 0;
                    if (!faces[faceId] || confidence > faces[faceId].confidence) {
                        faces[faceId] = {
                            age: item.estimated_age || 'Bilinmiyor',
                            confidence: confidence,
                            processed_image_path: item.processed_image_path || null
                        };
                    }
                });

                const faceIds = Object.keys(faces);
                const ageEstimationSection = document.createElement('div');
                ageEstimationSection.classList.add('age-estimations', 'mt-4');
                ageEstimationSection.innerHTML = `
                    <h5 class="mb-3"><i class="fas fa-user-alt me-2"></i>Yaş Tahminleri</h5>
                    <div class="alert alert-info mb-3">
                        <i class="fas fa-info-circle me-2"></i> Her tespit edilen benzersiz yüz için en yüksek güven skorlu tahmin gösterilmektedir.
                    </div>
                    <div class="row" id="ageEstimationList-${fileId}"></div>
                `;
                detailsTab.appendChild(ageEstimationSection);
                const ageEstimationList = ageEstimationSection.querySelector(`#ageEstimationList-${fileId}`);

                if (faceIds.length === 0) {
                    ageEstimationList.innerHTML = '<div class="col-12"><div class="alert alert-info">Bu dosyada tespit edilen yüz bulunmuyor.</div></div>';
                } else {
                    faceIds.forEach((faceId, index) => {
                        const face = faces[faceId];
                        console.log(`[DEBUG] Yüz kartı oluşturuluyor - Index: ${index}, FaceID: ${faceId}`);

                        const col = document.createElement('div');
                        col.className = 'col-md-6 mb-4';
                        
                        // ERSIN 18 yaş altı kontrolü
                        const isUnderAge = face.age < 18;
                        const ageClass = isUnderAge ? 'border-danger bg-danger-subtle' : '';
                        const ageWarning = isUnderAge ? 
                            `<div class="alert alert-danger mt-2 mb-0 p-2">
                                <small><i class="fas fa-exclamation-triangle me-1"></i> <strong>Dikkat:</strong> 18 yaş altında birey tespit edildi!</small>
                            </div>` : '';
                        
                        // ERSIN Görsel URL'sini oluştur (F5: getApiFilesUrl)
                        let frameUrl = '';
                        if (face.processed_image_path) {
                            const path = face.processed_image_path;
                            const cleanPath = path.startsWith('storage/processed/') ? path.substring('storage/'.length) : path;
                            frameUrl = getApiFilesUrl(path.startsWith('uploads/') ? path : cleanPath);
                            console.log("[DEBUG] İşlenmiş görsel URL'si:", frameUrl);
                            
                            col.innerHTML = `
                                <div class="card h-100 ${ageClass}">
                                    <div class="card-body">
                                        <div class="position-relative" style="height: 300px; overflow: hidden;">
                                            <img src="${frameUrl}" 
                                                 alt="Kişi ${index + 1}"
                                                 style="width: 100%; height: 100%; object-fit: contain; cursor: pointer;"
                                                 class="age-estimation-image"
                                                 onerror="this.onerror=null;this.src=(window.API_BASE||'')+'/static/img/image-not-found.svg';"
                                                 onload="console.log('[DEBUG] Görsel başarıyla yüklendi:', this.src)"
                                                 onclick="window.zoomImage && window.zoomImage(this.src, 'Yaş Tahmini - Kişi ${index + 1}')"
                                                 title="Büyütmek için tıklayın">
                                            <span class="position-absolute top-0 end-0 m-2 badge bg-info">ID: ${index + 1}</span>
                                            ${isUnderAge ? '<span class="position-absolute top-0 start-0 m-2 badge bg-danger"><i class="fas fa-exclamation-triangle me-1"></i> 18 yaş altı</span>' : ''}
                                        </div>
                                        <div class="mt-3">
                                            <h5 class="card-title mb-2">Tahmini Yaş: ${Math.round(face.age)}</h5>
                                            ${formatVideoFrameInfo(face.frame_path) && results.file_filename ? 
                                                `<p class="text-muted small mb-3 timestamp-clickable" style="cursor: pointer;" 
                                                   onclick="handleTimestampClick(event, '${face.frame_path}', '${results.file_filename}', '${formatVideoFrameInfo(face.frame_path)}')" 
                                                   title="Video timeline'ı açmak için tıklayın">
                                                   <i class="fas fa-clock me-1"></i>${formatVideoFrameInfo(face.frame_path)}
                                                   <i class="fas fa-external-link-alt ms-1" style="font-size: 0.8em;"></i>
                                                 </p>` : 
                                                (formatVideoFrameInfo(face.frame_path) ? `<p class="text-muted small mb-3"><i class="fas fa-clock me-1"></i>${formatVideoFrameInfo(face.frame_path)}</p>` : '')
                                            }
                                            <div class="mb-2">
                                                <div class="d-flex justify-content-between">
                                                    <span>Güvenilirlik:</span>
                                                    <span>${Math.round(face.confidence * 100)}%</span>
                                                </div>
                                                <div class="progress" style="height: 6px;">
                                                    <div class="progress-bar ${face.confidence > 0.7 ? 'bg-success' : 
                                                        face.confidence > 0.4 ? 'bg-warning' : 'bg-danger'}"
                                                        style="width: ${face.confidence * 100}%">
                                                    </div>
                                                </div>
                                            </div>
                                            ${ageWarning}
                                        </div>
                                    </div>
                                </div>
                            `;
                        } else {
                            console.warn("[DEBUG] İşlenmiş görsel bulunamadı - FaceID:", faceId);
                            col.innerHTML = `
                                <div class="card h-100 ${ageClass}">
                                    <div class="card-body">
                                        <div class="alert alert-warning">
                                            <i class="fas fa-exclamation-triangle me-2"></i>
                                            İşlenmiş (overlay'li) görsel bulunamadı.
                                        </div>
                                        <h5 class="card-title mb-2">Tahmini Yaş: ${Math.round(face.age)}</h5>
                                        ${formatVideoFrameInfo(face.frame_path) && results.file_filename ? 
                                            `<p class="text-muted small mb-3 timestamp-clickable" style="cursor: pointer;" 
                                               onclick="handleTimestampClick(event, '${face.frame_path}', '${results.file_filename}', '${formatVideoFrameInfo(face.frame_path)}')" 
                                               title="Video timeline'ı açmak için tıklayın">
                                               <i class="fas fa-clock me-1"></i>${formatVideoFrameInfo(face.frame_path)}
                                               <i class="fas fa-external-link-alt ms-1" style="font-size: 0.8em;"></i>
                                             </p>` : 
                                             (formatVideoFrameInfo(face.frame_path) ? `<p class="text-muted small mb-3"><i class="fas fa-clock me-1"></i>${formatVideoFrameInfo(face.frame_path)}</p>` : '')
                                        }
                                        <div class="mb-2">
                                            <div class="d-flex justify-content-between">
                                                <span>Güvenilirlik:</span>
                                                <span>${Math.round(face.confidence * 100)}%</span>
                                            </div>
                                            <div class="progress" style="height: 6px;">
                                                <div class="progress-bar ${face.confidence > 0.7 ? 'bg-success' : 
                                                    face.confidence > 0.4 ? 'bg-warning' : 'bg-danger'}"
                                                    style="width: ${face.confidence * 100}%">
                                                </div>
                                            </div>
                                        </div>
                                        ${ageWarning}
                                    </div>
                                </div>
                            `;
                        }
                        ageEstimationList.appendChild(col);
                    });
                }
            } catch (error) {
                console.error("Yaş tahminleri gösterilirken hata:", error);
                detailsTab.innerHTML += `<div class="alert alert-danger mb-4">Yaş tahminleri işlenirken hata oluştu: ${error.message}</div>`;
            }
        }
        
        // ERSIN 🎯 FEEDBACK TAB'ında yaş ve içerik geri bildirimi göster
        const feedbackTab = resultCard.querySelector('.tab-content .tab-pane:nth-child(3)') || resultCard.querySelector('#feedback');
        if (feedbackTab) {
            displayUnifiedFeedbackForm(feedbackTab, results);
        }
    } else if (results.include_age_analysis) {
        const detailsTab = resultCard.querySelector('.tab-content .tab-pane:nth-child(2)') || resultCard.querySelector('#details');
        if (detailsTab) {
            detailsTab.innerHTML += '<div class="alert alert-info mt-3">Bu dosya için yaş tahmini bulunmuyor.</div>';
        }
    }
    
    // ERSIN 🔧 FEEDBACK FORM HER DURUMDA GÖSTERİLMELİ
    const feedbackTab = resultCard.querySelector('.tab-content .tab-pane:nth-child(3)') || resultCard.querySelector('#feedback');
    if (feedbackTab && !feedbackTab.querySelector('.unified-feedback-form')) {
        displayUnifiedFeedbackForm(feedbackTab, results);
    }
    
    // ERSIN Detaylar tabını doldur (yedek main.js'ten)
    const detailsTab = resultCard.querySelector('.tab-content .tab-pane:nth-child(2)') || resultCard.querySelector('#details');
    if (detailsTab && results.highest_risk) {
        try {
            displayHighestRiskFrame(detailsTab, results, file);
            displayHighRiskFramesByCategory(detailsTab, results, file);
        } catch (error) {
            console.error('Detaylar tab doldurulurken hata:', error);
        }
    }

    // ERSIN Sonuç kartını DOM'a ekle (DUPLICATE ÖNLEMEKION)
    const resultsList = document.getElementById('resultsList');
    if (!resultsList) {
        console.error('resultsList bulunamadı!');
        return;
    }
    
    // ERSIN 🚨 ÖNEMLİ: Eğer bu fileId için sonuç kartı zaten varsa, yenisini ekleme
    const existingCard = document.querySelector(`.result-card[data-file-id="${fileId}"]`);
    if (existingCard) {
        console.log(`${file.name} için sonuç kartı zaten var, güncelleniyor...`);
        existingCard.remove();  // ERSIN Varolan kartı kaldır
    }
    
    // ERSIN data-file-id attribute ekle
    const resultCardEl = resultCard.querySelector('.result-card') || resultCard.querySelector('.card');
    if (resultCardEl) {
        resultCardEl.setAttribute('data-file-id', fileId);
        resultsList.appendChild(resultCardEl);
    } else {
        resultsList.appendChild(resultCard);
    }
    
    console.log('✅ Analiz sonuçları başarıyla gösterildi:', file.name);
    
    // ERSIN 🎯 Overall progress bar'ı güncelle (bir analiz daha tamamlandı)
    setTimeout(() => {
        updateOverallProgress({ queue_size: 0, is_processing: false });
    }, 100);
}

/* ERSIN Aciklama. */
function displayMainHighestRiskFrame(resultCard, results, file) {
    const highestRiskContainer = resultCard.querySelector('.highest-risk-frame');
    
    // ERSIN 🔍 DEBUG: Detaylı kontrol
    console.log('🔍 DEBUG - displayMainHighestRiskFrame:');
    console.log('  highestRiskContainer:', highestRiskContainer);
    console.log('  results.highest_risk:', results.highest_risk);
    console.log('  results struktur:', Object.keys(results));
    
    if (!highestRiskContainer) {
        console.error('❌ highest-risk-frame container bulunamadı!');
        return;
    }
    
    if (!results.highest_risk) {
        console.error('❌ results.highest_risk verisi yok!');
        console.log('📄 Mevcut results keys:', Object.keys(results));
        return;
    }
    
    if (!results.highest_risk.processed_image_path) {
        console.error('❌ results.highest_risk.processed_image_path yok!');
        console.log('📄 highest_risk keys:', Object.keys(results.highest_risk));
        return;
    }

    console.log(`✅ Ana kartta en yüksek riskli kare gösteriliyor (${file.name}):`, results.highest_risk);
    
    const imgElement = highestRiskContainer.querySelector('img');
    const badgeElement = highestRiskContainer.querySelector('.risk-category-badge');
    
    if (imgElement) {
        // ERSIN Resim yolu - F5: getApiFilesUrl
        const path = results.highest_risk.processed_image_path;
        const cleanPath = path.startsWith('storage/processed/') ? path.substring('storage/'.length) : path;
        const imageSrc = getApiFilesUrl(path.startsWith('uploads/') ? path : cleanPath);
        console.log('Main card highest risk image URL:', imageSrc);
        
        imgElement.src = imageSrc;
        imgElement.style.cursor = 'pointer';
        imgElement.title = 'Büyütmek için tıklayın';
        imgElement.onclick = () => {
            if (window.zoomImage) {
                window.zoomImage(imageSrc, 'En Yüksek Riskli Kare');
            }
        };
        imgElement.onerror = () => {
            console.error('En yüksek riskli kare yüklenemedi:', imageSrc);
            imgElement.src = (window.API_BASE || '') + '/static/img/image-not-found.svg';
        };
    }
    
    if (badgeElement) {
        // ERSIN Kategori badge'i
        const category = results.highest_risk.category;
        let categoryName = getCategoryDisplayName(category);
        let badgeClass = 'bg-warning';
        
        switch (category) {
            case 'violence': badgeClass = 'bg-danger'; break;
            case 'adult_content': badgeClass = 'bg-danger'; break;
            case 'harassment': badgeClass = 'bg-warning'; break;
            case 'weapon': badgeClass = 'bg-danger'; break;
            case 'drug': badgeClass = 'bg-warning'; break;
            case 'safe': badgeClass = 'bg-success'; break;
        }
        
        // ERSIN Güç dönüşümü uygula (backend ile tutarlılık için)
        const powerValue = 1.5;
        const transformedScore = Math.pow(results.highest_risk.score, powerValue);
        badgeElement.textContent = `${categoryName}: ${(transformedScore * 100).toFixed(0)}%`;
        badgeElement.className = `position-absolute bottom-0 end-0 m-2 badge ${badgeClass}`;
    }
}

/* ERSIN Aciklama. */
function displayHighestRiskFrame(detailsTab, results, file) {
    if (!results.highest_risk || !results.highest_risk.processed_image_path) return;
    
    const container = document.createElement('div');
    container.className = 'highest-risk-section mt-4';
    container.innerHTML = `
        <h6><i class="fas fa-exclamation-triangle me-2 text-danger"></i>En Yüksek Riskli Kare</h6>
        <div class="alert alert-warning mb-3">
            <small>İçerikte tespit edilen en yüksek risk skoruna sahip kare gösterilmektedir.</small>
            ${formatVideoFrameInfo(results.highest_risk.frame) && results.file_filename ? 
                `<div class="mt-2">
                   <small class="text-dark timestamp-clickable" style="cursor: pointer;" 
                          onclick="handleTimestampClick(event, '${results.highest_risk.frame}', '${results.file_filename}', '${formatVideoFrameInfo(results.highest_risk.frame)}')" 
                          title="Video timeline'ı açmak için tıklayın">
                          <i class="fas fa-clock me-1"></i><strong>${formatVideoFrameInfo(results.highest_risk.frame)}</strong>
                          <i class="fas fa-external-link-alt ms-1" style="font-size: 0.8em;"></i>
                   </small>
                 </div>` : 
                (formatVideoFrameInfo(results.highest_risk.frame) ? `<div class="mt-2"><small class="text-dark"><i class="fas fa-clock me-1"></i><strong>${formatVideoFrameInfo(results.highest_risk.frame)}</strong></small></div>` : '')
            }
        </div>
        <div class="position-relative">
            <img src="${getApiFilesUrl((() => {
                const path = results.highest_risk.processed_image_path;
                return path.startsWith('uploads/') ? path : (path.startsWith('storage/processed/') ? path.substring('storage/'.length) : path);
            })())}" 
                 class="img-fluid rounded border" 
                 alt="En yüksek riskli kare"
                 style="max-height: 300px; cursor: pointer;"
                 onclick="window.zoomImage && window.zoomImage(this.src, 'En Yüksek Riskli Kare')"
                 onerror="this.onerror=null;this.src=(window.API_BASE||'')+'/static/img/image-not-found.svg';">
            ${results.highest_risk.category ? `
                <span class="position-absolute top-0 end-0 m-2 badge bg-danger">
                    ${getCategoryDisplayName(results.highest_risk.category)}: ${Math.round(Math.pow(results.highest_risk.score, 1.5) * 100)}%
                </span>
            ` : ''}
        </div>
    `;
    detailsTab.appendChild(container);
}

/* ERSIN Aciklama. */
function displayHighRiskFramesByCategory(detailsTab, results, file) {
    if (!results.category_specific_highest_risks_data) return;
    
    let categoryData = {};
    try {
        categoryData = JSON.parse(results.category_specific_highest_risks_data);
    } catch (e) {
        console.error('Category specific data parse hatası:', e);
        return;
    }
    
    const container = document.createElement('div');
    container.className = 'category-frames-section mt-4';
    container.innerHTML = `
        <h6><i class="fas fa-th-large me-2"></i>Kategori Bazlı Yüksek Risk Kareleri</h6>
        <div class="row" id="categoryFramesGrid"></div>
    `;
    
    const grid = container.querySelector('#categoryFramesGrid');
    const categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug'];
    
    categories.forEach(category => {
        const data = categoryData[category];
        if (!data || data.score < 0.3) return;
        
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4 mb-3';
        col.innerHTML = `
            <div class="card">
                <img src="${getApiFilesUrl(getRelativeStoragePath(data.frame_path))}" 
                     class="card-img-top" 
                     alt="${getCategoryDisplayName(category)}"
                     style="height: 200px; object-fit: cover; cursor: pointer;"
                     onclick="window.zoomImage && window.zoomImage(this.src, '${getCategoryDisplayName(category)}')"
                     onerror="this.onerror=null;this.src=(window.API_BASE||'')+'/static/img/image-not-found.svg';">
                <div class="card-body p-2">
                    <h6 class="card-title mb-1">${getCategoryDisplayName(category)}</h6>
                    <small class="text-muted d-block">Risk: ${Math.round(Math.pow(data.score, 1.5) * 100)}%</small>
                    ${formatVideoFrameInfo(data.frame_path) && results.file_filename ? 
                        `<small class="text-secondary timestamp-clickable" style="cursor: pointer;" 
                               onclick="handleTimestampClick(event, '${data.frame_path}', '${results.file_filename}', '${formatVideoFrameInfo(data.frame_path)}')" 
                               title="Video timeline'ı açmak için tıklayın">
                               <i class="fas fa-clock me-1"></i>${formatVideoFrameInfo(data.frame_path)}
                               <i class="fas fa-external-link-alt ms-1" style="font-size: 0.7em;"></i>
                        </small>` : 
                        (formatVideoFrameInfo(data.frame_path) ? `<small class="text-secondary"><i class="fas fa-clock me-1"></i>${formatVideoFrameInfo(data.frame_path)}</small>` : '')
                    }
                </div>
            </div>
        `;
        grid.appendChild(col);
    });
    
    if (grid.children.length > 0) {
        detailsTab.appendChild(container);
    }
}

/* ERSIN Aciklama. */
function getCategoryDisplayName(category) {
    const names = {
        'violence': 'Şiddet',
        'adult_content': 'Yetişkin İçeriği',
        'harassment': 'Taciz',
        'weapon': 'Silah',
        'drug': 'Madde Kullanımı',
        'safe': 'Güvenli'
    };
    return names[category] || category;
}

/* ERSIN Aciklama. */
function normalizePath(path) {
    if (!path) return '';
    return path.replace(/\\/g, '/').replace(/\/+/g, '/');
}

/* ERSIN F5 path prefix: /api/files/ URL'leri için ortak base */
function getApiFilesUrl(pathSegment) {
    if (!pathSegment) return '';
    const base = (typeof window !== 'undefined' && window.API_BASE) ? window.API_BASE : '';
    const path = String(pathSegment).replace(/^\/+/, '');
    return `${base}/api/files/${path}`;
}

/* ERSIN Aciklama. */
function getRelativeStoragePath(fullPath) {
    if (!fullPath) return '';
    const normalizedPath = fullPath.replace(/\\/g, '/');
    const storageIndex = normalizedPath.indexOf('/storage/');
    if (storageIndex !== -1) {
        return normalizedPath.substring(storageIndex + '/storage/'.length);
    }
    
    // ERSIN Eğer /storage/ bulunamazsa path analizi yap
    const filename = normalizedPath.split('/').pop() || '';
    
    // ERSIN Overlay dosyası ise processed/ prefix kullan
    if (filename.includes('_person_') || normalizedPath.includes('overlay')) {
        // ERSIN Overlay dosyaları için processed/ endpoint'i kullan
        return `processed/${filename}`;
    }
    
    // ERSIN Normal dosyalar için uploads/ prefix ekle
    if (filename && !filename.includes('/')) {
        return `uploads/${filename}`;
    }
    return filename;
}

/* ERSIN Aciklama. */
function extractFrameTimestamp(framePath) {
    if (!framePath) return '';
    
    try {
        // ERSIN Path'i normalize et - eksik backslash'leri düzelt
        let normalizedPath = framePath.replace(/([A-Z]):/g, '$1:\\'); // ERSIN Aciklama.
        normalizedPath = normalizedPath.replace(/([^\\])([A-Za-z]+)/g, '$1\\$2');  // ERSIN Eksik backslash'leri ekle
        
        // ERSIN Windows ve Unix path'lerinden dosya adını çıkar
        const fileName = normalizedPath.split(/[\/\\]/).pop();
        
        // ERSIN frame_000072_2.89.jpg formatından 2.89 kısmını çıkar
        const match = fileName.match(/frame_\d+_(\d+\.\d+)\.jpg$/);
        if (match && match[1]) {
            const seconds = parseFloat(match[1]);
            return `${seconds.toFixed(2)}s`;
        }
        
        // ERSIN Alternatif format için ikinci deneme (frame_000072_2-89.jpg gibi)
        const matchAlt = fileName.match(/frame_\d+_(\d+)-(\d+)\.jpg$/);
        if (matchAlt && matchAlt[1] && matchAlt[2]) {
            const seconds = parseFloat(`${matchAlt[1]}.${matchAlt[2]}`);
            return `${seconds.toFixed(2)}s`;
        }
        
        // ERSIN Son çare - raw path'te timestamp arama
        const rawMatch = framePath.match(/(\d+\.\d+)\.jpg$/);
        if (rawMatch && rawMatch[1]) {
            const seconds = parseFloat(rawMatch[1]);
            return `${seconds.toFixed(2)}s`;
        }
        
        return '';
    } catch (error) {
        console.warn('Frame timestamp extract hatası:', error, 'Path:', framePath);
        return '';
    }
}

/* ERSIN Aciklama. */
function formatVideoFrameInfo(framePath) {
    if (!framePath) return '';
    
    try {
        // ERSIN Frame numarasını çıkar
        const frameMatch = framePath.match(/frame_(\d+)_/);
        const frameNumber = frameMatch ? parseInt(frameMatch[1]) : null;
        
        // ERSIN Timestamp'ı çıkar
        const timestamp = extractFrameTimestamp(framePath);
        
        if (frameNumber && timestamp) {
            return `Kare #${frameNumber} (${timestamp})`;
        } else if (timestamp) {
            return `Video: ${timestamp}`;
        } else if (frameNumber) {
            return `Kare #${frameNumber}`;
        }
        
        return '';
    } catch (error) {
        console.warn('Frame info format hatası:', error);
        return '';
    }
}

/* ERSIN Aciklama. */
function displayAgeFeedback(feedbackTab, results) {
    if (!feedbackTab || !results.age_estimations || !results.age_estimations.length) {
        // ERSIN Eğer yaş tahmini yoksa mesaj göster
        const ageFeedbackContainer = feedbackTab.querySelector('.age-feedback-container');
        if (ageFeedbackContainer) {
            ageFeedbackContainer.innerHTML = '<div class="alert alert-secondary">Bu analiz için yaş tahmini geri bildirim alanı bulunmamaktadır.</div>';
        }
        return;
    }

    const ageFeedbackContainer = feedbackTab.querySelector('.age-feedback-container');
    if (!ageFeedbackContainer) {
        console.error("'.age-feedback-container' bulunamadı.");
        return;
    }
    ageFeedbackContainer.innerHTML = '';  // ERSIN Mevcut içeriği temizle

    const analysisId = results.analysis_id; 
    if (!analysisId) {
        console.error("displayAgeFeedback: results objesinde analysis_id bulunamadı!", results);
        ageFeedbackContainer.innerHTML = '<div class="alert alert-danger">Analiz ID alınamadığı için yaş geri bildirimleri gösterilemiyor.</div>';
        return;
    }

    const ageFeedbackTemplate = document.getElementById('ageFeedbackTemplate');
    if (!ageFeedbackTemplate) {
        console.error("'ageFeedbackTemplate' bulunamadı.");
        return;
    }
    
    const facesMap = new Map();
    results.age_estimations.forEach(item => {
        const personId = item.person_id || `unknown-${Date.now()}-${Math.random()}`; 
        const confidence = item.confidence_score || item.confidence || 0;
        if (!facesMap.has(personId) || confidence > facesMap.get(personId).confidence) {
            facesMap.set(personId, {
                age: item.estimated_age !== undefined && item.estimated_age !== null ? Math.round(item.estimated_age) : 'Bilinmiyor',
                confidence: confidence,
                frame_path: item.processed_image_path || item.frame_path || null, 
                face_image_src: item.face_image_path || item.processed_image_path || '/static/img/placeholder-face.png' 
            });
        }
    });

    let personCounter = 0;
    facesMap.forEach((face, personId) => {
        personCounter++;
        const templateClone = ageFeedbackTemplate.content.cloneNode(true);
        const feedbackItem = templateClone.querySelector('.age-feedback-item');
        
        const faceImageElement = feedbackItem.querySelector('.face-image');
        if (faceImageElement) {
            // ERSIN Görsel yolunu /api/files/ ile başlatacak şekilde düzelt (F5: getApiFilesUrl)
            let imgSrc = face.face_image_src;
            if (imgSrc && !imgSrc.startsWith('/api/files/') && !imgSrc.startsWith('http') && !imgSrc.startsWith('/static/')) {
                imgSrc = getApiFilesUrl(getRelativeStoragePath(imgSrc));
            }
            faceImageElement.src = imgSrc;
            faceImageElement.alt = `Kişi ${personCounter}`;
            faceImageElement.style.cursor = 'pointer';
            faceImageElement.title = 'Büyütmek için tıklayın';
            faceImageElement.onclick = () => {
                if (window.zoomImage) {
                    window.zoomImage(imgSrc, `Kişi ${personCounter}`);
                }
            };
        }
        
        const personIdElement = feedbackItem.querySelector('.person-id');
        if (personIdElement) {
            personIdElement.textContent = personCounter;
        }
        
        const estimatedAgeElement = feedbackItem.querySelector('.estimated-age');
        if (estimatedAgeElement) {
            estimatedAgeElement.textContent = face.age;
        }
        
        const correctedAgeInput = feedbackItem.querySelector('.corrected-age');
        if (correctedAgeInput) {
            // ERSIN Set data attributes on the input field
            correctedAgeInput.dataset.personId = personId;
            correctedAgeInput.dataset.analysisId = analysisId;
            correctedAgeInput.dataset.framePath = face.frame_path || '';
        }
        
        // ERSIN Individual submit button event (basit versiyon)
        const submitButton = feedbackItem.querySelector('.age-feedback-submit');
        if (submitButton) {
            submitButton.onclick = () => {
                const correctedAge = parseInt(correctedAgeInput.value);
                if (correctedAge && correctedAge > 0 && correctedAge <= 100) {
                    // ERSIN API'ye yaş feedback gönder
                    const payload = {
                        person_id: personId,
                        corrected_age: correctedAge,
                        analysis_id: analysisId,
                        frame_path: face.frame_path || ''
                    };
                    fetch('/api/feedback/age', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            if (window.showToast) {
                                window.showToast('Başarılı', 'Yaş geri bildirimi kaydedildi!', 'success');
                            }
                            correctedAgeInput.disabled = true;
                            submitButton.disabled = true;
                            submitButton.innerHTML = '<i class="fas fa-check me-1"></i> Gönderildi';
                        } else {
                            if (window.showToast) {
                                window.showToast('Hata', data.error || 'Yaş geri bildirimi kaydedilemedi.', 'error');
                            }
                        }
                    })
                    .catch(error => {
                        if (window.showToast) {
                            window.showToast('Hata', 'Sunucuya bağlanırken hata oluştu: ' + error.message, 'error');
                        }
                    });
                } else {
                    if (window.showToast) {
                        window.showToast('Hata', 'Lütfen 1-100 arasında geçerli bir yaş girin.', 'error');
                    }
                }
            };
        }
        
        ageFeedbackContainer.appendChild(feedbackItem);
    });
}

/* ERSIN Aciklama. */
function closeAllVideoPlayers() {
    const modal = document.getElementById('videoPlayerModal');
    const video = document.getElementById('timelineVideo');
    const videoSource = video?.querySelector('source');
    
    if (modal && video) {
        // ERSIN Video'yu durdur
        video.pause();
        video.currentTime = 0;
        
        // ERSIN Event listener'ları temizle
        if (video._currentTimeUpdateHandler) {
            video.removeEventListener('timeupdate', video._currentTimeUpdateHandler);
            video._currentTimeUpdateHandler = null;
        }
        
        // ERSIN Video source'u temizle
        if (videoSource) {
            videoSource.src = '';
        }
        video.load();
        
        // ERSIN Modal'ı kapat (eğer açıksa)
        const bsModal = bootstrap.Modal.getInstance(modal);
        if (bsModal) {
            bsModal.hide();
        }
        
        console.log('🛑 Tüm video player\'ları kapatıldı');
        
        // ERSIN Navbar'daki close button'u gizle
        const navCloseBtn = document.getElementById('closeVideoNavItem');
        if (navCloseBtn) {
            navCloseBtn.style.display = 'none';
        }
    }
    
    // ERSIN Gelecekte başka video player'lar da eklenirse burada kapatılabilir
    // ERSIN Örn: Picture-in-picture, fullscreen video'lar vs.
}

// ERSIN ESC tuşu ile tüm video player'ları kapat
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modal = document.getElementById('videoPlayerModal');
        const bsModal = bootstrap.Modal.getInstance(modal);
        if (bsModal && modal.classList.contains('show')) {
            event.preventDefault();
            closeAllVideoPlayers();
        }
    }
});

/* ERSIN Aciklama. */
function openVideoTimeline(framePath, videoFilename, frameInfo) {
    if (!framePath || !videoFilename) {
        console.warn('Video timeline: Eksik parametreler', { framePath, videoFilename });
        return;
    }
    
    const timestamp = extractFrameTimestamp(framePath);
    if (!timestamp) {
        console.warn('Video timeline: Timestamp çıkarılamadı', framePath);
        return;
    }
    
    // ERSIN Saniye değerini float olarak al
    const targetSeconds = parseFloat(timestamp.replace('s', ''));
    const startSeconds = Math.max(0, targetSeconds - 1);  // ERSIN 1 saniye öncesi (minimum 0)
    const endSeconds = targetSeconds + 1;  // ERSIN 1 saniye sonrası
    
    // ERSIN Modal elementlerini al
    const modal = document.getElementById('videoPlayerModal');
    const video = document.getElementById('timelineVideo');
    const videoSource = video.querySelector('source');
    const frameInfoElement = document.getElementById('videoFrameInfo');
    const startTimeElement = document.getElementById('videoStartTime');
    const targetTimeElement = document.getElementById('videoTargetTime');
    const endTimeElement = document.getElementById('videoEndTime');
    
    if (!modal || !video || !videoSource) {
        console.error('Video timeline: Modal elementler bulunamadı');
        return;
    }
    
    // ERSIN Video URL: F5 path prefix ile (window.API_BASE = request.script_root)
    const apiBase = (typeof window !== 'undefined' && window.API_BASE) ? window.API_BASE : '';
    const videoUrl = getApiFilesUrl(`uploads/${encodeURIComponent(videoFilename)}`);
    videoSource.src = videoUrl;
    video.load();  // ERSIN Video'yu yeniden yükle
    
    // ERSIN UI elementlerini güncelle
    frameInfoElement.textContent = frameInfo || `Video: ${timestamp}`;
    startTimeElement.textContent = `${startSeconds.toFixed(2)}s`;
    targetTimeElement.textContent = `${targetSeconds.toFixed(2)}s`;
    endTimeElement.textContent = `${endSeconds.toFixed(2)}s`;
    
    // ERSIN Önceki event listener'ları temizle
    video.onloadedmetadata = null;
    video.ontimeupdate = null;
    
    // ERSIN Video yüklendiğinde timeline'ı ayarla
    video.addEventListener('loadedmetadata', function onVideoLoaded() {
        try {
            // ERSIN Video süresini kontrol et
            if (targetSeconds > video.duration) {
                console.warn(`Video timeline: Target time (${targetSeconds}s) video süresinden büyük (${video.duration}s)`);
                return;
            }
            
            // ERSIN Başlangıç zamanına atla
            video.currentTime = startSeconds;
            
            // ERSIN 3 saniyelik loop için event listener (öncekini temizle)
            video.removeEventListener('timeupdate', video._currentTimeUpdateHandler);
            video._currentTimeUpdateHandler = function onTimeUpdate() {
                if (video.currentTime >= endSeconds) {
                    video.currentTime = startSeconds;  // ERSIN Loop başa dön
                }
            };
            video.addEventListener('timeupdate', video._currentTimeUpdateHandler);
            
            console.log(`📺 Video timeline açıldı: ${frameInfo} (${startSeconds}s - ${endSeconds}s)`);
            
        } catch (error) {
            console.error('Video timeline setup hatası:', error);
        }
        
        // ERSIN Event listener'ı temizle
        video.removeEventListener('loadedmetadata', onVideoLoaded);
    }, { once: true });
    
    // ERSIN Modal'ı göster
    const bsModal = new bootstrap.Modal(modal);
    
    // ERSIN Accessibility: Modal açıldığında aria-hidden kaldır
    modal.addEventListener('shown.bs.modal', () => {
        modal.removeAttribute('aria-hidden');
        
        // ERSIN Focus'u video player'a taşı (accessibility uyarısını önler)
        if (video) {
            video.focus();
        }
        
        // ERSIN Navbar'daki close button'u göster
        const navCloseBtn = document.getElementById('closeVideoNavItem');
        if (navCloseBtn) {
            navCloseBtn.style.display = 'block';
        }
    });
    
    // ERSIN Accessibility: Modal kapandığında aria-hidden ekle
    modal.addEventListener('hidden.bs.modal', () => {
        modal.setAttribute('aria-hidden', 'true');
        
        // ERSIN Video'yu durdur ve temizle
        video.pause();
        video.currentTime = 0;
        
        // ERSIN Event listener'ları temizle
        if (video._currentTimeUpdateHandler) {
            video.removeEventListener('timeupdate', video._currentTimeUpdateHandler);
            video._currentTimeUpdateHandler = null;
        }
        
        // ERSIN Video source'u temizle
        videoSource.src = '';
        video.load();  // ERSIN Video elementini temizle
        
        console.log('📺 Video timeline kapatıldı - video durduruldu ve temizlendi');
    });
    
    bsModal.show();
}

/* ERSIN Aciklama. */
function toggleVideoPlayback() {
    const video = document.getElementById('timelineVideo');
    const playIcon = document.getElementById('playPauseIcon');
    const playText = document.getElementById('playPauseText');
    
    if (!video) return;
    
    if (video.paused) {
        video.play();
        playIcon.className = 'fas fa-pause';
        playText.textContent = 'Duraklat';
    } else {
        video.pause();
        playIcon.className = 'fas fa-play';
        playText.textContent = 'Oynat';
    }
}

/* ERSIN Aciklama. */
function handleTimestampClick(event, framePath, videoFilename, frameInfo) {
    event.preventDefault();
    event.stopPropagation();
    
    if (!videoFilename) {
        console.warn('Video filename bulunamadı, video player açılamıyor');
        return;
    }
    
    console.log(`🕐 Timestamp tıklandı: ${frameInfo}`);
    openVideoTimeline(framePath, videoFilename, frameInfo);
}

// ERSIN Global olarak erişilebilir yap
window.toggleVideoPlayback = toggleVideoPlayback;
window.openVideoTimeline = openVideoTimeline;
window.handleTimestampClick = handleTimestampClick;

/* ERSIN Aciklama. */
export function exposeAnalysisManagerToWindow() {
    window.analysisManager = {
        startAnalysisForAllFiles,
        startAnalysis,
        stopAnalysis,
        resetAnalyzeButton,
        changeButtonsToStopMode,
        handleAnalysisProgress,
        handleAnalysisCompleted,
        checkAllAnalysesCompleted: checkAllAnalysesCompleted,
        getAnalysisResults,  // ERSIN Yeni eklenen
        updateOverallProgress,  // ERSIN Overall progress fonksiyonu
        updateQueueDisplay,  // ERSIN Queue display fonksiyonu
        updateButtonStateBasedOnQueue  // ERSIN BUG FIX: Buton state güncelleme fonksiyonu
    };
    
    // ERSIN Global window fonksiyonları (backward compatibility)
    window.getAnalysisResults = getAnalysisResults;
    window.updateOverallProgress = updateOverallProgress;
    window.updateQueueDisplay = updateQueueDisplay;
}

// ERSIN Initialize window exposure
exposeAnalysisManagerToWindow(); 

// ERSIN 🎯 FEEDBACK TAB'ında yaş ve içerik geri bildirimi göster
function displayUnifiedFeedbackForm(feedbackTab, results) {
    if (!feedbackTab) return;
    feedbackTab.innerHTML = '';

    // ERSIN Formu oluştur
    const form = document.createElement('form');
    form.className = 'unified-feedback-form';

    // ERSIN İçerik feedback alanları (örnek: kategori feedback)
    const categories = [
        { key: 'violence', label: 'Şiddet' },
        { key: 'adult_content', label: 'Yetişkin İçeriği' },
        { key: 'harassment', label: 'Taciz' },
        { key: 'weapon', label: 'Silah' },
        { key: 'drug', label: 'Madde Kullanımı' }
    ];
    const contentFeedbackSection = document.createElement('div');
    contentFeedbackSection.innerHTML = `<h5>İçerik Geri Bildirimi</h5>`;
    categories.forEach(cat => {
        // ERSIN Model skorunu ve tahminini al
        let score = null;
        let scoreText = '';
        let badgeClass = 'bg-secondary';
        if (results.overall_scores && results.overall_scores[cat.key] !== undefined) {
            score = Math.round(results.overall_scores[cat.key] * 100);
            scoreText = `Model: %${score}`;
            if (score >= 70) badgeClass = 'bg-danger';
            else if (score >= 40) badgeClass = 'bg-warning';
            else badgeClass = 'bg-info';
        }
        // ERSIN Model tahmini (var/yok) - 50 eşik örneği
        let prediction = '';
        let predictionClass = 'bg-info';
        if (score !== null) {
            if (score >= 50) { prediction = 'Var'; predictionClass = 'bg-success'; }
            else { prediction = 'Yok'; predictionClass = 'bg-info'; }
        }
        // ERSIN Flex row ile select ve rozetleri yan yana hizala
        contentFeedbackSection.innerHTML += `
            <div class="mb-3 d-flex align-items-center">
                <div class="flex-grow-1">
                    <label for="${cat.key}-feedback" class="form-label">${cat.label}</label>
                    <select class="form-select" id="${cat.key}-feedback" name="${cat.key}">
                        <option value="">Seçiniz</option>
                        <option value="accurate">Model doğru tespit etti</option>
                        <option value="false_negative">Model tespit etmedi, aslında VAR</option>
                        <option value="false_positive">Model yanlış tespit etti, aslında YOK</option>
                        <option value="over_estimated">Model fazla risk verdi</option>
                        <option value="under_estimated">Model az risk verdi</option>
                    </select>
                </div>
                <div class="ms-2 d-flex flex-column align-items-end">
                    ${scoreText ? `<span class="badge ${badgeClass} mb-1">${scoreText}</span>` : ''}
                    ${prediction ? `<span class="badge ${predictionClass}">Tahmin: ${prediction}</span>` : ''}
                </div>
            </div>
        `;
    });
    form.appendChild(contentFeedbackSection);

    // ERSIN === YAN YANA GRID BAŞLANGIÇ ===
    const feedbackGrid = document.createElement('div');
    feedbackGrid.className = 'row g-4';

    // ERSIN İçerik geri bildirimi sol sütun
    const contentCol = document.createElement('div');
    contentCol.className = 'col-md-6';
    contentCol.appendChild(contentFeedbackSection);
    feedbackGrid.appendChild(contentCol);

    // ERSIN Yaş geri bildirimi sağ sütun
    if (results.age_estimations && results.age_estimations.length > 0) {
        const ageCol = document.createElement('div');
        ageCol.className = 'col-md-6';
        const ageFeedbackSection = document.createElement('div');
        ageFeedbackSection.innerHTML = `<h5>Yaş Geri Bildirimi</h5>`;
        const ageGrid = document.createElement('div');
        ageGrid.className = 'row g-3';
        results.age_estimations.forEach((item, idx) => {
            const personId = item.person_id || `unknown-${idx}`;
            const faceImg = item.processed_image_path || item.face_image_path || '/static/img/placeholder-face.png';
            const card = document.createElement('div');
            card.className = 'col-12';
            card.innerHTML = `
                <div class="card h-100 shadow-sm p-2">
                    <div class="d-flex align-items-center">
                        <img src="${getApiFilesUrl(faceImg.startsWith('storage/') ? faceImg : 'processed/' + faceImg)}" alt="Kişi ${idx + 1}" class="rounded me-3" style="width: 80px; height: 80px; object-fit: cover; border: 1px solid #ccc; cursor: pointer;" onclick="window.zoomImage && window.zoomImage(this.src, 'Kişi ${idx + 1}')">
                        <div class="flex-grow-1">
                            <div class="mb-1"><strong>Kişi ${idx + 1}</strong></div>
                            <div class="mb-2 text-muted">Tahmini Yaş: <strong>${Math.round(item.estimated_age)}</strong></div>
                            <input type="number" class="form-control age-feedback-input" name="age_${personId}" min="1" max="100" placeholder="Gerçek Yaş (1-100)" data-person-id="${personId}" data-analysis-id="${results.analysis_id}" data-frame-path="${item.processed_image_path || ''}">
                        </div>
                    </div>
                </div>
            `;
            ageGrid.appendChild(card);
        });
        ageFeedbackSection.appendChild(ageGrid);
        ageCol.appendChild(ageFeedbackSection);
        feedbackGrid.appendChild(ageCol);
    }
    // ERSIN === YAN YANA GRID SONU ===
    form.appendChild(feedbackGrid);

    // ERSIN Tek bir gönderim butonu
    const submitBtn = document.createElement('button');
    submitBtn.type = 'submit';
    submitBtn.className = 'btn btn-primary mt-3';
    submitBtn.textContent = 'Geri Bildirim Gönder';
    form.appendChild(submitBtn);

    // ERSIN Submit event
    form.onsubmit = function(e) {
        e.preventDefault();
        const categoryFeedback = {
            violence: form.querySelector('#violence-feedback') ? form.querySelector('#violence-feedback').value : '',
            adult_content: form.querySelector('#adult-content-feedback') ? form.querySelector('#adult-content-feedback').value : '',
            harassment: form.querySelector('#harassment-feedback') ? form.querySelector('#harassment-feedback').value : '',
            weapon: form.querySelector('#weapon-feedback') ? form.querySelector('#weapon-feedback').value : '',
            drug: form.querySelector('#drug-feedback') ? form.querySelector('#drug-feedback').value : ''
        };
        // ERSIN Analizden kategoriye göre frame_path'leri al
        let categoryFrames = {};
        try {
            categoryFrames = JSON.parse(results.category_specific_highest_risks_data || '{}');
        } catch (e) { categoryFrames = {}; }

        // ERSIN Her kategori için ayrı feedback kaydı gönder
        let feedbackPromises = [];
        Object.keys(categoryFeedback).forEach(cat => {
            const feedbackValue = categoryFeedback[cat];
            if (feedbackValue) {
                const framePath = categoryFrames[cat]?.frame_path || '';
                const payload = {
            content_id: results.content_id || results.analysis_id,
            analysis_id: results.analysis_id,
                    category: cat,
                    feedback: feedbackValue,
                    frame_path: framePath
                };
                feedbackPromises.push(
                    fetch('/api/feedback/submit', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    })
                    .then(res => res.json())
                );
            }
        });
        // ERSIN Yaş feedback'lerini topla ve gönder (değiştirilmedi)
        const ageInputs = form.querySelectorAll('.age-feedback-input');
        const ageFeedbacks = [];
        ageInputs.forEach(input => {
            const val = input ? parseInt(input.value) : null;
            if (val && val > 0 && val <= 100) {
                ageFeedbacks.push({
                    person_id: input.dataset.personId,
                    corrected_age: val,
                    analysis_id: input.dataset.analysisId,
                    frame_path: input.dataset.framePath || ''
                });
            }
        });
        ageFeedbacks.forEach(payload => {
            feedbackPromises.push(
            fetch('/api/feedback/age', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            );
        });
        // ERSIN Tüm feedbackler gönderildikten sonra kullanıcıya bilgi ver
        Promise.all(feedbackPromises).then(results => {
            if (window.showToast) window.showToast('Başarılı', 'Geri bildirim kaydedildi!', 'success');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Gönderildi ✓';
            // ERSIN Otomatik yönlendirme kaldırıldı
        }).catch(err => {
            if (window.showToast) window.showToast('Hata', 'Sunucuya bağlanırken hata oluştu: ' + err.message, 'error');
        });
    };

    feedbackTab.appendChild(form);
}

/* ERSIN Aciklama. */
function redirectToNextPendingAnalysis() {
    fetch('/api/analysis/pending-feedback')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.pending_analyses && data.pending_analyses.length > 0) {
                // ERSIN Bir sonraki bekleyen analiz var
                const nextAnalysis = data.pending_analyses[0];
                if (window.showToast) {
                    window.showToast('Bilgi', `${data.count} analiz daha feedback bekliyor. Bir sonrakine yönlendiriliyorsunuz...`, 'info');
                }
                
                setTimeout(() => {
                    // ERSIN Bir sonraki analiz sonucuna git
                    window.location.href = `/analysis-results?fileId=${nextAnalysis.file_id}&analysisId=${nextAnalysis.analysis_id}`;
                }, 1000);
            } else {
                // ERSIN Artık bekleyen analiz yok, anasayfaya dön
                if (window.showToast) {
                    window.showToast('Tamamlandı', 'Tüm analizler için feedback verildi! Anasayfaya yönlendiriliyorsunuz.', 'success');
                }
                
                setTimeout(() => {
                    window.location.href = '/';
                }, 1000);
            }
        })
        .catch(error => {
            console.error('Bekleyen analizler alınırken hata:', error);
            // ERSIN Hata durumunda anasayfaya dön
            setTimeout(() => {
                window.location.href = '/';
            }, 1000);
        });
}