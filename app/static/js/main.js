// Global değişkenler
const API_URL = '/api';
let socket; // Global socket - tek instance
let uploadedFiles = [];
let analysisResults = {};
let currentAnalysisIds = [];
let hideLoaderTimeout; // Add this missing variable
let globalAnalysisParamsModalElement; // Global modal element

// Global flags for training
window.currentTrainingSessionId = null;
window.isModalTraining = false;

// Dosya yolu normalleştirme fonksiyonu
function normalizePath(path) {
    // Windows ve Unix yol ayraçlarını normalize et
    if (path) {
        // Önce tüm backslash'leri slash'e çevir
        return path.replace(/\\/g, '/');
    }
    return path;
}

// Globals for tracking analysis state
const fileStatuses = new Map();  // Maps fileId to status
const fileAnalysisMap = new Map();  // Maps analysisId to fileId
const cancelledAnalyses = new Set();  // Set of cancelled analysisId values
const fileErrorCounts = new Map();  // Maps fileId to error count
let totalAnalysisCount = 0;
let MAX_STATUS_CHECK_RETRIES = 5;

// window üzerinden global state paylaşımı
window.fileAnalysisMap = fileAnalysisMap;
window.uploadedFiles = uploadedFiles;

// Analiz parametreleri butonu için uyarı gösterme fonksiyonu
function handleParamsAlert(e) {
    e.preventDefault();
    e.stopPropagation();
    alert('Analiz parametrelerini değiştirmeden önce lütfen yüklenmiş dosyaları kaldırın veya analizi tamamlayın.');
}

// Manual server restart fonksiyonu (production için)
function manualServerRestart() {
    const restartBtn = document.querySelector('.restart-btn');
    if (restartBtn) {
        restartBtn.disabled = true;
        restartBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Yeniden Başlatılıyor...';
    }
    
    showToast('Bilgi', 'Sunucu yeniden başlatılıyor...', 'info');
    
    fetch('/api/restart_server', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Bilgi', 'Sunucu yeniden başlatıldı. Sayfa yenileniyor...', 'success');
            
            // 3 saniye sonra sayfayı yenile
            setTimeout(() => {
                window.location.reload();
            }, 3000);
        } else {
            showToast('Hata', 'Restart hatası: ' + (data.error || 'Bilinmeyen hata'), 'error');
            if (restartBtn) {
                restartBtn.disabled = false;
                restartBtn.innerHTML = '<i class="fas fa-sync me-2"></i>Sunucuyu Yeniden Başlat';
            }
        }
    })
    .catch(error => {
        console.error('Manual restart error:', error);
        // Restart başarılı olmuş olabilir, connection error olabilir
        showToast('Bilgi', 'Restart signal gönderildi. Sayfa yenileniyor...', 'info');
        
        setTimeout(() => {
            window.location.reload();
        }, 5000);
    });
}

// Model butonları için uyarı gösterme fonksiyonu
function handleModelAlert(e) {
    e.preventDefault();
    e.stopPropagation();
    alert('Model işlemlerini yapmadan önce lütfen yüklenmiş dosyaları kaldırın veya analizi tamamlayın.');
}

// Analiz parametreleri ve model yönetimi butonlarının durumunu güncelleme fonksiyonu (sadece yüklü dosyalara göre)
function updateAnalysisParamsButtonState() {
    // Bu fonksiyon sadece dosya ekleme/çıkarma durumlarında çağrılır
    // Kuyruk durumu kontrolü updateAnalysisParamsButtonStateWithQueue() fonksiyonunda yapılır
    updateAnalysisParamsButtonStateWithQueue(null);
}

// Analiz parametreleri ve model yönetimi butonlarının durumunu güncelleme fonksiyonu (hem yüklü dosya hem kuyruk durumuna göre)
function updateAnalysisParamsButtonStateWithQueue(queueData) {
    const analysisParamsBtn = document.getElementById('openAnalysisParamsModalBtn');
    const modelMetricsBtn = document.getElementById('modelMetricsBtn');
    const trainModelBtn = document.getElementById('trainModelBtn');
    const modelManagementBtn = document.getElementById('modelManagementBtn');

    // Yüklü dosya kontrolü
    const hasUploadedFiles = uploadedFiles.length > 0;
    
    // Kuyruk durumu kontrolü
    let hasFilesInQueue = false;
    if (queueData) {
        // Backend response formatına göre düzelt
        const data = queueData?.data || queueData;
        hasFilesInQueue = (data?.queue_size > 0) || (data?.is_processing === true);
    }
    
    // Butonlar sadece analiz devam ederken devre dışı olmalı
    // Dosya yüklendiyse VE analiz devam ediyorsa → devre dışı
    // Sadece dosya yüklendiyse → aktif (kullanıcı analiz başlatabilir)
    const shouldDisableButtons = hasFilesInQueue; // Sadece kuyruk durumuna göre

    // Debug logları (sadece durumda değişiklik varsa)
    const currentState = `files:${hasUploadedFiles}_queue:${hasFilesInQueue}_disabled:${shouldDisableButtons}`;
    if (window.lastButtonState !== currentState) {
        console.log('🔄 Buton durumu değişti:', {
            'Yüklü dosya': hasUploadedFiles,
            'Kuyrukta dosya': hasFilesInQueue, 
            'Butonlar devre dışı': shouldDisableButtons
        });
        window.lastButtonState = currentState;
    }

    if (shouldDisableButtons) {
        // Analiz Parametreleri butonu
        if (analysisParamsBtn) {
            analysisParamsBtn.classList.add('disabled');
            analysisParamsBtn.setAttribute('aria-disabled', 'true');
            analysisParamsBtn.removeAttribute('data-bs-toggle');
            analysisParamsBtn.removeAttribute('data-bs-target');
            analysisParamsBtn.removeEventListener('click', handleParamsAlert);
            analysisParamsBtn.addEventListener('click', handleParamsAlert);
        }

        // Model Metrikleri butonu
        if (modelMetricsBtn) {
            modelMetricsBtn.classList.add('disabled');
            modelMetricsBtn.setAttribute('aria-disabled', 'true');
            modelMetricsBtn.removeEventListener('click', handleModelAlert);
            modelMetricsBtn.addEventListener('click', handleModelAlert);
        }

        // Model Eğitimi butonu
        if (trainModelBtn) {
            trainModelBtn.classList.add('disabled');
            trainModelBtn.setAttribute('aria-disabled', 'true');
            trainModelBtn.removeEventListener('click', handleModelAlert);
            trainModelBtn.addEventListener('click', handleModelAlert);
        }

        // Model Yönetimi butonu
        if (modelManagementBtn) {
            modelManagementBtn.classList.add('disabled');
            modelManagementBtn.setAttribute('aria-disabled', 'true');
            modelManagementBtn.removeAttribute('data-bs-toggle');
            modelManagementBtn.removeAttribute('data-bs-target');
            modelManagementBtn.removeEventListener('click', handleModelAlert);
            modelManagementBtn.addEventListener('click', handleModelAlert);
        }
    } else {
        // Analiz Parametreleri butonu
        if (analysisParamsBtn) {
            analysisParamsBtn.classList.remove('disabled');
            analysisParamsBtn.setAttribute('aria-disabled', 'false');
            analysisParamsBtn.setAttribute('data-bs-toggle', 'modal');
            analysisParamsBtn.setAttribute('data-bs-target', '#analysisParamsModal');
            analysisParamsBtn.removeEventListener('click', handleParamsAlert);
        }

        // Model Metrikleri butonu
        if (modelMetricsBtn) {
            modelMetricsBtn.classList.remove('disabled');
            modelMetricsBtn.setAttribute('aria-disabled', 'false');
            modelMetricsBtn.removeEventListener('click', handleModelAlert);
        }

        // Model Eğitimi butonu
        if (trainModelBtn) {
            trainModelBtn.classList.remove('disabled');
            trainModelBtn.setAttribute('aria-disabled', 'false');
            trainModelBtn.removeEventListener('click', handleModelAlert);
        }

        // Model Yönetimi butonu
        if (modelManagementBtn) {
            modelManagementBtn.classList.remove('disabled');
            modelManagementBtn.setAttribute('aria-disabled', 'false');
            modelManagementBtn.setAttribute('data-bs-toggle', 'modal');
            modelManagementBtn.setAttribute('data-bs-target', '#modelManagementModal');
            modelManagementBtn.removeEventListener('click', handleModelAlert);
        }
    }
}

// Sayfa yüklendiğinde çalışacak fonksiyon
document.addEventListener('DOMContentLoaded', () => {
    const settingsSaveLoader = document.getElementById('settingsSaveLoader'); // Yükleyici elementi
    
    // Socket.io bağlantısı
    initializeSocket(settingsSaveLoader); // Yükleyici elementini initializeSocket'a parametre olarak geç
    
    // Event Listeners
    initializeEventListeners();
    
    // Eğitim butonu kurulumu
    setupTrainingButton();
    updateAnalysisParamsButtonState(); // Butonun başlangıç durumunu ayarla
    
    // Resim tıklama özelliğini etkinleştir
    addImageClickListeners();

    // --- Yeni Analiz Parametreleri Modalı (GLOBAL) için Event Listener'lar ve Fonksiyonlar ---
    globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal'); 
    if (globalAnalysisParamsModalElement) {
        const globalAnalysisParamsModal = new bootstrap.Modal(globalAnalysisParamsModalElement);
        const globalAnalysisParamsForm = document.getElementById('analysisParamsForm'); 
        const saveGlobalAnalysisParamsBtn = document.getElementById('saveAnalysisParamsBtn');
        const loadDefaultAnalysisParamsBtn = document.getElementById('loadDefaultAnalysisParamsBtn');

        // Helper function to setup slider and its value display
        function setupSliderWithValueDisplay(sliderId, valueDisplayId, defaultValue) {
            const slider = document.getElementById(sliderId);
            const valueDisplay = document.getElementById(valueDisplayId);
            if (slider && valueDisplay) {
                slider.addEventListener('input', () => {
                    valueDisplay.textContent = slider.value;
                });
                valueDisplay.textContent = slider.value || defaultValue;
            }
            return slider;
        }

        const faceDetectionConfidenceSlider = setupSliderWithValueDisplay('faceDetectionConfidence', 'faceDetectionConfidenceValue', '0.5');
        const trackingReliabilityThresholdSlider = setupSliderWithValueDisplay('trackingReliabilityThreshold', 'trackingReliabilityThresholdValue', '0.5');
        const idChangeThresholdSlider = setupSliderWithValueDisplay('idChangeThreshold', 'idChangeThresholdValue', '0.45');
        const embeddingDistanceThresholdSlider = setupSliderWithValueDisplay('embeddingDistanceThreshold', 'embeddingDistanceThresholdValue', '0.4');
        const maxLostFramesInput = document.getElementById('maxLostFrames');

        // Modal açıldığında mevcut ayarları yükle
        globalAnalysisParamsModalElement.addEventListener('show.bs.modal', function () {
            fetch('/api/get_analysis_params')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data) {
                        console.log('Fetched global params:', data);
                        if (faceDetectionConfidenceSlider && data.face_detection_confidence !== null && data.face_detection_confidence !== undefined) {
                            faceDetectionConfidenceSlider.value = data.face_detection_confidence;
                            document.getElementById('faceDetectionConfidenceValue').textContent = data.face_detection_confidence;
                        }
                        if (trackingReliabilityThresholdSlider && data.tracking_reliability_threshold !== null && data.tracking_reliability_threshold !== undefined) {
                            trackingReliabilityThresholdSlider.value = data.tracking_reliability_threshold;
                            document.getElementById('trackingReliabilityThresholdValue').textContent = data.tracking_reliability_threshold;
                        }
                        if (idChangeThresholdSlider && data.id_change_threshold !== null && data.id_change_threshold !== undefined) {
                            idChangeThresholdSlider.value = data.id_change_threshold;
                            document.getElementById('idChangeThresholdValue').textContent = data.id_change_threshold;
                        }
                        if (maxLostFramesInput && data.max_lost_frames !== null && data.max_lost_frames !== undefined) {
                            maxLostFramesInput.value = data.max_lost_frames;
                        }
                        if (embeddingDistanceThresholdSlider && data.embedding_distance_threshold !== null && data.embedding_distance_threshold !== undefined) {
                            embeddingDistanceThresholdSlider.value = data.embedding_distance_threshold;
                            document.getElementById('embeddingDistanceThresholdValue').textContent = data.embedding_distance_threshold;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching global analysis params:', error);
                    alert('Global analiz parametreleri yüklenirken bir hata oluştu: ' + error.message);
                });
        });

        // Varsayılan ayarları yükle butonu
        if (loadDefaultAnalysisParamsBtn) {
            loadDefaultAnalysisParamsBtn.addEventListener('click', function() {
                fetch('/api/get_analysis_params?use_defaults=true') 
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        if (data) {
                            console.log('Loading default global params:', data);
                            if (faceDetectionConfidenceSlider && data.face_detection_confidence !== null && data.face_detection_confidence !== undefined) {
                                faceDetectionConfidenceSlider.value = data.face_detection_confidence;
                                document.getElementById('faceDetectionConfidenceValue').textContent = data.face_detection_confidence;
                            }
                            if (trackingReliabilityThresholdSlider && data.tracking_reliability_threshold !== null && data.tracking_reliability_threshold !== undefined) {
                                trackingReliabilityThresholdSlider.value = data.tracking_reliability_threshold;
                                document.getElementById('trackingReliabilityThresholdValue').textContent = data.tracking_reliability_threshold;
                            }
                            if (idChangeThresholdSlider && data.id_change_threshold !== null && data.id_change_threshold !== undefined) {
                                idChangeThresholdSlider.value = data.id_change_threshold;
                                document.getElementById('idChangeThresholdValue').textContent = data.id_change_threshold;
                            }
                            if (maxLostFramesInput && data.max_lost_frames !== null && data.max_lost_frames !== undefined) {
                                maxLostFramesInput.value = data.max_lost_frames;
                            }
                            if (embeddingDistanceThresholdSlider && data.embedding_distance_threshold !== null && data.embedding_distance_threshold !== undefined) {
                                embeddingDistanceThresholdSlider.value = data.embedding_distance_threshold;
                                document.getElementById('embeddingDistanceThresholdValue').textContent = data.embedding_distance_threshold;
                            }
                            showToast('Bilgi', 'Varsayılan analiz parametreleri yüklendi.', 'info');
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching default global analysis params:', error);
                        alert('Varsayılan global analiz parametreleri yüklenirken bir hata oluştu: ' + error.message);
                    });
            });
        }

        // Ayarları kaydet
        if (saveGlobalAnalysisParamsBtn && globalAnalysisParamsForm) {
            saveGlobalAnalysisParamsBtn.addEventListener('click', function () {
                const formData = new FormData(globalAnalysisParamsForm);
                const params = {};
                let formIsValid = true;

                for (let [key, value] of formData.entries()) {
                    const inputElement = globalAnalysisParamsForm.elements[key];
                    if (inputElement.type === 'number' || inputElement.type === 'range') {
                        if (value === '') {
                            params[key] = null; 
                        } else {
                            const numValue = Number(value);
                            if (isNaN(numValue)) {
                                alert(`Geçersiz sayısal değer: ${inputElement.name || inputElement.id}`);
                                formIsValid = false;
                                break;
                            }
                            if (inputElement.min && numValue < Number(inputElement.min)) {
                                alert(`${inputElement.name || inputElement.id} için minimum değer ${inputElement.min} olmalıdır.`);
                                formIsValid = false;
                                break;
                            }
                            if (inputElement.max && numValue > Number(inputElement.max)) {
                                alert(`${inputElement.name || inputElement.id} için maksimum değer ${inputElement.max} olmalıdır.`);
                                formIsValid = false;
                                break;
                            }
                            params[key] = numValue;
                        }
                    } else {
                        params[key] = value;
                    }
                }

                if (!formIsValid) return;
                console.log('Saving global params:', params);

                if(settingsSaveLoader) settingsSaveLoader.style.display = 'flex'; // Yükleyiciyi göster

                fetch('/api/set_analysis_params', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(params),
                })
                .then(response => response.json().then(data => ({ status: response.status, body: data })))
                .then(({ status, body }) => {
                    if (status === 200 && body.message) {
                        if (body.restart_required) {
                            // Production mode - manual restart gerekli
                            showToast('Bilgi', body.message, 'warning');
                            
                            // Manual restart butonu göster
                            const restartBtn = document.createElement('button');
                            restartBtn.className = 'btn btn-warning mt-2';
                            restartBtn.innerHTML = '<i class="fas fa-sync me-2"></i>Sunucuyu Yeniden Başlat';
                            restartBtn.onclick = () => manualServerRestart();
                            
                            // Modal içinde restart butonu göster
                            const modalBody = document.querySelector('#analysisParamsModal .modal-body');
                            if (modalBody) {
                                // Önceki restart butonunu kaldır
                                const existingBtn = modalBody.querySelector('.restart-btn');
                                if (existingBtn) existingBtn.remove();
                                
                                restartBtn.classList.add('restart-btn');
                                modalBody.appendChild(restartBtn);
                            }
                            
                            // Loader'ı gizle
                            if(settingsSaveLoader) settingsSaveLoader.style.display = 'none';
                        } else {
                            // Development mode - auto reload
                            showToast('Bilgi', body.message + ' Sunucu yeniden başlatılıyor, lütfen bekleyin...', 'info');
                            // Yükleyici zaten gösteriliyor, WebSocket bağlantısı ve modalın kapanması bekleniyor.
                            // globalAnalysisParamsModal.hide(); // Hemen gizleme, socket connect'te gizlenecek
                        }
                    } else {
                        if(settingsSaveLoader) settingsSaveLoader.style.display = 'none';
                        if (hideLoaderTimeout) { // Add this check
                            clearTimeout(hideLoaderTimeout);
                            hideLoaderTimeout = null; // Optional: reset after clearing
                        }
                        let errorMessage = 'Global ayarlar kaydedilirken bir hata oluştu.';
                        if (body.error) errorMessage += '\nSunucu Mesajı: ' + body.error;
                        if (body.details && Array.isArray(body.details)) errorMessage += '\nDetaylar: ' + body.details.join('\n');
                        else if (body.details) errorMessage += '\nDetaylar: ' + body.details;
                        alert(errorMessage);
                        console.error('Error saving global params:', body);
                    }
                })
                .catch(error => {
                    if(settingsSaveLoader) settingsSaveLoader.style.display = 'none';
                    if (hideLoaderTimeout) { // Add this check
                        clearTimeout(hideLoaderTimeout);
                        hideLoaderTimeout = null; // Optional: reset after clearing
                    }
                    console.error('Error saving global analysis params:', error);
                    alert('Global ayarlar kaydedilirken bir ağ hatası oluştu: ' + error.message);
                });
            });
        }
    } // --- Yeni Analiz Parametreleri Modalı (GLOBAL) için SON ---

    // Modal accessibility düzeltmesi - aria-hidden attribute'unu düzelt
    const analysisModal = document.getElementById('runAnalysisSettingsModal');
    if (analysisModal) {
        analysisModal.addEventListener('show.bs.modal', function () {
            this.removeAttribute('aria-hidden');
            // Body scroll'unu engelle
            document.body.style.overflow = 'hidden';
            console.log('[DEBUG] Analysis modal açıldı, body scroll engellendi');
        });
        analysisModal.addEventListener('hide.bs.modal', function () {
            this.setAttribute('aria-hidden', 'true');
            console.log('[DEBUG] Analysis modal kapandı, aria-hidden eklendi');
        });
        analysisModal.addEventListener('hidden.bs.modal', function () {
            // Modal tamamen kapandığında backdrop'ı temizle ve scroll'u geri getir
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => {
                backdrop.remove();
                console.log('[DEBUG] Backdrop temizlendi');
            });
            document.body.style.overflow = '';
            console.log('[DEBUG] Body scroll geri getirildi');
        });
    }

    // Image zoom modal için de aynı düzeltmeyi uygula
    const imageModal = document.getElementById('imageZoomModal');
    if (imageModal) {
        imageModal.addEventListener('show.bs.modal', function () {
            this.removeAttribute('aria-hidden');
            // Body scroll'unu engelle
            document.body.style.overflow = 'hidden';
            console.log('[DEBUG] Image modal açıldı, body scroll engellendi');
        });
        imageModal.addEventListener('hide.bs.modal', function () {
            this.setAttribute('aria-hidden', 'true');
            console.log('[DEBUG] Image modal kapandı, aria-hidden eklendi');
        });
        imageModal.addEventListener('hidden.bs.modal', function () {
            // Modal tamamen kapandığında backdrop'ı temizle ve scroll'u geri getir
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => {
                backdrop.remove();
                console.log('[DEBUG] Image modal backdrop temizlendi');
            });
            document.body.style.overflow = '';
            console.log('[DEBUG] Body scroll geri getirildi');
        });
    }

    // Analiz Et butonu tıklama olayı
    document.getElementById('analyzeBtn').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (uploadedFiles.length > 0) {
            // Analiz parametreleri modalını aç (ANLIK AYARLAR İÇİN YENİ MODAL)
            const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
            modal.show();
        }
    });
});

// WebSocket sistemi başlatılıyor 
function initializeSocket(settingsSaveLoader) { 
    console.log('WebSocket sistemi aktif');
    
    // 🔥 WebSocket client instance'ını oluştur ve global variable'a ata
    if (typeof WebSocketClient !== 'undefined') {
        window.socketioClient = new WebSocketClient();
        
        // Browser background detection ve visibility API
        try {
            // Page Visibility API ile browser arka plan durumunu takip et
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    window.socketioClient.backgroundMode = true;
                    console.log('🌙 Browser arka plana geçti, background mode aktif');
                } else {
                    window.socketioClient.backgroundMode = false;
                    console.log('🌞 Browser ön plana geçti, normal mode aktif');
                }
            });
            
            // Browser notification'ları user gesture olmadan engelle
            const originalNotification = window.Notification;
            if (originalNotification) {
                window.Notification = function() {
                    // User gesture olmadan notification'ları engelle
                    return { close: () => {} };
                };
                window.Notification.permission = 'denied';
                window.Notification.requestPermission = () => Promise.resolve('denied');
            }
        } catch (error) {
            // Browser API hatalarını sessizce yakala
        }
        
        window.socketioClient.connect();
        console.log('✅ WebSocket client oluşturuldu ve bağlantı başlatıldı');
    } else {
        console.error('❌ WebSocketClient class bulunamadı!');
    }
    
    // Model değişikliği kontrolü
    if (localStorage.getItem('modelChangedReloadRequired') === 'true') {
        localStorage.removeItem('modelChangedReloadRequired');
        setTimeout(() => {
            window.location.reload();
        }, 500);
        return;
    }
    
    // Settings save loader kontrolü
    if (settingsSaveLoader && settingsSaveLoader.style.display === 'flex') {
        // Model değişikliği veya parametre değişikliği sonrası yeniden yükleme
        settingsSaveLoader.style.display = 'none';
        if (hideLoaderTimeout) {
            clearTimeout(hideLoaderTimeout);
            hideLoaderTimeout = null;
        }
        if (globalAnalysisParamsModalElement) {
            const modalInstance = bootstrap.Modal.getInstance(globalAnalysisParamsModalElement);
            if (modalInstance) {
                modalInstance.hide();
            }
        }
        showToast('Bilgi', 'Ayarlar kaydedildi ve sunucu bağlantısı yeniden kuruldu.', 'success');
    }
}

// Olay dinleyicileri
function initializeEventListeners() {
    // Dosya yükleme butonları
    document.getElementById('uploadFileBtn').addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });
    
    document.getElementById('uploadFolderBtn').addEventListener('click', () => {
        document.getElementById('folderInput').click();
    });
    
    // Dosya seçme inputları
    document.getElementById('fileInput').addEventListener('change', handleFileSelection);
    document.getElementById('folderInput').addEventListener('change', handleFileSelection);
    
    // Sürükle bırak işlemleri
    const uploadArea = document.getElementById('uploadArea');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });
    
    // Analiz Başlatma Butonu
    document.getElementById('analyzeBtn').addEventListener('click', () => {
        if (uploadedFiles.length > 0) {
            // Analiz parametreleri modalını aç (ANLIK AYARLAR İÇİN YENİ MODAL)
            const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
            modal.show();
        }
    });
    
    // Analiz Başlatma Onay Butonu (ANLIK AYARLAR MODALI İÇİNDEKİ)
    document.getElementById('startAnalysisBtn').addEventListener('click', () => {
        // Analiz parametrelerini al
        const framesPerSecondInput = document.getElementById('framesPerSecond');
        const includeAgeAnalysisInput = document.getElementById('includeAgeAnalysis');

        const framesPerSecond = framesPerSecondInput ? parseFloat(framesPerSecondInput.value) : 1;
        const includeAgeAnalysis = includeAgeAnalysisInput ? includeAgeAnalysisInput.checked : false;
        
        // Modalı kapat
        const modalElement = document.getElementById('runAnalysisSettingsModal');
        if (modalElement) {
            const modalInstance = bootstrap.Modal.getInstance(modalElement);
            if (modalInstance) {
                modalInstance.hide();
            }
        }
        
        // Tüm yüklenen dosyalar için analiz başlat
        startAnalysisForAllFiles(framesPerSecond, includeAgeAnalysis);
    });
    
    // Yapay Zeka Model Metrikleri Butonu
    document.getElementById('modelMetricsBtn').addEventListener('click', () => {
        loadModelMetrics();
        const modal = new bootstrap.Modal(document.getElementById('modelMetricsModal'));
        modal.show();
    });
    
    // Model Metrikleri modalı açıldığında Model Eğitimi tab'ında istatistikleri yükle
    const modelMetricsModal = document.getElementById('modelMetricsModal');
    if (modelMetricsModal) {
        modelMetricsModal.addEventListener('shown.bs.modal', () => {
            // Model Eğitimi tab'ı aktif hale geldiğinde istatistikleri yükle
            const trainingTab = document.getElementById('model-training-tab');
            if (trainingTab) {
                trainingTab.addEventListener('shown.bs.tab', () => {
                    refreshTrainingStats();
                });
            }
            
            // Eğitim Verisi Sayaçları tab'ı event listener
            const trainingDataTab = document.getElementById('training-data-tab');
            if (trainingDataTab) {
                trainingDataTab.addEventListener('shown.bs.tab', () => {
                    console.log('Egitim Verisi Sayaclari tabi acildi, API cagrilari yapiliyor...');
                    
                    // İçerik modeli verilerini çek
                    Promise.all([
                        fetch('/api/ensemble/stats/content').then(r => r.json()),
                        fetch('/api/model/metrics/content').then(r => r.json())
                    ])
                    .then(([ensembleData, modelData]) => {
                        const contentModelData = modelData.content || {};
                        const mergedData = {
                            ...ensembleData,
                            feedback_sources: contentModelData.feedback_sources || { manual: 0, pseudo: 0 },
                            feedback_count: contentModelData.feedback_count || 0
                        };
                        displayContentModelMetrics(mergedData);
                    })
                    .catch(error => {
                        console.error('Icerik model verisi cekilirken hata:', error);
                        displayContentModelMetrics({}); // Bos veri ile cagir
                    });
                    
                    // Yaş modeli verilerini çek
                    Promise.all([
                        fetch('/api/ensemble/stats/age').then(r => r.json()),
                        fetch('/api/model/metrics/age').then(r => r.json())
                    ])
                    .then(([ensembleData, modelData]) => {
                        const ageModelData = modelData.age || {};
                        const mergedData = {
                            ...ensembleData,
                            feedback_sources: ageModelData.feedback_sources || { manual: 0, pseudo: 0 },
                            feedback_count: ageModelData.feedback_count || 0
                        };
                        displayAgeModelMetrics(mergedData);
                    })
                    .catch(error => {
                        console.error('Yas model verisi cekilirken hata:', error);
                        displayAgeModelMetrics({}); // Bos veri ile cagir
                    });
                });
            }
        });
    }
    
    // Model türü seçildiğinde content model ayarlarını göster/gizle
    const trainingModelType = document.getElementById('trainingModelType');
    if (trainingModelType) {
        trainingModelType.addEventListener('change', function() {
            const contentSettings = document.getElementById('contentModelSettings');
            const analyzeConflictsBtn = document.getElementById('analyzeConflictsBtn');
            const conflictAnalysisInfo = document.getElementById('conflictAnalysisInfo');
            
            if (this.value === 'content') {
                // Content model seçildiğinde
                if (contentSettings) contentSettings.style.display = 'block';
                if (analyzeConflictsBtn) analyzeConflictsBtn.style.display = 'inline-block';
                if (conflictAnalysisInfo) conflictAnalysisInfo.style.display = 'none';
            } else {
                // Age model seçildiğinde
                if (contentSettings) contentSettings.style.display = 'none';
                if (analyzeConflictsBtn) analyzeConflictsBtn.style.display = 'none';
                if (conflictAnalysisInfo) conflictAnalysisInfo.style.display = 'block';
            }
        });
    }
    
   
    // Model Yönetimi Butonu
    const modelManagementBtn = document.getElementById('modelManagementBtn');
    if (modelManagementBtn) {
        // Modal element'i bir kez al
        const modalElement = document.getElementById('modelManagementModal');
        let modalInstance = null;
        
        // Event listener'ları sadece bir kez ekle
        modalElement.addEventListener('shown.bs.modal', () => {
            console.log('Model Management Modal açıldı');
            initializeModelManagementModal();
        });
        
        modalElement.addEventListener('hidden.bs.modal', () => {
            console.log('Model Management Modal kapandı');
            cleanupModelManagementModal();
            
            // Modal instance'ını temizle
            if (modalInstance) {
                modalInstance.dispose();
                modalInstance = null;
            }
            
            // Backdrop'ı zorla temizle
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());
            
            // Body'den modal class'larını temizle
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        });
        
        // Butona tıklandığında modal'ı aç
        modelManagementBtn.addEventListener('click', () => {
            // Önceki instance varsa temizle
            if (modalInstance) {
                modalInstance.dispose();
            }
            
            // Eski backdrop'ları temizle
            const backdrops = document.querySelectorAll('.modal-backdrop');
            backdrops.forEach(backdrop => backdrop.remove());
            
            // Body'yi temizle
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
            
            // Yeni modal instance oluştur ve aç
            modalInstance = new bootstrap.Modal(modalElement, {
                backdrop: true,
                keyboard: true
            });
            
            modalInstance.show();
    });
    }
    
    // Eğitim Başlatma Butonu
        
    // Model Sıfırlama Butonları - Kaldırıldı, Model Yönetimi modalında mevcut
        // Dosya kaldırma butonu için olay dinleyicisi
    document.getElementById('fileList').addEventListener('click', function(e) {
        if (e.target.closest('.remove-file-btn')) {
            const fileCard = e.target.closest('.file-card');
            removeFile(fileCard.id);
        }
    });
    
    // Uygulama başlangıcında kuyruk durumu kontrolünü başlat
    startQueueStatusChecker();
}

// Sayfa yüklendiğinde kuyruk durumunu periyodik olarak kontrol et
let mainQueueStatusInterval = null;

function startQueueStatusChecker() {
    // Önceki interval varsa temizle
    if (mainQueueStatusInterval) {
        clearInterval(mainQueueStatusInterval);
    }
    
    // İlk kontrol
    checkQueueStatus();
    
    // 30 saniyede bir kontrol et (daha az spam için)
            // WebSocket üzerinden gerçek zamanlı bilgi alınıyor
    mainQueueStatusInterval = setInterval(checkQueueStatus, 30000);
}

function stopQueueStatusChecker() {
    if (mainQueueStatusInterval) {
        clearInterval(mainQueueStatusInterval);
        mainQueueStatusInterval = null;
    }
}

// Kuyruk durumunu kontrol et
function checkQueueStatus() {
    fetch('/api/queue/status')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        updateQueueStatus(data);
    })
    .catch(error => {
        console.error('Kuyruk durumu kontrol hatası:', error);
    });
}

// Kuyruk durumunu güncelle
function updateQueueStatus(response) {
    const queueStatusElement = document.getElementById('queueStatus');
    if (!queueStatusElement) return;
    
    // Backend response formatını parse et
    const data = response?.data || response;
    const queueSize = data?.queue_size || 0;
    const isProcessing = data?.is_processing || false;
    
    if (data && (isProcessing || queueSize > 0)) {
        // Kuyruk aktif veya bekleyen dosya varsa
        const waitingCount = queueSize;
        const statusText = `Kuyruk: ${waitingCount} dosya bekliyor`;
        
        queueStatusElement.innerHTML = `
            <i class="fas fa-hourglass-half"></i> ${statusText}
        `;
        queueStatusElement.style.display = 'block';
        
        // Global ilerleme alanını da göster
        const globalProgressSection = document.getElementById('globalProgressSection');
        if (globalProgressSection) {
            globalProgressSection.style.display = 'block';
        }
        
        // Analiz durumu metnini de güncelle
        const statusElement = document.getElementById('analysisStatus');
        if (statusElement) {
            const completedCount = getCompletedAnalysesCount();
            const totalCount = fileStatuses.size;
            statusElement.textContent = `${completedCount} / ${totalCount} dosya analizi tamamlandı`;
        }
    } else {
        // Kuyruk aktif değilse ve bekleyen dosya yoksa
        queueStatusElement.style.display = 'none';
    }
    
    // Buton durumlarını güncelle (sadece gerçekten gerekli olduğunda)
    // WebSocket queue_status sürekli geldiği için, sadece kuyruk durumu değiştiğinde güncelle
    const currentQueueSize = data?.queue_size || 0;
    const currentProcessing = data?.is_processing || false;
    
    // Önceki durumla karşılaştır
    const prevQueueSize = window.lastQueueSize || 0;
    const prevProcessing = window.lastProcessing || false;
    
    if (currentQueueSize !== prevQueueSize || currentProcessing !== prevProcessing) {
        updateAnalysisParamsButtonStateWithQueue(data);
        window.lastQueueSize = currentQueueSize;
        window.lastProcessing = currentProcessing;
    }
}

// Dosya seçimini işle
function handleFileSelection(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
    
    // Input değerini sıfırla (aynı dosyayı tekrar seçebilmek için)
    event.target.value = null;
}

// Dosyaları işle
function handleFiles(files) {
    // Dosya listesi bölümünü görünür yap
    document.getElementById('fileListSection').style.display = 'block';
    
    // Dosyaları filtrele ve ekle
    Array.from(files).forEach(file => {
        // Sadece görüntü ve video dosyalarını kabul et
        if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
            addFileToList(file);
        } else {
            showToast('Hata', `${file.name} desteklenmeyen bir dosya formatı.`, 'danger');
        }
    });
    
    // Analiz butonunu aktifleştir
    document.getElementById('analyzeBtn').disabled = uploadedFiles.length === 0;
    
    // Dosyaları yüklemeye başla
    uploadFilesSequentially(0);
    updateAnalysisParamsButtonState(); // Dosya eklendiğinde buton durumunu güncelle
}

// Dosyaları sırayla yükle
function uploadFilesSequentially(index) {
    // Tüm dosyalar yüklendiyse çık
    if (index >= uploadedFiles.length) {
        console.log("Tüm dosyalar yüklendi");
        return;
    }
    
    const file = uploadedFiles[index];
    
    // Eğer dosya zaten yüklendiyse sonraki dosyaya geç
    if (file.fileId) {
        uploadFilesSequentially(index + 1);
        return;
    }
    
    // Dosya durumunu güncelle
    updateFileStatus(file.id, 'Yükleniyor', 0);
    
    // FormData nesnesi oluştur
    const formData = new FormData();
    formData.append('file', file.originalFile); // Send the original File object
    
    // Dosyayı yükle
    fetch('/api/files/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Yükleme hatası: ${response.status} ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log(`Dosya yüklendi: ${file.name}, ID: ${data.file_id}`);
        
        // Dosyaya sunucu tarafı ID ata
        file.fileId = data.file_id;
        
        // CRITICAL FIX: Server file ID ile frontend card ID mapping'ini oluştur
        if (!window.fileIdToCardId) window.fileIdToCardId = {};
        window.fileIdToCardId[String(data.file_id)] = file.id;
        console.log(`[DEBUG] fileIdToCardId mapping oluşturuldu: ${data.file_id} → ${file.id}`);
        
        // Dosya durumunu güncelle
        updateFileStatus(file.id, 'Sırada', 100, null, null);
        
        // Bir sonraki dosyayı yükle
        uploadFilesSequentially(index + 1);
    })
    .catch(error => {
        console.error(`Dosya yükleme hatası (${file.name}):`, error);
        
        // Durumu hataya çevir
        updateFileStatus(file.id, 'Hata', 0);
        
        // Hatayı göster
        showToast('Hata', `${file.name} yüklenirken hata oluştu: ${error.message}`, 'danger');
        
        // Yine de bir sonraki dosyaya geç
        uploadFilesSequentially(index + 1);
    });
}

// Dosyayı listeye ekle
function addFileToList(file) {
    const newFile = {
        id: 'file-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9),
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'pending',
        progress: 0,
        originalFile: file, // Orijinal File nesnesini sakla
        fileId: null, // Sunucudan gelen file_id, analiz başladığında atanacak
        analysisId: null // Sunucudan gelen analysis_id, analiz başladığında atanacak
    };

    // Dosya zaten listeye eklenmişse tekrar ekleme
    if (uploadedFiles.some(f => f.name === newFile.name && f.size === newFile.size)) {
        console.warn(`File ${newFile.name} already in list. Skipping.`);
        return null; // Veya uygun bir değer döndür
    }
    
    uploadedFiles.push(newFile);
    window.uploadedFiles = uploadedFiles;
    updateAnalysisParamsButtonState(); // Add this line

    const fileList = document.getElementById('fileList');
    if (!fileList) return null;

    const fileCard = createFileCard(newFile);
    fileList.appendChild(fileCard);
    
    // "Analiz Başlat" butonunu etkinleştir
    const analyzeBtn = document.getElementById('analyzeBtn');
    if(analyzeBtn) analyzeBtn.disabled = false;

    return newFile; // Eklenen dosya nesnesini döndür
}

// Dosya kartı oluştur
function createFileCard(file) {
    // Template'i klonla
    const template = document.getElementById('fileCardTemplate');
    const fileCard = template.content.cloneNode(true);
    
    // Karta dosya ID'si ata
    const cardElem = fileCard.querySelector('.file-card');
    cardElem.id = file.id;
    // Analiz için benzersiz data-analysis-id attribute'u ekle
    cardElem.setAttribute('data-analysis-id', file.analysisId || '');
    
    // Dosya adı ve boyutu ayarla
    fileCard.querySelector('.filename').textContent = file.name;
    fileCard.querySelector('.filesize').textContent = formatFileSize(file.size);
    
    // Dosya önizlemesi oluştur
    createFilePreview(file.originalFile, fileCard.querySelector('.file-preview'));
    
    // Dosya silme butonuna olay dinleyicisi ekle
    fileCard.querySelector('.remove-file-btn').addEventListener('click', () => removeFile(file.id));
    
    // Status message elementine .status-message class'ı ekle ve temiz duruma sıfırla
    const statusElem = fileCard.querySelector('.file-status-text');
    if (statusElem) {
        statusElem.classList.add('status-message');
        statusElem.textContent = 'Sırada'; // Temiz duruma sıfırla
    }
    
    // Status badge'i de temiz duruma sıfırla
    const statusBadge = fileCard.querySelector('.file-status');
    if (statusBadge) {
        statusBadge.textContent = 'Sırada';
        statusBadge.className = 'file-status bg-secondary'; // Temiz CSS sınıfları
    }
    
    // Progress bar'ı da temiz duruma sıfırla
    const progressBar = fileCard.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', '0');
        progressBar.className = 'progress-bar'; // Animasyon sınıflarını temizle
    }
    
    return cardElem;
}

// Dosya önizlemesi oluştur
function createFilePreview(file, previewElement) {
    // Dosya URL'si oluştur
    const fileURL = URL.createObjectURL(file);
    
    if (file.type.startsWith('image/')) {
        // Resim dosyası
        previewElement.src = fileURL;
        
        // Resim yüklendiğinde blob URL'i temizle
        previewElement.onload = () => {
            URL.revokeObjectURL(fileURL);
        };
    } else if (file.type.startsWith('video/')) {
        // Video dosyası
        previewElement.src = '';
        
        // Video ilk karesini almak için
        const video = document.createElement('video');
        video.src = fileURL;
        video.onloadeddata = () => {
            // Video yüklendikten sonra ilk kareyi al
            video.currentTime = 0.1;
        };
        video.onseeked = () => {
            // Canvas oluştur ve ilk kareyi çiz
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Canvas'taki resmi önizleme olarak ayarla
            previewElement.src = canvas.toDataURL();
            
            // Belleği temizle
            URL.revokeObjectURL(fileURL);
        };
        
        // Hata durumunda veya zaman aşımında blob URL'i temizle
        video.onerror = () => {
            URL.revokeObjectURL(fileURL);
        };
        
        // 5 saniye sonra hala işlenmemişse URL'i temizle (zaman aşımı güvenlik önlemi)
        setTimeout(() => {
            URL.revokeObjectURL(fileURL);
        }, 5000);
    }
}

// Dosyayı kaldır
function removeFile(fileId) {
    console.log("Attempting to remove file with ID:", fileId);
    const fileToRemove = uploadedFiles.find(f => f.id === fileId);

    if (fileToRemove) {
        // Eğer analiz devam ediyorsa ve bir analysisId varsa, iptal etmeyi dene
        if (fileToRemove.status !== 'pending' && fileToRemove.status !== 'failed' && fileToRemove.status !== 'completed' && fileToRemove.analysisId) {
            // Analiz iptal etme HTTP API ile yapılır
            console.log(`Analysis cancellation for ID: ${fileToRemove.analysisId} of file ${fileToRemove.name}`);
            cancelledAnalyses.add(fileToRemove.analysisId);
            // Sunucudan onay beklemeden UI'ı hemen güncellemek yerine,
            // sunucudan bir 'analysis_cancelled' veya 'status_update' olayı bekleyebiliriz.
            // Şimdilik, kullanıcıya işlemin başlatıldığını bildirelim.
            showToast('Bilgi', `${fileToRemove.name} için analiz iptal isteği gönderildi.`, 'info');
        }

        // Dosyayı listeden ve UI'dan kaldır
        uploadedFiles = uploadedFiles.filter(f => f.id !== fileId);
        window.uploadedFiles = uploadedFiles;
        updateAnalysisParamsButtonState(); // Add this line

        const fileCard = document.getElementById(fileId);
        if (fileCard) {
            fileCard.remove();
        }
        
        // Analiz butonunu güncelle
        document.getElementById('analyzeBtn').disabled = uploadedFiles.length === 0;
    }
}

// Dosya boyutunu formatla
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Bildirim göster
function showToast(title, message, type = 'info') {
    // Toast oluştur
    const toastEl = document.createElement('div');
    toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <strong>${title}</strong>: ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Toast container oluştur veya seç
    let toastContainer = document.querySelector('.toast-container');
    
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Toast'u container'a ekle
    toastContainer.appendChild(toastEl);
    
    // Bootstrap Toast nesnesini oluştur ve göster
    const toast = new bootstrap.Toast(toastEl, {
        animation: true,
        autohide: true,
        delay: 5000
    });
    
    toast.show();
    
    // Toast kapandığında DOM'dan kaldır
    toastEl.addEventListener('hidden.bs.toast', () => {
        toastEl.remove();
    });
}

// Tüm dosyalar için analiz başlat
function startAnalysisForAllFiles(framesPerSecond, includeAgeAnalysis) {
    // Analiz başlatılırken loading overlay'i göster
    const settingsSaveLoader = document.getElementById('settingsSaveLoader');
    console.log('[DEBUG] startAnalysisForAllFiles: settingsSaveLoader element:', settingsSaveLoader);
    if (settingsSaveLoader) {
        settingsSaveLoader.style.display = 'flex';
        settingsSaveLoader.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;"><div class="spinner-border text-primary" role="status" style="width:4rem;height:4rem;"></div><div style="margin-top:20px;font-size:1.2rem;">Modeller yükleniyor, lütfen bekleyin...</div></div>';
        console.log('[DEBUG] startAnalysisForAllFiles: Loading spinner GÖSTERILDI');
    } else {
        console.error('[DEBUG] startAnalysisForAllFiles: settingsSaveLoader element BULUNAMADI!');
    }
    
    // Analiz edilecek dosya sayısını belirle
    const filesToAnalyze = uploadedFiles.filter(file => file.fileId && !file.analysisId);
    totalAnalysisCount = filesToAnalyze.length;
    
    if (totalAnalysisCount === 0) {
        showToast('Bilgi', 'Analiz edilecek dosya bulunamadı.', 'info');
        return;
    }
    
    // Genel ilerleme çubuğunu sıfırla ve göster
    updateGlobalProgress(0, totalAnalysisCount);
    document.getElementById('globalProgressSection').style.display = 'block';
    
    // "Analiz Et" butonunu "Analizi Durdur" olarak değiştir
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-stop me-1"></i> Analizi Durdur';
        analyzeBtn.className = 'btn btn-danger';
        // Direkt onclick kullan (daha güvenilir)
        analyzeBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[DEBUG] Analizi Durdur butonu tıklandı!');
            stopAnalysis();
        };
        console.log('[DEBUG] Analiz Et butonu -> Analizi Durdur olarak değiştirildi');
    }
    
    // Ana sayfadaki "Analiz Başlat" butonunu da değiştir
    const startAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
    if (startAnalysisMainBtn) {
        startAnalysisMainBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Analizi Durdur';
        startAnalysisMainBtn.className = 'btn btn-danger btn-lg me-3';
        // Direkt onclick kullan (daha güvenilir)
        startAnalysisMainBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[DEBUG] Ana sayfa Analizi Durdur butonu tıklandı!');
            stopAnalysis();
        };
        console.log('[DEBUG] Analiz Başlat butonu -> Analizi Durdur olarak değiştirildi');
    }
    
    // Her bir dosya için analiz başlat
    filesToAnalyze.forEach(file => {
        startAnalysis(file.id, file.fileId, framesPerSecond, includeAgeAnalysis);
    });
}

// REMOVED: HTTP fallback function - all communication is via WebSocket now

// Analiz işlemini başlat
function startAnalysis(fileId, serverFileId, framesPerSecond, includeAgeAnalysis) {
    // Dosya durumunu "kuyruğa eklendi" olarak ayarla - backend'den gerçek durum gelecek
    updateFileStatus(fileId, "Sırada", 0, null, null);
    fileStatuses.set(fileId, "queued");
    
    // Analiz parametrelerini hazırla
    const analysisParams = {
        file_id: serverFileId,
        frames_per_second: framesPerSecond,
        include_age_analysis: includeAgeAnalysis
    };

    console.log("Analiz başlatılıyor:", analysisParams);

    // FileAnalysisMap'i hazırla (race condition'ı önlemek için)
    if (!window.fileAnalysisMap) window.fileAnalysisMap = new Map();
    
    // IMMEDIATE MAPPING: Server file ID ile fileId'yi hemen eşleştir
    const tempAnalysisKey = `temp_${serverFileId}`;
    window.fileAnalysisMap.set(tempAnalysisKey, fileId);
    console.log('[DEBUG] Immediate temporary mapping:', tempAnalysisKey, '→', fileId);

    // API'ye analiz isteği gönder
    fetch('/api/analysis/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(analysisParams)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(response => {
        console.log("Analysis started", response);
        
        // Analiz ID'sini doğru şekilde çıkar
        let analysisId = null;
        if (response.analysis && response.analysis.id) {
            // Yeni API formatı (response.analysis.id)
            analysisId = response.analysis.id;
        } else if (response.analysis_id) {
            // Eski API formatı (response.analysis_id)
            analysisId = response.analysis_id;
        }
        
        if (!analysisId) {
            console.error("Analiz ID alınamadı:", response);
            throw new Error("Analiz ID alınamadı");
        }
        
        // fileAnalysisMap'i mutlaka tanımla ve güncelle
        if (!window.fileAnalysisMap) window.fileAnalysisMap = new Map();
        
        // Temp mapping'i temizle (eğer henüz temp_ ile progress gelmemişse)
        const tempKey = `temp_${serverFileId}`;
        if (window.fileAnalysisMap.has(tempKey)) {
            window.fileAnalysisMap.delete(tempKey);
            console.log('[DEBUG] Temp mapping temizlendi:', tempKey);
        }
        
        window.fileAnalysisMap.set(analysisId, fileId);
        console.log('[DEBUG] fileAnalysisMap güncellendi:', analysisId, fileId, window.fileAnalysisMap);
        
        // Pending progress queue'sunu kontrol et ve uygula
        if (window.pendingProgress && window.pendingProgress.has(analysisId)) {
            const pendingUpdates = window.pendingProgress.get(analysisId);
            console.log('[DEBUG] Pending progress uygulanıyor:', pendingUpdates.length, 'adet');
            
            pendingUpdates.forEach(progressData => {
                updateFileStatus(fileId, 'processing', progressData.progress, progressData.message);
                console.log('[DEBUG] Pending progress uygulandı:', progressData.progress + '%', progressData.message);
            });
            
            // Queue'yu temizle
            window.pendingProgress.delete(analysisId);
            console.log('[DEBUG] Pending progress queue temizlendi');
        }
        
        // DOM'a da analysis-id attribute'unu set et (fallback için)
        const fileCard = document.getElementById(fileId);
        if (fileCard) {
            fileCard.dataset.analysisId = analysisId;
            console.log('[DEBUG] DOM fileCard analysis-id set edildi:', fileId, analysisId);
        }
        
        // Dosyaya analiz ID'sini ekle
        const fileIndex = uploadedFiles.findIndex(f => f.id === fileId);
        if (fileIndex !== -1) {
            uploadedFiles[fileIndex].analysisId = analysisId;
            window.uploadedFiles = uploadedFiles;
            // DOM'daki file-card'ın data-analysis-id attribute'unu güncelle
            const cardElem = document.getElementById(fileId);
            if (cardElem) {
                cardElem.setAttribute('data-analysis-id', analysisId);
                console.log('[DEBUG] file-card data-analysis-id güncellendi:', cardElem);
            } else {
                console.warn('[DEBUG] file-card bulunamadı! fileId:', fileId);
            }
        }
        
        // Hata sayacını sıfırla
        fileErrorCounts.set(fileId, 0);
        
        // 🔥 WebSocket analysis room'una katıl (HER ZAMAN)
        console.log('[DEBUG] WebSocket join kontrolleri:', {
            socketioClient: typeof socketioClient,
            connected: socketioClient ? socketioClient.connected : 'N/A',
            analysisId: analysisId
        });
        
        if (typeof socketioClient !== 'undefined' && socketioClient && socketioClient.connected) {
            console.log(`🚀 WebSocket analysis room'una katılıyor: ${analysisId}`);
            socketioClient.joinAnalysis(analysisId);
        } else {
            console.warn('⚠️ WebSocket bağlı değil, room join edilemiyor');
            // WebSocket bağlantısını tekrar dene
            if (typeof socketioClient !== 'undefined' && socketioClient && !socketioClient.connected) {
                console.log('🔄 WebSocket yeniden bağlanmayı deniyor...');
                socketioClient.connect();
                // Kısa bir gecikme sonrası tekrar dene
                setTimeout(() => {
                    if (socketioClient.connected) {
                        console.log(`🚀 WebSocket yeniden bağlandı, room'a katılıyor: ${analysisId}`);
                        socketioClient.joinAnalysis(analysisId);
                    }
                }, 1000);
            }
        }
        
        // 🔥 FIX: WebSocket timeout timer'ını global olarak saklayalım
        const timeoutId = setTimeout(() => {
            alert('WebSocket üzerinden analiz ilerleme bilgisi alınamadı! Sunucu ile gerçek zamanlı bağlantı kurulamıyor. Lütfen sayfayı yenileyin veya sistem yöneticisine başvurun.');
        }, 600000); // 10 dakika
        
        // Timer'ı dosya objesi üzerinde saklayalım ki completed olduğunda temizleyebilelim
        const file = uploadedFiles.find(f => f.id === fileId);
        if (file) {
            file.alertTimeoutId = timeoutId;
            console.log('[DEBUG] 🔥 Alert timeout set for file:', fileId, timeoutId);
        }
    })
    .catch(error => {
        console.error("Error starting analysis:", error);
        updateFileStatus(fileId, "failed", 0, null, null);
        fileStatuses.set(fileId, "failed");
        showToast('Hata', `${fileNameFromId(fileId)} dosyası için analiz başlatılamadı: ${error.message}`, 'danger');
        updateGlobalProgress();
    });
}

// Analiz durumunu kontrol et
// REMOVED: checkAnalysisStatus function - replaced with SocketIO real-time events
function checkAllAnalysesCompleted() {
    // Tüm dosya durumlarını kontrol et
    for (const [fileId, status] of fileStatuses.entries()) {
        // İptal edilmiş analizleri tamamlanmış olarak kabul et
        // analysisId'yi bulma - fileAnalysisMap'ten ters lookup gerekli
        let analysisId = null;
        for (const [aid, fid] of fileAnalysisMap.entries()) {
            if (fid === fileId) {
                analysisId = aid;
                break;
            }
        }
        if (status !== "completed" && status !== "failed" && !cancelledAnalyses.has(analysisId)) {
            return false;  // Hala işlemde olan veya başarısız olmayan analiz var
        }
    }
    return true;  // Tüm analizler tamamlandı veya başarısız oldu
}

// Tamamlanan analiz sayısını döndüren yardımcı fonksiyon
function getCompletedAnalysesCount() {
    let count = 0;
    for (const status of fileStatuses.values()) {
        if (status === "completed") {
            count++;
        }
    }
    return count;
}

// Dosya durumunu güncelle
function updateFileStatus(fileId, status, progress, message = null, error = null) {
    // Spinner'ı herhangi bir dosya analiz ediliyora geçtiğinde gizle
    if (status === 'processing') {
        const settingsSaveLoader = document.getElementById('settingsSaveLoader');
        console.log('[DEBUG] updateFileStatus - Processing status tespit edildi, progress:', progress, ', mesaj:', message);
        if (settingsSaveLoader && settingsSaveLoader.style.display !== 'none') {
            settingsSaveLoader.style.display = 'none';
            settingsSaveLoader.innerHTML = '';
            console.log('[DEBUG] updateFileStatus: Processing status ile Loading spinner GİZLENDI');
        }
    }
    // DEBUG LOG EKLE
    console.log('[DEBUG] updateFileStatus çağrıldı:', fileId, status, progress);
    const fileCard = document.getElementById(fileId);
    console.log('[DEBUG] fileCard bulundu mu?', !!fileCard, fileCard);
    if (!fileCard) return;
    
    // Durum metnini düzenle (API'den gelen İngilizce durumları Türkçe'ye çevirelim)
    let displayStatus = status;
    if (status === 'completed') {
        displayStatus = 'Tamamlandı';
    } else if (status === 'processing') {
        displayStatus = 'Analiz Ediliyor';
    } else if (status === 'failed') {
        displayStatus = 'Hata';
    } else if (status === 'queued') {
        displayStatus = 'Sırada';
    } else if (status === 'cancelled') {
        displayStatus = 'İptal Edildi';
    }
    
    // Durum metni
    const statusText = fileCard.querySelector('.file-status-text');
    console.log('[DEBUG] statusText bulundu mu?', !!statusText, statusText);
    if (statusText) {
        statusText.textContent = displayStatus;
        console.log('[DEBUG] statusText güncellendi:', displayStatus);
    } else {
        console.error('[DEBUG] statusText elementi bulunamadı!');
    }
    
    // Durum etiketi
    const statusBadge = fileCard.querySelector('.file-status');
    console.log('[DEBUG] statusBadge bulundu mu?', !!statusBadge, statusBadge);
    if (statusBadge) {
        statusBadge.textContent = displayStatus;
        console.log('[DEBUG] statusBadge güncellendi:', displayStatus);
    } else {
        console.error('[DEBUG] statusBadge elementi bulunamadı!');
    }
    
    // Status badge rengi
    statusBadge.className = 'file-status';
    
    switch (displayStatus) {
        case 'Tamamlandı':
            statusBadge.classList.add('bg-success');
            break;
        case 'Analiz Başlatıldı':
        case 'Analiz Ediliyor':
        case 'Analiz: ':  // Analiz: X/Y kare gibi durumlar
            statusBadge.classList.add('bg-primary');
            break;
        case 'Yükleniyor':
            statusBadge.classList.add('bg-info');
            break;
        case 'Sırada':
            statusBadge.classList.add('bg-secondary');
            break;
        case 'Hata':
            statusBadge.classList.add('bg-danger');
            break;
        default:
            if (status.startsWith('Analiz:')) {
                statusBadge.classList.add('bg-primary');
            } else {
                statusBadge.classList.add('bg-secondary');
            }
    }
    
    // İlerleme çubuğu
    const progressBar = fileCard.querySelector('.progress-bar');
    console.log('[DEBUG] progressBar bulundu mu?', !!progressBar, progressBar);
    if (progressBar) {
        // Progress değeri güvenli hale getirilsin
        const safeProgress = Math.max(0, Math.min(100, progress || 0));
        
        // 🔥 ÖNCEDEN width değerini kontrol et
        console.log('[DEBUG] 🔥 Progress bar BEFORE - width:', progressBar.style.width, 'computed:', window.getComputedStyle(progressBar).width);
        
        progressBar.style.width = `${safeProgress}%`;
        progressBar.setAttribute('aria-valuenow', safeProgress);
        
        // 🔥 SONRADAN width değerini kontrol et
        console.log('[DEBUG] 🔥 Progress bar AFTER - width:', progressBar.style.width, 'computed:', window.getComputedStyle(progressBar).width);
        console.log('[DEBUG] Progress bar güncellendi:', safeProgress + '%');
    }
    
    // İlerleme yüzdesini ekle
    if (progress > 0 && progress < 100) {
        progressBar.textContent = `${Math.round(progress)}%`;
    } else {
        progressBar.textContent = '';
    }
    
    // Tamamlandı veya Hata durumları için ilerleme çubuğunu güncelle
    if (displayStatus === 'Tamamlandı' || status === 'completed') {
        progressBar.style.width = '100%';
        progressBar.setAttribute('aria-valuenow', 100);
        progressBar.classList.add('bg-success');
        
        // 🔥 FIX: Alert timeout timer'ını temizle!
        const file = uploadedFiles.find(f => f.id === fileId);
        if (file && file.alertTimeoutId) {
            clearTimeout(file.alertTimeoutId);
            console.log('[DEBUG] 🔥 Alert timeout cleared for completed file:', fileId);
            delete file.alertTimeoutId;
        }
        
        // 🔥 CRITICAL: Analysis tamamlandığında buton durumlarını kontrol et!
        console.log('🎉 [updateFileStatus] Analysis completed - buton durumları kontrol ediliyor');
        setTimeout(() => {
            // Queue status'ü kontrol et ve butonları güncelle
            fetch('/api/queue/status')
            .then(response => response.json())
            .then(response => {
                console.log('🎉 [updateFileStatus] Queue status alındı:', response);
                updateAnalysisParamsButtonStateWithQueue(response.data || response);
            })
            .catch(error => {
                console.warn('Queue status kontrol hatası, butonları yine de aktif et:', error);
                // Hata durumunda yine de butonları aktif et
                updateAnalysisParamsButtonStateWithQueue({queue_size: 0, is_processing: false});
            });
        }, 500); // Kısa gecikme ile queue'nun güncellendiğinden emin ol
        
    } else if (displayStatus === 'Hata' || status === 'failed') {
        progressBar.classList.add('bg-danger');
        
        // Hata durumunda da buton durumlarını kontrol et
        setTimeout(() => {
            fetch('/api/queue/status')
            .then(response => response.json())
            .then(response => {
                updateAnalysisParamsButtonStateWithQueue(response.data || response);
            })
            .catch(() => {
                updateAnalysisParamsButtonStateWithQueue({queue_size: 0, is_processing: false});
            });
        }, 500);
        
    } else if (displayStatus === 'Analiz Ediliyor' || status === 'processing' || displayStatus.startsWith('Analiz:') || displayStatus === 'Analiz Başlatıldı') {
        // Analiz sırasında daha göze çarpan renk
        progressBar.classList.add('bg-primary');
        progressBar.classList.add('progress-bar-striped');
        progressBar.classList.add('progress-bar-animated');
    }
    
    // 🔥 CRITICAL FIX: Dosya durumu değiştiğinde global progress bar'ı güncelle
    // fileStatuses map'ine durumu kaydet (eğer henüz kaydedilmemişse)
    fileStatuses.set(fileId, status);
    
    // Global progress bar'ı güncelle
    updateGlobalProgress();
    
    console.log('[DEBUG] updateFileStatus tamamlandı - fileId:', fileId, 'status:', status, 'global progress güncellendi');
}

// Genel ilerlemeyi güncelle
function updateGlobalProgress(current, total) {
    // Global ilerleme çubuğu kontrol
    const progressBar = document.getElementById('globalProgressBar');
    if (!progressBar) return;
    
    // Eğer parametreler verilmemişse, tamamlanan analizleri say
    if (current === undefined || total === undefined) {
        let completed = getCompletedAnalysesCount();
        let totalFiles = fileStatuses.size;
        
        // Hiç dosya yoksa çık
        if (totalFiles === 0) return;
        
        current = completed;
        total = totalFiles;
    }
    
    // İlerleme yüzdesini hesapla
    const progress = Math.round((current / total) * 100);
    
    // İlerleme çubuğunu güncelle
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
    // İlerleme durumu metnini güncelle
    const statusElement = document.getElementById('analysisStatus');
    if (statusElement) {
        statusElement.textContent = `${current} / ${total} dosya analizi tamamlandı`;
    }
    
    // Tüm analizler tamamlandıysa
    if (current >= total) {
        // Tamamlandı mesajını göster
        const completedElement = document.getElementById('completedMessage');
        if (completedElement) {
            completedElement.style.display = 'block';
        }
    }
}

// Analiz sonuçlarını al - GLOBAL SCOPE
window.getAnalysisResults = function getAnalysisResults(fileId, analysisId, isPartial = false) {
    console.log(`Analiz sonuçları alınıyor: fileId=${fileId}, analysisId=${analysisId}, partial=${isPartial}`);
    
    if (!analysisId) {
        console.error(`Analiz ID bulunamadı, fileId=${fileId}`);
        if (!isPartial) { // Sadece final results için hata göster
            showToast('Hata', `Analiz ID'si bulunamadı. Bu beklenmeyen bir durum.`, 'danger');
        }
        return;
    }
    
    // Yükleme göstergesi ekleyin (sadece final results için)
    const resultsList = document.getElementById('resultsList');
    if (resultsList && !isPartial) {
        const existingLoading = document.getElementById(`loading-${fileId}`);
        if (!existingLoading) { // Zaten varsa ekleme
            const loadingEl = document.createElement('div');
            loadingEl.id = `loading-${fileId}`;
            loadingEl.className = 'text-center my-3';
            loadingEl.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Yükleniyor...</span></div><p class="mt-2">Sonuçlar yükleniyor...</p>';
            resultsList.appendChild(loadingEl);
        }
    }
    
    fetch(`/api/analysis/${analysisId}/detailed-results`)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log(`Analiz sonuçları alındı (${analysisId}):`, data);
        
        // Eğer data string ise (double-encoded JSON), tekrar parse et
        if (typeof data === 'string') {
            console.log('JSON string detected, parsing again...');
            data = JSON.parse(data);
        }
        
        // Yükleme göstergesini kaldır
        const loadingEl = document.getElementById(`loading-${fileId}`);
        if (loadingEl) loadingEl.remove();
        
        // Veri doğrulama
        if (!data) {
            throw new Error("Analiz sonuç verisi boş");
        }
        
        // Sonuçları göster
        try {
            displayAnalysisResults(fileId, data);
        } catch (displayError) {
            console.error("Sonuçları gösterirken hata oluştu:", displayError);
            showToast('Hata', `Sonuçlar alındı fakat gösterilirken hata oluştu: ${displayError.message}`, 'danger');
        }
        
        // Genel ilerlemeyi güncelle
        updateGlobalProgress();
        
        // Tüm analizlerin tamamlanıp tamamlanmadığını kontrol et
        if (checkAllAnalysesCompleted()) {
            console.log("Tüm analizler tamamlandı");
            
            // Sadece TÜM analizler tamamlandığında sonuçlar bölümünü görünür yap
            document.getElementById('resultsSection').style.display = 'block';
            
            // Tamamlandı mesajını göster
            const completedElement = document.getElementById('completedMessage');
            if (completedElement) {
                completedElement.style.display = 'block';
            }
            
            // Butonu tekrar "Analiz Et" olarak değiştir
            resetAnalyzeButton();
        }
    })
    .catch(error => {
        console.error(`Analiz sonuçları alınırken hata (${analysisId}):`, error);
        
        // Yükleme göstergesini kaldır
        const loadingEl = document.getElementById(`loading-${fileId}`);
        if (loadingEl) loadingEl.remove();
        
        // Hata mesajını göster
        showToast('Hata', `${fileNameFromId(fileId)} dosyası için sonuçlar alınırken hata oluştu: ${error.message}`, 'danger');
        
        // Dosya kartına hata durumunu yansıt
        updateFileStatus(fileId, "error", 0, error.message);
    });
}

// Analiz sonuçlarını göster
function displayAnalysisResults(fileId, results) {
    console.log(`Analiz sonuçları gösteriliyor: fileId=${fileId}`, results);
    
    // Sonuçlar bölümünü görünür yap
    document.getElementById('resultsSection').style.display = 'block';
    
    // Dosya bilgisini al
    const file = uploadedFiles.find(f => f.id === fileId);
    
    if (!file) {
        console.error(`Sonuçları göstermek için dosya bulunamadı: fileId=${fileId}`);
        return;
    }
    
    // Sonuç kartı template'ini klonla
    const template = document.getElementById('resultCardTemplate');
    if (!template) {
        console.error('resultCardTemplate bulunamadı!');
        return;
    }
    
    const resultCard = template.content.cloneNode(true);
    
    // Benzersiz ID'ler için rastgele bir son ek oluştur
    const uniqueSuffix = Math.random().toString(36).substr(2, 9);
    
    // Tab ID'lerini benzersiz yap
    const tabs = resultCard.querySelectorAll('[id$="-tab"]');
    const tabPanes = resultCard.querySelectorAll('[id$="summary"],[id$="details"],[id$="feedback"]');
    
    tabs.forEach(tab => {
        const originalId = tab.id;
        const newId = `${originalId}-${uniqueSuffix}`;
        tab.id = newId;
        
        // data-bs-target değerini güncelle
        const targetId = tab.getAttribute('data-bs-target');
        if (targetId) {
            const newTargetId = `${targetId}-${uniqueSuffix}`;
            tab.setAttribute('data-bs-target', newTargetId);
            
            // Hedef paneyi güncelle
            const targetPane = resultCard.querySelector(targetId);
            if (targetPane) {
                targetPane.id = newTargetId.substring(1); // # işaretini kaldır
            }
        }
    });
    
    // 18 yaş altında birey kontrolü
    let hasUnder18 = false;
    if (results.age_estimations && Array.isArray(results.age_estimations) && results.age_estimations.length > 0) {
        hasUnder18 = results.age_estimations.some(item => {
            const estimatedAge = item.estimated_age || 0;
            return estimatedAge < 18;
        });
    }
    
    // Kart başlığını al
    const cardHeader = resultCard.querySelector('.card-header');
    
    // 18 yaş altı tespiti varsa, başlık üstünde bir uyarı ekle
    if (hasUnder18 && cardHeader) {
        const warningAlert = document.createElement('div');
        warningAlert.className = 'alert alert-danger mb-3 mt-0 py-2';
        warningAlert.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i><strong>DİKKAT:</strong> Bu içerikte 18 yaşından küçük birey tespiti yapılmıştır!';
        cardHeader.parentNode.insertBefore(warningAlert, cardHeader);
    }
    
    // Dosya adını ayarla
    const fileNameElement = resultCard.querySelector('.result-filename');
    fileNameElement.textContent = file.name;
    
    // 18 yaş altı birey tespiti varsa, uyarı ekle ve kart stilini değiştir
    if (hasUnder18) {
        // Kart stilini değiştir - arkaplan rengini kırmızımsı yap
        const cardElement = resultCard.querySelector('.card');
        if (cardElement) {
            cardElement.classList.add('bg-danger-subtle');
            cardElement.classList.add('border-danger');
        }
        
        // Dosya adının yanına uyarı ekle
        const warningBadge = document.createElement('span');
        warningBadge.className = 'badge bg-danger ms-2';
        warningBadge.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i> 18 yaş altı birey tespit edildi!';
        fileNameElement.appendChild(warningBadge);
    }
    
    // Content ID'sini gizli alana ekle
    const contentIdInput = resultCard.querySelector('.content-id');
    if (contentIdInput) {
        contentIdInput.value = results.content_id || '';
    }
    
    // Analysis ID ve Frame Path'i geri bildirim formuna ekle (İÇERİK GERİ BİLDİRİMİ İÇİN)
    const feedbackForm = resultCard.querySelector(`#feedback-${uniqueSuffix} form`); // Geri bildirim formunu bul
    if (feedbackForm) {
        // Önce mevcut gizli inputları temizle (varsa)
        let existingAnalysisIdInput = feedbackForm.querySelector('input[name="analysis_id"]');
        if (existingAnalysisIdInput) existingAnalysisIdInput.remove();
        let existingFramePathInput = feedbackForm.querySelector('input[name="frame_path"]');
        if (existingFramePathInput) existingFramePathInput.remove();

        const analysisIdInput = document.createElement('input');
        analysisIdInput.type = 'hidden';
        analysisIdInput.name = 'analysis_id';
        analysisIdInput.value = results.analysis_id || ''; 
        feedbackForm.appendChild(analysisIdInput);

        const framePathInput = document.createElement('input');
        framePathInput.type = 'hidden';
        framePathInput.name = 'frame_path';
        
        // Resimler için orijinal dosya yolu, videolar için en yüksek riskli karenin yolu (eğer varsa)
        let determinedFramePath = results.file_path || '';
        if (results.file_type === 'video' && results.highest_risk_frame_details && results.highest_risk_frame_details.frame_path) {
            determinedFramePath = results.highest_risk_frame_details.frame_path;
        } else if (results.file_type === 'image' && results.file_path) { 
            determinedFramePath = results.file_path;
        }
        
        if (!determinedFramePath || determinedFramePath === 'undefined') {
            console.warn('determinedFramePath geçersiz:', determinedFramePath);
            determinedFramePath = '';
        }
        
        framePathInput.value = normalizePath(determinedFramePath);
        
        feedbackForm.appendChild(framePathInput);
        
        console.log('Feedback formuna eklendi: analysis_id=', analysisIdInput.value, ', frame_path=', framePathInput.value);
        console.log('[DEBUG] Full results object received by displayAnalysisResults:', JSON.stringify(results, null, 2)); // Log the whole results object
        console.log('[DEBUG] Raw category_specific_highest_risks_data from results:', results.category_specific_highest_risks_data);

        // Parse the category-specific highest risk data
        let categorySpecificHighestRisks = {};
        if (results.category_specific_highest_risks_data) {
            try {
                categorySpecificHighestRisks = JSON.parse(results.category_specific_highest_risks_data);
                console.log('[DEBUG] Parsed categorySpecificHighestRisks:', JSON.stringify(categorySpecificHighestRisks, null, 2));
            } catch (e) {
                console.error("Error parsing category_specific_highest_risks_data:", e);
                console.log('[DEBUG] Failed to parse category_specific_highest_risks_data. Raw data was:', results.category_specific_highest_risks_data);
            }
        } else {
            console.log('[DEBUG] results.category_specific_highest_risks_data is undefined, null, or empty.');
        }

        // Populate feedback form with scores and set data attributes
        const categoriesForFeedback = ['violence', 'adult_content', 'harassment', 'weapon', 'drug'];

        for (const categoryKey of categoriesForFeedback) {
            const scoreDisplayElement = feedbackForm.querySelector(`.${categoryKey.replace('_', '-')}-model-score`);
            const feedbackSelectElement = feedbackForm.querySelector(`.${categoryKey.replace('_', '-')}-feedback`);

            let modelScoreValue = null;
            console.log(`[DEBUG] Processing feedback score for category: ${categoryKey}`);
            
            if (categorySpecificHighestRisks && typeof categorySpecificHighestRisks === 'object' && categorySpecificHighestRisks.hasOwnProperty(categoryKey)) {
                const categoryData = categorySpecificHighestRisks[categoryKey];
                if (categoryData && categoryData.score !== undefined && categoryData.score !== null && categoryData.score !== -1) {
                    modelScoreValue = parseFloat(categoryData.score);
                    console.log(`[DEBUG] ${categoryKey} - Found score in categorySpecificHighestRisks: ${categoryData.score}, parsed as: ${modelScoreValue}`);
                } else {
                    console.log(`[DEBUG] ${categoryKey} - Score is undefined, null, or -1. Data for category:`, categoryData);
                }
            } else {
                 console.log(`[DEBUG] ${categoryKey} - Key not found in categorySpecificHighestRisks or categorySpecificHighestRisks is not a valid object. categorySpecificHighestRisks:`, categorySpecificHighestRisks);
            }

            if (scoreDisplayElement) {
                scoreDisplayElement.textContent = `Model Skoru: ${modelScoreValue !== null && !isNaN(modelScoreValue) ? (modelScoreValue * 100).toFixed(0) : 'N/A'}%`;
            }
            if (feedbackSelectElement) {
                feedbackSelectElement.dataset.modelScore = modelScoreValue !== null && !isNaN(modelScoreValue) ? modelScoreValue.toFixed(4) : '';
                console.log(`[DEBUG] Set data-model-score for ${categoryKey} (from highest specific risk): ${feedbackSelectElement.dataset.modelScore}`);
            }
        }
    }
    
    // Detaylar sekmesini al
    const detailsTab = resultCard.querySelector(`#details-${uniqueSuffix}`);
    
    // Risk skorlarını göster - eğer sonuçlar boş değilse
    if (!results || Object.keys(results).length === 0) {
        console.error('Analiz sonuçları boş!', results);
        showToast('Hata', 'Analiz sonuçları boş veya hatalı format!', 'danger');
        return;
    }
    
    // Risk skorlarını göster
    const riskScoresContainer = resultCard.querySelector('.risk-scores-container');
    
    if (results.overall_scores && typeof results.overall_scores === 'object' && Object.keys(results.overall_scores).length > 0) {
        console.log(`Risk skorları gösteriliyor (${file.name}):`, results.overall_scores);
        
        // Açıklama ekle
        const infoText = document.createElement('div');
        infoText.className = 'alert alert-info mb-3';
        infoText.innerHTML = '<small><i class="fas fa-info-circle me-1"></i> Bu skorlar içeriğin tamamı için hesaplanan <strong>ortalama</strong> risk değerlerini gösterir.</small>';
        riskScoresContainer.appendChild(infoText);
        
        // Şüpheli skorları tespit et
        // const suspiciousScores = detectSuspiciousScores(results);
        
        // Şüpheli skor varsa uyarı göster (BU KISIM KALDIRILDI)
        /*
        if (suspiciousScores.length > 0) {
            const warningEl = document.createElement('div');
            warningEl.className = 'alert alert-warning mb-3';
            warningEl.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Dikkat:</strong> Bazı kategorilerde skorlar beklenenden yüksek çıkmış olabilir.
                <small>(${suspiciousScores.join(', ')}) kategorilerinde değerlendirme yaparken dikkatli olunuz)</small>
            `;
            riskScoresContainer.appendChild(warningEl);
        }
        */
        
        const scores = results.overall_scores;
        
        // Skorların formatını incele
        console.log("Skorların ham değerleri:", scores);
        
        // Skorlar 0-1 aralığında geliyorsa 0-100 aralığına dönüştür
        const normalizedScores = {};
        for (const [category, score] of Object.entries(scores)) {
            // Eğer skor 0-1 aralığındaysa (yani 1'den küçükse), 100 ile çarp
            if (score <= 1.0) {
                normalizedScores[category] = score * 100;
                console.log(`${category} skoru normalize edildi: ${score} → ${normalizedScores[category]}`);
            } else {
                // Skor zaten 0-100 aralığındaysa olduğu gibi kullan
                normalizedScores[category] = score;
            }
        }
        
        // Orijinal scores değişkeni yerine normalizedScores kullan
        const scoresForDisplay = normalizedScores;
        
        // Güven skorlarını kontrol et
        const confidenceScores = results.confidence_scores || results.score_confidences || {};
        const hasConfidenceScores = Object.keys(confidenceScores).length > 0;
        
        for (const [category, score] of Object.entries(scoresForDisplay)) {
            const scoreElement = document.createElement('div');
            scoreElement.className = 'mb-2';
            
            // Kategori adını düzenle
            let categoryName = category;
            switch (category) {
                case 'violence': categoryName = 'Şiddet'; break;
                case 'adult_content': categoryName = 'Yetişkin İçeriği'; break;
                case 'harassment': categoryName = 'Taciz'; break;
                case 'weapon': categoryName = 'Silah'; break;
                case 'drug': categoryName = 'Madde Kullanımı'; break;
                case 'safe': categoryName = 'Güvenli'; break;
            }
            
            // Risk seviyesi
            let riskLevel = '';
            let riskClass = '';
            
            if (category === 'safe') {
                // Güvenli kategori için farklı risk yorumlaması (0-100 aralığı)
                if (score >= 80) { 
                    riskLevel = 'Yüksek Güven';
                    riskClass = 'risk-level-low'; // Yeşil renk
                } else if (score >= 50) { 
                    riskLevel = 'Orta Güven';
                    riskClass = 'risk-level-medium'; // Sarı renk
                } else { 
                    riskLevel = 'Düşük Güven';
                    riskClass = 'risk-level-high'; // Kırmızı renk
                }
            } else {
                // Diğer kategoriler için yeni risk seviyesi sistemi (0-100 aralığı)
                if (score < 20) {
                    riskLevel = 'Çok Düşük Risk';
                    riskClass = 'risk-level-low';
                } else if (score < 35) {
                    riskLevel = 'Düşük Risk';
                    riskClass = 'risk-level-low';
                } else if (score < 55) {
                    riskLevel = 'Belirsiz';
                    riskClass = 'risk-level-medium';
                } else if (score < 70) {
                    riskLevel = 'Orta Risk';
                    riskClass = 'risk-level-medium';
                } else if (score < 85) {
                    riskLevel = 'Yüksek Risk';
                    riskClass = 'risk-level-high';
                } else {
                    riskLevel = 'Çok Yüksek Risk';
                    riskClass = 'risk-level-high fw-bold';
                }
            }
            
            // Şüpheli skor ise işaretle
            // const isSuspicious = suspiciousScores.includes(categoryName);
            
            // Kategori rengini belirle
            let progressBarClass = '';
            
            if (category === 'safe') {
                // Güvenli kategorisi için: yüksek skor = yeşil, düşük skor = kırmızı
                if (score >= 80) {
                    progressBarClass = 'bg-success'; // Yeşil - yüksek güven
                } else if (score >= 50) {
                    progressBarClass = 'bg-warning'; // Sarı - orta güven  
                } else {
                    progressBarClass = 'bg-danger'; // Kırmızı - düşük güven
                }
            } else {
                // Diğer kategoriler için yeni 5-seviye renk sistemi
                if (score < 20) {
                    progressBarClass = 'bg-primary'; // Mavi - çok düşük risk
                } else if (score < 35) {
                    progressBarClass = 'bg-info'; // Lacivert - düşük risk  
                } else if (score < 55) {
                    progressBarClass = 'bg-warning'; // Turuncu - belirsiz
                } else if (score < 85) {
                    progressBarClass = 'progress-bar-pink'; // Pembe - yüksek risk
                } else {
                    progressBarClass = 'bg-danger'; // Kırmızı - çok yüksek risk
                }
            }
            
            // Varsa güven skorunu al
            const confidenceScore = hasConfidenceScores ? (confidenceScores[category] || 0) : 0;
            const showConfidence = hasConfidenceScores && confidenceScore > 0;
            
            // Skor elementi HTML'i - sadece görsel bar ve risk seviyesi
            scoreElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="fw-medium">${categoryName}</span>
                    <span class="risk-score ${riskClass}">${riskLevel}</span>
                </div>
                <div class="progress mb-1" style="height: 12px; border-radius: 6px;">
                    <div class="progress-bar ${progressBarClass}" 
                         role="progressbar" style="width: ${score}%" 
                         aria-valuenow="${score}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                ${showConfidence ? `
                <div class="d-flex justify-content-between align-items-center small text-muted">
                    <span>Güven Seviyesi:</span>
                    <span class="fw-medium">${confidenceScore > 0.8 ? 'Yüksek' : confidenceScore > 0.5 ? 'Orta' : 'Düşük'}</span>
                </div>
                <div class="progress" style="height: 4px; border-radius: 2px;">
                    <div class="progress-bar bg-info" 
                         role="progressbar" style="width: ${confidenceScore * 100}%" 
                         aria-valuenow="${confidenceScore * 100}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                ` : ''}
            `;
            
            riskScoresContainer.appendChild(scoreElement);
        }
    } else {
        console.warn(`Risk skorları bulunamadı veya geçersiz format (${file.name}):`, results.overall_scores);
        riskScoresContainer.innerHTML = '<div class="alert alert-warning">Risk skorları bulunamadı veya işlenemiyor.</div>';
    }
    
    // En yüksek riskli kareyi göster
    if (results.highest_risk) {
        console.log(`En yüksek riskli kare gösteriliyor (${file.name}):`, results.highest_risk);
        
        const highestRiskFrame = resultCard.querySelector('.highest-risk-frame img');
        const highestRiskCategory = resultCard.querySelector('.highest-risk-category');
        const highestRiskScore = resultCard.querySelector('.highest-risk-score');
        const highestRiskTimestamp = resultCard.querySelector('.highest-risk-timestamp');
        const riskCategoryBadge = resultCard.querySelector('.risk-category-badge');
        
        // Açıklama ekle
        const frameContainer = resultCard.querySelector('.highest-risk-frame');
        const infoText = document.createElement('div');
        infoText.className = 'alert alert-warning mb-2';
        infoText.innerHTML = '<small><i class="fas fa-exclamation-triangle me-1"></i> İçerikte tespit edilen <strong>en yüksek risk skoruna sahip</strong> kare gösterilmektedir.</small>';
        frameContainer.insertBefore(infoText, frameContainer.firstChild);
        
        if (highestRiskFrame && results.highest_risk.frame) {
            try {
                // Analiz ID ve Frame bilgilerini al
                const frameFilename = results.highest_risk.frame;
                const frameDir = results.highest_risk.frame_dir;
                const analysisId = results.highest_risk.analysis_id;
                
                // Resim dosyası için görsel kaynağını belirle
                let imageSource = '';
                if (file.type && file.type.startsWith('image/')) {
                    // Eğer dosya bir görsel ise, her zaman direkt dosyayı kullan
                    const fileId = file.fileId || '';
                    imageSource = `/api/files/${fileId}/download`;
                } else if (results.highest_risk.frame) {
                    // Video kareleri için API endpoint ile dosyaya erişim sağla
                    const frameFilename = results.highest_risk.frame;
                    const analysisId = results.highest_risk.analysis_id;
                    imageSource = `/api/files/frames/${analysisId}/${encodeURIComponent(frameFilename)}`;
                }
                
                console.log(`Yüksek riskli kare URL'si:`, imageSource);
                
                // İmage error handling ekle
                highestRiskFrame.onerror = function() {
                    console.error("Görsel yüklenemedi:", imageSource);
                    this.src = '/static/img/image-not-found.svg';
                    this.onerror = null; // Sonsuz döngüyü önle
                };
                
                // Tıklama özelliği ekle
                highestRiskFrame.style.cursor = 'pointer';
                highestRiskFrame.title = 'Büyütmek için tıklayın';
                
                highestRiskFrame.src = imageSource;
                
                // Kategori adını düzenle
                let categoryName = results.highest_risk.category;
                let badgeClass = 'bg-warning';
                
                switch (results.highest_risk.category) {
                    case 'violence': 
                        categoryName = 'Şiddet'; 
                        badgeClass = 'bg-danger';
                        break;
                    case 'adult_content': 
                        categoryName = 'Yetişkin İçeriği'; 
                        badgeClass = 'bg-danger';
                        break;
                    case 'harassment': 
                        categoryName = 'Taciz'; 
                        badgeClass = 'bg-warning';
                        break;
                    case 'weapon': 
                        categoryName = 'Silah'; 
                        badgeClass = 'bg-danger';
                        break;
                    case 'drug': 
                        categoryName = 'Madde Kullanımı'; 
                        badgeClass = 'bg-warning';
                        break;
                    case 'safe': 
                        categoryName = 'Güvenli'; 
                        badgeClass = 'bg-success';
                        break;
                }
                
                if (highestRiskCategory) {
                    highestRiskCategory.textContent = categoryName;
                    highestRiskCategory.className = `highest-risk-category badge ${badgeClass}`;
                }
                
                if (riskCategoryBadge) {
                    riskCategoryBadge.textContent = categoryName;
                    riskCategoryBadge.className = `position-absolute bottom-0 end-0 m-2 badge ${badgeClass}`;
                }
                
                if (highestRiskScore) {
                    // Skor muhtemelen 0-1 aralığında, kontrol edip 0-100 aralığına dönüştür
                    let displayScore = results.highest_risk.score;
                    
                    // Eğer skor 0-1 aralığındaysa
                    if (displayScore <= 1.0) {
                        displayScore = displayScore * 100;
                        console.log(`En yüksek risk skoru normalize edildi: ${results.highest_risk.score} → ${displayScore}`);
                    }
                    
                    highestRiskScore.textContent = `Skor: ${displayScore.toFixed(0)}%`;
                }
                
                // Zaman bilgisini ekle
                if (highestRiskTimestamp && results.highest_risk.timestamp) {
                    const timestamp = results.highest_risk.timestamp;
                    const minutes = Math.floor(timestamp / 60);
                    const seconds = Math.floor(timestamp % 60);
                    highestRiskTimestamp.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
            } catch (error) {
                console.error("Görsel URL'si oluşturulurken hata:", error);
                highestRiskFrame.src = '/static/img/image-not-found.svg';
            }
        } else {
            console.warn(`En yüksek riskli kare bilgileri eksik`, results.highest_risk);
            const highestRiskContainer = resultCard.querySelector('.highest-risk-frame');
            if (highestRiskContainer) {
                highestRiskContainer.innerHTML = '<div class="alert alert-warning">Görsel bilgileri alınamadı.</div>';
            }
        }
    } else {
        console.warn(`En yüksek riskli kare bulunamadı (${file.name})`);
        const highestRiskContainer = resultCard.querySelector('.highest-risk-frame');
        if (highestRiskContainer) {
            highestRiskContainer.innerHTML = '<div class="alert alert-warning">Yüksek riskli kare bulunamadı.</div>';
        }
    }
    
    // ===== DETAY TAB - İÇERİK TESPİTLERİ =====
    if (detailsTab) {
        // Detaylar sayfasını temizleyelim
        detailsTab.innerHTML = '';
        
        // İçerik tespitleri
        const contentDetections = results.content_detections || [];
        
        if (contentDetections.length > 0) {
            try {
                // Detaylar sayfasını temizleyelim
                detailsTab.innerHTML = '';
                
                // İçerik tespitleri bölümü
                const contentDetectionsSection = document.createElement('div');
                contentDetectionsSection.classList.add('content-detections', 'mb-4');
                contentDetectionsSection.innerHTML = `
                    <h5 class="mb-3"><i class="fas fa-exclamation-triangle me-2"></i>Kategori Bazında En Yüksek Riskli Kareler</h5>
                    <div class="row" id="categoryTopDetectionsList-${uniqueSuffix}"></div>
                `;
                detailsTab.appendChild(contentDetectionsSection);
                
                const categoryDetectionsList = contentDetectionsSection.querySelector(`#categoryTopDetectionsList-${uniqueSuffix}`);
                
                // Her kategori için en yüksek skorlu tespitleri bul
                const categoryTopDetections = {
                    'violence': null,
                    'adult_content': null,
                    'harassment': null,
                    'weapon': null,
                    'drug': null,
                    'safe': null
                };
                
                // En yüksek skoru takip etmek için değişken tanımla
                const highestScores = {
                    'violence': 0,
                    'adult_content': 0,
                    'harassment': 0,
                    'weapon': 0,
                    'drug': 0,
                    'safe': 0
                };
                
                // Her kategori için en yüksek skorlu kareleri bul
                contentDetections.forEach(detection => {
                    // Eski kontrol:
                    // if (!detection.content_scores || typeof detection.content_scores !== 'object') { ... }
                    
                    // Doğrudan skor alanlarını kontrol edelim
                    const categoryScores = {
                        'violence': detection.violence_score,
                        'adult_content': detection.adult_content_score,
                        'harassment': detection.harassment_score,
                        'weapon': detection.weapon_score,
                        'drug': detection.drug_score,
                        'safe': detection.safe_score
                    };
                    
                    console.log('Tespit edilen skorlar:', categoryScores);
                    
                    // Her kategori için skoru kontrol et
                    for (const [category, score] of Object.entries(categoryScores)) {
                        if (score && !isNaN(score)) {
                            // Skor 0-1 aralığında mı kontrol et
                            let normalizedScore = score;
                            if (score <= 1.0) {
                                normalizedScore = score * 100;
                                console.log(`Detay tabı ${category} skoru normalize edildi: ${score} → ${normalizedScore}`);
                            }
                            
                            if (!categoryTopDetections[category] || normalizedScore > highestScores[category]) {
                                console.log(`Daha yüksek ${category} skoru bulundu:`, normalizedScore);
                                categoryTopDetections[category] = {
                                    score: normalizedScore, // normalize edilmiş skoru kullan
                                    frame_path: detection.frame_path,
                                    timestamp: detection.frame_timestamp // frame_timestamp alanını kullan
                                };
                                highestScores[category] = normalizedScore; // En yüksek skoru güncelle
                            }
                        }
                    }
                });
                
                console.log('Bulunan en yüksek kategoriler:', categoryTopDetections);
                
                // Her kategori için en yüksek skorlu kareyi göster
                let detectionCount = 0;
                for (const [category, detection] of Object.entries(categoryTopDetections)) {
                    if (!detection || detection.score < 0.1) continue; // Çok düşük skorları atla
                    
                    detectionCount++;
                    const detectionCard = document.createElement('div');
                    detectionCard.classList.add('col-md-4', 'mb-3');
                    
                    // Kategori adını düzenle
                    let categoryName = category;
                    let badgeClass = 'bg-success';
                    
                    switch (category) {
                        case 'violence': 
                            categoryName = 'Şiddet'; 
                            badgeClass = (detection.score >= 70) ? 'bg-danger' : (detection.score >= 30) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'adult_content': 
                            categoryName = 'Yetişkin İçeriği'; 
                            badgeClass = (detection.score >= 70) ? 'bg-danger' : (detection.score >= 30) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'harassment': 
                            categoryName = 'Taciz'; 
                            badgeClass = (detection.score >= 70) ? 'bg-danger' : (detection.score >= 30) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'weapon': 
                            categoryName = 'Silah'; 
                            badgeClass = (detection.score >= 70) ? 'bg-danger' : (detection.score >= 30) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'drug': 
                            categoryName = 'Madde Kullanımı'; 
                            badgeClass = (detection.score >= 70) ? 'bg-danger' : (detection.score >= 30) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'safe': 
                            categoryName = 'Güvenli'; 
                            badgeClass = (detection.score >= 70) ? 'bg-success' : (detection.score >= 30) ? 'bg-info' : 'bg-warning';
                            break;
                    }
                    
                    // Zaman bilgisini formatla
                    let timeText = '';
                    if (detection.timestamp) {
                        const minutes = Math.floor(detection.timestamp / 60);
                        const seconds = Math.floor(detection.timestamp % 60);
                        timeText = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                    }
                    
                    // Karşılık gelen görseli yükle
                    let frameUrl = '';
                    if (detection.frame_path) {
                        // Resim dosyası mı yoksa video karesi mi?
                        if (file.type && file.type.startsWith('image/')) {
                            // Eğer dosya bir görsel ise, direkt dosyayı kullan
                            const fileId = file.fileId || '';
                            frameUrl = `/api/files/${fileId}/download`;
                        } else {
                            // Video kareleri için
                            const frameName = normalizePath(detection.frame_path).split(/[\\/]/).pop();
                            frameUrl = `/api/files/frames/${results.analysis_id}/${frameName}`;
                        }
                    } else if (file.type && file.type.startsWith('image/')) {
                        // Eğer frame_path yoksa ama bu bir görsel dosyasıysa, görsel dosyasını kullan
                        const fileId = file.fileId || '';
                        frameUrl = `/api/files/${fileId}/download`;
                    }
                    
                    console.log(`${category} için frame URL:`, frameUrl);
                    
                    // Kart içeriğini oluştur
                    detectionCard.innerHTML = `
                        <div class="card h-100">
                            <div class="position-relative">
                                <div style="height: 240px; overflow: hidden;">
                                    <img src="${frameUrl}" class="card-img-top detection-img" alt="${categoryName}" 
                                        style="width: 100%; height: 100%; object-fit: cover; cursor: pointer;"
                                        title="Büyütmek için tıklayın"
                                        onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';">
                                </div>
                                <span class="position-absolute top-0 end-0 m-2 badge ${badgeClass}">${categoryName}</span>
                                ${timeText ? `<span class="position-absolute bottom-0 start-0 m-2 badge bg-dark">${timeText}</span>` : ''}
                            </div>
                            <div class="card-body">
                                <h6 class="card-title">${categoryName}</h6>
                                <div class="d-flex justify-content-between mb-1">
                                    <span>${category === 'safe' ? 'Güven Skoru:' : 'Risk Skoru:'}</span>
                                    <strong>${highestScores[category].toFixed(0)}%</strong>
                                </div>
                                <div class="progress">
                                    <div class="progress-bar ${badgeClass}" 
                                        style="width: ${highestScores[category]}%" 
                                        role="progressbar" 
                                        aria-valuenow="${highestScores[category]}" 
                                        aria-valuemin="0" 
                                        aria-valuemax="100">
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    categoryDetectionsList.appendChild(detectionCard);
                }
                
                // Eğer kategorilerde hiç kart yoksa bilgi mesajı göster
                if (detectionCount === 0) {
                    categoryDetectionsList.innerHTML = '<div class="col-12"><div class="alert alert-info">Bu dosyada önemli içerik tespiti yapılmadı.</div></div>';
                }
            } catch (error) {
                console.error("İçerik tespitleri gösterilirken hata:", error);
                detailsTab.innerHTML += `<div class="alert alert-danger mb-4">İçerik tespitleri işlenirken hata oluştu: ${error.message}</div>`;
            }
        } else {
            detailsTab.innerHTML += '<div class="alert alert-info mb-4">Bu dosya için içerik tespiti bulunmuyor.</div>';
        }
    }
    
    // ===== DETAY TAB - YAŞ TAHMİNİ =====
    // Yaş tahmini varsa göster
    console.log('YAŞ TAHMİNİ - API YANITI İNCELEME:', results);
    
    // API yanıtındaki yaş verilerini detaylı incele
    if (results.age_estimations) {
        console.log('YAŞ TAHMİNİ - age_estimations mevcut:', results.age_estimations);
    } else if (results.age_analysis) {
        console.log('YAŞ TAHMİNİ - age_analysis mevcut:', results.age_analysis);
    } else {
        console.warn('YAŞ TAHMİNİ - Yaş verisi bulunamadı. API yanıtı:', results);
    }
    
    // Yaş tahmini verilerini uygun şekilde işlemeye çalış
    if ((results.age_estimations && results.age_estimations.length > 0) || 
        (results.age_analysis && results.age_analysis.length > 0)) {
        try {
            // Backend'in döndüğü veri yapısına göre uygun değişkeni seç
            const ageData = results.age_estimations || results.age_analysis || [];
            console.log('Yaş tahmini işlenen veriler:', ageData.length, 'kayıt bulundu');

            // Geri bildirimdekiyle aynı mapping: en yüksek confidence'lı kaydı seç
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

            // Geri bildirimdeki gibi kartları oluştur
            const faceIds = Object.keys(faces);
            const ageEstimationSection = document.createElement('div');
            ageEstimationSection.classList.add('age-estimations', 'mt-4');
            ageEstimationSection.innerHTML = `
                <h5 class="mb-3"><i class="fas fa-user-alt me-2"></i>Yaş Tahminleri</h5>
                <div class="alert alert-info mb-3">
                    <i class="fas fa-info-circle me-2"></i> Her tespit edilen benzersiz yüz için en yüksek güven skorlu tahmin gösterilmektedir.
                </div>
                <div class="row" id="ageEstimationList-${uniqueSuffix}"></div>
            `;
            detailsTab.appendChild(ageEstimationSection);
            const ageEstimationList = ageEstimationSection.querySelector(`#ageEstimationList-${uniqueSuffix}`);

            if (faceIds.length === 0) {
                ageEstimationList.innerHTML = '<div class="col-12"><div class="alert alert-info">Bu dosyada tespit edilen yüz bulunmuyor.</div></div>';
            } else {
                faceIds.forEach((faceId, index) => {
                    const face = faces[faceId];
                    console.log(`[DEBUG] Yüz kartı oluşturuluyor - Index: ${index}, FaceID: ${faceId}`);
                    console.log("[DEBUG] Yüz verisi:", face);

                    const col = document.createElement('div');
                    col.className = 'col-md-6 mb-4';
                    
                    // 18 yaş altı kontrolü
                    const isUnderAge = face.age < 18;
                    const ageClass = isUnderAge ? 'border-danger bg-danger-subtle' : '';
                    const ageWarning = isUnderAge ? 
                        `<div class="alert alert-danger mt-2 mb-0 p-2">
                            <small><i class="fas fa-exclamation-triangle me-1"></i> <strong>Dikkat:</strong> 18 yaş altında birey tespit edildi!</small>
                        </div>` : '';
                    
                    // Görsel URL'sini oluştur
                    let frameUrl = '';
                    if (face.processed_image_path) {
                        frameUrl = `/api/files/${normalizePath(face.processed_image_path).replace(/^\/+|\/+/g, '/')}`;
                        console.log("[DEBUG] İşlenmiş görsel URL'si:", frameUrl);
                        console.log('[LOG][FRONTEND] Backendden gelen processed_image_path:', face.processed_image_path);
                        console.log('[LOG][FRONTEND] Frontendde gösterilen img src:', frameUrl);
                        
                        col.innerHTML = `
                            <div class="card h-100 ${ageClass}">
                                <div class="card-body">
                                    <div class="row align-items-center">
                                        <div class="col-md-12">
                                            <div class="position-relative" style="height: 300px; overflow: hidden;">
                                                <img src="${frameUrl}" 
                                                     alt="ID: ${faceId.includes('_person_') ? faceId.split('_person_').pop() : index + 1}"
                                                     style="width: 100%; height: 100%; object-fit: contain; cursor: pointer;"
                                                     class="age-estimation-image"
                                                     onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';"
                                                     onload="console.log('[DEBUG] Görsel başarıyla yüklendi:', this.src)"
                                                     title="Büyütmek için tıklayın">
                                                <span class="position-absolute top-0 end-0 m-2 badge bg-info">ID: ${faceId.includes('_person_') ? faceId.split('_person_').pop() : index + 1}</span>
                                                ${isUnderAge ? '<span class="position-absolute top-0 start-0 m-2 badge bg-danger"><i class="fas fa-exclamation-triangle me-1"></i> 18 yaş altı</span>' : ''}
                                            </div>
                                            <div class="mt-3">
                                                <h5 class="card-title mb-3">Tahmini Yaş: ${Math.round(face.age)}</h5>
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
                                    <h5 class="card-title mb-3">Tahmini Yaş: ${Math.round(face.age)}</h5>
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
    } else if (results.include_age_analysis) {
        detailsTab.innerHTML += '<div class="alert alert-info mt-3">Bu dosya için yaş tahmini bulunmuyor.</div>';
    }
    
    // Yaş tahminleri geri bildirimini göster
    const feedbackTab = resultCard.querySelector(`#feedback-${uniqueSuffix}`);
    if (feedbackTab) {
        displayAgeFeedback(feedbackTab, results);
    }

    // Sonuç kartını listeye ekle
    const resultsList = document.getElementById('resultsList');
    if (!resultsList) {
        console.error('resultsList bulunamadı!');
        return;
    }
    
    // Eğer bu fileId için sonuç kartı zaten varsa, yenisini ekleme
    const existingCard = document.querySelector(`.result-card[data-file-id="${fileId}"]`);
    if (existingCard) {
        console.log(`${file.name} için sonuç kartı zaten var, güncelleniyor...`);
        existingCard.remove(); // Varolan kartı kaldır (yenisiyle değiştirmek için)
    }
    
    // Sonuç kartını ekle ve görünür olduğundan emin ol
    const resultCardEl = resultCard.querySelector('.result-card');
    resultCardEl.setAttribute('data-file-id', fileId);
    resultsList.appendChild(resultCardEl);
    
    // Sonuçlar bölümünü görünür yap
    document.getElementById('resultsSection').style.display = 'block';
    
    console.log(`Analiz sonuç kartı eklendi (${file.name})`);
}

// Zaman formatı
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Geri bildirim gönder
function submitFeedback(event) {
    event.preventDefault();
    
    const form = event.target;
    const resultCard = form.closest('.result-card');
    const contentId = form.querySelector('.content-id').value;
    const analysisIdForContent = form.querySelector('input[name="analysis_id"]').value;
    const framePathForContent = form.querySelector('input[name="frame_path"]').value;
    
    const mainSubmitButton = form.querySelector('button[type="submit"]');
    mainSubmitButton.disabled = true;
    mainSubmitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Gönderiliyor...';

    const categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug'];
    const categoryFeedback = {};
    const categoryCorrectValues = {};

    categories.forEach(category => {
        const feedbackSelect = form.querySelector(`.${category.replace('_', '-')}-feedback`);
        const correctValueInput = form.querySelector(`.${category.replace('_', '-')}-correct-value`);
        
        const feedbackValue = feedbackSelect ? feedbackSelect.value : "";
        categoryFeedback[category] = feedbackValue;

        let correctValue = null;
        if (correctValueInput && correctValueInput.value !== "") {
            correctValue = parseFloat(correctValueInput.value);
            if (isNaN(correctValue) || correctValue < 0 || correctValue > 100) {
                showToast('Uyarı', `Kategori '${category}' için geçersiz skor: ${correctValueInput.value}. Lütfen 0-100 arası bir değer girin.`, 'warning');
                // Hatalı durumda butonu tekrar aktif et ve işlemi durdur
                mainSubmitButton.disabled = false;
                mainSubmitButton.innerHTML = 'Tekrar Dene';
                // throw new Error(`Invalid score for ${category}`); // Daha katı bir hata yönetimi için
                return; // Fonksiyondan erken çıkış yapabilir veya kategori için null gönderebilir
            }
        }

        if (feedbackValue === 'false_positive') {
            categoryCorrectValues[category] = 0;
        } else if (feedbackValue === 'correct') {
            categoryCorrectValues[category] = null; // Modelin skoru doğru kabul ediliyor, özel bir skor yok
        } else if (feedbackValue === 'false_negative' || feedbackValue === 'score_too_low' || feedbackValue === 'score_too_high') {
            // Kullanıcı bir skor girdiyse onu kullan, girmediyse null (veya backend'de varsayılan bir işlem)
            categoryCorrectValues[category] = (correctValueInput && correctValueInput.value !== "") ? correctValue : null;
        } else {
            // Eğer feedbackValue boşsa (Değerlendirme seçin) veya beklenmeyen bir değerse
            categoryCorrectValues[category] = null; // Ya da bu kategori için veri gönderme
        }
    });
    
    // Eğer bir kategori için geçersiz skor girildiyse ve yukarıda return ile çıkıldıysa, devam etme.
    // Bu kontrol, forEach içindeki return'ün sadece döngünün o adımını atladığını, fonksiyonu sonlandırmadığını dikkate alır.
    // Daha sağlam bir yapı için, forEach yerine for...of döngüsü ve erken return kullanılabilir veya bir flag tutulabilir.
    // Şimdilik, her kategori için uyarı verip null göndermeye devam edecek şekilde bırakıyoruz, 
    // ama en az bir hata varsa butonun aktif kalmasını sağlıyoruz.
    let hasErrorInScores = false;
    categories.forEach(category => {
        const correctValueInput = form.querySelector(`.${category.replace('_', '-')}-correct-value`);
        if (correctValueInput && correctValueInput.value !== "") {
            const val = parseFloat(correctValueInput.value);
            if (isNaN(val) || val < 0 || val > 100) {
                hasErrorInScores = true;
            }
        }
    });

    if (hasErrorInScores) {
        // mainSubmitButton.disabled = false; // Zaten yukarıda yapılıyor
        // mainSubmitButton.innerHTML = 'Tekrar Dene';
        return; // Hata varsa gönderme işlemi yapma
    }

    const contentFeedbackData = {
        content_id: contentId,
        analysis_id: analysisIdForContent,
        frame_path: framePathForContent,
        rating: parseInt(form.querySelector('.general-rating').value),
        comment: form.querySelector('.feedback-comment').value,
        category_feedback: categoryFeedback,
        category_correct_values: categoryCorrectValues
    };
    
    fetch('/api/feedback/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(contentFeedbackData)
    })
    .then(response => {
        if (!response.ok) {
            // Try to parse error from backend
            return response.json().then(err => { throw new Error(err.error || `İçerik geri bildirimi HTTP hatası! Durum: ${response.status}`) });
        }
        return response.json();
    })
    .then(data => {
        console.log('İçerik geri bildirimi başarıyla gönderildi:', data);
        showToast('Başarılı', 'İçerik geri bildiriminiz kaydedildi.', 'success');

        // 2. Collect and Send Age Feedback
        const allAgeFeedbacks = [];
        if (resultCard) { // Ensure resultCard is found
            const ageInputFields = resultCard.querySelectorAll('.age-feedback-container .corrected-age');
            ageInputFields.forEach(input => {
                const correctedAgeValue = input.value.trim();
                if (correctedAgeValue !== "") { // Only process if a value is entered
                    const correctedAge = parseInt(correctedAgeValue);
                    const personId = input.dataset.personId;
                    const analysisIdForAge = input.dataset.analysisId; // Should be same as analysisIdForContent
                    const framePathForAge = input.dataset.framePath;

                    if (isNaN(correctedAge) || correctedAge <= 0 || correctedAge > 100) {
                        showToast('Uyarı', `Kişi ${personId} için geçersiz yaş değeri: ${correctedAgeValue}. Lütfen 1-100 arası bir değer girin.`, 'warning');
                        // Optionally, re-enable the main button and return if strict validation is needed here
                        // mainSubmitButton.disabled = false;
                        // mainSubmitButton.innerHTML = 'Gönder';
                        // throw new Error("Invalid age input"); 
                        return; // Skip this invalid age feedback
                    }
                    
                    if (!personId || !analysisIdForAge || !framePathForAge) {
                        console.error('Yaş geri bildirimi için eksik data attribute: ', {personId, analysisIdForAge, framePathForAge});
                        showToast('Hata', `Kişi ${personId} için yaş geri bildirimi gönderilemedi (eksik bilgi).`, 'danger');
                        return; // Skip this age feedback
                    }

                    allAgeFeedbacks.push({
                        person_id: personId,
                        corrected_age: correctedAge,
                        is_age_range_correct: false, // This field seems to be gone from the simplified UI
                        analysis_id: analysisIdForAge,
                        frame_path: framePathForAge
                    });
                }
            });
        } else {
            console.warn("submitFeedback: .result-card bulunamadı, yaş geri bildirimleri toplanamadı.");
        }
        

        if (allAgeFeedbacks.length > 0) {
            const ageFeedbackPromises = allAgeFeedbacks.map(ageFeedback => {
                return fetch('/api/feedback/age', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(ageFeedback)
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => { throw new Error(err.error || `Yaş geri bildirimi (${ageFeedback.person_id}) HTTP Hatası! Durum: ${response.status}`) });
                    }
                    return response.json();
                })
                .then(ageData => {
                    console.log(`Yaş geri bildirimi (${ageFeedback.person_id}) başarıyla gönderildi:`, ageData);
                });
            });

            return Promise.allSettled(ageFeedbackPromises)
                .then(results => {
                    let allSuccessful = true;
                    results.forEach(result => {
                        if (result.status === 'rejected') {
                            allSuccessful = false;
                            console.error('Bir yaş geri bildirimi gönderme hatası:', result.reason);
                            showToast('Hata', `Bir yaş geri bildirimi gönderilemedi: ${result.reason.message}`, 'danger');
                        }
                    });
                    if (allSuccessful && allAgeFeedbacks.length > 0) {
                        showToast('Başarılı', 'Tüm yaş geri bildirimleri kaydedildi.', 'success');
                    }
                    return allSuccessful; // Propagate success status
                });
        }
        return true; // Content feedback was successful, no age feedback to send
    })
    .then((allFeedbacksSuccessful) => {
        if (allFeedbacksSuccessful) { // Check if content and all age feedbacks were processed successfully
            mainSubmitButton.innerHTML = '<i class="fas fa-check me-1"></i> Gönderildi';
            // Keep it disabled
        } else {
             mainSubmitButton.disabled = false; // Re-enable if there were issues
             mainSubmitButton.innerHTML = 'Tekrar Dene';
        }
    })
    .catch(error => {
        console.error('Geri bildirim gönderme sırasında genel hata:', error);
        showToast('Hata', `Geri bildirim gönderilirken genel bir hata oluştu: ${error.message}`, 'danger');
        mainSubmitButton.disabled = false;
        mainSubmitButton.innerHTML = 'Tekrar Dene';
    });
}

// Yaş geri bildirimi gönder
// submitAgeFeedback fonksiyonunu güncelliyoruz: buttonElement parametresi alacak
// Bu fonksiyon artık kullanılmıyor, kaldırıldı.

// Geliştirilmiş yaş tahmini display için yardımcı fonksiyon
// createAgeFeedbackElements fonksiyonu artık kullanılmıyor, kaldırıldı.

// Model metrikleri yükle
function loadModelMetrics() {
    // Settings save loader'ı gizle (eğer görünürse)
    const settingsSaveLoader = document.getElementById('settingsSaveLoader');
    if (settingsSaveLoader && settingsSaveLoader.style.display === 'flex') {
        settingsSaveLoader.style.display = 'none';
    }
    
    let contentPromise, agePromise;
    
    // CLIP ensemble metriklerini yükle
    contentPromise = Promise.all([
        fetch('/api/ensemble/stats/content').then(r => r.json()),
        fetch('/api/model/metrics/content').then(r => r.json())
    ])
    .then(([ensembleData, modelData]) => {
        console.log('CLIP ensemble metrikleri:', ensembleData);
        console.log('İçerik model metrikleri:', modelData);
        
        // Model service verilerinden feedback_sources'ı al
        const contentModelData = modelData.content || {};
        const mergedData = {
            ...ensembleData,
            feedback_sources: contentModelData.feedback_sources || { manual: 0, pseudo: 0 },
            feedback_count: contentModelData.feedback_count || 0
        };
        
        // Ensemble versiyonlarını al
        return fetch('/api/ensemble/versions/content')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(versionData => {
                // Versiyon bilgilerini ekle
                mergedData.versions = versionData.versions;
                return mergedData;
            })
            .catch(error => {
                console.error('CLIP ensemble versiyonları alınamadı:', error);
                return mergedData;
            });
    })
    .then(data => {
        displayContentModelMetrics(data);
    })
    .catch(error => {
        console.error('CLIP ensemble metrikleri alınırken hata:', error);
        document.getElementById('contentMetricsTab').innerHTML = `
            <div class="alert alert-danger">Ensemble metrikler yüklenirken hata oluştu: ${error.message}</div>
        `;
    });
    
    // Yaş model metriklerini yükle (hem ensemble hem model service verilerini birleştir)
    agePromise = Promise.all([
        fetch('/api/ensemble/stats/age').then(r => r.json()),
        fetch('/api/model/metrics/age').then(r => r.json())
    ])
    .then(([ensembleData, modelData]) => {
        console.log('Yaş ensemble metrikleri:', ensembleData);
        console.log('Yaş model metrikleri:', modelData);
        
        // Model service verilerinden feedback_sources'ı al
        const ageModelData = modelData.age || {};
        const mergedData = {
            ...ensembleData,
            feedback_sources: ageModelData.feedback_sources || { manual: 0, pseudo: 0 },
            feedback_count: ageModelData.feedback_count || 0
        };
        
        // Ensemble versiyonlarını al
        return fetch('/api/ensemble/versions/age')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(versionData => {
                // Versiyon bilgilerini ekle
                mergedData.versions = versionData.versions;
                return mergedData;
            })
            .catch(error => {
                console.error('Yaş ensemble versiyonları alınamadı:', error);
                return mergedData;
            });
    })
    .then(data => {
        displayAgeModelMetrics(data);
    })
    .catch(error => {
        console.error('Yaş ensemble metrikleri alınırken hata:', error);
        document.getElementById('ageMetricsTab').innerHTML = `
            <div class="alert alert-danger">Ensemble metrikler yüklenirken hata oluştu: ${error.message}</div>
        `;
    });
    
    // Her iki yükleme de tamamlandığında settings loader'ını kesin olarak gizle
    Promise.allSettled([contentPromise, agePromise]).finally(() => {
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'none';
        }
    });
}

// İçerik analiz modeli metriklerini göster
function displayContentModelMetrics(data) {
    console.log('displayContentModelMetrics called with data:', data);
    
    // Veri kontrolü
    if (!data) {
        console.warn('displayContentModelMetrics: No data provided');
        data = {};
    }
    
    // Eğitim verisi sayaçlarını güncelle
    const feedbackSources = data.feedback_sources || {};
    const manualCount = feedbackSources.manual || 0;
    const pseudoCount = feedbackSources.pseudo || 0;
    const totalCount = (manualCount + pseudoCount) || (data.feedback_count || 0);
    const manualEl = document.getElementById('content-manual-count');
    const pseudoEl = document.getElementById('content-pseudo-count');
    const totalEl = document.getElementById('content-total-count');
    if (manualEl) manualEl.textContent = `Manuel: ${manualCount}`;
    if (pseudoEl) pseudoEl.textContent = `Pseudo: ${pseudoCount}`;
    if (totalEl) totalEl.textContent = `Toplam: ${totalCount}`;
    
    // Loading spinner'ı kaldır
    const contentTab = document.getElementById('contentMetricsTab');
    if (contentTab) {
        const loadingSpinner = contentTab.querySelector('.spinner-border');
        if (loadingSpinner) {
            loadingSpinner.remove();
        }
    }
    
    // CLIP ensemble metrikler
    const ensembleMetrics = data.ensemble_metrics || {};
    const baseModel = data.base_model || {};
    
    // Ensemble performans gösterimi
    const accuracyEl = document.querySelector('.content-accuracy');
    const precisionEl = document.querySelector('.content-precision');
    const recallEl = document.querySelector('.content-recall');
    const f1El = document.querySelector('.content-f1');
    
    if (ensembleMetrics.content_corrections > 0 || ensembleMetrics.confidence_adjustments > 0) {
        if (accuracyEl) accuracyEl.textContent = 'Ensemble Enhanced';
        if (precisionEl) precisionEl.textContent = '100% (Lookup)';
        if (recallEl) recallEl.textContent = '100% (Lookup)';
        if (f1El) f1El.textContent = '100% (Lookup)';
    } else {
        if (accuracyEl) accuracyEl.textContent = 'Base OpenCLIP Performance';
        if (precisionEl) precisionEl.textContent = 'Base OpenCLIP';
        if (recallEl) recallEl.textContent = 'Base OpenCLIP';
        if (f1El) f1El.textContent = 'Base OpenCLIP';
    }
    
    // CLIP ensemble kategori performansı
    const categoryMetricsTable = document.getElementById('contentCategoryMetrics');
    categoryMetricsTable.innerHTML = '';
    
    // Sabit kategori listesi
    const categories = [
        { key: 'violence', name: 'Şiddet' },
        { key: 'adult_content', name: 'Yetişkin İçeriği' }, 
        { key: 'harassment', name: 'Taciz' },
        { key: 'weapon', name: 'Silah' },
        { key: 'drug', name: 'Madde Kullanımı' },
        { key: 'safe', name: 'Güvenli' }
    ];
    
    const hasEnsembleCorrections = ensembleMetrics.content_corrections > 0 || ensembleMetrics.confidence_adjustments > 0;
    
    categories.forEach(cat => {
        const row = document.createElement('tr');
        if (hasEnsembleCorrections) {
            row.innerHTML = `
                <td>${cat.name}</td>
                <td>Ensemble Enhanced</td>
                <td>Lookup Based</td>
                <td>Lookup Based</td>
                <td>Perfect (100%)</td>
            `;
        } else {
            row.innerHTML = `
                <td>${cat.name}</td>
                <td>Base OpenCLIP</td>
                <td>Base OpenCLIP</td>
                <td>Base OpenCLIP</td>
                <td>Base OpenCLIP</td>
            `;
        }
        categoryMetricsTable.appendChild(row);
    });
    
    // Eğitim geçmişi
    const trainingHistoryContainer = document.getElementById('contentTrainingHistory');
    
    if (data.training_history && data.training_history.length > 0) {
        trainingHistoryContainer.innerHTML = '';
        
        const table = document.createElement('table');
        table.className = 'table table-bordered table-sm';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Tarih</th>
                    <th>Epoch Sayısı</th>
                    <th>Eğitim Kümesi</th>
                    <th>Doğrulama Kümesi</th>
                    <th>Süre</th>
                </tr>
            </thead>
            <tbody id="contentTrainingHistoryBody"></tbody>
        `;
        
        trainingHistoryContainer.appendChild(table);
        
        const tbody = document.getElementById('contentTrainingHistoryBody');
        
        data.training_history.forEach(history => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${new Date(history.date).toLocaleString()}</td>
                <td>${history.epochs}</td>
                <td>${history.training_samples}</td>
                <td>${history.validation_samples}</td>
                <td>${formatDuration(history.duration)}</td>
            `;
            
            tbody.appendChild(row);
        });
    } else {
        trainingHistoryContainer.innerHTML = '<div class="alert alert-info">Henüz eğitim yapılmamış.</div>';
    }
    
    // Versiyon bilgisi ekle
    if (data.versions && data.versions.length > 0) {
        displayModelVersions('content', data.versions);
    }
}

// Yaş analiz modeli metriklerini göster
function displayAgeModelMetrics(data) {
    console.log('displayAgeModelMetrics called with data:', data);
    
    // Veri kontrolü
    if (!data) {
        console.warn('displayAgeModelMetrics: No data provided');
        data = {};
    }
    
    // Eğitim verisi sayaçlarını güncelle
    const feedbackSources = data.feedback_sources || {};
    const manualCount = feedbackSources.manual || 0;
    const pseudoCount = feedbackSources.pseudo || 0;
    const totalCount = (manualCount + pseudoCount) || (data.feedback_count || 0);
    const manualEl = document.getElementById('age-manual-count');
    const pseudoEl = document.getElementById('age-pseudo-count');
    const totalEl = document.getElementById('age-total-count');
    if (manualEl) manualEl.textContent = `Manuel: ${manualCount}`;
    if (pseudoEl) pseudoEl.textContent = `Pseudo: ${pseudoCount}`;
    if (totalEl) totalEl.textContent = `Toplam: ${totalCount}`;
    
    // Loading spinner'ı kaldır
    const ageTab = document.getElementById('ageMetricsTab');
    if (ageTab) {
        const loadingSpinner = ageTab.querySelector('.spinner-border');
        if (loadingSpinner) {
            loadingSpinner.remove();
        }
    }
    // Ensemble metrikler - ensemble formatı
    const ensembleMetrics = data.ensemble_metrics || {};
    const baseModel = data.base_model || {};
    
    // MAE gösterimi - safe element access
    const maeEl = document.querySelector('.age-mae');
    const accuracyEl = document.querySelector('.age-accuracy');
    const countEl = document.querySelector('.age-count');
    
    if (ensembleMetrics.people_corrections > 0) {
        if (maeEl) maeEl.textContent = '0.00 yaş (Ensemble Perfect)';
        if (accuracyEl) accuracyEl.textContent = '100.0% (Lookup)';
    } else {
        if (maeEl) maeEl.textContent = baseModel.mae ? `${baseModel.mae} yaş (Base Model)` : '-';
        if (accuracyEl) accuracyEl.textContent = 'Base Model Performance';
    }
    
    // Ensemble düzeltme sayısı
    const totalCorrections = ensembleMetrics.people_corrections || 0;
    if (countEl) countEl.textContent = `${totalCorrections} ensemble correction`;
    
    // Yaş dağılımı grafiği
    if (data.age_distribution) {
        const ageDistributionCanvas = document.getElementById('ageDistributionChart');
        const ageDistributionCtx = ageDistributionCanvas.getContext('2d');
        
        // Mevcut grafiği temizle
        if (window.ageDistributionChart && typeof window.ageDistributionChart.destroy === 'function') {
            window.ageDistributionChart.destroy();
        }
        
        const labels = Object.keys(data.age_distribution);
        const values = Object.values(data.age_distribution);
        
        window.ageDistributionChart = new Chart(ageDistributionCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Kişi Sayısı',
                    data: values,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Kişi Sayısı'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Yaş Aralığı'
                        }
                    }
                }
            }
        });
    }
    
    // Yaş hata dağılımı grafiği
    if (data.error_distribution) {
        const ageErrorCanvas = document.getElementById('ageErrorChart');
        const ageErrorCtx = ageErrorCanvas.getContext('2d');
        
        // Mevcut grafiği temizle
        if (window.ageErrorChart && typeof window.ageErrorChart.destroy === 'function') {
            window.ageErrorChart.destroy();
        }
        
        const labels = Object.keys(data.error_distribution);
        const values = Object.values(data.error_distribution);
        
        window.ageErrorChart = new Chart(ageErrorCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Hata Dağılımı',
                    data: values,
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Sayı'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Yaş Farkı'
                        }
                    }
                }
            }
        });
    }
    
    // Eğitim geçmişi
    const trainingHistoryContainer = document.getElementById('ageTrainingHistory');
    
    if (data.training_history && data.training_history.length > 0) {
        trainingHistoryContainer.innerHTML = '';
        
        const table = document.createElement('table');
        table.className = 'table table-bordered table-sm';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Tarih</th>
                    <th>Epoch Sayısı</th>
                    <th>Eğitim Kümesi</th>
                    <th>Doğrulama Kümesi</th>
                    <th>MAE</th>
                    <th>Süre</th>
                </tr>
            </thead>
            <tbody id="ageTrainingHistoryBody"></tbody>
        `;
        
        trainingHistoryContainer.appendChild(table);
        
        const tbody = document.getElementById('ageTrainingHistoryBody');
        
        data.training_history.forEach(history => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${new Date(history.date).toLocaleString()}</td>
                <td>${history.epochs}</td>
                <td>${history.training_samples}</td>
                <td>${history.validation_samples}</td>
                <td>${history.mae.toFixed(2)}</td>
                <td>${formatDuration(history.duration)}</td>
            `;
            
            tbody.appendChild(row);
        });
    } else {
        trainingHistoryContainer.innerHTML = '<div class="alert alert-info">Henüz eğitim yapılmamış.</div>';
    }
    
    // Versiyon bilgisi ekle
    if (data.versions && data.versions.length > 0) {
        displayModelVersions('age', data.versions);
    }
}

// Model versiyonlarını göster (Model Metrics modal için - sadece görüntüleme)
function displayModelVersions(modelType, versions) {
    const containerId = modelType === 'content' ? 'contentVersionsContainer' : 'ageVersionsContainer';
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error(`Container not found: ${containerId}`);
        return;
    }
    
    // Loading spinner'ı kaldır
    const loadingSpinner = container.querySelector('.spinner-border');
    if (loadingSpinner) {
        loadingSpinner.remove();
    }
    
    if (!versions || versions.length === 0) {
        container.innerHTML = '<p class="text-muted">Hiç model versiyonu bulunamadı.</p>';
        return;
    }
    
    // Versiyonları sırala (en yeni önce)
    const sortedVersions = versions.sort((a, b) => b.version - a.version);
    
    const versionsList = document.createElement('div');
    versionsList.className = 'list-group';
    
    sortedVersions.forEach(version => {
        const versionItem = document.createElement('div');
        versionItem.className = `list-group-item ${version.is_active ? 'list-group-item-success' : ''}`;
        
        // Metrikleri hazırla
        let metricsHtml = '';
        if (version.metrics) {
            if (modelType === 'content') {
                metricsHtml = `
                    <div class="metrics-container mt-2">
                        <div class="row">
                            <div class="col-md-3">
                                <small>Doğruluk: <strong>${version.metrics.accuracy ? (version.metrics.accuracy*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-3">
                                <small>Kesinlik: <strong>${version.metrics.precision ? (version.metrics.precision*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-3">
                                <small>Duyarlılık: <strong>${version.metrics.recall ? (version.metrics.recall*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-3">
                                <small>F1: <strong>${version.metrics.f1 ? (version.metrics.f1*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                        </div>
                    </div>
                `;
            } else { // age model
                metricsHtml = `
                    <div class="metrics-container mt-2">
                        <div class="row">
                            <div class="col-md-4">
                                <small>MAE: <strong>${version.metrics.mae ? version.metrics.mae.toFixed(1) + ' yaş' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-4">
                                <small>±3 Yaş Doğruluğu: <strong>${version.metrics.accuracy ? (version.metrics.accuracy*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-4">
                                <small>Örnek Sayısı: <strong>${version.metrics.count || 'N/A'}</strong></small>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        // Eğitim bilgilerini hazırla
        const trainingInfo = `
            <div class="training-info mt-1">
                <small class="text-muted">
                    ${version.training_samples || 0} eğitim, ${version.validation_samples || 0} doğrulama örneği,
                    ${version.epochs || 0} epoch (${new Date(version.created_at).toLocaleString()})
                </small>
            </div>
        `;
        
        versionItem.innerHTML = `
            <div class="d-flex w-100 justify-content-between align-items-center">
                <h6 class="mb-1">Model Versiyonu ${version.version}</h6>
                <div>
                    ${version.is_active 
                        ? '<span class="badge bg-success">Aktif</span>' 
                        : '<span class="badge bg-secondary">Pasif</span>'
                    }
                </div>
            </div>
            ${metricsHtml}
            ${trainingInfo}
        `;
        
        versionsList.appendChild(versionItem);
    });
    
    container.appendChild(versionsList);
    
    // Sıfırlama butonu ekle (sadece yaş modeli için)
    if (modelType === 'age') {
        const resetButton = document.createElement('button');
        resetButton.className = 'btn btn-danger mt-3';
        resetButton.innerHTML = '<i class="fas fa-undo-alt me-2"></i>Modeli Sıfırla';
        resetButton.onclick = () => confirmModelReset(modelType);
        container.appendChild(resetButton);
    }
}

// Model versiyonunu aktifleştir
function activateModelVersion(versionId, modelType) {
    if (!confirm(`Model versiyonunu aktifleştirmek istediğinizden emin misiniz?`)) {
        return;
    }
    
    fetch('/api/model/activate/' + versionId, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showToast('Başarılı', `Model versiyonu başarıyla aktifleştirildi.`, 'success');
            // Metrikleri yenile
            loadModelMetrics();
        } else {
            showToast('Hata', `Model aktifleştirilemedi: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('Model aktifleştirme hatası:', error);
        showToast('Hata', `Model aktifleştirilemedi: ${error.message}`, 'danger');
    });
}

// Model sıfırlama onayı
function confirmModelReset(modelType) {
    if (!confirm(`${modelType === 'content' ? 'İçerik analiz' : 'Yaş tahmin'} modelini sıfırlamak istediğinizden emin misiniz? Bu işlem geri alınamaz.`)) {
        return;
    }
    
    resetModel(modelType);
}



// Modeli sıfırla
function resetModel(modelType) {
    fetch('/api/model/reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_type: modelType
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Model sıfırlama başarılı:', data);
        showToast('Başarılı', `${modelType === 'content' ? 'İçerik analiz' : 'Yaş tahmin'} modeli başarıyla sıfırlandı.`, 'success');
        
        // Metrikleri yenile
        loadModelMetrics();
    })
    .catch(error => {
        console.error('Model sıfırlama hatası:', error);
        showToast('Hata', `Model sıfırlanırken hata oluştu: ${error.message}`, 'danger');
    });
}

// Geri bildirimlerle model eğitimi başlat
function startTrainingWithFeedback() {
    // Eğitim modelini ve parametreleri al
    const modelType = document.getElementById('modelType').value;
    const epochCount = parseInt(document.getElementById('epochCount').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    
    // Eğitim durumu bölümünü göster
    document.querySelector('.training-info').style.display = 'block';
    document.getElementById('trainingResultsSection').style.display = 'none';
    
    // Eğitim durumunu sıfırla
    const progressBar = document.getElementById('trainingProgressBar');
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';
    progressBar.setAttribute('aria-valuenow', 0);
    
    // Durum metnini güncelle
    document.getElementById('trainingStatusText').textContent = 'Geri bildirim verileri hazırlanıyor...';
    
    // Eğitim butonunu devre dışı bırak
    document.getElementById('startTrainingBtn').disabled = true;
    
    // Eğitim isteği gönder
    fetch('/api/model/train-with-feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_type: modelType,
            epochs: epochCount,
            batch_size: batchSize,
            learning_rate: learningRate
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            console.log('Eğitim tamamlandı:', data);
            document.getElementById('trainingStatusText').textContent = 'Eğitim tamamlandı.';
            
            // İlerleme çubuğunu güncelle
            progressBar.style.width = '100%';
            progressBar.textContent = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            
            // Eğitim sonuçlarını göster
            displayTrainingResults(data);
            
            // Eğitim butonunu aktif et
            document.getElementById('startTrainingBtn').disabled = false;
            
            // Metrikleri yenile
            loadModelMetrics();
        } else {
            throw new Error(data.message || 'Bilinmeyen bir hata oluştu');
        }
    })
    .catch(error => {
        console.error('Eğitim başlatma hatası:', error);
        document.getElementById('trainingStatusText').textContent = `Eğitim başlatılamadı: ${error.message}`;
        document.getElementById('startTrainingBtn').disabled = false;
        showToast('Hata', `Eğitim başlatılırken hata oluştu: ${error.message}`, 'danger');
    });
}

// Eğitim butonunun işlevini güncelle
function setupTrainingButton() {
    const trainingBtn = document.getElementById('startTrainingBtn');
    if (trainingBtn) {

        trainingBtn.addEventListener('click', startTrainingWithFeedback);
    }
}

// Socket.io eğitim ilerleme güncellemesi
function updateTrainingProgress(data) {
    const { progress, message } = data;
    
    // İlerleme çubuğunu güncelle
    const progressBar = document.getElementById('trainingProgressBar');
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
    // Durum metnini güncelle
    document.getElementById('trainingStatusText').textContent = message || `${progress}% tamamlandı`;
}

// Socket.io eğitim tamamlandı
function handleTrainingCompleted(data) {
    // Eğitim sonuçlarını göster
    displayTrainingResults(data.results);
    
    // Butonları aktif et (ama gizli tut)
    document.getElementById('startTrainingBtn').disabled = false;
    
    // Başarı mesajı göster
    showToast('Başarılı', 'Model eğitimi başarıyla tamamlandı.', 'success');
}



// Süre formatla
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)} saniye`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes} dakika ${remainingSeconds} saniye`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours} saat ${minutes} dakika`;
    }
}

// Hata mesajını göster
function showError(message) {
    // Toast kullanarak hata mesajını göster
    showToast('Hata', message, 'danger');
    
    // Konsola da kaydet
    console.error(message);
}

// Dosya adını fileId'den çıkar
function fileNameFromId(fileId) {
    const file = uploadedFiles.find(f => f.id === fileId);
    if (file) {
        return file.name;
    }
    return "Bilinmeyen dosya";
}

// 1. Yüksek riskli kare URL oluşturma fonksiyonunu düzeltme
function getFrameUrl(frame, analysisId, fileId, fileType) {
    // Sadece processed_image_path varsa URL döndür
    if (frame && frame.startsWith('processed/')) {
        return `/api/files/${frame}`;
    }
    
    // Diğer tüm durumlarda null döndür
    return null;
}

// Yüksek riskli kare görüntüleme kısmını düzelt
function displayHighestRiskFrame(results) {
    console.log(`En yüksek riskli kare gösteriliyor (${results.file_name}):`, results.highest_risk);
    
    const container = document.getElementById('highestRiskFrameContainer');
    if (!container) return;
    
    if (results.highest_risk && results.highest_risk.processed_image_path) {
        const frameUrl = `/api/files/${results.highest_risk.processed_image_path}`;
        console.log(`Yüksek riskli kare URL'si:`, frameUrl);
        
        const highestRiskFrame = document.createElement('img');
        highestRiskFrame.className = 'img-fluid highest-risk-frame';
        highestRiskFrame.alt = 'En yüksek riskli kare';
        highestRiskFrame.src = frameUrl;
        
        container.innerHTML = '';
        container.appendChild(highestRiskFrame);
        
        // Kategori ve skor bilgisini ekle
        if (results.highest_risk.category && results.highest_risk.score) {
            const categoryName = getCategoryDisplayName(results.highest_risk.category);
            const scoreLabel = document.createElement('div');
            scoreLabel.className = 'position-absolute bottom-0 end-0 bg-danger text-white px-2 py-1 rounded-start';
            scoreLabel.innerHTML = `${categoryName}: ${Math.round(results.highest_risk.score)}%`;
            container.appendChild(scoreLabel);
            
            // Zaman bilgisi varsa ekle
            if (results.highest_risk.timestamp) {
                const timeLabel = document.createElement('div');
                timeLabel.className = 'position-absolute bottom-0 start-0 bg-dark text-white px-2 py-1 rounded-end';
                timeLabel.innerHTML = formatTime(results.highest_risk.timestamp);
                container.appendChild(timeLabel);
            }
        }
    } else {
        container.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>İşlenmiş (overlay\'li) görsel bulunamadı.</div>';
    }
}

// Kategori bazlı yüksek riskli kareleri düzeltme
function displayHighRiskFramesByCategory(results) {
    console.log("Tespit edilen skorlar:", results.overall_scores);
    
    const grid = document.getElementById('categoryFramesGrid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    // Yeni sistem: category_specific_highest_risks_data kullan
    let categorySpecificHighestRisks = {};
    if (results.category_specific_highest_risks_data) {
        try {
            categorySpecificHighestRisks = JSON.parse(results.category_specific_highest_risks_data);
            console.log('[DEBUG] Using category_specific_highest_risks_data:', categorySpecificHighestRisks);
        } catch (e) {
            console.error("Error parsing category_specific_highest_risks_data:", e);
            // Fallback to old method
            categorySpecificHighestRisks = null;
        }
    }

    // Eğer yeni sistem verisi varsa onu kullan, yoksa eski yöntemi kullan
    if (categorySpecificHighestRisks) {
        // YENİ SİSTEM: Backend'den gelen category_specific_highest_risks_data
        const categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug', 'safe'];
        
        categories.forEach(category => {
            const categoryData = categorySpecificHighestRisks[category];
            if (!categoryData || categoryData.score <= 0) return;
            
            // Güvenli kategori için farklı eşik değeri (en az %50)
            const threshold = category === 'safe' ? 0.5 : 0.3;
            
            if (categoryData.score < threshold) return;
            
            // UI için skorları yüzdelik sisteme dönüştür
            const score = categoryData.score;
            const frameUrl = `/api/files/${normalizePath(categoryData.frame_path)}`;
            
            const categoryName = getCategoryDisplayName(category);
            let badgeClass = getCategoryBadgeClass(category);
            
            const cardDiv = document.createElement('div');
            cardDiv.className = 'col-lg-4 col-md-6 mb-4';
            
            console.log('[LOG][FRONTEND] Kategori kartı oluşturuluyor:', {
                category, 
                score: score,
                timestamp: categoryData.timestamp,
                frame_path: categoryData.frame_path,
                frameUrl: frameUrl
            });
            
            cardDiv.innerHTML = `
                <div class="card h-100">
                    <div class="position-relative">
                        <div style="height: 240px; overflow: hidden;">
                            <img src="${frameUrl}" class="card-img-top detection-img" alt="${categoryName}" 
                                style="width: 100%; height: 100%; object-fit: cover;"
                                onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';">
                        </div>
                        <span class="position-absolute top-0 end-0 m-2 badge ${badgeClass}">${categoryName}</span>
                        ${categoryData.timestamp !== null && categoryData.timestamp !== undefined ? `<span class="position-absolute bottom-0 start-0 m-2 badge bg-dark">${formatTime(categoryData.timestamp)}</span>` : ''}
                    </div>
                    <div class="card-body">
                        <h6 class="card-title">${categoryName}</h6>
                        <div class="d-flex justify-content-between mb-1">
                            <span>${category === 'safe' ? 'Güven Skoru:' : 'Risk Skoru:'}</span>
                            <strong>${Math.round(score * 100)}%</strong>
                        </div>
                        <div class="progress">
                            <div class="progress-bar ${badgeClass}" 
                                style="width: ${score * 100}%" 
                                role="progressbar" 
                                aria-valuenow="${score * 100}" 
                                aria-valuemin="0" 
                                aria-valuemax="100">
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            grid.appendChild(cardDiv);
        });
    } else {
        // ESKİ SİSTEM: Fallback
        console.log("Fallback to old detection method");
        
        // En yüksek skorları ve kare bilgilerini saklayacak objeler
        let highestScores = {
            violence: 0,
            adult_content: 0,
            harassment: 0,
            weapon: 0,
            drug: 0,
            safe: 0
        };
        
        let highestFrames = {
            violence: null,
            adult_content: null,
            harassment: null,
            weapon: null,
            drug: null,
            safe: null
        };
        
        // İçerik tespitlerini gözden geçir ve en yüksek skorları bul
        if (results.content_detections && results.content_detections.length > 0) {
            results.content_detections.forEach(detection => {
                // Her kategori için en yüksek skoru kontrol et
                if (detection.violence_score > highestScores.violence) {
                    highestScores.violence = detection.violence_score;
                    highestFrames.violence = {
                        processed_image_path: detection.processed_image_path,
                        timestamp: detection.frame_timestamp
                    };
                }
                
                if (detection.adult_content_score > highestScores.adult_content) {
                    highestScores.adult_content = detection.adult_content_score;
                    highestFrames.adult_content = {
                        processed_image_path: detection.processed_image_path,
                        timestamp: detection.frame_timestamp
                    };
                }
                
                if (detection.harassment_score > highestScores.harassment) {
                    highestScores.harassment = detection.harassment_score;
                    highestFrames.harassment = {
                        processed_image_path: detection.processed_image_path,
                        timestamp: detection.frame_timestamp
                    };
                }
                
                if (detection.weapon_score > highestScores.weapon) {
                    highestScores.weapon = detection.weapon_score;
                    highestFrames.weapon = {
                        processed_image_path: detection.processed_image_path,
                        timestamp: detection.frame_timestamp
                    };
                }
                
                if (detection.drug_score > highestScores.drug) {
                    highestScores.drug = detection.drug_score;
                    highestFrames.drug = {
                        processed_image_path: detection.processed_image_path,
                        timestamp: detection.frame_timestamp
                    };
                }
                
                if (detection.safe_score > highestScores.safe) {
                    highestScores.safe = detection.safe_score;
                    highestFrames.safe = {
                        processed_image_path: detection.processed_image_path,
                        timestamp: detection.frame_timestamp
                    };
                }
            });
        }
        
        console.log("Fallback: Bulunan en yüksek kategoriler:", highestFrames);
        
        // Her kategori için en yüksek riskli kareyi göster
        const categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug', 'safe'];
        
        categories.forEach(category => {
            // Güvenli kategori için farklı eşik değeri (en az %50)
            const threshold = category === 'safe' ? 0.5 : 0.3;
            
            if (highestScores[category] >= threshold) { 
                const frameData = highestFrames[category];
                if (!frameData || !frameData.processed_image_path) return;
                
                let categoryName = getCategoryDisplayName(category);
                const cardDiv = document.createElement('div');
                cardDiv.className = 'col-lg-4 col-md-6 mb-4';
                
                const frameUrl = `/api/files/${normalizePath(frameData.processed_image_path)}`;
                let badgeClass = getCategoryBadgeClass(category);
                
                cardDiv.innerHTML = `
                    <div class="card h-100">
                        <div class="position-relative">
                            <div style="height: 240px; overflow: hidden;">
                                <img src="${frameUrl}" class="card-img-top detection-img" alt="${categoryName}" 
                                    style="width: 100%; height: 100%; object-fit: cover;"
                                    onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';">
                            </div>
                            <span class="position-absolute top-0 end-0 m-2 badge ${badgeClass}">${categoryName}</span>
                            ${frameData.timestamp ? `<span class="position-absolute bottom-0 start-0 m-2 badge bg-dark">${formatTime(frameData.timestamp)}</span>` : ''}
                        </div>
                        <div class="card-body">
                            <h6 class="card-title">${categoryName}</h6>
                            <div class="d-flex justify-content-between mb-1">
                                <span>${category === 'safe' ? 'Güven Skoru:' : 'Risk Skoru:'}</span>
                                <strong>${Math.round(highestScores[category] * 100)}%</strong>
                            </div>
                            <div class="progress">
                                <div class="progress-bar ${badgeClass}" 
                                    style="width: ${highestScores[category] * 100}%" 
                                    role="progressbar" 
                                    aria-valuenow="${highestScores[category] * 100}" 
                                    aria-valuemin="0" 
                                    aria-valuemax="100">
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                grid.appendChild(cardDiv);
            }
        });
    }
    
    // Eğer hiç kart eklenmemişse bilgi mesajı göster
    if (grid.children.length === 0) {
        grid.innerHTML = '<div class="col-12"><div class="alert alert-info">Bu dosyada önemli içerik tespiti yapılmadı.</div></div>';
    }
}

// Yaş tahminleri görüntüleme fonksiyonu - Sadeleştirilmiş versiyon
function displayAgeEstimations(results) {
    console.log("[DEBUG] displayAgeEstimations başladı:", results);

    // Yaş tahminleri olup olmadığını kontrol et
    if (!results || !results.age_estimations) {
        console.warn("[DEBUG] Yaş tahminleri bulunamadı:", results);
        const ageContainer = document.getElementById('ageEstimationsContainer');
        if (ageContainer) {
            ageContainer.innerHTML = '<div class="alert alert-warning">Yaş tahminleri bulunamadı veya dosya formatı hatalı.</div>';
        }
        return;
    }

    const ageContainer = document.getElementById('ageEstimationsContainer');
    if (!ageContainer) {
        console.error('[DEBUG] ageEstimationsContainer bulunamadı!');
        return;
    }

    try {
        console.log("[DEBUG] Yaş tahminlerini işlemeye başlıyorum...");
        
        // Benzersiz yüzleri bul
        const faces = {};
        results.age_estimations.forEach(item => {
            const faceId = item.person_id || item.face_id || 'unknown';
            const confidence = item.confidence_score || item.confidence || 0;
            
            console.log(`[DEBUG] Yüz işleniyor - ID: ${faceId}, Confidence: ${confidence}`);
            console.log("[DEBUG] Tam veri:", item);
            
            if (!faces[faceId] || confidence > faces[faceId].confidence) {
                faces[faceId] = {
                    age: item.estimated_age || 'Bilinmiyor',
                    confidence: confidence,
                    processed_image_path: item.processed_image_path || null
                };
                console.log(`[DEBUG] Yüz kaydedildi/güncellendi:`, faces[faceId]);
            }
        });

        // Her yüz için kart oluştur
        const faceIds = Object.keys(faces);
        console.log('[DEBUG] Tespit edilen toplam benzersiz yüz sayısı:', faceIds.length);

        if (faceIds.length === 0) {
            console.warn('[DEBUG] Hiç yüz tespit edilmedi');
            ageContainer.innerHTML = '<div class="alert alert-info">Bu içerikte tespit edilen yüz bulunmamaktadır.</div>';
            return;
        }

        // Container'ı temizle
        ageContainer.innerHTML = '';

        // Her yüz için kart oluştur
        const row = document.createElement('div');
        row.className = 'row';
        ageContainer.appendChild(row);

        faceIds.forEach((faceId, index) => {
            const face = faces[faceId];
            console.log(`[DEBUG] Yüz kartı oluşturuluyor - Index: ${index}, FaceID: ${faceId}`);
            console.log("[DEBUG] Yüz verisi:", face);

            const col = document.createElement('div');
            col.className = 'col-md-6 mb-4';
            
            // Görsel URL'sini oluştur
            let frameUrl = '';
            if (face.processed_image_path) {
                frameUrl = `/api/files/${normalizePath(face.processed_image_path).replace(/^\/+|\/+/g, '/')}`;
                console.log("[DEBUG] İşlenmiş görsel URL'si:", frameUrl);
                
                col.innerHTML = `
                    <div class="card h-100">
                        <div class="card-body">
                            <div class="row align-items-center">
                                <div class="col-md-12">
                                    <div class="position-relative" style="height: 300px; overflow: hidden;">
                                        <img src="${frameUrl}" 
                                             alt="ID: ${faceId.includes('_person_') ? faceId.split('_person_').pop() : index + 1}"
                                             style="width: 100%; height: 100%; object-fit: contain;"
                                             onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';"
                                             onload="console.log('[DEBUG] Görsel başarıyla yüklendi:', this.src)">
                                        <span class="position-absolute top-0 end-0 m-2 badge bg-info">ID: ${faceId.includes('_person_') ? faceId.split('_person_').pop() : index + 1}</span>
                                    </div>
                                    <div class="mt-3">
                                        <h5 class="card-title mb-3">Tahmini Yaş: ${Math.round(face.age)}</h5>
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
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                console.warn("[DEBUG] İşlenmiş görsel bulunamadı - FaceID:", faceId);
                col.innerHTML = `
                    <div class="card h-100">
                        <div class="card-body">
                            <div class="alert alert-warning">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                İşlenmiş (overlay'li) görsel bulunamadı.
                            </div>
                            <h5 class="card-title mb-3">Tahmini Yaş: ${Math.round(face.age)}</h5>
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
                        </div>
                    </div>
                `;
            }
            
            console.log("[DEBUG] Kart DOM'a ekleniyor");
            row.appendChild(col);
            console.log("[DEBUG] Kart DOM'a eklendi");
        });

    } catch (e) {
        console.error('[DEBUG] Yaş tahminleri gösterilirken hata:', e);
        console.error('[DEBUG] Hata stack:', e.stack);
        ageContainer.innerHTML = `<div class="alert alert-danger">Yaş tahminleri işlenirken hata oluştu: ${e.message}</div>`;
    }
}

// Yaş geri bildirimi görüntüleme fonksiyonu - Sadeleştirilmiş versiyon
// displayAgeFeedback fonksiyonunu güncelliyoruz: results objesinden analysis_id alacak
function displayAgeFeedback(feedbackTab, results) { // results objesi analysis_id ve frame_path içermeli
    if (!feedbackTab || !results.age_estimations || !results.age_estimations.length) {
        // Eğer yaş tahmini yoksa mesaj göster ve geri bildirim alanını temizle/gizle
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
    ageFeedbackContainer.innerHTML = ''; // Mevcut içeriği temizle

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
                // frame_path için de processed_image_path'i önceliklendir, eğer yoksa item.frame_path'e fallback yap
                frame_path: item.processed_image_path || item.frame_path || null, 
                face_image_src: item.face_image_path || item.processed_image_path || '/static/img/placeholder-face.png' 
            });
        }
    });

    let personCounter = 0; // Kişi sayacı eklendi
    facesMap.forEach((face, personId) => {
        personCounter++; // Sayaç artırıldı
        const templateClone = ageFeedbackTemplate.content.cloneNode(true);
        const feedbackItem = templateClone.querySelector('.age-feedback-item');
        
        const faceImageElement = feedbackItem.querySelector('.face-image');
        if (faceImageElement) {
            // Görsel yolunu /api/files/ ile başlatacak şekilde düzelt
            let imgSrc = face.face_image_src;
            if (imgSrc && !imgSrc.startsWith('/api/files/') && !imgSrc.startsWith('http') && !imgSrc.startsWith('/static/')) {
                imgSrc = '/api/files/' + imgSrc.replace(/^\/+/, '');
            }
            faceImageElement.src = imgSrc;
            faceImageElement.alt = `Kişi ${personCounter}`;
            faceImageElement.style.cursor = 'pointer';
            faceImageElement.title = 'Büyütmek için tıklayın';
        }
        
        const personIdElement = feedbackItem.querySelector('.person-id');
        if (personIdElement) {
            personIdElement.textContent = personCounter; // Sıralı numara atandı
        }
        
        const estimatedAgeElement = feedbackItem.querySelector('.estimated-age');
        if (estimatedAgeElement) {
            estimatedAgeElement.textContent = face.age;
        }
        
        const correctedAgeInput = feedbackItem.querySelector('.corrected-age');
        if (correctedAgeInput) {
            // Set data attributes on the input field
            correctedAgeInput.dataset.personId = personId;
            correctedAgeInput.dataset.analysisId = analysisId; // analysis_id from the main results
            correctedAgeInput.dataset.framePath = face.frame_path || ''; // original frame_path for this specific face
        }
        
        // Remove individual submit button if it exists in the template
        const individualSubmitButton = feedbackItem.querySelector('.age-feedback-submit');
        if (individualSubmitButton) {
            individualSubmitButton.remove();
        }
        
        ageFeedbackContainer.appendChild(feedbackItem);
    });
}

// ... (rest of the code remains unchanged)

// Model Yönetimi Modal JavaScript fonksiyonları
let modalTrainingInterval = null;
let modalQueueStatusInterval = null;

// Model Yönetimi Modal açıldığında çalışacak fonksiyon
function initializeModelManagementModal() {
    console.log('Initializing Model Management Modal...');
    
    // Ana sayfa queue checker'ını durdur
    stopQueueStatusChecker();
    
    // Önce butonları aktif et (varsayılan olarak)
    const trainButtons = document.querySelectorAll('[onclick*="trainModelFromModal"]');
    const resetButtons = document.querySelectorAll('[onclick*="resetModelFromModal"]');
    const deleteButtons = document.querySelectorAll('[onclick*="deleteLatestModelVersion"]');
    
    // Tüm butonları başlangıçta aktif yap
    [...trainButtons, ...resetButtons, ...deleteButtons].forEach(btn => {
        if (btn) {
            btn.disabled = false;
            btn.classList.remove('disabled');
            btn.title = '';
        }
    });
    
    loadModalModelVersions();
    loadModalModelStats();
    startModalQueueStatusChecker();
}

// Model Yönetimi Modal kapatıldığında çalışacak fonksiyon
function cleanupModelManagementModal() {
    if (modalQueueStatusInterval) {
        clearInterval(modalQueueStatusInterval);
        modalQueueStatusInterval = null;
    }
    
    // Ana sayfa queue checker'ını yeniden başlat
    startQueueStatusChecker();
}

// Modal kuyruk durumu kontrolünü başlat
function startModalQueueStatusChecker() {
    // İlk kontrol
    checkModalQueueStatus();
    
    // 10 saniyede bir kontrol et (rate limiting için azaltıldı)
    modalQueueStatusInterval = setInterval(checkModalQueueStatus, 10000);
}

// Modal kuyruk durumunu kontrol et
function checkModalQueueStatus() {
    // Sadece kuyruk durumunu al, dosya sayısını frontend'den kullan
    fetch('/api/queue/status')
    .then(response => response.json())
    .then(response => {
        // Backend response formatını parse et
        const queueData = response?.data || response;
        
        // Frontend'deki dosya sayısını kullan
        const frontendUploadedFiles = uploadedFiles.length;
        const uploadedFilesData = {
            uploaded_files_count: frontendUploadedFiles
        };
        
        updateModalButtonsState(queueData, uploadedFilesData);
    })
    .catch(error => {
        console.error('Modal kuyruk durumu kontrol hatası:', error);
        // Hata durumunda butonları aktif et
        updateModalButtonsState({queue_size: 0, is_processing: false}, {uploaded_files_count: 0});
    });
}

// Modal butonlarının durumunu güncelle
function updateModalButtonsState(queueData, uploadedFilesData) {
    console.log('Modal - Kuyruk durumu:', queueData);
    console.log('Modal - Yüklü dosya durumu:', uploadedFilesData);
    
    // Ana sayfadaki mantık: Yüklü dosya varsa veya kuyrukta dosya varsa veya aktif analiz varsa devre dışı bırak
    const hasUploadedFiles = uploadedFilesData.uploaded_files_count > 0;
    const hasFilesInQueue = queueData.queue_size > 0 || queueData.is_processing === true;
    const shouldDisableButtons = hasUploadedFiles || hasFilesInQueue;
    
    console.log('Modal - Ana sayfada yüklü dosya var mı?', hasUploadedFiles);
    console.log('Modal - Kuyrukta dosya var mı?', hasFilesInQueue);
    console.log('Modal - Butonlar devre dışı mı?', shouldDisableButtons);
    
    // Modal içindeki tüm model yönetimi butonlarını bul
    const trainButtons = document.querySelectorAll('[onclick*="trainModelFromModal"]');
    const resetButtons = document.querySelectorAll('[onclick*="resetModelFromModal"]');
    const activateButtons = document.querySelectorAll('[onclick*="activateVersionFromModal"]');
    
    if (shouldDisableButtons) {
        // Dosya yüklü veya kuyrukta dosya varken butonları devre dışı bırak
        trainButtons.forEach(btn => {
            btn.disabled = true;
            btn.classList.add('disabled');
            btn.title = 'Dosya yüklü veya analiz devam ederken model eğitimi yapılamaz';
        });
        
        resetButtons.forEach(btn => {
            btn.disabled = true;
            btn.classList.add('disabled');
            btn.title = 'Dosya yüklü veya analiz devam ederken model sıfırlanamaz';
        });
        
        activateButtons.forEach(btn => {
            btn.disabled = true;
            btn.classList.add('disabled');
            btn.title = 'Dosya yüklü veya analiz devam ederken model değiştirilemez';
        });
        
    } else {
        // Dosya yüklü değil ve analiz yokken butonları aktif et
        trainButtons.forEach(btn => {
            btn.disabled = false;
            btn.classList.remove('disabled');
            btn.title = '';
        });
        
        resetButtons.forEach(btn => {
            btn.disabled = false;
            btn.classList.remove('disabled');
            btn.title = '';
        });
        
        activateButtons.forEach(btn => {
            btn.disabled = false;
            btn.classList.remove('disabled');
            btn.title = '';
        });
    }
}

// Modal model versiyonlarını yükle
async function loadModalModelVersions() {
    try {
        // Yaş modeli versiyonları
        const ageResponse = await fetch('/api/model/versions/age');
        if (ageResponse.ok) {
            const ageData = await ageResponse.json();
            console.log('Modal Age API Response:', ageData);
            
            const ageVersions = ageData.versions || [];
            console.log('Modal Age Versions:', ageVersions);
            displayModalVersions('age', ageVersions);
        } else {
            console.error('Modal Age API Error:', ageResponse.status, ageResponse.statusText);
            document.getElementById('modal-age-versions').innerHTML = '<span class="text-danger">API hatası</span>';
        }

        // İçerik modeli versiyonları
        const contentResponse = await fetch('/api/model/versions/content');
        if (contentResponse.ok) {
            const contentData = await contentResponse.json();
            console.log('Modal Content API Response:', contentData);
            
            const contentVersions = contentData.versions || [];
            console.log('Modal Content Versions:', contentVersions);
            displayModalVersions('content', contentVersions);
        } else {
            console.error('Modal Content API Error:', contentResponse.status, contentResponse.statusText);
            document.getElementById('modal-content-versions').innerHTML = '<span class="text-danger">API hatası</span>';
        }
    } catch (error) {
        console.error('Modal model versiyonları yüklenirken hata:', error);
        document.getElementById('modal-age-versions').innerHTML = '<span class="text-danger">Yükleme hatası</span>';
    }
}

// Modal model istatistiklerini yükle
async function loadModalModelStats() {
    try {
        // Yaş modeli istatistikleri
        const ageResponse = await fetch('/api/model/metrics/age');
        if (ageResponse.ok) {
            const ageStats = await ageResponse.json();
            updateModalModelStats('age', ageStats);
        }

        // İçerik modeli istatistikleri
        const contentResponse = await fetch('/api/model/metrics/content');
        if (contentResponse.ok) {
            const contentStats = await contentResponse.json();
            updateModalModelStats('content', contentStats);
        }
    } catch (error) {
        console.error('Modal model istatistikleri yüklenirken hata:', error);
    }
}

// Modal'daki model versiyonlarını göster (tıklanabilir)
function displayModalVersions(modelType, versions) {
    const containerId = `modal-${modelType}-versions`;
    const container = document.getElementById(containerId);
    
    if (!container) {
        console.error(`Container not found: ${containerId}`);
        return;
    }
    
    if (!versions || versions.length === 0) {
        container.innerHTML = '<span class="badge bg-secondary">Versiyon bulunamadı</span>';
        return;
    }
    
    // Versiyonları sırala (en yeni önce)
    const sortedVersions = versions.sort((a, b) => b.version - a.version);
    
    let html = '';
    sortedVersions.forEach((version, index) => {
        const badgeClass = version.is_active ? 'bg-success' : 'bg-secondary';
        const activeText = version.is_active ? ' (Aktif)' : '';
        const isLatest = index === 0;
        
        // Version display
        let versionDisplay = '';
        if (version.version === 0) {
            versionDisplay = modelType === 'content' ? 'Base OpenCLIP' : 'Base Model';
        } else {
            versionDisplay = `v${version.version}`;
        }
        
        html += `
            <span class="badge ${badgeClass} version-badge me-2 mb-2 clickable-version" 
                  data-version-id="${version.id}" 
                  data-model-type="${modelType}"
                  title="${version.metrics && version.metrics.mae ? `MAE: ${version.metrics.mae.toFixed(2)} yaş` : 'Versiyon seç'}"
                  style="cursor: pointer;">
                ${versionDisplay}${activeText}${isLatest ? ' (En Son)' : ''}
            </span>
        `;
    });
    
    container.innerHTML = html;
    
    // Versiyon seçme olayları ekle
    const versionBadges = container.querySelectorAll('.clickable-version');
    versionBadges.forEach(badge => {
        badge.addEventListener('click', function() {
            const versionId = this.dataset.versionId;
            const modelType = this.dataset.modelType;
            activateModelVersionFromModal(versionId, modelType);
        });
    });
    
    // Silme butonunu güncelle
    if (modelType === 'age' || modelType === 'content') {
        updateDeleteButton(modelType, sortedVersions);
    }
}

// Modal'dan model versiyonu aktifleştir
function activateModelVersionFromModal(versionId, modelType) {
    if (!confirm(`Bu model versiyonunu aktifleştirmek istediğinizden emin misiniz?`)) {
        return;
    }
    
    showModalTrainingStatus('Model versiyonu aktifleştiriliyor...', 'info');
    
    fetch(`/api/model/activate/${versionId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showModalTrainingStatus(`Model versiyonu başarıyla aktifleştirildi!`, 'success');
            
            // Model versiyonlarını ve istatistikleri yenile
            setTimeout(() => {
                loadModalModelVersions();
                loadModalModelStats();
                hideModalTrainingStatus();
            }, 2000);
            
            showToast('Başarılı', `Model versiyonu aktifleştirildi.`, 'success');
        } else {
            showModalTrainingStatus(`Model aktifleştirilemedi: ${data.message}`, 'danger');
            setTimeout(hideModalTrainingStatus, 3000);
        }
    })
    .catch(error => {
        console.error('Model aktifleştirme hatası:', error);
        showModalTrainingStatus(`Model aktifleştirilemedi: ${error.message}`, 'danger');
        setTimeout(hideModalTrainingStatus, 3000);
    });
}

// Silme butonunu güncelle
function updateDeleteButton(modelType, versions) {
    const deleteBtn = document.getElementById('deleteLatestVersionBtn');
    if (deleteBtn) {
        const latestVersion = versions[0];
        // Base model (v0) veya aktif versiyon veya sadece 1 versiyon varsa silme butonunu devre dışı bırak
        if (latestVersion.version === 0 || latestVersion.is_active || versions.length <= 1) {
            deleteBtn.disabled = true;
            if (latestVersion.version === 0) {
                deleteBtn.title = 'Base model (v0) silinemez';
            } else if (latestVersion.is_active) {
                deleteBtn.title = 'Aktif versiyon silinemez';
            } else {
                deleteBtn.title = 'En az bir versiyon bulunmalıdır';
            }
        } else {
            deleteBtn.disabled = false;
            deleteBtn.title = `v${latestVersion.version} versiyonunu sil`;
        }
    }
}

// Modal model istatistiklerini güncelle
function updateModalModelStats(modelType, stats) {
    console.log(`Modal - Updating ${modelType} stats:`, stats);
    
    if (modelType === 'age') {
        // Aktif versiyon güncelle
        const activeVersion = stats.age?.active_version || 'ensemble_v1';
        const versionDisplay = activeVersion === 'base_model' ? 'v0' : 'ensemble_v1';
        const versionElement = document.getElementById('modal-age-active-version');
        if (versionElement) {
            versionElement.textContent = versionDisplay;
        }
        
        // Durum güncelle
        const status = stats.age?.status || 'active';
        const statusElement = document.getElementById('modal-age-status');
        if (statusElement) {
            statusElement.innerHTML = '<i class="fas fa-check-circle text-success me-1"></i>Aktif';
        }
        
        // Geri bildirim sayısını güncelle
        const feedbackCount = stats.age?.feedback_count || 0;
        const trainingDataElement = document.getElementById('modal-age-training-data');
        if (trainingDataElement) {
            trainingDataElement.textContent = `${feedbackCount} örnek`;
        }
        
        // MAE bilgisini güncelle
        const maeElement = document.getElementById('modal-age-mae');
        if (maeElement && stats.age?.metrics?.mae) {
            maeElement.textContent = `${stats.age.metrics.mae.toFixed(2)} yaş`;
        }
    }
    
    if (modelType === 'content') {
        // Aktif versiyon güncelle
        const activeVersion = stats.content?.active_version || 'CLIP-v1.0';
        const versionDisplay = activeVersion.includes('v') ? activeVersion : 'CLIP-v1.0';
        const versionElement = document.getElementById('modal-clip-active-version');
        if (versionElement) {
            versionElement.textContent = versionDisplay;
        }
        
        // Durum güncelle
        const status = stats.content?.status || 'active';
        const statusElement = document.getElementById('modal-clip-status');
        if (statusElement) {
            statusElement.innerHTML = '<i class="fas fa-check-circle text-success me-1"></i>Aktif';
        }
        
        // Geri bildirim sayısını güncelle
        const feedbackCount = stats.content?.feedback_count || 0;
        const trainingDataElement = document.getElementById('modal-content-training-data');
        if (trainingDataElement) {
            trainingDataElement.textContent = `${feedbackCount} örnek`;
        }
    }
}

// Modal'dan model eğitimi başlat
function trainModelFromModal(modelType) {
    console.log(`trainModelFromModal called with modelType: ${modelType}`);
    
    // Global flag set et
    window.isModalTraining = true;
    
    const button = document.querySelector(`.btn-train-${modelType}`);
    const progressDiv = document.getElementById('modal-training-progress');
    
    if (!button || !progressDiv) {
        console.error('Required elements not found for modal training');
        return;
    }
    
    // UI durumunu ayarla
    button.disabled = true;
    
    if (modelType === 'age') {
        // Yaş modeli için ensemble refresh
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Corrections Yenileniyor...';
        refreshEnsembleCorrections();
        return;
    } else {
        // İçerik modeli için normal training
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Eğitim Başlatılıyor...';
    }
    
    progressDiv.style.display = 'block';
    progressDiv.classList.remove('d-none');
    
    const statusElement = document.getElementById('modal-training-status');
    if (statusElement) {
        statusElement.textContent = 'Eğitim başlatılıyor...';
        statusElement.className = 'alert alert-info';
    }
    
    const progressBar = document.getElementById('modal-progress-bar');
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', '0');
    }
    
    console.log('Modal UI elements configured, making API call');
    
    // API çağrısı (sadece content modeli için)
    fetch(`/api/model/train-web`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_type: modelType,
            epochs: 20,
            learning_rate: 0.001,
            batch_size: 1
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Modal training API response:', data);
        
        if (data.success) {
            // Global session tracking
            window.currentTrainingSessionId = data.session_id;
            console.log('Set global session ID for modal:', data.session_id);
            
            showModalTrainingStatus(`Eğitim başlatıldı! Session ID: ${data.session_id.substring(0, 8)}...`, 'info');
            
            // WebSocket training room'a katıl
            if (window.wsClient && window.wsClient.connected) {
                window.wsClient.joinTraining(data.session_id);
            } else {
                console.log('WebSocket henüz bağlı değil, room join atlandı');
            }
            
        } else {
            throw new Error(data.error || 'Eğitim başlatılamadı');
        }
    })
    .catch(error => {
        console.error('Modal training error:', error);
        
        // UI sıfırla
        button.disabled = false;
        button.innerHTML = `<i class="fas fa-play me-2"></i>Eğitimi Başlat`;
        
        progressDiv.style.display = 'none';
        window.isModalTraining = false;
        
        showModalTrainingStatus(`Hata: ${error.message}`, 'danger');
    });
}

// WebSocket sistemi kuruldu

// Modal training tamamlandı
function handleModalTrainingCompleted(data, modelType) {
    console.log('Modal training completed:', data);
    
    const progressDiv = document.getElementById('modal-training-progress');
    
    // Progress bar'ı 100% yap
    const progressBar = document.getElementById('modal-progress-bar');
    if (progressBar) {
        progressBar.style.width = '100%';
        progressBar.setAttribute('aria-valuenow', 100);
    }
    
    // Tamamlanma mesajı
    const metrics = data.metrics || {};
    let successMessage = `${modelType.toUpperCase()} eğitimi başarıyla tamamlandı!`;
    
    if (metrics.mae) {
        successMessage += ` (MAE: ${metrics.mae.toFixed(3)})`;
    } else if (metrics.accuracy) {
        successMessage += ` (Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%)`;
    }
    
    showModalTrainingStatus(successMessage, 'success');
    
    // Eğitim butonlarını aktif et
    const trainButtons = document.querySelectorAll('[onclick*="trainModelFromModal"]');
    trainButtons.forEach(btn => {
        btn.disabled = false;
        btn.innerHTML = `<i class="fas fa-play me-2"></i>Yeni Eğitim Başlat`;
    });
    
    // Model versiyonlarını ve istatistikleri yenile
    setTimeout(() => {
        loadModalModelVersions();
        loadModalModelStats();
        
        // Progress'i gizle
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
        
        window.isModalTraining = false;
    }, 3000);
    
    // Toast notification
    showToast('Başarılı', `${modelType.toUpperCase()} modeli eğitimi tamamlandı!`, 'success');
}

// Modal training error
function handleModalTrainingError(data, modelType) {
    console.error('Modal training error:', data);
    
    // UI sıfırla
    const button = document.querySelector(`.btn-train-${modelType}`);
    if (button) {
        button.disabled = false;
        button.innerHTML = `<i class="fas fa-play me-2"></i>Eğitimi Başlat`;
    }
    
    const progressDiv = document.getElementById('modal-training-progress');
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
    
    window.isModalTraining = false;
    
    showModalTrainingStatus(`Eğitim hatası: ${data.error_message || 'Bilinmeyen hata'}`, 'danger');
    showToast('Hata', `${modelType.toUpperCase()} eğitimi başarısız oldu`, 'error');
}

// Resim büyütme fonksiyonu
function zoomImage(imageSrc, imageTitle = 'Resim Görüntüleyici') {
    console.log('[DEBUG] zoomImage çağrıldı:', imageSrc, imageTitle);
    
    // Mevcut modal'ı kapat
    const existingModal = document.getElementById('imageZoomModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Yeni modal oluştur
    const modalHTML = `
        <div class="modal fade show" id="imageZoomModal" tabindex="-1" style="display: block; background: rgba(0,0,0,0.5); position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1050;">
            <div class="modal-dialog modal-lg" style="margin: 50px auto; max-width: 90%; width: 800px; position: relative;">
                <div class="modal-content" style="background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div class="modal-header" style="padding: 15px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center;">
                        <h5 class="modal-title" style="margin: 0;">${imageTitle}</h5>
                        <button type="button" class="btn-close" onclick="closeZoomModal()" style="background: none; border: none; font-size: 24px; cursor: pointer;">&times;</button>
                    </div>
                    <div class="modal-body" style="padding: 20px; text-align: center;">
                        <img src="${imageSrc}" alt="${imageTitle}" style="max-width: 100%; max-height: 70vh; height: auto; display: block; margin: 0 auto;">
                    </div>
                    <div class="modal-footer" style="padding: 15px; border-top: 1px solid #ddd; text-align: right;">
                        <button type="button" class="btn btn-secondary" onclick="closeZoomModal()" style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">Kapat</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Modal'ı sayfaya ekle
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Body scroll'unu engelle
    document.body.style.overflow = 'hidden';
    
    console.log('[DEBUG] Manuel modal oluşturuldu ve açıldı');
}
// Modal kapatma fonksiyonu
function closeZoomModal() {
    const modal = document.getElementById('imageZoomModal');
    if (modal) {
        modal.remove();
        console.log('[DEBUG] Manuel modal kapatıldı');
    }
    // Body scroll'unu geri getir
    document.body.style.overflow = '';
}

// Resim tıklama event listener'ını ekle
function addImageClickListeners() {
    // Tüm analiz sonuç resimlerine tıklama özelliği ekle
    document.addEventListener('click', function(e) {
        // Modal backdrop tıklamalarını atla
        if (e.target.classList.contains('modal-backdrop')) {
            return;
        }
        
        console.log('[DEBUG] Resim tıklama testi - Element:', e.target);
        console.log('[DEBUG] Element sınıfları:', e.target.classList);
        console.log('[DEBUG] Element tag:', e.target.tagName);
        
        // Yaş tahminleri resimleri
        if (e.target.matches('.age-estimations img, .age-feedback-container img, .face-image, .age-estimation-image')) {
            console.log('[DEBUG] Yaş tahmini resmi tıklandı!');
            e.preventDefault();
            e.stopPropagation();
            const imageSrc = e.target.src;
            const imageAlt = e.target.alt || 'Yaş Tahmini Resmi';
            console.log('[DEBUG] Resim zoom açılıyor:', imageSrc);
            zoomImage(imageSrc, imageAlt);
        }
        
        // İçerik tespiti resimleri
        if (e.target.matches('.content-detections img, .detection-img')) {
            console.log('[DEBUG] İçerik tespiti resmi tıklandı!');
            e.preventDefault();
            e.stopPropagation();
            const imageSrc = e.target.src;
            const imageAlt = e.target.alt || 'İçerik Tespiti Resmi';
            zoomImage(imageSrc, imageAlt);
        }
        
        // En yüksek riskli kare resimleri
        if (e.target.matches('.highest-risk-frame img, .risk-frame-img')) {
            console.log('[DEBUG] En yüksek riskli kare resmi tıklandı!');
            e.preventDefault();
            e.stopPropagation();
            const imageSrc = e.target.src;
            const imageAlt = e.target.alt || 'En Yüksek Riskli Kare';
            zoomImage(imageSrc, imageAlt);
        }
    });
}

// En son model versiyonunu sil
async function deleteLatestModelVersion(modelType) {
    // Önce mevcut versiyonları kontrol et
    try {
        const versionsResponse = await fetch(`/api/model/versions/${modelType}`);
        if (!versionsResponse.ok) {
            throw new Error('Model versiyonları alınamadı');
        }
        
        const versionsData = await versionsResponse.json();
        const versions = versionsData.versions || [];
        
        // Sadece 1 versiyon varsa silmeye izin verme
        if (versions.length <= 1) {
            alert('En az bir model versiyonu bulunmalıdır. Son versiyon silinemez!');
            return;
        }
        
        // En son versiyonun aktif olup olmadığını ve base model olup olmadığını kontrol et
        const sortedVersions = versions.sort((a, b) => b.version - a.version);
        const latestVersion = sortedVersions[0];
        
        // Base model (v0) silinemez
        if (latestVersion.version === 0) {
            alert('Base model (v0) silinemez! Bu model sistemin temel modelidir.');
            return;
        }
        
        if (latestVersion.is_active) {
            alert('Aktif model versiyonu silinemez! Önce başka bir versiyonu aktif yapın.');
            return;
        }
        
        // Silme onayı al
        const confirmMessage = `En son model versiyonu (v${latestVersion.version}) silinecek.\n\n` +
                              `Oluşturulma Tarihi: ${new Date(latestVersion.created_at).toLocaleString()}\n` +
                              `Eğitim Örnekleri: ${latestVersion.training_samples || 0}\n` +
                              (latestVersion.metrics && latestVersion.metrics.mae ? `MAE: ${latestVersion.metrics.mae.toFixed(2)} yaş\n` : '') +
                              '\nBu işlem geri alınamaz. Devam etmek istiyor musunuz?';
        
        if (!confirm(confirmMessage)) {
            return;
        }
        
        // Silme işlemini başlat
        showModalTrainingStatus('Model versiyonu siliniyor...', 'info');
        
        const deleteResponse = await fetch(`/api/model/delete-latest/${modelType}`, {
            method: 'DELETE'
        });
        
        const result = await deleteResponse.json();
        
        if (deleteResponse.ok && result.success) {
            showModalTrainingStatus(`Model versiyonu v${result.deleted_version.version} başarıyla silindi!`, 'success');
            
            // Model versiyonlarını yenile
            setTimeout(() => {
                loadModalModelVersions();
                loadModalModelStats();
                hideModalTrainingStatus();
            }, 2000);
        } else {
            showModalTrainingStatus(result.message || 'Model versiyonu silinirken hata oluştu', 'danger');
            setTimeout(hideModalTrainingStatus, 3000);
        }
    } catch (error) {
        console.error('Model silme hatası:', error);
        showModalTrainingStatus(`Model silme hatası: ${error.message}`, 'danger');
        setTimeout(hideModalTrainingStatus, 3000);
    }
}

// Modal eğitim durumu mesajını göster
function showModalTrainingStatus(message, type = 'info') {
    const statusDiv = document.getElementById('modal-training-status');
    const messageSpan = document.getElementById('modal-training-message');
    
    if (statusDiv && messageSpan) {
        // Alert sınıfını güncelle
        statusDiv.className = `alert alert-${type}`;
        
        // İkonu güncelle
        let icon = 'info-circle';
        switch(type) {
            case 'success': icon = 'check-circle'; break;
            case 'danger': icon = 'exclamation-triangle'; break;
            case 'warning': icon = 'exclamation-circle'; break;
        }
        
        messageSpan.innerHTML = `<i class="fas fa-${icon} me-2"></i>${message}`;
        statusDiv.style.display = 'block';
    }
}

// Modal eğitim durumu mesajını gizle
function hideModalTrainingStatus() {
    const statusDiv = document.getElementById('modal-training-status');
    if (statusDiv) {
        statusDiv.style.display = 'none';
    }
}

// ===============================
// WEB ARAYÜZÜ MODEL EĞİTİMİ
// ===============================

// Model türü değişiminde özel ayarları göster/gizle
document.addEventListener('DOMContentLoaded', function() {
    const trainingModelTypeSelect = document.getElementById('trainingModelType');
    const contentModelSettings = document.getElementById('contentModelSettings');
    
    if (trainingModelTypeSelect) {
        trainingModelTypeSelect.addEventListener('change', function() {
            if (this.value === 'content') {
                contentModelSettings.style.display = 'block';
            } else {
                contentModelSettings.style.display = 'none';
            }
        });
        
        // Sayfa yüklendiğinde de kontrol et
        if (trainingModelTypeSelect.value === 'content') {
            contentModelSettings.style.display = 'block';
        }
    }
    
    // Eğitim istatistiklerini yükle
    refreshTrainingStats();
});

// Eğitim istatistiklerini yenile
async function refreshTrainingStats() {
    const container = document.getElementById('trainingStatsContainer');
    const modelType = document.getElementById('trainingModelType')?.value || 'content';
    
    if (!container) return;
    
    try {
        container.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Yükleniyor...</span>
                </div>
            </div>
        `;
        
        const response = await fetch(`/api/model/training-stats/${modelType}`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            
            // Model türüne göre farklı display
            if (modelType === 'age') {
                container.innerHTML = `
                    <div class="row g-3">
                        <div class="col-md-4">
                            <div class="card border-primary">
                                <div class="card-body text-center">
                                    <h5 class="card-title text-primary">Manuel Feedbacks</h5>
                                    <h3 class="mb-0">${stats.manual_samples || 0}</h3>
                                    <small class="text-muted">Kullanıcı düzelttikleri gerçek geri bildirimler</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card border-info">
                                <div class="card-body text-center">
                                    <h5 class="card-title text-info">Otomatik Feedbacks</h5>
                                    <h3 class="mb-0">${stats.pseudo_samples || 0}</h3>
                                    <small class="text-muted">Sistem etiketleri</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card border-success">
                                <div class="card-body text-center">
                                    <h5 class="card-title text-success">Eğitim Örneği</h5>
                                    <h3 class="mb-0">${stats.total_samples || stats.total_feedbacks}</h3>
                                    <small class="text-muted">Çelişki çözümlemeli</small>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    ${stats.age_distribution ? `
                    <div class="mt-3">
                        <h6>Yaş Dağılımı:</h6>
                        <div class="table-responsive">
                            <table class="table table-sm table-bordered">
                                <thead class="table-light">
                                    <tr>
                                        <th>Yaş Grubu</th>
                                        <th>Örnek Sayısı</th>
                                        <th>Oran (%)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${Object.entries(stats.age_distribution).map(([ageGroup, count]) => {
                                        const percentage = ((count / stats.total_feedbacks) * 100).toFixed(1);
                                        return `
                                            <tr>
                                                <td>${ageGroup}</td>
                                                <td><span class="badge bg-primary">${count}</span></td>
                                                <td>${percentage}%</td>
                                            </tr>
                                        `;
                                    }).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    ` : ''}
                    
                    <div class="alert alert-info mt-3">
                        <h6><i class="fas fa-info-circle me-2"></i>Eğitim Verisi Açıklaması</h6>
                        <ul class="mb-0">
                            <li><strong>Manuel Feedbacks:</strong> Kullanıcıların yaş tahminlerini düzelttikleri gerçek geri bildirimler</li>
                            <li><strong>Otomatik Feedbacks:</strong> Buffalo modeli tarafından yüksek güvenle etiketlenen veriler</li>
                            <li><strong>Eğitim Örneği:</strong> Aynı kişi için hem manuel hem otomatik feedback varsa, manuel feedback öncelikli</li>
                        </ul>
                    </div>
                    
                    ${stats.message ? `<div class="alert alert-warning mt-3">${stats.message}</div>` : ''}
                `;
            } else {
                // Content modeli için mevcut display
                container.innerHTML = `
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="card border-primary">
                                <div class="card-body text-center">
                                    <h5 class="card-title text-primary">Toplam Feedback</h5>
                                    <h3 class="mb-0">${stats.total_feedbacks}</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card border-success">
                                <div class="card-body text-center">
                                    <h5 class="card-title text-success">Eğitim Örneği</h5>
                                    <h3 class="mb-0">${stats.total_samples}</h3>
                                </div>
                            </div>
                        </div>
                        ${stats.conflicts_detected ? `
                        <div class="col-md-6">
                            <div class="card border-warning">
                                <div class="card-body text-center">
                                    <h5 class="card-title text-warning">Çelişkiler</h5>
                                    <h3 class="mb-0">${stats.conflicts_detected}</h3>
                                </div>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    
                    ${stats.category_stats ? `
                    <div class="mt-3">
                        <h6>Kategori Dağılımı:</h6>
                        <div class="table-responsive">
                            <table class="table table-sm table-bordered">
                                <thead class="table-light">
                                    <tr>
                                        <th>Kategori</th>
                                        <th>Pozitif</th>
                                        <th>Negatif</th>
                                        <th>Toplam</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${Object.entries(stats.category_stats).map(([category, data]) => `
                                        <tr>
                                            <td>${getCategoryDisplayName(category)}</td>
                                            <td><span class="badge bg-danger">${data.positive || 0}</span></td>
                                            <td><span class="badge bg-success">${data.negative || 0}</span></td>
                                            <td><span class="badge bg-primary">${data.total || 0}</span></td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    ` : ''}
                    
                    ${stats.message ? `<div class="alert alert-info mt-3">${stats.message}</div>` : ''}
                `;
            }
        } else {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${data.error || 'İstatistikler yüklenemedi'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Training stats error:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Bağlantı hatası: ${error.message}
            </div>
        `;
    }
}



// Web eğitimi başlat
let currentTrainingSession = null;
let trainingStartTime = null;

async function startWebTraining() {
    console.log('[WebSocket] startWebTraining called');
    
    try {
        const response = await fetch('/api/model/train-web', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_type: 'content',
                epochs: 20,
                learning_rate: 0.001,
                batch_size: 1
            })
        });
        
        const data = await response.json();
        console.log('[WebSocket] Backend response:', data);
        
        if (data.success) {
            // Global session tracking için session ID'yi kaydet
            window.currentTrainingSessionId = data.session_id;
            window.isModalTraining = false; // Bu web training, modal training değil
            
            console.log('[WebSocket] Setting up WebSocket connection for web training with session_id:', data.session_id);
            
            // WebSocket training room'a katıl
            if (window.wsClient && window.wsClient.connected) {
                window.wsClient.joinTraining(data.session_id);
            } else {
                console.log('WebSocket henüz bağlı değil, room join atlandı');
            }
            
            showToast('Bilgi', `Eğitim başlatıldı. Tahmini süre: ${data.estimated_duration}`, 'info');
        } else {
            showError(`Eğitim başlatılamadı: ${data.error}`);
        }
    } catch (error) {
        console.error('Eğitim başlatma hatası:', error);
        showError('Eğitim başlatılırken bir hata oluştu.');
    }
}

// Eğitimi durdur
function stopWebTraining() {
    const startBtn = document.getElementById('startWebTrainingBtn');
    const stopBtn = document.getElementById('stopWebTrainingBtn');
    const statusDiv = document.getElementById('webTrainingStatus');
    const progressDiv = document.getElementById('webTrainingProgress');
    
    // UI sıfırla
    if (startBtn && stopBtn) {
        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
    }
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
    
    if (statusDiv) {
        statusDiv.className = 'alert alert-warning';
        document.getElementById('webTrainingMessage').textContent = 'Eğitim kullanıcı tarafından durduruldu.';
    }
    
    currentTrainingSession = null;
    trainingStartTime = null;
    
    console.log('Web training connection closed by user');
}

// WebSocket event listeners
function setupTrainingWebSocketListeners(sessionId) {
    console.log('[DEBUG] setupTrainingWebSocketListeners called with sessionId:', sessionId);
    console.log('[DEBUG] Socket connected:', socket.connected);
    console.log('[DEBUG] Socket object:', socket);
    console.log('[DEBUG] isModalTraining flag:', window.isModalTraining);
    
    // Add a global listener to catch ALL events for debugging
    socket.onAny((eventName, data) => {
        console.log('[SOCKET DEBUG] ANY EVENT RECEIVED:', eventName, data);
        if (eventName === 'training_progress') {
            console.log('[SOCKET DEBUG] TRAINING_PROGRESS EVENT DETECTED!', data);
        }
    });
    
    // Add a global catch-all listener for ANY training_progress event
    socket.off('training_progress');
    socket.on('training_progress', (data) => {
        console.log('[GLOBAL CATCH-ALL] training_progress event received:', data);
        console.log('[GLOBAL CATCH-ALL] Expected sessionId:', sessionId);
        console.log('[GLOBAL CATCH-ALL] Received sessionId:', data.session_id);
        console.log('[GLOBAL CATCH-ALL] Session ID match:', data.session_id === sessionId);
        console.log('[GLOBAL CATCH-ALL] isModalTraining:', window.isModalTraining);
        
        // If this is our session, handle the progress update
        if (data.session_id === sessionId) {
            console.log('[GLOBAL CATCH-ALL] Processing training progress for our session');
            
            // Check if we should update modal or web training progress
            const isUsingModal = window.isModalTraining === true;
            console.log('[GLOBAL CATCH-ALL] Using modal training:', isUsingModal);
            
            if (isUsingModal) {
                // Update modal progress
                const modalProgressDiv = document.getElementById('modal-training-progress');
                console.log('[GLOBAL CATCH-ALL] Modal progress div found:', !!modalProgressDiv);
                console.log('[GLOBAL CATCH-ALL] Modal progress div display style:', modalProgressDiv ? modalProgressDiv.style.display : 'not found');
                console.log('[GLOBAL CATCH-ALL] Modal progress div computed display:', modalProgressDiv ? window.getComputedStyle(modalProgressDiv).display : 'not found');
                
                if (modalProgressDiv) {
                    const progressPercent = (data.current_epoch / data.total_epochs) * 100;
                    const progressBar = document.getElementById('modal-progress-bar');
                    const currentEpoch = document.getElementById('modal-current-epoch');
                    const currentLoss = document.getElementById('modal-current-loss');
                    const currentMAE = document.getElementById('modal-current-mae');
                    
                    console.log('[GLOBAL CATCH-ALL] Modal elements found:', {
                        progressDiv: !!modalProgressDiv,
                        progressBar: !!progressBar,
                        currentEpoch: !!currentEpoch,
                        currentLoss: !!currentLoss,
                        currentMAE: !!currentMAE
                    });
                    
                    if (progressBar) {
                        progressBar.style.width = progressPercent + '%';
                        progressBar.setAttribute('aria-valuenow', Math.round(progressPercent));
                        console.log('[GLOBAL CATCH-ALL] Progress bar updated to:', progressPercent + '%');
                    }
                    if (currentEpoch) {
                        currentEpoch.textContent = `${data.current_epoch}/${data.total_epochs}`;
                        console.log('[GLOBAL CATCH-ALL] Epoch updated to:', `${data.current_epoch}/${data.total_epochs}`);
                    }
                    if (currentLoss) {
                        currentLoss.textContent = data.current_loss?.toFixed(4) || '-';
                        console.log('[GLOBAL CATCH-ALL] Loss updated to:', data.current_loss?.toFixed(4));
                    }
                    if (currentMAE) {
                        currentMAE.textContent = data.current_mae?.toFixed(4) || '-';
                        console.log('[GLOBAL CATCH-ALL] MAE updated to:', data.current_mae?.toFixed(4));
                    }
                    
                    // Durum mesajını güncelle
                    showModalTrainingStatus(`Eğitim devam ediyor... Epoch ${data.current_epoch}/${data.total_epochs} (${Math.round(progressPercent)}%)`, 'info');
                    console.log('[GLOBAL CATCH-ALL] Modal status updated');
                } else {
                    console.log('[GLOBAL CATCH-ALL] Modal training requested but modal not found or not visible');
                }
            } else {
                // Update web training progress (the original way)
                console.log('[GLOBAL CATCH-ALL] Using web training progress');
                const webProgressDiv = document.getElementById('webTrainingProgress');
                if (webProgressDiv) {
                    webProgressDiv.style.display = 'block';
                    const progressPercent = (data.current_epoch / data.total_epochs) * 100;
                    const webProgressBar = document.getElementById('webProgressBar');
                    const webProgressText = document.getElementById('webProgressText');
                    
                    if (webProgressBar) {
                        webProgressBar.style.width = progressPercent + '%';
                        webProgressBar.setAttribute('aria-valuenow', Math.round(progressPercent));
                    }
                    if (webProgressText) {
                        webProgressText.textContent = Math.round(progressPercent) + '%';
                    }
                    console.log('[GLOBAL CATCH-ALL] Web progress updated to:', progressPercent + '%');
                }
            }
            
            // Convert backend data format to frontend expected format for compatibility
            const progressData = {
                progress: (data.current_epoch / data.total_epochs) * 100,
                epoch: data.current_epoch,
                total_epochs: data.total_epochs,
                metrics: {
                    val_loss: data.current_loss,
                    current_loss: data.current_loss,
                    val_mae: data.current_mae,
                    current_mae: data.current_mae,
                    val_r2: data.current_r2,
                    current_r2: data.current_r2
                }
            };
            
            // Also call the standard update functions for compatibility
            try {
                updateWebTrainingProgress(progressData);
                console.log('[GLOBAL CATCH-ALL] updateWebTrainingProgress called successfully');
            } catch (error) {
                console.error('[GLOBAL CATCH-ALL] Error in updateWebTrainingProgress:', error);
            }
            
            try {
                updateModalTrainingProgress(progressData);
                console.log('[GLOBAL CATCH-ALL] updateModalTrainingProgress called successfully');
            } catch (error) {
                console.error('[GLOBAL CATCH-ALL] Error in updateModalTrainingProgress:', error);
            }
        }
    });
    
    // Global listener for all training_progress events (for debugging)
    socket.off('training_progress_global_debug');
    socket.on('training_progress_global', (data) => {
        console.log('[GLOBAL DEBUG] ANY training_progress event received:', data);
        
        // Modal progress güncellemesi (eğer modal açıksa)
        const modalProgressDiv = document.getElementById('modal-training-progress');
        if (modalProgressDiv && modalProgressDiv.style.display === 'block') {
            // Modal progress bar güncelle
            const progressPercent = (data.current_epoch / data.total_epochs) * 100;
            const progressBar = document.getElementById('modal-progress-bar');
            const currentEpoch = document.getElementById('modal-current-epoch');
            const currentLoss = document.getElementById('modal-current-loss');
            const currentMAE = document.getElementById('modal-current-mae');
            
            if (progressBar) {
                progressBar.style.width = progressPercent + '%';
                progressBar.setAttribute('aria-valuenow', Math.round(progressPercent));
            }
            if (currentEpoch) {
                currentEpoch.textContent = `${data.current_epoch}/${data.total_epochs}`;
            }
            if (currentLoss) {
                currentLoss.textContent = data.current_loss?.toFixed(4) || '-';
            }
            if (currentMAE) {
                currentMAE.textContent = data.current_mae?.toFixed(4) || '-';
            }
            
            // Durum mesajını güncelle
            showModalTrainingStatus(`Eğitim devam ediyor... Epoch ${data.current_epoch}/${data.total_epochs} (${Math.round(progressPercent)}%)`, 'info');
            
            console.log('[GLOBAL DEBUG] Modal progress updated:', {
                epoch: `${data.current_epoch}/${data.total_epochs}`,
                progress: progressPercent + '%',
                loss: data.current_loss,
                mae: data.current_mae
            });
        }
    });
    
    // Training started
    socket.off('training_started');
    socket.on('training_started', (data) => {
        console.log('[DEBUG] training_started event received:', data);
        if (data.session_id === sessionId) {
            console.log('Training started:', data);
            document.getElementById('webTrainingMessage').textContent = 
                `Eğitim başladı (${data.total_samples} örnek)`;
        }
        
        // Modal için de güncelle
        const modalProgressDiv = document.getElementById('modal-training-progress');
        if (modalProgressDiv && modalProgressDiv.style.display === 'block') {
            showModalTrainingStatus(`Eğitim başladı (${data.total_samples} örnek)`, 'info');
            console.log('[GLOBAL DEBUG] Modal training started updated');
        }
    });
    
    // Training progress
    socket.on('training_progress', (data) => {
        console.log('[DEBUG] training_progress event received:', data, 'expected sessionId:', sessionId);
        console.log('[DEBUG] session_id match:', data.session_id === sessionId);
        if (data.session_id === sessionId) {
            console.log('[DEBUG] Calling updateWebTrainingProgress and updateModalTrainingProgress');
            
            // Convert backend data format to frontend expected format
            const progressData = {
                progress: (data.current_epoch / data.total_epochs) * 100,
                epoch: data.current_epoch,
                total_epochs: data.total_epochs,
                metrics: {
                    val_loss: data.current_loss,
                    current_loss: data.current_loss,
                    val_mae: data.current_mae,
                    current_mae: data.current_mae,
                    val_r2: data.current_r2,
                    current_r2: data.current_r2
                }
            };
            
            updateWebTrainingProgress(progressData);
            
            // Modal progress'i de güncelle
            updateModalTrainingProgress(progressData);
            
            // Also directly update modal elements
            const modalProgressDiv = document.getElementById('modal-training-progress');
            if (modalProgressDiv && modalProgressDiv.style.display === 'block') {
                const progressPercent = (data.current_epoch / data.total_epochs) * 100;
                const progressBar = document.getElementById('modal-progress-bar');
                const currentEpoch = document.getElementById('modal-current-epoch');
                const currentLoss = document.getElementById('modal-current-loss');
                const currentMAE = document.getElementById('modal-current-mae');
                
                if (progressBar) {
                    progressBar.style.width = progressPercent + '%';
                    progressBar.setAttribute('aria-valuenow', Math.round(progressPercent));
                }
                if (currentEpoch) {
                    currentEpoch.textContent = `${data.current_epoch}/${data.total_epochs}`;
                }
                if (currentLoss) {
                    currentLoss.textContent = data.current_loss?.toFixed(4) || '-';
                }
                if (currentMAE) {
                    currentMAE.textContent = data.current_mae?.toFixed(4) || '-';
                }
                
                // Durum mesajını güncelle
                showModalTrainingStatus(`Eğitim devam ediyor... Epoch ${data.current_epoch}/${data.total_epochs} (${Math.round(progressPercent)}%)`, 'info');
                
                console.log('[DEBUG] Direct modal update completed:', {
                    epoch: `${data.current_epoch}/${data.total_epochs}`,
                    progress: progressPercent + '%',
                    loss: data.current_loss,
                    mae: data.current_mae
                });
            }
        } else {
            console.log('[DEBUG] Session ID mismatch - ignoring event');
        }
    });
    
    // Training completed
    socket.off('training_completed');
    socket.on('training_completed', (data) => {
        console.log('[DEBUG] training_completed event received:', data);
        console.log('[DEBUG] isModalTraining flag:', window.isModalTraining);
        
        if (data.session_id === sessionId) {
            // Reset modal training flag
            if (window.isModalTraining) {
                console.log('[DEBUG] Resetting isModalTraining flag');
                window.isModalTraining = false;
            }
            
            handleWebTrainingCompleted(data);
            
            // Modal completion'ı da handle et
            handleModalTrainingCompleted(data);
        }
        
        // Modal için de global güncelle - ensure modal gets completion message
        const modalProgressDiv = document.getElementById('modal-training-progress');
        if (modalProgressDiv && modalProgressDiv.style.display === 'block') {
            const progressBar = document.getElementById('modal-progress-bar');
            if (progressBar) {
                progressBar.style.width = '100%';
                progressBar.setAttribute('aria-valuenow', 100);
            }
            showModalTrainingStatus(`Eğitim tamamlandı! Model: ${data.model_version}`, 'success');
            console.log('[DEBUG] Modal training completion updated');
        }
    });
    
    // Training error handler
    socket.off('training_error');
    socket.on('training_error', (data) => {
        console.log('[DEBUG] training_error event received:', data);
        if (data.session_id === sessionId) {
            // Reset modal training flag
            if (window.isModalTraining) {
                console.log('[DEBUG] Resetting isModalTraining flag due to error');
                window.isModalTraining = false;
            }
            
            handleWebTrainingError(data);
            
            // Modal için error message
            const modalProgressDiv = document.getElementById('modal-training-progress');
            if (modalProgressDiv && modalProgressDiv.style.display === 'block') {
                showModalTrainingStatus(`Eğitim hatası: ${data.error}`, 'danger');
                console.log('[DEBUG] Modal training error updated');
            }
        }
    });
}

// Eğitim progress güncelle
function updateWebTrainingProgress(data) {
    const progressBar = document.getElementById('webProgressBar');
    const progressText = document.getElementById('webProgressText');
    const currentEpoch = document.getElementById('webCurrentEpoch');
    const currentLoss = document.getElementById('webCurrentLoss');
    const currentMAE = document.getElementById('webCurrentMAE');
    const currentR2 = document.getElementById('webCurrentR2');
    const trainingDuration = document.getElementById('webTrainingDuration');
    const trainingETA = document.getElementById('webTrainingETA');
    
    console.log('Training progress update:', data);
    
    // Progress bar güncelleme
    const progress = Math.round(data.progress || 0);
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
    if (progressText) {
        progressText.textContent = `${progress}%`;
    }
    
    // Epoch bilgisi
    if (currentEpoch && data.epoch && data.total_epochs) {
        currentEpoch.textContent = `${data.epoch}/${data.total_epochs}`;
    }
    
    // Metrics güncelleme (model tipine göre)
    if (data.metrics) {
        // Yaş modeli için
        if (data.metrics.val_loss && currentLoss) {
            currentLoss.textContent = data.metrics.val_loss.toFixed(4);
        }
        if (data.metrics.val_mae && currentMAE) {
            currentMAE.textContent = data.metrics.val_mae.toFixed(3);
        }
        
        // Content modeli için de destekle
        if (data.metrics.current_loss && currentLoss) {
            currentLoss.textContent = data.metrics.current_loss.toFixed(4);
        }
        if (data.metrics.current_mae && currentMAE) {
            currentMAE.textContent = data.metrics.current_mae.toFixed(3);
        }
        if (data.metrics.current_r2 && currentR2) {
            currentR2.textContent = data.metrics.current_r2.toFixed(3);
        }
    }
    
    // Süre hesaplamaları
    if (trainingStartTime) {
        const elapsed = (Date.now() - trainingStartTime) / 1000;
        if (trainingDuration) {
            trainingDuration.textContent = formatDuration(elapsed);
        }
        
        if (data.epoch && data.total_epochs && data.epoch > 0) {
            const avgTimePerEpoch = elapsed / data.epoch;
            const remainingEpochs = data.total_epochs - data.epoch;
            const eta = remainingEpochs * avgTimePerEpoch;
            if (trainingETA) {
                trainingETA.textContent = formatDuration(eta);
            }
        }
    }
    
    // Durum mesajını güncelle
    const statusMessage = document.getElementById('webTrainingMessage');
    if (statusMessage && data.epoch && data.total_epochs) {
        statusMessage.textContent = `Eğitim devam ediyor... Epoch ${data.epoch}/${data.total_epochs} (${progress}%)`;
    }
}

// Eğitim tamamlandı
function handleWebTrainingCompleted(data) {
    const startBtn = document.getElementById('startWebTrainingBtn');
    const stopBtn = document.getElementById('stopWebTrainingBtn');
    const statusDiv = document.getElementById('webTrainingStatus');
    const progressDiv = document.getElementById('webTrainingProgress');
    const resultsDiv = document.getElementById('webTrainingResults');
    const metricsDiv = document.getElementById('webTrainingMetrics');
    
    // UI sıfırla
    startBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    progressDiv.style.display = 'none';
    
    // Success mesajı
    statusDiv.className = 'alert alert-success';
    document.getElementById('webTrainingMessage').textContent = 
        `Eğitim tamamlandı! Yeni model versiyonu: ${data.model_version}`;
    
    // Results göster
    resultsDiv.style.display = 'block';
    
    const metrics = data.metrics;
    
    // Model tipine göre farklı metrik display
    if (data.model_type === 'age') {
        // Yaş modeli metrikleri
        metricsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card border-primary">
                        <div class="card-body text-center">
                            <h6 class="card-title">MAE (Ortalama Hata)</h6>
                            <h5 class="text-primary">${metrics.mae ? metrics.mae.toFixed(2) : '-'} yaş</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-success">
                        <div class="card-body text-center">
                            <h6 class="card-title">±3 Yaş Doğruluğu</h6>
                            <h5 class="text-success">${metrics.within_3_years ? (metrics.within_3_years * 100).toFixed(1) : '-'}%</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-info">
                        <div class="card-body text-center">
                            <h6 class="card-title">±5 Yaş Doğruluğu</h6>
                            <h5 class="text-info">${metrics.within_5_years ? (metrics.within_5_years * 100).toFixed(1) : '-'}%</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-warning">
                        <div class="card-body text-center">
                            <h6 class="card-title">RMSE</h6>
                            <h5 class="text-warning">${metrics.rmse ? metrics.rmse.toFixed(2) : '-'} yaş</h5>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-6">
                    <div class="card border-secondary">
                        <div class="card-body text-center">
                            <h6 class="card-title">Eğitim Örnekleri</h6>
                            <h5 class="text-secondary">${metrics.training_samples || data.training_samples || '-'}</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card border-secondary">
                        <div class="card-body text-center">
                            <h6 class="card-title">Doğrulama Örnekleri</h6>
                            <h5 class="text-secondary">${metrics.validation_samples || data.validation_samples || '-'}</h5>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        // Content modeli metrikleri (mevcut kod)
        metricsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card border-success">
                        <div class="card-body text-center">
                            <h6 class="card-title">Accuracy</h6>
                            <h5 class="text-success">${metrics.accuracy ? (metrics.accuracy * 100).toFixed(1) : '-'}%</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-info">
                        <div class="card-body text-center">
                            <h6 class="card-title">Loss</h6>
                            <h5 class="text-info">${metrics.loss ? metrics.loss.toFixed(4) : '-'}</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-warning">
                        <div class="card-body text-center">
                            <h6 class="card-title">F1 Score</h6>
                            <h5 class="text-warning">${metrics.f1_score ? metrics.f1_score.toFixed(3) : '-'}</h5>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-primary">
                        <div class="card-body text-center">
                            <h6 class="card-title">Çelişki Çözüldü</h6>
                            <h5 class="text-primary">${data.conflicts_resolved || '-'}</h5>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    currentTrainingSession = null;
    trainingStartTime = null;
    
    // Model versiyonlarını yenile
    setTimeout(() => {
        refreshTrainingStats();
    }, 1000);
}

// Eğitim hatası
function handleWebTrainingError(data) {
    const startBtn = document.getElementById('startWebTrainingBtn');
    const stopBtn = document.getElementById('stopWebTrainingBtn');
    const statusDiv = document.getElementById('webTrainingStatus');
    const progressDiv = document.getElementById('webTrainingProgress');
    
    // UI sıfırla
    startBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    progressDiv.style.display = 'none';
    
    // Error mesajı
    statusDiv.className = 'alert alert-danger';
    document.getElementById('webTrainingMessage').textContent = `Eğitim hatası: ${data.error}`;
    
    currentTrainingSession = null;
    trainingStartTime = null;
}

// Kategori adlarını düzenle
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

// Risk seviyesi belirleme fonksiyonu (4 seviyeli sistem)
function getRiskLevel(score, category) {
    // Safe kategorisi için ters logic (yüksek skor = güvenli = yeşil)
    if (category === 'safe') {
        if (score >= 0.8) return { level: 'very-low', color: 'success', text: 'Çok Güvenli' };
        if (score >= 0.6) return { level: 'low', color: 'info', text: 'Güvenli' };
        if (score >= 0.4) return { level: 'medium', color: 'warning', text: 'Belirsiz' };
        if (score >= 0.2) return { level: 'high', color: 'danger', text: 'Riskli' };
        return { level: 'very-high', color: 'dark', text: 'Çok Riskli' };
    }
    
    // Diğer kategoriler için normal logic (yüksek skor = riskli)
    if (score < 0.2) return { level: 'very-low', color: 'success', text: 'Çok Düşük' };
    if (score < 0.4) return { level: 'low', color: 'info', text: 'Düşük' };
    if (score < 0.6) return { level: 'medium', color: 'warning', text: 'Orta' };
    if (score < 0.8) return { level: 'high', color: 'danger', text: 'Yüksek' };
    return { level: 'very-high', color: 'dark', text: 'Çok Yüksek' };
}

// Modal training progress güncelle
function updateModalTrainingProgress(data) {
    console.log('[DEBUG] updateModalTrainingProgress called with data:', data);
    const progressBar = document.getElementById('modal-progress-bar');
    const currentEpoch = document.getElementById('modal-current-epoch');
    const currentLoss = document.getElementById('modal-current-loss');
    const currentMAE = document.getElementById('modal-current-mae');
    const trainingDuration = document.getElementById('modal-training-duration');
    
    console.log('Modal training progress update:', data);
    
    // Progress bar güncelleme
    const progress = Math.round(data.progress || 0);
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
    
    // Epoch bilgisi
    if (currentEpoch && data.epoch && data.total_epochs) {
        currentEpoch.textContent = `${data.epoch}/${data.total_epochs}`;
    }
    
    // Metrics güncelleme
    if (data.metrics) {
        if (data.metrics.val_loss && currentLoss) {
            currentLoss.textContent = data.metrics.val_loss.toFixed(4);
        }
        if (data.metrics.val_mae && currentMAE) {
            currentMAE.textContent = data.metrics.val_mae.toFixed(3);
        }
        
        // Fallback untuk current metrics
        if (data.metrics.current_loss && currentLoss) {
            currentLoss.textContent = data.metrics.current_loss.toFixed(4);
        }
        if (data.metrics.current_mae && currentMAE) {
            currentMAE.textContent = data.metrics.current_mae.toFixed(3);
        }
    }
    
    // Süre hesaplaması
    if (trainingStartTime && trainingDuration) {
        const elapsed = (Date.now() - trainingStartTime) / 1000;
        trainingDuration.textContent = formatDuration(elapsed);
    }
    
    // Durum mesajını güncelle
    showModalTrainingStatus(`Eğitim devam ediyor... Epoch ${data.epoch || 0}/${data.total_epochs || 0} (${progress}%)`, 'info');
}

// Modal training tamamlandı
function handleModalTrainingCompleted(data) {
    const progressDiv = document.getElementById('modal-training-progress');
    
    // Progress bar'ı 100% yap
    const progressBar = document.getElementById('modal-progress-bar');
    if (progressBar) {
        progressBar.style.width = '100%';
        progressBar.setAttribute('aria-valuenow', 100);
    }
    
    // Tamamlanma mesajı
    const metrics = data.metrics || {};
    let successMessage = 'Eğitim başarıyla tamamlandı!';
    
    if (metrics.mae) {
        successMessage += ` (MAE: ${metrics.mae.toFixed(3)})`;
    } else if (metrics.accuracy) {
        successMessage += ` (Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%)`;
    }
    
    showModalTrainingStatus(successMessage, 'success');
    
    // Eğitim butonlarını aktif et
    const trainButtons = document.querySelectorAll('[onclick*="trainModelFromModal"]');
    trainButtons.forEach(btn => {
        btn.disabled = false;
        const modelType = btn.onclick.toString().includes("'age'") ? 'age' : 'content';
        btn.innerHTML = `<i class="fas fa-play me-2"></i>Yeni Eğitim Başlat`;
    });
    
    // Model versiyonlarını ve istatistikleri yenile
    setTimeout(() => {
        loadModalModelVersions();
        loadModalModelStats();
        
        // Progress'i gizle
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
    }, 3000);
    
    // Toast notification
    showToast('Başarılı', 'Model eğitimi başarıyla tamamlandı!', 'success');
}

// Modal'dan model sıfırla
function resetModelFromModal(modelType) {
    const isAgeModel = modelType === 'age';
    const confirmMessage = isAgeModel 
        ? 'Yaş tahmin modeli ensemble düzeltmelerini temizlemek istediğinizden emin misiniz?\n\nBu işlem base model\'e döner ve düzeltmeler silinir.'
        : 'İçerik analiz modelini sıfırlamak istediğinizden emin misiniz?\n\nDikkat: Model sıfırlama işlemi sistem yeniden başlatılmasını gerektirir.';
    
    if (confirm(confirmMessage)) {
        console.log(`Modal - Resetting ${modelType} model`);
        
        showModalTrainingStatus('Model sıfırlanıyor...', 'info');
        
        // Yükleyici göster
        const settingsSaveLoader = document.getElementById('settingsSaveLoader');
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'flex';
        }
        
        // Yaş modeli için ensemble reset, diğerleri için normal reset
        const endpoint = isAgeModel ? `/api/ensemble/reset/${modelType}` : `/api/model/reset/${modelType}`;
        
        // Model sıfırlama API çağrısı
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Model reset response:', data);
            
            if (data.success) {
                // Başarılı mesaj
                let message = `${modelType} modeli başarıyla sıfırlandı!\n`;
                message += `Temizlenen düzeltmeler: ${data.corrections_cleared || 0}`;
                
                // Otomatik temizlik sonuçlarını göster
                if (data.auto_cleanup && data.auto_cleanup.enabled) {
                    message += `\n\n🧹 Otomatik Temizlik:\n`;
                    message += `Toplam temizlenen: ${data.auto_cleanup.total_cleaned} öğe\n`;
                    
                    if (data.auto_cleanup.summary) {
                        message += `\nDetaylar:\n${data.auto_cleanup.summary.join('\n')}`;
                    }
                    
                    if (data.auto_cleanup.error) {
                        message += `\n⚠️ Temizlik uyarısı: ${data.auto_cleanup.message}`;
                    }
                }
                
                if (modelType === 'age' && data.restart_required) {
                    // Yaş modeli sıfırlandığında sistem yeniden başlatılmalı
                    showModalTrainingStatus('Model sıfırlandı. Sistem yeniden başlatılıyor...', 'success');
                    showToast('Bilgi', 'Model başarıyla sıfırlandı. Sistem yeniden başlatılıyor, lütfen bekleyin...', 'info');
                    
                    // Yeniden başlatma sonrası sayfa yenilenmesi için işaret koy
                    localStorage.setItem('modelChangedReloadRequired', 'true');
                    
                    // Modal'ı kapat
                    const modalElement = document.getElementById('modelManagementModal');
                    if (modalElement) {
                        const modalInstance = bootstrap.Modal.getInstance(modalElement);
                        if (modalInstance) {
                            modalInstance.hide();
                        }
                    }
                } else {
                    // Ensemble reset için
                    showModalTrainingStatus(message.replace(/\n/g, '<br>'), 'success');
                    showToast('Başarılı', `${modelType} modeli sıfırlandı ve otomatik temizlik tamamlandı!`, 'success');
                    
                    // Model metriklerini yenile
                    loadModelMetrics();
                }
                
                // Yükleyiciyi gizle
                if (settingsSaveLoader) {
                    settingsSaveLoader.style.display = 'none';
                }
                
            } else {
                throw new Error(data.error || 'Model sıfırlama başarısız');
            }
        })
        .catch(error => {
            console.error('❌ Model reset hatası:', error);
            
            showModalTrainingStatus(`Model sıfırlama hatası: ${error.message}`, 'danger');
            showToast('Hata', `Model sıfırlama hatası: ${error.message}`, 'danger');
            
            // Yükleyiciyi gizle
            if (settingsSaveLoader) {
                settingsSaveLoader.style.display = 'none';
            }
        });
    }
}



// Ensemble corrections yenileme fonksiyonu
function refreshEnsembleCorrections() {
    console.log('🔄 Ensemble corrections yenileniyor...');
    
    const button = document.querySelector('.btn-train-age');
    const statusElement = document.getElementById('modal-training-status');
    const progressDiv = document.getElementById('modal-training-progress');
    
    // UI durumunu ayarla
    if (statusElement) {
        statusElement.textContent = 'Ensemble corrections yenileniyor...';
        statusElement.className = 'alert alert-info';
    }
    
    if (progressDiv) {
        progressDiv.style.display = 'block';
        progressDiv.classList.remove('d-none');
    }
    
    // API çağrısı
    fetch('/api/ensemble/refresh', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('✅ Ensemble refresh response:', data);
        
        if (data.success) {
            // Başarılı mesaj
            let message = `Ensemble corrections başarıyla yenilendi!\n`;
            message += `Yaş düzeltmeleri: ${data.age_corrections}\n`;
            message += `İçerik düzeltmeleri: ${data.clip_corrections}`;
            
            // Model versiyon bilgilerini göster
            if (data.models_created) {
                message += `\n\n📦 Oluşturulan Model Versiyonları:\n`;
                if (data.models_created.age_model_created && data.age_version) {
                    message += `✅ Yaş Modeli: ${data.age_version}\n`;
                }
                if (data.models_created.clip_model_created && data.clip_version) {
                    message += `✅ İçerik Modeli: ${data.clip_version}\n`;
                }
                if (!data.models_created.age_model_created && !data.models_created.clip_model_created) {
                    message += `ℹ️ Yeni düzeltme bulunmadığı için versiyon oluşturulmadı\n`;
                }
            }
            
            // Otomatik temizlik sonuçlarını göster
            if (data.auto_cleanup && data.auto_cleanup.enabled) {
                message += `\n\n🧹 Otomatik Temizlik:\n`;
                message += `Toplam temizlenen: ${data.auto_cleanup.total_cleaned} öğe\n`;
                
                if (data.auto_cleanup.summary) {
                    message += `\nDetaylar:\n${data.auto_cleanup.summary.join('\n')}`;
                }
                
                if (data.auto_cleanup.error) {
                    message += `\n⚠️ Temizlik uyarısı: ${data.auto_cleanup.message}`;
                }
            }
            
            if (statusElement) {
                statusElement.innerHTML = message.replace(/\n/g, '<br>');
                statusElement.className = 'alert alert-success';
            }
            
            // Toast bildirimi - model versiyonu bilgisi ile
            let toastMessage = 'Ensemble corrections yenilendi';
            if (data.models_created && (data.models_created.age_model_created || data.models_created.clip_model_created)) {
                toastMessage += ' ve yeni model versiyonları oluşturuldu';
            }
            toastMessage += '!';
            
            showToast('Başarılı', toastMessage, 'success');
            
            // Buton durumunu sıfırla
            if (button) {
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-sync me-2"></i>Corrections Yenile';
            }
            
            // Model metriklerini yenile
            loadModelMetrics();
            
        } else {
            throw new Error(data.error || 'Ensemble refresh başarısız');
        }
    })
    .catch(error => {
        console.error('❌ Ensemble refresh hatası:', error);
        
        if (statusElement) {
            statusElement.textContent = `Ensemble refresh hatası: ${error.message}`;
            statusElement.className = 'alert alert-danger';
        }
        
        if (button) {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-sync me-2"></i>Corrections Yenile';
        }
        
        showToast('Hata', `Ensemble refresh hatası: ${error.message}`, 'danger');
    });
}



// ... existing code ...
// Modal kapandıktan sonra focus'u kaldır (vanilla JS)
const modal = document.getElementById('runAnalysisSettingsModal');
if (modal) {
    modal.addEventListener('hidden.bs.modal', function () {
        const btn = document.getElementById('startAnalysisBtn');
        if (btn) btn.blur();
    });
}
// ... existing code ...

// ... existing code ...
// Analizi durdur fonksiyonu
function stopAnalysis() {
    console.log('[DEBUG] stopAnalysis çağrıldı');
    
    // Kullanıcıdan onay al
    if (!confirm('Tüm analizler durdurulacak ve kuyruk temizlenecek. Emin misiniz?')) {
        console.log('[DEBUG] stopAnalysis: Kullanıcı işlemi iptal etti');
        return;
    }
    
    console.log('[DEBUG] stopAnalysis: Kullanıcı onayladı, API çağrısı yapılıyor...');
    
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
                fileStatuses.set(fileId, "cancelled");
            }
        }
        
        // Global progress'i gizle
        document.getElementById('globalProgressSection').style.display = 'none';
        
        // Butonu tekrar "Analiz Et" olarak değiştir
        resetAnalyzeButton();
        
        // Toast mesajından sonra sayfayı yenile
        setTimeout(() => {
            console.log('[DEBUG] stopAnalysis: Sayfa yenileniyor...');
            window.location.reload();
        }, 1500); // 1.5 saniye bekle ki toast mesajı görünsün
        
    })
    .catch(error => {
        console.error('[DEBUG] stopAnalysis: Fetch error:', error);
        console.log('[DEBUG] stopAnalysis: Error type:', typeof error);
        console.log('[DEBUG] stopAnalysis: Error message:', error.message);
        showToast('Hata', 'Analizler durdurulurken hata oluştu: ' + error.message, 'danger');
        console.log('[DEBUG] stopAnalysis: Error toast gösterildi');
    });
}

// Analiz Et butonunu sıfırla
function resetAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<i class="fas fa-play me-1"></i> Analiz Et';
        analyzeBtn.className = 'btn btn-success';
        analyzeBtn.disabled = false;
        // Önceki tüm event listener'ları temizle
        analyzeBtn.replaceWith(analyzeBtn.cloneNode(true));
        // Yeni referansı al ve orijinal event listener'ı ekle
        const newAnalyzeBtn = document.getElementById('analyzeBtn');
        newAnalyzeBtn.addEventListener('click', () => {
            if (uploadedFiles.length > 0) {
                // Analiz parametreleri modalını aç (ANLIK AYARLAR İÇİN YENİ MODAL)
                const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
                modal.show();
            }
        });
        console.log('[DEBUG] Analizi Durdur butonu -> Analiz Et olarak değiştirildi');
    }
    
    // Ana sayfadaki "Analiz Başlat" butonunu da sıfırla
    const startAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
    if (startAnalysisMainBtn) {
        startAnalysisMainBtn.innerHTML = '<i class="fas fa-shield-alt me-2"></i>Analiz Başlat';
        startAnalysisMainBtn.className = 'btn btn-primary btn-lg me-3';
        // Önceki tüm event listener'ları temizle
        startAnalysisMainBtn.replaceWith(startAnalysisMainBtn.cloneNode(true));
        // Yeni referansı al ve orijinal event listener'ı ekle
        const newStartAnalysisMainBtn = document.getElementById('startAnalysisMainBtn');
        newStartAnalysisMainBtn.addEventListener('click', function() { 
            document.getElementById('uploadFileBtn').click(); 
        });
        console.log('[DEBUG] Analizi Durdur butonu -> Analiz Başlat olarak değiştirildi');
    }
}

// Modal kapatıldığında odak ana butona taşınsın
document.addEventListener('DOMContentLoaded', function() {
    const modelMetricsModal = document.getElementById('modelMetricsModal');
    if (modelMetricsModal) {
        modelMetricsModal.addEventListener('hidden.bs.modal', function () {
            document.getElementById('openAnalysisParamsModalBtn')?.focus();
        });
    }
});
// ... existing code ...