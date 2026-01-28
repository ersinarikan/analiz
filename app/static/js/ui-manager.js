/* ERSIN Aciklama. */

import { 
    uploadedFiles,
    showToast,
    showError,
    setGlobalAnalysisParamsModalElement
} from './globals.js';

import { handleFileSelection } from './file-manager.js';
import { startAnalysisForAllFiles } from './analysis-manager.js';

// ERSIN =====================================
// ERSIN UI MANAGEMENT
// ERSIN =====================================

/* ERSIN Aciklama. */
export function handleParamsAlert(e) {
    e.preventDefault();
    e.stopPropagation();
    alert('Analiz parametrelerini değiştirmeden önce lütfen yüklenmiş dosyaları kaldırın veya analizi tamamlayın.');
}

/* ERSIN Aciklama. */
export function handleModelAlert(e) {
    e.preventDefault();
    e.stopPropagation();
    alert('Model işlemlerini yapmadan önce lütfen yüklenmiş dosyaları kaldırın veya analizi tamamlayın.');
}

/* ERSIN Aciklama. */
export function updateAnalysisParamsButtonState() {
    updateAnalysisParamsButtonStateWithQueue(null);
}

/* ERSIN Aciklama. */
export function updateAnalysisParamsButtonStateWithQueue(queueData) {
    const analysisParamsBtn = document.getElementById('openAnalysisParamsModalBtn');
    const modelMetricsBtn = document.getElementById('modelMetricsBtn');
    const trainModelBtn = document.getElementById('trainModelBtn');
    const modelManagementBtn = document.getElementById('modelManagementBtn');

    // ERSIN Yüklü dosya kontrolü
    const hasUploadedFiles = uploadedFiles.length > 0;
    
    // ERSIN Kuyruk durumu kontrolü
    let hasFilesInQueue = false;
    if (queueData) {
        // ERSIN Backend response formatına göre düzelt
        const data = queueData?.data || queueData;
        hasFilesInQueue = (data?.queue_size > 0) || (data?.is_processing === true);
    }
    
    // ERSIN Butonlar sadece analiz devam ederken devre dışı olmalı
    const shouldDisableButtons = hasFilesInQueue;  // ERSIN Sadece kuyruk durumuna göre

    // ERSIN Debug logları (sadece durumda değişiklik varsa)
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
        // ERSIN Analiz Parametreleri butonu
        if (analysisParamsBtn) {
            analysisParamsBtn.classList.add('disabled');
            analysisParamsBtn.setAttribute('aria-disabled', 'true');
            analysisParamsBtn.removeAttribute('data-bs-toggle');
            analysisParamsBtn.removeAttribute('data-bs-target');
            analysisParamsBtn.removeEventListener('click', handleParamsAlert);
            analysisParamsBtn.addEventListener('click', handleParamsAlert);
        }

        // ERSIN Model Metrikleri butonu
        if (modelMetricsBtn) {
            modelMetricsBtn.classList.add('disabled');
            modelMetricsBtn.setAttribute('aria-disabled', 'true');
            modelMetricsBtn.removeEventListener('click', handleModelAlert);
            modelMetricsBtn.addEventListener('click', handleModelAlert);
        }

        // ERSIN Model Eğitimi butonu
        if (trainModelBtn) {
            trainModelBtn.classList.add('disabled');
            trainModelBtn.setAttribute('aria-disabled', 'true');
            trainModelBtn.removeEventListener('click', handleModelAlert);
            trainModelBtn.addEventListener('click', handleModelAlert);
        }

        // ERSIN Model Yönetimi butonu
        if (modelManagementBtn) {
            modelManagementBtn.classList.add('disabled');
            modelManagementBtn.setAttribute('aria-disabled', 'true');
            modelManagementBtn.removeAttribute('data-bs-toggle');
            modelManagementBtn.removeAttribute('data-bs-target');
            modelManagementBtn.removeEventListener('click', handleModelAlert);
            modelManagementBtn.addEventListener('click', handleModelAlert);
        }
    } else {
        // ERSIN Analiz Parametreleri butonu
        if (analysisParamsBtn) {
            analysisParamsBtn.classList.remove('disabled');
            analysisParamsBtn.setAttribute('aria-disabled', 'false');
            analysisParamsBtn.setAttribute('data-bs-toggle', 'modal');
            analysisParamsBtn.setAttribute('data-bs-target', '#analysisParamsModal');
            analysisParamsBtn.removeEventListener('click', handleParamsAlert);
        }

        // ERSIN Model Metrikleri butonu
        if (modelMetricsBtn) {
            modelMetricsBtn.classList.remove('disabled');
            modelMetricsBtn.setAttribute('aria-disabled', 'false');
            modelMetricsBtn.removeEventListener('click', handleModelAlert);
            // ERSIN Gerçek model metrics event listener'ını yeniden ekle
            if (window.loadModelMetrics) {
                // ERSIN Modal instance'ını sakla
                let modalInstance = null;
                modelMetricsBtn.addEventListener('click', () => {
                    window.loadModelMetrics();
                    // ERSIN 🎯 MODEL VERSIONS DE YÜKLE
                    if (window.loadModalModelVersions) {
                        window.loadModalModelVersions();
                    }
                    // ERSIN Var olan modal instance'ını kullan veya yeni oluştur
                    const modalElement = document.getElementById('modelMetricsModal');
                    if (!modalInstance && modalElement) {
                        modalInstance = new bootstrap.Modal(modalElement);
                    }
                    if (modalInstance) {
                        modalInstance.show();
                    }
                });
            }
        }

        // ERSIN Model Eğitimi butonu
        if (trainModelBtn) {
            trainModelBtn.classList.remove('disabled');
            trainModelBtn.setAttribute('aria-disabled', 'false');
            trainModelBtn.removeEventListener('click', handleModelAlert);
        }

        // ERSIN Model Yönetimi butonu
        if (modelManagementBtn) {
            modelManagementBtn.classList.remove('disabled');
            modelManagementBtn.setAttribute('aria-disabled', 'false');
            modelManagementBtn.setAttribute('data-bs-toggle', 'modal');
            modelManagementBtn.setAttribute('data-bs-target', '#modelManagementModal');
            modelManagementBtn.removeEventListener('click', handleModelAlert);
        }
    }
}

/* ERSIN Aciklama. */
export function manualServerRestart() {
    const restartBtn = document.querySelector('.restart-btn');
    if (restartBtn) {
        restartBtn.textContent = 'Yeniden başlatılıyor...';
        restartBtn.disabled = true;
    }
    
    fetch('/api/model/restart-server', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Başarılı', 'Sunucu yeniden başlatıldı. Sayfa yenileniyor...', 'success');
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        } else {
            showToast('Hata', data.error || 'Sunucu yeniden başlatılamadı', 'error');
            if (restartBtn) {
                restartBtn.textContent = 'Sunucuyu Yeniden Başlat';
                restartBtn.disabled = false;
            }
        }
    })
    .catch(error => {
        console.error('Restart error:', error);
        showToast('Hata', 'Sunucu yeniden başlatılırken bir hata oluştu', 'error');
        if (restartBtn) {
            restartBtn.textContent = 'Sunucuyu Yeniden Başlat';
            restartBtn.disabled = false;
        }
    });
}

// ERSIN =====================================
// ERSIN MODAL MANAGEMENT
// ERSIN =====================================

/* ERSIN Aciklama. */
export function setupSliderWithValueDisplay(sliderId, valueDisplayId, defaultValue) {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(valueDisplayId);
    
    if (slider && valueDisplay) {
        slider.value = defaultValue;
        valueDisplay.textContent = defaultValue;
        
        slider.addEventListener('input', function() {
            valueDisplay.textContent = this.value;
        });
    }
    
    return slider;
}

/* ERSIN Aciklama. */
export function setupModalAccessibility(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    
    modal.addEventListener('show.bs.modal', function () {
        this.removeAttribute('aria-hidden');
        document.body.style.overflow = 'hidden';
        console.log(`[DEBUG] ${modalId} modal açıldı, body scroll engellendi`);
    });
    
    modal.addEventListener('hide.bs.modal', function () {
        this.setAttribute('aria-hidden', 'true');
        console.log(`[DEBUG] ${modalId} modal kapandı, aria-hidden eklendi`);
    });
    
    modal.addEventListener('hidden.bs.modal', function () {
        // ERSIN Modal tamamen kapandığında backdrop'ı temizle ve scroll'u geri getir
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => {
            backdrop.remove();
            console.log(`[DEBUG] ${modalId} backdrop temizlendi`);
        });
        document.body.style.overflow = '';
        console.log(`[DEBUG] ${modalId} body scroll geri getirildi`);
    });
}

/* ERSIN Aciklama. */
export function zoomImage(imageSrc, imageTitle = 'Resim Görüntüleyici') {
    const modal = new bootstrap.Modal(document.getElementById('imageZoomModal'));
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('imageZoomModalLabel');
    
    if (modalImage) {
        modalImage.src = imageSrc;
        modalImage.alt = imageTitle;
    }
    
    if (modalTitle) {
        modalTitle.textContent = imageTitle;
    }
    
    modal.show();
}

/* ERSIN Aciklama. */
export function closeZoomModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('imageZoomModal'));
    if (modal) {
        modal.hide();
    }
}

/* ERSIN Aciklama. */
export function addImageClickListeners() {
    // ERSIN Tüm analiz sonucu resimlerine click listener ekle
    document.addEventListener('click', function(e) {
        console.log('[DEBUG] Resim tıklama testi - Element:', e.target);
        console.log('[DEBUG] Element sınıfları:', e.target.classList);
        console.log('[DEBUG] Element tag:', e.target.tagName);
        
        if (e.target.classList && e.target.classList.contains('analysis-image')) {
            e.preventDefault();
            const imageSrc = e.target.src;
            const imageTitle = e.target.alt || 'Analiz Sonucu';
            zoomImage(imageSrc, imageTitle);
        }
    });
}

// ERSIN =====================================
// ERSIN EVENT LISTENERS INITIALIZATION
// ERSIN =====================================

/* ERSIN Aciklama. */
export function initializeEventListeners() {
    // ERSIN Dosya yükleme event'leri
    const uploadBtn = document.getElementById('uploadFileBtn');
    const folderBtn = document.getElementById('uploadFolderBtn');
    const fileInput = document.getElementById('fileInput');
    const folderInput = document.getElementById('folderInput');
    
    if (uploadBtn && fileInput) {
        // ERSIN Click event: Upload butonuna basıldığında file input'u aç
        uploadBtn.addEventListener('click', () => {
            console.log('📁 [DEBUG] Upload button clicked, opening file dialog...');
            fileInput.click();
        });
        
        // ERSIN Change event: Dosya seçildiğinde işle
        fileInput.addEventListener('change', handleFileSelection);
    }
    
    if (folderBtn && folderInput) {
        // ERSIN Click event: Folder butonuna basıldığında folder input'u aç
        folderBtn.addEventListener('click', () => {
            console.log('📁 [DEBUG] Folder button clicked, opening folder dialog...');
            folderInput.click();
        });
        
        // ERSIN Change event: Klasör seçildiğinde işle
        folderInput.addEventListener('change', handleFileSelection);
    }
    
    // ERSIN Drag & Drop event'leri
    const dropZone = document.getElementById('fileDropZone');
    if (dropZone) {
        setupDragAndDrop(dropZone);
    }
    
    // ERSIN Analiz başlatma event'leri
    setupAnalysisButtons();
    
    // ERSIN Modal event'leri
    setupModals();
    
    // ERSIN Image click listener'ları
    addImageClickListeners();
    
    console.log('✅ Event listeners başlatıldı');
}

/* ERSIN Aciklama. */
function setupDragAndDrop(dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropZone.classList.add('drag-over');
    }
    
    function unhighlight() {
        dropZone.classList.remove('drag-over');
    }
    
    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFileSelection({ target: { files } });
        }
    }
}

/* ERSIN Aciklama. */
function setupAnalysisButtons() {
    // ERSIN Analiz Et butonu
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (uploadedFiles.length > 0) {
                // ERSIN Analiz parametreleri modalını aç
                const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
                modal.show();
            }
        };
    }
    
    // ERSIN Analiz Başlatma Onay Butonu (Modal içindeki)
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', () => {
            // ERSIN Analiz parametrelerini al
            const framesPerSecondInput = document.getElementById('framesPerSecond');
            const includeAgeAnalysisInput = document.getElementById('includeAgeAnalysis');

            const framesPerSecond = framesPerSecondInput ? parseFloat(framesPerSecondInput.value) : 1;
            const includeAgeAnalysis = includeAgeAnalysisInput ? includeAgeAnalysisInput.checked : false;
            
            // ERSIN 🔍 DEBUG: Checkbox state'ini logla
            console.log("🔍 CHECKBOX DEBUG:");
            console.log("🔍 includeAgeAnalysisInput element:", includeAgeAnalysisInput);
            console.log("🔍 includeAgeAnalysisInput.checked:", includeAgeAnalysisInput ? includeAgeAnalysisInput.checked : 'element not found');
            console.log("🔍 Final includeAgeAnalysis value:", includeAgeAnalysis);
            
            // ERSIN Modalı kapat
            const modalElement = document.getElementById('runAnalysisSettingsModal');
            if (modalElement) {
                const modalInstance = bootstrap.Modal.getInstance(modalElement);
                if (modalInstance) {
                    modalInstance.hide();
                }
            }
            
            // ERSIN Tüm yüklenen dosyalar için analiz başlat
            startAnalysisForAllFiles(framesPerSecond, includeAgeAnalysis);
        });
    }
}

/* ERSIN Aciklama. */
function setupModals() {
    // ERSIN Analysis settings modal
    setupModalAccessibility('runAnalysisSettingsModal');
    
    // ERSIN Image zoom modal
    setupModalAccessibility('imageZoomModal');
    
    // ERSIN Analysis params modal setup
    const globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal');
    if (globalAnalysisParamsModalElement) {
        setGlobalAnalysisParamsModalElement(globalAnalysisParamsModalElement);
        setupModalAccessibility('analysisParamsModal');
        setupAnalysisParamsModal(globalAnalysisParamsModalElement);
    }
}

/* ERSIN Aciklama. */
function setupAnalysisParamsModal(modalElement) {
    console.log('🔧 setupAnalysisParamsModal çağrıldı');
    const form = document.getElementById('analysisParamsForm');
    const saveBtn = document.getElementById('saveAnalysisParamsBtn');
    const loadDefaultBtn = document.getElementById('loadDefaultAnalysisParamsBtn');
    
    console.log('🔍 Form elements:', { form, saveBtn, loadDefaultBtn });
    
    if (!form) return;
    
    // ERSIN Slider setup
    const faceDetectionConfidenceSlider = setupSliderWithValueDisplay('faceDetectionConfidence', 'faceDetectionConfidenceValue', '0.5');
    const trackingReliabilityThresholdSlider = setupSliderWithValueDisplay('trackingReliabilityThreshold', 'trackingReliabilityThresholdValue', '0.5');
    const idChangeThresholdSlider = setupSliderWithValueDisplay('idChangeThreshold', 'idChangeThresholdValue', '0.45');
    const embeddingDistanceThresholdSlider = setupSliderWithValueDisplay('embeddingDistanceThreshold', 'embeddingDistanceThresholdValue', '0.4');
    
    // ERSIN Modal show event
    modalElement.addEventListener('show.bs.modal', function () {
        loadCurrentAnalysisParams();
    });
    
    // ERSIN 🎯 SAVE BUTTON EVENT LISTENER (from main.js.backup)
    if (saveBtn) {
        saveBtn.addEventListener('click', function() {
            console.log('🔧 Settings kaydediliyor...');
            
            // ERSIN Form validation
            const params = {};
            let formIsValid = true;
            
            // ERSIN Get all form values
            const faceDetectionConfidence = document.getElementById('faceDetectionConfidence');
            const trackingReliabilityThreshold = document.getElementById('trackingReliabilityThreshold');
            const idChangeThreshold = document.getElementById('idChangeThreshold');
            const embeddingDistanceThreshold = document.getElementById('embeddingDistanceThreshold');
            const maxLostFrames = document.getElementById('maxLostFrames');
            
            // ERSIN Collect parameters
            if (faceDetectionConfidence) params.face_detection_confidence = parseFloat(faceDetectionConfidence.value);
            if (trackingReliabilityThreshold) params.tracking_reliability_threshold = parseFloat(trackingReliabilityThreshold.value);
            if (idChangeThreshold) params.id_change_threshold = parseFloat(idChangeThreshold.value);
            if (embeddingDistanceThreshold) params.embedding_distance_threshold = parseFloat(embeddingDistanceThreshold.value);
            if (maxLostFrames) params.max_lost_frames = parseInt(maxLostFrames.value);
            
            if (!formIsValid) return;
            console.log('Saving global params:', params);
            
            // ERSIN Show loading
            const settingsSaveLoader = document.getElementById('settingsSaveLoader');
            console.log('🔧 settingsSaveLoader element:', settingsSaveLoader);
            if (settingsSaveLoader) {
                settingsSaveLoader.style.display = 'flex';
                settingsSaveLoader.style.visibility = 'visible';
                console.log('✅ Loading spinner gösterildi');
            } else {
                console.error('❌ settingsSaveLoader elementi bulunamadı!');
            }
            
            // ERSIN API call
            fetch('/api/settings/analysis-params', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params),
            })
            .then(response => response.json().then(data => ({ status: response.status, body: data })))
            .then(({ status, body }) => {
                console.log('Settings response:', { status, body });
                
                if (status === 200 && body.message) {
                    if (body.restart_required || body.restart_initiated) {
                        // ERSIN Production mode - restart başlatıldı
                        if (window.showToast) {
                            window.showToast('Başarılı', body.message || 'Analiz parametreleri kaydedildi. Sistem yeniden başlatılıyor...', 'success');
                        }
                        console.log('🔄 Production mode - restart başlatıldı');
                        
                        // ERSIN Modal'ı kapat
                        const modal = bootstrap.Modal.getInstance(modalElement);
                        if (modal) modal.hide();
                        
                        // ERSIN Restart sonrası sayfa yenile (force stop gibi)
                        if (body.restart_initiated) {
                            // ERSIN Restart sonrası eski UI state'in (uploadedFiles / overall progress) kalmaması için
                            // ERSIN local restore mekanizmasını bir seferlik devre dışı bırak.
                            try {
                                sessionStorage.setItem('wsanaliz_skip_restore', '1');
                                localStorage.removeItem('wsanaliz_recent_analyses');
                            } catch (e) {
                                console.warn('Restart cleanup storage erişilemedi:', e);
                            }

                            // ERSIN Loading mesajını güncelle (eğer varsa)
                            const loadingMessage = document.getElementById('loadingMessage');
                            if (loadingMessage) {
                                loadingMessage.textContent = 'Sistem yeniden başlatılıyor, lütfen bekleyin...';
                            }
                            
                            // ERSIN 8 saniye bekle sonra sayfa yenile (restart tamamlanması için)
                            setTimeout(() => {
                                console.log('[DEBUG] Analiz parametreleri güncellendi, sayfa yeniden yükleniyor (restart bekleniyor)...');
                                // ERSIN Cache bypass + temiz init için query param ekle
                                const url = new URL(window.location.href);
                                url.searchParams.set('restarted', String(Date.now()));
                                window.location.href = url.toString();
                            }, 8000);
                        }
                    } else {
                        // ERSIN Development mode - auto reload
                        if (window.showToast) {
                            window.showToast('Başarılı', body.message + ' Ayarlar kaydedildi!', 'success');
                        }
                        console.log('✅ Settings başarıyla kaydedildi');
                        
                        // ERSIN Modal'ı kapat
                        const modal = bootstrap.Modal.getInstance(modalElement);
                        if (modal) modal.hide();
                    }
                } else {
                    console.error('Settings kaydetme hatası:', body);
                    if (window.showToast) {
                        window.showToast('Hata', 'Ayarlar kaydedilirken bir hata oluştu: ' + (body.error || 'Bilinmeyen hata'), 'error');
                    }
                }
                
                // ERSIN Hide loading
                if (settingsSaveLoader) {
                    settingsSaveLoader.style.display = 'none';
                    console.log('🔄 Loading spinner gizlendi');
                }
            })
            .catch(error => {
                console.error('Settings fetch hatası:', error);
                if (window.showToast) {
                    window.showToast('Hata', 'Bağlantı hatası: ' + error.message, 'error');
                }
                if (settingsSaveLoader) {
                    settingsSaveLoader.style.display = 'none';
                    console.log('🔄 Loading spinner gizlendi (catch)');
                }
            });
        });
    }
    
    // ERSIN 🎯 LOAD DEFAULTS BUTTON (from main.js.backup)
    if (loadDefaultBtn) {
        console.log('✅ Load defaults button bulundu:', loadDefaultBtn);
        loadDefaultBtn.addEventListener('click', function() {
            console.log('🔧 Default ayarlar yükleniyor...');
            console.log('📡 API call: /api/settings/analysis-params/defaults');
            
            fetch('/api/settings/analysis-params/defaults')
                .then(response => {
                    console.log('📥 Defaults response status:', response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('✅ Default settings loaded:', data);
                    populateFormWithParams(data);
                    if (window.showToast) {
                        window.showToast('Bilgi', 'Varsayılan ayarlar yüklendi', 'info');
                    }
                })
                .catch(error => {
                    console.error('❌ Default settings yükleme hatası:', error);
                    if (window.showToast) {
                        window.showToast('Hata', 'Varsayılan ayarlar yüklenemedi: ' + error.message, 'error');
                    }
                });
        });
    } else {
        console.error('❌ loadDefaultAnalysisParamsBtn elementi bulunamadı!');
    }
    
    // ERSIN Load default button
    if (loadDefaultBtn) {
        loadDefaultBtn.addEventListener('click', function () {
            loadDefaultAnalysisParams();
        });
    }
}

/* ERSIN Aciklama. */
function loadCurrentAnalysisParams() {
    fetch('/api/settings/analysis-params')
        .then(response => response.json())
        .then(data => {
            populateAnalysisParamsForm(data);
            // ERSIN EK: max_lost_frames yoksa inputa 30 yaz
            const el = document.getElementById('maxLostFrames');
            if (el && (data.max_lost_frames === undefined || data.max_lost_frames === null || data.max_lost_frames === '')) {
                el.value = 30;
                console.log('✅ Max Lost Frames default (30) olarak atandı. [loadCurrentAnalysisParams]');
            }
        })
        .catch(error => {
            console.error('loadCurrentAnalysisParams error:', error);
        });
}

/* ERSIN Aciklama. */
function loadDefaultAnalysisParams() {
    fetch('/api/settings/analysis-params/defaults')
    .then(response => response.json())
    .then(data => {
        if (data.success && data.params) {
            populateAnalysisParamsForm(data.params);
            showToast('Bilgi', 'Varsayılan değerler yüklendi.', 'info');
        }
    })
    .catch(error => {
        console.error('Load defaults error:', error);
        showError('Varsayılan değerler yüklenirken bir hata oluştu.');
    });
}

/* ERSIN Aciklama. */
function populateAnalysisParamsForm(params) {
    for (const [key, value] of Object.entries(params)) {
        const element = document.getElementById(key);
        if (element) {
            if (key === 'max_lost_frames' && (value === undefined || value === null || value === '')) {
                element.value = 30;
                console.log('✅ Max Lost Frames default (30) olarak atandı.');
            } else if (element.type === 'range') {
                element.value = value;
                // ERSIN Value display'ini de güncelle
                const valueDisplay = document.getElementById(key + 'Value');
                if (valueDisplay) {
                    valueDisplay.textContent = value;
                }
            } else if (element.type === 'number') {
                element.value = value;
            } else if (element.type === 'checkbox') {
                element.checked = value;
            } else {
                element.value = value;
            }
        }
    }
}

/* ERSIN Aciklama. */
export function exposeUIManagerToWindow() {
    window.uiManager = {
        handleParamsAlert,
        handleModelAlert,
        updateAnalysisParamsButtonState,
        updateAnalysisParamsButtonStateWithQueue,
        manualServerRestart,
        setupSliderWithValueDisplay,
        zoomImage,
        closeZoomModal,
        addImageClickListeners,
        initializeEventListeners
    };
}

// ERSIN 🎯 HELPER FUNCTIONS için Settings
function populateFormWithParams(data) {
    console.log('Populating form with params:', data);
    
    // ERSIN Populate form fields
    if (data.face_detection_confidence !== undefined) {
        const el = document.getElementById('faceDetectionConfidence');
        if (el) {
            el.value = data.face_detection_confidence;
            const valueDisplay = document.getElementById('faceDetectionConfidenceValue');
            if (valueDisplay) {
                valueDisplay.textContent = el.value;
                console.log('✅ Face Detection Confidence güncellendi:', el.value);
            }
            // ERSIN Trigger input event için consistency
            el.dispatchEvent(new Event('input'));
        }
    }
    
    if (data.tracking_reliability_threshold !== undefined) {
        const el = document.getElementById('trackingReliabilityThreshold');
        if (el) {
            el.value = data.tracking_reliability_threshold;
            const valueDisplay = document.getElementById('trackingReliabilityThresholdValue');
            if (valueDisplay) {
                valueDisplay.textContent = el.value;
                console.log('✅ Tracking Reliability güncellendi:', el.value);
            }
            el.dispatchEvent(new Event('input'));
        }
    }
    
    if (data.id_change_threshold !== undefined) {
        const el = document.getElementById('idChangeThreshold');
        if (el) {
            el.value = data.id_change_threshold;
            const valueDisplay = document.getElementById('idChangeThresholdValue');
            if (valueDisplay) {
                valueDisplay.textContent = el.value;
                console.log('✅ ID Change Threshold güncellendi:', el.value);
            }
            el.dispatchEvent(new Event('input'));
        }
    }
    
    if (data.embedding_distance_threshold !== undefined) {
        const el = document.getElementById('embeddingDistanceThreshold');
        if (el) {
            el.value = data.embedding_distance_threshold;
            const valueDisplay = document.getElementById('embeddingDistanceThresholdValue');
            if (valueDisplay) {
                valueDisplay.textContent = el.value;
                console.log('✅ Embedding Distance güncellendi:', el.value);
            }
            el.dispatchEvent(new Event('input'));
        }
    }
    
    if (data.max_lost_frames !== undefined && data.max_lost_frames !== null && data.max_lost_frames !== '') {
        const el = document.getElementById('maxLostFrames');
        if (el) {
            el.value = data.max_lost_frames;
            console.log('✅ Max Lost Frames güncellendi:', el.value);
        }
    } else {
        // ERSIN Eğer değer yoksa default olarak 30 ata
        const el = document.getElementById('maxLostFrames');
        if (el) {
            el.value = 30;
            console.log('✅ Max Lost Frames default (30) olarak atandı.');
        }
    }
}

// ERSIN showToast already defined in globals.js - removed duplicate

// ERSIN Initialize window exposure
exposeUIManagerToWindow(); 