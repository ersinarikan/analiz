/**
 * WSANALIZ - UI Manager Module
 * 
 * Bu modül UI interactions, modal management ve button states'lerini yönetir.
 * main.js'ten extract edilmiştir.
 */

import { 
    uploadedFiles,
    showToast,
    showError,
    setGlobalAnalysisParamsModalElement
} from './globals.js';

import { handleFileSelection } from './file-manager.js';
import { startAnalysisForAllFiles } from './analysis-manager.js';

// =====================================
// UI MANAGEMENT
// =====================================

/**
 * Analiz parametreleri butonu için uyarı gösterme fonksiyonu
 */
export function handleParamsAlert(e) {
    e.preventDefault();
    e.stopPropagation();
    alert('Analiz parametrelerini değiştirmeden önce lütfen yüklenmiş dosyaları kaldırın veya analizi tamamlayın.');
}

/**
 * Model butonları için uyarı gösterme fonksiyonu
 */
export function handleModelAlert(e) {
    e.preventDefault();
    e.stopPropagation();
    alert('Model işlemlerini yapmadan önce lütfen yüklenmiş dosyaları kaldırın veya analizi tamamlayın.');
}

/**
 * Analiz parametreleri ve model yönetimi butonlarının durumunu günceller (sadece yüklü dosyalara göre)
 */
export function updateAnalysisParamsButtonState() {
    updateAnalysisParamsButtonStateWithQueue(null);
}

/**
 * Analiz parametreleri ve model yönetimi butonlarının durumunu günceller (hem yüklü dosya hem kuyruk durumuna göre)
 * @param {Object} queueData - Queue status data
 */
export function updateAnalysisParamsButtonStateWithQueue(queueData) {
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

/**
 * Manual server restart fonksiyonu (production için)
 */
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

// =====================================
// MODAL MANAGEMENT
// =====================================

/**
 * Slider ve value display'ini kurar
 * @param {string} sliderId - Slider element ID
 * @param {string} valueDisplayId - Value display element ID  
 * @param {string} defaultValue - Default value
 * @returns {HTMLElement} - Slider element
 */
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

/**
 * Modal accessibility düzeltmeleri uygular
 * @param {string} modalId - Modal element ID
 */
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
        // Modal tamamen kapandığında backdrop'ı temizle ve scroll'u geri getir
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => {
            backdrop.remove();
            console.log(`[DEBUG] ${modalId} backdrop temizlendi`);
        });
        document.body.style.overflow = '';
        console.log(`[DEBUG] ${modalId} body scroll geri getirildi`);
    });
}

/**
 * Image zoom modal'ını kurar
 * @param {string} imageSrc - Image source URL
 * @param {string} imageTitle - Image title
 */
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

/**
 * Image zoom modal'ını kapatır
 */
export function closeZoomModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('imageZoomModal'));
    if (modal) {
        modal.hide();
    }
}

/**
 * Image click listener'larını kurar
 */
export function addImageClickListeners() {
    // Tüm analiz sonucu resimlerine click listener ekle
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

// =====================================
// EVENT LISTENERS INITIALIZATION
// =====================================

/**
 * Event listener'ları başlatır
 */
export function initializeEventListeners() {
    // Dosya yükleme event'leri
    const uploadBtn = document.getElementById('uploadFileBtn');
    if (uploadBtn) {
        uploadBtn.addEventListener('change', handleFileSelection);
    }
    
    // Drag & Drop event'leri
    const dropZone = document.getElementById('fileDropZone');
    if (dropZone) {
        setupDragAndDrop(dropZone);
    }
    
    // Analiz başlatma event'leri
    setupAnalysisButtons();
    
    // Modal event'leri
    setupModals();
    
    // Image click listener'ları
    addImageClickListeners();
    
    console.log('✅ Event listeners başlatıldı');
}

/**
 * Drag & Drop functionality'sini kurar
 * @param {HTMLElement} dropZone - Drop zone element
 */
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

/**
 * Analiz butonlarını kurar
 */
function setupAnalysisButtons() {
    // Analiz Et butonu
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            if (uploadedFiles.length > 0) {
                // Analiz parametreleri modalını aç
                const modal = new bootstrap.Modal(document.getElementById('runAnalysisSettingsModal'));
                modal.show();
            }
        };
    }
    
    // Analiz Başlatma Onay Butonu (Modal içindeki)
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', () => {
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
    }
}

/**
 * Modal'ları kurar
 */
function setupModals() {
    // Analysis settings modal
    setupModalAccessibility('runAnalysisSettingsModal');
    
    // Image zoom modal
    setupModalAccessibility('imageZoomModal');
    
    // Analysis params modal setup
    const globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal');
    if (globalAnalysisParamsModalElement) {
        setGlobalAnalysisParamsModalElement(globalAnalysisParamsModalElement);
        setupModalAccessibility('analysisParamsModal');
        setupAnalysisParamsModal(globalAnalysisParamsModalElement);
    }
}

/**
 * Analysis parameters modal'ını kurar
 * @param {HTMLElement} modalElement - Modal element
 */
function setupAnalysisParamsModal(modalElement) {
    const form = document.getElementById('analysisParamsForm');
    const saveBtn = document.getElementById('saveAnalysisParamsBtn');
    const loadDefaultBtn = document.getElementById('loadDefaultAnalysisParamsBtn');
    
    if (!form) return;
    
    // Slider setup
    const faceDetectionConfidenceSlider = setupSliderWithValueDisplay('faceDetectionConfidence', 'faceDetectionConfidenceValue', '0.5');
    const trackingReliabilityThresholdSlider = setupSliderWithValueDisplay('trackingReliabilityThreshold', 'trackingReliabilityThresholdValue', '0.5');
    const idChangeThresholdSlider = setupSliderWithValueDisplay('idChangeThreshold', 'idChangeThresholdValue', '0.45');
    const embeddingDistanceThresholdSlider = setupSliderWithValueDisplay('embeddingDistanceThreshold', 'embeddingDistanceThresholdValue', '0.4');
    
    // Modal show event
    modalElement.addEventListener('show.bs.modal', function () {
        // Current settings'leri yükle
        loadCurrentAnalysisParams();
    });
    
    // Load default button
    if (loadDefaultBtn) {
        loadDefaultBtn.addEventListener('click', function () {
            loadDefaultAnalysisParams();
        });
    }
}

/**
 * Current analysis params'ları yükler
 */
function loadCurrentAnalysisParams() {
    fetch('/api/settings/analysis-params')
    .then(response => response.json())
    .then(data => {
        if (data.success && data.params) {
            populateAnalysisParamsForm(data.params);
        }
    })
    .catch(error => {
        console.error('Load params error:', error);
    });
}

/**
 * Default analysis params'ları yükler
 */
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

/**
 * Analysis params form'unu doldurur
 * @param {Object} params - Parameters object
 */
function populateAnalysisParamsForm(params) {
    for (const [key, value] of Object.entries(params)) {
        const element = document.getElementById(key);
        if (element) {
            if (element.type === 'range') {
                element.value = value;
                // Value display'ini de güncelle
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

/**
 * UI Manager fonksiyonlarını window'a expose et
 */
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

// Initialize window exposure
exposeUIManagerToWindow(); 