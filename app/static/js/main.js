// Global değişkenler
let uploadedFiles = [];
let analysisInProgress = false;
let socket;
let hideLoaderTimeout; // Yükleyici için zaman aşımı ID'si

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
const fileAnalysisMap = new Map();  // Maps fileId to analysisId
const cancelledAnalyses = new Set();  // Set of cancelled analysisId values
const fileErrorCounts = new Map();  // Maps fileId to error count
let totalAnalysisCount = 0;
let MAX_STATUS_CHECK_RETRIES = 5;

// Sayfa yüklendiğinde çalışacak fonksiyon
document.addEventListener('DOMContentLoaded', () => {
    const settingsSaveLoader = document.getElementById('settingsSaveLoader'); // Yükleyici elementi
    
    // Socket.io bağlantısı
    initializeSocket(settingsSaveLoader); // Yükleyici elementini initializeSocket'a parametre olarak geç
    
    // Event Listeners
    initializeEventListeners();
    
    // Eğitim butonu kurulumu
    setupTrainingButton();

    // --- Yeni Analiz Parametreleri Modalı (GLOBAL) için Event Listener'lar ve Fonksiyonlar ---
    const globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal'); 
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

                clearTimeout(hideLoaderTimeout); // Önceki zaman aşımını temizle
                hideLoaderTimeout = setTimeout(() => {
                    if (settingsSaveLoader && settingsSaveLoader.style.display === 'flex') {
                        settingsSaveLoader.style.display = 'none';
                        showToast('Uyarı', 'Sunucuyla bağlantı beklenenden uzun sürdü. Sayfayı yenilemeniz gerekebilir.', 'warning');
                        // Modal hala açıksa ve yükleyici zaman aşımına uğradıysa modalı da kapatabiliriz.
                        if (globalAnalysisParamsModalElement && bootstrap.Modal.getInstance(globalAnalysisParamsModalElement)) {
                             bootstrap.Modal.getInstance(globalAnalysisParamsModalElement).hide();
                        }
                    }
                }, 15000); // 15 saniye

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
                        showToast('Bilgi', body.message + ' Sunucu yeniden başlatılıyor, lütfen bekleyin...', 'info');
                        // Yükleyici zaten gösteriliyor, WebSocket bağlantısı ve modalın kapanması bekleniyor.
                        // globalAnalysisParamsModal.hide(); // Hemen gizleme, socket connect'te gizlenecek
                    } else {
                        if(settingsSaveLoader) settingsSaveLoader.style.display = 'none';
                        clearTimeout(hideLoaderTimeout);
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
                    clearTimeout(hideLoaderTimeout);
                    console.error('Error saving global analysis params:', error);
                    alert('Global ayarlar kaydedilirken bir ağ hatası oluştu: ' + error.message);
                });
            });
        }
    } // --- Yeni Analiz Parametreleri Modalı (GLOBAL) için SON ---
});

// Socket.io bağlantısını başlat
function initializeSocket(settingsSaveLoader) { // settingsSaveLoader parametre olarak alındı
    socket = io();
    
    socket.on('connect', () => {
        console.log('WebSocket bağlantısı kuruldu.');
        // const settingsSaveLoader = document.getElementById('settingsSaveLoader'); // Burada tekrar seçmeye gerek yok, parametre olarak geldi
        const globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal');
        
        if (settingsSaveLoader && settingsSaveLoader.style.display === 'flex') {
            settingsSaveLoader.style.display = 'none'; 
            clearTimeout(hideLoaderTimeout); // Başarılı bağlantıda zaman aşımını temizle
            if (globalAnalysisParamsModalElement) {
                const modalInstance = bootstrap.Modal.getInstance(globalAnalysisParamsModalElement);
                if (modalInstance) {
                    modalInstance.hide(); 
                }
            }
            showToast('Bilgi', 'Ayarlar kaydedildi ve sunucu bağlantısı yeniden kuruldu.', 'success');
        }
    });
    
    socket.on('disconnect', () => {
        console.log('WebSocket bağlantısı kesildi.');
    });
    
    socket.on('queue_status', (data) => {
        console.log('Kuyruk durumu güncellendi:', data);
        updateQueueStatus(data);
    });
    
    socket.on('analysis_status_update', (data) => {
        console.log('Analiz durumu güncellendi:', data);
        const { analysis_id, file_id, status, progress, message } = data;
        
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            fileStatuses.set(file.id, status);
            updateFileStatus(file.id, status, progress);
            if (status === 'completed') {
                file.analysisId = analysis_id;
                fileAnalysisMap.set(file.id, analysis_id);
                getAnalysisResults(file.id, analysis_id);
            }
            updateGlobalProgress();
        }
    });
    
    socket.on('analysis_started', (data) => {
        console.log('Analiz başladı:', data);
        const { analysis_id, file_id, file_name, file_type } = data;
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            file.analysisId = analysis_id;
            fileAnalysisMap.set(file.id, analysis_id);
            updateFileStatus(file.id, 'Analiz Başlatıldı', 10);
            console.log(`Analiz başlatıldı: ${file_name} (${file_type}), ID: ${analysis_id}`);
            showToast('Bilgi', `${file_name} analizi başlatıldı.`, 'info');
            setTimeout(() => checkAnalysisStatus(analysis_id, file.id), 1000);
        }
    });
    
    socket.on('analysis_progress', (data) => {
        console.log('Analiz ilerliyor:', data);
        const { analysis_id, file_id, current_frame, total_frames, progress, detected_faces, high_risk_frames } = data;
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            const status_msg = `Analiz: ${current_frame}/${total_frames} kare`; // status değişken adını status_msg olarak değiştirdim.
            updateFileStatus(file.id, status_msg, progress);
            const fileCard = document.getElementById(file.id);
            if (fileCard) {
                const statusElement = fileCard.querySelector('.file-status-text');
                if (statusElement) {
                    statusElement.title = `İşlenen kare: ${current_frame}/${total_frames}\nTespit edilen yüz: ${detected_faces || 0}\nYüksek riskli kare: ${high_risk_frames || 0}\nİlerleme: %${progress.toFixed(1)}`;
                }
            }
            if (current_frame % 10 === 0 || current_frame === total_frames) {
                console.log(`Analiz ilerliyor: ${file.name}, Kare: ${current_frame}/${total_frames}, İlerleme: %${progress.toFixed(1)}`);
            }
        }
    });
    
    socket.on('analysis_completed', (data) => {
        console.log('Analiz tamamlandı:', data);
        const { analysis_id, file_id, elapsed_time, message } = data;
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            updateFileStatus(file.id, 'completed', 100);
            console.log(`Analiz tamamlandı: ${file.name}, Süre: ${elapsed_time ? elapsed_time.toFixed(1) + 's' : 'bilinmiyor'}`);
            file.analysisId = analysis_id;
            fileAnalysisMap.set(file.id, analysis_id);
            getAnalysisResults(file.id, analysis_id);
            showToast('Başarılı', `${file.name} analizi tamamlandı (${elapsed_time ? elapsed_time.toFixed(1) + ' saniye' : 'bilinmiyor'}).`, 'success');
            updateGlobalProgress();
        }
    });
    
    socket.on('analysis_failed', (data) => {
        console.error('Analiz hatası:', data);
        const { analysis_id, file_id, error, elapsed_time } = data;
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            updateFileStatus(file.id, 'failed', 0);
            console.error(`Analiz hatası: ${file.name}, Süre: ${elapsed_time ? elapsed_time.toFixed(1) + 's' : 'bilinmiyor'}, Hata: ${error}`);
            showToast('Hata', `Analiz hatası: ${error}`, 'danger');
        }
    });
    
    socket.on('training_progress', (data) => {
        updateTrainingProgress(data);
    });
    
    socket.on('training_completed', (data) => {
        handleTrainingCompleted(data);
    });
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
    
    // Yapay Zeka Eğitim Butonu
    document.getElementById('trainModelBtn').addEventListener('click', () => {
        const modal = new bootstrap.Modal(document.getElementById('trainModelModal'));
        modal.show();
    });
    
    // Eğitim Başlatma Butonu
    document.getElementById('startTrainingBtn').addEventListener('click', startModelTraining);
    
    // Model Sıfırlama Butonları
    document.getElementById('resetContentModelBtn').addEventListener('click', () => resetModel('content'));
    document.getElementById('resetAgeModelBtn').addEventListener('click', () => resetModel('age'));
    
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
function startQueueStatusChecker() {
    // İlk kontrol
    checkQueueStatus();
    
    // 5 saniyede bir kontrol et
    setInterval(checkQueueStatus, 5000);
}

// Kuyruk durumunu kontrol et
function checkQueueStatus() {
    fetch('/api/debug/queue-status')
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
function updateQueueStatus(data) {
    const queueStatusElement = document.getElementById('queueStatus');
    if (!queueStatusElement) return;
    
    if (data && (data.active || data.size > 0)) {
        // Kuyruk aktif veya bekleyen dosya varsa
        const waitingCount = data.size || 0;
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
    formData.append('file', file);
    
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
        
        // Dosya durumunu güncelle
        updateFileStatus(file.id, 'Sırada', 100);
        
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
    // Dosya zaten listeye eklenmişse tekrar ekleme
    if (uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
        showToast('Bilgi', `${file.name} dosyası zaten eklenmiş.`, 'info');
        return;
    }
    
    // Dosya ID'si oluştur
    const fileId = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Dosyaya ID ekle
    file.id = fileId;
    
    // Dosyayı global listeye ekle
    uploadedFiles.push(file);
    
    // Dosya önizleme kartını oluştur
    const fileCard = createFileCard(file);
    
    // Dosya listesine ekle
    document.getElementById('fileList').appendChild(fileCard);
}

// Dosya kartı oluştur
function createFileCard(file) {
    // Template'i klonla
    const template = document.getElementById('fileCardTemplate');
    const fileCard = template.content.cloneNode(true);
    
    // Karta dosya ID'si ata
    fileCard.querySelector('.file-card').id = file.id;
    
    // Dosya adı ve boyutu ayarla
    fileCard.querySelector('.filename').textContent = file.name;
    fileCard.querySelector('.filesize').textContent = formatFileSize(file.size);
    
    // Dosya önizlemesi oluştur
    createFilePreview(file, fileCard.querySelector('.file-preview'));
    
    // Dosya silme butonuna olay dinleyicisi ekle
    fileCard.querySelector('.remove-file-btn').addEventListener('click', () => removeFile(file.id));
    
    return fileCard.querySelector('.file-card');
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
    // Global listeden dosyayı kaldır
    const fileIndex = uploadedFiles.findIndex(file => file.id === fileId);
    
    if (fileIndex !== -1) {
        uploadedFiles.splice(fileIndex, 1);
        
        // DOM'dan dosya kartını kaldır
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
    
    // Her bir dosya için analiz başlat
    filesToAnalyze.forEach(file => {
        startAnalysis(file.id, file.fileId, framesPerSecond, includeAgeAnalysis);
    });
}

// Analiz işlemini başlat
function startAnalysis(fileId, serverFileId, framesPerSecond, includeAgeAnalysis) {
    // Dosya durumunu "işleniyor" olarak ayarla
    updateFileStatus(fileId, "processing", 0);
    fileStatuses.set(fileId, "processing");
    
    // Analiz parametrelerini hazırla
    const analysisParams = {
        file_id: serverFileId,
        frames_per_second: framesPerSecond,
        include_age_analysis: includeAgeAnalysis
    };
    
    // Analiz başlatma API çağrısı
    fetch("/api/analysis/start", {
        method: "POST",
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
        
        // Socket.io tarafından zaten işlenmemişse analiz durumunu kontrol et
        // (Yani fileAnalysisMap'te bu dosya için bir analysisId yoksa)
        if (!fileAnalysisMap.has(fileId)) {
            fileAnalysisMap.set(fileId, analysisId);
            
            // Dosyaya analiz ID'sini ekle
            const fileIndex = uploadedFiles.findIndex(f => f.id === fileId);
            if (fileIndex !== -1) {
                uploadedFiles[fileIndex].analysisId = analysisId;
            }
            
            // Hata sayacını sıfırla
            fileErrorCounts.set(fileId, 0);
            
            // İlerlemeyi kontrol etmeye başla
            setTimeout(() => checkAnalysisStatus(analysisId, fileId), 1000);
        }
    })
    .catch(error => {
        console.error("Error starting analysis:", error);
        updateFileStatus(fileId, "failed", 0);
        fileStatuses.set(fileId, "failed");
        showToast('Hata', `${fileNameFromId(fileId)} dosyası için analiz başlatılamadı: ${error.message}`, 'danger');
        updateGlobalProgress();
    });
}

// Analiz durumunu kontrol et
function checkAnalysisStatus(analysisId, fileId) {
    // Analiz ID'si yoksa işlemi durdur
    if (!analysisId) {
        console.error(`No analysis ID for file ${fileId}, cannot check status`);
        return;
    }
    
    // İptal edilen analizleri kontrol etme
    if (cancelledAnalyses.has(analysisId)) {
        console.log(`Analysis ${analysisId} was cancelled, stopping status checks`);
        return;
    }

    // Hata sayacını kontrol et
    let errorCount = fileErrorCounts.get(fileId) || 0;
    if (errorCount > MAX_STATUS_CHECK_RETRIES) {
        console.error(`Max retries (${MAX_STATUS_CHECK_RETRIES}) exceeded for analysis ${analysisId}`);
        updateFileStatus(fileId, "failed", 0);
        fileStatuses.set(fileId, "failed");
        showError(`${fileNameFromId(fileId)} dosyası için durum kontrolü başarısız oldu: Çok sayıda hata oluştu.`);
        updateGlobalProgress();
        return;
    }

    fetch(`/api/analysis/${analysisId}/status`)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(response => {
        console.log(`Analysis status for ${analysisId}:`, response);
        
        // Duruma göre işle
        const status = response.status;
        const progress = response.progress || 0;
        
        // Dosya durumunu güncelle
        fileStatuses.set(fileId, status);
        
        // Dosya nesnesini bul ve güncelle
        const fileIndex = uploadedFiles.findIndex(f => f.id === fileId);
        if (fileIndex !== -1) {
            uploadedFiles[fileIndex].analysisId = analysisId;
            uploadedFiles[fileIndex].status = status;
        }
        
        // Kuyrukta bekliyor durumu için özel mesaj
        if (status === "queued") {
            const queueMessage = "Sırada";
            updateFileStatus(fileId, queueMessage, 0);
            
            // Kuyrukta bekleyen öğeyi kontrol etmeye devam et
            setTimeout(() => checkAnalysisStatus(analysisId, fileId), 2000);
        } else if (status === "processing") {
            // İşlem yapılıyorsa ilerleyişi göster
            updateFileStatus(fileId, status, progress);
            
            // Analiz devam ediyorsa durumu kontrol etmeye devam et
            setTimeout(() => checkAnalysisStatus(analysisId, fileId), 2000);
        } else if (status === "completed") {
            // Analiz tamamlandıysa sonuçları göster
            updateFileStatus(fileId, status, 100);
            getAnalysisResults(fileId, analysisId);
        } else if (status === "failed") {
            // Analiz başarısız olduysa hata mesajı göster
            updateFileStatus(fileId, status, 0);
            showError(`${fileNameFromId(fileId)} dosyası için analiz başarısız oldu: ${response.error || "Bilinmeyen hata"}`);
        } else {
            // Diğer durumlar için (cancelled vb)
            updateFileStatus(fileId, status, progress);
            
            // İşlem devam ediyorsa kontrol etmeye devam et
            if (status !== "completed" && status !== "failed") {
                setTimeout(() => checkAnalysisStatus(analysisId, fileId), 2000);
            }
        }
        
        // Genel ilerlemeyi güncelle
        updateGlobalProgress();
    })
    .catch(error => {
        console.error(`Error checking analysis status for ${analysisId}:`, error);
        
        // Hata sayacını arttır
        fileErrorCounts.set(fileId, errorCount + 1);
        
        // Bir süre bekleyip tekrar dene
        setTimeout(() => checkAnalysisStatus(analysisId, fileId), 5000);
    });
}

// Tüm analizlerin tamamlanıp tamamlanmadığını kontrol eden yardımcı fonksiyon
function checkAllAnalysesCompleted() {
    // Tüm dosya durumlarını kontrol et
    for (const [fileId, status] of fileStatuses.entries()) {
        // İptal edilmiş analizleri tamamlanmış olarak kabul et
        const analysisId = fileAnalysisMap.get(fileId);
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
function updateFileStatus(fileId, status, progress, error = null) {
    const fileCard = document.getElementById(fileId);
    
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
    }
    
    // Durum metni
    const statusText = fileCard.querySelector('.file-status-text');
    statusText.textContent = displayStatus;
    
    // Durum etiketi
    const statusBadge = fileCard.querySelector('.file-status');
    statusBadge.textContent = displayStatus;
    
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
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
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
    } else if (displayStatus === 'Hata' || status === 'failed') {
        progressBar.classList.add('bg-danger');
    } else if (displayStatus === 'Analiz Ediliyor' || status === 'processing' || displayStatus.startsWith('Analiz:') || displayStatus === 'Analiz Başlatıldı') {
        // Analiz sırasında daha göze çarpan renk
        progressBar.classList.add('bg-primary');
        progressBar.classList.add('progress-bar-striped');
        progressBar.classList.add('progress-bar-animated');
    }
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

// Analiz sonuçlarını al
function getAnalysisResults(fileId, analysisId) {
    console.log(`Analiz sonuçları alınıyor: fileId=${fileId}, analysisId=${analysisId}`);
    
    if (!analysisId) {
        console.error(`Analiz ID bulunamadı, fileId=${fileId}`);
        showToast('Hata', `Analiz ID'si bulunamadı. Bu beklenmeyen bir durum.`, 'danger');
        return;
    }
    
    // Yükleme göstergesi ekleyin
    const resultsList = document.getElementById('resultsList');
    if (resultsList) {
        const loadingEl = document.createElement('div');
        loadingEl.id = `loading-${fileId}`;
        loadingEl.className = 'text-center my-3';
        loadingEl.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Yükleniyor...</span></div><p class="mt-2">Sonuçlar yükleniyor...</p>';
        resultsList.appendChild(loadingEl);
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
        
        // Sonuçlar bölümünü görünür yap
        document.getElementById('resultsSection').style.display = 'block';
        
        // Genel ilerlemeyi güncelle
        updateGlobalProgress();
        
        // Tüm analizlerin tamamlanıp tamamlanmadığını kontrol et
        if (checkAllAnalysesCompleted()) {
            console.log("Tüm analizler tamamlandı");
            // Tamamlandı mesajını göster
            const completedElement = document.getElementById('completedMessage');
            if (completedElement) {
                completedElement.style.display = 'block';
            }
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
        const suspiciousScores = detectSuspiciousScores(results);
        
        // Şüpheli skor varsa uyarı göster
        if (suspiciousScores.length > 0) {
            const warningEl = document.createElement('div');
            warningEl.className = 'alert alert-warning mb-3';
            warningEl.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Dikkat:</strong> Bazı kategorilerde skorlar beklenenden yüksek çıkmış olabilir.
                <small>(${suspiciousScores.join(', ')} kategorilerinde değerlendirme yaparken dikkatli olunuz)</small>
            `;
            riskScoresContainer.appendChild(warningEl);
        }
        
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
                // Güvenli kategori için farklı risk yorumlaması
                if (score >= 70) {
                    riskLevel = 'Yüksek Güven';
                    riskClass = 'risk-level-low'; // Yeşil renk
                } else if (score >= 30) {
                    riskLevel = 'Orta Güven';
                    riskClass = 'risk-level-medium'; // Sarı renk
                } else {
                    riskLevel = 'Düşük Güven';
                    riskClass = 'risk-level-high'; // Kırmızı renk
                }
            } else {
                // Diğer kategoriler için normal risk yorumlaması
                if (score >= 70) {
                    riskLevel = 'Yüksek Risk';
                    riskClass = 'risk-level-high';
                } else if (score >= 30) {
                    riskLevel = 'Orta Risk';
                    riskClass = 'risk-level-medium';
                } else {
                    riskLevel = 'Düşük Risk';
                    riskClass = 'risk-level-low';
                }
            }
            
            // Şüpheli skor ise işaretle
            const isSuspicious = suspiciousScores.includes(categoryName);
            
            // Kategori rengini belirle
            let progressBarClass = '';
            if (category === 'safe') {
                // Güvenli kategorisi için yeşil ton kullan, değer yükseldikçe daha koyu yeşil
                progressBarClass = score >= 70 ? 'bg-success' : score >= 30 ? 'bg-info' : 'bg-warning';
            } else {
                // Diğer kategoriler için risk arttıkça kırmızılaşan renk
                progressBarClass = riskClass === 'risk-level-high' ? 'bg-danger' : 
                                  riskClass === 'risk-level-medium' ? 'bg-warning' : 'bg-success';
            }
            
            // Varsa güven skorunu al
            const confidenceScore = hasConfidenceScores ? (confidenceScores[category] || 0) : 0;
            const showConfidence = hasConfidenceScores && confidenceScore > 0;
            
            // Skor elementi HTML'i - güven skoru varsa ekle
            scoreElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span>${categoryName} ${isSuspicious ? '<i class="fas fa-question-circle text-warning" title="Bu kategori skoru tutarsız olabilir"></i>' : ''}</span>
                    <span class="risk-score ${riskClass}">${score.toFixed(0)}% - ${riskLevel}</span>
                </div>
                <div class="progress mb-1">
                    <div class="progress-bar ${progressBarClass}" 
                         role="progressbar" style="width: ${score}%" 
                         aria-valuenow="${score}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                ${showConfidence ? `
                <div class="d-flex justify-content-between align-items-center small text-muted">
                    <span>Güven Skoru:</span>
                    <span>${(confidenceScore * 100).toFixed(0)}%</span>
                </div>
                <div class="progress" style="height: 4px;">
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
                                        style="width: 100%; height: 100%; object-fit: cover;"
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
                        frameUrl = `/${face.processed_image_path}`;
                        console.log("[DEBUG] İşlenmiş görsel URL'si:", frameUrl);
                        
                        col.innerHTML = `
                            <div class="card h-100 ${ageClass}">
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
    const contentId = form.querySelector('.content-id').value;
    
    // Form verilerini topla
    const feedbackData = {
        content_id: contentId,
        rating: parseInt(form.querySelector('.general-rating').value),
        comment: form.querySelector('.feedback-comment').value,
        category_feedback: {
            violence: form.querySelector('.violence-feedback').value,
            adult_content: form.querySelector('.adult-content-feedback').value,
            harassment: form.querySelector('.harassment-feedback').value,
            weapon: form.querySelector('.weapon-feedback').value,
            drug: form.querySelector('.drug-feedback').value
        },
        // Kategori için doğru değer geri bildirimleri
        category_correct_values: {
            violence: form.querySelector('.violence-correct-value') ? parseFloat(form.querySelector('.violence-correct-value').value) : null,
            adult_content: form.querySelector('.adult-content-correct-value') ? parseFloat(form.querySelector('.adult-content-correct-value').value) : null,
            harassment: form.querySelector('.harassment-correct-value') ? parseFloat(form.querySelector('.harassment-correct-value').value) : null,
            weapon: form.querySelector('.weapon-correct-value') ? parseFloat(form.querySelector('.weapon-correct-value').value) : null,
            drug: form.querySelector('.drug-correct-value') ? parseFloat(form.querySelector('.drug-correct-value').value) : null
        }
    };
    
    // Geri bildirimi gönder
    fetch('/api/feedback/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(feedbackData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Geri bildirim başarıyla gönderildi:', data);
        showToast('Başarılı', 'Geri bildiriminiz için teşekkürler. Model eğitiminde kullanılacaktır.', 'success');
        
        // Geri bildirim butonunu pasif yap
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-check me-1"></i> Gönderildi';
    })
    .catch(error => {
        console.error('Geri bildirim gönderme hatası:', error);
        showToast('Hata', `Geri bildirim gönderilirken hata oluştu: ${error.message}`, 'danger');
    });
}

// Yaş geri bildirimi gönder
function submitAgeFeedback(personId, ageInput) {
    const correctedAge = parseInt(ageInput.value);
    const ageRangeCorrect = document.querySelector(`#ageRange-${personId}`).checked;
    
    if (isNaN(correctedAge) || correctedAge <= 0 || correctedAge > 100) {
        showToast('Uyarı', 'Lütfen geçerli bir yaş değeri girin (1-100).', 'warning');
        return;
    }
    
    // Yaş geri bildirimini gönder
    fetch('/api/feedback/age', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            person_id: personId,
            corrected_age: correctedAge,
            age_range_correct: ageRangeCorrect
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Yaş geri bildirimi başarıyla gönderildi:', data);
        showToast('Başarılı', 'Yaş geri bildirimi kaydedildi. Model eğitiminde kullanılacaktır.', 'success');
        
        // Geri bildirim butonunu pasif yap ve metni değiştir
        const button = document.querySelector(`button[onclick*="submitAgeFeedback('${personId}"]`);
        if (button) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-check"></i>';
            button.classList.remove('btn-primary');
            button.classList.add('btn-success');
        }
        
        // Input'u readonly yap
        const input = document.getElementById(`realAge-${personId}`);
        if (input) {
            input.readOnly = true;
        }
    })
    .catch(error => {
        console.error('Yaş geri bildirimi gönderme hatası:', error);
        showToast('Hata', `Yaş geri bildirimi gönderilirken hata oluştu: ${error.message}`, 'danger');
    });
}

// Geliştirilmiş yaş tahmini display için yardımcı fonksiyon
function createAgeFeedbackElements(faceId, estimatedAge, personId) {
    const feedbackContainer = document.createElement('div');
    feedbackContainer.className = 'age-feedback-container mt-2';
    
    feedbackContainer.innerHTML = `
        <div class="input-group">
            <span class="input-group-text">Doğru Yaş:</span>
            <input type="number" class="form-control age-feedback-input" 
                   min="1" max="100" placeholder="Gerçek yaş" 
                   value="${estimatedAge}" data-person-id="${personId}">
            <button class="btn btn-primary age-feedback-btn" 
                    data-person-id="${personId}" 
                    onclick="submitAgeFeedback('${personId}', this.previousElementSibling)">
                Gönder
            </button>
        </div>
        <div class="form-check mt-1">
            <input class="form-check-input age-accuracy-check" type="checkbox" id="accuracy-${faceId}">
            <label class="form-check-label" for="accuracy-${faceId}">
                Yaş aralığı doğru ama kesin yaş yanlış
            </label>
        </div>
    `;
    
    return feedbackContainer;
}

// Model metrikleri yükle
function loadModelMetrics() {
    // Önce içerik analiz modeli metrikleri
    fetch('/api/model/metrics/content')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('İçerik analiz modeli metrikleri:', data);
        
        // Model versiyonlarını al
        return fetch('/api/model/versions/content')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(versionData => {
                // Versiyon bilgilerini ekle
                data.versions = versionData.versions;
                return data;
            })
            .catch(error => {
                console.error('Model versiyonları alınamadı:', error);
                return data;
            });
    })
    .then(data => {
        displayContentModelMetrics(data);
    })
    .catch(error => {
        console.error('İçerik analiz modeli metrikleri alınırken hata:', error);
        document.getElementById('contentMetricsTab').innerHTML = `
            <div class="alert alert-danger">Metrikler yüklenirken hata oluştu: ${error.message}</div>
        `;
    });
    
    // Sonra yaş analiz modeli metrikleri
    fetch('/api/model/metrics/age')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Yaş analiz modeli metrikleri:', data);
        
        // Model versiyonlarını al
        return fetch('/api/model/versions/age')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(versionData => {
                // Versiyon bilgilerini ekle
                data.versions = versionData.versions;
                return data;
            })
            .catch(error => {
                console.error('Model versiyonları alınamadı:', error);
                return data;
            });
    })
    .then(data => {
        displayAgeModelMetrics(data);
    })
    .catch(error => {
        console.error('Yaş analiz modeli metrikleri alınırken hata:', error);
        document.getElementById('ageMetricsTab').innerHTML = `
            <div class="alert alert-danger">Metrikler yüklenirken hata oluştu: ${error.message}</div>
        `;
    });
}

// İçerik analiz modeli metriklerini göster
function displayContentModelMetrics(data) {
    // Genel metrikler
    document.querySelector('.content-accuracy').textContent = data.metrics.accuracy ? `${(data.metrics.accuracy * 100).toFixed(1)}%` : '-';
    document.querySelector('.content-precision').textContent = data.metrics.precision ? `${(data.metrics.precision * 100).toFixed(1)}%` : '-';
    document.querySelector('.content-recall').textContent = data.metrics.recall ? `${(data.metrics.recall * 100).toFixed(1)}%` : '-';
    document.querySelector('.content-f1').textContent = data.metrics.f1 ? `${(data.metrics.f1 * 100).toFixed(1)}%` : '-';
    
    // Kategori bazında metrikler
    const categoryMetricsTable = document.getElementById('contentCategoryMetrics');
    categoryMetricsTable.innerHTML = '';
    
    if (data.category_metrics) {
        for (const [category, metrics] of Object.entries(data.category_metrics)) {
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
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${categoryName}</td>
                <td>${metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : '-'}</td>
                <td>${metrics.precision ? `${(metrics.precision * 100).toFixed(1)}%` : '-'}</td>
                <td>${metrics.recall ? `${(metrics.recall * 100).toFixed(1)}%` : '-'}</td>
                <td>${metrics.f1 ? `${(metrics.f1 * 100).toFixed(1)}%` : '-'}</td>
            `;
            
            categoryMetricsTable.appendChild(row);
        }
    } else {
        categoryMetricsTable.innerHTML = '<tr><td colspan="5" class="text-center">Kategori metrikler mevcut değil.</td></tr>';
    }
    
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
    // Genel metrikler
    document.querySelector('.age-mae').textContent = data.metrics.mae ? `${data.metrics.mae.toFixed(1)} yaş` : '-';
    document.querySelector('.age-accuracy').textContent = data.metrics.accuracy ? `${(data.metrics.accuracy * 100).toFixed(1)}%` : '-';
    document.querySelector('.age-count').textContent = data.metrics.count ? data.metrics.count : '-';
    
    // Yaş dağılımı grafiği
    if (data.age_distribution) {
        const ageDistributionCanvas = document.getElementById('ageDistributionChart');
        const ageDistributionCtx = ageDistributionCanvas.getContext('2d');
        
        // Mevcut grafiği temizle
        if (window.ageDistributionChart) {
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
        if (window.ageErrorChart) {
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

// Model versiyonlarını göster
function displayModelVersions(modelType, versions) {
    const container = document.getElementById(`${modelType}VersionsContainer`);
    if (!container) {
        console.error(`${modelType}VersionsContainer bulunamadı`);
        return;
    }
    
    // Container'ı temizle
    container.innerHTML = `<h5 class="mb-3">Model Versiyonları</h5>`;
    
    if (!versions || versions.length === 0) {
        container.innerHTML += '<div class="alert alert-info">Henüz kaydedilmiş model versiyonu bulunmuyor.</div>';
        return;
    }
    
    // Versiyon listesi oluştur
    const versionsList = document.createElement('div');
    versionsList.className = 'list-group mb-3';
    
    versions.forEach(version => {
        const versionItem = document.createElement('div');
        versionItem.className = `list-group-item ${version.is_active ? 'list-group-item-success' : ''}`;
        
        // Metrik bilgilerini hazırla
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
                                <small>F1: <strong>${version.metrics.f1 ? (version.metrics.f1*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-3">
                                <small>Kesinlik: <strong>${version.metrics.precision ? (version.metrics.precision*100).toFixed(1) + '%' : 'N/A'}</strong></small>
                            </div>
                            <div class="col-md-3">
                                <small>Duyarlılık: <strong>${version.metrics.recall ? (version.metrics.recall*100).toFixed(1) + '%' : 'N/A'}</strong></small>
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
                        : `<button class="btn btn-sm btn-outline-primary activate-version-btn" data-version-id="${version.id}">Aktifleştir</button>`
                    }
                </div>
            </div>
            ${metricsHtml}
            ${trainingInfo}
        `;
        
        versionsList.appendChild(versionItem);
    });
    
    container.appendChild(versionsList);
    
    // Sıfırlama butonu ekle
    const resetButton = document.createElement('button');
    resetButton.className = 'btn btn-danger mt-3';
    resetButton.innerHTML = '<i class="fas fa-undo-alt me-2"></i>Modeli Sıfırla';
    resetButton.onclick = () => confirmModelReset(modelType);
    container.appendChild(resetButton);
    
    // Aktifleştirme butonlarına olay dinleyici ekle
    const activateButtons = container.querySelectorAll('.activate-version-btn');
    activateButtons.forEach(button => {
        button.addEventListener('click', function() {
            const versionId = this.dataset.versionId;
            activateModelVersion(versionId, modelType);
        });
    });
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

// Model eğitimini başlat
function startModelTraining() {
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
    document.getElementById('trainingStatusText').textContent = 'Eğitim hazırlanıyor...';
    
    // Eğitim butonunu devre dışı bırak
    document.getElementById('startTrainingBtn').disabled = true;
    
    // Eğitim isteği gönder
    fetch('/api/model/train', {
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
        console.log('Eğitim başlatıldı:', data);
        document.getElementById('trainingStatusText').textContent = 'Eğitim başlatıldı. Bu işlem biraz zaman alabilir...';
        
        // Eğitim durumunu güncellemek için timer başlat
        checkTrainingStatus(data.training_id);
    })
    .catch(error => {
        console.error('Eğitim başlatma hatası:', error);
        document.getElementById('trainingStatusText').textContent = `Eğitim başlatılamadı: ${error.message}`;
        document.getElementById('startTrainingBtn').disabled = false;
        showToast('Hata', `Eğitim başlatılırken hata oluştu: ${error.message}`, 'danger');
    });
}

// Eğitim durumunu kontrol et
function checkTrainingStatus(trainingId) {
    fetch(`/api/model/training-status/${trainingId}`)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // İlerlemeyi güncelle
        const progressBar = document.getElementById('trainingProgressBar');
        progressBar.style.width = `${data.progress}%`;
        progressBar.textContent = `${data.progress}%`;
        progressBar.setAttribute('aria-valuenow', data.progress);
        
        // Durum metnini güncelle
        document.getElementById('trainingStatusText').textContent = data.status_message;
        
        if (data.status === 'completed') {
            // Eğitim tamamlandı, sonuçları göster
            displayTrainingResults(data.results);
            document.getElementById('startTrainingBtn').disabled = false;
        } else if (data.status === 'failed') {
            // Eğitim başarısız oldu
            document.getElementById('trainingStatusText').textContent = `Eğitim başarısız oldu: ${data.status_message}`;
            document.getElementById('startTrainingBtn').disabled = false;
            showToast('Hata', `Eğitim başarısız oldu: ${data.status_message}`, 'danger');
        } else {
            // Eğitim devam ediyor, tekrar kontrol et
            setTimeout(() => {
                checkTrainingStatus(trainingId);
            }, 2000);
        }
    })
    .catch(error => {
        console.error('Eğitim durumu kontrolü hatası:', error);
        document.getElementById('trainingStatusText').textContent = `Eğitim durumu kontrol edilemedi: ${error.message}`;
        document.getElementById('startTrainingBtn').disabled = false;
        showToast('Hata', `Eğitim durumu kontrol edilirken hata oluştu: ${error.message}`, 'danger');
    });
}

// Eğitim sonuçlarını göster
function displayTrainingResults(results) {
    const resultsSection = document.getElementById('trainingResultsSection');
    resultsSection.style.display = 'block';
    
    const resultsContainer = document.getElementById('trainingResults');
    
    if (results.model_type === 'content') {
        // İçerik modeli sonuçları
        resultsContainer.innerHTML = `
            <div class="alert alert-success">
                <strong>Eğitim Başarıyla Tamamlandı!</strong>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-sm">
                        <tr>
                            <th>Eğitim Süresi:</th>
                            <td>${formatDuration(results.duration)}</td>
                        </tr>
                        <tr>
                            <th>Örnek Sayısı:</th>
                            <td>${results.samples} (Eğitim: ${results.training_samples}, Doğrulama: ${results.validation_samples})</td>
                        </tr>
                        <tr>
                            <th>Epoch Sayısı:</th>
                            <td>${results.epochs}</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <table class="table table-sm">
                        <tr>
                            <th>Doğruluk:</th>
                            <td>${(results.metrics.accuracy * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>F1 Skoru:</th>
                            <td>${(results.metrics.f1 * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <th>Hassasiyet:</th>
                            <td>${(results.metrics.precision * 100).toFixed(2)}%</td>
                        </tr>
                    </table>
                </div>
            </div>
        `;
    } else {
        // Yaş modeli sonuçları
        resultsContainer.innerHTML = `
            <div class="alert alert-success">
                <strong>Eğitim Başarıyla Tamamlandı!</strong>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-sm">
                        <tr>
                            <th>Eğitim Süresi:</th>
                            <td>${formatDuration(results.duration)}</td>
                        </tr>
                        <tr>
                            <th>Örnek Sayısı:</th>
                            <td>${results.samples} (Eğitim: ${results.training_samples}, Doğrulama: ${results.validation_samples})</td>
                        </tr>
                        <tr>
                            <th>Epoch Sayısı:</th>
                            <td>${results.epochs}</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <table class="table table-sm">
                        <tr>
                            <th>Ortalama Mutlak Hata (MAE):</th>
                            <td>${results.metrics.mae.toFixed(2)} yaş</td>
                        </tr>
                        <tr>
                            <th>±3 Yaş Başarı Oranı:</th>
                            <td>${(results.metrics.accuracy * 100).toFixed(2)}%</td>
                        </tr>
                    </table>
                </div>
            </div>
        `;
    }
    
    // Metrikleri yenile
    loadModelMetrics();
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
        trainingBtn.removeEventListener('click', startModelTraining);
        trainingBtn.addEventListener('click', startTrainingWithFeedback);
    }
}

// Socket.io eğitim ilerleme güncellemesi
function updateTrainingProgress(data) {
    const { progress, status_message } = data;
    
    // İlerleme çubuğunu güncelle
    const progressBar = document.getElementById('trainingProgressBar');
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
    // Durum metnini güncelle
    document.getElementById('trainingStatusText').textContent = status_message;
}

// Socket.io eğitim tamamlandı
function handleTrainingCompleted(data) {
    // Eğitim sonuçlarını göster
    displayTrainingResults(data.results);
    
    // Butonları aktif et
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

// Şüpheli skorları tespit etme yardımcı fonksiyonu
function detectSuspiciousScores(results) {
    const suspiciousCategories = [];
    const scores = results.overall_scores;
    
    // Güvenli skoru yüksek ama diğer kategorilerde de yüksek skorlar varsa
    if (scores.safe > 0.5 && (scores.violence > 0.4 || scores.adult_content > 0.4 || scores.harassment > 0.4 || scores.weapon > 0.4 || scores.drug > 0.4)) {
        suspiciousCategories.push('Güvenli');
    }
    
    // Anormal skor kombinasyonları
    if (scores.drug > 0.8 && scores.violence < 0.4 && scores.weapon < 0.3) {
        suspiciousCategories.push('Madde Kullanımı');
    }
    
    // Yüksek şiddet skoru ama silah skoru düşük
    if (scores.violence > 0.8 && scores.weapon < 0.3) {
        // Eğer içerik bir haber veya TV programı tarzı ise
        if (isNewsOrTVContent(results)) {
            suspiciousCategories.push('Şiddet');
        }
    }
    
    // Eğer tüm skorlar çok yüksekse şüpheli olabilir
    if (scores.violence > 0.7 && scores.harassment > 0.7 && scores.adult_content > 0.7 && scores.weapon > 0.5) {
        suspiciousCategories.push('Genel Skorlar');
    }
    
    return suspiciousCategories;
}

// İçerik türünü tahmin etme (haber veya TV programı mı?)
function isNewsOrTVContent(results) {
    // Detay bilgilerinden tahmin etmeye çalış
    if (results.content_detections && results.content_detections.length > 0) {
        // Tespit edilen yüz sayısı
        const detectedFaces = results.age_estimations ? results.age_estimations.length : 0;
        
        // Farklı yüz sayısı
        const uniqueFaces = new Set();
        if (results.age_estimations) {
            results.age_estimations.forEach(est => {
                const faceId = est.person_id || est.face_id || '';
                if (faceId) uniqueFaces.add(faceId);
            });
        }
        
        // Eğer çok sayıda yüz ve çeşitli kişi varsa TV/haber olabilir
        if (detectedFaces > 10 && uniqueFaces.size > 2) {
            return true;
        }
    }
    
    return false;
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
    
    console.log("Bulunan en yüksek kategoriler:", highestFrames);
    
    // Her kategori için en yüksek riskli kareyi göster
    const categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug', 'safe'];
    const grid = document.getElementById('categoryFramesGrid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    categories.forEach(category => {
        // Güvenli kategori için farklı eşik değeri (en az %50)
        const threshold = category === 'safe' ? 0.5 : 0.3;
        
        if (highestScores[category] >= threshold) { 
            const frameData = highestFrames[category];
            if (!frameData || !frameData.processed_image_path) return;
            
            let categoryName = getCategoryDisplayName(category);
            const cardDiv = document.createElement('div');
            cardDiv.className = 'col-md-4 mb-4';
            
            const frameUrl = `/${frameData.processed_image_path}`;  // /api/files/ prefix'ini kaldırdık
            console.log(`${categoryName} için frame URL:`, frameUrl);
            
            // Kategori badge'inin rengini belirle
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
            
            grid.appendChild(cardDiv);
        }
    });
    
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
                frameUrl = `/${face.processed_image_path}`;
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
function displayAgeFeedback(feedbackTab, results) {
    if (!feedbackTab || !results.age_estimations || !results.age_estimations.length) {
        return;
    }

    const ageFeedbackContainer = feedbackTab.querySelector('.age-feedback-container');
    if (!ageFeedbackContainer) {
        return;
    }

    // Benzersiz yüzleri bul
    const faces = {};
    results.age_estimations.forEach(item => {
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
    
    // Her yüz için geri bildirim kartı oluştur
    Object.entries(faces).forEach(([faceId, face], index) => {
        const feedbackCard = document.createElement('div');
        feedbackCard.className = 'card mb-3';
        let frameUrl = '';
        
        // 18 yaş altı kontrolü
        const isUnderAge = face.age < 18;
        if (isUnderAge) {
            feedbackCard.classList.add('border-danger');
            feedbackCard.classList.add('bg-danger-subtle');
        }
        
        // Sadece processed_image_path varsa görsel göster
        if (face.processed_image_path) {
            frameUrl = `/${face.processed_image_path}`;
            console.log("Geri bildirim tabı - İşlenmiş görsel kullanılıyor:", frameUrl);
            feedbackCard.innerHTML = `
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-4">
                            <div class="position-relative" style="height: 200px; overflow: hidden;">
                                <img src="${frameUrl}" alt="Yüz ${index + 1}"
                                    style="width: 100%; height: 100%; object-fit: contain;"
                                    onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';">
                                <span class="position-absolute top-0 end-0 m-2 badge bg-info">Yüz #${index + 1}</span>
                                ${isUnderAge ? '<span class="position-absolute top-0 start-0 m-2 badge bg-danger"><i class="fas fa-exclamation-triangle me-1"></i> 18 yaş altı</span>' : ''}
                            </div>
                        </div>
                        <div class="col-md-8">
                            <div class="d-flex align-items-center mb-2">
                                <h6 class="mb-0 me-2">Tahmin Edilen Yaş:</h6>
                                <span class="badge bg-primary fs-5">${Math.round(face.age)}</span>
                            </div>
                            <div class="mb-3">
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <small class="text-muted">Güvenilirlik:</small>
                                    <span class="badge ${face.confidence > 0.7 ? 'bg-success' : 
                                        face.confidence > 0.4 ? 'bg-warning' : 'bg-danger'}">
                                        ${(face.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div class="progress" style="height: 6px;">
                                    <div class="progress-bar ${face.confidence > 0.7 ? 'bg-success' : 
                                        face.confidence > 0.4 ? 'bg-warning' : 'bg-danger'}"
                                        style="width: ${face.confidence * 100}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            console.log("Geri bildirim tabı - İşlenmiş görsel bulunamadı");
            feedbackCard.innerHTML = `
                <div class="card-body">
                    <div class="alert alert-warning">İşlenmiş (overlay'li) görsel bulunamadı.</div>
                </div>
            `;
        }
        ageFeedbackContainer.appendChild(feedbackCard);
    });
}

// Kişi kartı oluşturma fonksiyonu
function createPersonCard(person, personId, analysisId) {
    // Kart temel yapısı
    const card = document.createElement('div');
    card.className = 'card mb-3';
    card.id = `person-${personId}`;
    
    // Kart içeriği
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    // Kişi başlığı - Person 1, Person 2, etc. veya daha anlamlı bir isim
    const personNumber = personId.split('_').pop();
    const cardTitle = document.createElement('h5');
    cardTitle.className = 'card-title';
    cardTitle.textContent = `Kişi ${personNumber}`;
    cardBody.appendChild(cardTitle);
    
    // İçerik satırı
    const cardContent = document.createElement('div');
    cardContent.className = 'row align-items-center';
    
    // Yüz görüntüsü (sol taraf)
    const faceCol = document.createElement('div');
    faceCol.className = 'col-md-4';
    
    // Görüntü konteynerı
    const faceImgContainer = document.createElement('div');
    faceImgContainer.className = 'position-relative mb-2';
    
    if (person.bestFrame) {
        // Yüz görseli
        const faceImg = document.createElement('img');
        const frameUrl = getFrameUrl(person.bestFrame.split('/').pop(), analysisId, null);
        faceImg.src = frameUrl;
        faceImg.className = 'img-fluid rounded face-image';
        faceImg.alt = `Kişi ${personNumber} yüz`;
        faceImg.style.maxHeight = '150px';
        faceImg.style.maxWidth = '100%';
        
        // Görsel yüklenemezse
        faceImg.onerror = function() {
            faceImgContainer.innerHTML = '<div class="text-center py-4 bg-light rounded">Görsel Bulunamadı</div>';
        };
        
        faceImgContainer.appendChild(faceImg);
        
        // Zaman bilgisi varsa ekle
        if (person.bestTimestamp) {
            const timeLabel = document.createElement('span');
            timeLabel.className = 'position-absolute bottom-0 start-0 badge bg-dark';
            timeLabel.textContent = formatTime(person.bestTimestamp);
            faceImgContainer.appendChild(timeLabel);
        }
    } else {
        faceImgContainer.innerHTML = '<div class="text-center py-4 bg-light rounded">Görsel Yok</div>';
    }
    
    faceCol.appendChild(faceImgContainer);
    
    // Bilgi metni
    const infoText = document.createElement('div');
    infoText.className = 'text-center small text-muted';
    infoText.textContent = `${person.appearances} karede görünüyor`;
    faceCol.appendChild(infoText);
    
    cardContent.appendChild(faceCol);
    
    // Yaş bilgileri (sağ taraf)
    const ageCol = document.createElement('div');
    ageCol.className = 'col-md-8';
    
    // Yaş bilgisi kartı
    const ageCard = document.createElement('div');
    ageCard.className = 'card h-100';
    
    const ageCardBody = document.createElement('div');
    ageCardBody.className = 'card-body';
    
    // Yaş değeri
    const ageValue = document.createElement('h2');
    ageValue.className = 'display-4 text-center mb-0';
    ageValue.textContent = Math.round(person.bestEstimation);
    ageCardBody.appendChild(ageValue);
    
    // Yaş etiketi
    const ageLabel = document.createElement('p');
    ageLabel.className = 'text-center text-muted mb-2';
    ageLabel.textContent = 'Yaş';
    ageCardBody.appendChild(ageLabel);
    
    // Güvenilirlik skoru
    if (person.bestConfidence) {
        const confidenceWrapper = document.createElement('div');
        confidenceWrapper.className = 'mt-3';
        
        const confidenceLabel = document.createElement('label');
        confidenceLabel.className = 'form-label d-flex justify-content-between mb-1';
        confidenceLabel.innerHTML = `<span>Güvenilirlik:</span><span>${Math.round(person.bestConfidence * 100)}%</span>`;
        confidenceWrapper.appendChild(confidenceLabel);
        
        const confidenceBar = document.createElement('div');
        confidenceBar.className = 'progress';
        confidenceBar.style.height = '8px';
        
        const confidenceFill = document.createElement('div');
        confidenceFill.className = 'progress-bar bg-success';
        confidenceFill.style.width = `${person.bestConfidence * 100}%`;
        confidenceFill.setAttribute('role', 'progressbar');
        confidenceFill.setAttribute('aria-valuenow', person.bestConfidence * 100);
        confidenceFill.setAttribute('aria-valuemin', '0');
        confidenceFill.setAttribute('aria-valuemax', '100');
        
        confidenceBar.appendChild(confidenceFill);
        confidenceWrapper.appendChild(confidenceBar);
        
        ageCardBody.appendChild(confidenceWrapper);
    }
    
    // Geri bildirim formu
    if (enableFeedback) {
        const feedbackForm = document.createElement('div');
        feedbackForm.className = 'mt-3 pt-3 border-top';
        
        const feedbackTitle = document.createElement('div');
        feedbackTitle.className = 'form-text text-muted mb-2';
        feedbackTitle.innerHTML = '<i class="fas fa-comment-alt me-1"></i> Geri Bildirim:';
        feedbackForm.appendChild(feedbackTitle);
        
        // Yas skoru geri bildirim butonları
        const feedbackButtons = document.createElement('div');
        feedbackButtons.className = 'd-flex age-feedback';
        
        // Doğru ve yanlış butonları
        const correctBtn = document.createElement('button');
        correctBtn.type = 'button';
        correctBtn.className = 'btn btn-sm btn-outline-success me-2 feedback-correct';
        correctBtn.innerHTML = '<i class="fas fa-check"></i> Doğru';
        correctBtn.onclick = function() {
            document.querySelectorAll(`#person-${personId} .age-feedback-input`).forEach(el => el.classList.add('d-none'));
            this.classList.remove('btn-outline-success');
            this.classList.add('btn-success');
            document.querySelector(`#person-${personId} .feedback-wrong`).classList.remove('btn-danger');
            document.querySelector(`#person-${personId} .feedback-wrong`).classList.add('btn-outline-danger');
        };
        feedbackButtons.appendChild(correctBtn);
        
        const wrongBtn = document.createElement('button');
        wrongBtn.type = 'button';
        wrongBtn.className = 'btn btn-sm btn-outline-danger feedback-wrong';
        wrongBtn.innerHTML = '<i class="fas fa-times"></i> Yanlış';
        wrongBtn.onclick = function() {
            document.querySelectorAll(`#person-${personId} .age-feedback-input`).forEach(el => el.classList.remove('d-none'));
            this.classList.remove('btn-outline-danger');
            this.classList.add('btn-danger');
            document.querySelector(`#person-${personId} .feedback-correct`).classList.remove('btn-success');
            document.querySelector(`#person-${personId} .feedback-correct`).classList.add('btn-outline-success');
        };
        feedbackButtons.appendChild(wrongBtn);
        
        feedbackForm.appendChild(feedbackButtons);
        
        // Doğru yaş giriş formu (başlangıçta gizli)
        const ageInputGroup = document.createElement('div');
        ageInputGroup.className = 'input-group mt-2 age-feedback-input d-none';
        
        const ageInputLabel = document.createElement('span');
        ageInputLabel.className = 'input-group-text';
        ageInputLabel.textContent = 'Doğru yaş:';
        ageInputGroup.appendChild(ageInputLabel);
        
        const ageInput = document.createElement('input');
        ageInput.type = 'number';
        ageInput.className = 'form-control';
        ageInput.min = '1';
        ageInput.max = '100';
        ageInput.value = Math.round(person.bestEstimation);
        ageInputGroup.appendChild(ageInput);
        
        const ageSubmitBtn = document.createElement('button');
        ageSubmitBtn.type = 'button';
        ageSubmitBtn.className = 'btn btn-primary';
        ageSubmitBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        ageSubmitBtn.onclick = function() {
            submitAgeFeedback(personId, ageInput.value);
            this.disabled = true;
            this.innerHTML = '<i class="fas fa-check"></i>';
        };
        ageInputGroup.appendChild(ageSubmitBtn);
        
        feedbackForm.appendChild(ageInputGroup);
        
        ageCardBody.appendChild(feedbackForm);
    }
    
    ageCard.appendChild(ageCardBody);
    ageCol.appendChild(ageCard);
    
    cardContent.appendChild(ageCol);
    cardBody.appendChild(cardContent);
    
    card.appendChild(cardBody);
    
    return card;
}

// ... (rest of the code remains unchanged)
