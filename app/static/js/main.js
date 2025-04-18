// Global değişkenler
let uploadedFiles = [];
let analysisInProgress = false;
let socket;

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
    // Socket.io bağlantısı
    initializeSocket();
    
    // Event Listeners
    initializeEventListeners();
});

// Socket.io bağlantısını başlat
function initializeSocket() {
    socket = io();
    
    // Socket.io olayları dinle
    socket.on('connect', () => {
        console.log('WebSocket bağlantısı kuruldu.');
    });
    
    socket.on('disconnect', () => {
        console.log('WebSocket bağlantısı kesildi.');
    });
    
    // Analiz başladı
    socket.on('analysis_started', (data) => {
        console.log('Analiz başladı:', data);
        const { analysis_id, file_id, file_name, file_type } = data;
        
        // Dosyanın durumunu güncelle
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            // Analiz ID'sini kaydet
            file.analysisId = analysis_id;
            fileAnalysisMap.set(file.id, analysis_id);
            
            updateFileStatus(file.id, 'Analiz Başlatıldı', 10);
            console.log(`Analiz başlatıldı: ${file_name} (${file_type}), ID: ${analysis_id}`);
            showToast('Bilgi', `${file_name} analizi başlatıldı.`, 'info');
            
            // Analiz durumunu kontrol etmeye başla
            setTimeout(() => checkAnalysisStatus(analysis_id, file.id), 1000);
        }
    });
    
    // Analiz ilerleme durumu
    socket.on('analysis_progress', (data) => {
        console.log('Analiz ilerliyor:', data);
        const { analysis_id, file_id, current_frame, total_frames, progress, detected_faces, high_risk_frames } = data;
        
        // İlerleme bilgisini ekranda göster
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            const status = `Analiz: ${current_frame}/${total_frames} kare`;
            updateFileStatus(file.id, status, progress);
            
            // İlerleme detaylarını tooltip'te göster
            const fileCard = document.getElementById(file.id);
            const statusElement = fileCard.querySelector('.file-status-text');
            statusElement.title = `İşlenen kare: ${current_frame}/${total_frames}
Tespit edilen yüz: ${detected_faces || 0}
Yüksek riskli kare: ${high_risk_frames || 0}
İlerleme: %${progress.toFixed(1)}`;
            
            // Her 10 karede bir mesaj göster
            if (current_frame % 10 === 0 || current_frame === total_frames) {
                console.log(`Analiz ilerliyor: ${file.name}, Kare: ${current_frame}/${total_frames}, İlerleme: %${progress.toFixed(1)}`);
            }
        }
    });
    
    // Analiz tamamlandı
    socket.on('analysis_completed', (data) => {
        console.log('Analiz tamamlandı:', data);
        const { analysis_id, file_id, elapsed_time, message } = data;
        
        // Dosya durumunu güncelle
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            updateFileStatus(file.id, 'completed', 100);
            console.log(`Analiz tamamlandı: ${file.name}, Süre: ${elapsed_time ? elapsed_time.toFixed(1) + 's' : 'bilinmiyor'}`);
            
            // Analiz ID'sini file.analysisId olarak kaydet
            file.analysisId = analysis_id;
            fileAnalysisMap.set(file.id, analysis_id);
            
            // Analiz sonuçlarını getir ve göster
            getAnalysisResults(file.id, analysis_id);
            
            // Tamamlandı bildirimi göster
            showToast('Başarılı', `${file.name} analizi tamamlandı (${elapsed_time ? elapsed_time.toFixed(1) + ' saniye' : 'bilinmiyor'}).`, 'success');
            
            // Genel ilerlemeyi güncelle
            updateGlobalProgress();
        }
    });
    
    // Analiz hatası
    socket.on('analysis_failed', (data) => {
        console.error('Analiz hatası:', data);
        const { analysis_id, file_id, error, elapsed_time } = data;
        
        // Dosya durumunu güncelle
        const file = uploadedFiles.find(f => f.fileId == file_id);
        if (file) {
            updateFileStatus(file.id, 'failed', 0);
            console.error(`Analiz hatası: ${file.name}, Süre: ${elapsed_time ? elapsed_time.toFixed(1) + 's' : 'bilinmiyor'}, Hata: ${error}`);
            showToast('Hata', `Analiz hatası: ${error}`, 'danger');
        }
    });
    
    // Model eğitim durumu
    socket.on('training_progress', (data) => {
        updateTrainingProgress(data);
    });
    
    // Model eğitimi tamamlandı
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
            // Analiz parametreleri modalını aç
            const modal = new bootstrap.Modal(document.getElementById('analysisParamsModal'));
            modal.show();
        }
    });
    
    // Analiz Başlatma Onay Butonu
    document.getElementById('startAnalysisBtn').addEventListener('click', () => {
        // Analiz parametrelerini al
        const framesPerSecond = parseFloat(document.getElementById('framesPerSecond').value);
        const includeAgeAnalysis = document.getElementById('includeAgeAnalysis').checked;
        
        // Modalı kapat
        const modal = bootstrap.Modal.getInstance(document.getElementById('analysisParamsModal'));
        modal.hide();
        
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
        
        // Analiz ID'sini kaydet - response.analysis_id yerine response.analysis.id kullan
        const analysisId = response.analysis ? response.analysis.id : null;
        
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
        updateFileStatus(fileId, status, progress);
        
        // Genel ilerlemeyi güncelle
        updateGlobalProgress();
        
        if (status === "completed") {
            // Analiz tamamlandıysa sonuçları göster
            getAnalysisResults(fileId, analysisId);
        } else if (status === "failed") {
            // Analiz başarısız olduysa hata mesajı göster
            showError(`${fileNameFromId(fileId)} dosyası için analiz başarısız oldu: ${response.error || "Bilinmeyen hata"}`);
        } else if (status === "processing" || status === "queued") {
            // Analiz devam ediyorsa durumu kontrol etmeye devam et
            setTimeout(() => checkAnalysisStatus(analysisId, fileId), 2000);
        }
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
    
    // Dosya adını ayarla
    resultCard.querySelector('.result-filename').textContent = file.name;
    
    // Content ID'sini gizli alana ekle
    const contentIdInput = resultCard.querySelector('.content-id');
    if (contentIdInput) {
        contentIdInput.value = results.content_id || '';
    }
    
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
        
        const scores = results.overall_scores;
        
        for (const [category, score] of Object.entries(scores)) {
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
            }
            
            // Risk seviyesi
            let riskLevel = '';
            let riskClass = '';
            
            if (score >= 0.7) {
                riskLevel = 'Yüksek Risk';
                riskClass = 'risk-level-high';
            } else if (score >= 0.3) {
                riskLevel = 'Orta Risk';
                riskClass = 'risk-level-medium';
            } else {
                riskLevel = 'Düşük Risk';
                riskClass = 'risk-level-low';
            }
            
            // Skor elementi HTML'i
            scoreElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span>${categoryName}</span>
                    <span class="risk-score ${riskClass}">${(score * 100).toFixed(0)}% - ${riskLevel}</span>
                </div>
                <div class="progress">
                    <div class="progress-bar ${riskClass === 'risk-level-high' ? 'bg-danger' : riskClass === 'risk-level-medium' ? 'bg-warning' : 'bg-success'}" 
                         role="progressbar" style="width: ${score * 100}%" 
                         aria-valuenow="${score * 100}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
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
                    imageSource = `/api/files/frames/${analysisId}/${frameFilename}`;
                }
                
                console.log(`Yüksek riskli kare URL'si: ${imageSource}`);
                
                // İmage error handling ekle
                highestRiskFrame.onerror = function() {
                    console.error("Görsel yüklenemedi:", imageSource);
                    // Alternatif kaynak dene 
                    const alternativeUrl = `/api/files/processed/${results.highest_risk.frame || ''}`;
                    console.log(`Alternatif URL deneniyor: ${alternativeUrl}`);
                    
                    // Alternatif URL'yi bir kere deneyelim
                    this.onerror = function() {
                        // İkinci hata durumunda placeholder göster
                        console.error("Alternatif görsel de yüklenemedi");
                        this.src = '/static/img/image-not-found.svg';
                        this.onerror = null; // Sonsuz döngüyü önle
                    };
                    
                    this.src = alternativeUrl;
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
                    highestRiskScore.textContent = `Skor: ${(results.highest_risk.score * 100).toFixed(0)}%`;
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
    const detailsTab = resultCard.querySelector('#details-' + uniqueSuffix);
    if (detailsTab) {
        // Detaylar sayfasını temizleyelim
        detailsTab.innerHTML = '';
        
        // İçerik tespitleri
        const contentDetections = results.content_detections || [];
        
        if (contentDetections.length > 0) {
            try {
                // Her kategori için en yüksek skorlu tespitleri bul
                const categoryTopDetections = {
                    'violence': null,
                    'adult_content': null,
                    'harassment': null,
                    'weapon': null,
                    'drug': null
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
                        'drug': detection.drug_score
                    };
                    
                    console.log('Tespit edilen skorlar:', categoryScores);
                    
                    // Her kategori için skoru kontrol et
                    for (const [category, score] of Object.entries(categoryScores)) {
                        if (score && !isNaN(score)) {
                            if (!categoryTopDetections[category] || score > categoryTopDetections[category].score) {
                                console.log(`Daha yüksek ${category} skoru bulundu:`, score);
                                categoryTopDetections[category] = {
                                    score: score,
                                    frame_path: detection.frame_path,
                                    timestamp: detection.frame_timestamp // frame_timestamp alanını kullan
                                };
                            }
                        }
                    }
                });
                
                console.log('Bulunan en yüksek kategoriler:', categoryTopDetections);
                
                // İçerik tespitleri bölümü
                const contentDetectionsSection = document.createElement('div');
                contentDetectionsSection.classList.add('content-detections', 'mb-4');
                contentDetectionsSection.innerHTML = `
                    <h5 class="mb-3"><i class="fas fa-exclamation-triangle me-2"></i>Kategori Bazında En Yüksek Riskli Kareler</h5>
                    <div class="row" id="categoryTopDetectionsList-${uniqueSuffix}"></div>
                `;
                detailsTab.appendChild(contentDetectionsSection);
                
                const categoryDetectionsList = contentDetectionsSection.querySelector(`#categoryTopDetectionsList-${uniqueSuffix}`);
                
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
                            badgeClass = (detection.score >= 0.7) ? 'bg-danger' : (detection.score >= 0.3) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'adult_content': 
                            categoryName = 'Yetişkin İçeriği'; 
                            badgeClass = (detection.score >= 0.7) ? 'bg-danger' : (detection.score >= 0.3) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'harassment': 
                            categoryName = 'Taciz'; 
                            badgeClass = (detection.score >= 0.7) ? 'bg-danger' : (detection.score >= 0.3) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'weapon': 
                            categoryName = 'Silah'; 
                            badgeClass = (detection.score >= 0.7) ? 'bg-danger' : (detection.score >= 0.3) ? 'bg-warning' : 'bg-success';
                            break;
                        case 'drug': 
                            categoryName = 'Madde Kullanımı'; 
                            badgeClass = (detection.score >= 0.7) ? 'bg-danger' : (detection.score >= 0.3) ? 'bg-warning' : 'bg-success';
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
                            const frameName = normalizePath(detection.frame_path).split('/').pop();
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
                                    <span>Risk Skoru:</span>
                                    <strong>${(detection.score * 100).toFixed(0)}%</strong>
                                </div>
                                <div class="progress">
                                    <div class="progress-bar ${badgeClass}" style="width: ${detection.score * 100}%" 
                                         role="progressbar" aria-valuenow="${detection.score * 100}" 
                                         aria-valuemin="0" aria-valuemax="100"></div>
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
            // Her yüz için en güvenilir skorları bul
            const faceConfidenceMap = new Map();
            
            // Backend'in döndüğü veri yapısına göre uygun değişkeni seç
            const ageData = results.age_estimations || results.age_analysis || [];
            
            console.log('Yaş tahmini işlenen veriler:', ageData.length, 'kayıt bulundu');
            
            ageData.forEach(analysis => {
                // Her analiz için yüz ID'si ve güvenilirlik skorunu al
                const faceId = analysis.person_id || analysis.face_id || 'unknown';
                const confidence = analysis.confidence_score || analysis.confidence || 0;
                const age = analysis.estimated_age || 'Bilinmiyor';
                const timestamp = analysis.frame_timestamp || analysis.timestamp || null;
                const frame_path = analysis.frame_path || null;
                
                // Eğer bu yüz daha önce görülmediyse veya bu analiz daha güvenilirse güncelle
                if (!faceConfidenceMap.has(faceId) || confidence > faceConfidenceMap.get(faceId).confidence) {
                    faceConfidenceMap.set(faceId, {
                        confidence,
                        age,
                        timestamp,
                        frame_path
                    });
                }
            });
            
            // Yaş tahminleri bölümü
            const ageEstimationSection = document.createElement('div');
            ageEstimationSection.classList.add('age-estimations', 'mt-4');
            ageEstimationSection.innerHTML = `
                <h5 class="mb-3"><i class="fas fa-user-alt me-2"></i>Yaş Tahminleri</h5>
                <div class="row" id="ageEstimationList-${uniqueSuffix}"></div>
            `;
            detailsTab.appendChild(ageEstimationSection);
            
            const ageEstimationList = ageEstimationSection.querySelector(`#ageEstimationList-${uniqueSuffix}`);
            
            // Tespit edilen toplam yüz sayısı
            const person_count = faceConfidenceMap.size;
            console.log(`Tespit edilen toplam benzersiz yüz sayısı: ${person_count}`);
            
            // Her yüz için en güvenilir tahmini göster
            let faceCount = 0;
            for (const [faceId, data] of faceConfidenceMap.entries()) {
                faceCount++;
                
                // Karşılık gelen görseli yükle
                let frameUrl = '';
                if (data.frame_path) {
                    // Resim dosyası mı yoksa video karesi mi?
                    if (file.type && file.type.startsWith('image/')) {
                        // Eğer dosya bir görsel ise, direkt dosyayı kullan
                        const fileId = file.fileId || '';
                        frameUrl = `/api/files/${fileId}/download`;
                    } else {
                        // Video kareleri için
                        const frameName = normalizePath(data.frame_path).split('/').pop();
                        frameUrl = `/api/files/frames/${results.analysis_id}/${frameName}`;
                    }
                }
                
                // Zaman bilgisini formatla
                let timeText = '';
                if (data.timestamp) {
                    const minutes = Math.floor(data.timestamp / 60);
                    const seconds = Math.floor(data.timestamp % 60);
                    timeText = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                }
                
                // Güvenilirlik seviyesine göre renk seç
                let confidenceClass = 'bg-success';
                if (data.confidence < 0.6) {
                    confidenceClass = 'bg-warning';
                } else if (data.confidence < 0.4) {
                    confidenceClass = 'bg-danger';
                }
                
                const faceCard = document.createElement('div');
                faceCard.classList.add('col-md-4', 'mb-3');
                
                // Yaş değerini düzgün formata çevirelim
                let displayAge = data.age;
                
                // Eğer yaş çok küçük bir değerse (0.5'den küçük) ve "Bilinmiyor" değilse, 
                // muhtemelen farklı bir birimde (0-1 arası normalize edilmiş) olabilir
                if (typeof displayAge === 'number' && displayAge < 0.5 && displayAge > 0) {
                    // 0-1 aralığını 0-100 yaş aralığına dönüştür
                    displayAge = Math.round(displayAge * 100);
                    console.log(`Düşük yaş değeri (${data.age}) tespit edildi, ${displayAge} yaşına dönüştürüldü`);
                } else if (typeof displayAge === 'number') {
                    // Sayısal değeri yuvarla
                    displayAge = Math.round(displayAge);
                }
                
                // Yüz konumu bilgisi varsa
                let faceLocationHtml = '';
                if (data.face_location) {
                    const faceLocation = data.face_location;
                    // Kare üzerinde yüzü dikdörtgen ile işaretle
                    faceLocationHtml = `
                        <div class="face-highlight" style="
                            position: absolute;
                            top: ${faceLocation[1] / 4}px;
                            left: ${faceLocation[0] / 4}px;
                            width: ${faceLocation[2] / 4}px;
                            height: ${faceLocation[3] / 4}px;
                            border: 3px solid #00e676;
                            border-radius: 4px;
                            z-index: 10;
                            box-shadow: 0 0 5px rgba(0,0,0,0.5);
                        "></div>
                    `;
                }

                // Birden fazla kişi olduğunda hangi yüzün analiz edildiğini belirtmek için
                const faceIndicatorHTML = `
                    <div class="alert alert-info py-1 mb-2">
                        <small><i class="fas fa-info-circle me-1"></i> 
                        Bu karede ${person_count > 1 ? `<b>${person_count} kişi</b> tespit edildi. Yüz <b>#${faceCount}</b> görüntüleniyor.` : '1 kişi tespit edildi.'}
                        </small>
                    </div>
                `;

                faceCard.innerHTML = `
                    <div class="card h-100">
                        <div class="position-relative">
                            <div style="height: 240px; overflow: hidden;">
                                <img src="${frameUrl}" class="card-img-top face-img" alt="Yüz ${faceCount}" 
                                    style="width: 100%; height: 100%; object-fit: cover;"
                                    onerror="this.onerror=null;this.src='/static/img/image-not-found.svg';">
                            </div>
                            <span class="position-absolute top-0 end-0 m-2 badge bg-info">Yüz ${faceCount}</span>
                            ${timeText ? `<span class="position-absolute bottom-0 start-0 m-2 badge bg-dark">${timeText}</span>` : ''}
                        </div>
                        <div class="card-body">
                            ${faceIndicatorHTML}
                            <h6 class="card-title">Yaş Tahmini: <strong>${displayAge}</strong></h6>
                            <div class="d-flex justify-content-between mb-1">
                                <span>Güvenilirlik:</span>
                                <strong>${(data.confidence * 100).toFixed(0)}%</strong>
                            </div>
                            <div class="progress">
                                <div class="progress-bar ${confidenceClass}" style="width: ${data.confidence * 100}%" 
                                    role="progressbar" aria-valuenow="${data.confidence * 100}" 
                                    aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                        </div>
                    </div>
                `;
                
                ageEstimationList.appendChild(faceCard);
            }
            
            // Eğer yüz tespiti yapılmadıysa bilgi mesajı göster
            if (faceCount === 0) {
                ageEstimationList.innerHTML = '<div class="col-12"><div class="alert alert-info">Bu dosyada tespit edilen yüz bulunmuyor.</div></div>';
            }
        } catch (error) {
            console.error("Yaş tahminleri gösterilirken hata:", error);
            detailsTab.innerHTML += `<div class="alert alert-danger mb-4">Yaş tahminleri işlenirken hata oluştu: ${error.message}</div>`;
        }
    } else if (results.include_age_analysis) {
        detailsTab.innerHTML += '<div class="alert alert-info mt-3">Bu dosya için yaş tahmini bulunmuyor.</div>';
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
        showToast('Başarılı', 'Geri bildiriminiz için teşekkürler.', 'success');
        
        // Geri bildirim butonunu pasif yap
        form.querySelector('.submit-feedback-btn').disabled = true;
        form.querySelector('.submit-feedback-btn').textContent = 'Gönderildi';
    })
    .catch(error => {
        console.error('Geri bildirim gönderme hatası:', error);
        showToast('Hata', `Geri bildirim gönderilirken hata oluştu: ${error.message}`, 'danger');
    });
}

// Yaş geri bildirimi gönder
function submitAgeFeedback(input) {
    const personId = input.dataset.personId;
    const correctedAge = parseInt(input.value);
    
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
            corrected_age: correctedAge
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
        showToast('Başarılı', 'Yaş geri bildirimi kaydedildi.', 'success');
        
        // Geri bildirim butonunu pasif yap
        const button = input.nextElementSibling;
        button.disabled = true;
        button.textContent = 'Kaydedildi';
    })
    .catch(error => {
        console.error('Yaş geri bildirimi gönderme hatası:', error);
        showToast('Hata', `Yaş geri bildirimi gönderilirken hata oluştu: ${error.message}`, 'danger');
    });
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
    if (!confirm(`${modelType === 'content' ? 'İçerik analiz' : 'Yaş tahmin'} modelini sıfırlamak istediğinizden emin misiniz? Bu işlem geri alınamaz.`)) {
        return;
    }
    
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