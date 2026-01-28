/* ERSIN Aciklama. */

import { 
    uploadedFiles, 
    fileStatuses, 
    fileAnalysisMap, 
    API_URL,
    formatFileSize,
    showToast,
    showError
} from './globals.js';

// ERSIN =====================================
// ERSIN FILE OPERATIONS
// ERSIN =====================================

/* ERSIN Aciklama. */
export function handleFileSelection(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

/* ERSIN Aciklama. */
export function handleFiles(files) {
    const fileArray = Array.from(files);
    
    // ERSIN Dosya türü kontrolü
    const validFiles = [];
    for (const file of fileArray) {
        if (isValidFileType(file)) {
            validFiles.push(file);
        } else {
            showError(`Desteklenmeyen dosya türü: ${file.name}`);
        }
    }
    
    if (validFiles.length === 0) {
        return;
    }
    
    // ERSIN Dosyaları listeye ekle
    validFiles.forEach(file => addFileToList(file));
    
    // ERSIN Upload işlemini başlat
    uploadFilesSequentially(0);
}

/* ERSIN Aciklama. */
function isValidFileType(file) {
    const validTypes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp',
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/flv',
        'video/webm', 'video/mkv'
    ];
    return validTypes.includes(file.type);
}

/* ERSIN Aciklama. */
function addFileToList(file) {
    // ERSIN Unique ID oluştur
    const fileId = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const fileInfo = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
        serverFileId: null,  // ERSIN Server'dan gelecek
        status: 'pending'
    };
    
    uploadedFiles.push(fileInfo);
    
    // ERSIN 🎯 Overall progress bar'ı güncelle (yeni dosya eklendi)
    if (typeof window.updateOverallProgress === 'function') {
        window.updateOverallProgress({ queue_size: 0, is_processing: false });
    }
    
    // ERSIN UI'da file card oluştur
    createFileCard(fileInfo);
    
    // ERSIN File status'ını map'e ekle
    fileStatuses.set(fileId, 'pending');
    
    console.log(`Dosya listeye eklendi: ${file.name} (ID: ${fileId})`);
}

/* ERSIN Aciklama. */
function createFileCard(fileInfo) {
    const fileList = document.getElementById('fileList');
    if (!fileList) return;
    
    // ERSIN File list section'ı görünür yap (ilk dosya eklendiğinde)
    const fileListSection = document.getElementById('fileListSection');
    if (fileListSection && fileListSection.style.display === 'none') {
        fileListSection.style.display = 'block';
        console.log('📁 File list section görünür hale getirildi');
    }
    
    // ERSIN 🎨 ORİJİNAL DESIGN: HTML template'dan güzel tasarımı kullan
    const fileCard = document.createElement('div');
    fileCard.className = 'col-12 mb-3';
    fileCard.id = fileInfo.id;
    fileCard.setAttribute('data-file-id', fileInfo.id);
    
    fileCard.innerHTML = `
        <div class="file-card file-list-layout">
            <div class="file-preview-area">
                ${createFilePreviewHTML(fileInfo)}
                <div class="file-status bg-secondary">Sırada</div>
            </div>
            <div class="file-details-area">
                <div class="file-status-area">
                    <div>
                        <div class="filename fw-bold" title="${fileInfo.name}">${fileInfo.name}</div>
                        <small class="filesize text-muted">Boyut: ${formatFileSize(fileInfo.size)}</small>
                    </div>
                    <button class="btn btn-sm btn-danger remove-file-btn" onclick="window.removeFile('${fileInfo.id}')">
                        <i class="fas fa-times"></i> Kaldır
                    </button>
                </div>
                <div class="file-status-text mb-2">Sırada</div>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </div>
        </div>
    `;
    
    fileList.appendChild(fileCard);
    
    // ERSIN 🎬 VIDEO DOSYASI İÇİN GERÇEK THUMBNAIL OLUŞTUR
    if (fileInfo.type.startsWith('video/')) {
        const previewImg = fileCard.querySelector('.file-preview');
        if (previewImg) {
            createVideoThumbnail(fileInfo, previewImg);
        }
    }
    
    // ERSIN 🔘 FILE YÜKLENINCE ANALYZE BUTTON'I ENABLE ET
    enableAnalyzeButton();
}

/* ERSIN Aciklama. */
function enableAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn && uploadedFiles.length > 0) {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('disabled');
        console.log('🔘 Analyze button enabled - dosya sayısı:', uploadedFiles.length);
    }
}

/* ERSIN Aciklama. */
function createFilePreviewHTML(fileInfo) {
    if (fileInfo.type.startsWith('image/')) {
        const imageUrl = URL.createObjectURL(fileInfo.file);
        return `<img class="file-preview" src="${imageUrl}" alt="${fileInfo.name}">`;
    } else if (fileInfo.type.startsWith('video/')) {
        // ERSIN Video için boş img tag oluştur, JavaScript ile thumbnail ayarlanacak
        return `<img class="file-preview" src="" alt="Video önizlemesi" data-file-id="${fileInfo.id}">`;
    } else {
        return `<img class="file-preview" src="/static/img/placeholder-face.png" alt="Dosya önizlemesi">`;
    }
}

/* ERSIN Aciklama. */
function createVideoThumbnail(fileInfo, previewElement) {
    console.log(`🎬 Video thumbnail oluşturuluyor: ${fileInfo.name}`);
    
    const fileURL = URL.createObjectURL(fileInfo.file);
    const video = document.createElement('video');
    video.src = fileURL;
    
    video.onloadeddata = () => {
        // ERSIN Video yüklendikten sonra ilk kareyi al
        video.currentTime = 0.1;
    };
    
    video.onseeked = () => {
        // ERSIN Canvas oluştur ve ilk kareyi çiz
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 320;
        canvas.height = video.videoHeight || 240;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // ERSIN Canvas'taki resmi önizleme olarak ayarla
        previewElement.src = canvas.toDataURL();
        console.log(`✅ Video thumbnail oluşturuldu: ${fileInfo.name}`);
        
        // ERSIN Belleği temizle
        URL.revokeObjectURL(fileURL);
    };
    
    // ERSIN Hata durumunda veya zaman aşımında blob URL'i temizle
    video.onerror = () => {
        console.log(`❌ Video thumbnail oluşturulamadı: ${fileInfo.name}`);
        // ERSIN Fallback placeholder kullan
        previewElement.src = '/static/img/placeholder-face.png';
        URL.revokeObjectURL(fileURL);
    };
    
    // ERSIN 5 saniye sonra hala işlenmemişse URL'i temizle (zaman aşımı güvenlik önlemi)
    setTimeout(() => {
        if (video.src) {
            URL.revokeObjectURL(fileURL);
        }
    }, 5000);
}

/* ERSIN Aciklama. */
export function uploadFilesSequentially(index) {
    if (index >= uploadedFiles.length) {
        console.log('Tüm dosyalar yüklendi');
        return;
    }
    
    const fileInfo = uploadedFiles[index];
    if (fileInfo.status !== 'pending') {
        // ERSIN Bu dosya zaten yüklenmiş, bir sonrakine geç
        uploadFilesSequentially(index + 1);
        return;
    }
    
    // ERSIN Dosya yükleme durumunu güncelle
    updateFileStatus(fileInfo.id, 'Yükleniyor', 0);
    
    const formData = new FormData();
    formData.append('file', fileInfo.file);
    
    fetch(`${API_URL}/files/`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.file_id) {
            // ERSIN Server file ID'yi kaydet
            fileInfo.serverFileId = data.file_id;
            fileInfo.status = 'uploaded';
            
            // ERSIN File status'ını güncelle
            updateFileStatus(fileInfo.id, 'uploaded', 100);
            
            console.log(`Dosya yüklendi: ${fileInfo.name}, ID: ${data.file_id}`);
            
            // ERSIN ID mapping'i oluştur
            createFileIdMapping(data.file_id, fileInfo.id);
            
            // ERSIN Bir sonraki dosyayı yükle
            uploadFilesSequentially(index + 1);
        } else {
            throw new Error(data.error || 'Dosya yüklenemedi');
        }
    })
    .catch(error => {
        console.error(`Dosya yükleme hatası (${fileInfo.name}):`, error);
        updateFileStatus(fileInfo.id, 'Hata', 0, null, error.message);
        showError(`Dosya yükleme hatası: ${fileInfo.name} - ${error.message}`);
        
        // ERSIN Hata olsa bile diğer dosyalara devam et
        uploadFilesSequentially(index + 1);
    });
}

/* ERSIN Aciklama. */
function createFileIdMapping(serverFileId, clientFileId) {
    // ERSIN Global mapping objesi
    if (!window.fileIdToCardId) {
        window.fileIdToCardId = {};
    }
    window.fileIdToCardId[serverFileId] = clientFileId;
    
    console.log(`[DEBUG] fileIdToCardId mapping oluşturuldu: ${serverFileId} → ${clientFileId}`);
}

/* ERSIN Aciklama. */
export function updateFileStatus(fileId, status, progress, message = null, error = null) {
    console.log(`[DEBUG] updateFileStatus çağrıldı: ${fileId} ${status} ${progress}`);
    
    const fileCard = document.getElementById(fileId);
    if (!fileCard) {
        // ERSIN Sadece analiz devam ederken veya hata durumunda warning göster
        // ERSIN Eğer dosya zaten temizlendiyse (completed/error status), sessizce çık
        if (status === 'processing' || status === 'queued' || status === 'failed') {
            console.warn(`[DEBUG] fileCard bulunamadı (${status}): ${fileId}`);
        }
        return;
    }
    
    console.log(`[DEBUG] fileCard bulundu mu? ${!!fileCard}`);
    
    // ERSIN 🎨 YENİ TEMPLATE STRUCTURE - eski design ile uyumlu selectors
    const statusDiv = fileCard.querySelector('.file-status');  // ERSIN status badge area
    const statusText = fileCard.querySelector('.file-status-text');  // ERSIN status text
    const progressBar = fileCard.querySelector('.progress-bar');
    
    if (statusText) {
        let nextStatusText = message || getStatusMessage(status);
        if (typeof progress === 'number' && status === 'processing' && nextStatusText && !nextStatusText.includes('%')) {
            nextStatusText = `${nextStatusText} (%${Math.max(0, Math.min(100, progress))})`;
        }
        statusText.textContent = nextStatusText;
        console.log(`[DEBUG] statusText güncellendi: ${statusText.textContent}`);
    }
    
    if (statusDiv) {
        statusDiv.textContent = getStatusDisplayName(status);
        statusDiv.className = `file-status ${getStatusBadgeClass(status)}`;
        console.log(`[DEBUG] statusBadge güncellendi: ${statusDiv.textContent}`);
    }
    
    if (progressBar) {
        const safeProgress = Math.max(0, Math.min(100, progress));
        const oldWidth = progressBar.style.width;
        
        // ERSIN 🎨 PROGRESS BAR VISUAL UPDATE (yedek main.js'teki logic)
        progressBar.style.width = `${safeProgress}%`;
        progressBar.setAttribute('aria-valuenow', safeProgress);
        
        // ERSIN Progress bar text content (önemli!)
        progressBar.textContent = safeProgress > 0 ? `${safeProgress}%` : '';
        
        // ERSIN CSS classes for animation (processing status için)
        if (status === 'processing') {
            progressBar.classList.remove('bg-success', 'bg-danger');
            progressBar.classList.add('bg-primary', 'progress-bar-striped', 'progress-bar-animated');
        } else if (status === 'completed') {
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            progressBar.classList.add('bg-success');
        } else if (status === 'failed' || status === 'error') {
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            progressBar.classList.add('bg-danger');
        }
        
        console.log(`[DEBUG] 🔥 Progress bar BEFORE - width: ${oldWidth} computed: ${getComputedStyle(progressBar).width}`);
        console.log(`[DEBUG] 🔥 Progress bar AFTER - width: ${progressBar.style.width} computed: ${getComputedStyle(progressBar).width}`);
        console.log(`[DEBUG] Progress bar güncellendi: ${safeProgress}% (text: "${progressBar.textContent}")`);
    }
    
    // ERSIN Global status tracking
    fileStatuses.set(fileId, status);
    
    // ERSIN Error handling
    if (error) {
        const errorElement = fileCard.querySelector('.error-message');
        if (errorElement) {
            errorElement.textContent = error;
            errorElement.style.display = 'block';
        }
    }
    
    // ERSIN 🎯 BUG FIX: Sadece analiz ile ilgili status değişikliklerinde buton state'ini güncelle
    // ERSIN Dosya yükleme durumları (pending, uploading, uploaded) için buton güncellemesi yapma
    const analysisStatuses = ['processing', 'queued', 'completed', 'failed', 'cancelled', 'error'];
    if (analysisStatuses.includes(status.toLowerCase()) && 
        window.analysisManager && window.analysisManager.updateButtonStateBasedOnQueue) {
        // ERSIN Mevcut queue bilgisini alarak buton state'ini güncelle
        window.analysisManager.updateButtonStateBasedOnQueue(0, false);
        console.log(`[DEBUG] 🎯 Analiz status değişikliği: ${status} - buton state güncellendi`);
    }
    
    console.log(`[DEBUG] updateFileStatus tamamlandı - fileId: ${fileId} status: ${status} global progress güncellendi`);
}

/* ERSIN Aciklama. */
function getStatusDisplayName(status) {
    const statusMap = {
        'uploaded': 'Yüklendi',
        'queued': 'Sırada',
        'processing': 'Analiz Ediliyor',
        'completed': 'Tamamlandı',
        'failed': 'Hata',
        'cancelled': 'İptal Edildi',
        'error': 'Hata',
        'pending': 'Bekliyor',
        'yükleniyor': 'Yükleniyor',
        'hazır': 'Hazır'
    };
    return statusMap[status.toLowerCase()] || status;
}

/* ERSIN Aciklama. */
function getStatusBadgeClass(status) {
    const classMap = {
        'uploaded': 'bg-success',
        'queued': 'bg-warning',
        'processing': 'bg-primary',
        'completed': 'bg-success',
        'failed': 'bg-danger',
        'cancelled': 'bg-secondary',
        'error': 'bg-danger',
        'pending': 'bg-secondary',
        'yükleniyor': 'bg-info',
        'hazır': 'bg-info'
    };
    return classMap[status.toLowerCase()] || 'bg-secondary';
}

/* ERSIN Aciklama. */
function getStatusMessage(status) {
    const messageMap = {
        'uploaded': 'Yüklendi, analiz için hazır',
        'queued': 'Analiz sırasında bekliyor',
        'processing': 'Analiz yapılıyor...',
        'completed': 'Analiz tamamlandı',
        'failed': 'Analiz başarısız',
        'cancelled': 'Analiz iptal edildi',
        'error': 'Hata oluştu',
        'pending': 'Bekliyor',
        'yükleniyor': 'Yükleniyor',
        'hazır': 'Hazır'
    };
    return messageMap[status.toLowerCase()] || status;
}

/* ERSIN Aciklama. */
export function removeFile(fileId) {
    // ERSIN Uploaded files array'den kaldır
    const fileIndex = uploadedFiles.findIndex(f => f.id === fileId);
    if (fileIndex !== -1) {
        const file = uploadedFiles[fileIndex];
        uploadedFiles.splice(fileIndex, 1);
        console.log(`Dosya array'den kaldırıldı: ${file.name}`);
    }
    
    // ERSIN DOM'dan kaldır
    const fileCard = document.getElementById(fileId);
    if (fileCard) {
        fileCard.remove();
        console.log(`File card DOM'dan kaldırıldı: ${fileId}`);
    }
    
    // ERSIN Status tracking'den kaldır
    fileStatuses.delete(fileId);
    
    // ERSIN Analysis mapping'den kaldır
    for (const [analysisId, mappedFileId] of fileAnalysisMap.entries()) {
        if (mappedFileId === fileId) {
            fileAnalysisMap.delete(analysisId);
            break;
        }
    }
    
    console.log(`Dosya tamamen kaldırıldı: ${fileId}`);
    
    // ERSIN 🔘 HİÇ DOSYA KALMADIYSA ANALYZE BUTTON'I DISABLE ET
    if (uploadedFiles.length === 0) {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add('disabled');
            console.log('🔘 Analyze button disabled - hiç dosya yok');
        }
    }
}

/* ERSIN Aciklama. */
export function clearAllFiles() {
    // ERSIN Array'i temizle
    uploadedFiles.length = 0;
    
    // ERSIN DOM'u temizle
    const fileList = document.getElementById('fileList');
    if (fileList) {
        fileList.innerHTML = '';
    }
    
    // ERSIN Status tracking'i temizle
    fileStatuses.clear();
    fileAnalysisMap.clear();
    
    console.log('Tüm dosyalar temizlendi');
}

/* ERSIN Aciklama. */
export function getUploadedFileCount() {
    return uploadedFiles.filter(f => f.status === 'uploaded').length;
}

/* ERSIN Aciklama. */
export function exposeFileManagerToWindow() {
    window.fileManager = {
        handleFileSelection,
        handleFiles,
        removeFile,
        clearAllFiles,
        updateFileStatus,
        getUploadedFileCount
    };
}

// ERSIN Initialize window exposure
exposeFileManagerToWindow(); 