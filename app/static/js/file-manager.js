/**
 * WSANALIZ - File Manager Module
 * 
 * Bu modül dosya upload, yönetimi ve file operations'larını içerir.
 * main.js'ten extract edilmiştir.
 */

import { 
    uploadedFiles, 
    fileStatuses, 
    fileAnalysisMap, 
    API_URL,
    formatFileSize,
    showToast,
    showError
} from './globals.js';

// =====================================
// FILE OPERATIONS
// =====================================

/**
 * Dosya seçimi işlemini yönetir
 * @param {Event} event - File input change event
 */
export function handleFileSelection(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

/**
 * Dosyaları işler (drag&drop veya file input'tan gelen)
 * @param {FileList} files - İşlenecek dosyalar
 */
export function handleFiles(files) {
    const fileArray = Array.from(files);
    
    // Dosya türü kontrolü
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
    
    // Dosyaları listeye ekle
    validFiles.forEach(file => addFileToList(file));
    
    // Upload işlemini başlat
    uploadFilesSequentially(0);
}

/**
 * Dosya türünün geçerli olup olmadığını kontrol eder
 * @param {File} file - Kontrol edilecek dosya
 * @returns {boolean}
 */
function isValidFileType(file) {
    const validTypes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp',
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/flv',
        'video/webm', 'video/mkv'
    ];
    return validTypes.includes(file.type);
}

/**
 * Dosyayı file listesine ekler
 * @param {File} file - Eklenecek dosya
 */
function addFileToList(file) {
    // Unique ID oluştur
    const fileId = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const fileInfo = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
        serverFileId: null, // Server'dan gelecek
        status: 'pending'
    };
    
    uploadedFiles.push(fileInfo);
    
    // 🎯 Overall progress bar'ı güncelle (yeni dosya eklendi)
    if (typeof window.updateOverallProgress === 'function') {
        window.updateOverallProgress({ queue_size: 0, is_processing: false });
    }
    
    // UI'da file card oluştur
    createFileCard(fileInfo);
    
    // File status'ını map'e ekle
    fileStatuses.set(fileId, 'pending');
    
    console.log(`Dosya listeye eklendi: ${file.name} (ID: ${fileId})`);
}

/**
 * File card UI elementi oluşturur
 * @param {Object} fileInfo - Dosya bilgileri
 */
function createFileCard(fileInfo) {
    const fileList = document.getElementById('fileList');
    if (!fileList) return;
    
    // File list section'ı görünür yap (ilk dosya eklendiğinde)
    const fileListSection = document.getElementById('fileListSection');
    if (fileListSection && fileListSection.style.display === 'none') {
        fileListSection.style.display = 'block';
        console.log('📁 File list section görünür hale getirildi');
    }
    
    // 🎨 ORİJİNAL DESIGN: HTML template'dan güzel tasarımı kullan
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
    
    // 🎬 VIDEO DOSYASI İÇİN GERÇEK THUMBNAIL OLUŞTUR
    if (fileInfo.type.startsWith('video/')) {
        const previewImg = fileCard.querySelector('.file-preview');
        if (previewImg) {
            createVideoThumbnail(fileInfo, previewImg);
        }
    }
    
    // 🔘 FILE YÜKLENINCE ANALYZE BUTTON'I ENABLE ET
    enableAnalyzeButton();
}

/**
 * Analyze button'ını enable eder
 */
function enableAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn && uploadedFiles.length > 0) {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('disabled');
        console.log('🔘 Analyze button enabled - dosya sayısı:', uploadedFiles.length);
    }
}

/**
 * File preview HTML'i oluşturur
 * @param {Object} fileInfo - Dosya bilgileri
 * @returns {string}
 */
function createFilePreviewHTML(fileInfo) {
    if (fileInfo.type.startsWith('image/')) {
        const imageUrl = URL.createObjectURL(fileInfo.file);
        return `<img class="file-preview" src="${imageUrl}" alt="${fileInfo.name}">`;
    } else if (fileInfo.type.startsWith('video/')) {
        // Video için boş img tag oluştur, JavaScript ile thumbnail ayarlanacak
        return `<img class="file-preview" src="" alt="Video önizlemesi" data-file-id="${fileInfo.id}">`;
    } else {
        return `<img class="file-preview" src="/static/img/placeholder-face.png" alt="Dosya önizlemesi">`;
    }
}

/**
 * Video dosyası için gerçek thumbnail oluşturur (yedek main.js'teki logic)
 * @param {Object} fileInfo - Video dosya bilgisi
 * @param {HTMLImageElement} previewElement - Preview image element
 */
function createVideoThumbnail(fileInfo, previewElement) {
    console.log(`🎬 Video thumbnail oluşturuluyor: ${fileInfo.name}`);
    
    const fileURL = URL.createObjectURL(fileInfo.file);
    const video = document.createElement('video');
    video.src = fileURL;
    
    video.onloadeddata = () => {
        // Video yüklendikten sonra ilk kareyi al
        video.currentTime = 0.1;
    };
    
    video.onseeked = () => {
        // Canvas oluştur ve ilk kareyi çiz
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 320;
        canvas.height = video.videoHeight || 240;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Canvas'taki resmi önizleme olarak ayarla
        previewElement.src = canvas.toDataURL();
        console.log(`✅ Video thumbnail oluşturuldu: ${fileInfo.name}`);
        
        // Belleği temizle
        URL.revokeObjectURL(fileURL);
    };
    
    // Hata durumunda veya zaman aşımında blob URL'i temizle
    video.onerror = () => {
        console.log(`❌ Video thumbnail oluşturulamadı: ${fileInfo.name}`);
        // Fallback placeholder kullan
        previewElement.src = '/static/img/placeholder-face.png';
        URL.revokeObjectURL(fileURL);
    };
    
    // 5 saniye sonra hala işlenmemişse URL'i temizle (zaman aşımı güvenlik önlemi)
    setTimeout(() => {
        if (video.src) {
            URL.revokeObjectURL(fileURL);
        }
    }, 5000);
}

/**
 * Dosyaları sıralı olarak upload eder
 * @param {number} index - Şu anki dosya index'i
 */
export function uploadFilesSequentially(index) {
    if (index >= uploadedFiles.length) {
        console.log('Tüm dosyalar yüklendi');
        return;
    }
    
    const fileInfo = uploadedFiles[index];
    if (fileInfo.status !== 'pending') {
        // Bu dosya zaten yüklenmiş, bir sonrakine geç
        uploadFilesSequentially(index + 1);
        return;
    }
    
    // Dosya yükleme durumunu güncelle
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
            // Server file ID'yi kaydet
            fileInfo.serverFileId = data.file_id;
            fileInfo.status = 'uploaded';
            
            // File status'ını güncelle
            updateFileStatus(fileInfo.id, 'Sırada', 100);
            
            console.log(`Dosya yüklendi: ${fileInfo.name}, ID: ${data.file_id}`);
            
            // ID mapping'i oluştur
            createFileIdMapping(data.file_id, fileInfo.id);
            
            // Bir sonraki dosyayı yükle
            uploadFilesSequentially(index + 1);
        } else {
            throw new Error(data.error || 'Dosya yüklenemedi');
        }
    })
    .catch(error => {
        console.error(`Dosya yükleme hatası (${fileInfo.name}):`, error);
        updateFileStatus(fileInfo.id, 'Hata', 0, null, error.message);
        showError(`Dosya yükleme hatası: ${fileInfo.name} - ${error.message}`);
        
        // Hata olsa bile diğer dosyalara devam et
        uploadFilesSequentially(index + 1);
    });
}

/**
 * File ID mapping oluşturur (server ID -> client ID)
 * @param {number} serverFileId - Server'daki dosya ID'si
 * @param {string} clientFileId - Client'taki dosya ID'si
 */
function createFileIdMapping(serverFileId, clientFileId) {
    // Global mapping objesi
    if (!window.fileIdToCardId) {
        window.fileIdToCardId = {};
    }
    window.fileIdToCardId[serverFileId] = clientFileId;
    
    console.log(`[DEBUG] fileIdToCardId mapping oluşturuldu: ${serverFileId} → ${clientFileId}`);
}

/**
 * Dosya durumunu günceller
 * @param {string} fileId - Dosya ID'si
 * @param {string} status - Yeni durum
 * @param {number} progress - İlerleme yüzdesi (0-100)
 * @param {string} message - Durum mesajı
 * @param {string} error - Hata mesajı
 */
export function updateFileStatus(fileId, status, progress, message = null, error = null) {
    console.log(`[DEBUG] updateFileStatus çağrıldı: ${fileId} ${status} ${progress}`);
    
    const fileCard = document.getElementById(fileId);
    if (!fileCard) {
        console.warn(`[DEBUG] fileCard bulunamadı: ${fileId}`);
        return;
    }
    
    console.log(`[DEBUG] fileCard bulundu mu? ${!!fileCard}`);
    
    // 🎨 YENİ TEMPLATE STRUCTURE - eski design ile uyumlu selectors
    const statusDiv = fileCard.querySelector('.file-status');  // status badge area
    const statusText = fileCard.querySelector('.file-status-text');  // status text
    const progressBar = fileCard.querySelector('.progress-bar');
    
    if (statusText) {
        statusText.textContent = message || getStatusMessage(status);
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
        
        // 🎨 PROGRESS BAR VISUAL UPDATE (yedek main.js'teki logic)
        progressBar.style.width = `${safeProgress}%`;
        progressBar.setAttribute('aria-valuenow', safeProgress);
        
        // Progress bar text content (önemli!)
        if (safeProgress > 0) {
            progressBar.textContent = `${Math.round(safeProgress)}%`;
        } else {
            progressBar.textContent = '';
        }
        
        // CSS classes for animation (processing status için)
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
    
    // Global status tracking
    fileStatuses.set(fileId, status);
    
    // Error handling
    if (error) {
        const errorElement = fileCard.querySelector('.error-message');
        if (errorElement) {
            errorElement.textContent = error;
            errorElement.style.display = 'block';
        }
    }
    
    // 🎯 BUG FIX: File status değiştiğinde buton state'ini de güncelle
    // Analysis manager'ın updateButtonStateBasedOnQueue fonksiyonunu çağır
    if (window.analysisManager && window.analysisManager.updateButtonStateBasedOnQueue) {
        // Mevcut queue bilgisini alarak buton state'ini güncelle
        window.analysisManager.updateButtonStateBasedOnQueue(0, false);
    }
    
    console.log(`[DEBUG] updateFileStatus tamamlandı - fileId: ${fileId} status: ${status} global progress güncellendi`);
}

/**
 * Status display adını döndürür
 */
function getStatusDisplayName(status) {
    const statusMap = {
        'pending': 'Bekleniyor',
        'uploading': 'Yükleniyor',
        'uploaded': 'Yüklendi',
        'queued': 'Sırada',
        'processing': 'Analiz Ediliyor',
        'completed': 'Tamamlandı',
        'failed': 'Başarısız',
        'cancelled': 'İptal Edildi',
        'error': 'Hata'
    };
    return statusMap[status.toLowerCase()] || status;
}

/**
 * Status badge class'ını döndürür
 */
function getStatusBadgeClass(status) {
    const classMap = {
        'pending': 'bg-secondary',
        'uploading': 'bg-info',
        'uploaded': 'bg-success',
        'queued': 'bg-warning',
        'processing': 'bg-primary',
        'completed': 'bg-success',
        'failed': 'bg-danger',
        'cancelled': 'bg-secondary',
        'error': 'bg-danger'
    };
    return classMap[status.toLowerCase()] || 'bg-secondary';
}

/**
 * Status mesajını döndürür
 */
function getStatusMessage(status) {
    const messageMap = {
        'pending': 'Yükleme bekleniyor...',
        'uploading': 'Dosya yükleniyor...',
        'uploaded': 'Yükleme tamamlandı',
        'queued': 'Analiz sırasında bekliyor',
        'processing': 'Analiz yapılıyor...',
        'completed': 'Analiz tamamlandı',
        'failed': 'Analiz başarısız',
        'cancelled': 'Analiz iptal edildi',
        'error': 'Hata oluştu'
    };
    return messageMap[status.toLowerCase()] || status;
}

/**
 * Dosyayı listeden kaldırır
 * @param {string} fileId - Kaldırılacak dosya ID'si
 */
export function removeFile(fileId) {
    // Uploaded files array'den kaldır
    const fileIndex = uploadedFiles.findIndex(f => f.id === fileId);
    if (fileIndex !== -1) {
        const file = uploadedFiles[fileIndex];
        uploadedFiles.splice(fileIndex, 1);
        console.log(`Dosya array'den kaldırıldı: ${file.name}`);
    }
    
    // DOM'dan kaldır
    const fileCard = document.getElementById(fileId);
    if (fileCard) {
        fileCard.remove();
        console.log(`File card DOM'dan kaldırıldı: ${fileId}`);
    }
    
    // Status tracking'den kaldır
    fileStatuses.delete(fileId);
    
    // Analysis mapping'den kaldır
    for (const [analysisId, mappedFileId] of fileAnalysisMap.entries()) {
        if (mappedFileId === fileId) {
            fileAnalysisMap.delete(analysisId);
            break;
        }
    }
    
    console.log(`Dosya tamamen kaldırıldı: ${fileId}`);
    
    // 🔘 HİÇ DOSYA KALMADIYSA ANALYZE BUTTON'I DISABLE ET
    if (uploadedFiles.length === 0) {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add('disabled');
            console.log('🔘 Analyze button disabled - hiç dosya yok');
        }
    }
}

/**
 * Tüm dosyaları temizler
 */
export function clearAllFiles() {
    // Array'i temizle
    uploadedFiles.length = 0;
    
    // DOM'u temizle
    const fileList = document.getElementById('fileList');
    if (fileList) {
        fileList.innerHTML = '';
    }
    
    // Status tracking'i temizle
    fileStatuses.clear();
    fileAnalysisMap.clear();
    
    console.log('Tüm dosyalar temizlendi');
}

/**
 * Upload edilmiş dosya sayısını döndürür
 */
export function getUploadedFileCount() {
    return uploadedFiles.filter(f => f.status === 'uploaded').length;
}

/**
 * Dosya yönetim fonksiyonlarını window'a expose et
 */
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

// Initialize window exposure
exposeFileManagerToWindow(); 