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
    
    const fileCard = document.createElement('div');
    fileCard.className = 'col-md-6 col-lg-4 mb-3';
    fileCard.id = fileInfo.id;
    fileCard.setAttribute('data-file-id', fileInfo.id);
    
    fileCard.innerHTML = `
        <div class="card file-card h-100">
            <div class="file-preview">
                ${createFilePreviewHTML(fileInfo)}
            </div>
            <div class="card-body">
                <h6 class="card-title text-truncate" title="${fileInfo.name}">
                    ${fileInfo.name}
                </h6>
                <p class="card-text">
                    <small class="text-muted">
                        Boyut: ${formatFileSize(fileInfo.size)}
                    </small>
                </p>
                <div class="file-status">
                    <span class="badge bg-secondary status-badge">Bekleniyor</span>
                    <div class="progress mt-2" style="height: 8px;">
                        <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                    </div>
                    <small class="status-text text-muted">Yükleme bekleniyor...</small>
                </div>
            </div>
            <div class="card-footer">
                <button type="button" class="btn btn-sm btn-outline-danger remove-file-btn" 
                        onclick="window.fileManager.removeFile('${fileInfo.id}')">
                    <i class="fas fa-trash-alt me-1"></i>Kaldır
                </button>
            </div>
        </div>
    `;
    
    fileList.appendChild(fileCard);
}

/**
 * File preview HTML'i oluşturur
 * @param {Object} fileInfo - Dosya bilgileri
 * @returns {string}
 */
function createFilePreviewHTML(fileInfo) {
    if (fileInfo.type.startsWith('image/')) {
        const imageUrl = URL.createObjectURL(fileInfo.file);
        return `
            <img src="${imageUrl}" alt="${fileInfo.name}" 
                 class="file-preview-img" 
                 onload="URL.revokeObjectURL(this.src)">
        `;
    } else if (fileInfo.type.startsWith('video/')) {
        return `
            <div class="video-preview">
                <i class="fas fa-video fa-3x text-primary"></i>
                <div class="mt-2">Video</div>
            </div>
        `;
    } else {
        return `
            <div class="file-preview-placeholder">
                <i class="fas fa-file fa-3x text-secondary"></i>
                <div class="mt-2">Dosya</div>
            </div>
        `;
    }
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
    
    // Status badge'i güncelle
    const statusBadge = fileCard.querySelector('.status-badge');
    const statusText = fileCard.querySelector('.status-text');
    const progressBar = fileCard.querySelector('.progress-bar');
    
    if (statusText) {
        statusText.textContent = message || getStatusMessage(status);
        console.log(`[DEBUG] statusText güncellendi: ${statusText.textContent}`);
    }
    
    if (statusBadge) {
        statusBadge.textContent = getStatusDisplayName(status);
        statusBadge.className = `badge ${getStatusBadgeClass(status)} status-badge`;
        console.log(`[DEBUG] statusBadge güncellendi: ${statusBadge.textContent}`);
    }
    
    if (progressBar) {
        const oldWidth = progressBar.style.width;
        progressBar.style.width = `${progress}%`;
        console.log(`[DEBUG] 🔥 Progress bar BEFORE - width: ${oldWidth} computed: ${getComputedStyle(progressBar).width}`);
        console.log(`[DEBUG] 🔥 Progress bar AFTER - width: ${progressBar.style.width} computed: ${getComputedStyle(progressBar).width}`);
        console.log(`[DEBUG] Progress bar güncellendi: ${progress}%`);
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