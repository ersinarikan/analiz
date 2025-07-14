/**
 * İçerik Analiz Sistemi JavaScript Kodu
 * HTTP API tabanlı - SocketIO kaldırıldı
 */

// Global değişkenler
const API_URL = '/api';
let uploadedFiles = [];
let analysisResults = {};
let currentAnalysisIds = [];

// DOM elementlerini al
document.addEventListener('DOMContentLoaded', () => {
    // HTTP API kullanacağız - SocketIO kaldırıldı
    console.log('HTTP API tabanlı sistem başlatıldı');
    
    // Event listenerları ekle
    setupEventListeners();
});

/**
 * Olay dinleyicileri ekler
 */
function setupEventListeners() {
    // Dosya seçimi ve sürükle bırak olayları
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const folderInput = document.getElementById('folder-input');
    const selectFilesBtn = document.getElementById('select-files-btn');
    const selectFolderBtn = document.getElementById('select-folder-btn');
    
    // Dosya sürükle bırak işlemleri
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('highlight');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('highlight');
        }, false);
    });
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Dosya seçimi butonları
    selectFilesBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    selectFolderBtn.addEventListener('click', () => {
        folderInput.click();
    });
    
    fileInput.addEventListener('change', e => {
        handleFiles(e.target.files);
        e.target.value = null; // Aynı dosyayı tekrar seçebilmek için reset
    });
    
    folderInput.addEventListener('change', e => {
        handleFiles(e.target.files);
        e.target.value = null; // Aynı klasörü tekrar seçebilmek için reset
    });
    
    // Analiz başlatma
    const startAnalysisBtn = document.getElementById('start-analysis-btn');
    startAnalysisBtn.addEventListener('click', showAnalysisSettingsModal);
    
    // Analiz ayarları modal
    const startAnalysisConfirmBtn = document.getElementById('start-analysis-confirm-btn');
    startAnalysisConfirmBtn.addEventListener('click', startAnalysis);
    
    // Analiz ayarları - FPS slider
    const framesPerSecondSlider = document.getElementById('frames-per-second');
    const framesPerSecondValue = document.getElementById('frames-per-second-value');
    
    framesPerSecondSlider.addEventListener('input', () => {
        framesPerSecondValue.textContent = framesPerSecondSlider.value;
    });
    
    // Yapay zeka yeniden eğitim
    const retrainAiBtn = document.getElementById('retrain-ai-btn');
    retrainAiBtn.addEventListener('click', showRetrainModal);
    
    // Yapay zeka eğitim modal
    const startTrainingBtn = document.getElementById('start-training-btn');
    startTrainingBtn.addEventListener('click', startTraining);
    
    const resetModelBtn = document.getElementById('reset-model-btn');
    resetModelBtn.addEventListener('click', resetModel);
    
    // Epoch slider
    const epochCountSlider = document.getElementById('epoch-count');
    const epochCountValue = document.getElementById('epoch-count-value');
    
    epochCountSlider.addEventListener('input', () => {
        epochCountValue.textContent = epochCountSlider.value;
    });
    
    // Geri bildirim
    const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
    submitFeedbackBtn.addEventListener('click', submitFeedback);
    
    // Sayfa yüklendiğinde model istatistiklerini al
    getModelStats();
}

/**
 * Dosya sürükle bırak işlemi
 */
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

/**
 * Dosyaları işle ve arayüze ekle
 */
function handleFiles(files) {
    if (files.length === 0) return;
    
    // Dosyaları kontrol et, sadece resim ve video dosyalarını kabul et
    let validFiles = Array.from(files).filter(file => {
        return file.type.startsWith('image/') || file.type.startsWith('video/');
    });
    
    if (validFiles.length === 0) {
        showAlert('Lütfen geçerli resim veya video dosyaları yükleyin.', 'warning');
        return;
    }
    
    // Geçerli olmayan dosyaları rapor et
    if (validFiles.length < files.length) {
        showAlert(`${files.length - validFiles.length} dosya desteklenmeyen formatta ve atlandı.`, 'warning');
    }
    
    // Dosyaları yükle
    validFiles.forEach(file => {
        uploadFile(file);
    });
}

/**
 * Dosyayı API'ye yükler
 */
function uploadFile(file) {
    // Dosya kartını ekle
    const fileId = 'file-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    const fileCard = createFileCard(file, fileId);
    document.getElementById('file-list').appendChild(fileCard);
    
    // FormData oluştur
    const formData = new FormData();
    formData.append('file', file);
    
    // Fetch API ile dosyayı yükle
    fetch(`${API_URL}/files/`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Dosya yükleme hatası');
        }
        return response.json();
    })
    .then(data => {
        // Dosya yükleme başarılı
        console.log('Dosya yüklendi:', data);
        
        // Dosyayı listeye ekle
        uploadedFiles.push({
            id: data.id,
            filename: data.original_filename,
            file_type: data.file_type,
            elementId: fileId
        });
        
        // Dosya durum bilgisini güncelle
        updateFileStatus(fileId, 'success', 'Yüklendi');
        
        // Analiz butonunu etkinleştir
        document.getElementById('start-analysis-btn').disabled = false;
    })
    .catch(error => {
        console.error('Dosya yükleme hatası:', error);
        updateFileStatus(fileId, 'error', 'Yükleme hatası');
    });
}

/**
 * Dosya kartı oluşturur
 */
function createFileCard(file, fileId) {
    const col = document.createElement('div');
    col.className = 'col';
    
    const card = document.createElement('div');
    card.className = 'card file-card h-100';
    card.id = fileId;
    
    const thumbnail = document.createElement('div');
    thumbnail.className = 'file-thumbnail';
    
    // Dosya türüne göre ön izleme göster
    if (file.type.startsWith('image/')) {
        const img = document.createElement('img');
        const imgUrl = URL.createObjectURL(file);
        img.src = imgUrl;
        img.alt = file.name;
        // Clean up the blob URL after the image has loaded
        img.onload = () => {
            URL.revokeObjectURL(imgUrl);
        };
        thumbnail.appendChild(img);
    } else if (file.type.startsWith('video/')) {
        const video = document.createElement('video');
        const videoUrl = URL.createObjectURL(file);
        video.src = videoUrl;
        video.muted = true;
        video.controls = false;
        video.autoplay = false;
        video.loop = true;
        
        // Clean up the blob URL after the video metadata has loaded
        video.onloadedmetadata = () => {
            URL.revokeObjectURL(videoUrl);
        };
        
        video.addEventListener('mouseover', () => {
            video.play();
        });
        video.addEventListener('mouseout', () => {
            video.pause();
        });
        thumbnail.appendChild(video);
    } else {
        // Desteklenmeyen dosya tipi için ikon
        const icon = document.createElement('i');
        icon.className = 'fas fa-file fa-3x text-muted';
        thumbnail.appendChild(icon);
    }
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body file-info';
    
    const fileName = document.createElement('h6');
    fileName.className = 'card-title file-name';
    fileName.textContent = file.name;
    
    const fileSize = document.createElement('p');
    fileSize.className = 'card-text text-muted small mb-2';
    fileSize.textContent = formatFileSize(file.size);
    
    const statusDiv = document.createElement('div');
    statusDiv.className = 'd-flex justify-content-between align-items-center mt-2';
    
    const statusBadge = document.createElement('span');
    statusBadge.className = 'badge bg-secondary status-badge';
    statusBadge.textContent = 'Yükleniyor...';
    
    const progressDiv = document.createElement('div');
    progressDiv.className = 'progress file-progress mt-2';
    progressDiv.style.height = '5px';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.role = 'progressbar';
    progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', '0');
    progressBar.setAttribute('aria-valuemin', '0');
    progressBar.setAttribute('aria-valuemax', '100');
    
    progressDiv.appendChild(progressBar);
    statusDiv.appendChild(statusBadge);
    
    cardBody.appendChild(fileName);
    cardBody.appendChild(fileSize);
    cardBody.appendChild(statusDiv);
    cardBody.appendChild(progressDiv);
    
    card.appendChild(thumbnail);
    card.appendChild(cardBody);
    
    col.appendChild(card);
    
    return col;
} 