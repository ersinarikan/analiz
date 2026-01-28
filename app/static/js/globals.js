/* ERSIN Aciklama. */

// ERSIN =====================================
// ERSIN GLOBAL CONSTANTS & VARIABLES
// ERSIN =====================================

export const API_URL = '/api';

// ERSIN Global state variables
export let socket = null;  // ERSIN Global socket - tek instance
export let uploadedFiles = [];
export let analysisResults = {};
export let currentAnalysisIds = [];
export let hideLoaderTimeout = null;
export let globalAnalysisParamsModalElement = null;

// ERSIN Global flags için training
export const globalTrainingState = {
    currentTrainingSessionId: null,
    isModalTraining: false
};

// ERSIN Analysis state tracking
export const fileStatuses = new Map();  // ERSIN Maps fileId to status
export const fileAnalysisMap = new Map();  // ERSIN Maps analysisId to fileId
export const cancelledAnalyses = new Set();  // ERSIN Set of cancelled analysisId values
export const fileErrorCounts = new Map();  // ERSIN Maps fileId to error count
export let totalAnalysisCount = 0;
export let MAX_STATUS_CHECK_RETRIES = 5;

// ERSIN =====================================
// ERSIN UTILITY FUNCTIONS
// ERSIN =====================================

/* ERSIN Aciklama. */
export function normalizePath(path) {
    if (path) {
        // ERSIN Önce tüm backslash'leri slash'e çevir
        return path.replace(/\\/g, '/');
    }
    return path;
}

/* ERSIN Aciklama. */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/* ERSIN Aciklama. */
export function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/* ERSIN Aciklama. */
export function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/* ERSIN Aciklama. */
export function showToast(title, message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <strong>${title}</strong><br>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    const toastContainer = document.getElementById('toast-container');
    if (toastContainer) {
        toastContainer.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // ERSIN Toast gizlendiğinde DOM'dan kaldır
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
}

/* ERSIN Aciklama. */
export function showError(message) {
    showToast('Hata', message, 'error');
}

/* ERSIN Aciklama. */
export function fileNameFromId(fileId) {
    const file = uploadedFiles.find(f => f.id === fileId);
    return file ? file.name : 'Bilinmeyen Dosya';
}

/* ERSIN Aciklama. */
export function exposeGlobalState() {
    window.fileAnalysisMap = fileAnalysisMap;
    window.uploadedFiles = uploadedFiles;
    window.currentTrainingSessionId = globalTrainingState.currentTrainingSessionId;
    window.isModalTraining = globalTrainingState.isModalTraining;
}

/* ERSIN Aciklama. */
export function setSocket(newSocket) {
    socket = newSocket;
}

export function setGlobalAnalysisParamsModalElement(element) {
    globalAnalysisParamsModalElement = element;
}

export function setCurrentTrainingSessionId(sessionId) {
    globalTrainingState.currentTrainingSessionId = sessionId;
    window.currentTrainingSessionId = sessionId;
}

export function setIsModalTraining(isTraining) {
    globalTrainingState.isModalTraining = isTraining;
    window.isModalTraining = isTraining;
}

// ERSIN Initialize global state exposure
exposeGlobalState(); 