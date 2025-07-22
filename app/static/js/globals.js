/**
 * WSANALIZ - Global Variables & Utilities Module
 * 
 * Bu modül tüm global değişkenleri ve yardımcı fonksiyonları içerir.
 * main.js'ten extract edilmiştir.
 */

// =====================================
// GLOBAL CONSTANTS & VARIABLES
// =====================================

export const API_URL = '/api';

// Global state variables
export let socket = null; // Global socket - tek instance
export let uploadedFiles = [];
export let analysisResults = {};
export let currentAnalysisIds = [];
export let hideLoaderTimeout = null;
export let globalAnalysisParamsModalElement = null;

// Global flags for training
export const globalTrainingState = {
    currentTrainingSessionId: null,
    isModalTraining: false
};

// Analysis state tracking
export const fileStatuses = new Map();  // Maps fileId to status
export const fileAnalysisMap = new Map();  // Maps analysisId to fileId
export const cancelledAnalyses = new Set();  // Set of cancelled analysisId values
export const fileErrorCounts = new Map();  // Maps fileId to error count
export let totalAnalysisCount = 0;
export let MAX_STATUS_CHECK_RETRIES = 5;

// =====================================
// UTILITY FUNCTIONS
// =====================================

/**
 * Dosya yolu normalleştirme fonksiyonu
 * Windows ve Unix yol ayraçlarını normalize eder
 */
export function normalizePath(path) {
    if (path) {
        // Önce tüm backslash'leri slash'e çevir
        return path.replace(/\\/g, '/');
    }
    return path;
}

/**
 * Dosya boyutunu human-readable format'a çevirir
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Saniyeyi okunabilir zaman formatına çevirir
 */
export function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Süreyi dakika:saniye formatında döndürür
 */
export function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Toast bildirimi gösterir
 */
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
        
        // Toast gizlendiğinde DOM'dan kaldır
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
}

/**
 * Hata mesajı gösterir
 */
export function showError(message) {
    showToast('Hata', message, 'error');
}

/**
 * File ID'den dosya adını döndürür
 */
export function fileNameFromId(fileId) {
    const file = uploadedFiles.find(f => f.id === fileId);
    return file ? file.name : 'Bilinmeyen Dosya';
}

/**
 * Global state'i window objesine paylaş
 */
export function exposeGlobalState() {
    window.fileAnalysisMap = fileAnalysisMap;
    window.uploadedFiles = uploadedFiles;
    window.currentTrainingSessionId = globalTrainingState.currentTrainingSessionId;
    window.isModalTraining = globalTrainingState.isModalTraining;
}

/**
 * Global değişkenleri setter fonksiyonları
 */
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

// Initialize global state exposure
exposeGlobalState(); 