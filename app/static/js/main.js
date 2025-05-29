/**
 * WSANALIZ Frontend JavaScript
 * ==========================
 * 
 * Bu dosya WSANALIZ web uygulamasının frontend işlevselliğini sağlar.
 * Dosya yükleme, analiz başlatma, sonuç görüntüleme ve model yönetimi
 * işlemlerini yönetir.
 */

// Global değişkenler
let uploadedFiles = [];
let socket = null;
let queueStatusChecker = null;
let modalQueueStatusChecker = null;

/**
 * Dosya yolunu normalize eder - Windows ve Unix path'lerini düzenler
 */
function normalizePath(path) {
    if (!path || path === 'undefined' || path === '../undefined') {
        return '';
    }
    
    // Windows backslash'leri forward slash'e çevir
    let normalized = path.replace(/\\/g, '/');
    
    // Başındaki ../ kısımları temizle
    while (normalized.startsWith('../') || normalized.startsWith('./')) {
        normalized = normalized.substring(3);
    }
    
    return normalized;
}
