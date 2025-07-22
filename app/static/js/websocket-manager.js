/**
 * WSANALIZ - WebSocket Manager Module
 * 
 * Bu modül WebSocket bağlantıları ve notification engelleme sistemini yönetir.
 * main.js'ten extract edilmiştir.
 */

import { setSocket, showToast } from './globals.js';

// =====================================
// WEBSOCKET MANAGEMENT
// =====================================

let socketioClient = null;

/**
 * WebSocket sistemi başlatır
 * @param {HTMLElement} settingsSaveLoader - Loading spinner elementi
 */
export function initializeSocket(settingsSaveLoader) {
    console.log('🚀 WebSocket sistemi aktif - Async Age Estimation desteği ile');
    
    // 🔥 Kapsamlı notification engelleme sistemi
    setupNotificationBlocking();
    
    // WebSocket client instance'ını oluştur
    if (typeof WebSocketClient !== 'undefined') {
        socketioClient = new WebSocketClient();
        window.socketioClient = socketioClient;
        setSocket(socketioClient);
        
        setupWebSocketEventListeners();
        
        console.log('✅ WebSocket client oluşturuldu ve bağlantı başlatıldı');
    } else {
        console.error('❌ WebSocketClient class bulunamadı!');
        return;
    }
    
    // Analysis params modal setup
    setupAnalysisParamsModal(settingsSaveLoader);
}

/**
 * Notification engelleme sistemini kurar
 */
function setupNotificationBlocking() {
    console.log('🚫 Kapsamlı notification engelleme sistemi aktifleştiriliyor...');
    
    // 1. Web Notification API'yi engelle
    if ('Notification' in window) {
        const originalNotification = window.Notification;
        try {
            window.Notification = function() {
                console.log('🚫 Notification constructor engellendi');
                return {};
            };
            window.Notification.requestPermission = function() {
                console.log('🚫 Notification.requestPermission engellendi');
                return Promise.resolve('denied');
            };
            window.Notification.permission = 'denied';
        } catch(e) {
            console.log('⚠️ Notification API engelleme hatası:', e);
        }
    }
    
    // 2. Service Worker notifications engelle
    if ('serviceWorker' in navigator) {
        try {
            const originalRegister = navigator.serviceWorker.register;
            navigator.serviceWorker.register = function() {
                console.log('🚫 ServiceWorker register engellendi');
                return Promise.reject(new Error('ServiceWorker blocked'));
            };
        } catch(e) {}
    }
    
    // 3. Push API engelle
    if ('PushManager' in window) {
        try {
            const originalSubscribe = PushManager.prototype.subscribe;
            PushManager.prototype.subscribe = function() {
                console.log('🚫 PushManager subscribe engellendi');
                return Promise.reject(new Error('Push notifications blocked'));
            };
        } catch(e) {}
    }
    
    // 4. Chrome notifications engelle
    if (window.chrome && window.chrome.notifications) {
        try {
            const originalCreate = window.chrome.notifications.create;
            window.chrome.notifications.create = function() {
                console.log('🚫 Chrome notification create engellendi');
            };
        } catch(e) {}
    }
    
    // 5. Window focus events engelle (bazı popup'ların tetikleyicisi)
    try {
        const originalAddEventListener = window.addEventListener;
        window.addEventListener = function(type, listener, options) {
            if (type === 'focus' || type === 'blur' || type === 'beforeunload') {
                console.log(`🚫 ${type} event listener engellendi`);
                return;
            }
            return originalAddEventListener.call(this, type, listener, options);
        };
    } catch(e) {}
    
    // 6. Console warning/error filtreleme (WebSocket mesajları için)
    try {
        const originalConsoleWarn = console.warn;
        const originalConsoleError = console.error;
        
        console.warn = function(...args) {
            const message = args.join(' ').toLowerCase();
            if (message.includes('websocket') || message.includes('disconnect') || 
                message.includes('connection') || message.includes('socket')) {
                // WebSocket ile ilgili warning'leri sustur
                return;
            }
            return originalConsoleWarn.apply(console, args);
        };
        
        console.error = function(...args) {
            const message = args.join(' ').toLowerCase();
            if (message.includes('websocket') || message.includes('disconnect') || 
                message.includes('connection') || message.includes('socket')) {
                // WebSocket ile ilgili error'ları sustur
                return;
            }
            return originalConsoleError.apply(console, args);
        };
    } catch(e) {}
    
    console.log('✅ Kapsamlı notification engelleme sistemi aktif');
}

/**
 * WebSocket event listener'larını kurar
 */
function setupWebSocketEventListeners() {
    if (!socketioClient) return;
    
    // Browser background detection ve visibility API
    try {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                socketioClient.backgroundMode = true;
                console.log('🌙 Browser arka plana geçti, background mode aktif');
            } else {
                socketioClient.backgroundMode = false;
                console.log('🌞 Browser ön plana geçti, normal mode aktif');
            }
        });
    } catch(e) {
        console.log('⚠️ Visibility API desteklenmiyor:', e);
    }
}

/**
 * Analysis parameters modal kurulumu
 */
function setupAnalysisParamsModal(settingsSaveLoader) {
    // Global analiz parametreleri modalı
    const globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal');
    if (globalAnalysisParamsModalElement) {
        const globalAnalysisParamsModal = new bootstrap.Modal(globalAnalysisParamsModalElement);
        const globalAnalysisParamsForm = document.getElementById('analysisParamsForm');
        const saveGlobalAnalysisParamsBtn = document.getElementById('saveAnalysisParamsBtn');
        
        // Modal event listeners
        setupModalEventListeners(globalAnalysisParamsModalElement);
        
        // Form submit handler
        if (saveGlobalAnalysisParamsBtn) {
            saveGlobalAnalysisParamsBtn.addEventListener('click', function () {
                handleAnalysisParamsSave(globalAnalysisParamsForm, settingsSaveLoader);
            });
        }
    }
}

/**
 * Modal event listener'larını kurar
 */
function setupModalEventListeners(modalElement) {
    if (!modalElement) return;
    
    modalElement.addEventListener('show.bs.modal', function () {
        this.removeAttribute('aria-hidden');
        document.body.style.overflow = 'hidden';
        console.log('[DEBUG] Analysis modal açıldı, body scroll engellendi');
    });
    
    modalElement.addEventListener('hide.bs.modal', function () {
        this.setAttribute('aria-hidden', 'true');
        console.log('[DEBUG] Analysis modal kapandı, aria-hidden eklendi');
    });
    
    modalElement.addEventListener('hidden.bs.modal', function () {
        // Modal tamamen kapandığında backdrop'ı temizle ve scroll'u geri getir
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => {
            backdrop.remove();
            console.log('[DEBUG] Backdrop temizlendi');
        });
        document.body.style.overflow = '';
        console.log('[DEBUG] Body scroll geri getirildi');
    });
}

/**
 * Analysis parameters form save handler
 */
function handleAnalysisParamsSave(form, settingsSaveLoader) {
    const formData = new FormData(form);
    const params = {};
    let formIsValid = true;

    // Form verilerini validate et ve params objesine ekle
    for (const [key, value] of formData.entries()) {
        const inputElement = form.elements[key];
        
        if (inputElement && inputElement.type === 'number') {
            const numValue = Number(value);
            if (isNaN(numValue)) {
                formIsValid = false;
                showToast('Hata', `${key} için geçerli bir sayı giriniz.`, 'error');
                break;
            }
            params[key] = numValue;
        } else {
            params[key] = value;
        }
    }

    if (!formIsValid) return;

    // Loading göster
    if (settingsSaveLoader) {
        settingsSaveLoader.style.display = 'inline-block';
        settingsSaveLoader.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Kaydediliyor...';
    }

    // API'ye kaydet
    fetch('/api/settings/analysis-params', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'none';
        }
        
        if (data.success) {
            showToast('Başarılı', 'Analiz parametreleri kaydedildi.', 'success');
            
            // WebSocket yeniden bağlan
            if (socketioClient && socketioClient.disconnect) {
                socketioClient.disconnect();
                setTimeout(() => {
                    if (socketioClient.connect) {
                        socketioClient.connect();
                    }
                }, 1000);
            }
        } else {
            showToast('Hata', data.error || 'Global ayarlar kaydedilirken bir hata oluştu.', 'error');
        }
    })
    .catch(error => {
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'none';
        }
        console.error('Save params error:', error);
        showToast('Hata', 'Ayarlar kaydedilirken bir hata oluştu.', 'error');
    });
}

/**
 * WebSocket bağlantısını döndürür
 */
export function getSocketConnection() {
    return socketioClient;
}

/**
 * WebSocket bağlantı durumunu kontrol eder
 */
export function isSocketConnected() {
    return socketioClient && socketioClient.connected;
}

/**
 * WebSocket event emit eder
 */
export function emitSocketEvent(eventName, data) {
    if (isSocketConnected()) {
        socketioClient.emit(eventName, data);
        return true;
    }
    console.warn('⚠️ WebSocket bağlı değil, event emit edilemedi:', eventName);
    return false;
} 