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
    
    // 🔧 WebSocket client bağlantısını kurmak için biraz bekle (websocket-client.js yüklensin)
    setTimeout(() => {
        // WebSocket client instance'ını oluştur
        if (typeof WebSocketClient !== 'undefined') {
            socketioClient = new WebSocketClient();
            window.socketioClient = socketioClient;
            setSocket(socketioClient);
            
            // Explicit connection başlat
            socketioClient.connect();
            
            setupWebSocketEventListeners();
            
            console.log('✅ WebSocket client oluşturuldu ve bağlantı başlatıldı');
        } else {
            console.error('❌ WebSocketClient class bulunamadı!');
            // FALLBACK: Direct socket.io connection
            if (typeof io !== 'undefined') {
                console.log('🔄 Fallback: Direct socket.io connection');
                socketioClient = io();
                window.socketioClient = socketioClient;
                setSocket(socketioClient);
                
                setupWebSocketEventListeners();
                console.log('✅ Fallback WebSocket connection kuruldu');
            }
        }
    }, 100);
    
    // Analysis params modal setup - MOVED TO ui-manager.js
    // setupAnalysisParamsModal(settingsSaveLoader);
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
    
    console.log('🔧 WebSocket event listeners kuruluyor...');
    
    // WebSocketClient wrapper'dan native socket'a erişim
    const nativeSocket = socketioClient.socket || socketioClient;
    
    if (typeof nativeSocket.on !== 'function') {
        console.error('❌ Socket.io native instance bulunamadı!');
        return;
    }
    
    // Connection event'leri
    nativeSocket.on('connect', () => {
        console.log('✅ WebSocket bağlantısı kuruldu');
    });
    
    nativeSocket.on('disconnect', () => {
        console.log('⚠️ WebSocket bağlantısı kesildi');
    });
    
    // Analysis progress events - ASIL PROGRESS LISTENER!
    nativeSocket.on('analysis_progress', (data) => {
        console.log('📊 Analysis progress alındı:', data);
        if (window.handleAnalysisProgress) {
            window.handleAnalysisProgress(data);
        } else {
            console.error('❌ handleAnalysisProgress fonksiyonu bulunamadı!');
        }
    });
    
    nativeSocket.on('analysis_completed', (data) => {
        console.log('✅ Analysis completed alındı:', data);
        if (window.handleAnalysisCompleted) {
            window.handleAnalysisCompleted(data);
        }
    });
    
    // Browser background detection ve visibility API
    try {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                if (socketioClient.backgroundMode !== undefined) {
                    socketioClient.backgroundMode = true;
                }
                console.log('🌙 Browser arka plana geçti, background mode aktif');
            } else {
                if (socketioClient.backgroundMode !== undefined) {
                    socketioClient.backgroundMode = false;
                }
                console.log('🌞 Browser ön plana geçti, normal mode aktif');
            }
        });
    } catch(e) {
        console.log('⚠️ Visibility API desteklenmiyor:', e);
    }
    
    console.log('✅ WebSocket event listeners kuruldu');
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
    .then(response => response.json().then(data => ({ status: response.status, body: data })))
    .then(({ status, body: data }) => {
        if (settingsSaveLoader) {
            settingsSaveLoader.style.display = 'none';
        }
        
        // Response status kontrolü (200 OK ise başarılı)
        if (status === 200 && data.message) {
            // Restart durumunu kontrol et
            if (data.restart_required || data.restart_initiated) {
                // Production mode - restart başlatıldı
                showToast('Başarılı', data.message || 'Analiz parametreleri kaydedildi. Sistem yeniden başlatılıyor...', 'success');
                
                // Restart sonrası sayfa yenile
                if (data.restart_initiated) {
                    // Restart sonrası eski UI state'in (uploadedFiles / overall progress) kalmaması için
                    // local restore mekanizmasını bir seferlik devre dışı bırak.
                    try {
                        sessionStorage.setItem('wsanaliz_skip_restore', '1');
                        localStorage.removeItem('wsanaliz_recent_analyses');
                    } catch (e) {
                        console.warn('Restart cleanup storage erişilemedi:', e);
                    }

                    setTimeout(() => {
                        console.log('[DEBUG] Analiz parametreleri güncellendi, sayfa yeniden yükleniyor (restart bekleniyor)...');
                        const url = new URL(window.location.href);
                        url.searchParams.set('restarted', String(Date.now()));
                        window.location.href = url.toString();
                    }, 8000);
                }
            } else {
                // Development mode - normal kayıt
                showToast('Başarılı', data.message || 'Analiz parametreleri kaydedildi.', 'success');
                
                // WebSocket yeniden bağlan (development mode için)
                if (socketioClient && socketioClient.disconnect) {
                    socketioClient.disconnect();
                    setTimeout(() => {
                        if (socketioClient.connect) {
                            socketioClient.connect();
                        }
                    }, 1000);
                }
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
    if (socketioClient && socketioClient.connected) {
        const nativeSocket = socketioClient.socket || socketioClient;
        
        // join_analysis için özel format (backend dict bekliyor)
        if (eventName === 'join_analysis') {
            const joinData = { analysis_id: data };
            console.log('🔗 WebSocket join_analysis emit:', joinData);
            nativeSocket.emit(eventName, joinData);
        } else {
            nativeSocket.emit(eventName, data);
        }
        return true;
    }
    console.warn('⚠️ WebSocket bağlantısı yok, event emit edilemedi:', eventName);
    return false;
} 