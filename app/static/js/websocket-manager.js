/* ERSIN Aciklama. */

import { setSocket, showToast } from './globals.js';

// ERSIN =====================================
// ERSIN WEBSOCKET MANAGEMENT
// ERSIN =====================================

let socketioClient = null;

/* ERSIN Aciklama. */
export function initializeSocket(settingsSaveLoader) {
    console.log('🚀 WebSocket sistemi aktif - Async Age Estimation desteği ile');
    
    // ERSIN 🔥 Kapsamlı notification engelleme sistemi
    setupNotificationBlocking();
    
    // ERSIN 🔧 WebSocket client bağlantısını kurmak için biraz bekle (websocket-client.js yüklensin)
    setTimeout(() => {
        // ERSIN WebSocket client instance'ını oluştur
        if (typeof WebSocketClient !== 'undefined') {
            socketioClient = new WebSocketClient();
            window.socketioClient = socketioClient;
            setSocket(socketioClient);
            
            // ERSIN Explicit connection başlat
            socketioClient.connect();
            
            setupWebSocketEventListeners();
            
            console.log('✅ WebSocket client oluşturuldu ve bağlantı başlatıldı');
        } else {
            console.error('❌ WebSocketClient class bulunamadı!');
            // ERSIN FALLBACK: Direct socket.io connection
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
    
    // ERSIN Analysis params modal setup - MOVED TO ui-manager.js
    // ERSIN setupAnalysisParamsModal(settingsSaveLoader);
}

/* ERSIN Aciklama. */
function setupNotificationBlocking() {
    console.log('🚫 Kapsamlı notification engelleme sistemi aktifleştiriliyor...');
    
    // ERSIN 1. Web Notification API'yi engelle
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
    
    // ERSIN 2. Service Worker notifications engelle
    if ('serviceWorker' in navigator) {
        try {
            const originalRegister = navigator.serviceWorker.register;
            navigator.serviceWorker.register = function() {
                console.log('🚫 ServiceWorker register engellendi');
                return Promise.reject(new Error('ServiceWorker blocked'));
            };
        } catch(e) {}
    }
    
    // ERSIN 3. Push API engelle
    if ('PushManager' in window) {
        try {
            const originalSubscribe = PushManager.prototype.subscribe;
            PushManager.prototype.subscribe = function() {
                console.log('🚫 PushManager subscribe engellendi');
                return Promise.reject(new Error('Push notifications blocked'));
            };
        } catch(e) {}
    }
    
    // ERSIN 4. Chrome notifications engelle
    if (window.chrome && window.chrome.notifications) {
        try {
            const originalCreate = window.chrome.notifications.create;
            window.chrome.notifications.create = function() {
                console.log('🚫 Chrome notification create engellendi');
            };
        } catch(e) {}
    }
    
    // ERSIN 5. Window focus events engelle (bazı popup'ların tetikleyicisi)
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
    
    // ERSIN 6. Console warning/error filtreleme (WebSocket mesajları için)
    try {
        const originalConsoleWarn = console.warn;
        const originalConsoleError = console.error;
        
        console.warn = function(...args) {
            const message = args.join(' ').toLowerCase();
            if (message.includes('websocket') || message.includes('disconnect') || 
                message.includes('connection') || message.includes('socket')) {
                // ERSIN WebSocket ile ilgili warning'leri sustur
                return;
            }
            // ERSIN Amplitude Logger warning'lerini sustur (browser extension'dan geliyor)
            if (message.includes('amplitude') || message.includes('defaulttracking')) {
                return;  // ERSIN Browser extension warning'lerini filtrele
            }
            return originalConsoleWarn.apply(console, args);
        };
        
        console.error = function(...args) {
            const message = args.join(' ').toLowerCase();
            if (message.includes('websocket') || message.includes('disconnect') || 
                message.includes('connection') || message.includes('socket')) {
                // ERSIN WebSocket ile ilgili error'ları sustur
                return;
            }
            // ERSIN 404 hatalarını sustur - analiz sonuçları temizlenmiş olabilir
            if (message.includes('404') && (message.includes('detailed-results') || message.includes('analysis'))) {
                return;  // ERSIN Sessizce çık, bu normal bir durum
            }
            return originalConsoleError.apply(console, args);
        };
    } catch(e) {}
    
    console.log('✅ Kapsamlı notification engelleme sistemi aktif');
}

/* ERSIN Aciklama. */
function setupWebSocketEventListeners() {
    if (!socketioClient) return;
    
    console.log('🔧 WebSocket event listeners kuruluyor...');
    
    // ERSIN WebSocketClient wrapper'dan native socket'a erişim
    const nativeSocket = socketioClient.socket || socketioClient;
    
    if (typeof nativeSocket.on !== 'function') {
        console.error('❌ Socket.io native instance bulunamadı!');
        return;
    }
    
    // ERSIN Connection event'leri
    nativeSocket.on('connect', () => {
        console.log('✅ WebSocket bağlantısı kuruldu');
    });
    
    nativeSocket.on('disconnect', () => {
        console.log('⚠️ WebSocket bağlantısı kesildi');
    });
    
    // ERSIN Analysis progress events - ASIL PROGRESS LISTENER!
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
    
    // ERSIN Browser background detection ve visibility API
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

/* ERSIN Aciklama. */
function setupAnalysisParamsModal(settingsSaveLoader) {
    // ERSIN Global analiz parametreleri modalı
    const globalAnalysisParamsModalElement = document.getElementById('analysisParamsModal');
    if (globalAnalysisParamsModalElement) {
        const globalAnalysisParamsModal = new bootstrap.Modal(globalAnalysisParamsModalElement);
        const globalAnalysisParamsForm = document.getElementById('analysisParamsForm');
        const saveGlobalAnalysisParamsBtn = document.getElementById('saveAnalysisParamsBtn');
        
        // ERSIN Modal event listeners
        setupModalEventListeners(globalAnalysisParamsModalElement);
        
        // ERSIN Form submit handler
        if (saveGlobalAnalysisParamsBtn) {
            saveGlobalAnalysisParamsBtn.addEventListener('click', function () {
                handleAnalysisParamsSave(globalAnalysisParamsForm, settingsSaveLoader);
            });
        }
    }
}

/* ERSIN Aciklama. */
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
        // ERSIN Modal tamamen kapandığında backdrop'ı temizle ve scroll'u geri getir
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => {
            backdrop.remove();
            console.log('[DEBUG] Backdrop temizlendi');
        });
        document.body.style.overflow = '';
        console.log('[DEBUG] Body scroll geri getirildi');
    });
}

/* ERSIN Aciklama. */
function handleAnalysisParamsSave(form, settingsSaveLoader) {
    const formData = new FormData(form);
    const params = {};
    let formIsValid = true;

    // ERSIN Form verilerini validate et ve params objesine ekle
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

    // ERSIN Loading göster
    if (settingsSaveLoader) {
        settingsSaveLoader.style.display = 'inline-block';
        settingsSaveLoader.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Kaydediliyor...';
    }

    // ERSIN API'ye kaydet
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
        
        // ERSIN Response status kontrolü (200 OK ise başarılı)
        if (status === 200 && data.message) {
            // ERSIN Restart durumunu kontrol et
            if (data.restart_required || data.restart_initiated) {
                // ERSIN Production mode - restart başlatıldı
                showToast('Başarılı', data.message || 'Analiz parametreleri kaydedildi. Sistem yeniden başlatılıyor...', 'success');
                
                // ERSIN Restart sonrası sayfa yenile
                if (data.restart_initiated) {
                    // ERSIN Restart sonrası eski UI state'in (uploadedFiles / overall progress) kalmaması için
                    // ERSIN local restore mekanizmasını bir seferlik devre dışı bırak.
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
                // ERSIN Development mode - normal kayıt
                showToast('Başarılı', data.message || 'Analiz parametreleri kaydedildi.', 'success');
                
                // ERSIN WebSocket yeniden bağlan (development mode için)
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

/* ERSIN Aciklama. */
export function getSocketConnection() {
    return socketioClient;
}

/* ERSIN Aciklama. */
export function isSocketConnected() {
    return socketioClient && socketioClient.connected;
}

/* ERSIN Aciklama. */
export function emitSocketEvent(eventName, data) {
    if (socketioClient && socketioClient.connected) {
        const nativeSocket = socketioClient.socket || socketioClient;
        
        // ERSIN join_analysis için özel format (backend dict bekliyor)
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