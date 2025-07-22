/**
 * WebSocket Client - Temiz ve minimal implementasyon
 */

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 1000; // Sınırsız reconnect için çok yüksek değer
        this.reconnectDelay = 1000; // 1 saniye
        this.pingInterval = null; // Otomatik ping için interval
    }

    // WebSocket bağlantısını başlat
    connect() {
        console.log('[WebSocket] Bağlantı başlatılıyor...');
        
        try {
            this.socket = io({
                transports: ['websocket'],
                upgrade: false,
                timeout: 20000
            });

            this.setupEventListeners();

            // Otomatik ping başlat
            this.startAutoPing();
        } catch (error) {
            console.error('[WebSocket] Bağlantı hatası:', error);
        }
    }

    // Event listener'ları kur
    setupEventListeners() {
        console.log('🔥🔥🔥 SETTING UP EVENT LISTENERS - Socket object:', this.socket);
        console.log('🔥🔥🔥 SETTING UP EVENT LISTENERS - Socket ID:', this.socket.id);
        console.log('🔥🔥🔥 SETTING UP EVENT LISTENERS - Socket connected:', this.socket.connected);
        
        // Bağlantı olayları
        this.socket.on('connect', () => {
            console.log('[WebSocket] Bağlantı başarılı - ID:', this.socket.id);
            console.log('🔥🔥🔥 CONNECT EVENT RECEIVED - Socket object:', this.socket);
            this.connected = true;
            this.reconnectAttempts = 0;
            
            // Event listeners kuruldu, test ping gönder
            console.log('🔥 [WebSocket] Event listeners kuruldu, test eventi emit ediliyor...');
            console.log('🔥 [DEBUG] About to emit ping event...');
            console.log('🔥🔥🔥 SOCKET STATE BEFORE PING:', {
                connected: this.socket.connected,
                id: this.socket.id,
                disconnected: this.socket.disconnected
            });
            try {
                this.socket.emit('ping', 'test');
                console.log('🔥 [DEBUG] Ping event emitted successfully!');
            } catch (error) {
                console.error('❌ [DEBUG] Error emitting ping:', error);
            }
            
            this.onConnected();
        });
        console.log('🔥🔥🔥 CONNECT LISTENER REGISTERED');

        this.socket.on('disconnect', (reason) => {
            console.log('[WebSocket] Bağlantı kesildi - Sebep:', reason);
            console.log('🔥🔥🔥 DISCONNECT EVENT RECEIVED - Reason:', reason);
            this.connected = false;
            this.onDisconnected(reason);
            
            // Otomatik yeniden bağlantı
            if (reason !== 'io client disconnect') {
                this.attemptReconnect();
            }
        });
        console.log('🔥🔥🔥 DISCONNECT LISTENER REGISTERED');

        this.socket.on('connect_error', (error) => {
            console.error('[WebSocket] Bağlantı hatası:', error);
            console.log('🔥🔥🔥 CONNECT_ERROR EVENT RECEIVED - Error:', error);
            this.onConnectError(error);
        });
        console.log('🔥🔥🔥 CONNECT_ERROR LISTENER REGISTERED');

        // Sistem olayları
        this.socket.on('connected', (data) => {
            console.log('[WebSocket] Server onayı:', data);
            console.log('🔥🔥🔥 CONNECTED EVENT RECEIVED - Data:', data);
        });
        console.log('🔥🔥🔥 CONNECTED LISTENER REGISTERED');

        this.socket.on('pong', (data) => {
            console.log('🔥 [WebSocket] PONG received - event listeners çalışıyor!:', data);
            console.log('🔥🔥🔥 PONG EVENT RECEIVED - Data:', data);
        });
        console.log('🔥🔥🔥 PONG LISTENER REGISTERED');

        // Analiz olayları
        this.socket.on('analysis_progress', (data) => {
            const analysisId = data.analysis_id;
            let fileId = window.fileAnalysisMap.get(analysisId);
            
            // Eğer fileId bulunamazsa, uploadedFiles dizisinden ara
            if (!fileId && window.uploadedFiles) {
                // Hem analysisId hem de analysis_id alanlarını kontrol et
                const file = window.uploadedFiles.find(f => 
                    f.analysisId === analysisId || 
                    f.analysis_id === analysisId
                );
                if (file) {
                    fileId = file.id;
                    // Eksikse map'e ekle
                    window.fileAnalysisMap.set(analysisId, fileId);
                    console.warn('[WebSocket] fileId fallback ile bulundu ve map eklendi! analysisId:', analysisId, 'fileId:', fileId);
                }
            }

            
            // Hâlâ bulunamazsa, DOM'dan ara (race condition için son çare)
            if (!fileId) {
                const fileCards = document.querySelectorAll('.file-card');
                for (const card of fileCards) {
                    if (card.dataset.analysisId === analysisId) {
                        fileId = card.id;
                        window.fileAnalysisMap.set(analysisId, fileId);
                        console.warn('[WebSocket] fileId DOM fallback ile bulundu! analysisId:', analysisId, 'fileId:', fileId);
                        break;
                    }
                }
            }
            
            // Son çare: temp_ mapping'leri kontrol et (immediate mapping için)
            if (!fileId) {
                for (const [key, value] of window.fileAnalysisMap.entries()) {
                    if (key.startsWith('temp_')) {
                        // Bu dosya için analiz başlatılmış, gerçek analysis ID ile update et
                        window.fileAnalysisMap.delete(key);
                        window.fileAnalysisMap.set(analysisId, value);
                        fileId = value;
                        console.log('[WebSocket] Immediate mapping kullanıldı:', key, '→', analysisId, '→', fileId);
                        break;
                    }
                }
            }
            
            if (!fileId) {
                console.info('[WebSocket] analysis_progress: fileId henüz mapping\'e eklenmemiş, progress queue\'ya alınıyor. analysisId:', analysisId);
                // Progress'i queue'ya al, daha sonra mapping geldiğinde uygula
                if (!window.pendingProgress) window.pendingProgress = new Map();
                if (!window.pendingProgress.has(analysisId)) {
                    window.pendingProgress.set(analysisId, []);
                }
                window.pendingProgress.get(analysisId).push(data);
                console.info('[WebSocket] Progress queue\'ya eklendi. Toplam bekleyen:', window.pendingProgress.get(analysisId).length);
                return;
            }
            updateFileStatus(fileId, 'processing', data.progress, data.message);
        });
        console.log('🔥🔥🔥 ANALYSIS_PROGRESS LISTENER REGISTERED');
        
        this.socket.on('joined_analysis', (data) => {
            console.log('🔥 [WebSocket] JOINED_ANALYSIS received:', data);
            console.log('🔥🔥🔥 JOINED_ANALYSIS EVENT RECEIVED - Data:', data);
            // Oda katılımı onayı alındıktan sonra analysis_ready event'i gönder
            if (data && data.analysis_id) {
                setTimeout(() => {
                    this.socket.emit('analysis_ready', { analysis_id: data.analysis_id });
                    console.log('🔥 [WebSocket] analysis_ready event emitted:', data.analysis_id);
                }, 200); // 200ms gecikme ile güvenli oda katılımı
            }
        });
        console.log('🔥🔥🔥 JOINED_ANALYSIS LISTENER REGISTERED');

        this.socket.on('analysis_completed', (data) => {
            const analysisId = data.analysis_id;
            console.log('🎉 [WebSocket] ANALYSIS_COMPLETED received:', data);
            
            // Doğru mapping: analysisId → fileId
            let fileId = null;
            if (window.fileAnalysisMap && window.fileAnalysisMap.has(analysisId)) {
                fileId = window.fileAnalysisMap.get(analysisId);
                console.log('🎉 [WebSocket] FileId bulundu mapping\'te:', analysisId, '→', fileId);
            }
            
            // Fallback: uploadedFiles'dan ara
            if (!fileId && window.uploadedFiles) {
                const file = window.uploadedFiles.find(f => f.analysisId === analysisId);
                if (file) {
                    fileId = file.id;
                    console.log('🎉 [WebSocket] FileId bulundu uploadedFiles\'ta:', file.id);
                }
            }
            
                            if (fileId) {
                    console.log('🎉 [WebSocket] Analysis completed - updateFileStatus çağrılıyor:', fileId);
                    updateFileStatus(fileId, 'completed', 100, data.message || 'Analiz tamamlandı');
                    
                    // 🔥 CRITICAL: Analiz sonuçlarını çek ve göster!
                    console.log('🎉 [WebSocket] Analysis completed - getAnalysisResults çağrılıyor:', fileId, analysisId);
                    setTimeout(() => {
                        if (typeof getAnalysisResults === 'function') {
                            getAnalysisResults(fileId, analysisId);
                        } else {
                            console.error('🔥 getAnalysisResults function bulunamadı!');
                        }
                    }, 500); // Backend'de tüm işlemlerin bitmesi için kısa gecikme
                } else {
                    console.warn('⚠️ [WebSocket] analysis_completed: fileId bulunamadı!', {
                        analysisId,
                        fileAnalysisMap: window.fileAnalysisMap,
                        uploadedFiles: window.uploadedFiles
                    });
                }
        });

        // Tüm event'leri yakala (debug amaçlı)
        this.socket.onAny((eventName, ...args) => {
            console.log('🔥 [WebSocket] ANY EVENT received:', eventName, args);
            
            // Analysis progress özel debug
            if (eventName === 'analysis_progress') {
                console.log('🚨🚨🚨 ANALYSIS_PROGRESS DETECTED IN ANY LISTENER!', args);
                console.log('🚨 Event data:', args[0]);
                // Manuel olarak onAnalysisProgress çağır
                if (args[0]) {
                    this.onAnalysisProgress(args[0]);
                }
            }
        });

        // 🔥 joined_analysis confirmation event'ini dinle
        this.socket.on('joined_analysis', (data) => {
            console.log('🔥 [WebSocket] JOINED_ANALYSIS confirmation received:', data);
        });

        // Eğitim olayları
        this.socket.on('training_started', (data) => {
            console.log('[WebSocket] Training started:', data);
            this.onTrainingStarted(data);
        });

        this.socket.on('training_progress', (data) => {
            console.log('[WebSocket] Training progress:', data);
            this.onTrainingProgress(data);
        });

        this.socket.on('training_completed', (data) => {
            console.log('🔥 [WebSocket] Training completed RECEIVED:', data);
            this.onTrainingCompleted(data);
        });
        console.log('🔥🔥🔥 TRAINING_COMPLETED LISTENER REGISTERED');
        
        console.log('🔥🔥🔥 ALL EVENT LISTENERS SETUP COMPLETED!');
        console.log('🔥🔥🔥 FINAL SOCKET STATE:', {
            connected: this.socket.connected,
            id: this.socket.id,
            disconnected: this.socket.disconnected
        });
    }

    // Yeniden bağlantı deneme
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[WebSocket] Maksimum yeniden bağlantı denemesi aşıldı');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`[WebSocket] Yeniden bağlantı deneniyor... ${this.reconnectAttempts}/${this.maxReconnectAttempts} (${delay}ms)`);
        
        setTimeout(() => {
            if (!this.connected) {
                this.connect();
            }
        }, delay);
    }

    // Ping gönder
    ping() {
        if (this.connected) {
            this.socket.emit('ping');
        }
    }

    // Otomatik ping başlat
    startAutoPing() {
        if (this.pingInterval) clearInterval(this.pingInterval);
        this.pingInterval = setInterval(() => {
            if (this.connected && this.socket) {
                this.socket.emit('ping', 'auto');
                console.log('[WebSocket] Otomatik ping gönderildi');
            }
        }, 30000); // 30 saniyede bir ping
    }

    // Otomatik ping'i durdur
    stopAutoPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    // Analiz room'una katıl
    joinAnalysis(analysisId) {
        console.log('🔥 [DEBUG] joinAnalysis called with:', analysisId);
        console.log('🔥 [DEBUG] Connected status:', this.connected);
        console.log('🔥 [DEBUG] Socket object:', this.socket);
        console.log('🔥🔥🔥 SOCKET STATE IN JOIN_ANALYSIS:', {
            connected: this.socket.connected,
            id: this.socket.id,
            disconnected: this.socket.disconnected,
            transport: this.socket.io.engine.transport.name
        });
        
        if (!this.connected || !this.socket) {
            console.error('❌ [WebSocket] Socket bağlı değil, join_analysis gönderilemedi');
            return;
        }
        
        console.log('🔥 [DEBUG] About to emit join_analysis event...');
        try {
            this.socket.emit('join_analysis', { analysis_id: analysisId });
            console.log('🔥 [DEBUG] join_analysis event emitted successfully!');
            console.log('[WebSocket] Analiz odasına katılındı:', `analysis_${analysisId}`);
        } catch (error) {
            console.error('❌ [DEBUG] Error emitting join_analysis:', error);
        }
    }

    // Eğitim odasına katıl
    joinTraining(sessionId) {
        if (this.connected) {
            this.socket.emit('join_training', { session_id: sessionId });
            console.log(`[WebSocket] Eğitim odasına katılındı: training_${sessionId}`);
        }
    }

    // Bağlantıyı kapat
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
            this.stopAutoPing();
        }
    }

    // Event handler'lar - override edilebilir
    onConnected() {
        // UI güncellemeleri
        this.updateConnectionStatus('connected', 'WebSocket bağlantısı aktif');
    }

    onDisconnected(reason) {
        // UI güncellemeleri
        this.updateConnectionStatus('disconnected', `Bağlantı kesildi: ${reason}`);
    }

    onConnectError(error) {
        // UI güncellemeleri
        this.updateConnectionStatus('error', `Bağlantı hatası: ${error.message || error}`);
    }

    onAnalysisProgress(data) {
        // Analysis progress UI güncellemeleri
        const { analysis_id, progress, message, status } = data;
        
        // Progress bar güncelle
        const progressBar = document.querySelector(`[data-analysis-id="${analysis_id}"] .progress-bar`);
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            progressBar.textContent = `${progress}%`;
        }

        // Status message güncelle
        const statusMessage = document.querySelector(`[data-analysis-id="${analysis_id}"] .status-message`);
        if (statusMessage) {
            statusMessage.textContent = message;
        }

        console.log(`[WebSocket] Analysis ${analysis_id} progress: ${progress}% - ${message}`);
    }

    onAnalysisCompleted(data) {
        // Analysis completion UI güncellemeleri
        const { analysis_id, message } = data;
        
        // Progress bar'ı 100% yap
        const progressBar = document.querySelector(`[data-analysis-id="${analysis_id}"] .progress-bar`);
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.classList.remove('progress-bar-animated');
            progressBar.classList.add('bg-success');
        }

        // Status message güncelle
        const statusMessage = document.querySelector(`[data-analysis-id="${analysis_id}"] .status-message`);
        if (statusMessage) {
            statusMessage.textContent = message;
            statusMessage.classList.add('text-success');
        }

        console.log(`[WebSocket] Analysis ${analysis_id} completed: ${message}`);
        
        // Sayfayı yenile (sonuçları göstermek için)
        setTimeout(() => {
            window.location.reload();
        }, 2000);
    }

    onTrainingStarted(data) {
        // Training başlatıldı UI güncellemeleri
        const { session_id, model_type, total_samples, message } = data;
        
        // Modal training status güncelle
        this.showModalTrainingStatus(message, 'info');
        
        // Progress div'i görünür yap
        const modalProgressDiv = document.getElementById('modal-training-progress');
        if (modalProgressDiv) {
            modalProgressDiv.style.display = 'block';
        }

        console.log(`[WebSocket] Training started: ${model_type} model with ${total_samples} samples`);
    }

    onTrainingProgress(data) {
        // Training progress UI güncellemeleri
        const { session_id, current_epoch, total_epochs, progress, metrics } = data;
        
        // Modal training progress güncelle
        const modalProgressBar = document.getElementById('modal-progress-bar');
        const modalCurrentEpoch = document.getElementById('modal-current-epoch');
        const modalCurrentLoss = document.getElementById('modal-current-loss');
        
        if (modalProgressBar) {
            modalProgressBar.style.width = `${progress}%`;
            modalProgressBar.setAttribute('aria-valuenow', Math.round(progress));
        }
        
        if (modalCurrentEpoch) {
            modalCurrentEpoch.textContent = `${current_epoch}/${total_epochs}`;
        }
        
        if (modalCurrentLoss && metrics && metrics.loss) {
            modalCurrentLoss.textContent = metrics.loss.toFixed(4);
        }

        console.log(`[WebSocket] Training ${session_id} progress: ${current_epoch}/${total_epochs} (${Math.round(progress)}%)`);
    }

    onTrainingCompleted(data) {
        // Training completion UI güncellemeleri
        const { session_id, model_version, metrics } = data;
        
        // Modal progress'i 100% yap
        const modalProgressBar = document.getElementById('modal-progress-bar');
        if (modalProgressBar) {
            modalProgressBar.style.width = '100%';
            modalProgressBar.setAttribute('aria-valuenow', 100);
            modalProgressBar.classList.add('bg-success');
        }

        // Success mesajı göster
        this.showModalTrainingStatus(`Eğitim tamamlandı! Model: ${model_version}`, 'success');

        console.log(`[WebSocket] Training ${session_id} completed: ${model_version}`);
    }

    onTrainingError(data) {
        // Training error UI güncellemeleri
        const { session_id, error } = data;
        
        // Error mesajı göster
        this.showModalTrainingStatus(`Eğitim hatası: ${error}`, 'danger');

        console.error(`[WebSocket] Training ${session_id} error: ${error}`);
    }

    // UI Helper metodlar
    updateConnectionStatus(status, message) {
        const statusElement = document.getElementById('websocket-status');
        if (statusElement) {
            statusElement.className = `websocket-status ${status}`;
            statusElement.textContent = message;
        }
    }

    showModalTrainingStatus(message, type = 'info') {
        const statusDiv = document.getElementById('modal-training-status');
        if (statusDiv) {
            statusDiv.className = `alert alert-${type}`;
            statusDiv.textContent = message;
            statusDiv.style.display = 'block';
        }
    }

    // 🔥 Once method - tek seferlik event listener
    once(event, callback) {
        if (this.socket) {
            this.socket.once(event, callback);
        } else {
            console.warn('[WebSocket] Socket mevcut değil, once event eklenemiyor');
        }
    }

    // 🔥 Ping method - test için
    ping() {
        if (this.socket && this.connected) {
            this.socket.emit('ping', 'Client ping');
            console.log('[WebSocket] Ping gönderildi');
        } else {
            console.warn('[WebSocket] Socket bağlı değil, ping gönderilemedi');
        }
    }
}

// 🔥 WebSocket client class hazır - instance main.js'de oluşturulacak
// (Dublicate connection önlemek için burada instance oluşturmuyoruz)

// Global state'i ana dosya ile paylaş
if (!window.fileAnalysisMap) window.fileAnalysisMap = new Map();
if (!window.uploadedFiles) window.uploadedFiles = [];

// Sayfa yüklendiğinde setup
document.addEventListener('DOMContentLoaded', function() {
    console.log('[WebSocket] DOM yüklendi, WebSocket başlatılıyor...');
    // Instance oluşturma main.js'e taşındı
    
    // Ping test butonu
    window.testWebSocket = function() {
        console.log('[WebSocket] Test ping gönderiliyor...');
        if (window.socketioClient) {
            window.socketioClient.ping();
        } else {
            console.warn('[WebSocket] Client bulunamadı!');
        }
    };
});

// Sayfa kapanırken bağlantıyı kapat
window.addEventListener('beforeunload', function() {
    if (window.socketioClient) {
        window.socketioClient.disconnect();
    }
}); 