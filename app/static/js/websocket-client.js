/**
 * WebSocket Client - Temiz ve minimal implementasyon
 */

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // 1 saniye
    }

    // WebSocket bağlantısını başlat
    connect() {
        console.log('[WebSocket] Bağlantı başlatılıyor...');
        
        try {
            this.socket = io({
                transports: ['websocket', 'polling'],
                upgrade: true,
                timeout: 20000
            });

            this.setupEventListeners();
        } catch (error) {
            console.error('[WebSocket] Bağlantı hatası:', error);
        }
    }

    // Event listener'ları kur
    setupEventListeners() {
        // Bağlantı olayları
        this.socket.on('connect', () => {
            console.log('[WebSocket] Bağlantı başarılı - ID:', this.socket.id);
            this.connected = true;
            this.reconnectAttempts = 0;
            this.onConnected();
        });

        this.socket.on('disconnect', (reason) => {
            console.log('[WebSocket] Bağlantı kesildi - Sebep:', reason);
            this.connected = false;
            this.onDisconnected(reason);
            
            // Otomatik yeniden bağlantı
            if (reason !== 'io client disconnect') {
                this.attemptReconnect();
            }
        });

        this.socket.on('connect_error', (error) => {
            console.error('[WebSocket] Bağlantı hatası:', error);
            this.onConnectError(error);
        });

        // Sistem olayları
        this.socket.on('connected', (data) => {
            console.log('[WebSocket] Server onayı:', data);
        });

        this.socket.on('pong', (data) => {
            console.log('[WebSocket] Pong alındı:', data);
        });

        // Analiz olayları
        this.socket.on('analysis_progress', (data) => {
            console.log('[WebSocket] Analysis progress:', data);
            this.onAnalysisProgress(data);
        });

        this.socket.on('analysis_completed', (data) => {
            console.log('[WebSocket] Analysis completed:', data);
            this.onAnalysisCompleted(data);
        });

        // Eğitim olayları
        this.socket.on('training_progress', (data) => {
            console.log('[WebSocket] Training progress:', data);
            this.onTrainingProgress(data);
        });

        this.socket.on('training_completed', (data) => {
            console.log('[WebSocket] Training completed:', data);
            this.onTrainingCompleted(data);
        });

        this.socket.on('training_error', (data) => {
            console.log('[WebSocket] Training error:', data);
            this.onTrainingError(data);
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

    // Analiz odasına katıl
    joinAnalysis(analysisId) {
        if (this.connected) {
            this.socket.emit('join_analysis', { analysis_id: analysisId });
            console.log(`[WebSocket] Analiz odasına katılındı: analysis_${analysisId}`);
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
}

// Global WebSocket client instance
window.wsClient = new WebSocketClient();

// Sayfa yüklendiğinde WebSocket'i başlat
document.addEventListener('DOMContentLoaded', function() {
    console.log('[WebSocket] DOM yüklendi, WebSocket başlatılıyor...');
    window.wsClient.connect();
    
    // Ping test butonu
    window.testWebSocket = function() {
        console.log('[WebSocket] Test ping gönderiliyor...');
        window.wsClient.ping();
    };
});

// Sayfa kapanırken bağlantıyı kapat
window.addEventListener('beforeunload', function() {
    if (window.wsClient) {
        window.wsClient.disconnect();
    }
}); 