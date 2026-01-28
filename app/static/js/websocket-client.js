/* ERSIN Aciklama. */

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;  // ERSIN Uzun analizler için daha fazla deneme
        this.reconnectDelay = 1000;  // ERSIN 1 saniye
        this.pingInterval = null;  // ERSIN Otomatik ping için interval
        this.silentMode = false;  // ERSIN Silent mode flag
        this.backgroundMode = false;  // ERSIN Browser arka plan modu
    }

    // ERSIN WebSocket bağlantısını başlat
    connect() {
        console.log('[WebSocket] Bağlantı başlatılıyor...');
        
        try {
            // ERSIN F5 load balancer desteği: polling fallback ekle, upgrade'e izin ver
            // ERSIN F5 path-based routing için path environment variable'dan alınabilir
            const socketPath = window.SOCKETIO_PATH || '/socket.io/';
            
            // ERSIN F5 arkasında mutlak URL kullan - window.location.origin kullan
            const socketUrl = window.SOCKETIO_URL || window.location.origin;
            
            console.log('[WebSocket] Bağlantı ayarları:', {
                url: socketUrl,
                path: socketPath,
                transports: ['websocket', 'polling']
            });
            
            this.socket = io(socketUrl, {
                transports: ['polling', 'websocket'],  // ERSIN F5 arkasında önce polling dene, sonra websocket upgrade
                upgrade: true,  // ERSIN F5 load balancer için upgrade'e izin ver
                timeout: 20000,
                reconnection: true,
                reconnectionAttempts: 10,  // ERSIN Uzun analizler için daha fazla deneme
                reconnectionDelay: 1000,
                reconnectionDelayMax: 10000,
                maxHttpBufferSize: 1e6,
                pingTimeout: 720000,  // ERSIN 12 dakika - sunucu ile sync
                pingInterval: 60000,  // ERSIN 1 dakika - sunucu ile sync
                autoConnect: true,
                forceNew: false,
                path: socketPath,  // ERSIN F5 path-based routing için yapılandırılabilir path
                rememberUpgrade: false,  // ERSIN Her seferinde polling'den başla, F5 için daha güvenli
                withCredentials: true  // ERSIN Session cookie'leri gönder
            });
            
            console.log('[WebSocket] SocketIO instance oluşturuldu:', {
                socket: this.socket,
                id: this.socket.id,
                connected: this.socket.connected,
                io: this.socket.io,
                transport: this.socket.io?.engine?.transport?.name
            });
            
            // ERSIN F5 arkasında bağlantı gecikmesi olabilir, manuel kontrol ekle
            setTimeout(() => {
                if (!this.socket.connected) {
                    console.warn('[WebSocket] Bağlantı kurulmadı, durum kontrolü yapılıyor...');
                    console.log('[WebSocket] Socket durumu:', {
                        connected: this.socket.connected,
                        disconnected: this.socket.disconnected,
                        id: this.socket.id,
                        io: this.socket.io,
                        transport: this.socket.io?.engine?.transport?.name,
                        readyState: this.socket.io?.engine?.readyState
                    });
                    
                    // ERSIN Engine durumunu kontrol et
                    if (this.socket.io && this.socket.io.engine) {
                        console.log('[WebSocket] Engine durumu:', {
                            readyState: this.socket.io.engine.readyState,
                            transport: this.socket.io.engine.transport?.name,
                            upgrading: this.socket.io.engine.upgrading
                        });
                    }
                    
                    // ERSIN Manuel connect denemesi - sadece disconnected ise
                    if (this.socket.disconnected) {
                        console.log('[WebSocket] Manuel connect() çağrılıyor...');
                        this.socket.connect();
                    }
                }
            }, 2000);

            this.setupEventListeners();

            // ERSIN Otomatik ping başlat
            this.startAutoPing();
        } catch (error) {
            console.error('[WebSocket] Bağlantı hatası:', error);
        }
    }

    // ERSIN Event listener'ları kur
    setupEventListeners() {
        console.log('🔥🔥🔥 SETTING UP EVENT LISTENERS - Socket object:', this.socket);
        console.log('🔥🔥🔥 SETTING UP EVENT LISTENERS - Socket ID:', this.socket.id);
        console.log('🔥🔥🔥 SETTING UP EVENT LISTENERS - Socket connected:', this.socket.connected);
        
        // ERSIN Bağlantı olayları
        this.socket.on('connect', () => {
            console.log('[WebSocket] Bağlantı başarılı - ID:', this.socket.id);
            console.log('🔥🔥🔥 CONNECT EVENT RECEIVED - Socket object:', this.socket);
            this.connected = true;
            this.reconnectAttempts = 0;
            
            // ERSIN Event listeners kuruldu, test ping gönder
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
            
            // ERSIN Otomatik yeniden bağlantı
            if (reason !== 'io client disconnect') {
                this.attemptReconnect();
            }
        });
        console.log('🔥🔥🔥 DISCONNECT LISTENER REGISTERED');

        this.socket.on('connect_error', (error) => {
            console.error('[WebSocket] Bağlantı hatası:', error);
            console.error('[WebSocket] Bağlantı hatası detayları:', {
                message: error.message,
                description: error.description,
                context: error.context,
                type: error.type,
                transport: error.transport,
                socket: this.socket,
                io: this.socket.io
            });
            console.log('🔥🔥🔥 CONNECT_ERROR EVENT RECEIVED - Error:', error);
            this.onConnectError(error);
        });
        console.log('🔥🔥🔥 CONNECT_ERROR LISTENER REGISTERED');

        // ERSIN Sistem olayları
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

        // ERSIN Analiz olayları
        this.socket.on('analysis_started', (data) => {
            const analysisId = data.analysis_id;
            let fileId = data.file_id;
            let cardId = fileId;
            // ERSIN Önce fileIdToCardId mapping'ini dene
            if (window.fileIdToCardId && window.fileIdToCardId[String(fileId)]) {
                cardId = window.fileIdToCardId[String(fileId)];
            }
            // ERSIN Sonra eski mapping'leri dene
            if (!document.getElementById(cardId) && window.fileAnalysisMap && window.fileAnalysisMap.has(analysisId)) {
                cardId = window.fileAnalysisMap.get(analysisId);
            }
            if (cardId) {
                window.fileAnalysisMap.set(analysisId, cardId);
                // ERSIN analysis_started sadece kuyruğa eklendiği anlamına gelir - "Sırada" durumunda kalır
                updateFileStatus(cardId, 'queued', 0, data.message || 'Analiz kuyruğa eklendi');
                console.log('🚀 [WebSocket] Analysis started - kuyruğa eklendi (cardId):', cardId);
            } else {
                console.warn('[WebSocket] analysis_started: fileId bulunamadı!', data);
            }
        });
        console.log('🔥🔥🔥 ANALYSIS_STARTED LISTENER REGISTERED');

        this.socket.on('analysis_progress', (data) => {
            const analysisId = data.analysis_id;
            let fileId = data.file_id || window.fileAnalysisMap.get(analysisId);
            let cardId = fileId;
            if (window.fileIdToCardId && window.fileIdToCardId[String(fileId)]) {
                cardId = window.fileIdToCardId[String(fileId)];
            }
            if (!document.getElementById(cardId) && window.fileAnalysisMap && window.fileAnalysisMap.has(analysisId)) {
                cardId = window.fileAnalysisMap.get(analysisId);
            }
            if (!cardId && window.uploadedFiles) {
                const file = window.uploadedFiles.find(f => f.analysisId === analysisId || f.analysis_id === analysisId);
                if (file) {
                    cardId = file.id;
                    window.fileAnalysisMap.set(analysisId, cardId);
                }
            }
            if (!cardId) {
                // ERSIN Progress'i queue'ya al, daha sonra mapping geldiğinde uygula
                if (!window.pendingProgress) window.pendingProgress = new Map();
                if (!window.pendingProgress.has(analysisId)) {
                    window.pendingProgress.set(analysisId, []);
                }
                window.pendingProgress.get(analysisId).push(data);
                return;
            }
            updateFileStatus(cardId, 'processing', data.progress, data.message, null);
        });
        console.log('🔥🔥🔥 ANALYSIS_PROGRESS LISTENER REGISTERED');
        
        this.socket.on('joined_analysis', (data) => {
            console.log('🔥 [WebSocket] JOINED_ANALYSIS received:', data);
            console.log('🔥🔥🔥 JOINED_ANALYSIS EVENT RECEIVED - Data:', data);
            // ERSIN Oda katılımı onayı alındıktan sonra analysis_ready event'i gönder
            if (data && data.analysis_id) {
                setTimeout(() => {
                    this.socket.emit('analysis_ready', { analysis_id: data.analysis_id });
                    console.log('🔥 [WebSocket] analysis_ready event emitted:', data.analysis_id);
                }, 200);  // ERSIN 200ms gecikme ile güvenli oda katılımı
            }
        });
        console.log('🔥🔥🔥 JOINED_ANALYSIS LISTENER REGISTERED');

        this.socket.on('analysis_completed', (data) => {
            const analysisId = data.analysis_id;
            let fileId = data.file_id;
            let cardId = fileId;
            if (window.fileIdToCardId && window.fileIdToCardId[String(fileId)]) {
                cardId = window.fileIdToCardId[String(fileId)];
            }
            if (!document.getElementById(cardId) && window.fileAnalysisMap && window.fileAnalysisMap.has(analysisId)) {
                cardId = window.fileAnalysisMap.get(analysisId);
            }
            if (!cardId && window.uploadedFiles) {
                const file = window.uploadedFiles.find(f => f.analysisId === analysisId);
                if (file) {
                    cardId = file.id;
                }
            }
            if (cardId) {
                updateFileStatus(cardId, 'completed', 100, data.message || 'Analiz tamamlandı', null);
                setTimeout(() => {
                    if (typeof getAnalysisResults === 'function') {
                        getAnalysisResults(cardId, analysisId);
                    }
                }, 500);
            } else {
                console.warn('⚠️ [WebSocket] analysis_completed: fileId bulunamadı!', {
                    analysisId,
                    fileAnalysisMap: window.fileAnalysisMap,
                    uploadedFiles: window.uploadedFiles
                });
            }
        });

        // ERSIN Kuyruk durumu olayları
                this.socket.on('queue_status', (data) => {
            console.log('📊 [WebSocket] QUEUE_STATUS received:', data);

            // ERSIN Kuyruk bilgilerini güncelle (eğer UI'da gösteriliyorsa)
            if (data) {
                window.queueStatus = data;
                
                // ERSIN 🎯 BUTTON STATE UPDATE - Queue durumuna göre butonları güncelle
                if (window.updateAnalysisParamsButtonStateWithQueue) {
                    window.updateAnalysisParamsButtonStateWithQueue(data);
                }
                
                // ERSIN updateQueueStatus fonksiyonunu çağır (main.js'te)
                if (typeof updateQueueStatus === 'function') {
                    updateQueueStatus(data);
                    console.log('📊 [WebSocket] Queue status UI güncellendi');
                }
            }
        });
        console.log('🔥🔥🔥 QUEUE_STATUS LISTENER REGISTERED');

        // ERSIN Tüm event'leri yakala (debug amaçlı)
        this.socket.onAny((eventName, ...args) => {
            console.log('🔥 [WebSocket] ANY EVENT received:', eventName, args);
            
            // ERSIN Analysis progress özel debug
            if (eventName === 'analysis_progress') {
                console.log('🚨🚨🚨 ANALYSIS_PROGRESS DETECTED IN ANY LISTENER!', args);
                console.log('🚨 Event data:', args[0]);
                // ERSIN Manuel olarak onAnalysisProgress çağır
                if (args[0]) {
                    this.onAnalysisProgress(args[0]);
                }
            }
        });

        // ERSIN 🔥 joined_analysis confirmation event'ini dinle
        this.socket.on('joined_analysis', (data) => {
            console.log('🔥 [WebSocket] JOINED_ANALYSIS confirmation received:', data);
        });

        // ERSIN Eğitim olayları
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

    // ERSIN Yeniden bağlantı deneme
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.warn('[WebSocket] Maksimum yeniden bağlantı denemesi aşıldı');
            // ERSIN Silent mode yerine daha uzun interval ile denemeye devam et
            setTimeout(() => {
                this.reconnectAttempts = 0;  // ERSIN Reset attempts
                if (!this.connected) {
                    this.attemptReconnect();
                }
            }, 30000);  // ERSIN 30 saniye bekle
            return;
        }

        this.reconnectAttempts++;
        const baseDelay = this.backgroundMode ? 5000 : this.reconnectDelay;  // ERSIN Arka planda daha uzun delay
        const delay = Math.min(baseDelay * this.reconnectAttempts, 15000);  // ERSIN Max 15 saniye delay
        
        const mode = this.backgroundMode ? '(arka plan)' : '';
        console.log(`[WebSocket] Yeniden bağlantı deneniyor... ${this.reconnectAttempts}/${this.maxReconnectAttempts} (${delay}ms) ${mode}`);
        
        setTimeout(() => {
            if (!this.connected) {
                try {
                    this.connect();
                } catch (error) {
                    console.error('[WebSocket] Reconnect sırasında hata:', error);
                }
            }
        }, delay);
    }

    // ERSIN Ping gönder
    ping() {
        if (this.connected) {
            this.socket.emit('ping');
        }
    }

    // ERSIN Otomatik ping başlat
    startAutoPing() {
        if (this.pingInterval) clearInterval(this.pingInterval);
        this.pingInterval = setInterval(() => {
            if (this.connected && this.socket) {
                this.socket.emit('ping', 'auto');
                const mode = this.backgroundMode ? '(arka plan)' : '';
                console.log(`[WebSocket] Otomatik ping gönderildi ${mode}`);
            }
        }, 45000);  // ERSIN 45 saniyede bir ping (sunucu 60s interval'ından biraz önce)
    }

    // ERSIN Otomatik ping'i durdur
    stopAutoPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    // ERSIN Analiz room'una katıl
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

    // ERSIN Eğitim odasına katıl
    joinTraining(sessionId) {
        if (this.connected) {
            this.socket.emit('join_training', { session_id: sessionId });
            console.log(`[WebSocket] Eğitim odasına katılındı: training_${sessionId}`);
        }
    }

    // ERSIN Bağlantıyı kapat
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
            this.stopAutoPing();
        }
    }

    // ERSIN Event handler'lar - override edilebilir
    onConnected() {
        // ERSIN UI güncellemeleri
        this.updateConnectionStatus('connected', 'WebSocket bağlantısı aktif');
    }

    onDisconnected(reason) {
        // ERSIN UI güncellemeleri
        this.updateConnectionStatus('disconnected', `Bağlantı kesildi: ${reason}`);
    }

    onConnectError(error) {
        // ERSIN UI güncellemeleri
        this.updateConnectionStatus('error', `Bağlantı hatası: ${error.message || error}`);
    }

    onAnalysisProgress(data) {
        // ERSIN Analysis progress UI güncellemeleri
        const { analysis_id, progress, message, status } = data;
        
        // ERSIN Progress bar güncelle
        const progressBar = document.querySelector(`[data-analysis-id="${analysis_id}"] .progress-bar`);
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            // ERSIN textContent kaldırıldı - sadece visual bar yeterli
        }

        // ERSIN Status message güncelle
        const statusMessage = document.querySelector(`[data-analysis-id="${analysis_id}"] .status-message`);
        if (statusMessage) {
            statusMessage.textContent = message;
        }

        console.log(`[WebSocket] Analysis ${analysis_id} progress: ${progress}% - ${message}`);
    }

    onAnalysisCompleted(data) {
        // ERSIN Analysis completion UI güncellemeleri
        const { analysis_id, message } = data;
        
        // ERSIN Progress bar'ı 100% yap
        const progressBar = document.querySelector(`[data-analysis-id="${analysis_id}"] .progress-bar`);
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.classList.remove('progress-bar-animated');
            progressBar.classList.add('bg-success');
        }

        // ERSIN Status message güncelle
        const statusMessage = document.querySelector(`[data-analysis-id="${analysis_id}"] .status-message`);
        if (statusMessage) {
            statusMessage.textContent = message;
            statusMessage.classList.add('text-success');
        }

        console.log(`[WebSocket] Analysis ${analysis_id} completed: ${message}`);
        
        // ERSIN Sayfayı yenile (sonuçları göstermek için)
        setTimeout(() => {
            window.location.reload();
        }, 2000);
    }

    onTrainingStarted(data) {
        // ERSIN Training başlatıldı UI güncellemeleri
        const { session_id, model_type, total_samples, message } = data;
        
        // ERSIN Modal training status güncelle
        this.showModalTrainingStatus(message, 'info');
        
        // ERSIN Progress div'i görünür yap
        const modalProgressDiv = document.getElementById('modal-training-progress');
        if (modalProgressDiv) {
            modalProgressDiv.style.display = 'block';
        }

        // ERSIN Training istatistiklerini temizle (özellikle CLIP ensemble için)
        this.clearTrainingStats();

        console.log(`[WebSocket] Training started: ${model_type} model with ${total_samples} samples`);
    }

    clearTrainingStats() {
        // ERSIN Training istatistiklerini "-" ile sıfırla
        const epochEl = document.getElementById('modal-current-epoch');
        const lossEl = document.getElementById('modal-current-loss');
        const maeEl = document.getElementById('modal-current-mae');
        const durationEl = document.getElementById('modal-training-duration');
        
        if (epochEl) epochEl.textContent = '-';
        if (lossEl) lossEl.textContent = '-';
        if (maeEl) maeEl.textContent = '-';
        if (durationEl) durationEl.textContent = '-';
        
        // ERSIN Progress bar'ı da sıfırla
        const modalProgressBar = document.getElementById('modal-progress-bar');
        if (modalProgressBar) {
            modalProgressBar.style.width = '0%';
            modalProgressBar.setAttribute('aria-valuenow', 0);
            modalProgressBar.classList.remove('bg-success');
        }
        
        console.log('🧹 Training stats temizlendi');
    }

    onTrainingProgress(data) {
        // ERSIN Training progress UI güncellemeleri
        const { session_id, current_epoch, total_epochs, progress, metrics } = data;
        
        // ERSIN Modal training progress güncelle
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
        // ERSIN Training completion UI güncellemeleri
        const { session_id, model_path, metrics } = data;
        
        // ERSIN Modal progress'i 100% yap
        const modalProgressBar = document.getElementById('modal-progress-bar');
        if (modalProgressBar) {
            modalProgressBar.style.width = '100%';
            modalProgressBar.setAttribute('aria-valuenow', 100);
            modalProgressBar.classList.add('bg-success');
        }

        // ERSIN Success mesajı göster
        this.showModalTrainingStatus(`Eğitim tamamlandı! Model: ${model_path}`, 'success');

        // ERSIN CLIP Ensemble metrics varsa istatistikleri güncelle
        if (metrics && model_path.includes('Content')) {
            this.updateClipEnsembleStats(metrics);
        }

        // ERSIN Modal'ı yenile (küçük delay ile database commit işlemini bekle)
        setTimeout(() => {
            if (window.initializeModelManagementModal) {
                console.log('🔄 Modal yenileniyor (CLIP training completed)...');
                window.initializeModelManagementModal();
            }
        }, 1000);  // ERSIN 1 saniye bekle

        console.log(`[WebSocket] Training ${session_id} completed: ${model_path}`, metrics);
    }

    updateClipEnsembleStats(metrics) {
        // ERSIN CLIP Ensemble için özel istatistik gösterimi
        console.log('🎯 CLIP Ensemble stats güncelleniyor:', metrics);
        
        const epochEl = document.getElementById('modal-current-epoch');
        const lossEl = document.getElementById('modal-current-loss');
        const maeEl = document.getElementById('modal-current-mae');
        const durationEl = document.getElementById('modal-training-duration');
        
        if (epochEl && metrics.total_content_corrections !== undefined) {
            epochEl.textContent = `${metrics.total_content_corrections} Düzeltme`;
        }
        
        if (lossEl && metrics.avg_confidence_adjustment !== undefined) {
            lossEl.textContent = `${parseFloat(metrics.avg_confidence_adjustment).toFixed(3)}`;
        }
        
        if (maeEl && metrics.total_confidence_adjustments !== undefined) {
            maeEl.textContent = `${metrics.total_confidence_adjustments} Ayar`;
        }
        
        if (durationEl && metrics.manual_corrections !== undefined && metrics.auto_corrections !== undefined) {
            durationEl.textContent = `${metrics.manual_corrections}M/${metrics.auto_corrections}A`;
        }
        
        console.log('✅ CLIP Ensemble stats güncellendi');
    }

    onTrainingError(data) {
        // ERSIN Training error UI güncellemeleri
        const { session_id, error } = data;
        
        // ERSIN Error mesajı göster
        this.showModalTrainingStatus(`Eğitim hatası: ${error}`, 'danger');

        console.error(`[WebSocket] Training ${session_id} error: ${error}`);
    }

    // ERSIN UI Helper metodlar
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

    // ERSIN 🔥 Once method - tek seferlik event listener
    once(event, callback) {
        if (this.socket) {
            this.socket.once(event, callback);
        } else {
            console.warn('[WebSocket] Socket mevcut değil, once event eklenemiyor');
        }
    }

    // ERSIN 🔥 Ping method - test için
    ping() {
        if (this.socket && this.connected) {
            this.socket.emit('ping', 'Client ping');
            console.log('[WebSocket] Ping gönderildi');
        } else {
            console.warn('[WebSocket] Socket bağlı değil, ping gönderilemedi');
        }
    }
}

// ERSIN 🔥 WebSocket client class hazır - instance main.js'de oluşturulacak
// ERSIN (Dublicate connection önlemek için burada instance oluşturmuyoruz)

// ERSIN Global state'i ana dosya ile paylaş
if (!window.fileAnalysisMap) window.fileAnalysisMap = new Map();
if (!window.uploadedFiles) window.uploadedFiles = [];

// ERSIN Sayfa yüklendiğinde setup
document.addEventListener('DOMContentLoaded', function() {
    console.log('[WebSocket] DOM yüklendi, WebSocket başlatılıyor...');
    // ERSIN Instance oluşturma main.js'e taşındı
    
    // ERSIN Ping test butonu
    window.testWebSocket = function() {
        console.log('[WebSocket] Test ping gönderiliyor...');
        if (window.socketioClient) {
            window.socketioClient.ping();
        } else {
            console.warn('[WebSocket] Client bulunamadı!');
        }
    };
});

// ERSIN Sayfa kapanırken bağlantıyı kapat
window.addEventListener('beforeunload', function() {
    if (window.socketioClient) {
        window.socketioClient.disconnect();
    }
}); 