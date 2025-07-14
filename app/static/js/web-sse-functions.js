// Web training için SSE bağlantısını kur
function setupWebSSEConnection(sessionId) {
    console.log(`[SSE] Setting up web SSE connection for session: ${sessionId}`);
    
    // Mevcut SSE bağlantısını kapat
    if (window.webEventSource) {
        window.webEventSource.close();
        console.log('[SSE] Closed existing web SSE connection');
    }
    
    // Yeni SSE bağlantısı oluştur
    const eventSource = new EventSource(`/api/model/training-events/${sessionId}`);
    window.webEventSource = eventSource;
    
    // UI elementlerini hazırla
    const startBtn = document.getElementById('startWebTrainingBtn');
    const stopBtn = document.getElementById('stopWebTrainingBtn');
    const statusDiv = document.getElementById('webTrainingStatus');
    const progressDiv = document.getElementById('webTrainingProgress');
    
    if (startBtn && stopBtn) {
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
    }
    
    if (progressDiv) {
        progressDiv.style.display = 'block';
    }
    
    eventSource.onopen = function() {
        console.log('[SSE] Web training SSE connection opened');
        if (statusDiv) {
            statusDiv.className = 'alert alert-info';
            document.getElementById('webTrainingMessage').textContent = 'SSE bağlantısı kuruldu, eğitim takibi başlatıldı...';
        }
    };
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('[SSE] Web training event received:', data);
            
            if (data.type === 'connected') {
                console.log('[SSE] Web connection confirmed for session:', data.session_id);
                if (statusDiv) {
                    statusDiv.className = 'alert alert-info';
                    document.getElementById('webTrainingMessage').textContent = 'Eğitim verisi işleniyor...';
                }
                
            } else if (data.type === 'training_started') {
                console.log('[SSE] Web training started:', data);
                trainingStartTime = Date.now();
                if (statusDiv) {
                    statusDiv.className = 'alert alert-info';
                    document.getElementById('webTrainingMessage').textContent = 
                        `Eğitim başladı! (${data.total_samples || 'N/A'} örnek)`;
                }
                
            } else if (data.type === 'training_progress') {
                console.log('[SSE] Web training progress:', data);
                updateWebTrainingProgressSSE(data);
                
            } else if (data.type === 'training_completed') {
                console.log('[SSE] Web training completed:', data);
                handleWebTrainingCompletedSSE(data);
                eventSource.close();
                
            } else if (data.type === 'training_error') {
                console.log('[SSE] Web training error:', data);
                handleWebTrainingErrorSSE(data);
                eventSource.close();
                
            } else if (data.type === 'session_ended') {
                console.log('[SSE] Web session ended:', data);
                if (statusDiv) {
                    statusDiv.className = 'alert alert-warning';
                    document.getElementById('webTrainingMessage').textContent = 'Eğitim oturumu sona erdi';
                }
                eventSource.close();
            }
            
        } catch (error) {
            console.error('[SSE] Error parsing web training event data:', error);
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('[SSE] Web training SSE connection error:', error);
        if (statusDiv) {
            statusDiv.className = 'alert alert-danger';
            document.getElementById('webTrainingMessage').textContent = 'SSE bağlantısında hata oluştu';
        }
        
        // UI sıfırla
        if (startBtn && stopBtn) {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        }
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
        
        eventSource.close();
    };
    
    // Otomatik kapatma (60 saniye)
    setTimeout(() => {
        if (eventSource.readyState !== EventSource.CLOSED) {
            console.log('[SSE] Auto-closing web SSE connection after timeout');
            eventSource.close();
        }
    }, 60000);
}

// SSE web progress güncellemesi
function updateWebTrainingProgressSSE(data) {
    console.log('[SSE] Updating web training progress:', data);
    
    const progressBar = document.getElementById('webProgressBar');
    const progressText = document.getElementById('webProgressText');
    const currentEpoch = document.getElementById('webCurrentEpoch');
    const currentLoss = document.getElementById('webCurrentLoss');
    const currentMAE = document.getElementById('webCurrentMAE');
    const currentR2 = document.getElementById('webCurrentR2');
    const trainingDuration = document.getElementById('webTrainingDuration');
    const trainingETA = document.getElementById('webTrainingETA');
    
    // Progress bar güncelleme
    const progressPercent = (data.current_epoch / data.total_epochs) * 100;
    if (progressBar) {
        progressBar.style.width = progressPercent + '%';
        progressBar.setAttribute('aria-valuenow', Math.round(progressPercent));
    }
    if (progressText) {
        progressText.textContent = Math.round(progressPercent) + '%';
    }
    
    // Epoch bilgisi
    if (currentEpoch) {
        currentEpoch.textContent = `${data.current_epoch}/${data.total_epochs}`;
    }
    
    // Metrics güncelleme
    if (currentLoss && data.current_loss !== undefined) {
        currentLoss.textContent = data.current_loss.toFixed(4);
    }
    if (currentMAE && data.current_mae !== undefined) {
        currentMAE.textContent = data.current_mae.toFixed(4);
    }
    if (currentR2 && data.current_r2 !== undefined) {
        currentR2.textContent = data.current_r2.toFixed(4);
    }
    
    // Süre hesaplaması
    if (trainingStartTime && trainingDuration) {
        const elapsed = (Date.now() - trainingStartTime) / 1000;
        trainingDuration.textContent = formatDuration(elapsed);
        
        // ETA hesaplaması
        if (trainingETA && data.current_epoch > 0) {
            const avgTimePerEpoch = elapsed / data.current_epoch;
            const remainingEpochs = data.total_epochs - data.current_epoch;
            const eta = avgTimePerEpoch * remainingEpochs;
            trainingETA.textContent = formatDuration(eta);
        }
    }
    
    // Durum mesajını güncelle
    document.getElementById('webTrainingMessage').textContent = 
        `Eğitim devam ediyor... Epoch ${data.current_epoch}/${data.total_epochs} (${Math.round(progressPercent)}%) - Loss: ${data.current_loss?.toFixed(4) || '-'}`;
}

// SSE web training tamamlandı
function handleWebTrainingCompletedSSE(data) {
    console.log('[SSE] Web training completed:', data);
    
    const startBtn = document.getElementById('startWebTrainingBtn');
    const stopBtn = document.getElementById('stopWebTrainingBtn');
    const statusDiv = document.getElementById('webTrainingStatus');
    const progressDiv = document.getElementById('webTrainingProgress');
    const resultsDiv = document.getElementById('webTrainingResults');
    
    // UI sıfırla
    if (startBtn && stopBtn) {
        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
    }
    if (progressDiv) {
        // Progress bar'ı 100% yap
        const progressBar = document.getElementById('webProgressBar');
        const progressText = document.getElementById('webProgressText');
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
        }
        if (progressText) {
            progressText.textContent = '100%';
        }
    }
    
    // Success mesajı
    if (statusDiv) {
        statusDiv.className = 'alert alert-success';
        document.getElementById('webTrainingMessage').textContent = 
            `Eğitim tamamlandı! Yeni model versiyonu: ${data.model_version}`;
    }
    
    // Results göster
    if (resultsDiv) {
        resultsDiv.style.display = 'block';
        
        const metrics = data.metrics || {};
        if (metrics.mae) {
            const finalMAEElement = document.getElementById('finalMAE');
            if (finalMAEElement) {
                finalMAEElement.textContent = metrics.mae.toFixed(4);
            }
        }
        if (metrics.accuracy) {
            const finalAccuracyElement = document.getElementById('finalAccuracy');
            if (finalAccuracyElement) {
                finalAccuracyElement.textContent = (metrics.accuracy * 100).toFixed(2) + '%';
            }
        }
        if (metrics.training_samples) {
            const finalTrainingSamplesElement = document.getElementById('finalTrainingSamples');
            if (finalTrainingSamplesElement) {
                finalTrainingSamplesElement.textContent = metrics.training_samples;
            }
        }
        if (metrics.validation_samples) {
            const finalValidationSamplesElement = document.getElementById('finalValidationSamples');
            if (finalValidationSamplesElement) {
                finalValidationSamplesElement.textContent = metrics.validation_samples;
            }
        }
    }
    
    currentTrainingSession = null;
    trainingStartTime = null;
    
    // SSE connection temizle
    if (window.webEventSource) {
        window.webEventSource.close();
        window.webEventSource = null;
    }
    
    // Toast notification
    showToast('Başarılı', 'Web eğitimi tamamlandı!', 'success');
    
    // Model listelerini yenile
    setTimeout(() => {
        loadModelMetrics();
    }, 2000);
}

// SSE web training error
function handleWebTrainingErrorSSE(data) {
    console.error('[SSE] Web training error:', data);
    
    const startBtn = document.getElementById('startWebTrainingBtn');
    const stopBtn = document.getElementById('stopWebTrainingBtn');
    const statusDiv = document.getElementById('webTrainingStatus');
    const progressDiv = document.getElementById('webTrainingProgress');
    
    // UI sıfırla
    if (startBtn && stopBtn) {
        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
    }
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
    
    // Error mesajı
    if (statusDiv) {
        statusDiv.className = 'alert alert-danger';
        document.getElementById('webTrainingMessage').textContent = 
            `Eğitim hatası: ${data.error_message || 'Bilinmeyen hata'}`;
    }
    
    currentTrainingSession = null;
    trainingStartTime = null;
    
    // SSE connection temizle
    if (window.webEventSource) {
        window.webEventSource.close();
        window.webEventSource = null;
    }
    
    showToast('Hata', 'Web eğitimi başarısız oldu', 'error');
} 