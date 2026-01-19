// Training Statistics Module
const TrainingStats = {
    
    // Eğitim verisi istatistiklerini al
    async fetchTrainingDataStats() {
        try {
            const response = await fetch('/api/model/age/training-data-stats');
            if (!response.ok) {
                throw new Error('Veri istatistikleri alınamadı');
            }
            return await response.json();
        } catch (error) {
            console.error('Eğitim verisi istatistikleri alınırken hata:', error);
            return null;
        }
    },
    
    // İstatistikleri modal'da göster
    async displayTrainingStats(modalId = 'trainModelModal') {
        const stats = await this.fetchTrainingDataStats();
        
        if (!stats || !stats.success) {
            this.showNoDataMessage(modalId);
            return;
        }
        
        const statsData = stats.stats;
        
        // İstatistik kartını oluştur
        const statsHtml = `
            <div class="alert alert-info mb-3" id="trainingDataStats">
                <h6 class="alert-heading">
                    <i class="fas fa-database me-2"></i>Mevcut Eğitim Verisi
                </h6>
                <div class="row mt-3">
                    <div class="col-md-3">
                        <div class="text-center">
                            <h4 class="mb-0">${statsData.total_samples}</h4>
                            <small class="text-muted">Toplam Örnek</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h4 class="mb-0 text-primary">${statsData.manual_samples}</h4>
                            <small class="text-muted">Manuel Geri Bildirim</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h4 class="mb-0 text-success">${statsData.pseudo_samples}</h4>
                            <small class="text-muted">Otomatik Etiket</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h4 class="mb-0">${statsData.mean_age ? statsData.mean_age.toFixed(1) : '-'}</h4>
                            <small class="text-muted">Ortalama Yaş</small>
                        </div>
                    </div>
                </div>
                ${this.getDataQualityMessage(statsData)}
            </div>
        `;
        
        // Modal'a ekle
        const modal = document.getElementById(modalId);
        if (modal) {
            const modalBody = modal.querySelector('.modal-body');
            const existingStats = modalBody.querySelector('#trainingDataStats');
            
            if (existingStats) {
                existingStats.outerHTML = statsHtml;
            } else {
                // Alert'ten sonra ekle
                const alertInfo = modalBody.querySelector('.alert-info');
                if (alertInfo) {
                    alertInfo.insertAdjacentHTML('afterend', statsHtml);
                }
            }
        }
        
        // Eğitim butonunu güncelle
        this.updateTrainingButton(statsData.total_samples);
    },
    
    // Veri kalitesi mesajı oluştur
    getDataQualityMessage(stats) {
        const totalSamples = stats.total_samples;
        const manualRatio = stats.manual_samples / totalSamples;
        
        let message = '<div class="mt-3">';
        
        if (totalSamples < 10) {
            message += `
                <div class="alert alert-warning mb-0">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Yetersiz Veri!</strong> Eğitim için en az 10 örnek gerekli. 
                    ${10 - totalSamples} örnek daha toplamanız gerekiyor.
                </div>
            `;
        } else if (totalSamples < 50) {
            message += `
                <div class="alert alert-info mb-0">
                    <i class="fas fa-info-circle me-2"></i>
                    <strong>Temel Eğitim Yapılabilir.</strong> Daha iyi sonuçlar için 50+ örnek önerilir.
                </div>
            `;
        } else if (totalSamples < 100) {
            message += `
                <div class="alert alert-success mb-0">
                    <i class="fas fa-check-circle me-2"></i>
                    <strong>İyi Veri Miktarı!</strong> Kaliteli bir eğitim yapılabilir.
                </div>
            `;
        } else {
            message += `
                <div class="alert alert-success mb-0">
                    <i class="fas fa-star me-2"></i>
                    <strong>Mükemmel Veri Miktarı!</strong> Yüksek kaliteli eğitim yapılabilir.
                </div>
            `;
        }
        
        // Manuel veri oranı uyarısı
        if (totalSamples >= 10 && manualRatio < 0.2) {
            message += `
                <div class="alert alert-info mt-2 mb-0">
                    <i class="fas fa-hand-point-up me-2"></i>
                    Manuel geri bildirim oranı düşük (%${(manualRatio * 100).toFixed(0)}). 
                    Daha fazla manuel düzeltme yapmanız model kalitesini artırabilir.
                </div>
            `;
        }
        
        message += '</div>';
        return message;
    },
    
    // Veri yok mesajı göster
    showNoDataMessage(modalId) {
        const noDataHtml = `
            <div class="alert alert-warning mb-3" id="trainingDataStats">
                <h6 class="alert-heading">
                    <i class="fas fa-database me-2"></i>Eğitim Verisi Bulunamadı
                </h6>
                <p class="mb-0">
                    Henüz hiç geri bildirim toplanmamış. Model eğitimi için önce analiz yapıp 
                    yaş tahminlerini düzeltmeniz gerekiyor.
                </p>
            </div>
        `;
        
        const modal = document.getElementById(modalId);
        if (modal) {
            const modalBody = modal.querySelector('.modal-body');
            const existingStats = modalBody.querySelector('#trainingDataStats');
            
            if (existingStats) {
                existingStats.outerHTML = noDataHtml;
            } else {
                const alertInfo = modalBody.querySelector('.alert-info');
                if (alertInfo) {
                    alertInfo.insertAdjacentHTML('afterend', noDataHtml);
                }
            }
        }
        
        // Eğitim butonunu devre dışı bırak
        this.updateTrainingButton(0);
    },
    
    // Eğitim butonunu güncelle
    updateTrainingButton(sampleCount) {
        const trainBtn = document.getElementById('startTrainingBtn');
        if (trainBtn) {
            if (sampleCount < 10) {
                trainBtn.disabled = true;
                trainBtn.innerHTML = `
                    <i class="fas fa-lock me-1"></i> 
                    ${10 - sampleCount} Örnek Daha Gerekli
                `;
                trainBtn.classList.remove('btn-primary');
                trainBtn.classList.add('btn-secondary');
            } else {
                trainBtn.disabled = false;
                trainBtn.innerHTML = `
                    <i class="fas fa-play me-1"></i> 
                    Eğitimi Başlat
                `;
                trainBtn.classList.remove('btn-secondary');
                trainBtn.classList.add('btn-primary');
            }
        }
    },
    
    // Periyodik güncelleme başlat
    startPeriodicUpdate(intervalMs = 30000) {
        // İlk yükleme
        this.displayTrainingStats();
        
        // Periyodik güncelleme
        this.updateInterval = setInterval(() => {
            this.displayTrainingStats();
        }, intervalMs);
    },
    
    // Periyodik güncellemeyi durdur
    stopPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
};

// Export for use in other modules
window.TrainingStats = TrainingStats; 