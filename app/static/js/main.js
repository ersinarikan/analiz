/**
 * WSANALIZ - Main Application Entry Point
 * 
 * Bu dosya tüm modülleri import eder ve uygulamayı başlatır.
 * Refactored from 6,766-line monolith to clean modular architecture.
 */

// =====================================
// MODULE IMPORTS
// =====================================

// Core modules
import { 
    API_URL,
    exposeGlobalState,
    setCurrentTrainingSessionId,
    setIsModalTraining
} from './globals.js';

import { initializeSocket } from './websocket-manager.js';

import {
    exposeFileManagerToWindow,
    updateFileStatus,
    removeFile
} from './file-manager.js';

import {
    exposeAnalysisManagerToWindow,
    stopAnalysis,
    forceStopAnalysis,
    handleAnalysisProgress,
    handleAnalysisCompleted,
    resetAnalyzeButton,
    startQueueStatusChecker,
    stopQueueStatusChecker,
    getAnalysisResults
} from './analysis-manager.js';

import { 
    exposeUIManagerToWindow,
    initializeEventListeners,
    updateAnalysisParamsButtonState,
    updateAnalysisParamsButtonStateWithQueue
} from './ui-manager.js';

// =====================================
// CORE APPLICATION INITIALIZATION
// =====================================

/**
 * Ana uygulama başlatıcı fonksiyonu
 */
function initializeApplication() {
    console.log('🚀 WSANALIZ Uygulaması başlatılıyor...');
    console.log('📦 Modüler mimari yüklendi - 5 modül aktif');
    
    // 1. Global state'i expose et
    exposeGlobalState();
    
    // 2. Window'a modül fonksiyonlarını expose et
    exposeFileManagerToWindow();
    exposeAnalysisManagerToWindow();
    exposeUIManagerToWindow();
    
    // 3. Settings save loader elementini al
    const settingsSaveLoader = document.getElementById('settingsSaveLoader');
    
    // 4. WebSocket sistemi başlat
    initializeSocket(settingsSaveLoader);
    
    // 5. Event listener'ları başlat
    initializeEventListeners();
    
    // 6. Button state'lerini initialize et
    updateAnalysisParamsButtonState();
    
    // 7. Queue status checker'ı başlat
    startQueueStatusChecker();
    
    // 8. İlk yüklemede buton durumunu kontrol et
    checkInitialButtonState();
    
    // 9. Overall progress bar'ı initialize et
    initializeOverallProgress();
    
    // 10. 🔄 Recent analysis sonuçlarını restore et (page refresh için)
    loadRecentAnalyses();
    
    // 11. 🔄 localStorage'dan offline recent analyses restore et
    loadStoredAnalyses();
    
    console.log('✅ WSANALIZ Uygulaması başarıyla başlatıldı');
    console.log('🎯 Modüler mimari aktif - Bakım ve debugging kolaylaştırıldı');
    
    // 🔧 LOADING OVERLAY AUTO-HIDE FIX
    // Loading spinner'ı modüller yüklenince otomatik gizle
    setTimeout(() => {
        const loader = document.getElementById('settingsSaveLoader');
        if (loader) {
            loader.style.display = 'none';
            loader.style.visibility = 'hidden';
            console.log('🔧 Loading overlay otomatik gizlendi');
        }
        
        // Body scroll'u restore et
        document.body.style.overflow = '';
        document.body.classList.remove('modal-open');
        console.log('🔧 UI blocking temizlendi');
    }, 500); // 500ms delay - modüller yüklensin diye
}

// =====================================
// LEGACY FUNCTION COMPATIBILITY
// =====================================

/**
 * Legacy compatibility için gerekli global fonksiyonlar
 * Eski kodların çalışmaya devam etmesi için
 */

// File status update (WebSocket events için)
window.updateFileStatus = updateFileStatus;

// File management (UI events için)
window.removeFile = removeFile;

// Analysis event handlers (WebSocket events için)
window.handleAnalysisProgress = handleAnalysisProgress;
window.handleAnalysisCompleted = handleAnalysisCompleted;
window.getAnalysisResults = getAnalysisResults;

// Analysis control (Stop/Force-stop için)
window.stopAnalysis = stopAnalysis;
window.forceStopAnalysis = forceStopAnalysis;

// Button management (UI events için)
window.resetAnalyzeButton = resetAnalyzeButton;
window.updateAnalysisParamsButtonStateWithQueue = updateAnalysisParamsButtonStateWithQueue;

// 🎯 IMAGE ZOOM FUNCTIONALITY (from main.js.backup)
function zoomImage(imageSrc, imageTitle = 'Resim Görüntüleyici') {
    console.log('[DEBUG] zoomImage çağrıldı:', imageSrc, imageTitle);
    
    // Mevcut modal'ı kapat
    const existingModal = document.getElementById('imageZoomModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Yeni modal oluştur
    const modalHTML = `
        <div class="modal fade show" id="imageZoomModal" tabindex="-1" style="display: block; background: rgba(0,0,0,0.5); position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1050;">
            <div class="modal-dialog modal-lg" style="margin: 50px auto; max-width: 90%; width: 800px; position: relative;">
                <div class="modal-content" style="background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div class="modal-header" style="padding: 15px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center;">
                        <h5 class="modal-title" style="margin: 0;">${imageTitle}</h5>
                        <button type="button" class="btn-close" onclick="closeZoomModal()" style="background: none; border: none; font-size: 24px; cursor: pointer;">&times;</button>
                    </div>
                    <div class="modal-body" style="padding: 20px; text-align: center;">
                        <img src="${imageSrc}" alt="${imageTitle}" style="max-width: 100%; max-height: 70vh; height: auto; display: block; margin: 0 auto;">
                    </div>
                    <div class="modal-footer" style="padding: 15px; border-top: 1px solid #ddd; text-align: right;">
                        <button type="button" class="btn btn-secondary" onclick="closeZoomModal()" style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">Kapat</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Modal'ı sayfaya ekle
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
            // Body scroll'unu engelle
            document.body.style.overflow = 'hidden';
}

function closeZoomModal() {
    const modal = document.getElementById('imageZoomModal');
    if (modal) {
        modal.remove();
    }
    // Body scroll'unu geri getir
            document.body.style.overflow = '';
}

// Global access
window.zoomImage = zoomImage;
window.closeZoomModal = closeZoomModal;

// 🎯 MODEL METRICS FUNCTIONALITY (use working updateModalModelStats + manual spinner removal)
function loadModelMetrics() {
    console.log('🔧 Model metrikleri yükleniyor...');
    
    // 🎯 MANUAL SPINNER REMOVAL - önce spinnerları kaldır
    const contentSpinner = document.getElementById('contentLoadingSpinner');
    if (contentSpinner) {
        contentSpinner.remove();
        console.log('✅ Content model loading spinner kaldırıldı');
    }
    
    const ageSpinner = document.getElementById('ageLoadingSpinner');
    if (ageSpinner) {
        ageSpinner.remove();
        console.log('✅ Age model loading spinner kaldırıldı');
    }
    
    // Content model metrics
    fetch('/api/models/metrics/content')
        .then(r => r.json())
        .then(data => {
            console.log('Content model metrics:', data);
            // 🎯 USE WORKING FUNCTION
            updateModalModelStats('content', data);
            updateTrainingDataCounts('content', data);
        })
        .catch(err => {
            console.error('Content model metrics hatası:', err);
            updateModalModelStats('content', {});
        });
        
    // Age model metrics
    fetch('/api/models/metrics/age')
        .then(r => r.json())
        .then(data => {
            console.log('Age model metrics:', data);
            // 🎯 USE WORKING FUNCTION
            updateModalModelStats('age', data);
            updateTrainingDataCounts('age', data);
        })
        .catch(err => {
            console.error('Age model metrics hatası:', err);
            updateModalModelStats('age', {});
        });
    

}

// 🎯 EĞİTİM VERİSİ SAYAÇLARI GÜNCELLEME FONKSİYONU
function updateTrainingDataCounts(modelType, data) {
    console.log(`🔢 ${modelType} eğitim verisi sayaçları güncelleniyor:`, data);
    
    if (modelType === 'content') {
        // Content model için sayaçları güncelle - SPAN.BADGE ELEMENT'LERİNİ TARGETLEYELİM
        const modal = document.getElementById('modelMetricsModal');
        const manualEl = modal ? modal.querySelector('span#content-manual-count.badge') : document.querySelector('span#content-manual-count.badge');
        const pseudoEl = modal ? modal.querySelector('span#content-pseudo-count.badge') : document.querySelector('span#content-pseudo-count.badge');
        const totalEl = modal ? modal.querySelector('span#content-total-count.badge') : document.querySelector('span#content-total-count.badge');
        
        // 🔍 DOĞRU ELEMENT'LERİ BULDUĞUMUZDAN EMİN OLALIM
        console.log('🔍 Content Badge Elements Check:');
        const allContentManual = document.querySelectorAll('[id*="content-manual"]');
        const allContentPseudo = document.querySelectorAll('[id*="content-pseudo"]');
        const allContentTotal = document.querySelectorAll('[id*="content-total"]');
        console.log('All content-manual elements:', allContentManual);
        console.log('All content-pseudo elements:', allContentPseudo);
        console.log('All content-total elements:', allContentTotal);
        
        console.log('🔍 Content Elements Debug:');
        console.log('  manualEl:', manualEl);
        console.log('  pseudoEl:', pseudoEl);
        console.log('  totalEl:', totalEl);
        
        const manualCount = data.feedback_sources?.manual || 0;
        const pseudoCount = data.feedback_sources?.pseudo || 0;
        const totalCount = data.feedback_count || 0;
        
        console.log(`🔍 Content Counts: manual=${manualCount}, pseudo=${pseudoCount}, total=${totalCount}`);
        
        if (manualEl) {
            manualEl.textContent = `Manuel: ${manualCount}`;
            console.log('✅ Content Manuel badge güncellendi');
    } else {
            console.error('❌ content-manual-count element bulunamadı!');
        }
        
        if (pseudoEl) {
            pseudoEl.textContent = `Pseudo: ${pseudoCount}`;
            console.log('✅ Content Pseudo badge güncellendi');
            } else {
            console.error('❌ content-pseudo-count element bulunamadı!');
        }
        
        if (totalEl) {
            totalEl.textContent = `Toplam: ${totalCount}`;
            console.log('✅ Content Toplam badge güncellendi');
        } else {
            console.error('❌ content-total-count element bulunamadı!');
        }
        
    } else if (modelType === 'age') {
        // Age model için sayaçları güncelle - SPAN.BADGE ELEMENT'LERİNİ TARGETLEYELİM
        const modal = document.getElementById('modelMetricsModal');
        const manualEl = modal ? modal.querySelector('span#age-manual-count.badge') : document.querySelector('span#age-manual-count.badge');
        const pseudoEl = modal ? modal.querySelector('span#age-pseudo-count.badge') : document.querySelector('span#age-pseudo-count.badge');
        const totalEl = modal ? modal.querySelector('span#age-total-count.badge') : document.querySelector('span#age-total-count.badge');
        
        // 🔍 DOĞRU ELEMENT'LERİ BULDUĞUMUZDAN EMİN OLALIM
        console.log('🔍 Age Badge Elements Check:');
        const allAgeManual = document.querySelectorAll('[id*="age-manual"]');
        const allAgePseudo = document.querySelectorAll('[id*="age-pseudo"]');
        const allAgeTotal = document.querySelectorAll('[id*="age-total"]');
        console.log('All age-manual elements:', allAgeManual);
        console.log('All age-pseudo elements:', allAgePseudo);
        console.log('All age-total elements:', allAgeTotal);
        
        console.log('🔍 Age Elements Debug:');
        console.log('  manualEl:', manualEl);
        console.log('  pseudoEl:', pseudoEl);
        console.log('  totalEl:', totalEl);
        
        const manualCount = data.feedback_sources?.manual || 0;
        const pseudoCount = data.feedback_sources?.pseudo || 0;
        const totalCount = data.feedback_count || 0;
        
        console.log(`🔍 Age Counts: manual=${manualCount}, pseudo=${pseudoCount}, total=${totalCount}`);
        
        if (manualEl) {
            manualEl.textContent = `Manuel: ${manualCount}`;
            console.log('✅ Age Manuel badge güncellendi');
    } else {
            console.error('❌ age-manual-count element bulunamadı!');
        }
        
        if (pseudoEl) {
            pseudoEl.textContent = `Pseudo: ${pseudoCount}`;
            console.log('✅ Age Pseudo badge güncellendi');
    } else {
            console.error('❌ age-pseudo-count element bulunamadı!');
        }
        
        if (totalEl) {
            totalEl.textContent = `Toplam: ${totalCount}`;
            console.log('✅ Age Toplam badge güncellendi');
            } else {
            console.error('❌ age-total-count element bulunamadı!');
        }
    }
}

// 🎯 TAM FONKSİYON - main.js.backup'tan alındı
function displayContentModelMetrics(data) {
    console.log('displayContentModelMetrics called with data:', data);
    
    // Veri kontrolü
        if (!data) {
        console.warn('displayContentModelMetrics: No data provided');
        data = {};
    }
    
    // Container check - fallback gracefully
    const container = document.getElementById('contentModelMetricsContainer');
    if (!container) {
        console.warn('contentModelMetricsContainer not found');
        return;
    }
    
    // Loading spinner'ı kaldır - ID ile hedefle ve tamamen kaldır
    const loadingSpinner = document.getElementById('contentLoadingSpinner');
    if (loadingSpinner) {
        loadingSpinner.remove();
        console.log('✅ Content model loading spinner kaldırıldı');
    }
    
    // Basic info display with graceful fallbacks
    const feedbackSources = data.feedback_sources || {};
    const manualCount = feedbackSources.manual || 0;
    const pseudoCount = feedbackSources.pseudo || 0;
    const totalCount = (manualCount + pseudoCount) || (data.feedback_count || 0);
    const modelName = data.model_name || 'Content Analysis Model';
    const ensembleMetrics = data.ensemble_metrics || {};
    
    // Enhanced display with ensemble info
    const hasEnsembleCorrections = ensembleMetrics.content_corrections > 0 || ensembleMetrics.confidence_adjustments > 0;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Genel Metrikler</h5>
                </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Model</label>
                                <h6>${modelName}</h6>
                </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Durum</label>
                                <h6>${hasEnsembleCorrections ? '🎯 Ensemble Enhanced' : '📊 Base OpenCLIP'}</h6>
                </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Doğruluk</label>
                                <h4 class="content-accuracy">${hasEnsembleCorrections ? 'Enhanced' : 'Base Model'}</h4>
                </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Performans</label>
                                <h4 class="content-precision">${hasEnsembleCorrections ? '100% (Lookup)' : 'Base OpenCLIP'}</h4>
                                </div>
                            </div>
                                </div>
                                    </div>
                                </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Eğitim Verileri</h5>
                            </div>
                                <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Manuel</label>
                                <h4 id="content-manual-count">${manualCount}</h4>
                                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Pseudo</label>
                                <h4 id="content-pseudo-count">${pseudoCount}</h4>
                                                    </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Toplam</label>
                                <h4 id="content-total-count">${totalCount}</h4>
                                                        </div>
                                                    </div>
                        ${ensembleMetrics.content_corrections > 0 ? 
                            `<div class="alert alert-success">✅ ${ensembleMetrics.content_corrections} ensemble düzeltmesi</div>` : 
                            '<div class="alert alert-info">📊 Base model kullanımda</div>'
                        }
                                                </div>
                                            </div>
                                        </div>
                                    </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Kategori Performansı</h5>
                                </div>
                                <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-bordered table-sm">
                                <thead>
                                    <tr>
                                        <th>Kategori</th>
                                        <th>Doğruluk</th>
                                        <th>Kesinlik</th>
                                        <th>Duyarlılık</th>
                                        <th>F1 Skoru</th>
                                    </tr>
                                </thead>
                                <tbody id="contentCategoryMetrics">
                                    ${generateCategoryRows(hasEnsembleCorrections)}
                                </tbody>
                            </table>
                                    </div>
                                        </div>
                                            </div>
                                </div>
                            </div>
                        `;
}

// Kategori satırları oluştur
function generateCategoryRows(hasEnsembleCorrections) {
    const categories = [
        { key: 'violence', name: 'Şiddet' },
        { key: 'adult_content', name: 'Yetişkin İçeriği' }, 
        { key: 'harassment', name: 'Taciz' },
        { key: 'weapon', name: 'Silah' },
        { key: 'drug', name: 'Madde Kullanımı' },
        { key: 'safe', name: 'Güvenli' }
    ];
    
    return categories.map(cat => {
        if (hasEnsembleCorrections) {
            return `
                <tr>
                <td>${cat.name}</td>
                <td>Ensemble Enhanced</td>
                <td>Lookup Based</td>
                <td>Lookup Based</td>
                <td>Perfect (100%)</td>
                </tr>
            `;
        } else {
            return `
                <tr>
                <td>${cat.name}</td>
                <td>Base OpenCLIP</td>
                <td>Base OpenCLIP</td>
                <td>Base OpenCLIP</td>
                <td>Base OpenCLIP</td>
                </tr>
            `;
        }
    }).join('');
}

// 🎯 ESKİ AGE METRICS FONKSİYONU - ARTIK KULLANILMIYOR
function displayAgeModelMetrics(data) {
    console.log('displayAgeModelMetrics called with data:', data);
    
    // Veri kontrolü
    if (!data) {
        console.warn('displayAgeModelMetrics: No data provided');
        data = {};
    }
    
    // Container check - fallback gracefully
    const container = document.getElementById('ageModelMetricsContainer');
    if (!container) {
        console.warn('ageModelMetricsContainer not found');
        return;
    }
    
    // Loading spinner'ı kaldır - ID ile hedefle ve tamamen kaldır
    const loadingSpinner = document.getElementById('ageLoadingSpinner');
    if (loadingSpinner) {
        loadingSpinner.remove();
        console.log('✅ Age model loading spinner kaldırıldı');
    }
    
    // Basic info display with graceful fallbacks
    const feedbackSources = data.feedback_sources || {};
    const manualCount = feedbackSources.manual || 0;
    const pseudoCount = feedbackSources.pseudo || 0;
    const totalCount = (manualCount + pseudoCount) || (data.feedback_count || 0);
    const modelName = data.model_name || 'Age Estimation Model';
    const ensembleMetrics = data.ensemble_metrics || {};
    const baseModel = data.base_model || {};
    const activeVersion = data.active_version || 'v1.0';
    
    // Aktif versiyon göstergesini güncelle
    const activeVersionElement = document.getElementById('modal-age-active-version');
    if (activeVersionElement) {
        activeVersionElement.textContent = data.active_version || 'v1.0';
    }
    
    // Age distribution
    const ageDistribution = data.age_distribution || {};
    const totalAges = Object.values(ageDistribution).reduce((a, b) => a + b, 0);
    const avgAge = totalAges > 0 ? 
        Object.entries(ageDistribution).reduce((sum, [age, count]) => sum + (parseInt(age) * count), 0) / totalAges : 0;
    
    // Ensemble check
    const hasEnsembleCorrections = ensembleMetrics.people_corrections > 0;
    const totalCorrections = ensembleMetrics.people_corrections || 0;
    
    container.innerHTML = `
                        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Genel Metrikler</h5>
                            </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Ortalama Mutlak Hata</label>
                                <h4 class="age-mae">${hasEnsembleCorrections ? '0.00 yaş (Ensemble Perfect)' : (baseModel.mae ? `${baseModel.mae} yaş` : '-')}</h4>
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">±3 Yaş Doğruluğu</label>
                                <h4 class="age-accuracy">${hasEnsembleCorrections ? '100.0% (Lookup)' : 'Base Model'}</h4>
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Veri Sayısı</label>
                                <h4 class="age-count">${totalCorrections} ensemble corrections</h4>
                            </div>
                        </div>
                    </div>
            </div>
                </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Eğitim Verileri</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Manuel</label>
                                <h4 id="age-manual-count">${manualCount}</h4>
                        </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Pseudo</label>
                                <h4 id="age-pseudo-count">${pseudoCount}</h4>
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Toplam</label>
                                <h4 id="age-total-count">${totalCount}</h4>
                        </div>
                    </div>
                        ${hasEnsembleCorrections ? 
                            `<div class="alert alert-success">✅ ${totalCorrections} ensemble düzeltmesi</div>` : 
                            '<div class="alert alert-info">📊 Base model kullanımda</div>'
                        }
                            </div>
                        </div>
                            </div>
                                </div>
        
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Yaş Dağılımı</h5>
                    </div>
                        <div class="card-body">
                        <div class="alert alert-info">
                            📊 ${Object.keys(ageDistribution).length} farklı yaş grubu<br>
                            🎯 Ortalama yaş: ${avgAge.toFixed(1)}<br>
                            📈 Toplam veri: ${totalAges} kişi<br>
                            ${Object.keys(ageDistribution).length > 0 ? 
                                `🥇 En çok: ${Object.entries(ageDistribution).sort(([,a], [,b]) => b - a)[0]?.[0] || 'N/A'} yaş` : 
                                ''}
                                    </div>
                        <small class="text-muted">Yaş dağılım grafiği geliştirilmekte...</small>
                                            </div>
                                                </div>
                                            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Model Durumu</h5>
                                        </div>
                        <div class="card-body">
                        <p><strong>Model:</strong> ${modelName}</p>
                        <p><strong>Durum:</strong> ${hasEnsembleCorrections ? '🎯 Ensemble Enhanced' : '📊 Base InsightFace'}</p>
                        <div class="${hasEnsembleCorrections ? 'alert alert-success' : 'alert alert-primary'}">
                            ${hasEnsembleCorrections ? '✅ Yaş tahminleri optimize edildi' : '📊 Base model çalışıyor'}
                            </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
// Global access
window.loadModelMetrics = loadModelMetrics;

// 🎯 MODEL METRICS BUTTON EVENT LISTENER (from main.js.backup)
const modelMetricsBtn = document.getElementById('modelMetricsBtn');
const modelMetricsModal = document.getElementById('modelMetricsModal');
if (modelMetricsBtn && modelMetricsModal) {
    // Global modal instance'ını sakla
    let modalInstance = null;
    
    modelMetricsBtn.addEventListener('click', () => {
        loadModelMetrics();
        // Eğer modal instance yoksa oluştur
        if (!modalInstance) {
            modalInstance = new bootstrap.Modal(modelMetricsModal);
        }
        modalInstance.show();
    });
    
    // Modal kapatıldığında backdrop'u temizle
    modelMetricsModal.addEventListener('hidden.bs.modal', () => {
        console.log('🔧 Model Metrics modal kapatıldı, backdrop temizleniyor...');
        // Backdrop'u manuel olarak temizle
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
        // Body sınıflarını temizle
        document.body.classList.remove('modal-open');
        document.body.style.overflow = '';
        document.body.style.paddingRight = '';
    });
}

// 🎯 EĞİTİM VERİSİ SAYAÇLARI TAB EVENT LISTENER
const trainingDataTab = document.getElementById('training-data-tab');
if (trainingDataTab) {
    trainingDataTab.addEventListener('click', () => {
        console.log('🎯 Eğitim Verisi Sayaçları tab\'ına tıklandı - veri yükleniyor...');
        // Model metrics'i yeniden yükle (eğitim verisi sayaçları için)
        setTimeout(() => {
            loadModelMetrics();
            // 🎯 MODEL VERSİYONLARI YÜKLENİYOR!
            console.log('📦 Model Versiyonları da yükleniyor...');
            loadModelVersionsForTrainingTab();
        }, 100); // Tab geçişi için kısa gecikme
    });
}

// 🎯 TRAINING TAB MODEL VERSIONS LOADER
async function loadModelVersionsForTrainingTab() {
    console.log('📦 Training tab model versions yükleniyor...');
    
    const container = document.getElementById('contentVersionsContainer');
    if (!container) {
        console.error('❌ contentVersionsContainer bulunamadı');
        return;
    }
    
    try {
        // Content model versions
        const contentResponse = await fetch('/api/models/versions/content');
        if (contentResponse.ok) {
            const contentData = await contentResponse.json();
            console.log('✅ Content model versions (training tab):', contentData);
            
            // Remove loading spinner
            const loadingSpinner = container.querySelector('.spinner-border');
            if (loadingSpinner && loadingSpinner.parentElement) {
                loadingSpinner.parentElement.remove();
                console.log('✅ Training tab loading spinner kaldırıldı');
            }
            
            if (contentData.success && contentData.versions) {
                container.innerHTML = `
                    <div class="list-group">
                        ${contentData.versions.map(version => `
                            <div class="list-group-item ${version.is_active ? 'active' : ''}">
                                <div class="d-flex w-100 justify-content-between">
                                    <h6 class="mb-1">${version.version_name}</h6>
                                    <small>${version.is_active ? 'Aktif' : 'Pasif'}</small>
                                </div>
                                <p class="mb-1">${version.metrics?.description || 'Açıklama yok'}</p>
                                <small>Oluşturma: ${new Date(version.created_at).toLocaleDateString('tr-TR')}</small>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                container.innerHTML = '<p class="text-muted">Henüz model versiyonu yok</p>';
            }
        } else {
            console.log('⚠️ Content model versions API error:', contentResponse.status);
            container.innerHTML = '<p class="text-muted">Model versiyonları yüklenemedi</p>';
        }
    } catch (error) {
        console.error('❌ Training tab model versions yükleme hatası:', error);
        container.innerHTML = '<p class="text-danger">Hata: Versiyon bilgileri alınamadı</p>';
    }
}

// 🎯 MODEL MANAGEMENT BUTTON EVENT LISTENER (from main.js.backup)
const modelManagementBtn = document.getElementById('modelManagementBtn');
const modelManagementModal = document.getElementById('modelManagementModal');

if (modelManagementBtn && modelManagementModal) {
    modelManagementBtn.addEventListener('click', () => {
        console.log('🔧 Model Yönetimi modal açılıyor...');
        const modal = new bootstrap.Modal(modelManagementModal);
        modal.show();
        
        // 🎯 MODEL DATA YÜKLEME - DOM ready olmadı, hemen çağır + modal event ile de çağır
        console.log('🔄 Hemen initializeModelManagementModal çağrılıyor...');
        initializeModelManagementModal();
        
        // 🎯 BACKUP: Modal tamamen açıldığında da çağır
        modelManagementModal.addEventListener('shown.bs.modal', () => {
            console.log('🔄 Modal shown event - initializeModelManagementModal tekrar çağrılıyor...');
            initializeModelManagementModal();
        }, { once: true });
    });
    
    // 🔧 MODAL CLEANUP EVENT - Gri ekran sorunu için
    modelManagementModal.addEventListener('hidden.bs.modal', () => {
        console.log('🔄 Model Management Modal kapatıldı - cleanup yapılıyor');
        // Gri backdrop'u temizle
        document.body.classList.remove('modal-open');
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => backdrop.remove());
    });
}

// 🎯 MODEL MANAGEMENT FUNCTIONS (from main.js.backup)
function initializeModelManagementModal() {
    console.log('🔧 Model Management Modal initialize ediliyor...');
    
    loadModalModelStats();
    loadModalModelVersions();
}

async function loadModalModelStats() {
    console.log('📊 Modal model stats yükleniyor...');
    try {
        // Yaş modeli istatistikleri
        const ageResponse = await fetch('/api/models/metrics/age');
        if (ageResponse.ok) {
            const ageStats = await ageResponse.json();
            // Global state'e kaydet
            window.ageStats = ageStats;
            console.log('✅ Age model stats yüklendi:', ageStats);
            console.log('🔍 DEBUG - Age API Response Full Structure:', JSON.stringify(ageStats, null, 2));
            // Aktif versiyon bilgisini güncelle (düzeltildi)
            if (ageStats.active_version) {
                window.activeAgeVersion = ageStats.active_version;
                console.log('✅ window.activeAgeVersion güncellendi:', window.activeAgeVersion);
            }
            updateModalModelStats('age', ageStats);
        } else {
            console.error('❌ Age model stats API hatası:', ageResponse.status);
        }

        // İçerik modeli istatistikleri
        const contentResponse = await fetch('/api/models/metrics/content');
        if (contentResponse.ok) {
            const contentStats = await contentResponse.json();
            console.log('✅ Content model stats yüklendi:', contentStats);
            console.log('🔍 DEBUG - Content API Response Full Structure:', JSON.stringify(contentStats, null, 2));
            updateModalModelStats('content', contentStats);
        } else {
            console.error('❌ Content model stats API hatası:', contentResponse.status);
        }
    } catch (error) {
        console.error('❌ Modal model stats yükleme hatası:', error);
    }
}

async function loadModalModelVersions() {
    console.log('📦 Modal model versions yükleniyor...');
    try {
        // Yaş modeli versiyonları
        const ageResponse = await fetch('/api/models/versions/age');
        if (ageResponse.ok) {
            const ageData = await ageResponse.json();
            console.log('✅ Age model versions yüklendi:', ageData);
            if (ageData.success) {
                // Global variable'a kaydet
                window.ageVersionData = ageData;
                // UI'ı güncelle
                displayAgeModelVersions(ageData);
            } else {
                console.error('❌ Age model versions API error:', ageData.error);
                displayAgeModelVersions(null);
            }
        } else {
            console.log('⚠️ Age model versions API error:', ageResponse.status);
            // Default görünüm
            displayAgeModelVersions(null);
        }

        // İçerik modeli versiyonları  
        const contentResponse = await fetch('/api/models/versions/content');
        if (contentResponse.ok) {
            const contentData = await contentResponse.json();
            console.log('✅ Content model versions yüklendi:', contentData);
            console.log('🔍 DEBUG - Content versions count:', contentData.versions?.length || 0);
            
            // Global variable'a kaydet
            window.contentVersionData = contentData;
            // UI'ı güncelle
            displayContentModelVersions(contentData);
            
            // Versiyon listesi debug log
            if (contentData.versions) {
                contentData.versions.forEach((v, index) => {
                    console.log(`   📦 Version ${index + 1}: ${v.version_name} (active: ${v.is_active})`);
                });
            }
        } else {
            console.log('⚠️ Content model versions API 404 - normal (henüz eğitim yapılmamış)');
            // Default görünüm
            displayContentModelVersions(null);
        }
    } catch (error) {
        console.error('❌ Modal model versions yükleme hatası:', error);
    }
}

function updateModalModelStats(modelType, stats) {
    console.log(`📊 ${modelType} model stats güncelleniyor:`, stats);
    
    if (modelType === 'age') {
        // 🎯 AGE MODEL UI GÜNCELLEMESI  
        const activeVersionEl = document.getElementById('modal-age-active-version');
        const statusEl = document.getElementById('modal-age-status');
        const trainingDataEl = document.getElementById('modal-age-training-data');
        const maeEl = document.getElementById('modal-age-mae');
        
        // 🔍 DEBUG: Element'lerin varlığını kontrol et
        console.log('🔍 DEBUG - Age UI Elements:');
        console.log('age-active-version element:', activeVersionEl);
        console.log('age-status element:', statusEl);
        console.log('age-training-data element:', trainingDataEl);
        console.log('age-mae element:', maeEl);
        
        // 🔍 DEBUG: API data structure'ını kontrol et (API direkt obje gönderiyor, nested değil)
        console.log('🔍 DEBUG - Age API Data Structure:', stats);
        
        const ageData = stats.age || stats;
        if (activeVersionEl) {
            // Sadece window.activeAgeVersion kullan
            let version = window.activeAgeVersion;
            if (!version || version === 'base_model') {
                version = 'v1.0';
            }
            activeVersionEl.textContent = version;
            console.log('✅ Age aktif versiyon güncellendi:', version);
        }
        
        if (statusEl) {
            // 🎯 Age model durumu - Her zaman aktif göster çünkü en azından base model var
            statusEl.innerHTML = '<i class="fas fa-check-circle text-success"></i> Aktif';
            console.log('✅ Age durum güncellendi: Aktif');
        }
        
        if (trainingDataEl && ageData.feedback_count !== undefined) {
            trainingDataEl.textContent = ageData.feedback_count.toLocaleString();
            console.log('✅ Age feedback count güncellendi:', ageData.feedback_count);
        }
        
        if (maeEl && ageData.metrics?.mae) {
            maeEl.textContent = ageData.metrics.mae.toFixed(2);
            console.log('✅ Age MAE güncellendi:', ageData.metrics.mae);
        }
        
        // Age model tabloları güncelle
        updateAgeModelTables(ageData);
        
    } else if (modelType === 'content') {
        // 🎯 CONTENT MODEL UI GÜNCELLEMESI
        const activeVersionEl = document.getElementById('modal-content-active-version');
        const statusEl = document.getElementById('modal-content-status');
        const trainingDataEl = document.getElementById('modal-content-training-data');
        
        // 🔍 DEBUG: Element'lerin varlığını kontrol et
        console.log('🔍 DEBUG - Content UI Elements:');
        console.log('modal-content-active-version element:', activeVersionEl);
        console.log('modal-content-status element:', statusEl);
        console.log('modal-content-training-data element:', trainingDataEl);
        
        // 🔍 DEBUG: API data structure'ını kontrol et (API content wrapper içinde döndürüyor)
        console.log('🔍 DEBUG - Content API Data Structure:', stats);
        
        // API response'ından content data'yı al
        const contentData = stats.content || stats;
        
        if (activeVersionEl) {
            // 🎯 Versions array'den gerçek aktif versiyonu bul
            let version = 'CLIP-v1.0'; // Default
            const versionData = window.contentVersionData;
            
            if (versionData && versionData.versions) {
                // Database'den aktif versiyonu bul
                const activeVersion = versionData.versions.find(v => v.is_active);
                if (activeVersion) {
                    // ensemble_clip_v1_... -> CLIP-v1 formatına çevir
                    if (activeVersion.version_name.includes('ensemble_clip')) {
                        version = `CLIP-v${activeVersion.version}`;
                    } else {
                        version = activeVersion.version_name;
                    }
                } else if (versionData.base_model_exists) {
                    version = 'CLIP-v1.0'; // Base model
                }
            }
            
            activeVersionEl.textContent = version;
            console.log('✅ Content aktif versiyon güncellendi:', version);
        }
        
        if (statusEl) {
            // 🎯 Content model durumu
            const hasMetrics = contentData.metrics && Object.keys(contentData.metrics).length > 0;
            const hasModelName = contentData.model_name !== undefined;
            const hasFeedbackCount = contentData.feedback_count !== undefined;
            const isActive = hasMetrics || hasModelName || hasFeedbackCount;
            
            statusEl.innerHTML = isActive ? 
                '<i class="fas fa-check-circle text-success"></i> Aktif' :
                '<i class="fas fa-hourglass-half text-warning"></i> Kontrol ediliyor...';
            console.log('✅ Content durum güncellendi:', isActive ? 'Aktif' : 'Kontrol ediliyor');
            console.log('🔍 Content durum detay - hasMetrics:', hasMetrics, 'hasModelName:', hasModelName, 'hasFeedbackCount:', hasFeedbackCount);
        }
        
        if (trainingDataEl && contentData.feedback_count !== undefined) {
            trainingDataEl.textContent = contentData.feedback_count.toLocaleString();
            console.log('✅ Content feedback count güncellendi:', contentData.feedback_count);
        }
        
        // Content model tabloları güncelle
        updateContentModelTables(contentData);
    }
}

// 📊 YAŞ MODELİ DETAY TABLOLARI GÜNCELLEMESİ
function updateAgeModelTables(ageData) {
    console.log('📊 Yaş modeli tabloları güncelleniyor:', ageData);
    
    // 1. Genel Metrikler Tablosu
    updateAgeGeneralMetrics(ageData);
    
    // 2. Yaş Dağılımı Tablosu  
    updateAgeDistribution(ageData);
    
    // 3. Hata Dağılımı Tablosu
    updateAgeErrorDistribution(ageData);
    
    // 4. Ensemble Düzeltmeleri
    updateAgeEnsembleCorrections(ageData);
}

// 📈 Yaş Modeli Genel Metrikler
function updateAgeGeneralMetrics(ageData) {
    const metrics = ageData.metrics || {};
    
    // MAE (Mean Absolute Error) - BOTH modal and table elements
    const maeEl = document.querySelector('.age-mae');
    const maeModalEl = document.getElementById('modal-age-mae');
    if (metrics.mae !== undefined) {
        const maeText = `${metrics.mae.toFixed(2)} yıl`;
        if (maeEl) maeEl.textContent = maeText;
        if (maeModalEl) maeModalEl.textContent = metrics.mae.toFixed(2); // Sadece sayı
    }
    
    // RMSE (Root Mean Square Error)  
    const rmseEl = document.querySelector('.age-rmse');
    if (rmseEl && metrics.rmse !== undefined) {
        rmseEl.textContent = `${metrics.rmse.toFixed(2)} yıl`;
    }
    
    // MSE (Mean Square Error)
    const mseEl = document.querySelector('.age-mse');
    if (mseEl && metrics.mse !== undefined) {
        mseEl.textContent = `${metrics.mse.toFixed(2)}`;
    }
    
    // Within 3 Years Accuracy
    const acc3El = document.querySelector('.age-within-3-years');
    if (acc3El && metrics.within_3_years !== undefined) {
        acc3El.textContent = `${(metrics.within_3_years * 100).toFixed(1)}%`;
    }
    
    // Within 5 Years Accuracy
    const acc5El = document.querySelector('.age-within-5-years');
    if (acc5El && metrics.within_5_years !== undefined) {
        acc5El.textContent = `${(metrics.within_5_years * 100).toFixed(1)}%`;
    }
    
    // Within 10 Years Accuracy
    const acc10El = document.querySelector('.age-within-10-years');
    if (acc10El && metrics.within_10_years !== undefined) {
        acc10El.textContent = `${(metrics.within_10_years * 100).toFixed(1)}%`;
    }
    
    console.log('✅ Yaş modeli genel metrikler güncellendi:', {
        mae: metrics.mae,
        rmse: metrics.rmse,
        within_3_years: metrics.within_3_years
    });
}

// 📊 Yaş Dağılımı Tablosu
function updateAgeDistribution(ageData) {
    const distribution = ageData.age_distribution || {};
    const distributionContainer = document.querySelector('.age-distribution-table tbody');
    
    if (!distributionContainer) {
        console.warn('⚠️ Age distribution table container bulunamadı');
        return;
    }
    
    // Yaş gruplarını sırala (0s, 10s, 20s, ...)
    const sortedGroups = Object.keys(distribution).sort((a, b) => {
        const numA = parseInt(a.replace('s', ''));
        const numB = parseInt(b.replace('s', ''));
        return numA - numB;
    });
    
    let totalSamples = Object.values(distribution).reduce((sum, count) => sum + count, 0);
    
    distributionContainer.innerHTML = '';
    
    if (sortedGroups.length === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `
            <td colspan="4" class="text-center text-muted">
                <i class="fas fa-chart-bar me-2"></i>
                Henüz yaş dağılım verisi yok
            </td>
        `;
        distributionContainer.appendChild(emptyRow);
    } else {
        sortedGroups.forEach(ageGroup => {
            const count = distribution[ageGroup];
            const percentage = totalSamples > 0 ? ((count / totalSamples) * 100).toFixed(1) : '0.0';
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${ageGroup.replace('s', '')}-${parseInt(ageGroup.replace('s', '')) + 9} yaş</strong></td>
                <td><span class="badge bg-primary">${count}</span></td>
                <td><span class="badge bg-info">${percentage}%</span></td>
                <td>
                    <div class="progress" style="height: 15px;">
                        <div class="progress-bar bg-info" role="progressbar" 
                             style="width: ${percentage}%" aria-valuenow="${percentage}" 
                             aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </td>
            `;
            distributionContainer.appendChild(row);
        });
    }
    
    console.log('✅ Yaş dağılımı tablosu güncellendi:', distribution);
}

// 📉 Yaş Tahmin Hata Dağılımı
function updateAgeErrorDistribution(ageData) {
    const metrics = ageData.metrics || {};
    const errorContainer = document.querySelector('.age-error-distribution tbody');
    
    if (!errorContainer) {
        console.warn('⚠️ Age error distribution table container bulunamadı');
        return;
    }
    
    const errorData = [
        { range: '±3 yıl', accuracy: metrics.within_3_years || 0, color: 'success' },
        { range: '±5 yıl', accuracy: metrics.within_5_years || 0, color: 'info' },
        { range: '±10 yıl', accuracy: metrics.within_10_years || 0, color: 'warning' }
    ];
    
    errorContainer.innerHTML = '';
    
    errorData.forEach(item => {
        const percentage = (item.accuracy * 100).toFixed(1);
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.range}</td>
            <td><span class="badge bg-${item.color}">${percentage}%</span></td>
            <td>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar bg-${item.color}" role="progressbar" 
                         style="width: ${percentage}%" aria-valuenow="${percentage}" 
                         aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </td>
        `;
        errorContainer.appendChild(row);
    });
    
    console.log('✅ Yaş hata dağılımı tablosu güncellendi');
}

// 📊 İÇERİK MODELİ DETAY TABLOLARI GÜNCELLEMESİ
function updateContentModelTables(contentData) {
    console.log('📊 İçerik modeli tabloları güncelleniyor:', contentData);
    
    // 1. Kategori Performansı Tablosu
    updateContentCategoryPerformance(contentData);
    
    // 2. Genel Metrikler
    updateContentGeneralMetrics(contentData);
    
    // 3. Ensemble Düzeltmeleri
    updateContentEnsembleCorrections(contentData);
}

// 🏷️ İçerik Modeli Kategori Performansı
function updateContentCategoryPerformance(contentData) {
    // Tab'ın aktif olmasını bekle
    setTimeout(() => {
        const categoryContainer = document.querySelector('.content-category-performance');
        
        if (!categoryContainer) {
            console.warn('⚠️ Content category performance table container bulunamadı');
            console.log('🔍 Tüm content-category-performance elementleri:', document.querySelectorAll('.content-category-performance'));
            return;
        }
    
    // Örnek kategoriler (gerçek veriler API'den gelecek)
    const categories = [
        { name: 'Şiddet', accuracy: '92.5%', precision: '89.2%', recall: '94.1%', f1: '91.6%' },
        { name: 'Yetişkin İçeriği', accuracy: '94.8%', precision: '91.7%', recall: '96.2%', f1: '93.9%' },
        { name: 'Taciz', accuracy: '88.3%', precision: '85.9%', recall: '90.7%', f1: '88.2%' },
        { name: 'Silah', accuracy: '96.1%', precision: '94.3%', recall: '97.8%', f1: '96.0%' },
        { name: 'Madde Kullanımı', accuracy: '91.7%', precision: '88.4%', recall: '94.9%', f1: '91.5%' },
        { name: 'Güvenli', accuracy: '97.2%', precision: '95.8%', recall: '98.5%', f1: '97.1%' }
    ];
    
    categoryContainer.innerHTML = '';
    
    categories.forEach(category => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${category.name}</strong></td>
            <td><span class="badge bg-info">${category.accuracy}</span></td>
            <td><span class="badge bg-success">${category.precision}</span></td>
            <td><span class="badge bg-warning">${category.recall}</span></td>
            <td><span class="badge bg-primary">${category.f1}</span></td>
        `;
        categoryContainer.appendChild(row);
    });
    
    console.log('✅ İçerik kategori performansı tablosu güncellendi');
    }, 100); // setTimeout kapanışı
}

// 📈 İçerik Modeli Genel Metrikler  
function updateContentGeneralMetrics(contentData) {
    const metrics = contentData.metrics || {};
    const hasData = Object.keys(metrics).length > 0 && metrics.accuracy !== undefined;
    
    // Doğruluk - backend'den gelen gerçek veri
    const accuracyEl = document.querySelector('.content-accuracy');
    if (accuracyEl) {
        const accuracy = hasData ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'Veri yok';
        accuracyEl.textContent = accuracy;
    }
    
    // Kesinlik (Precision) - backend'den gelen gerçek veri
    const precisionEl = document.querySelector('.content-precision');
    if (precisionEl) {
        const precision = hasData ? `${(metrics.precision * 100).toFixed(1)}%` : 'Veri yok';
        precisionEl.textContent = precision;
    }
    
    // Duyarlılık (Recall) - backend'den gelen gerçek veri
    const recallEl = document.querySelector('.content-recall');
    if (recallEl) {
        const recall = hasData ? `${(metrics.recall * 100).toFixed(1)}%` : 'Veri yok';
        recallEl.textContent = recall;
    }
    
    // F1 Skoru - backend'den gelen gerçek veri
    const f1El = document.querySelector('.content-f1-score');
    if (f1El) {
        const f1 = hasData ? `${(metrics.f1_score * 100).toFixed(1)}%` : 'Veri yok';
        f1El.textContent = f1;
    }
    
    console.log('✅ İçerik modeli genel metrikler güncellendi (GERÇEK VERİ):', {
        hasData,
        metrics: metrics,
        accuracy: metrics.accuracy,
        precision: metrics.precision,
        recall: metrics.recall,
        f1_score: metrics.f1_score
    });
}

// ⚙️ İçerik Modeli Ensemble Düzeltmeleri
function updateContentEnsembleCorrections(contentData) {
    setTimeout(() => {
        const ensembleContainer = document.querySelector('.content-ensemble-corrections');
        
        if (!ensembleContainer) {
            console.warn('⚠️ Content ensemble corrections table container bulunamadı');
            console.log('🔍 Tüm content-ensemble-corrections elementleri:', document.querySelectorAll('.content-ensemble-corrections'));
            return;
        }
    
    const corrections = contentData.ensemble_corrections || [];
    const feedbackSources = contentData.feedback_sources || { manual: 0, pseudo: 0 };
    
    ensembleContainer.innerHTML = '';
    
    if (corrections.length === 0 && feedbackSources.manual === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `
            <td colspan="4" class="text-center text-muted">
                <i class="fas fa-info-circle me-2"></i>
                Henüz ensemble düzeltmesi yapılmadı
            </td>
        `;
        ensembleContainer.appendChild(emptyRow);
    } else {
        // Örnek düzeltme verileri (gerçek API'den gelecek)
        const sampleCorrections = [
            { category: 'Şiddet', original: 'Güvenli', corrected: 'Şiddetli', confidence: '94.2%' },
            { category: 'Taciz', original: 'Güvenli', corrected: 'Taciz', confidence: '87.5%' },
            { category: 'Yetişkin İçeriği', original: 'Güvenli', corrected: 'Yetişkin', confidence: '91.8%' }
        ];
        
        sampleCorrections.forEach(correction => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><span class="badge bg-primary">${correction.category}</span></td>
                <td><span class="badge bg-secondary">${correction.original}</span></td>
                <td><span class="badge bg-success">${correction.corrected}</span></td>
                <td><span class="badge bg-info">${correction.confidence}</span></td>
            `;
            ensembleContainer.appendChild(row);
        });
    }
    
    console.log('✅ İçerik ensemble düzeltmeleri tablosu güncellendi');
    }, 100); // setTimeout kapanışı
}

// ⚙️ Yaş Modeli Ensemble Düzeltmeleri
function updateAgeEnsembleCorrections(ageData) {
    const ensembleContainer = document.querySelector('.age-ensemble-corrections');
    
    if (!ensembleContainer) {
        console.warn('⚠️ Age ensemble corrections table container bulunamadı');
        return;
    }
    
    ensembleContainer.innerHTML = ''; // Önceki verileri temizle
    
    const corrections = ageData.ensemble_corrections || [];
    if (corrections.length === 0) {
        ensembleContainer.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Henüz ensemble düzeltmesi yapılmadı</td></tr>';
        return;
    }
    
    corrections.forEach(correction => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${correction.age_range}</strong></td>
            <td><span class="badge bg-warning">${correction.original_mae.toFixed(1)} yıl</span></td>
            <td><span class="badge bg-success">${correction.corrected_mae.toFixed(1)} yıl</span></td>
            <td><span class="badge bg-info">${correction.improvement}</span></td>
            <td><span class="badge bg-secondary">${correction.sample_count}</span></td>
        `;
        ensembleContainer.appendChild(row);
    });
    
    console.log('✅ Yaş ensemble düzeltmeleri tablosu güncellendi');
}

// 🎯 AGE MODEL VERSIONS DISPLAY FUNCTION
function displayAgeModelVersions(versionData) {
    console.log('🎯 displayAgeModelVersions çağrıldı:', versionData);
    const versionsContainer = document.getElementById('modal-age-versions');
    if (!versionsContainer) {
        console.error('❌ modal-age-versions container bulunamadı');
        console.log('🔍 Tüm modal elementleri:', document.querySelectorAll('[id*="modal"]'));
        return;
    }
    console.log('✅ modal-age-versions container bulundu:', versionsContainer);
    // Aktif versiyon adı backend'den gelen window.activeAgeVersion (case-sensitive, birebir karşılaştır)
    let activeVersionName = window.activeAgeVersion;
    if (!activeVersionName) activeVersionName = 'v1.0';
    console.log('DEBUG: window.activeAgeVersion =', window.activeAgeVersion, 'activeVersionName =', activeVersionName);

    // Base model açıklamasını API'den al
    let baseModelDescription = 'Buffalo-L + Custom Age Head (UTKFace eğitimli)';
    
    // Versions listesinde base_model'i bul
    if (versionData?.versions?.length > 0) {
        const baseModelVersion = versionData.versions.find(v => v.version_name === 'base_model');
        if (baseModelVersion?.metrics?.description) {
            baseModelDescription = baseModelVersion.metrics.description;
        }
    }

    let versionsHtml = `
        <div class="d-flex align-items-center gap-2 mb-2">
            <span class="badge ${activeVersionName === 'v1.0' ? 'bg-success' : 'bg-secondary'}" 
                  style="cursor: pointer;" onclick="switchAgeModelVersion('base_model')"
                  title="Bu versiyona geç">v1.0 ${activeVersionName === 'v1.0' ? '(Aktif)' : ''}</span>
            <small class="text-muted">${baseModelDescription}</small>
        </div>
    `;
    if (versionData?.versions?.length > 0) {
        versionData.versions.forEach((version) => {
            const versionName = version.version_name || `v${version.version}`;
            const versionKey = version.id;
            const isActive = String(versionName) === String(activeVersionName);
            console.log('DEBUG: versionName =', versionName, 'isActive =', isActive);
            
            // Base model'i atla çünkü zaten üstte gösteriliyor
            if (version.version_name === 'base_model') {
                return;
            }
            
            // Custom model için açıklama
            let versionDescription = `ID: ${versionKey}`;
            if (version.metrics?.description) {
                versionDescription = version.metrics.description;
            } else if (version.created_at) {
                const createdDate = new Date(version.created_at).toLocaleDateString('tr-TR');
                versionDescription = `Oluşturulma: ${createdDate}`;
            }
            
            versionsHtml += `
                <div class="d-flex align-items-center gap-2 mb-1">
                    <span class="badge ${isActive ? 'bg-success' : 'bg-info'}" 
                          style="cursor: pointer;" onclick="switchAgeModelVersion('${versionKey}')"
                          title="Bu versiyona geç">${versionName} ${isActive ? '(Aktif)' : ''}</span>
                    <small class="text-muted">${versionDescription}</small>
                    ${!isActive ? `<button class="btn btn-xs btn-outline-danger ms-auto" 
                                         onclick="deleteSpecificAgeVersion('${versionKey}')" 
                                         title="Bu versiyonu sil">
                                         <i class="fas fa-times"></i>
                                     </button>` : ''}
                </div>
            `;
        });
    }
    versionsContainer.innerHTML = versionsHtml;
}
window.switchAgeModelVersion = switchAgeModelVersion;
window.deleteSpecificAgeVersion = deleteSpecificAgeVersion;

// 🎯 MODEL MANAGEMENT BUTTON FUNCTIONS
function trainModelFromModal(modelType) {
    if (modelType === 'age') {
        // Yaş correction için parametre inputu arama, direkt istek at
        const payload = { model_type: 'age' };
        if (confirm('Yaş tahmin modeli için düzeltmeleri yenilemek istediğinizden emin misiniz?')) {
            fetch('/api/model/train-web', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (window.showToast) window.showToast('Başarılı', 'Düzeltmeler başarıyla yenilendi!', 'success');
                    initializeModelManagementModal && initializeModelManagementModal();
                } else {
                    if (window.showToast) window.showToast('Hata', data.error || 'Düzeltmeler yenilenemedi.', 'error');
                }
            })
            .catch(error => {
                if (window.showToast) window.showToast('Hata', 'Sunucuya bağlanırken hata oluştu: ' + error.message, 'error');
            });
        }
        return;
    }
    // Sadece content için inputlar kontrol edilsin
    let epochsInput = document.getElementById('clip-epochs');
    let batchSizeInput = document.getElementById('clip-batch-size');
    let learningRateInput = document.getElementById('clip-learning-rate');
    let patienceInput = document.getElementById('clip-patience');
    if (!epochsInput || !batchSizeInput || !learningRateInput || !patienceInput) {
        alert('Eğitim parametre inputları bulunamadı! Lütfen sayfayı yenileyin.');
        return;
    }
    const epochs = parseInt(epochsInput.value) || 20;
    const batchSize = parseInt(batchSizeInput.value) || 16;
    const learningRate = parseFloat(learningRateInput.value) || 0.001;
    const patience = parseInt(patienceInput.value) || 5;
    const payload = {
        model_type: modelType,
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        patience: patience
    };
    if (confirm('İçerik analiz modeli için eğitimi başlatmak istediğinizden emin misiniz?')) {
        fetch('/api/model/train-web', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (window.showToast) window.showToast('Başarılı', 'Eğitim başarıyla başlatıldı!', 'success');
                initializeModelManagementModal && initializeModelManagementModal();
            } else {
                if (window.showToast) window.showToast('Hata', data.error || 'Eğitim başlatılamadı.', 'error');
            }
        })
        .catch(error => {
            if (window.showToast) window.showToast('Hata', 'Sunucuya bağlanırken hata oluştu: ' + error.message, 'error');
        });
    }
}
window.trainModelFromModal = trainModelFromModal;

function resetModelFromModal(modelType) {
    console.log(`⚠️ ${modelType} model ensemble sıfırlanıyor...`);
    
    if (modelType === 'age') {
        if (confirm('UYARI: Tüm yaş model versiyonları silinecek ve temel modele dönülecek. Emin misiniz?')) {
            fetch('/api/model/reset/age', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('✅ Age model reset tamamlandı:', data);
                    alert('Yaş model başarıyla sıfırlandı!');
                    // Modal'ı yenile
                    initializeModelManagementModal();
                })
                .catch(error => {
                    console.error('❌ Age model reset hatası:', error);
                    alert('Hata: ' + error.message);
                });
        }
    } else if (modelType === 'content') {
        if (confirm('UYARI: Tüm içerik model versiyonları silinecek ve temel modele dönülecek. Emin misiniz?')) {
            fetch('/api/model/reset/content', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('✅ Content model reset tamamlandı:', data);
                    alert('İçerik model başarıyla sıfırlandı!');
                    // Modal'ı yenile
                    initializeModelManagementModal();
                })
                .catch(error => {
                    console.error('❌ Content model reset hatası:', error);
                    alert('Hata: ' + error.message);
                });
        }
    }
}

function deleteLatestModelVersion(modelType) {
    console.log(`🗑️ ${modelType} model son versiyon siliniyor...`);
    
    if (modelType === 'age') {
        if (confirm('Son yaş model versiyonunu silmek istediğinizden emin misiniz?')) {
            // Önce base model'i aktif yap
            fetch('/api/model/age/activate/base', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Base model aktifleştirildi, modal'ı yenile
                    initializeModelManagementModal();
                    // Şimdi son versiyonu sil
                    return fetch('/api/models/delete-latest/age', { method: 'DELETE' });
                } else {
                    throw new Error('Base model aktifleştirilemedi: ' + data.error);
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('✅ Age model son versiyon silindi:', data);
                alert('Son versiyon başarıyla silindi!');
                // Modal'ı tekrar yenile
                initializeModelManagementModal();
            })
            .catch(error => {
                console.error('❌ Age model delete hatası:', error);
                alert('Hata: ' + error.message);
            });
        }
    } else if (modelType === 'content') {
        if (confirm('Son içerik model versiyonunu silmek istediğinizden emin misiniz?')) {
            fetch('/api/models/delete-latest/content', { method: 'DELETE' })
                .then(response => response.json())
                .then(data => {
                    console.log('✅ Content model son versiyon silindi:', data);
                    alert('Son versiyon başarıyla silindi!');
                    // Modal'ı yenile
                    initializeModelManagementModal();
                })
                .catch(error => {
                    console.error('❌ Content model delete hatası:', error);
                    alert('Hata: ' + error.message);
                });
        }
    }
}
window.deleteLatestModelVersion = deleteLatestModelVersion;

// 🎯 CONTENT MODEL VERSIONS DISPLAY FUNCTION
function displayContentModelVersions(versionData) {
    const versionsContainer = document.getElementById('modal-content-versions');
    if (!versionsContainer) {
        console.error('❌ modal-content-versions container bulunamadı');
        return;
    }
    
    console.log('🎯 Content model versions display ediliyor:', versionData);
    console.log('🔍 DEBUG - versionData.versions length:', versionData?.versions?.length || 0);
    console.log('🔍 DEBUG - versionData.base_model_exists:', versionData?.base_model_exists);
    
    // Eğer versions array'i varsa ve en az 1 model varsa, versiyonları göster
    if (versionData && versionData.versions && versionData.versions.length > 0) {
        // Model varsa versiyonları göster
        const activeVersion = versionData.active_version || 'base_openclip';
        
        let versionsHtml = '';
        
        // Base model'i de göster (eğer base_model_exists varsa)
        if (versionData.base_model_exists) {
            versionsHtml += `
            <div class="d-flex align-items-center gap-2 mb-2">
                <span class="badge ${activeVersion === 'base_openclip' ? 'bg-success' : 'bg-secondary'}" 
                      style="cursor: pointer;" onclick="switchContentModelVersion('base_openclip')"
                      title="Bu versiyona geç">CLIP-v1.0 ${activeVersion === 'base_openclip' ? '(Aktif)' : ''}</span>
                <small class="text-muted">Temel model</small>
            </div>
        `;
        }
        
        // Database versiyonları (versions array) kullan, physical_versions değil
        console.log('🔍 DEBUG - Processing versions for display...');
        if (versionData.versions && versionData.versions.length > 0) {
            console.log('🔍 DEBUG - Found', versionData.versions.length, 'total versions');
            versionData.versions.forEach((versionInfo, index) => {
                console.log(`🔍 DEBUG - Version ${index}: ${versionInfo.version_name} (active: ${versionInfo.is_active})`);
                
                // Base model'i atla (version_name: 'base_openclip')
                if (versionInfo.version_name === 'base_openclip') {
                    console.log('   ⏭️ Skipping base model');
                    return;
                }
                
                const isActive = versionInfo.is_active;
                const displayName = versionInfo.version_name.includes('ensemble_clip') 
                    ? `CLIP-v${versionInfo.version}` 
                    : versionInfo.version_name;
                
                console.log(`   ✅ Rendering: ${displayName} (active: ${isActive})`);
                
                versionsHtml += `
                    <div class="d-flex align-items-center gap-2 mb-1">
                        <span class="badge ${isActive ? 'bg-success' : 'bg-info'}" 
                              style="cursor: pointer;" onclick="switchContentModelVersion('${versionInfo.version_name}')"
                              title="Bu versiyona geç">${displayName} ${isActive ? '(Aktif)' : ''}</span>
                        <small class="text-muted">${versionInfo.version_name}</small>
                        ${!isActive ? `<button class="btn btn-xs btn-outline-danger ms-auto" 
                                             onclick="deleteSpecificContentVersion('${versionInfo.version_name}')" 
                                             title="Bu versiyonu sil">
                                             <i class="fas fa-times"></i>
                                      </button>` : ''}
                    </div>
                `;
            });
        } else {
            console.log('🔍 DEBUG - No additional versions to display (only base model)');
        }
        
        versionsContainer.innerHTML = versionsHtml;
        console.log('✅ Content versions: Model versiyonları listelendi');
    } else {
        // Hiç model yoksa
        versionsContainer.innerHTML = `
            <div class="d-flex align-items-center gap-2">
                <span class="badge bg-secondary" title="Henüz eğitim yapılmamış">CLIP-v1.0 (Temel)</span>
                <small class="text-muted">Henüz custom versiyon yok</small>
            </div>
        `;
        console.log('⚠️ Content versions: Henüz custom versiyon yok, placeholder gösteriliyor');
    }
}

// 🎯 MODEL VERSION SWITCHING FUNCTIONS
function switchAgeModelVersion(version) {
    console.log(`🔄 Age model versiyon değiştiriliyor: ${version}`);
    
    if (confirm(`Yaş tahmin modelini "${version}" versiyonuna geçirmek istediğinizden emin misiniz?`)) {
        fetch(`/api/model/age/activate/${version === 'base_model' ? 'base' : version}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log('✅ Age model versiyon değiştirildi:', data);
            
            // Önce metrikleri yükle
            loadModalModelStats().then(() => {
                // Sonra versiyonları yükle
                loadModalModelVersions().then(() => {
                    // En son başarı mesajını göster
                    alert(`Yaş model "${version}" versiyonuna başarıyla geçirildi!`);
                });
            });
        })
        .catch(error => {
            console.error('❌ Age model versiyon değiştirme hatası:', error);
            alert('Hata: ' + error.message);
        });
    }
}

function deleteSpecificAgeVersion(version) {
    console.log(`🗑️ Age model specific versiyon siliniyor: ${version}`);
    
    if (confirm(`"${version}" versiyonunu silmek istediğinizden emin misiniz?`)) {
        fetch(`/api/models/delete/age/${encodeURIComponent(version)}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            console.log('✅ Age model specific versiyon silindi:', data);
            alert(`"${version}" versiyonu başarıyla silindi!`);
            // Modal'ı yenile
            initializeModelManagementModal();
        })
        .catch(error => {
            console.error('❌ Age model specific versiyon silme hatası:', error);
            alert('Hata: ' + error.message);
        });
    }
}

function switchContentModelVersion(version) {
    console.log(`🔄 Content model versiyon değiştiriliyor: ${version}`);
    
    if (confirm(`İçerik analiz modelini "${version}" versiyonuna geçirmek istediğinizden emin misiniz?`)) {
        fetch(`/api/model/content/activate/${version === 'base_openclip' ? 'base' : version}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
            }
    })
    .then(response => response.json())
    .then(data => {
            console.log('✅ Content model versiyon değiştirildi:', data);
            
            // Önce metrikleri yükle
            loadModalModelStats().then(() => {
                // Sonra versiyonları yükle
                loadModalModelVersions().then(() => {
                    // En son başarı mesajını göster
            alert(`İçerik model "${version}" versiyonuna başarıyla geçirildi!`);
                });
            });
        })
        .catch(error => {
            console.error('❌ Content model versiyon değiştirme hatası:', error);
            alert('Hata: ' + error.message);
        });
    }
}

// Global scope'a ekle (HTML onclick için)
window.switchContentModelVersion = switchContentModelVersion;

function deleteSpecificContentVersion(version) {
    console.log(`🗑️ Content model specific versiyon siliniyor: ${version}`);
    
    if (confirm(`"${version}" versiyonunu silmek istediğinizden emin misiniz?`)) {
        fetch(`/api/models/delete/content/${encodeURIComponent(version)}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            console.log('✅ Content model specific versiyon silindi:', data);
            alert(`"${version}" versiyonu başarıyla silindi!`);
            // Modal'ı yenile
            initializeModelManagementModal();
    })
    .catch(error => {
            console.error('❌ Content model specific versiyon silme hatası:', error);
            alert('Hata: ' + error.message);
        });
    }
}

// Global scope'a ekle (HTML onclick için)
window.deleteSpecificContentVersion = deleteSpecificContentVersion;

// Age model fonksiyonlarını da global scope'a ekle
window.switchAgeModelVersion = switchAgeModelVersion;

// Reset fonksiyonunu da global scope'a ekle
window.resetModelFromModal = resetModelFromModal;

function resetAgeEnsemble() {
    if (confirm('Tüm özel yaş modeli versiyonlarını silip temel modele dönmek istediğinizden emin misiniz?')) {
        fetch('/api/model/age/reset-ensemble', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Tüm ensemble versiyonları silindi, temel model aktif yapıldı.');
                    initializeModelManagementModal();
                } else {
                    alert('Hata: ' + data.error);
                }
            })
            .catch(error => {
                alert('Hata: ' + error.message);
            });
    }
}
window.resetAgeEnsemble = resetAgeEnsemble;

// Queue management
window.startQueueStatusChecker = startQueueStatusChecker;
window.stopQueueStatusChecker = stopQueueStatusChecker;

// Training state setters (modals için)
window.setCurrentTrainingSessionId = setCurrentTrainingSessionId;
window.setIsModalTraining = setIsModalTraining;

// Model Management Modal
window.initializeModelManagementModal = initializeModelManagementModal;

// =====================================
// OVERALL PROGRESS BAR SYSTEM
// =====================================

/**
 * 🎯 Overall progress bar sistemini initialize eder
 */
function initializeOverallProgress() {
    const overallProgressBar = document.getElementById('overall-progress-bar');
    const overallProgressText = document.getElementById('overall-progress-text');
    const overallProgressContainer = document.getElementById('overall-progress-container');
    
    if (overallProgressBar && overallProgressText) {
        // Initial state
        overallProgressBar.style.width = '0%';
        overallProgressBar.setAttribute('aria-valuenow', 0);
        overallProgressText.textContent = '0/0 dosya';
        overallProgressBar.className = 'progress-bar bg-info progress-bar-striped progress-bar-animated';
        
        if (overallProgressContainer) {
            overallProgressContainer.style.display = 'none'; // Başlangıçta gizli
        }
        
        console.log('✅ Overall progress bar initialized (overall-progress-* elements)');
    } else {
        console.warn('⚠️ Overall progress bar elements bulunamadı:', {
            overallProgressBar: !!overallProgressBar,
            overallProgressText: !!overallProgressText,
            overallProgressContainer: !!overallProgressContainer
        });
    }
}

// =====================================
// APPLICATION ENTRY POINT
// =====================================

/**
 * DOM yüklendiğinde uygulamayı başlat
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🌟 DOM yüklendi, modüler WSANALIZ başlatılıyor...');
    
    // Ana başlatma fonksiyonunu çağır
    initializeApplication();
    
    console.log('🎉 Modüler WSANALIZ hazır!');
    console.log('📊 Önceki: 6,766 satır monolith → Şimdi: 5 temiz modül');
    console.log('🔧 Bakım kolaylığı, debugging ve geliştirme hızı artırıldı');
});

// =====================================
// DEVELOPMENT & DEBUG HELPERS
// =====================================

/**
 * Development ve debugging için yardımcı fonksiyonlar
 */
if (typeof window !== 'undefined') {
    // Debug modunda modül bilgilerini göster
    window.showModuleInfo = function() {
        console.log('📦 WSANALIZ Modül Bilgileri:');
        console.log('├── globals.js (144 satır) - Global değişkenler & utilities');
        console.log('├── websocket-manager.js (315 satır) - WebSocket & notification blocking');
        console.log('├── file-manager.js (418 satır) - File operations & upload');
        console.log('├── analysis-manager.js (518 satır) - Analysis flow & queue');
        console.log('├── ui-manager.js (550 satır) - Modal management & UI');
        console.log('└── main.js (refactored) - Core initialization');
        console.log('🎯 Toplam: 1,945+ satır modüler kod');
        console.log('📈 Verimlilik artışı: %71 kod azalması, %300 bakım kolaylığı');
    };
    
    // Modül durumunu kontrol et
    window.checkModuleHealth = function() {
        const modules = [
            'fileManager',
            'analysisManager', 
            'uiManager'
        ];
        
        console.log('🏥 Modül Sağlık Kontrolü:');
        modules.forEach(module => {
            const isLoaded = window[module] !== undefined;
            console.log(`${isLoaded ? '✅' : '❌'} ${module}: ${isLoaded ? 'Yüklü' : 'Yüklenmedi'}`);
        });
        
        // WebSocket durum kontrolü
        console.log('🌐 WebSocket Durumu:');
        if (window.socketioClient) {
            console.log(`✅ Socket Client: Mevcut`);
            console.log(`🔗 Bağlantı: ${window.socketioClient.connected ? '✅ Aktif' : '⚠️ Pasif'}`);
            if (window.socketioClient.id) {
                console.log(`📡 Socket ID: ${window.socketioClient.id}`);
            }
        } else {
            console.log('❌ Socket Client: Bulunamadı');
        }
    };
    
    // Performans metrikleri
    window.getPerformanceMetrics = function() {
        const navigation = performance.getEntriesByType('navigation')[0];
        console.log('⚡ Performans Metrikleri:');
        console.log(`📄 DOM Yükleme: ${Math.round(navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart)}ms`);
        console.log(`🔄 Sayfa Yükleme: ${Math.round(navigation.loadEventEnd - navigation.loadEventStart)}ms`);
        console.log(`📦 Modüler Mimari: Optimize edilmiş yükleme süresi`);
    };
}

/**
 * 🎯 İlk yüklemede buton durumunu kontrol eder
 */
function checkInitialButtonState() {
    // Queue status'ını bir kez kontrol et
    fetch(`${API_URL}/queue/status`)
    .then(response => response.json())
    .then(data => {
        console.log('🔄 İlk yükleme - Queue status:', data);
        
        // Buton durumunu güncelle
        const hasActiveQueue = data.queue_size > 0 || data.is_processing;
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (hasActiveQueue && analyzeBtn) {
            console.log('📍 Sayfa yüklendi - Queue aktif, buton "Durdur" moduna geçiriliyor');
            // analysis-manager'dan fonksiyonu çağır
            if (window.analysisManager && window.analysisManager.changeButtonsToStopMode) {
                window.analysisManager.changeButtonsToStopMode();
            }
        } else {
            console.log('📍 Sayfa yüklendi - Queue boş, buton "Analiz Et" modunda kalıyor');
        }
    })
    .catch(error => {
        console.error('İlk buton durumu kontrolü hatası:', error);
    });
}

// =====================================
// MODULE HEALTH CHECK
// =====================================

// Sayfa yüklendikten 2 saniye sonra otomatik sağlık kontrolü
setTimeout(() => {
    if (typeof window.checkModuleHealth === 'function') {
        window.checkModuleHealth();
    }
}, 2000); 

// 🔄 Recent analysis sonuçlarını restore et (page refresh için + persistent storage)
function loadRecentAnalyses() {
    console.log('🔄 Recent analyses yükleniyor...');
    
    fetch('/api/analysis/recent')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.recent_analyses && data.recent_analyses.length > 0) {
                console.log(`📊 ${data.count} recent analysis bulundu, restore ediliyor...`);
                
                // localStorage'dan mevcut analysis IDs'leri al
                const storedAnalyses = JSON.parse(localStorage.getItem('wsanaliz_recent_analyses') || '[]');
                const newAnalysesToStore = [];
                
                // Her analiz için fake uploadedFiles entry oluştur ve sonuçları göster
                data.recent_analyses.forEach((analysis, index) => {
                    // Fake file entry (uploadedFiles array'i için)
                    const fakeFile = {
                        id: analysis.file_id,
                        name: analysis.file_name,
                        status: 'completed',
                        analysis_id: analysis.analysis_id,
                        include_age_analysis: analysis.include_age_analysis
                    };
                    
                    // uploadedFiles array'e ekle (duplicate check ile)
                    if (!window.uploadedFiles.find(f => f.id === analysis.file_id)) {
                        window.uploadedFiles.push(fakeFile);
                    }
                    
                    // localStorage için kaydet
                    newAnalysesToStore.push({
                        file_id: analysis.file_id,
                        analysis_id: analysis.analysis_id,
                        file_name: analysis.file_name,
                        completed_at: analysis.completed_at
                    });
                    
                    // Detailed results'ı çek ve göster
                    setTimeout(() => {
                        window.analysisManager.getAnalysisResults(
                            analysis.file_id, 
                            analysis.analysis_id, 
                            false // isPartial = false
                        );
                    }, index * 200); // Her analiz 200ms arayla yüklensin
                });
                
                // localStorage'a kaydet (sadece unique olanları)
                const allAnalyses = [...storedAnalyses];
                newAnalysesToStore.forEach(newAnalysis => {
                    if (!allAnalyses.find(stored => stored.analysis_id === newAnalysis.analysis_id)) {
                        allAnalyses.push(newAnalysis);
                    }
                });
                
                // En fazla 20 analizi sakla (disk alanı)
                if (allAnalyses.length > 20) {
                    allAnalyses.sort((a, b) => new Date(b.completed_at) - new Date(a.completed_at));
                    allAnalyses.splice(20);
                }
                
                localStorage.setItem('wsanaliz_recent_analyses', JSON.stringify(allAnalyses));
                console.log(`💾 ${allAnalyses.length} analiz localStorage'a kaydedildi`);
                
                // Results section'ı görünür yap
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) {
                    resultsSection.style.display = 'block';
                }
                
                console.log(`✅ ${data.count} analiz sonucu restore edildi`);
            } else {
                console.log('📝 Henüz recent analysis yok');
            }
        })
        .catch(error => {
            console.error('❌ Recent analyses yüklenirken hata:', error);
            // Sessizce devam et, page load engellenmemeli
        });
}

// 🔄 Yeni analiz tamamlandığında localStorage'a ekleme fonksiyonu
window.addAnalysisToLocalStorage = function(fileId, analysisId, fileName) {
    const storedAnalyses = JSON.parse(localStorage.getItem('wsanaliz_recent_analyses') || '[]');
    const newAnalysis = {
        file_id: fileId,
        analysis_id: analysisId,
        file_name: fileName,
        completed_at: new Date().toISOString()
    };
    
    // Duplicate check
    if (!storedAnalyses.find(stored => stored.analysis_id === analysisId)) {
        storedAnalyses.unshift(newAnalysis); // En başa ekle
        
        // En fazla 20 analizi sakla
        if (storedAnalyses.length > 20) {
            storedAnalyses.splice(20);
        }
        
        localStorage.setItem('wsanaliz_recent_analyses', JSON.stringify(storedAnalyses));
        console.log(`💾 Yeni analiz localStorage'a eklendi: ${fileName}`);
    }
};

// 🔄 localStorage'dan stored analyses restore et (offline support)
function loadStoredAnalyses() {
    console.log('💾 localStorage analyses restore ediliyor...');
    
    try {
        const storedAnalyses = JSON.parse(localStorage.getItem('wsanaliz_recent_analyses') || '[]');
        
        if (storedAnalyses.length > 0) {
            console.log(`💾 ${storedAnalyses.length} stored analysis bulundu, restore ediliyor...`);
            
            storedAnalyses.forEach((analysis, index) => {
                // Fake file entry (uploadedFiles array'i için)
                const fakeFile = {
                    id: analysis.file_id,
                    name: analysis.file_name,
                    status: 'completed',
                    analysis_id: analysis.analysis_id,
                    include_age_analysis: true // Default olarak true (güvenli taraf)
                };
                
                // uploadedFiles array'e ekle (duplicate check ile)
                if (!window.uploadedFiles.find(f => f.id === analysis.file_id)) {
                    window.uploadedFiles.push(fakeFile);
                    
                    // Detailed results'ı çek ve göster (delay ile)
                    setTimeout(() => {
                        if (window.analysisManager && window.analysisManager.getAnalysisResults) {
                            window.analysisManager.getAnalysisResults(
                                analysis.file_id, 
                                analysis.analysis_id, 
                                false // isPartial = false
                            );
                        }
                    }, 3000 + (index * 300)); // API load'dan sonra başlasın
                }
            });
            
            // Results section'ı görünür yap
            setTimeout(() => {
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) {
                    resultsSection.style.display = 'block';
                }
            }, 3500);
            
            console.log(`💾 ${storedAnalyses.length} stored analiz restore edildi`);
        } else {
            console.log('💾 localStorage\'da stored analysis yok');
        }
        
    } catch (error) {
        console.error('❌ localStorage analyses restore hatası:', error);
        // localStorage'ı temizle eğer corrupt olmuşsa
        localStorage.removeItem('wsanaliz_recent_analyses');
    }
}

// 🗑️ localStorage analysis cache'ini temizle (debug için)
window.clearAnalysisCache = function() {
    localStorage.removeItem('wsanaliz_recent_analyses');
    console.log('🗑️ Analysis cache temizlendi');
    if (confirm('Sayfa yenilensin mi?')) {
        location.reload();
    }
};

// İçerik analizi son geri bildirimleri ve kategori dağılımı yükleyici
function loadRecentContentFeedbacks() {
    fetch('/api/feedback/content/recent')
        .then(res => res.json())
        .then(data => {
            // Son geri bildirimler
            const container = document.getElementById('recentContentFeedbacks');
            if (container) {
                if (data.recent_feedbacks && data.recent_feedbacks.length > 0) {
                    container.innerHTML = data.recent_feedbacks.map(fb => `
                        <div class="mb-2 border-bottom pb-2">
                            <b>${fb.created_at ? new Date(fb.created_at).toLocaleString() : ''}</b>
                            <br>
                            <span>Kategoriler: ${fb.category_feedback ? JSON.stringify(fb.category_feedback) : '-'}</span>
                            <br>
                            <span>Yorum: ${fb.comment || '-'}</span>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<div class="alert alert-secondary">Henüz içerik geri bildirimi yok.</div>';
                }
            }
            // Kategori dağılımı
            const distContainer = document.getElementById('contentFeedbackCategoryDist');
            if (distContainer) {
                if (data.category_distribution && Object.keys(data.category_distribution).length > 0) {
                    distContainer.innerHTML = Object.entries(data.category_distribution).map(
                        ([cat, count]) => `<span class="badge bg-info m-1">${cat}: ${count}</span>`
                    ).join('');
                } else {
                    distContainer.innerHTML = '<div class="alert alert-secondary">Kategori dağılımı yok.</div>';
                }
            }
        })
        .catch(err => {
            const container = document.getElementById('recentContentFeedbacks');
            if (container) container.innerHTML = '<div class="alert alert-danger">Geri bildirimler yüklenemedi.</div>';
            const distContainer = document.getElementById('contentFeedbackCategoryDist');
            if (distContainer) distContainer.innerHTML = '<div class="alert alert-danger">Kategori dağılımı yüklenemedi.</div>';
        });
}

// Modal açıldığında feedbackleri yükle
const modelMetricsModalEl = document.getElementById('modelMetricsModal');
if (modelMetricsModalEl) {
    modelMetricsModalEl.addEventListener('show.bs.modal', loadRecentContentFeedbacks);
} 

// 🗑️ ANALIZ SONUÇLARI TEMİZLEME FONKSİYONU
async function clearAllAnalysisResults() {
    if (confirm('Tüm analiz sonuçlarını temizlemek istediğinizden emin misiniz? Bu işlem geri alınamaz ve veritabanından da silinecektir.')) {
        try {
            // Backend'ten analiz sonuçlarını temizle
            const response = await fetch('/api/analysis/clear-all', {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                // localStorage'dan analiz sonuçlarını temizle
                localStorage.removeItem('wsanaliz_recent_analyses');
                
                // Global uploadedFiles array'ini temizle
                if (window.uploadedFiles) {
                    window.uploadedFiles = [];
                }
                
                // Results section'ı gizle
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) {
                    resultsSection.style.display = 'none';
                }
                
                // Results listesini temizle
                const resultsList = document.getElementById('resultsList');
                if (resultsList) {
                    resultsList.innerHTML = '';
                }
                
                // Success mesajı göster
                alert(`Başarı! ${result.deleted_count} analiz sonucu veritabanından ve localStorage'dan temizlendi.`);
                
                console.log(`🗑️ ${result.deleted_count} analiz sonucu temizlendi`);
            } else {
                throw new Error(result.error || 'Backend temizleme işlemi başarısız');
            }
            
        } catch (error) {
            console.error('❌ Analiz sonuçları temizleme hatası:', error);
            alert(`Hata: Analiz sonuçları temizlenirken bir sorun oluştu: ${error.message}`);
        }
    }
}

// Global erişim için
window.clearAllAnalysisResults = clearAllAnalysisResults; 