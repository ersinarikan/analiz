#!/usr/bin/env python3
"""
CLIP Fine-tuning Production Deployment Script
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

class CLIPProductionDeployer:
    """CLIP Production Deployment Manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deployment_log = []
        
    def log(self, message, level="INFO"):
        """Deployment log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)
    
    def check_prerequisites(self):
        """Deployment ön koşullarını kontrol et"""
        self.log("🔍 Checking deployment prerequisites...")
        
        checks = []
        
        # 1. Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 8:
            checks.append(("Python Version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python Version", False, f"Required: 3.8+, Found: {python_version.major}.{python_version.minor}"))
        
        # 2. Required directories
        required_dirs = [
            "storage/models/clip",
            "storage/models/clip/versions",
            "storage/processed/logs",
            "app/services",
            "app/models",
            "app/routes"
        ]
        
        for dir_path in required_dirs:
            exists = (self.project_root / dir_path).exists()
            checks.append((f"Directory: {dir_path}", exists, "Exists" if exists else "Missing"))
        
        # 3. Required files
        required_files = [
            "app/services/clip_training_service.py",
            "app/services/clip_version_service.py",
            "app/models/clip_training.py",
            "app/routes/clip_training_routes.py",
            "requirements.txt"
        ]
        
        for file_path in required_files:
            exists = (self.project_root / file_path).exists()
            checks.append((f"File: {file_path}", exists, "Exists" if exists else "Missing"))
        
        # 4. Database file (wsanaliz_dev.db - tek veritabanı kullanılıyor)
        db_paths = [
            self.project_root / "app" / "wsanaliz_dev.db",      # Ana konum
            self.project_root / "instance" / "wsanaliz_dev.db", # Flask instance klasörü
            self.project_root / "wsanaliz_dev.db"               # Root klasör
        ]
        
        existing_dbs = [path for path in db_paths if path.exists()]
        if existing_dbs:
            db_path = existing_dbs[0]
            db_size = db_path.stat().st_size / 1024  # KB cinsinden
            db_info = f"wsanaliz_dev.db ({db_size:.1f} KB)"
            checks.append(("Database", True, db_info))
            self.log(f"📊 Veritabanı bulundu: {db_info}")
        else:
            # Database missing is not critical - will be created on first run
            self.log("⚠️ wsanaliz_dev.db bulunamadı - ilk çalıştırmada oluşturulacak")
            checks.append(("Database", True, "İlk çalıştırmada oluşturulacak"))
        
        # Results
        passed = sum(1 for _, status, _ in checks if status)
        total = len(checks)
        
        self.log(f"Prerequisites check: {passed}/{total} passed")
        
        for name, status, detail in checks:
            status_icon = "✅" if status else "❌"
            self.log(f"  {status_icon} {name}: {detail}")
        
        return passed == total
    
    def create_production_config(self):
        """Production konfigürasyonu oluştur"""
        self.log("⚙️ Creating production configuration...")
        
        production_config = {
            "CLIP_TRAINING": {
                "DEFAULT_PARAMS": {
                    "learning_rate": 1e-5,
                    "batch_size": 16,
                    "epochs": 10,
                    "train_split": 0.8,
                    "min_feedback_count": 100
                },
                "CATEGORIES": [
                    "violence",
                    "adult_content", 
                    "harassment",
                    "weapon",
                    "drug"
                ],
                "MODEL_SETTINGS": {
                    "model_name": "ViT-H-14-378-quickgelu",
                    "pretrained": "dfn5b",
                    "device": "auto"  # auto-detect GPU/CPU
                },
                "STORAGE": {
                    "models_path": "storage/models/clip",
                    "versions_path": "storage/models/clip/versions",
                    "backup_path": "storage/models/clip/backups",
                    "logs_path": "storage/processed/logs"
                },
                "TRAINING": {
                    "max_concurrent_sessions": 1,
                    "auto_cleanup_failed": True,
                    "checkpoint_frequency": 1,  # Save every epoch
                    "early_stopping_patience": 3
                }
            },
            "DEPLOYMENT": {
                "version": "1.0.0",
                "deployed_at": datetime.now().isoformat(),
                "environment": "production"
            }
        }
        
        config_path = self.project_root / "clip_production_config.json"
        with open(config_path, 'w') as f:
            json.dump(production_config, f, indent=2)
        
        self.log(f"✅ Production config created: {config_path}")
        return config_path
    
    def setup_production_directories(self):
        """Production dizin yapısını oluştur"""
        self.log("📁 Setting up production directories...")
        
        production_dirs = [
            "storage/models/clip/backups",
            "storage/models/clip/versions/archive",
            "storage/processed/logs/training",
            "storage/processed/logs/deployment",
            "storage/processed/exports"
        ]
        
        for dir_path in production_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.log(f"  ✅ Created: {dir_path}")
    
    def create_deployment_scripts(self):
        """Deployment script'leri oluştur"""
        self.log("📜 Creating deployment scripts...")
        
        # 1. Start script
        start_script = """#!/bin/bash
# CLIP Training Production Start Script

echo "Starting CLIP Training Production Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export FLASK_ENV=production
export FLASK_DEBUG=0

# Start the application
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 wsgi:app

echo "CLIP Training Server started on port 5000"
"""
        
        with open(self.project_root / "start_production.sh", 'w', encoding='utf-8') as f:
            f.write(start_script)
        
        # 2. Stop script
        stop_script = """#!/bin/bash
# CLIP Training Production Stop Script

echo "Stopping CLIP Training Production Server..."

# Find and kill gunicorn processes
pkill -f "gunicorn.*wsgi:app"

echo "CLIP Training Server stopped"
"""
        
        with open(self.project_root / "stop_production.sh", 'w', encoding='utf-8') as f:
            f.write(stop_script)
        
        # 3. Health check script
        health_script = """#!/bin/bash
# CLIP Training Health Check Script

echo "CLIP Training Health Check..."

# Check if server is running
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/clip-training/statistics)

if [ "$response" = "200" ]; then
    echo "CLIP Training Server is healthy"
    exit 0
else
    echo "CLIP Training Server is not responding (HTTP: $response)"
    exit 1
fi
"""
        
        with open(self.project_root / "health_check.sh", 'w', encoding='utf-8') as f:
            f.write(health_script)
        
        # Make scripts executable
        for script in ["start_production.sh", "stop_production.sh", "health_check.sh"]:
            os.chmod(self.project_root / script, 0o755)
        
        self.log("✅ Deployment scripts created")
    
    def create_monitoring_dashboard(self):
        """Monitoring dashboard oluştur"""
        self.log("📊 Creating monitoring dashboard...")
        
        dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>CLIP Training Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 12px; color: #666; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 CLIP Training Monitoring Dashboard</h1>
        
        <div class="card">
            <h3>System Status</h3>
            <div id="system-status">Loading...</div>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>
        
        <div class="card">
            <h3>Training Statistics</h3>
            <div id="training-stats">Loading...</div>
        </div>
        
        <div class="card">
            <h3>Active Sessions</h3>
            <div id="active-sessions">Loading...</div>
        </div>
        
        <div class="card">
            <h3>Model Versions</h3>
            <div id="model-versions">Loading...</div>
        </div>
    </div>
    
    <script>
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api/clip-training/${endpoint}`);
                return await response.json();
            } catch (error) {
                return { error: error.message };
            }
        }
        
        async function refreshData() {
            // System status
            const stats = await fetchData('statistics');
            document.getElementById('system-status').innerHTML = stats.success ? 
                `<span class="status-good">✅ System Online</span>` : 
                `<span class="status-error">❌ System Error</span>`;
            
            // Training statistics
            if (stats.success) {
                const data = stats.data;
                document.getElementById('training-stats').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${data.total_feedbacks}</div>
                        <div class="metric-label">Total Feedbacks</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.manual_feedbacks}</div>
                        <div class="metric-label">Manual Feedbacks</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.ready_for_training ? 'Yes' : 'No'}</div>
                        <div class="metric-label">Ready for Training</div>
                    </div>
                `;
            }
            
            // Active sessions
            const sessions = await fetchData('sessions');
            if (sessions.success) {
                const activeSessions = sessions.data.sessions.filter(s => s.status === 'training');
                document.getElementById('active-sessions').innerHTML = activeSessions.length > 0 ?
                    activeSessions.map(s => `<div>Session ${s.id}: ${s.version_name} (${s.status})</div>`).join('') :
                    '<div>No active training sessions</div>';
            }
            
            // Model versions
            const versions = await fetchData('versions');
            if (versions.success) {
                document.getElementById('model-versions').innerHTML = 
                    versions.data.versions.map(v => 
                        `<div>${v.name} (${v.type}) ${v.is_active ? '- Active' : ''}</div>`
                    ).join('');
            }
        }
        
        // Auto refresh every 30 seconds
        setInterval(refreshData, 30000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>"""
        
        dashboard_path = self.project_root / "app" / "templates" / "clip_monitoring.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        self.log(f"✅ Monitoring dashboard created: {dashboard_path}")
    
    def save_deployment_log(self):
        """Deployment log'unu kaydet"""
        log_path = self.project_root / "storage" / "processed" / "logs" / "deployment" / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.deployment_log))
        
        self.log(f"📝 Deployment log saved: {log_path}")
    
    def deploy(self):
        """Ana deployment fonksiyonu"""
        self.log("🚀 Starting CLIP Fine-tuning Production Deployment...")
        self.log("=" * 60)
        
        try:
            # 1. Prerequisites check
            if not self.check_prerequisites():
                self.log("❌ Prerequisites check failed. Deployment aborted.")
                return False
            
            # 2. Setup directories
            self.setup_production_directories()
            
            # 3. Create configuration
            self.create_production_config()
            
            # 4. Create deployment scripts
            self.create_deployment_scripts()
            
            # 5. Create monitoring dashboard
            self.create_monitoring_dashboard()
            
            # 6. Save deployment log
            self.save_deployment_log()
            
            self.log("=" * 60)
            self.log("🎉 CLIP Fine-tuning Production Deployment COMPLETED!")
            self.log("=" * 60)
            self.log("📋 Next Steps:")
            self.log("  1. Review production configuration: clip_production_config.json")
            self.log("  2. Start production server: ./start_production.sh")
            self.log("  3. Check health: ./health_check.sh")
            self.log("  4. Monitor at: http://localhost:5000/clip-monitoring")
            self.log("  5. View logs: storage/processed/logs/")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Deployment failed: {str(e)}", "ERROR")
            return False

def main():
    """Ana deployment fonksiyonu"""
    deployer = CLIPProductionDeployer()
    success = deployer.deploy()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 