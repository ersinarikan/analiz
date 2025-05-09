"""
Gerekli Kütüphaneler:
- torch
- torchvision
- insightface
- opencv-python
- pillow
- scikit-learn
- numpy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime
import cv2
import insightface

# Embedding çıkarma ve kaydetme fonksiyonu

def extract_and_save_embeddings(dataset_path, model_path, emb_path, age_path):
    print("Embedding dosyaları bulunamadı, şimdi çıkarılıyor...")
    print("NOT: İsimlendirmede buffalo_x kullanılmasına rağmen, şu anda buffalo_sc modeli kullanılmaktadır.")
    
    # buffalo_sc modeli kullanılıyor ama buffalo_x olarak geçiyoruz
    source_model_name = "buffalo_sc"  # Gerçekte kullanacağımız model
    
    face_analyzer = insightface.app.FaceAnalysis(name=source_model_name, providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    embeddings = []
    ages = []
    image_count = 0
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg.chip.jpg'):
            age = int(filename.split('_')[0])
            if age > 100:
                age = 100
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_analyzer.get(image)
            if faces and hasattr(faces[0], 'embedding'):
                embeddings.append(faces[0].embedding)
            else:
                embeddings.append(np.zeros(512))
            ages.append(age)
            image_count += 1
            if image_count % 100 == 0:
                print(f"{image_count} görsel işlendi...")
    embeddings = np.array(embeddings)
    ages = np.array(ages)
    np.save(emb_path, embeddings)
    np.save(age_path, ages)
    print(f"Embeddingler kaydedildi: {emb_path}")
    print(f"Yaşlar kaydedildi: {age_path}")
    return embeddings, ages

# Özel veri seti sınıfı
class AgeEstimationDataset(Dataset):
    def __init__(self, embeddings, ages):
        self.embeddings = embeddings
        self.ages = ages
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32),
            'age': torch.tensor(self.ages[idx], dtype=torch.float32)
        }

# Özel model sınıfı
class CustomAgeHead(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Ayarlar
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Embedding ve yaş dosyalarının yolları
    embedding_path = os.path.join(base_dir, 'storage', 'embeddings.npy')
    age_path = os.path.join(base_dir, 'storage', 'ages.npy')
    
    # Dataset yolu
    dataset_path = os.path.join(base_dir, 'storage', 'dataset', 'age', 'utkface_aligned_cropped', 'crop_part1')
    
    # Model kaydetme yolu
    model_path = os.path.join(base_dir, 'storage', 'models', 'age', 'buffalo_x')
    
    # Model versiyonlama
    model_dir = os.path.join(base_dir, 'storage', 'models', 'age', 'buffalo_x', 'versions', 'v1')
    os.makedirs(model_dir, exist_ok=True)
    
    # Embeddinglari yükle veya oluştur
    if os.path.exists(embedding_path) and os.path.exists(age_path):
        print("Var olan embedding ve yaş dosyaları yükleniyor...")
        embeddings = np.load(embedding_path)
        ages = np.load(age_path)
    else:
        embeddings, ages = extract_and_save_embeddings(
            dataset_path, model_path, embedding_path, age_path)
    
    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, ages, test_size=0.2, random_state=42)
    
    # Veri setlerini oluştur
    train_dataset = AgeEstimationDataset(X_train, y_train)
    valid_dataset = AgeEstimationDataset(X_test, y_test)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64)
    
    # Modeli, kaybı ve optimize ediciyi tanımla
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomAgeHead().to(device)
    criterion = nn.L1Loss()  # MAE kaybı
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Eğitim döngüsü
    num_epochs = 30
    train_losses = []
    valid_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            ages = batch['age'].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        model.eval()
        epoch_valid_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in valid_loader:
                embeddings = batch['embedding'].to(device)
                ages = batch['age'].unsqueeze(1).to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, ages)
                
                epoch_valid_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(ages.cpu().numpy().flatten())
        
        epoch_train_loss /= len(train_loader)
        epoch_valid_loss /= len(valid_loader)
        
        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)
        
        mae = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {epoch_train_loss:.4f}, Valid Loss = {epoch_valid_loss:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        
        # Her n epochta bir modeli kaydet
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model_file = os.path.join(model_dir, f"age_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_file)
            print(f"Model kaydedildi: {model_file}")
    
    # Eğitim sonuçlarını kaydet
    training_history = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'date': datetime.now().isoformat(),
        'epochs': num_epochs
    }
    
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f)
    
    print("Eğitim tamamlandı ve sonuçlar kaydedildi.") 