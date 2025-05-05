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
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', root=model_path, providers=['CPUExecutionProvider'])
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

class EmbeddingAgeDataset(Dataset):
    def __init__(self, embeddings, ages):
        self.embeddings = embeddings
        self.ages = ages

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.ages[idx], dtype=torch.float32)

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

def main():
    print("Eğitim başlıyor...")
    print("=" * 50)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'storage', 'dataset', 'age', 'UTKFace')
    model_path = os.path.join(base_dir, 'storage', 'models', 'age', 'buffalo_l')
    emb_path = os.path.join(base_dir, 'storage', 'dataset', 'age', 'utkface_embeddings.npy')
    age_path = os.path.join(base_dir, 'storage', 'dataset', 'age', 'utkface_ages.npy')
    model_dir = os.path.join(base_dir, 'storage', 'models', 'age', 'buffalo_l', 'versions', 'v1')
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Embedding dosyaları yoksa çıkar ve kaydet
    if not (os.path.exists(emb_path) and os.path.exists(age_path)):
        embeddings, ages = extract_and_save_embeddings(dataset_path, model_path, emb_path, age_path)
    else:
        embeddings = np.load(emb_path)
        ages = np.load(age_path)
        print(f"Toplam {len(embeddings)} embedding ve yaş yüklendi.")
        print(f"Yaş aralığı: {int(np.min(ages))} - {int(np.max(ages))}")

    X_train, X_val, y_train, y_val = train_test_split(embeddings, ages, test_size=0.2, random_state=42)
    print(f"Eğitim seti: {len(X_train)} örnek")
    print(f"Validasyon seti: {len(X_val)} örnek")
    train_dataset = EmbeddingAgeDataset(X_train, y_train)
    val_dataset = EmbeddingAgeDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nKullanılan cihaz: {device}")
    model = CustomAgeHead().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_state = None
    print("\nModel eğitimi başlıyor...")
    print("=" * 50)
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for emb, age in train_loader:
            emb, age = emb.to(device), age.to(device)
            optimizer.zero_grad()
            output = model(emb).squeeze()
            loss = criterion(output, age)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for emb, age in val_loader:
                emb, age = emb.to(device), age.to(device)
                output = model(emb).squeeze()
                loss = criterion(output, age)
                val_loss += loss.item()
                preds.extend(output.cpu().numpy())
                targets.extend(age.cpu().numpy())
        val_loss /= len(val_loader)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        print(f"Epoch {epoch+1}/50: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"Yeni en iyi model kaydedildi! (Val Loss: {val_loss:.4f})")

    # Model ve metadata kaydet
    model_path_out = os.path.join(model_dir, 'custom_age_head.pth')
    torch.save(best_model_state, model_path_out)
    print(f"\nEn iyi model kaydedildi: {model_path_out}")
    metadata = {
        "version": "v1",
        "base_version": "base",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "UTKFace",
        "sample_size": int(len(embeddings)),
        "age_range": {
            "min": int(np.min(ages)),
            "max": int(np.max(ages))
        },
        "training_params": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata kaydedildi: {metadata_path}")
    print("\nEğitim tamamlandı!")
    print("=" * 50)

if __name__ == '__main__':
    main() 