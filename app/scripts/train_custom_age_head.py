import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging

# Betik seviyesinde bir logger oluşturuyoruz
# Log seviyesini DEBUG yapalım ki her şey görünsün
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.critical("train_custom_age_head.py betiği ÇALIŞMAYA BAŞLADI.")

# Model Tanımlama
class CustomAgeHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super(CustomAgeHead, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def main(args):
    logger.info(f"Argümanlar: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"Kullanılacak cihaz: {device}")

    logger.info(f"Embedding ve yaş verisi yükleniyor: {args.embeddings_path}")
    try:
        data = np.load(args.embeddings_path)
        embeddings = data['embeddings']
        ages = data['ages']
        logger.info(f"{len(embeddings)} adet embedding ve yaş bilgisi yüklendi.")
    except Exception as e:
        logger.error(f"Veri yüklenirken hata oluştu: {e}", exc_info=True)
        return

    ages = ages.astype(np.float32).reshape(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, ages, test_size=args.test_size, random_state=42
    )
    logger.info(f"Eğitim seti boyutu: {len(X_train)}, Doğrulama seti boyutu: {len(X_val)}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # KRİTİK: EMBEDDING NORMALİZASYONU (inference ile tutarlılık için)
    X_train_tensor = X_train_tensor / torch.norm(X_train_tensor, dim=1, keepdim=True)
    X_val_tensor = X_val_tensor / torch.norm(X_val_tensor, dim=1, keepdim=True)
    logger.info("Embeddings normalized during training (to match inference normalization)")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    hidden_dims_list = [int(d.strip()) for d in args.hidden_dims.split(',') if d.strip()]
    model = CustomAgeHead(input_dim=args.input_embedding_dim, hidden_dims=hidden_dims_list, output_dim=1).to(device)
    logger.info(f"Model tanımlandı:\n{model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    output_model_name = args.output_model_name
    if not output_model_name.endswith(".pth"):
        output_model_name += ".pth"
    
    os.makedirs(args.output_model_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_model_dir, output_model_name)

    logger.info(f"Eğitim başlıyor ({args.epochs} epoch)...")
    for epoch in range(args.epochs):
        model.train()
        running_train_loss = 0.0
        for batch_embeddings, batch_ages in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_ages)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_embeddings.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_embeddings, batch_ages in val_loader:
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_ages)
                running_val_loss += loss.item() * batch_embeddings.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Eğitim Kaybı: {epoch_train_loss:.4f}, Doğrulama Kaybı: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"En iyi model kaydedildi: {model_save_path} (Doğrulama Kaybı: {best_val_loss:.4f})")

    logger.info("Eğitim tamamlandı.")
    logger.info(f"Son eğitilmiş model (en iyi doğrulama kaybına sahip) şuraya kaydedildi: {model_save_path}")

if __name__ == "__main__":
    logger.critical("train_custom_age_head.py betiği __main__ bloğuna GİRDİ.")
    parser = argparse.ArgumentParser(description="InsightFace embeddinglerini kullanarak özel bir yaş tahmin başlığı (CustomAgeHead) eğitir.")
    parser.add_argument("--embeddings_path", type=str, required=True, 
                        help="Hazırlanmış embeddingleri ve yaşları içeren .npz dosyasının yolu.")
    parser.add_argument("--output_model_dir", type=str, required=True, 
                        help="Eğitilmiş modelin kaydedileceği dizin.")
    parser.add_argument("--output_model_name", type=str, default="custom_age_head_v1.pth",
                        help="Kaydedilecek model dosyasının adı (örn: custom_age_head_v1.pth).")
    parser.add_argument("--input_embedding_dim", type=int, default=512, 
                        help="Giriş embedding vektörünün boyutu.")
    parser.add_argument("--hidden_dims", type=str, default="256,128", 
                        help="Modeldeki gizli katmanların boyutları (virgülle ayrılmış, örn: '256,128').")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Eğitim epoch sayısı.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch boyutu.")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Öğrenme oranı.")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Doğrulama seti için ayrılacak veri oranı.")
    parser.add_argument("--use_gpu", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="GPU kullanımı (True/False).")
        
    parsed_args = parser.parse_args()
    logger.critical(f"Argümanlar parse edildi: {parsed_args}")
    main(parsed_args)
    logger.critical("main(parsed_args) çağrısı TAMAMLANDI.") 