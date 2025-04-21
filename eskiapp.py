from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, url_for, Response
import os
import cv2
import torch
import sys
from PIL import Image
from werkzeug.utils import secure_filename
import tempfile
import time
import glob
import json
import shutil
import atexit
import threading
import schedule

# CLIP'i manuel olarak import etmeyi dene
try:
    import clip
except ImportError:
    print("CLIP modülü yüklenemedi. Yükleniyor...")
    os.system('pip install git+https://github.com/openai/CLIP.git')
    import clip

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
app.config['TEMP_FRAMES_FOLDER'] = os.path.join(app.root_path, 'temp_frames')
app.config['FEEDBACK_FOLDER'] = os.path.join(app.root_path, 'feedback_data')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Klasörleri oluştur
for folder in [app.config['UPLOAD_FOLDER'], 
               app.config['TEMP_FRAMES_FOLDER'],
               app.config['FEEDBACK_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# CLIP modelini yükle
device = "cpu"  # GPU olmadığı için direkt CPU kullanıyoruz
model, preprocess = clip.load("ViT-B/32", device=device)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    cleanup_old_frames()  # Eski frame'leri temizle
    
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Video mu görüntü mü kontrol et
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            results, frame_path = analyze_video(filepath)
            if frame_path:
                # Frame path'i URL'e çevir
                frame_url = url_for('serve_frame', filename=os.path.basename(frame_path))
            else:
                frame_url = None
        else:
            results = analyze_image(filepath)
            frame_url = None

        # Dosyayı sil
        os.remove(filepath)

        return jsonify({
            'results': results,
            'frame_path': frame_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_image(image_input):
    try:
        # Eğer input bir dosya yolu ise
        if isinstance(image_input, str):
            try:
                # Dosyayı güvenli bir şekilde aç ve kopyala
                with Image.open(image_input) as img:
                    # RGBA modundaysa RGB'ye dönüştür
                    if img.mode == 'RGBA':
                        # Beyaz arka plan oluştur
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        # RGBA görüntüyü RGB'ye dönüştür
                        background.paste(img, mask=img.split()[3])  # 3 = alpha channel
                        image_copy = background
                    else:
                        image_copy = img.convert('RGB')
                    
                    # Preprocessi bellekteki kopya üzerinde yap
                    image = preprocess(image_copy).unsqueeze(0).to(device)
                    
                # Dosya adını hazırla
                filename = os.path.basename(image_input)
                temp_filename = f'temp_{int(time.time()*1000)}_{secure_filename(filename)}'
                temp_path = os.path.join(app.config['TEMP_FRAMES_FOLDER'], temp_filename)
                
                # Kopya dosyayı temp_frames klasörüne kaydet
                image_copy.save(temp_path, 'JPEG', quality=95)
                image_url = url_for('serve_frame', filename=temp_filename)
                
            except Exception as e:
                raise Exception(f"Dosya işleme hatası: {str(e)}")
                
        # Eğer input bir PIL Image nesnesi ise
        elif isinstance(image_input, Image.Image):
            try:
                # RGBA modundaysa RGB'ye dönüştür
                if image_input.mode == 'RGBA':
                    background = Image.new('RGB', image_input.size, (255, 255, 255))
                    background.paste(image_input, mask=image_input.split()[3])
                    image_copy = background
                else:
                    image_copy = image_input.convert('RGB')
                
                image = preprocess(image_copy).unsqueeze(0).to(device)
                
                # Geçici dosya oluştur
                temp_filename = f'frame_{int(time.time()*1000)}.jpg'
                temp_path = os.path.join(app.config['TEMP_FRAMES_FOLDER'], temp_filename)
                
                # Kopya dosyayı kaydet
                image_copy.save(temp_path, 'JPEG', quality=95)
                image_url = url_for('serve_frame', filename=temp_filename)
                
            except Exception as e:
                raise Exception(f"Görüntü işleme hatası: {str(e)}")
        else:
            raise ValueError("Geçersiz input tipi")

        # CLIP analizi
        try:
            text = clip.tokenize(["safe", "nsfw", "violence", "weapon", "abuse"]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                scores = similarity[0].tolist()

            labels = ["Güvenli", "Tehlikeli/Yetişkin", "Şiddet", "Silah Kullanımı", "İstismar"]
            
            results = {
                'results': {labels[i]: float(scores[i]) for i in range(len(labels))},
                'image_path': image_url
            }
            return results
            
        except Exception as e:
            raise Exception(f"CLIP analiz hatası: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Görüntü analiz hatası: {str(e)}")

# Global değişkenler
progress_data = {
    'current_file': 0,
    'total_files': 0,
    'percent': 0,
    'frame': 0,
    'current_filename': ''
}

@app.route('/analyze_progress')
def progress():
    def generate():
        last_data = None
        while True:
            current_data = {
                'current_file': progress_data['current_file'],
                'total_files': progress_data['total_files'],
                'percent': progress_data['percent'],
                'frame': progress_data['frame'],
                'filename': progress_data['current_filename']
            }
            
            # Sadece veri değiştiğinde gönder
            if current_data != last_data:
                yield f"data: {json.dumps(current_data)}\n\n"
                last_data = current_data.copy()
            
            time.sleep(0.1)
            
    return Response(generate(), mimetype='text/event-stream')

def analyze_video(video_path, fps=1):
    try:
        cap = cv2.VideoCapture(video_path)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FPS kontrolü
        if fps > video_fps:
            fps = video_fps
        frame_interval = max(1, video_fps // fps)
        
        print(f"Video FPS: {video_fps}")
        print(f"Analiz FPS: {fps}")
        print(f"Frame interval: {frame_interval}")
        print(f"Toplam kare sayısı: {total_frames}")
        
        all_scores = []
        highest_score = -1
        highest_frame = None
        frame_number = 0
        processed_frames = 0

        # Progress verilerini sıfırla
        progress_data['percent'] = 0
        progress_data['frame'] = 0
        progress_data['current_filename'] = os.path.basename(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                processed_frames += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                try:
                    scores = analyze_image(pil_image)
                    all_scores.append(scores)
                    
                    current_max_score = max(scores['results'].values())
                    if current_max_score > highest_score:
                        highest_score = current_max_score
                        highest_frame = frame.copy()
                        
                    # İlerleme durumunu yazdır
                    current_percent = int((frame_number / total_frames) * 100)
                    progress_data['percent'] = current_percent
                    progress_data['frame'] = processed_frames
                    
                except Exception as e:
                    print(f"Kare analiz hatası: {str(e)}")
                    continue

            frame_number += 1

        cap.release()
        
        print(f"Toplam işlenen kare sayısı: {processed_frames}")

        if not all_scores:
            raise Exception("Video analizi başarısız oldu: Hiç kare analiz edilemedi")

        # Ortalama skorları hesapla
        avg_scores = {}
        for key in all_scores[0]['results'].keys():
            avg_scores[key] = sum(score['results'][key] for score in all_scores) / len(all_scores)

        if highest_frame is not None:
            # Dosya adındaki özel karakterleri temizle
            safe_filename = secure_filename(os.path.basename(video_path))
            frame_filename = f'highest_frame_{safe_filename}.jpg'
            frame_path = os.path.join(app.config['TEMP_FRAMES_FOLDER'], frame_filename)
            
            # Frame'i kaydet
            cv2.imwrite(frame_path, highest_frame)
            print(f"Frame kaydedildi: {frame_path}")  # Debug için
            
            # Frame URL'ini döndür
            return avg_scores, frame_filename  # Sadece dosya adını döndür

        # İşlem tamamlandığında %100'e tamamla
        progress_data['percent'] = 100
        
        return avg_scores, None

    except Exception as e:
        raise Exception(f"Video analiz hatası: {str(e)}")

@app.route('/help')
def help():
    return render_template('documentation.html')

# Temp frames için route ekle
@app.route('/temp_frames/<path:filename>')
def serve_frame(filename):
    try:
        # Güvenli dosya adı oluştur
        safe_filename = secure_filename(filename)
        # Tam yolu al
        frame_path = os.path.join(app.config['TEMP_FRAMES_FOLDER'], safe_filename)
        
        if not os.path.exists(frame_path):
            print(f"Frame bulunamadı: {frame_path}")  # Debug için
            return "Frame bulunamadı", 404
            
        return send_from_directory(app.config['TEMP_FRAMES_FOLDER'], safe_filename)
    except Exception as e:
        print(f"Frame servis hatası: {str(e)}")  # Debug için
        return str(e), 404

# Frame'leri temizlemek için yardımcı fonksiyon
def cleanup_old_frames():
    temp_dir = 'temp_frames'
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < time.time() - 3600:  # 1 saat önce
                os.remove(file_path)
        except Exception as e:
            print(f"Dosya temizleme hatası: {str(e)}")

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        file_path = data.get('file_path')
        is_correct = data.get('is_correct')
        selected_labels = data.get('selected_labels', [])
        frame_path = data.get('frame_path')

        if not is_correct and not selected_labels:
            return jsonify({'error': 'Yanlış tespit için etiket seçilmedi'}), 400

        # Geri bildirim klasörü oluştur
        feedback_dir = app.config['FEEDBACK_FOLDER']
        os.makedirs(feedback_dir, exist_ok=True)

        # Eğer frame_path varsa, frame'i feedback klasörüne kopyala
        if frame_path:
            frame_filename = os.path.basename(frame_path)
            frame_source = os.path.join(app.config['TEMP_FRAMES_FOLDER'], frame_filename)
            frame_dest = os.path.join(feedback_dir, frame_filename)
            if os.path.exists(frame_source):
                import shutil
                shutil.copy2(frame_source, frame_dest)
                feedback_image_path = frame_dest
            else:
                feedback_image_path = None
        else:
            # Orijinal dosyayı feedback klasörüne kopyala
            feedback_image_path = os.path.join(feedback_dir, secure_filename(file_path))
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file_path)):
                import shutil
                shutil.copy2(
                    os.path.join(app.config['UPLOAD_FOLDER'], file_path),
                    feedback_image_path
                )

        # Geri bildirimi kaydet
        feedback_file = os.path.join(app.root_path, "feedback.txt")
        
        # Dosya yoksa oluştur
        if not os.path.exists(feedback_file):
            open(feedback_file, 'w', encoding='utf-8').close()

        with open(feedback_file, "a", encoding='utf-8') as f:
            if feedback_image_path and os.path.exists(feedback_image_path):
                if not is_correct:
                    for label in selected_labels:
                        f.write(f"{feedback_image_path}\t{label}\n")
                else:
                    f.write(f"{feedback_image_path}\tGüvenli\n")

        print(f"Geri bildirim kaydedildi: {feedback_image_path}")  # Debug için
        return jsonify({'message': 'Geri bildirim kaydedildi'})

    except Exception as e:
        print(f"Geri bildirim hatası: {str(e)}")  # Hata ayıklama için
        return jsonify({'error': str(e)}), 500

@app.route('/get_labels')
def get_labels():
    labels = ["Güvenli", "Tehlikeli/Yetişkin", "Şiddet", "Silah Kullanımı", "İstismar"]
    return jsonify(labels)

@app.route('/analyze_folder', methods=['POST'])
def analyze_folder():
    try:
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({'error': 'Dosya yüklenmedi'}), 400

        # İlerleme bilgisini başlat
        total_files = len([f for f in files if f and allowed_file(f.filename)])
        progress_data['total_files'] = total_files
        progress_data['current_file'] = 0
        progress_data['percent'] = 0
        
        results = []
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        
        try:
            for i, file in enumerate(files, 1):
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(temp_dir, filename)
                    
                    # Dosyayı kaydet ve hemen analiz et
                    try:
                        file.save(file_path)
                        progress_data['current_file'] = i
                        progress_data['current_filename'] = filename
                        progress_data['percent'] = 0
                        
                        # Video mu görüntü mü kontrol et
                        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                            file_results, frame_path = analyze_video(file_path)
                            result_item = {
                                'filename': filename,
                                'results': file_results,
                                'frame_path': url_for('serve_frame', filename=frame_path) if frame_path else None
                            }
                        else:
                            analysis_result = analyze_image(file_path)
                            result_item = {
                                'filename': filename,
                                'results': analysis_result['results'],
                                'image_path': analysis_result['image_path']
                            }
                            progress_data['percent'] = 100
                        
                        results.append(result_item)
                        
                        # Her dosyadan sonra hemen temizle
                        os.remove(file_path)
                        
                    except Exception as e:
                        print(f"Dosya işleme hatası ({filename}): {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Analiz hatası: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Geçici klasörü temizle
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Geçici klasör temizleme hatası: {str(e)}")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Genel hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        data = request.json
        epochs = int(data.get('epochs', 10))
        batch_size = int(data.get('batch_size', 32))
        learning_rate = float(data.get('learning_rate', 0.001))
        optimizer_name = data.get('optimizer', 'adam')

        feedback_file = os.path.join(app.root_path, "feedback.txt")
        if not os.path.exists(feedback_file):
            return jsonify({'error': 'Geri bildirim verisi bulunamadı (feedback.txt yok)'}), 400

        # Geri bildirimleri oku
        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedback_data = f.readlines()

        if not feedback_data:
            return jsonify({'error': 'Yeterli geri bildirim verisi yok (feedback.txt boş)'}), 400

        print(f"Toplam geri bildirim sayısı: {len(feedback_data)}")  # Debug için

        # Geri bildirimlerden veri seti oluştur
        training_data = []
        missing_files = []
        for line in feedback_data:
            try:
                file_path, label = line.strip().split('\t')
                if os.path.exists(file_path):
                    training_data.append((file_path, label))
                else:
                    missing_files.append(file_path)
                    print(f"Dosya bulunamadı: {file_path}")  # Debug için
            except Exception as e:
                print(f"Veri okuma hatası: {str(e)}")  # Debug için
                continue

        if not training_data:
            error_msg = 'Geçerli eğitim verisi bulunamadı.\n'
            error_msg += f'Toplam geri bildirim: {len(feedback_data)}\n'
            error_msg += f'Bulunamayan dosyalar: {len(missing_files)}\n'
            if missing_files:
                error_msg += f'Örnek eksik dosya: {missing_files[0]}'
            return jsonify({'error': error_msg}), 400

        print(f"Eğitim için kullanılabilir veri sayısı: {len(training_data)}")  # Debug için

        # Model eğitimi başlat
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.train()

            # Optimizer seçimi
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

            total_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                for file_path, label in training_data:
                    try:
                        # Görüntüyü yükle ve ön işleme
                        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
                        text = clip.tokenize([label]).to(device)

                        # Forward pass
                        logits_per_image, logits_per_text = model(image, text)
                        loss = (logits_per_image + logits_per_text).mean()

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                    except Exception as e:
                        print(f"Eğitim hatası (dosya: {file_path}): {str(e)}")  # Debug için
                        continue

                avg_epoch_loss = epoch_loss / len(training_data)
                total_loss = avg_epoch_loss
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

            # Modeli kaydet
            model_path = os.path.join(app.root_path, 'model_weights.pth')
            torch.save(model.state_dict(), model_path)
            
            return jsonify({
                'message': 'Model başarıyla yeniden eğitildi',
                'epochs': epochs,
                'final_loss': total_loss,
                'training_samples': len(training_data)
            })

        except Exception as e:
            print(f"Model eğitim hatası: {str(e)}")  # Debug için
            return jsonify({'error': f'Model eğitimi sırasında hata: {str(e)}'}), 500

    except Exception as e:
        print(f"Genel hata: {str(e)}")  # Debug için
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_video', methods=['POST'])
def analyze_video_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'Video dosyası yüklenmedi'}), 400
    
    file = request.files['file']
    fps = int(request.form.get('fps', 1))  # Kullanıcının seçtiği FPS
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results, frame_path = analyze_video(filepath, fps)
        
        if frame_path:
            frame_url = url_for('serve_frame', filename=os.path.basename(frame_path))
        else:
            frame_url = None

        os.remove(filepath)  # Geçici dosyayı sil

        return jsonify({
            'results': results,
            'frame_path': frame_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_model', methods=['POST'])
def reset_model():
    try:
        # Mevcut model dosyasını sil
        model_path = os.path.join(app.root_path, 'model_weights.pth')
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Feedback klasörünü temizle
        feedback_folder = app.config['FEEDBACK_FOLDER']
        for label_folder in os.listdir(feedback_folder):
            label_path = os.path.join(feedback_folder, label_folder)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    file_path = os.path.join(label_path, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Dosya silinirken hata: {str(e)}")
                os.rmdir(label_path)
        
        # Global model değişkenini güncelle
        global model, preprocess
        
        # CLIP'i yeniden yükle
        try:
            import clip
        except ImportError:
            os.system('pip install git+https://github.com/openai/CLIP.git')
            import clip
        
        # Yeni modeli yükle
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Etiket klasörlerini yeniden oluştur
        labels = ["safe", "nsfw", "violence", "weapon", "abuse"]
        for label in labels:
            label_path = os.path.join(feedback_folder, label)
            os.makedirs(label_path, exist_ok=True)
        
        return jsonify({
            'message': 'Model ve geri bildirim verileri başarıyla sıfırlandı',
            'details': 'Varsayılan CLIP modeli yüklendi ve etiket klasörleri yeniden oluşturuldu'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya yüklenmedi'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400

        # Benzersiz bir dosya adı oluştur
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Dosyayı chunk'lar halinde kaydet
            chunk_size = 8192  # 8KB chunks
            with open(filepath, 'wb') as f:
                while True:
                    chunk = file.stream.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Kısa bir bekleme ekle
            time.sleep(0.1)
            
            return jsonify({
                'message': 'Dosya başarıyla yüklendi',
                'path': filepath,
                'filename': filename
            })
            
        except Exception as e:
            print(f"[DEBUG] Dosya yazma hatası: {str(e)}")
            # Hata durumunda dosyayı temizlemeyi dene
            try:
                if os.path.exists(filepath):
                    time.sleep(0.1)  # Kısa bekleme
                    os.remove(filepath)
            except Exception as cleanup_error:
                print(f"[DEBUG] Dosya temizleme hatası: {str(cleanup_error)}")
            raise e
            
    except Exception as e:
        print(f"[DEBUG] Upload hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    try:
        data = request.json
        filename = data.get('filename')
        file_path = data.get('path')
        file_type = data.get('type')
        fps = data.get('fps', 1)

        print(f"[DEBUG] Analiz başlıyor: {filename}")
        print(f"[DEBUG] Dosya yolu: {file_path}")

        if not os.path.exists(file_path):
            print(f"[DEBUG] Dosya bulunamadı: {file_path}")
            return jsonify({'error': f'Dosya bulunamadı: {filename}'}), 404

        # Dosyayı temp_frames klasörüne kopyala
        temp_filename = f'temp_{int(time.time()*1000)}_{secure_filename(filename)}'
        temp_path = os.path.join(app.config['TEMP_FRAMES_FOLDER'], temp_filename)
        
        try:
            # Önce dosyayı güvenli bir şekilde kopyala
            shutil.copy2(file_path, temp_path)
            print(f"[DEBUG] Dosya kopyalandı: {temp_path}")

            # Analizi temp dosya üzerinde yap
            if file_type == 'video':
                results, frame_path = analyze_video(temp_path, fps)
                result = {
                    'filename': filename,
                    'results': results,
                    'frame_path': url_for('serve_frame', filename=os.path.basename(frame_path)) if frame_path else None
                }
            else:
                results = analyze_image(temp_path)
                result = {
                    'filename': filename,
                    'results': results['results'],
                    'image_path': results['image_path']
                }

            return jsonify(result)

        except Exception as e:
            print(f"[DEBUG] Analiz hatası: {str(e)}")
            raise e

        finally:
            # Temizlik işlemleri
            try:
                # Orijinal dosyayı sil
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[DEBUG] Orijinal dosya silindi: {file_path}")
                
                # Geçici dosyayı sil (video analizi için frame hariç)
                if os.path.exists(temp_path) and file_type != 'video':
                    os.remove(temp_path)
                    print(f"[DEBUG] Geçici dosya silindi: {temp_path}")
            except Exception as e:
                print(f"[DEBUG] Dosya temizleme hatası: {str(e)}")

    except Exception as e:
        error_msg = f"{filename} analiz edilirken hata oluştu: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return jsonify({'error': error_msg}), 500

def cleanup_folders():
    """Upload ve temp_frames klasörlerini temizler"""
    try:
        # Upload klasörünü temizle
        upload_dir = app.config['UPLOAD_FOLDER']
        temp_dir = app.config['TEMP_FRAMES_FOLDER']
        
        print(f"[DEBUG] Upload ve temp klasörleri temizleniyor...")
        
        for directory in [upload_dir, temp_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            # Dosya kilitli olabilir, birkaç kez deneme yap
                            for _ in range(3):
                                try:
                                    os.remove(file_path)
                                    print(f"[DEBUG] Silindi: {file_path}")
                                    break
                                except Exception:
                                    time.sleep(0.5)  # Yarım saniye bekle ve tekrar dene
                    except Exception as e:
                        print(f"[DEBUG] Dosya silinirken hata ({file_path}): {str(e)}")
                        continue
        
        return True
    except Exception as e:
        print(f"[DEBUG] Klasör temizleme hatası: {str(e)}")
        return False

@app.route('/cleanup_uploads', methods=['POST'])
def cleanup_uploads():
    try:
        print("[DEBUG] Cleanup isteği alındı (browser kapanıyor/yenileniyor)")
        success = cleanup_folders()  # Yeni fonksiyonu kullan
        if success:
            print("[DEBUG] Klasörler temizlendi")
            return jsonify({'message': 'Klasörler temizlendi'})
        else:
            print("[DEBUG] Klasörler temizlenemedi")
            return jsonify({'error': 'Klasörler temizlenemedi'}), 500
    except Exception as e:
        print(f"[DEBUG] Temizleme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Periyodik temizlik için fonksiyon
def schedule_cleanup():
    def run_cleanup():
        while True:
            schedule.run_pending()
            time.sleep(1)

    # Her saat başı temizlik yap
    schedule.every().hour.do(cleanup_folders)
    
    # Arka planda çalışacak thread'i başlat
    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.daemon = True  # Ana uygulama kapandığında thread de kapansın
    cleanup_thread.start()

# Uygulama başlatılırken
def init_app(app):
    with app.app_context():
        cleanup_folders()  # İlk temizlik
        schedule_cleanup()  # Periyodik temizliği başlat

if __name__ == '__main__':
    init_app(app)  # Uygulama başlatılmadan önce init_app'i çağır
    app.run(host='0.0.0.0', port=5000, debug=True) 