import os 
import cv2 
import tempfile 
from datetime import timedelta 
from flask import current_app 
from moviepy .editor import VideoFileClip 
from app .utils .image_utils import save_image ,resize_image 

def extract_frames (video_path ,output_dir ,frames_per_second =1 ):
    """Bir videodan belirli saniyede kare sayısı ile kareler çıkarır."""
    try :
        if not os .path .exists (video_path ):
            current_app .logger .error (f"Video dosyası bulunamadı: {video_path }")
            return [],0 ,0 

            # ERSIN Çıktı dizinini oluştur
        os .makedirs (output_dir ,exist_ok =True )

        # ERSIN VideoCapture nesnesini oluştur
        cap =cv2 .VideoCapture (video_path )

        if not cap .isOpened ():
            current_app .logger .error (f"Video açılamadı: {video_path }")
            return [],0 ,0 

            # ERSIN Video özelliklerini al
        fps =cap .get (cv2 .CAP_PROP_FPS )
        frame_count =int (cap .get (cv2 .CAP_PROP_FRAME_COUNT ))
        duration =frame_count /fps 

        # ERSIN Her kaç karede bir kare alınacak
        frame_step =int (fps /frames_per_second )
        frame_step =max (1 ,frame_step )# ERSIN En az 1 olmalı

        # ERSIN Kare çıkarma
        frame_paths =[]
        frame_index =0 

        while True :
            ret ,frame =cap .read ()

            if not ret :
                break 

                # ERSIN frame_step'e göre kare al
            if frame_index %frame_step ==0 :
            # ERSIN Anlık zaman damgasını hesapla
                timestamp =frame_index /fps 

                # ERSIN Kare dosya adını oluştur
                frame_filename =f"frame_{frame_index :06d}_{timestamp :.2f}.jpg"
                frame_path =os .path .join (output_dir ,frame_filename )

                # ERSIN Kareyi kaydet
                save_image (frame ,frame_path )

                # ERSIN Kaydedilen kare yolunu ve zaman damgasını listeye ekle
                frame_paths .append ((frame_path ,timestamp ))

            frame_index +=1 

            # ERSIN VideoCapture'ı serbest bırak
        cap .release ()

        return frame_paths ,frame_count ,duration 

    except Exception as e :
        current_app .logger .error (f"Video karesi çıkarma hatası: {str (e )}")
        return [],0 ,0 

def generate_video_thumbnail (video_path ,thumbnail_path ,frame_position =0.1 ):
    """Video için bir küçük resim (thumbnail) oluşturur."""
    try :
        if not os .path .exists (video_path ):
            current_app .logger .error (f"Video dosyası bulunamadı: {video_path }")
            return False 

            # ERSIN VideoCapture nesnesini oluştur
        cap =cv2 .VideoCapture (video_path )

        if not cap .isOpened ():
            current_app .logger .error (f"Video açılamadı: {video_path }")
            return False 

            # ERSIN Video özelliklerini al
        frame_count =int (cap .get (cv2 .CAP_PROP_FRAME_COUNT ))

        # ERSIN frame_position, videonun yüzde kaçında bir kare alınacağını belirtir
        # ERSIN Örneğin, 0.1 videonun %10'unda bir kare anlamına gelir
        position =int (frame_count *frame_position )
        position =max (0 ,min (position ,frame_count -1 ))# ERSIN Sınırlarda tut

        # ERSIN Belirtilen konuma git
        cap .set (cv2 .CAP_PROP_POS_FRAMES ,position )

        # ERSIN Kareyi oku
        ret ,frame =cap .read ()

        if not ret :
            current_app .logger .error (f"Video karesi okunamadı: {video_path }")
            cap .release ()
            return False 

            # ERSIN Kareyi yeniden boyutlandır
        thumbnail =resize_image (frame ,width =320 )

        # ERSIN Küçük resmi kaydet
        result =save_image (thumbnail ,thumbnail_path )

        # ERSIN VideoCapture'ı serbest bırak
        cap .release ()

        return result 

    except Exception as e :
        current_app .logger .error (f"Video küçük resmi oluşturma hatası: {str (e )}")
        return False 

def get_video_info (video_path ):
    """Video hakkında bilgi döndürür."""
    try :
        if not os .path .exists (video_path ):
            current_app .logger .error (f"Video dosyası bulunamadı: {video_path }")
            return None 

            # ERSIN VideoCapture nesnesini oluştur
        cap =cv2 .VideoCapture (video_path )

        if not cap .isOpened ():
            current_app .logger .error (f"Video açılamadı: {video_path }")
            return None 

            # ERSIN Video özelliklerini al
        width =int (cap .get (cv2 .CAP_PROP_FRAME_WIDTH ))
        height =int (cap .get (cv2 .CAP_PROP_FRAME_HEIGHT ))
        fps =cap .get (cv2 .CAP_PROP_FPS )
        frame_count =int (cap .get (cv2 .CAP_PROP_FRAME_COUNT ))
        duration =frame_count /fps if fps >0 else 0 

        # ERSIN VideoCapture'ı serbest bırak
        cap .release ()

        # ERSIN Süreyi formatlı bir şekilde döndür
        duration_formatted =str (timedelta (seconds =int (duration )))

        return {
        'width':width ,
        'height':height ,
        'fps':fps ,
        'frame_count':frame_count ,
        'duration':duration ,
        'duration_formatted':duration_formatted ,
        'bitrate':None # ERSIN OpenCV ile bitrate alınamıyor
        }

    except Exception as e :
        current_app .logger .error (f"Video bilgisi alma hatası: {str (e )}")
        return None 

def extract_audio (video_path ,output_path =None ):
    """Videodan ses dosyasını çıkarır."""
    try :
        if not os .path .exists (video_path ):
            current_app .logger .error (f"Video dosyası bulunamadı: {video_path }")
            return None 

            # ERSIN MoviePy ile videoyu aç
        video =VideoFileClip (video_path )

        # ERSIN Ses yoksa None döndür
        if not video .audio :
            current_app .logger .warning (f"Videoda ses yok: {video_path }")
            video .close ()
            return None 

            # ERSIN Çıktı yolu belirtilmemişse geçici bir dosya oluştur
        if output_path is None :
            temp_file =tempfile .NamedTemporaryFile (delete =False ,suffix ='.mp3')
            output_path =temp_file .name 
            temp_file .close ()

            # ERSIN Sesi çıkar ve kaydet
        video .audio .write_audiofile (output_path ,logger =None )

        # ERSIN VideoFileClip'i kapat
        video .close ()

        return output_path if os .path .exists (output_path )else None 

    except Exception as e :
        current_app .logger .error (f"Video ses çıkarma hatası: {str (e )}")
        return None 

def create_video_from_frames (frame_paths ,output_path ,fps =30 ,audio_path =None ):
    """Karelerden video oluşturur."""
    try :
        if not frame_paths :
            current_app .logger .error ("Video oluşturmak için kare yok")
            return False 

            # ERSIN İlk kareyi yükle ve boyutlarını al
        first_frame =cv2 .imread (frame_paths [0 ])
        height ,width ,_ =first_frame .shape 

        # ERSIN VideoWriter nesnesini oluştur
        # ERSIN cv2.VideoWriter_fourcc exists but type checker doesn't recognize it
        fourcc =getattr (cv2 ,'VideoWriter_fourcc',lambda *args :0x7634706d )(*'mp4v')# ERSIN Fallback to mp4v codec
        out =cv2 .VideoWriter (output_path ,fourcc ,fps ,(width ,height ))

        # ERSIN Kareleri yaz
        for frame_path in frame_paths :
            frame =cv2 .imread (frame_path )
            out .write (frame )

            # ERSIN VideoWriter'ı serbest bırak
        out .release ()

        # ERSIN Eğer ses dosyası belirtilmişse, sesi ekle
        if audio_path and os .path .exists (audio_path ):
            video =VideoFileClip (output_path )
            audio =VideoFileClip (audio_path ).audio 

            # ERSIN Ses dosyasını ekleyerek yeni video oluştur
            video_with_audio =video .set_audio (audio )

            # ERSIN Geçici dosya oluştur
            temp_path =output_path +".temp.mp4"

            # ERSIN Video ve sesi kaydet
            video_with_audio .write_videofile (temp_path ,codec ='libx264',audio_codec ='aac',logger =None )

            # ERSIN Nesneleri kapat
            video .close ()

            # ERSIN Geçici dosyayı orijinal dosya ile değiştir
            os .replace (temp_path ,output_path )

        return os .path .exists (output_path )

    except Exception as e :
        current_app .logger .error (f"Video oluşturma hatası: {str (e )}")
        return False 