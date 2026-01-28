import os 
import cv2 
import numpy as np 
from PIL import Image ,ImageDraw ,ImageFont 
from flask import current_app 
from typing import Any 

def load_image (image_path :str )->np .ndarray [Any ,Any ]|None :
    """
    Belirtilen dosya yolundan resmi yükler.
    Args:
        image_path (str): Resim dosya yolu.
    Returns:
        np.ndarray: Yüklenen resim.
    """
    try :
        if not os .path .exists (image_path ):
            current_app .logger .error (f"Görüntü dosyası bulunamadı: {image_path }")
            return None 

            # ERSIN OpenCV ile görüntüyü yükle (BGR formatında)
        image =cv2 .imread (image_path )

        if image is None :
            current_app .logger .error (f"Görüntü yüklenemedi: {image_path }")
            return None 

        return image 

    except Exception as e :
        current_app .logger .error (f"Görüntü yükleme hatası: {str (e )}")
        return None 

def resize_image (image :np .ndarray [Any ,Any ]|None ,width :int |None =None ,height :int |None =None ,max_size :int |None =None )->np .ndarray [Any ,Any ]|None :
    """Bir görüntüyü belirtilen boyutlara yeniden boyutlandırır."""
    try :
        if image is None :
            return None 

            # ERSIN Görüntü boyutlarını al
        h ,w =image .shape [:2 ]

        # ERSIN En büyük boyut belirtilmişse ve görüntü daha büyükse
        if max_size and max (h ,w )>max_size :
            if h >w :
                height =max_size 
                width =int (w *(height /h ))
            else :
                width =max_size 
                height =int (h *(width /w ))

                # ERSIN Boyutları belirle
        if width is None and height is None :
            return image 

            # ERSIN width and height are guaranteed to be int at this point due to logic above
        if width is None :
        # ERSIN Sadece yükseklik belirtilmişse, oranı koru
            if height is not None :
                aspect_ratio =float (w )/float (h )
                width =int (height *aspect_ratio )
            else :
                return image 
        elif height is None :
        # ERSIN Sadece genişlik belirtilmişse, oranı koru
            if width is not None :
                aspect_ratio =float (h )/float (w )
                height =int (width *aspect_ratio )
            else :
                return image 

                # ERSIN width and height are now guaranteed to be int
        if width is not None and height is not None :
        # ERSIN Görüntüyü yeniden boyutlandır
            resized =cv2 .resize (image ,(int (width ),int (height )),interpolation =cv2 .INTER_AREA )
        else :
            return image 

        return resized 

    except Exception as e :
        current_app .logger .error (f"Görüntü yeniden boyutlandırma hatası: {str (e )}")
        return image 

def generate_image_thumbnail (image_path ,thumbnail_path ,size =(200 ,200 )):
    """Bir görüntü için küçük resim oluşturur."""
    try :
    # ERSIN Görüntüyü yükle
        image =load_image (image_path )

        if image is None :
            return False 

            # ERSIN Görüntüyü yeniden boyutlandır
        thumbnail =resize_image (image ,width =size [0 ],height =size [1 ])

        # ERSIN Küçük resmi kaydet
        if thumbnail is not None :
            cv2 .imwrite (thumbnail_path ,thumbnail )

        return os .path .exists (thumbnail_path )

    except Exception as e :
        current_app .logger .error (f"Küçük resim oluşturma hatası: {str (e )}")
        return False 

def save_image (image ,output_path ):
    """Bir NumPy görüntü dizisini belirtilen yola kaydeder."""
    try :
        if image is None :
            return False 

            # ERSIN Dizin yoksa oluştur
        os .makedirs (os .path .dirname (output_path ),exist_ok =True )

        # ERSIN Görüntüyü kaydet
        cv2 .imwrite (output_path ,image )

        return os .path .exists (output_path )

    except Exception as e :
        current_app .logger .error (f"Görüntü kaydetme hatası: {str (e )}")
        return False 

def overlay_text (image ,text ,position ,font =cv2 .FONT_HERSHEY_SIMPLEX ,font_scale =1.0 ,color =(255 ,255 ,255 ),thickness =2 ):
    """Bir görüntü üzerine metin çizer."""
    try :
        if image is None :
            return None 

            # ERSIN Metni çiz
        cv2 .putText (image ,text ,position ,font ,font_scale ,color ,thickness )

        return image 

    except Exception as e :
        current_app .logger .error (f"Metin ekleme hatası: {str (e )}")
        return image 

def overlay_text_turkish (image ,text ,position ,color =(0 ,255 ,0 ),font_size =20 ,bg_color =(0 ,0 ,0 ),bg_padding =5 ):
    """
    OpenCV görüntüsüne Türkçe karakter destekli metin çizer.
    
    Args:
        image: OpenCV görüntüsü (numpy array)
        text: Çizilecek metin (Türkçe karakter destekli)
        position: (x, y) pozisyonu
        color: Text rengi (BGR format)
        font_size: Font boyutu
        bg_color: Arka plan rengi (BGR format)
        bg_padding: Arka plan padding
    
    Returns:
        OpenCV görüntüsü (numpy array)
    """
    try :
        if image is None :
            return None 

            # ERSIN OpenCV'den PIL'e dönüştür
        pil_image =Image .fromarray (cv2 .cvtColor (image ,cv2 .COLOR_BGR2RGB ))
        draw =ImageDraw .Draw (pil_image )

        # ERSIN Varsayılan font kullan (sistem fontunu bulmaya çalış)
        try :
            font =ImageFont .truetype ("arial.ttf",font_size )
        except :
            try :
                font =ImageFont .truetype ("C:/Windows/Fonts/arial.ttf",font_size )
            except :
                font =ImageFont .load_default ()

                # ERSIN Text boyutlarını hesapla
        bbox =draw .textbbox ((0 ,0 ),text ,font =font )
        text_width =bbox [2 ]-bbox [0 ]
        text_height =bbox [3 ]-bbox [1 ]

        x ,y =position 

        # ERSIN Arka plan çiz
        if bg_color is not None :
            bg_coords =[
            x -bg_padding ,
            y -text_height -bg_padding ,
            x +text_width +bg_padding ,
            y +bg_padding 
            ]
            draw .rectangle (bg_coords ,fill =bg_color )

            # ERSIN Metni çiz (PIL RGB formatında)
        pil_color =(color [2 ],color [1 ],color [0 ])# ERSIN BGR -> RGB
        draw .text ((x ,y -text_height ),text ,font =font ,fill =pil_color )

        # ERSIN PIL'den OpenCV'ye geri dönüştür
        result =cv2 .cvtColor (np .array (pil_image ),cv2 .COLOR_RGB2BGR )

        return result 

    except Exception as e :
        current_app .logger .error (f"Türkçe metin ekleme hatası: {str (e )}")
        return image 

def draw_rectangle (image ,top_left ,bottom_right ,color =(0 ,255 ,0 ),thickness =2 ):
    """Bir görüntü üzerine dikdörtgen çizer."""
    try :
        if image is None :
            return None 

            # ERSIN Dikdörtgeni çiz
        cv2 .rectangle (image ,top_left ,bottom_right ,color ,thickness )

        return image 

    except Exception as e :
        current_app .logger .error (f"Dikdörtgen çizme hatası: {str (e )}")
        return image 

def crop_image (image ,x ,y ,width ,height ):
    """Bir görüntüyü belirtilen koordinatlara göre keser."""
    try :
        if image is None :
            return None 

            # ERSIN Görüntü boyutlarını al
        h ,w =image .shape [:2 ]

        # ERSIN Koordinatları sınırlar içinde tut
        x =max (0 ,min (x ,w -1 ))
        y =max (0 ,min (y ,h -1 ))
        width =max (1 ,min (width ,w -x ))
        height =max (1 ,min (height ,h -y ))

        # ERSIN Görüntüyü kes
        cropped =image [y :y +height ,x :x +width ]

        return cropped 

    except Exception as e :
        current_app .logger .error (f"Görüntü kesme hatası: {str (e )}")
        return None 