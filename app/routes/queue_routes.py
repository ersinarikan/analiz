"""Queue management routes for analysis processing"""

from flask import Blueprint ,jsonify 
import logging 
import os 
import shutil 
from app .services .queue_service import get_queue_status ,get_queue_stats ,clear_queue 

logger =logging .getLogger (__name__ )

queue_bp =Blueprint ('queue',__name__ ,url_prefix ='/api/queue')
"""
Analiz kuyruğu için blueprint.
- Analiz işlemlerinin kuyruk yönetimi endpointlerini içerir.
"""

@queue_bp .route ('/status',methods =['GET'])
def get_queue_status_route ():
    """Get current queue status"""
    try :
        status =get_queue_status ()
        return jsonify ({
        'status':'success',
        'data':status 
        }),200 

    except Exception as e :
        logger .error (f"Error getting queue status: {e }")
        return jsonify ({
        'status':'error',
        'message':str (e )
        }),500 

@queue_bp .route ('/stats',methods =['GET'])
def get_queue_stats_route ():
    """Get queue statistics"""
    try :
        stats =get_queue_stats ()
        return jsonify ({
        'status':'success',
        'data':stats 
        }),200 

    except Exception as e :
        logger .error (f"Error getting queue stats: {e }")
        return jsonify ({
        'status':'error',
        'message':str (e )
        }),500 

@queue_bp .route ('/stop',methods =['POST'])
def stop_queue_route ():
    """Stop all analyses and clear queue and uploads"""
    try :
    # ERSIN YÖNTEM 1: Normal kuyruk temizleme (hafif durdurma)
        cleared_count =clear_queue ()
        logger .info (f"Kuyruktan {cleared_count } analiz temizlendi")

        # ERSIN Upload klasörünü temizle
        upload_path =os .path .join ('storage','uploads')
        if os .path .exists (upload_path ):
            for filename in os .listdir (upload_path ):
                file_path =os .path .join (upload_path ,filename )
                try :
                    if os .path .isfile (file_path )or os .path .islink (file_path ):
                        os .unlink (file_path )
                    elif os .path .isdir (file_path ):
                        shutil .rmtree (file_path )
                except Exception as e :
                    logger .error (f"Upload dosyası silinemedi {file_path }: {e }")
            logger .info (f"Upload klasörü temizlendi: {upload_path }")

            # ERSIN Processed klasörünü temizle (isteğe bağlı)
        processed_path =os .path .join ('storage','processed')
        if os .path .exists (processed_path ):
            for filename in os .listdir (processed_path ):
                file_path =os .path .join (processed_path ,filename )
                try :
                    if os .path .isfile (file_path )or os .path .islink (file_path ):
                        os .unlink (file_path )
                    elif os .path .isdir (file_path ):
                        shutil .rmtree (file_path )
                except Exception as e :
                    logger .error (f"Processed dosyası silinemedi {file_path }: {e }")
            logger .info (f"Processed klasörü temizlendi: {processed_path }")

        return jsonify ({
        'status':'success',
        'message':f'Kuyruk temizlendi ({cleared_count } analiz), upload ve processed klasörleri temizlendi'
        }),200 

    except Exception as e :
        logger .error (f"Kuyruk durdurma hatası: {e }")
        return jsonify ({
        'status':'error',
        'message':str (e )
        }),500 

@queue_bp .route ('/force-stop',methods =['POST'])
def force_stop_and_restart ():
    """
    CTRL+C benzeri zorla durdurma + VT temizlik + restart
    Aktif analizleri zorla durdurur, veritabanından siler ve uygulamayı restart eder
    """
    try :
        import os 
        import sys 
        from app .models .analysis import Analysis 
        from app .models .file import File 
        from app import db 

        logger .info ("🚨 FORCE STOP başlatıldı - Tüm analizler zorla durduruluyor...")

        # ERSIN Initialize analysis_ids at the start
        analysis_ids :list [str ]=[]

        # ERSIN 1. CTRL+C benzeri - Thread interrupt ve kaynak temizleme
        logger .info ("1️⃣ Thread'ler ve kaynaklar zorla temizleniyor...")

        # ERSIN Global stop flag set et (eğer varsa)
        try :
            from app .services .queue_service import clear_queue 
            clear_queue ()
        except Exception as e :
            logger .warning (f"Queue clear hatası: {e }")

            # ERSIN 2. VERİTABANI TEMİZLİK - Aktif analizleri sil
        logger .info ("2️⃣ Veritabanından aktif analizler siliniyor...")
        try :
        # ERSIN Processing veya pending durumundaki analizleri bul
            from app .utils .sqlalchemy_helpers import column_in 
            active_analyses =Analysis .query .filter (
            column_in (Analysis .status ,['processing','pending'])
            ).all ()
            for analysis in active_analyses :
                analysis_ids .append (analysis .id )
                logger .info (f"Aktif analiz siliniyor: #{analysis .id } (status: {analysis .status })")

                # ERSIN İlgili dosyaları da sil (isteğe bağlı)
                if analysis .file_id :
                    file_record =File .query .get (analysis .file_id )
                    if file_record :
                        logger .info (f"İlgili dosya kaydı siliniyor: {file_record .original_filename }")
                        db .session .delete (file_record )

                db .session .delete (analysis )

            db .session .commit ()
            logger .info (f"✅ {len (analysis_ids )} aktif analiz veritabanından silindi")

        except Exception as db_err :
            logger .error (f"Veritabanı temizlik hatası: {db_err }")
            db .session .rollback ()

            # ERSIN 3. DOSYA SİSTEMİ TEMİZLİK
        logger .info ("3️⃣ Dosya sistemi temizleniyor...")
        try :
        # ERSIN Upload klasörünü temizle
            upload_path =os .path .join ('storage','uploads')
            if os .path .exists (upload_path ):
                for filename in os .listdir (upload_path ):
                    file_path =os .path .join (upload_path ,filename )
                    try :
                        if os .path .isfile (file_path ):
                            os .unlink (file_path )
                        elif os .path .isdir (file_path ):
                            import shutil 
                            shutil .rmtree (file_path )
                    except Exception as e :
                        logger .warning (f"Dosya silinemedi {file_path }: {e }")

                        # ERSIN Processed klasörünü temizle
            processed_path =os .path .join ('storage','processed')
            if os .path .exists (processed_path ):
                for filename in os .listdir (processed_path ):
                    file_path =os .path .join (processed_path ,filename )
                    try :
                        if os .path .isfile (file_path ):
                            os .unlink (file_path )
                        elif os .path .isdir (file_path ):
                            import shutil 
                            shutil .rmtree (file_path )
                    except Exception as e :
                        logger .warning (f"Processed dosya silinemedi {file_path }: {e }")

            logger .info ("✅ Dosya sistemi temizlendi")

        except Exception as fs_err :
            logger .error (f"Dosya sistemi temizlik hatası: {fs_err }")

            # ERSIN 4. RESPONSE GÖNDER ve RESTART BAŞLAT
        logger .info ("4️⃣ Başarı mesajı gönderiliyor ve restart başlatılacak...")

        # ERSIN Önce response'u gönder
        response_data ={
        'message':'Tüm analizler zorla durduruldu, sistem restart ediliyor...',
        'force_stopped':True ,
        'cleared_analyses':len (analysis_ids ),
        'restart_initiated':True 
        }

        # ERSIN Response gönderildikten sonra restart için thread başlat
        import threading 
        def delayed_restart ():
            import time 

            # ERSIN 1. Response gitmesi için kısa bekleme
            time .sleep (3 )# ERSIN 3 saniye response bekleme
            logger .info ("🔄 RESTART hazırlığı - aktif thread'ler kontrol ediliyor...")

            # ERSIN 2. Aktif thread'leri zorla durdur (ThreadPoolExecutor)
            try :
                from app .services .analysis_service import _age_estimation_executor 
                if _age_estimation_executor :
                    logger .info ("🛑 ThreadPoolExecutor kapatılıyor...")
                    _age_estimation_executor .shutdown (wait =False )# ERSIN Zorla kapat
                    logger .info ("✅ ThreadPoolExecutor kapatıldı")
            except Exception as e :
                logger .warning (f"ThreadPoolExecutor kapatma hatası: {e }")

                # ERSIN 3. Kısa ek bekleme (thread cleanup için)
            time .sleep (2 )# ERSIN 2 saniye thread cleanup

            logger .info ("🔄 RESTART başlatılıyor...")

            try :
            # ERSIN Systemd servisi olarak çalışıyorsak systemctl kullan
                if os .path .exists ('/etc/systemd/system/wsanaliz.service'):
                    import subprocess 
                    logger .info ("Systemd servisi bulundu, systemctl restart yapılıyor...")
                    # ERSIN Sudo şifresini environment'tan al (güvenlik için)
                    sudo_password =os .environ .get ('SUDO_PASSWORD','5ex5chan5ge4')
                    restart_cmd =f'echo "{sudo_password }" | sudo -S systemctl restart wsanaliz.service'
                    subprocess .Popen (restart_cmd ,shell =True ,stdout =subprocess .DEVNULL ,stderr =subprocess .DEVNULL )
                    logger .info ("✅ Systemctl restart komutu gönderildi")
                    # ERSIN Process'i sonlandır, systemd yeniden başlatacak
                    os ._exit (0 )
                    # ERSIN Windows için restart
                elif sys .platform =="win32":
                    import subprocess 
                    subprocess .Popen ([sys .executable ]+sys .argv )
                    os ._exit (0 )
                else :
                # ERSIN Linux/Mac için restart (systemd yoksa)
                    os .execv (sys .executable ,[sys .executable ]+sys .argv )
            except Exception as restart_err :
                logger .error (f"Restart hatası: {restart_err }")
                # ERSIN Restart başarısız olursa en azından process'i kill et
                os ._exit (1 )

        restart_thread =threading .Thread (target =delayed_restart )
        restart_thread .daemon =True 
        restart_thread .start ()

        return jsonify (response_data ),200 

    except Exception as e :
        logger .error (f"Force stop hatası: {str (e )}",exc_info =True )
        return jsonify ({
        'status':'error',
        'message':f'Force stop hatası: {str (e )}'
        }),500 