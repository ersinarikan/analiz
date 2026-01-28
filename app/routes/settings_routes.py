import re 
import os 
import signal 
from flask import Blueprint ,request ,jsonify ,current_app 
from config import Config # ERSIN config.py'daki ana Config sınıfını import ediyoruz
import datetime 

bp =Blueprint ('settings_bp',__name__ ,url_prefix ='/api')
"""
Ayar yönetimi için blueprint.
- Sistem ve model ayarlarını yönetmeye yönelik endpointleri içerir.
"""

# ERSIN config.py dosyasının yolu
# ERSIN Proje kök dizinini bulmak için __file__ (settings_routes.py) üzerinden ../../ yaparak git
# ERSIN WSANALIZ/app/routes/settings_routes.py -> WSANALIZ/config.py
CONFIG_PY_PATH =os .path .abspath (os .path .join (os .path .dirname (__file__ ),'..','..','config.py'))

# ERSIN Güncellenebilecek parametrelerin listesi ve türleri (config.py'daki isimleriyle)
UPDATABLE_PARAMS =Config .UPDATABLE_PARAMS 

# ERSIN Kullanıcının referans resmindeki sabit fabrika ayarları
FACTORY_DEFAULTS =Config .FACTORY_DEFAULTS 

def update_settings_state_file (params ):
    """
    Settings state dosyasını günceller (Flask auto-reload için)
    """
    try :
        state_file_path =os .path .join (os .path .dirname (__file__ ),'..','utils','settings_state.py')

        # ERSIN Dosyayı oku
        with open (state_file_path ,'r',encoding ='utf-8')as f :
            content =f .read ()

            # ERSIN LAST_UPDATE satırını güncelle
        timestamp =datetime .datetime .now ().isoformat ()
        new_content =re .sub (
        r'LAST_UPDATE = .*',
        f'LAST_UPDATE = "{timestamp }"',
        content 
        )

        # ERSIN Settings state'i güncelle
        for key ,value in params .items ():
            key_lower =key .lower ()
            new_content =re .sub (
            f"'{key_lower }': [^,\n]*",
            f"'{key_lower }': {value }",
            new_content 
            )

            # ERSIN Dosyayı yaz
        with open (state_file_path ,'w',encoding ='utf-8')as f :
            f .write (new_content )

        current_app .logger .info (f"Settings state dosyası güncellendi: {params }")
        return True 

    except Exception as e :
        current_app .logger .error (f"Settings state dosyası güncellenirken hata: {str (e )}")
        return False 

def update_config_file (params_to_update ):
    try :
        if not os .path .exists (CONFIG_PY_PATH ):
            current_app .logger .error (f"Config file not found at {CONFIG_PY_PATH }")
            return False ,f"Config file not found at {CONFIG_PY_PATH }"

        with open (CONFIG_PY_PATH ,'r',encoding ='utf-8')as f :
            lines =f .readlines ()

        new_lines =[]

        for line in lines :
            new_line_to_append =line 
            stripped_line =line .strip ()
            if not stripped_line or stripped_line .startswith ('#')or '='not in stripped_line :
                new_lines .append (new_line_to_append )
                continue 

                # ERSIN Try to find a için açıklama
                # ERSIN Bu logic parametrelerin "KEY = VALUE" veya "KEY=VALUE" olarak tanımlandığını varsayar
                # ERSIN Ve key satırın başında olmalı (leading whitespace ignore edilir)
                # ERSIN Ve key içinde boşluk olmamalı

            possible_key =stripped_line .split ('=')[0 ].strip ()

            if possible_key in params_to_update :
                key =possible_key 
                value =params_to_update [key ]

                # ERSIN Preserve indentation
                leading_whitespace =line [:len (line )-len (line .lstrip ())]

                # ERSIN Format value (add quotes için strings)
                if isinstance (value ,str ):
                    formatted_value =f'"{value }"'
                else :
                    formatted_value =str (value )

                    # ERSIN Preserve comments if any
                comment_part =''
                if '#'in line :
                    original_value_and_comment =line .split ('=',1 )[1 ].strip ()
                    if '#'in original_value_and_comment :# ERSIN Ensure comment dir after value
                    # ERSIN Find first # şu dir değil inside quotes
                        _ = original_value_and_comment  # val_part_before_comment unused
                        comment_idx =-1 
                        in_quotes =False 
                        quote_char =''
                        for i ,char in enumerate (original_value_and_comment ):
                            if char in ('"',"'")and (i ==0 or original_value_and_comment [i -1 ]!='\\'):# ERSIN Escape edilmemiş
                                if not in_quotes :
                                    in_quotes =True 
                                    quote_char =char 
                                elif char ==quote_char :
                                    in_quotes =False 
                                    quote_char =''
                            elif char =='#'and not in_quotes :
                                comment_idx =i 
                                break 
                        if comment_idx !=-1 :
                            comment_part =' '+original_value_and_comment [comment_idx :].strip ()


                new_line_to_append =f"{leading_whitespace }{key } = {formatted_value }{comment_part }\n"
                # ERSIN Bu yazma için güncellendi olarak işaretle, daha sonra bulunamazsa tekrar ekleme
                # ERSIN Bu logic önce tüm satırları kontrol edip eksik olanları eklemek daha iyi olabilir

            new_lines .append (new_line_to_append )

        with open (CONFIG_PY_PATH ,'w',encoding ='utf-8')as f :
            f .writelines (new_lines )
        current_app .logger .info (f"Config file {CONFIG_PY_PATH } updated with: {params_to_update }")
        return True ,"Config file updated successfully."
    except Exception as e :
        current_app .logger .error (f"Error updating config file {CONFIG_PY_PATH }: {e }",exc_info =True )
        return False ,str (e )

@bp .route ('/get_analysis_params',methods =['GET'])
@bp .route ('/settings/analysis-params',methods =['GET'])# ERSIN Frontend uyumluluk alias'ı
def get_analysis_params ():
    # ERSIN Config class defaults'ları zorla okumalı mıyız kontrol et
    use_defaults =request .args .get ('use_defaults','false').lower ()=='true'
    return _get_analysis_params_logic (use_defaults )

@bp .route ('/settings/analysis-params/defaults',methods =['GET'])# ERSIN Defaults endpoint alias'ı  
def get_analysis_params_defaults ():
    """Always return factory defaults"""
    return _get_analysis_params_logic (use_defaults =True )

def _get_analysis_params_logic (use_defaults =False ):
    params ={}

    for key_config ,param_type in UPDATABLE_PARAMS .items ():
        value =None 
        if use_defaults :
        # ERSIN FACTORY_DEFAULTS'tan zorla oku
            if key_config in FACTORY_DEFAULTS :
                value =FACTORY_DEFAULTS [key_config ]
            else :
                current_app .logger .warning (f"Factory default value for {key_config } not found in FACTORY_DEFAULTS. Falling back to Config class.")
                if hasattr (Config ,key_config ):# ERSIN Güvenlik için fallback, tüm UPDATABLE_PARAMS FACTORY_DEFAULTS'ta olmalı
                    value =getattr (Config ,key_config )
        else :
        # ERSIN Standart davranış: Önce live app.config'den al, sonra Config class'a fallback yap
            value =current_app .config .get (key_config )
            if value is None and hasattr (Config ,key_config ):
                value =getattr (Config ,key_config )

        key_frontend =key_config .lower ()# ERSIN Frontend küçük harf key'ler kullanır

        if value is not None :
            try :
                if param_type ==bool and isinstance (value ,str ):
                    params [key_frontend ]=value .lower ()in ('true','1','t')
                else :
                    params [key_frontend ]=param_type (value )
            except ValueError :
                current_app .logger .warning (f"Could not convert config value for {key_config } ('{value }') to {param_type }. Sending as string.")
                params [key_frontend ]=str (value )
            except TypeError :
                current_app .logger .warning (f"Type error converting config value for {key_config } ('{value }') to {param_type }. Sending as string.")
                params [key_frontend ]=str (value )
        else :
            params [key_frontend ]=None 
            # ERSIN Sadece kasıtlı olarak defaults çekilmiyorsa veya factory default beklenmedik şekilde eksikse uyarı log'la
            if not use_defaults :
                current_app .logger .warning (f"Parameter {key_config } not found in app.config or Config defaults.")
            elif key_config not in FACTORY_DEFAULTS :# ERSIN use_defaults true ama başlangıçta FACTORY_DEFAULTS'ta yoksa
                 current_app .logger .warning (f"Parameter {key_config } was requested as a default but not found in FACTORY_DEFAULTS or Config class.")

    return jsonify (params )

@bp .route ('/set_analysis_params',methods =['POST'])
@bp .route ('/settings/analysis-params',methods =['POST'])# ERSIN Frontend uyumluluk alias'ı
def set_analysis_params ():
    data =request .json 
    if not data :
        return jsonify ({"error":"No data provided"}),400 

    params_to_update_in_file ={}

    malformed_params =[]

    for key_frontend ,value in data .items ():
        key_config =key_frontend .upper ()# ERSIN Config.py BÜYÜK HARF key'ler kullanır
        if key_config in UPDATABLE_PARAMS :
            param_type =UPDATABLE_PARAMS [key_config ]
            try :
            # ERSIN Tipi doğrula ve dönüştür
                if value is None and param_type not in [str ]:# ERSIN None'a sadece mantıklı olduğunda izin ver (örn. sayılar için değil)
                    typed_value =None 
                elif param_type ==bool and isinstance (value ,str ):# ERSIN Bool için "True"/"False" string'lerini handle et
                     typed_value =value .lower ()in ('true','1','t')
                elif value ==''and param_type !=str :# ERSIN String olmayan tip için boş string sorunlu
                    malformed_params .append (f"Parameter '{key_frontend }' (for {key_config }) received an empty string, expected {param_type .__name__ }.")
                    continue 
                else :
                    typed_value =param_type (value )

                params_to_update_in_file [key_config ]=typed_value 
                # ERSIN Live app.config'i hemen güncelle
                current_app .config [key_config ]=typed_value 

            except (ValueError ,TypeError )as e :
                malformed_params .append (f"Invalid type for parameter '{key_frontend }' (for {key_config }). Expected {param_type .__name__ }, got '{value }'. Error: {e }")
        else :
            current_app .logger .warning (f"Unknown parameter received and ignored: {key_frontend }")

    if malformed_params :
        return jsonify ({"error":"Invalid parameter types provided.","details":malformed_params }),400 

    if not params_to_update_in_file :
        return jsonify ({"message":"No valid parameters to update or no changes detected."}),200 

    current_app .logger .info (f"Attempting to update app.config with: {params_to_update_in_file }")
    current_app .logger .info (f"Attempting to update config.py file ({CONFIG_PY_PATH }) with: {params_to_update_in_file }")

    success ,message =update_config_file (params_to_update_in_file )

    if success :
        current_app .logger .info (f"Successfully updated app.config and config.py file. New app.config for relevant keys: {{k: current_app.config.get(k) for k in params_to_update_in_file}}")

        # ERSIN Settings state dosyasını güncelle, Flask auto-reload için
        update_settings_state_file (params_to_update_in_file )

        # ERSIN Environment'a göre response mesajı ve restart
        environment =os .environ .get ('FLASK_ENV','development')
        restart_required =environment !='development'

        if restart_required :
        # ERSIN Production ortamında otomatik restart yap
            current_app .logger .info ("Production ortamı tespit edildi, otomatik restart başlatılıyor...")

            def delayed_restart ():
                import time 
                import sys 
                import subprocess 

                # ERSIN Response gönderilmesi için bekle
                time .sleep (3 )

                current_app .logger .info ("🔄 Analiz parametreleri güncellendi, RESTART başlatılıyor...")

                try :
                # ERSIN Systemd servisi olarak çalışıyorsak systemctl kullan
                    if os .path .exists ('/etc/systemd/system/wsanaliz.service'):
                        current_app .logger .info ("Systemd servisi bulundu, systemctl restart yapılıyor...")
                        # ERSIN Sudo şifresini environment'tan al, güvenlik için
                        sudo_password =os .environ .get ('SUDO_PASSWORD','5ex5chan5ge4')
                        restart_cmd =f'echo "{sudo_password }" | sudo -S systemctl restart wsanaliz.service'
                        subprocess .Popen (restart_cmd ,shell =True ,stdout =subprocess .DEVNULL ,stderr =subprocess .DEVNULL )
                        current_app .logger .info ("✅ Systemctl restart komutu gönderildi")
                        # ERSIN Process'i sonlandır, systemd yeniden başlatacak
                        os ._exit (0 )
                        # ERSIN Windows için restart
                    elif sys .platform =="win32":
                        subprocess .Popen ([sys .executable ]+sys .argv )
                        os ._exit (0 )
                    else :
                    # ERSIN Linux/Mac için restart, systemd yoksa
                        current_app .logger .info ("Systemd servisi bulunamadı, process restart yapılıyor...")
                        os .kill (os .getpid (),signal .SIGTERM )
                except Exception as restart_err :
                    current_app .logger .error (f"Restart hatası: {restart_err }")
                    # ERSIN Restart başarısız olursa en azından process'i kill et
                    os ._exit (1 )

                    # ERSIN Background thread'de restart
            import threading 
            restart_thread =threading .Thread (target =delayed_restart )
            restart_thread .daemon =True 
            restart_thread .start ()

            message ="Analysis parameters saved successfully. System restarting..."
        else :
            message ="Analysis parameters saved successfully."

            # ERSIN Parametreler başarıyla kaydedildi
        response_data ={
        "message":message ,
        "restart_required":restart_required ,
        "restart_initiated":restart_required 
        }
        response =jsonify (response_data )

        return response ,200 
    else :
    # ERSIN Potentially revert app.config changes if file write fails? For simplicity, we don'tity, not doing it now.
        current_app .logger .error (f"Failed to save parameters to config.py: {message }. app.config might be out of sync with config.py.")
        return jsonify ({"error":f"Failed to save parameters to config.py: {message }. Live configuration was updated but changes could not be persisted."}),500 

@bp .route ('/restart_server',methods =['POST'])
def restart_server ():
    """
    Production ortamında manuel server restart endpoint'i
    Systemd servisi üzerinden restart yapar
    """
    try :
    # ERSIN Sadece production ortamında izin ver
        environment =os .environ .get ('FLASK_ENV','development')

        if environment =='development':
            return jsonify ({
            "error":"Manual restart not needed in development mode. Use auto-reload instead."
            }),400 

        current_app .logger .info ("Manual server restart başlatılıyor...")

        # ERSIN Güvenli restart için timer kullan
        def delayed_restart ():
            import time 
            import sys 
            import subprocess 

            # ERSIN Response gönderilmesi için bekle
            time .sleep (2 )

            current_app .logger .info ("🔄 RESTART başlatılıyor...")

            try :
            # ERSIN Systemd servisi olarak çalışıyorsak systemctl kullan
                if os .path .exists ('/etc/systemd/system/wsanaliz.service'):
                    current_app .logger .info ("Systemd servisi bulundu, systemctl restart yapılıyor...")
                    # ERSIN Sudo şifresini environment'tan al, güvenlik için
                    sudo_password =os .environ .get ('SUDO_PASSWORD','5ex5chan5ge4')
                    restart_cmd =f'echo "{sudo_password }" | sudo -S systemctl restart wsanaliz.service'
                    subprocess .Popen (restart_cmd ,shell =True ,stdout =subprocess .DEVNULL ,stderr =subprocess .DEVNULL )
                    current_app .logger .info ("✅ Systemctl restart komutu gönderildi")
                    # ERSIN Process'i sonlandır, systemd yeniden başlatacak
                    os ._exit (0 )
                    # ERSIN Windows için restart
                elif sys .platform =="win32":
                    subprocess .Popen ([sys .executable ]+sys .argv )
                    os ._exit (0 )
                else :
                # ERSIN Linux/Mac için restart, systemd yoksa
                    current_app .logger .info ("Systemd servisi bulunamadı, process restart yapılıyor...")
                    os .kill (os .getpid (),signal .SIGTERM )
            except Exception as restart_err :
                current_app .logger .error (f"Restart hatası: {restart_err }")
                # ERSIN Restart başarısız olursa en azından process'i kill et
                os ._exit (1 )

                # ERSIN Background thread'de restart
        import threading 
        restart_thread =threading .Thread (target =delayed_restart )
        restart_thread .daemon =True 
        restart_thread .start ()

        return jsonify ({
        "message":"Server restart başlatıldı. Lütfen birkaç saniye bekleyin...",
        "success":True 
        }),200 

    except Exception as e :
        current_app .logger .error (f"Manual restart hatası: {str (e )}")
        return jsonify ({
        "error":f"Restart failed: {str (e )}",
        "success":False 
        }),500 