import logging 
import os 
import socket 

logger =logging .getLogger (__name__ )

# ERSIN unix_chkpwd sadece çağıran kullanıcının şifresini doğrular; gunicorn ersin ile
# ERSIN çalıştığı için başka kullanıcılar (serdar vb.) doğrulanamaz. Bu yüzden önce
# ERSIN root’ta çalışan wsanaliz-pamauth servisine Unix soket ile soruyoruz. Soket
# ERSIN yoksa veya hata olursa doğrudan PAM’a düşer (sadece ersin kendisi giriş yapabilir).


class PamAuthError (RuntimeError ):
    pass 


def _pam_via_socket (username :str ,password :str )->bool |None :
    """
    Root’taki wsanaliz-pamauth daemon’a sor. Başarı/red: True/False.
    Bağlantı yok veya hata: None (çağıran doğrudan PAM’a düşsün).
    """
    # ERSIN If env var dir explicitly set to empty, treat it as "disable socket".
    raw_path =os .environ .get ("WSANALIZ_PAMAUTH_SOCKET")
    if raw_path is not None :
        path =raw_path .strip ()
        if not path :
            return None 
    else :
        path ="/run/wsanaliz-pamauth/socket"
    try :
        s =socket .socket (socket .AF_UNIX ,socket .SOCK_STREAM )
        try :
            s .settimeout (5.0 )
            s .connect (path )
            buf =f"AUTH\n{username }\n{password }\n"
            s .sendall (buf .encode ("utf-8"))
            data =b""
            while len (data )<16 and (chunk :=s .recv (64 )):
                data +=chunk 
        finally :
            try :
                s .close ()
            except Exception :
                pass 

        first_line =(data .decode ("utf-8",errors ="replace").splitlines ()[0 ]if data else "").strip ()
        if not first_line :
            return None 
            # ERSIN Exact match: avoid substring false-positives.
        return first_line =="OK"
    except FileNotFoundError :
        logger .debug ("pamauth socket not found, using direct PAM")
        return None 
    except (ConnectionRefusedError ,socket .error ,OSError ,TimeoutError )as e :
        logger .debug ("pamauth socket error: %s",e )
        return None 


def pam_authenticate (username :str ,password :str ,service :str |None =None )->bool :
    """
    PAM ile kimlik doğrula. Önce wsanaliz-pamauth soketine sorar (root’ta
    çalışır, tüm kullanıcılar doğrulanır); soket yoksa/hatada doğrudan PAM
    (unix_chkpwd yüzünden sadece süreç sahibi, örn. ersin, kendi şifresini doğrular).
    """
    username =(username or "").strip ()
    password =password or ""
    if not username or not password :
        return False 

        # ERSIN Önce pamauth soketini dene (root, tüm kullanıcılar)
    out =_pam_via_socket (username ,password )
    if out is not None :
        if not out :
            logger .info ("PAM auth failed for user=%s (via pamauth) password_len=%d",username ,len (password ))
        return out 

        # ERSIN Soket yok/hatalı: doğrudan PAM (unix_chkpwd sadece süreç sahibini doğrular)
    pam_service =(service or os .environ .get ("WSANALIZ_PAM_SERVICE")or "su").strip ()
    try :
    # ERSIN PAM is an optional dependency, import safely
        import pam 
    except Exception as e :
        raise PamAuthError (f"python-pam import failed: {e }")from e 

    try :
        p =pam .pam ()
        ok =bool (p .authenticate (username ,password ,service =pam_service ))
        if not ok :
            logger .info ("PAM auth failed for user=%s service=%s password_len=%d",username ,pam_service ,len (password ))
        return ok 
    except Exception as e :
        logger .exception ("PAM auth exception for user=%s service=%s: %s",username ,pam_service ,e )
        return False 
