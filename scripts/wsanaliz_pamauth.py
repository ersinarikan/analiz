#!/usr/bin/env python3
"""
PAM auth yardımcı servisi (root).

unix_chkpwd sadece çağıran kullanıcının şifresini doğrular. Gunicorn ersin ile
çalıştığı için serdar/ayşe vb. doğrulanamıyor. Bu daemon root ile PAM çağırır;
tüm sistem kullanıcıları doğrulanabilir.

Protokol (Unicode, satır sonu \\n):
  İstemci: "AUTH\\n" + kullanıcı + "\\n" + şifre + "\\n"
  Sunucu: "OK\\n" veya "FAIL\\n"

Sadece WSANALIZ_PAMAUTH_ALLOW_UID ile belirtilen UID'den gelen bağlantılar
kabul edilir (varsayılan: 1001 ersin). Şifre asla loglanmaz.
"""

from __future__ import annotations 

import logging 
import os 
import socket 
import sys 

try :
# ERSIN PAM is an optional dependency, import safely
    import pam 
except ImportError :
    pam =None 

logging .basicConfig (
level =logging .INFO ,
format ="%(asctime)s wsanaliz-pamauth %(levelname)s: %(message)s",
datefmt ="%Y-%m-%d %H:%M:%S",
)
logger =logging .getLogger (__name__ )

SOCKET_PATH =os .environ .get ("WSANALIZ_PAMAUTH_SOCKET")or "/run/wsanaliz-pamauth/socket"
PAM_SERVICE =(os .environ .get ("WSANALIZ_PAM_SERVICE")or "su").strip ()
# ERSIN Sadece bu UID'den gelen bağlantılar kabul (wsanaliz-web User=ersin → 1001)
try :
    ALLOW_UID =int (os .environ .get ("WSANALIZ_PAMAUTH_ALLOW_UID","1001").strip ())
except Exception :
    ALLOW_UID =1001 

    # ERSIN Bağlantıda soket grubunu ayarlamak için (ersin'in gid'i)
SOCKET_GROUP =(os .environ .get ("WSANALIZ_PAMAUTH_SOCKET_GROUP")or "ersin").strip ()


def get_socket_gid ()->int |None :
    try :
        import grp 

        return grp .getgrnam (SOCKET_GROUP ).gr_gid 
    except Exception :
        return None 


def handle (conn :socket .socket )->None :
    try :
        cred =conn .getsockopt (socket .SOL_SOCKET ,socket .SO_PEERCRED ,struct_size ())
        _pid ,uid ,_gid =parse_peercred (cred )
        if uid !=ALLOW_UID :
            logger .warning ("Rejected connection from uid=%s (allowed=%s)",uid ,ALLOW_UID )
            conn .sendall (b"FAIL\n")
            return 
    except Exception as e :
        logger .warning ("SO_PEERCRED failed: %s",e )
        conn .sendall (b"FAIL\n")
        return 

    try :
        f =conn .makefile ("rwb")
        line1 =(f .readline ()or b"").decode ("utf-8",errors ="replace").strip ()
        if line1 !="AUTH":
            conn .sendall (b"FAIL\n")
            return 
        username =(f .readline ()or b"").decode ("utf-8",errors ="replace").rstrip ("\n\r").strip ()
        password =(f .readline ()or b"").decode ("utf-8",errors ="replace").rstrip ("\n\r")
        f .close ()
    except Exception :
        try :
            conn .sendall (b"FAIL\n")
        except Exception :
            pass 
        return 

    if "\x00"in username or "\x00"in password :
        conn .sendall (b"FAIL\n")
        return 
    if len (username )>256 or len (password )>1024 :
        conn .sendall (b"FAIL\n")
        return 

    if not pam :
        logger .error ("python-pam not installed")
        conn .sendall (b"FAIL\n")
        return 

    try :
        p =pam .pam ()
        ok =bool (p .authenticate (username ,password ,service =PAM_SERVICE ))
        conn .sendall (b"OK\n"if ok else b"FAIL\n")
        if not ok :
            logger .info ("PAM auth failed for user=%s",username )
    except Exception as e :
        logger .exception ("PAM exception for user=%s: %s",username ,e )
        conn .sendall (b"FAIL\n")


def struct_size ()->int :
    import struct
    _ = struct  # used for struct.calcsize in peercred; size fixed 12

    # ERSIN SO_PEERCRED: struct ucred { pid_t pid; uid_t uid; gid_t gid; }
    # ERSIN Linux: 4+4+4 = 12 on 32-bit, on 64-bit often 4+4+4 (uint32 each) = 12
    return 12 


def parse_peercred (data :bytes )->tuple [int ,int ,int ]:
    import struct 

    # ERSIN ucred: I=uint32 for pid, uid, gid (Linux)
    t =struct .unpack ("III",data [:12 ])
    return (t [0 ],t [1 ],t [2 ])


def main ()->None :
    if os .geteuid ()!=0 :
        logger .error ("Must run as root")
        sys .exit (1 )

    sockdir =os .path .dirname (SOCKET_PATH )
    os .makedirs (sockdir ,mode =0o750 ,exist_ok =True )
    if os .path .exists (SOCKET_PATH ):
        os .unlink (SOCKET_PATH )

    sock =socket .socket (socket .AF_UNIX ,socket .SOCK_STREAM )
    sock .setsockopt (socket .SOL_SOCKET ,socket .SO_REUSEADDR ,1 )
    sock .bind (SOCKET_PATH )
    sock .listen (16 )

    gid =get_socket_gid ()
    if gid is not None :
        try :
            os .chown (SOCKET_PATH ,0 ,gid )
            os .chmod (SOCKET_PATH ,0o660 )
        except Exception as e :
            logger .warning ("chown/chmod socket: %s",e )
    else :
        os .chmod (SOCKET_PATH ,0o660 )

    logger .info ("Listening on %s (PAM service=%s, allow_uid=%s)",SOCKET_PATH ,PAM_SERVICE ,ALLOW_UID )

    while True :
        try :
            conn ,_ =sock .accept ()
            try :
                conn .settimeout (10.0 )
                handle (conn )
            except Exception as e :
                logger .exception ("handle: %s",e )
                try :
                    conn .sendall (b"FAIL\n")
                except Exception :
                    pass 
            finally :
                try :
                    conn .close ()
                except Exception :
                    pass 
        except KeyboardInterrupt :
            break 
        except Exception as e :
            logger .exception ("accept: %s",e )

    sock .close ()
    if os .path .exists (SOCKET_PATH ):
        os .unlink (SOCKET_PATH )
    logger .info ("Stopped")


if __name__ =="__main__":
    main ()
