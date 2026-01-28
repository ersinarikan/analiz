"""
Run a single analysis in a fresh Python process.

Why:
- Gunicorn + eventlet workers occasionally crash with native SIGSEGV (code 139)
  during heavy model inference.
- Running the actual analysis inside a clean subprocess isolates native crashes
  so the web worker stays alive, and the system can reliably mark analysis as
  completed/failed.

Usage:
  python -m app.services.analysis_subprocess_runner <analysis_id> [database_uri]
  database_uri: optional; if given, set DATABASE_URL before create_app (same DB as worker).
Outputs a single JSON line to stdout.
"""

import json 
import sys 
import traceback 
import logging 
import os 

# ERSIN Logging yapılandırması - subprocess'te INFO seviyesinde loglar görünsün
logging .basicConfig (
level =os .environ .get ("LOG_LEVEL","INFO"),
format ="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
stream =sys .stderr ,# ERSIN stderr'e yaz ki log dosyasına gitsin
)


def main (argv :list [str ])->int :
    if len (argv )<2 :
        sys .stderr .write ("usage: python -m app.services.analysis_subprocess_runner <analysis_id> [database_uri]\n")
        return 2 

    analysis_id =argv [1 ]
    # ERSIN Worker ile aynı DB: argv ile geçir (env bazen child'a taşınmıyor)
    if len (argv )>=3 and argv [2 ]:
        os .environ ["DATABASE_URL"]=argv [2 ]

    try :
        from app import create_app ,initialize_app 
        from app .services .analysis_service import analyze_file 

        app ,_socketio =create_app (return_socketio =True )
        with app .app_context ():
            # ERSIN VT tabloları subprocess'te de gerekli (temiz başlangıç sonrası veya boş DB)
            initialize_app (app )
            success ,message =analyze_file (analysis_id )

        sys .stdout .write (json .dumps ({"success":bool (success ),"message":str (message )})+"\n")
        return 0 if success else 1 

    except BaseException as e :
    # ERSIN If we reach here, it's a Python-level failure (native segfaults won't be caught).
        sys .stdout .write (
        json .dumps (
        {
        "success":False ,
        "message":f"Subprocess exception: {e }",
        "traceback":traceback .format_exc (),
        }
        )
        +"\n"
        )
        return 1 


if __name__ =="__main__":
    raise SystemExit (main (sys .argv ))

