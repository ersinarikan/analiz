"""
Redis-backed analysis queue worker.

- Web prosesinden bağımsız çalışır.
- Redis list (BLPOP) üzerinden analysis_id tüketir.
- Her işi subprocess'te çalıştırır (analysis_subprocess_runner) + GPU lock.
"""

import logging 
import os 
import signal 
import sys 
import time 


logger =logging .getLogger ("wsanaliz.queue_worker")


def _setup_logging ():
    logging .basicConfig (
    level =os .environ .get ("LOG_LEVEL","INFO"),
    stream =sys .stdout ,
    format ="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main ()->int :
    _setup_logging ()

    # ERSIN Force redis backend için worker
    os .environ .setdefault ("WSANALIZ_QUEUE_BACKEND","redis")

    from app import create_app ,initialize_app 
    from app .services import queue_service 

    app ,_socketio =create_app (return_socketio =True )
    # ERSIN Worker'ın kullandığı DB'de tablolar olsun (web ile aynı dosya; create_all idempotent)
    try :
        initialize_app (app )
    except Exception as e :
        logger .warning (f"Worker initialize_app: {e }")

    if not queue_service .is_redis_backend ():
        logger .warning ("Worker redis backend ile çalışmalı; WSANALIZ_QUEUE_BACKEND=redis değil.")

    r =queue_service ._get_redis ()# ERSIN reuse init + url config
    queue_key =queue_service .REDIS_QUEUE_KEY 

    running ={"ok":True }

    def _handle_sig (_signum ,_frame ):
        running ["ok"]=False 
        try :
            queue_service ._set_worker_state (False ,0 )
        except Exception :
            pass 

    signal .signal (signal .SIGTERM ,_handle_sig )
    signal .signal (signal .SIGINT ,_handle_sig )

    logger .info (f"Worker started. redis={queue_service .REDIS_URL } queue_key={queue_key }")
    queue_service ._set_worker_state (False ,0 )

    while running ["ok"]:
    # ERSIN heartbeat (idle)
        queue_service ._set_worker_state (False ,0 )

        # ERSIN Redis client is sync, blpop returns tuple directly (not awaitable)
        blpop_result =r .blpop (queue_key ,timeout =5 )
        if not blpop_result :
            continue 

            # ERSIN blpop returns tuple (key, value) or None
        if isinstance (blpop_result ,tuple )and len (blpop_result )==2 :
            _key ,analysis_id =blpop_result 
        else :
            continue 
        analysis_id =str (analysis_id )
        logger .info (f"Dequeued analysis_id={analysis_id }")

        try :
            queue_service .process_one_analysis (analysis_id ,app )
        except Exception as e :
            logger .error (f"Worker job failed analysis_id={analysis_id }: {e }",exc_info =True )

            # ERSIN küçük nefes
        time .sleep (0.2 )

    logger .info ("Worker stopping.")
    return 0 


if __name__ =="__main__":
    raise SystemExit (main ())

