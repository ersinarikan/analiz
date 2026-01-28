#!/usr/bin/env python3
"""
Lightweight production monitor for analysis durations.

Polls the WSANALIZ API and logs per-analysis lifecycle timing to a JSONL file.

Use-case:
- User uploads/starts analyses via UI.
- This script records how long each analysis stays in each state and total wall time.

Example:
  ./venv/bin/python scripts/monitor_analysis_durations.py --poll 5 --out logs/analysis_monitor.jsonl
"""

import argparse 
import json 
import time 
from typing import Any 

import requests 


FINAL_STATUSES ={"completed","failed","cancelled"}


def _now ()->float :
    return time .time ()


def _safe_get (d :dict [str ,Any ],path :list [str ],default =None ):
    cur :Any =d 
    for k in path :
        if not isinstance (cur ,dict )or k not in cur :
            return default 
        cur =cur [k ]
    return cur 


def main (argv :list [str ])->int :
    p =argparse .ArgumentParser ()
    p .add_argument ("--base-url",default ="http://127.0.0.1:5000",help ="WSANALIZ web base URL")
    p .add_argument ("--poll",type =float ,default =5.0 ,help ="Poll interval seconds")
    p .add_argument ("--max-recent",type =int ,default =50 ,help ="Max recent analyses to scan each poll")
    p .add_argument ("--out",default ="/opt/wsanaliz/logs/analysis_monitor.jsonl",help ="JSONL output path")
    args =p .parse_args (argv [1 :])

    base_url =args .base_url .rstrip ("/")
    poll_s =max (1.0 ,float (args .poll ))

    # ERSIN state: analysis_id -> info
    seen :dict [str ,dict [str ,Any ]]={}

    session =requests .Session ()
    session .headers .update ({"Accept":"application/json"})

    # ERSIN ensure output dir exists, best effort
    try :
        import os 

        os .makedirs (os .path .dirname (args .out ),exist_ok =True )
    except Exception :
        pass 

    with open (args .out ,"a",encoding ="utf-8")as out_f :
        while True :
            t0 =_now ()
            # ERSIN We prefer in-flight visibility, so we enumerate files and query /api/analysis/file/<file_id>
            analyses :list [dict [str ,Any ]]=[]
            try :
                files_resp =session .get (f"{base_url }/api/files/",timeout =5 )
                files_resp .raise_for_status ()
                files =(files_resp .json ()or {}).get ("files")or []
                if isinstance (files ,list ):
                    files =files [:args .max_recent ]
                for f in files :
                    fid =f .get ("id")
                    if not fid :
                        continue 
                    try :
                        aresp =session .get (f"{base_url }/api/analysis/file/{fid }",timeout =5 )
                        if not aresp .ok :
                            continue 
                        arr =aresp .json ()
                        if isinstance (arr ,list ):
                            analyses .extend (arr )
                    except Exception :
                        continue 
            except Exception as e :
                out_f .write (json .dumps ({"ts":t0 ,"type":"error","where":"files+per_file","error":str (e )},ensure_ascii =False )+"\n")
                out_f .flush ()
                time .sleep (poll_s )
                continue 

            for a in analyses :
                aid =str (a .get ("id")or "")
                if not aid :
                    continue 

                status =str (a .get ("status")or "")
                file_id =a .get ("file_id")
                filename =_safe_get (a ,["file_info","original_filename"])or _safe_get (a ,["file_info","filename"])

                entry =seen .get (aid )
                if entry is None :
                    entry ={
                    "analysis_id":aid ,
                    "first_seen_ts":t0 ,
                    "first_status":status ,
                    "last_status":status ,
                    "file_id":file_id ,
                    "filename":filename ,
                    }
                    seen [aid ]=entry 
                    out_f .write (json .dumps ({"ts":t0 ,"type":"seen",**entry },ensure_ascii =False )+"\n")
                    out_f .flush ()
                else :
                    if status and status !=entry .get ("last_status"):
                        out_f .write (
                        json .dumps (
                        {
                        "ts":t0 ,
                        "type":"status_change",
                        "analysis_id":aid ,
                        "file_id":file_id ,
                        "filename":filename ,
                        "from":entry .get ("last_status"),
                        "to":status ,
                        "wall_elapsed_s":round (t0 -float (entry .get ("first_seen_ts")or t0 ),3 ),
                        },
                        ensure_ascii =False ,
                        )
                        +"\n"
                        )
                        entry ["last_status"]=status 
                        out_f .flush ()

                if status in FINAL_STATUSES and not entry .get ("final_logged"):
                # ERSIN Fetch full analysis record for timestamps/message
                    detail :dict [str ,Any ]={}
                    try :
                        rr =session .get (f"{base_url }/api/analysis/{aid }",timeout =5 )
                        if rr .ok :
                            detail =rr .json ()
                    except Exception :
                        detail ={}

                    out_f .write (
                    json .dumps (
                    {
                    "ts":t0 ,
                    "type":"final",
                    "analysis_id":aid ,
                    "file_id":detail .get ("file_id",file_id ),
                    "filename":_safe_get (detail ,["file_info","original_filename"])or filename ,
                    "status":detail .get ("status",status ),
                    "start_time":detail .get ("start_time"),
                    "end_time":detail .get ("end_time"),
                    "error_message":detail .get ("error_message"),
                    "wall_elapsed_s":round (t0 -float (entry .get ("first_seen_ts")or t0 ),3 ),
                    },
                    ensure_ascii =False ,
                    )
                    +"\n"
                    )
                    entry ["final_logged"]=True 
                    out_f .flush ()

                    # ERSIN sleep remaining
            dt =_now ()-t0 
            time .sleep (max (0.0 ,poll_s -dt ))


if __name__ =="__main__":
    raise SystemExit (main (__import__ ("sys").argv ))

