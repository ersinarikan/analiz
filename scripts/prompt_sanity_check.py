#!/usr/bin/env python3
"""
Offline prompt sanity-check tool (etiketsiz hızlı kontrol).

Amaç:
- Mevcut ContentAnalyzer prompt setleri ve template ensembling'in davranışını
  hızlıca gözlemlemek (video/frame/image üzerinde).
- Etiketli dataset gerekmez; sadece skor dağılımına bakılır.

Kullanım örnekleri:
  ./venv/bin/python scripts/prompt_sanity_check.py --input storage/processed/frames --limit 20
  ./venv/bin/python scripts/prompt_sanity_check.py --input /path/to/video.mp4 --video-frames 10
  ./venv/bin/python scripts/prompt_sanity_check.py --input /path/to/image.jpg
"""

import argparse 
import contextlib 
import json 
import os 
import sys 
import tempfile 
from pathlib import Path 


def _iter_images_from_dir (d :Path ):
    exts ={".jpg",".jpeg",".png",".webp",".bmp"}
    for p in sorted (d .rglob ("*")):
        if p .is_file ()and p .suffix .lower ()in exts :
            yield p 


def _extract_frames_from_video (video_path :Path ,out_dir :Path ,frame_count :int )->list [Path ]:
    import cv2 

    cap =cv2 .VideoCapture (str (video_path ))
    if not cap .isOpened ():
        raise RuntimeError (f"Cannot open video: {video_path }")

    total =int (cap .get (cv2 .CAP_PROP_FRAME_COUNT )or 0 )
    if total <=0 :
    # ERSIN Fall back: read sequentially
        total =frame_count 

    indices =[]
    if frame_count <=1 :
        indices =[0 ]
    else :
        step =max (1 ,total //frame_count )
        indices =[i *step for i in range (frame_count )]

    saved :list [Path ]=[]
    for idx in indices :
        cap .set (cv2 .CAP_PROP_POS_FRAMES ,idx )
        ok ,frame =cap .read ()
        if not ok or frame is None :
            continue 
        out =out_dir /f"{video_path .stem }_f{idx }.jpg"
        cv2 .imwrite (str (out ),frame )
        saved .append (out )

    cap .release ()
    return saved 


def main (argv :list [str ])->int :
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--input",required =True ,help ="Image file, directory, or video file")
    parser .add_argument ("--limit",type =int ,default =20 ,help ="Max number of images to score (directory mode)")
    parser .add_argument ("--video-frames",type =int ,default =10 ,help ="Frames to sample from a video input")
    parser .add_argument ("--json",action ="store_true",help ="Output JSON lines (default: pretty text)")
    args =parser .parse_args (argv [1 :])

    # ERSIN Keep stdout clean for JSONL output: route logs/prints to stderr
    # ERSIN Some parts of app initialization use print() (not logging), so we redirect stdout during init
    os .environ .setdefault ("LOG_LEVEL","WARNING")
    # ERSIN Sanity-check should be lightweight and not fight with the running production services
    # ERSIN Default to CPU + disable memory management unless user explicitly overrides via env
    os .environ .setdefault ("USE_GPU","False")
    os .environ .setdefault ("WSANALIZ_DISABLE_MEMORY_MANAGEMENT","1")
    os .environ .setdefault ("WSANALIZ_SANITY_MODE","1")

    inp =Path (args .input ).expanduser ().resolve ()
    if not inp .exists ():
        sys .stderr .write (f"Input not found: {inp }\n")
        return 2 

        # ERSIN Ensure project root in sys.path
    project_root =Path (__file__ ).resolve ().parents [1 ]
    sys .path .insert (0 ,str (project_root ))

    # ERSIN Import + init app/analyzer with stdout redirected to stderr to avoid contaminating JSON output
    with contextlib .redirect_stdout (sys .stderr ):
        from app import create_app 
        from app .ai .content_analyzer import ContentAnalyzer 

        created =create_app ("default")
        # ERSIN create_app may return (app, socketio) in some contexts, support both
        flask_app =created [0 ]if isinstance (created ,tuple )else created 

    paths :list [Path ]=[]
    tmpdir_ctx =None 

    try :
        if inp .is_dir ():
            for i ,p in enumerate (_iter_images_from_dir (inp )):
                if i >=args .limit :
                    break 
                paths .append (p )
        elif inp .suffix .lower ()in {".mp4",".mov",".mkv",".avi",".webm"}:
            tmpdir_ctx =tempfile .TemporaryDirectory (prefix ="wsanaliz_sanity_")
            out_dir =Path (tmpdir_ctx .name )
            paths =_extract_frames_from_video (inp ,out_dir ,args .video_frames )
        else :
            paths =[inp ]

        if not paths :
            sys .stderr .write ("No images/frames found.\n")
            return 1 

        with flask_app .app_context ():
            with contextlib .redirect_stdout (sys .stderr ):
                analyzer =ContentAnalyzer ()

            for p in paths :
                try :
                # ERSIN returns: (violence, adult_content, harassment, weapon, drug, safe, ...) + objects
                    result =analyzer .analyze_image (str (p ))
                    # ERSIN analyzer returns a long tuple, normalize into dict
                    # ERSIN See content_analyzer.py: all_category_keys_for_return = keys + ['safe']
                    categories =list (analyzer .category_prompts .keys ())+["safe"]
                    # ERSIN result is a tuple, extract numeric scores and convert to float
                    score_values =result [:len (categories )]
                    scores :dict [str ,float ]={}
                    for k ,v in zip (categories ,score_values ):
                        if isinstance (v ,(int ,float )):
                            scores [k ]=float (v )
                        else :
                            scores [k ]=0.0 
                    objects =result [-1 ]if isinstance (result [-1 ],list )else []

                    top =sorted (scores .items (),key =lambda kv :kv [1 ],reverse =True )[:3 ]
                    payload ={
                    "path":str (p ),
                    "top3":top ,
                    "scores":scores ,
                    "objects":objects ,
                    }

                    if args .json :
                        sys .stdout .write (json .dumps (payload ,ensure_ascii =False )+"\n")
                    else :
                        print (f"\n== {p } ==")
                        print ("top3:",", ".join ([f"{k }={v :.3f}"for k ,v in top ]))
                        print ("scores:",json .dumps (scores ,ensure_ascii =False ))

                except Exception as e :
                    sys .stderr .write (f"Failed scoring {p }: {e }\n")

        return 0 

    finally :
        if tmpdir_ctx is not None :
            tmpdir_ctx .cleanup ()


if __name__ =="__main__":
    raise SystemExit (main (sys .argv ))

