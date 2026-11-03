@echo off
rem Rotating which TORGO speakers are held out (docs/PAPER.md section 8: two dysarthric test speakers are too few).
rem   fold.cmd <fold> "<held-out speakers>" <recipe>     e.g.  fold.cmd B "F04,M04,FC01,MC01" big6
rem The quotes are required: cmd splits unquoted arguments at commas (the first launch held out one speaker and ran the wrong recipe).
rem Fold A is the official split (F03, M05, FC03, MC04: big6, big7 and every earlier run). Folds B and C each hold out one mild and one
rem severe dysarthric speaker and two controls; M01 and M02 stay in training in every fold.
rem   B: F04 (mild), M04 (severe), FC01, MC01        C: M03 (mild), F01 (severe), FC02, MC02
rem Each fold has its own test speakers; nothing is fitted on them (the base model is only scored).
rem Recipes, identical to the named run except for the TORGO split:
rem   big6: real data only (LibriSpeech 100 + TORGO train x2 + its dysarthric clips x8), 4000 steps of 150 s
rem   big7: big6 + the mel-level simulation on 30 % of LibriSpeech clips, 6000 steps of 100 s
rem Caveat for big7: the mel-level simulation's settings were calibrated on all of TORGO's training speakers of fold A (E5), which include
rem fold B's and C's test speakers; a small leak in the simulated arm only. The base model is scored on each fold's test set as the reference.
rem Run on the training PC after any other training: cmd /c G:\cosgc\repo3\train\runs\fold.cmd B "F04,M04,FC01,MC01" big6 > G:\cosgc\exp\foldB_big6_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
set F=%1
set H=%~2
set R=%3
set T=torgo_fold%F%
set X=
if "%R%"=="big6" set X=--steps 4000 --batch_sec 150 --val_every 500
if "%R%"=="big7" set X=--steps 6000 --batch_sec 100 --val_every 750 --p_dys 0.3 --dys_severity 0.2,1.0
if not defined X (echo recipe must be big6 or big7, got "%R%" & exit /b 1)
echo fold %F%, held out %H%, recipe %R%
if not exist %D%\%T%_test.jsonl %PY% make_manifest_torgo.py %D%\TORGO %D%\%T% --heldout %H% || (echo manifest failed & exit /b 1)
%PY% -c "import json,sys;s=sorted({json.loads(l)['speaker'] for l in open(r'%D%\%T%_test.jsonl',encoding='utf8')});w=sorted('%H%'.split(','));print('test speakers',s);sys.exit(s!=w)" || (echo held-out speakers do not match & exit /b 1)
if not exist %P%\%T%_train.jsonl %PY% pack_audio.py %D%\%T%_train.jsonl %P%\%T%_train || (echo pack failed & exit /b 1)
if not exist %P%\%T%_dys_x8.jsonl %PY% -c "import json;L=[l for l in open(r'%P%\%T%_train.jsonl',encoding='utf8') if json.loads(l)['group']=='dysarthric'];open(r'%P%\%T%_dys_x8.jsonl','w',encoding='utf8',newline='\n').writelines(L*8);print(len(L),'dysarthric clips x 8')"
%PY% -c "import json,os,sys;h=set('%H%'.split(','));R=[json.loads(l) for f in (r'%P%\%T%_train.jsonl',r'%P%\%T%_dys_x8.jsonl') for l in open(f,encoding='utf8')];bad=sorted({r['speaker'] for r in R}&h);miss=sorted({r['pack'] for r in R if not os.path.exists(r['pack'])});print('packed training: test speakers',bad,'missing packs',miss);sys.exit(bool(bad or miss))" || (echo packed training data is stale & exit /b 1)
if not exist %E%\eval\base_%T%_test.json %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\base24x\init.pt --manifest %D%\%T%_test.jsonl --out %E%\eval\base_%T%_test.json
set O=%E%\fold%F%_%R%
if not exist %O%\last.pt start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\%T%_train.jsonl,%P%\%T%_train.jsonl,%P%\%T%_dys_x8.jsonl ^
  --val %D%\%T%_test.jsonl,%D%\ls_testclean_train.jsonl --out %O% ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 %X% --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\fold%F%_%R%.log 2>&1
if not exist %O%\last.pt (echo training failed & exit /b 1)
for %%M in (%T%_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %O%\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\fold%F%_%R%_%%M.json
echo DONE
