@echo off
rem big12's recipe on an E19 fold (docs/EXPERIMENTS.md, E25), so its LM weight and token bonus can be tuned on speakers it never heard.
rem   foldkd.cmd <fold> "<held-out speakers>"      e.g.  foldkd.cmd B "F04,M04,FC01,MC01"
rem The quotes are required (cmd splits unquoted arguments at commas). Needs fold.cmd's packed fold lists and fold<F>_big7 (offset).
rem teacher110 trained on the official split, which includes this fold's test speakers, so the fold gets its own teacher:
rem   1. teacher110_fold<F>: teacher.cmd's teacher, trained on the fold's training lists only.
rem   2. its frame offset against fold<F>_big7.
rem   3. fold<F>_big12: big12.cmd's recipe (teacher on dysarthric clips only) on the fold's training lists.
rem   4. float scores (fold test, LibriSpeech, UASpeech) and the int8 export for the host engine.
rem Run on the training PC after any other GPU job: cmd /c G:\cosgc\repo3\train\runs\foldkd.cmd B "F04,M04,FC01,MC01" > G:\cosgc\exp\foldB_kd_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
set HF_HOME=G:\cosgc\hf
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
set NPY=G:\cosgc\nemovenv\Scripts\python.exe -u
set TOK=%E%\base24x\tokenizer.model
set F=%1
set H=%~2
set T=torgo_fold%F%
set TT=%E%\teacher110_fold%F%
set O=%E%\fold%F%_big12
echo fold %F%, held out %H%, start %date% %time%
if not exist %E%\fold%F%_big7\done (echo fold%F%_big7 missing & exit /b 1)
%PY% -c "import json,sys;s=sorted({json.loads(l)['speaker'] for l in open(r'%D%\%T%_test.jsonl',encoding='utf8')});w=sorted('%H%'.split(','));print('test speakers',s);sys.exit(s!=w)" || (echo held-out speakers do not match & exit /b 1)
%PY% -c "import json,os,sys;h=set('%H%'.split(','));R=[json.loads(l) for f in (r'%P%\%T%_train.jsonl',r'%P%\%T%_dys_x8.jsonl') for l in open(f,encoding='utf8')];bad=sorted({r['speaker'] for r in R}&h);miss=sorted({r['pack'] for r in R if not os.path.exists(r['pack'])});print('packed training: test speakers',bad,'missing packs',miss);sys.exit(bool(bad or miss))" || (echo packed training data is stale & exit /b 1)
set L=%P%\ls100_train.jsonl,%P%\%T%_train.jsonl,%P%\%T%_train.jsonl,%P%\%T%_dys_x8.jsonl
if not exist %TT%\teacher.pt start "" /belownormal /b /wait %NPY% teacher_ft.py train --train %L% ^
  --tokenizer %TOK% --out %TT% --steps 6000 --batch_sec 100 --lr 1e-4 --warmup 500 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\teacher110_fold%F%.log 2>&1
if not exist %TT%\teacher.pt (echo teacher training failed & exit /b 1)
echo teacher done %date% %time%
if not exist %E%\eval\teacher110_fold%F%_%T%_test.json %NPY% teacher_ft.py eval --ckpt %TT%\teacher.pt --tokenizer %TOK% --manifest %D%\%T%_test.jsonl --out %E%\eval\teacher110_fold%F%_%T%_test.json
if not exist %TT%_offset.txt %NPY% teacher_ft.py offset --ckpt %TT%\teacher.pt --nemo %E%\base24x --student %E%\fold%F%_big7\last.pt --manifest %P%\%T%_train.jsonl --limit 50 > %TT%_offset.txt
set SHIFT=
for /f "tokens=3" %%s in ('findstr /c:"best --teacher_shift" %TT%_offset.txt') do set SHIFT=%%s
if not defined SHIFT (echo offset failed & exit /b 1)
echo teacher shift %SHIFT%
if not exist %O%\done start "" /belownormal /b /wait %NPY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %L% --val %D%\%T%_test.jsonl,%D%\ls_testclean_train.jsonl --out %O% ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 6000 --batch_sec 100 --val_every 750 --save_every 250 ^
  --p_dys 0.3 --dys_severity 0.2,1.0 ^
  --teacher %TT%\teacher.pt --teacher_kd 0.5 --teacher_T 2 --teacher_shift %SHIFT% --teacher_groups dysarthric ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\fold%F%_big12.log 2>&1
if not exist %O%\done (echo training failed & exit /b 1)
echo student done %date% %time%
for %%M in (%T%_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %O%\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\fold%F%_big12_%%M.json
%PY% export_nemo.py %E%\base24x %O%\fold%F%_big12_int8.tnm 8 %O%\last.pt
echo DONE %date% %time%
