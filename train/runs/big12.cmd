@echo off
rem big12 (docs/EXPERIMENTS.md, E22): big11 exactly (teacher.cmd), but the teacher term only on dysarthric clips (--teacher_groups dysarthric).
rem The E21 teacher was better than big7 only on dysarthric speech, and passed its weakness on typical speech to big11.
rem Needs teacher.cmd's teacher110\teacher.pt and its offset (+4). The comparison is big7, paired.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big12.cmd > G:\cosgc\exp\big12_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
set HF_HOME=G:\cosgc\hf
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
set NPY=G:\cosgc\nemovenv\Scripts\python.exe -u
set T=%E%\teacher110
if not exist %T%\teacher.pt (echo no teacher & exit /b 1)
set SHIFT=
for /f "tokens=3" %%s in ('findstr /c:"best --teacher_shift" %E%\teacher110_offset.txt') do set SHIFT=%%s
if not defined SHIFT (echo offset missing & exit /b 1)
echo teacher shift %SHIFT%
if not exist %E%\big12\done start "" /belownormal /b /wait %NPY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big12 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 6000 --batch_sec 100 --val_every 750 --save_every 250 ^
  --p_dys 0.3 --dys_severity 0.2,1.0 ^
  --teacher %T%\teacher.pt --teacher_kd 0.5 --teacher_T 2 --teacher_shift %SHIFT% --teacher_groups dysarthric ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big12.log 2>&1
if not exist %E%\big12\done (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big12\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big12_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big12\big12_int8.tnm 8 %E%\big12\last.pt
echo DONE
