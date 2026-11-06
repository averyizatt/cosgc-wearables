@echo off
rem big13 (docs/EXPERIMENTS.md, E24): big7 plus UASpeech training blocks (B1, B3) from every speaker except eight held out whole:
rem dysarthric F03, M07, F04, M09 (one per intelligibility band of the UASpeech paper's table: 6, 28, 62, 86 %) and controls
rem CF03, CF05, CM05, CM10. E11 added all 28 speakers and was then tested on the same people; this tests on people it never heard.
rem Steps 6000 -> 6500: the list grows from 150 h to 162 h, so TORGO and LibriSpeech are seen as often as in big7.
rem UASpeech is added to --dys_sources so the mel-level simulation never distorts it (big7 simulates LibriSpeech only).
rem Waits for 01:00. Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big13.cmd > G:\cosgc\exp\big13_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
set N=big13
set H=F03,M07,F04,M09,CF03,CF05,CM05,CM10
:t
for /f %%h in ('powershell -NoProfile -Command "(Get-Date).Hour"') do set HH=%%h
if %HH% GEQ 7 (timeout /t 120 >nul & goto t)
if %HH% LSS 1 (timeout /t 120 >nul & goto t)
echo start %date% %time%
%PY% split_speakers.py %P%\uaspeech_train_b13.jsonl %P%\uaspeech_train_b13_held.jsonl --drop %H% || (echo split failed & exit /b 1)
%PY% split_speakers.py %D%\uaspeech_test_b2.jsonl %D%\uaspeech_test_b2_held.jsonl --keep %H% || (echo split failed & exit /b 1)
if not exist %E%\%N%\done start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl,%P%\uaspeech_train_b13_held.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\%N% ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 6500 --batch_sec 100 --val_every 750 --save_every 250 ^
  --p_dys 0.3 --dys_severity 0.2,1.0 --dys_sources torgo,uaspeech ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\%N%.log 2>&1
if not exist %E%\%N%\done (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2_held uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\%N%\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\%N%_%%M.json
rem the same held-out speakers for big7, the comparison
if not exist %E%\eval\big7_uaspeech_test_b2_held.json %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big7\last.pt --init %E%\base24x\init.pt --manifest %D%\uaspeech_test_b2_held.jsonl --out %E%\eval\big7_uaspeech_test_b2_held.json
%PY% export_nemo.py %E%\base24x %E%\%N%\%N%_int8.tnm 8 %E%\%N%\last.pt
echo DONE
