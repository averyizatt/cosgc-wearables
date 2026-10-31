@echo off
rem big2 + UASpeech blocks B1 and B3 (mics M5 and M2) with the corrected transcripts (docs/EXPERIMENTS.md, E7: atyp4's
rem UASpeech transcripts repeated each word about 10 times). Same 24-layer start and recipe as big2, so the difference
rem measures the added data. Tested on TORGO, LibriSpeech and UASpeech block B2. Waits for big2q to finish.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big3.cmd > G:\cosgc\exp\big3_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
:wait
findstr /c:"DONE" G:\cosgc\exp\big2q_all.log >nul 2>&1 || (timeout /t 120 /nobreak >nul & goto wait)
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
if not exist %P%\uaspeech_train_b13.jsonl %PY% pack_audio.py %D%\uaspeech_train_b13.jsonl %P%\uaspeech_train_b13 || (echo pack failed & exit /b 1)
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\uaspeech_train_b13.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big3 ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big3.log 2>&1
if not exist %E%\big3\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big3\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big3_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big3\big3_int8.tnm 8 %E%\big3\last.pt
echo DONE
