@echo off
rem atyp2 + UASpeech blocks B1 and B3 (microphones M5 and M2) in training (docs/EXPERIMENTS.md, E7). Everything else
rem identical to atyp2, so the difference measures the added data. Tested on the held-out TORGO speakers, LibriSpeech and
rem UASpeech block B2 (same UASpeech speakers, other recordings; the B2 "uncommon" words never occur in B1 or B3).
rem Also tests the shipped model, atyp2 and big2 on UASpeech B2 for comparison. Waits for big2 to finish.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\atyp4.cmd > G:\cosgc\exp\atyp4_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
:wait
findstr /c:"DONE" G:\cosgc\exp\big2_all.log >nul 2>&1 || (timeout /t 120 /nobreak >nul & goto wait)
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
if not exist %P%\uaspeech_train_b13.jsonl %PY% pack_audio.py %D%\uaspeech_train_b13.jsonl %P%\uaspeech_train_b13 || (echo pack failed & exit /b 1)
set U=--manifest %D%\uaspeech_test_b2.jsonl
%PY% eval_manifest.py --nemo %E%\base --init %E%\base\init.pt %U% --out %E%\eval\base_uaspeech_test_b2.json
%PY% eval_manifest.py --nemo %E%\base --ckpt %E%\atyp2\last.pt --init %E%\base\init.pt %U% --out %E%\eval\atyp2_uaspeech_test_b2.json
if exist %E%\big2\last.pt %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big2\last.pt --init %E%\base24x\init.pt %U% --out %E%\eval\big2_uaspeech_test_b2.json
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base --init %E%\base\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\uaspeech_train_b13.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\atyp4 ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\atyp4.log 2>&1
if not exist %E%\atyp4\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base --ckpt %E%\atyp4\last.pt --init %E%\base\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\atyp4_%%M.json
echo DONE
