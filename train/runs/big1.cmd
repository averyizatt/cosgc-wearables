@echo off
rem First deeper model for the ESP32-P4: the shipped 16-layer model grown to 24 layers (grow_nemo.py, 13 -> 19 M parameters),
rem then the atyp2 recipe unchanged, so the difference to atyp2 measures the extra depth alone (docs/EXPERIMENTS.md, E6).
rem Also evaluates the grown model before training (how much inserting the layers costs).
rem Waits for atyp3 (and its evaluations) to finish first. Runs code from G:\cosgc\repo3.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big1.cmd > G:\cosgc\exp\big1_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
:wait
findstr /c:"DONE" G:\cosgc\exp\atyp3_all.log >nul 2>&1 || (timeout /t 120 /nobreak >nul & goto wait)
set D=G:\cosgc\data
set E=G:\cosgc\exp
set PY=G:\cosgc\venv\Scripts\python.exe -u
if not exist %E%\base24\init.pt %PY% grow_nemo.py --nemo %E%\base --init %E%\base\init.pt --layers 24 --out %E%\base24 || (echo grow failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base24 --ckpt %E%\base24\init.pt --init %E%\base24\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\base24_untrained_%%M.json
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24 --init %E%\base24\init.pt --keep_head ^
  --train %D%\ls100_train.jsonl,%D%\torgo_train.jsonl,%D%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big1 ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 > %E%\big1.log 2>&1
if not exist %E%\big1\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base24 --ckpt %E%\big1\last.pt --init %E%\base24\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big1_%%M.json
%PY% export_nemo.py %E%\base24 %E%\big1\big1_int8.tnm 8 %E%\big1\last.pt
echo DONE
