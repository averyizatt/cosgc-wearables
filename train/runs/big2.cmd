@echo off
rem Deeper model for the ESP32-P4, second try (docs/EXPERIMENTS.md, E6): grow_nemo.py with the exact start (the grown
rem model computes what the shipped one does), 16 -> 24 layers, 13 -> 19 M parameters, then the atyp2 recipe unchanged.
rem big1 was the same with the first, broken start (98 % WER before training).
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big2.cmd > G:\cosgc\exp\big2_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
if not exist %E%\base24x\init.pt %PY% grow_nemo.py --nemo %E%\base --init %E%\base\init.pt --layers 24 --out %E%\base24x || (echo grow failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\base24x\init.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\base24x_untrained_%%M.json
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big2 ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big2.log 2>&1
if not exist %E%\big2\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big2\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big2_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big2\big2_int8.tnm 8 %E%\big2\last.pt
echo DONE
