@echo off
rem big6 with a still larger share of dysarthric audio (docs/EXPERIMENTS.md, E13): big4 6.5 %, big6 26 %, big8 38 %. big6 gained 5 points for dysarthric TORGO speakers in the engine
rem and lost about 3 on UASpeech. Differences from big6: torgo_dys_x8.jsonl listed twice in --train. Everything else is unchanged.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big8.cmd > G:\cosgc\exp\big8_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big8 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big8.log 2>&1
if not exist %E%\big8\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big8\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big8_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big8\big8_int8.tnm 8 %E%\big8\last.pt
echo DONE
