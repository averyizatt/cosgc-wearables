@echo off
rem big2's start, data and length, but trained for long chunks and full context (docs/EXPERIMENTS.md, E12): with a push
rem button the device can wait for the end of speech. The engine sweep showed big2 at 24.24 % with chunk 32 and 19.22 %
rem with full context on a TORGO test subset, although it saw full context on only a quarter of its batches.
rem Differences from big2: --chunks 32,64 (was 16,24,32), --p_full 0.5 (was 0.25), --val_chunk 64 (was 32).
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big4.cmd > G:\cosgc\exp\big4_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big4 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big4.log 2>&1
if not exist %E%\big4\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big4\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big4_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big4\big4_int8.tnm 8 %E%\big4\last.pt
echo DONE
