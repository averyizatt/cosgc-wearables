@echo off
rem big4 with three times the steps and twice the learning rate (docs/EXPERIMENTS.md, E13). big4's full-context validation
rem was still falling when its 4000 steps ended, and the 8 layers grown onto the shipped 16 have had little training.
rem Differences from big4: --steps 12000 (was 4000), --lr 1e-4 (was 5e-5), --warmup 500 (was 300), --val_every 1000 (was 500).
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big5.cmd > G:\cosgc\exp\big5_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big5 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 1e-4 --warmup 500 --steps 12000 --batch_sec 150 --val_every 1000 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big5.log 2>&1
if not exist %E%\big5\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big5\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big5_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big5\big5_int8.tnm 8 %E%\big5\last.pt
echo DONE
