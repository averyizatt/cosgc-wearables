@echo off
rem Second atypical-speech fine-tune (docs/TRAINING_PLAN.md, step 2), then its evaluations.
rem Starting settings; KD and LR get set from the step-1 test runs before launching.
rem Run on the training PC: cmd /c G:\cosgc\repo\train\runs\atyp2.cmd > G:\cosgc\exp\atyp2_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set PY=G:\cosgc\venv\Scripts\python.exe -u
set KD=0
set LR=5e-5
rem TORGO listed twice = repeated 2x (atyp1 used 3x)
%PY% train_nemo_stream.py --nemo %E%\base --init %E%\base\init.pt --keep_head ^
  --train %D%\ls100_train.jsonl,%D%\torgo_train.jsonl,%D%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\atyp2 ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd %KD% --lr %LR% --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 3 > %E%\atyp2.log 2>&1
if not exist %E%\atyp2\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base --ckpt %E%\atyp2\last.pt --init %E%\base\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\atyp2_%%M.json
echo DONE
