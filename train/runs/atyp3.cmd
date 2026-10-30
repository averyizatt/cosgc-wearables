@echo off
rem Third fine-tune: atyp2 + simulated dysarthria (simdys.py version 2, severity 0.2-1.0) on 30 %% of typical clips (docs/EXPERIMENTS.md, E4).
rem Everything else is identical to atyp2, so the difference measures the simulation alone.
rem Waits for atyp2 (and its evaluations) to finish first. Runs code from G:\cosgc\repo2 so atyp2 is not disturbed.
rem Run on the training PC: cmd /c G:\cosgc\repo2\train\runs\atyp3.cmd > G:\cosgc\exp\atyp3_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo2\train
:wait
findstr /c:"DONE" G:\cosgc\exp\atyp2_all.log >nul 2>&1 || (timeout /t 120 /nobreak >nul & goto wait)
set D=G:\cosgc\data
set E=G:\cosgc\exp
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base --init %E%\base\init.pt --keep_head ^
  --train %D%\ls100_train.jsonl,%D%\torgo_train.jsonl,%D%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\atyp3 ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --p_dys 0.3 --dys_severity 0.2,1.0 --workers 2 >> %E%\atyp3.log 2>&1
if not exist %E%\atyp3\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base --ckpt %E%\atyp3\last.pt --init %E%\base\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\atyp3_%%M.json
echo DONE
