@echo off
rem atyp2 + 1000 steps of quantization-aware training (--bits 8: int8 weights and activations simulated as in the
rem engine), docs/EXPERIMENTS.md E8. The float fine-tunes lose about 3.5 points of WER when exported to the int8 engine
rem (atyp2: 30.1 % PyTorch vs 33.7 % engine on TORGO test, streaming greedy); the shipped model loses nothing.
rem Waits for atyp4 to finish. Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\atyp2q.cmd > G:\cosgc\exp\atyp2q_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
:wait
findstr /c:"DONE" G:\cosgc\exp\atyp4_all.log >nul 2>&1 || (timeout /t 120 /nobreak >nul & goto wait)
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base --init %E%\atyp2\last.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\atyp2q ^
  --stream --chunks 16,24,32 --lefts 64,128 --p_full 0.25 --val_chunk 32 --val_left 128 ^
  --kd 0 --lr 2e-5 --warmup 100 --steps 1000 --batch_sec 150 --val_every 250 --save_every 250 --bits 8 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\atyp2q.log 2>&1
if not exist %E%\atyp2q\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train) do %PY% eval_manifest.py --nemo %E%\base --ckpt %E%\atyp2q\last.pt --init %E%\base\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\atyp2q_%%M.json --bits 8
%PY% export_nemo.py %E%\base %E%\atyp2q\atyp2q_int8.tnm 8 %E%\atyp2q\last.pt
echo DONE
