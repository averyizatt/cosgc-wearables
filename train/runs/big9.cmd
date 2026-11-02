@echo off
rem big6 plus the waveform simulator's pilot set (docs/EXPERIMENTS.md, E15): 100 h of LibriSpeech 100 through dysim.py (14 conditions, severity 0.2-1.4, one
rem condition per simulated speaker), made offline by: dysim.py gen --manifest ls100_train.jsonl --out sim100 --hours 100 --seed 21 --severity 0.2,1.4
rem Differences from big6: sim100.jsonl added to --train, and torgo_dys_x8.jsonl listed twice so real dysarthric audio stays near big6's share (about 24 % against 26 %).
rem big7's steps and batch (the same audio as big6): 6000 steps of 100 s. Slowing makes most simulated clips longer than 20 s (65 % of the first
rem 7 h of output), so --max_dur is 32 s, not 20; about 15 % of the simulated clips, the slowest, are still dropped.
rem big7 (the older mel-level simulation, online) and big6 (none) are the comparisons.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big9.cmd > G:\cosgc\exp\big9_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%E%\sim\sim100.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big9 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 6000 --batch_sec 100 --val_every 750 --max_dur 32 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big9.log 2>&1
if not exist %E%\big9\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big9\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big9_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big9\big9_int8.tnm 8 %E%\big9\last.pt
echo DONE
