@echo off
rem simpre: the waveform simulation as a first stage, then big7's recipe on real data (docs/EXPERIMENTS.md, E20).
rem big10 (Finding 45) mixed 100 source hours of simulated speech into big7's data and left TORGO's dysarthric speakers unchanged; the simulated
rem speech stays separable from real speech (AUC 0.99), so mixed in, the model can treat it as its own kind of speech. Here it comes first:
rem   stage 1 (simpre1): base24x on simtr360 (E16's transparent set) + LibriSpeech 100, 4000 steps of 100 s, no TORGO;
rem   stage 2 (simpre):  big7's recipe exactly (data, steps, mel-level simulation, seed 0), started from stage 1 instead of base24x.
rem The comparison is big7, paired: stage 2 differs from big7 only in its starting point.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\simpre.cmd > G:\cosgc\exp\simpre_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set S=G:\cosgc\exp\sim
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
if not exist %E%\simpre1\last.pt start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%S%\simtr360.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\simpre1 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 100 --max_dur 32 --val_every 1000 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\simpre1.log 2>&1
if not exist %E%\simpre1\last.pt (echo stage 1 training failed & exit /b 1)
if not exist %E%\simpre\last.pt start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\simpre1\last.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\simpre ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 6000 --batch_sec 100 --val_every 750 --save_every 250 ^
  --p_dys 0.3 --dys_severity 0.2,1.0 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\simpre.log 2>&1
if not exist %E%\simpre\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\simpre\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\simpre_%%M.json
%PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\simpre1\last.pt --init %E%\base24x\init.pt --manifest %D%\torgo_test.jsonl --out %E%\eval\simpre1_torgo_test.json
%PY% export_nemo.py %E%\base24x %E%\simpre\simpre_int8.tnm 8 %E%\simpre\last.pt
echo DONE
