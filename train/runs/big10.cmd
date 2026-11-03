@echo off
rem big10: big7's data plus ~120 h of the waveform simulator rendered WITHOUT the vocoder (docs/EXPERIMENTS.md, E16, Findings 40-41).
rem The simulated set: train-clean-360 (new voices and texts; big7 trains on train-clean-100), sources up to 16 s so that slowed clips stay
rem under --max_dur, severity 0.1-1.4 (E16 recalibration: TORGO training speakers sit at 0.3 or below to 1.47), phone alignments (sound errors, spliced from each clip's own phones), measured TORGO error tables, all 14 conditions.
rem Steps scaled so the real data is seen as often as in big7 (6000 steps x 100 s over 182 h real): 6000 x (182 + 120) / 182 = 9950.
rem big9 (WORLD-rendered sim100 at big7's 6000 steps) halved real exposure and was worse on TORGO; this run changes the renderer and keeps exposure.
rem big7 (deployed) and big9 are the comparisons. Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big10.cmd > G:\cosgc\exp\big10_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set S=G:\cosgc\exp\sim
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
if exist %E%\big10\last.pt goto evals
if exist %S%\simtr360.done goto train
set CUDA_VISIBLE_DEVICES=-1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
start "" /belownormal /b /wait %PY% dysim.py gen --manifest %D%\ls360_train.jsonl --out %S%\simtr360 --hours 100 --max_src 16 --workers 8 --seed 31 ^
  --severity 0.1,1.4 --align_dir %D%\ls_align --real_errors %S%\real_errors.json --render transparent >> %S%\simtr360.log 2>&1
findstr /c:finished %S%\simtr360.log >nul || (echo generation failed & exit /b 1)
echo done > %S%\simtr360.done
set CUDA_VISIBLE_DEVICES=
set OPENBLAS_NUM_THREADS=
set OMP_NUM_THREADS=
:train
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%S%\simtr360.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big10 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 9950 --batch_sec 100 --val_every 750 --max_dur 32 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big10.log 2>&1
if not exist %E%\big10\last.pt (echo training failed & exit /b 1)
:evals
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big10\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big10_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big10\big10_int8.tnm 8 %E%\big10\last.pt
echo DONE
