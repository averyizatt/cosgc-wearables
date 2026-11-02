@echo off
rem big6 with simulated dysarthria on 30 % of the LibriSpeech clips (docs/EXPERIMENTS.md, E13). E4 found no gain from this at 16 layers on a 6.5 % dysarthric mix;
rem this asks again at 24 layers on big6's 26 % mix, and whether LibriSpeech's wide vocabulary in dysarthric form offsets big6's loss on UASpeech.
rem Differences from big6: --p_dys 0.3 --dys_severity 0.2,1.0 (the same settings as atyp3).
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big7.cmd > G:\cosgc\exp\big7_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big7 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --p_dys 0.3 --dys_severity 0.2,1.0 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big7.log 2>&1
if not exist %E%\big7\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big7\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big7_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big7\big7_int8.tnm 8 %E%\big7\last.pt
echo DONE
