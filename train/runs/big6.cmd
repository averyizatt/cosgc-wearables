@echo off
rem big4 with four times the share of dysarthric audio (docs/EXPERIMENTS.md, E13). In big2 to big5 dysarthric speech is 6.5 % of the training
rem audio (3.9 h x 2 of 120 h), so 4000 steps show each dysarthric clip about 3 times. Differences from big4: torgo_dys_x8.jsonl added to --train
rem (the dysarthric TORGO training speakers repeated 8 more times, about 26 % of the audio). Everything else is unchanged, so the difference measures the mix.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\big6.cmd > G:\cosgc\exp\big6_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
if not exist %P%\torgo_dys_x8.jsonl %PY% -c "import json;L=[l for l in open(r'%P%\torgo_train.jsonl',encoding='utf8') if json.loads(l)['group']=='dysarthric'];open(r'%P%\torgo_dys_x8.jsonl','w',encoding='utf8',newline='\n').writelines(L*8);print(len(L),'dysarthric clips x 8')"
start "" /belownormal /b /wait %PY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big6 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 4000 --batch_sec 150 --val_every 500 --save_every 250 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big6.log 2>&1
if not exist %E%\big6\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big6\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big6_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big6\big6_int8.tnm 8 %E%\big6\last.pt
echo DONE
