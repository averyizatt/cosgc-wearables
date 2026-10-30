@echo off
rem Short test runs for the second atypical-speech fine-tune (docs/TRAINING_PLAN.md, step 1).
rem Each run changes one suspected cause of atyp1's streaming regression; compare their chunk32 validation WER.
rem Run on the training PC: cmd /c G:\cosgc\repo\train\runs\abl.cmd > G:\cosgc\exp\abl.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo\train
set D=G:\cosgc\data
set C=G:\cosgc\venv\Scripts\python.exe -u train_nemo_stream.py --nemo G:\cosgc\exp\base --init G:\cosgc\exp\base\init.pt --keep_head --train %D%\ls100_train.jsonl,%D%\torgo_train_x3.jsonl --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --stream --steps 400 --warmup 100 --batch_sec 150 --val_every 200 --save_every 400 --val_chunk 32 --val_left 128 --aug_dir %D%\aug --workers 3
rem atyp1's settings, for reference at this length
%C% --out G:\cosgc\exp\abl_ref --kd 0.5 --lr 1e-4 >> G:\cosgc\exp\abl_ref.log 2>&1
%C% --out G:\cosgc\exp\abl_nokd --kd 0 --lr 1e-4 >> G:\cosgc\exp\abl_nokd.log 2>&1
%C% --out G:\cosgc\exp\abl_lowlr --kd 0.5 --lr 2e-5 >> G:\cosgc\exp\abl_lowlr.log 2>&1
%C% --out G:\cosgc\exp\abl_bigchunk --kd 0.5 --lr 1e-4 --chunks 16,24,32 --lefts 64,128 >> G:\cosgc\exp\abl_bigchunk.log 2>&1
%C% --out G:\cosgc\exp\abl_combo --kd 0 --lr 3e-5 --chunks 16,24,32 --lefts 64,128 >> G:\cosgc\exp\abl_combo.log 2>&1
echo DONE
