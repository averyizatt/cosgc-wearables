@echo off
rem A Parakeet teacher for the on-device model (docs/EXPERIMENTS.md, E21).
rem   1. teacher110: nvidia/parakeet-tdt_ctc-110m's encoder (CC-BY-4.0) with our 1024-piece tokenizer and a new CTC head, fine-tuned on
rem      big6's real data (LibriSpeech 100, TORGO train x2, its dysarthric clips x8; no TORGO test speaker, no UASpeech), 6000 steps of 100 s.
rem   2. scored like every model here: TORGO test, LibriSpeech test-clean, UASpeech (float, greedy, full context).
rem   3. the frame offset between teacher and student (teacher_ft.py offset, against big7).
rem   4. big11: big7's recipe exactly, plus the teacher's posteriors on full-context batches (train_nemo_stream.py --teacher, weight 0.5,
rem      temperature 2). Clips the mel-level simulation changed are left out of the teacher term (the teacher hears the waveform).
rem The comparison is big7, paired. Student training runs in the NeMo environment because the teacher needs NeMo.
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\teacher.cmd > G:\cosgc\exp\teacher_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
set HF_HOME=G:\cosgc\hf
cd /d G:\cosgc\repo3\train
set D=G:\cosgc\data
set E=G:\cosgc\exp
set P=%D%\packed
set PY=G:\cosgc\venv\Scripts\python.exe -u
set NPY=G:\cosgc\nemovenv\Scripts\python.exe -u
set TOK=%E%\base24x\tokenizer.model
set T=%E%\teacher110
if not exist %T%\teacher.pt start "" /belownormal /b /wait %NPY% teacher_ft.py train ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl ^
  --tokenizer %TOK% --out %T% --steps 6000 --batch_sec 100 --lr 1e-4 --warmup 500 ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\teacher110.log 2>&1
if not exist %T%\teacher.pt (echo teacher training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do if not exist %E%\eval\teacher110_%%M.json %NPY% teacher_ft.py eval --ckpt %T%\teacher.pt --tokenizer %TOK% --manifest %D%\%%M.jsonl --out %E%\eval\teacher110_%%M.json
if not exist %E%\teacher110_offset.txt %NPY% teacher_ft.py offset --ckpt %T%\teacher.pt --nemo %E%\base24x --student %E%\big7\last.pt --manifest %P%\torgo_train.jsonl --limit 50 > %E%\teacher110_offset.txt
type %E%\teacher110_offset.txt
set SHIFT=
for /f "tokens=3" %%s in ('findstr /c:"best --teacher_shift" %E%\teacher110_offset.txt') do set SHIFT=%%s
if not defined SHIFT (echo offset failed & exit /b 1)
echo teacher shift %SHIFT%
if not exist %E%\big11\last.pt start "" /belownormal /b /wait %NPY% train_nemo_stream.py --nemo %E%\base24x --init %E%\base24x\init.pt --keep_head ^
  --train %P%\ls100_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_train.jsonl,%P%\torgo_dys_x8.jsonl ^
  --val %D%\torgo_test.jsonl,%D%\ls_testclean_train.jsonl --out %E%\big11 ^
  --stream --chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64 --val_left 128 ^
  --kd 0 --lr 5e-5 --warmup 300 --steps 6000 --batch_sec 100 --val_every 750 --save_every 250 ^
  --p_dys 0.3 --dys_severity 0.2,1.0 ^
  --teacher %T%\teacher.pt --teacher_kd 0.5 --teacher_T 2 --teacher_shift %SHIFT% ^
  --aug_dir %D%\aug --p_speed 0.3 --workers 2 >> %E%\big11.log 2>&1
if not exist %E%\big11\last.pt (echo training failed & exit /b 1)
for %%M in (torgo_test ls_testclean_train uaspeech_test_b2) do %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big11\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big11_%%M.json
%PY% export_nemo.py %E%\base24x %E%\big11\big11_int8.tnm 8 %E%\big11\last.pt
echo DONE
