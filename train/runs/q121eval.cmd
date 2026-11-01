@echo off
rem The new int8 grid (docs/EXPERIMENTS.md, E10) on the other test sets, in PyTorch with int8 fake quantization on export_nemo.py's
rem grid. Starts after big3, because a GPU job next to training overcommits the VRAM (E8b).
rem Run on the training PC: cmd /c G:\cosgc\repo3\train\runs\q121eval.cmd > G:\cosgc\exp\q121eval_all.log 2>&1
set TMP=G:\cosgc\tmp
set TEMP=G:\cosgc\tmp
cd /d G:\cosgc\repo3\train
:wait
findstr /c:"DONE" /c:"failed" G:\cosgc\exp\big3_all.log >nul 2>&1 || (timeout /t 120 /nobreak >nul & goto wait)
set D=G:\cosgc\data
set E=G:\cosgc\exp
set PY=G:\cosgc\venv\Scripts\python.exe -u
set TASR_FQ_QMAX=121
for %%M in (ls_testclean_train uaspeech_test_b2) do (
  %PY% eval_manifest.py --nemo %E%\base --ckpt %E%\atyp2\last.pt --init %E%\base\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\atyp2_q121_%%M.json --bits 8
  %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big2q\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big2q_q121_%%M.json --bits 8
  %PY% eval_manifest.py --nemo %E%\base24x --ckpt %E%\big2\last.pt --init %E%\base24x\init.pt --manifest %D%\%%M.jsonl --out %E%\eval\big2_q121_%%M.json --bits 8
)
echo DONE
