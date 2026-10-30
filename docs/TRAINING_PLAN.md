# Plan for the second atypical-speech fine-tune

Status: draft, 2026-10-30. Follows the first run (`atyp1`, results in [DATA.md](DATA.md#results-so-far)).

## What went wrong with atyp1

atyp1 helped every held-out TORGO speaker (streaming WER on F03 57.0 → 49.3, M05 89.7 → 74.8), but streaming on typical
speech got worse: LibriSpeech test-clean 6.2 → 9.7. Full-utterance WER barely moved (4.0 → 4.6).

The validation log shows when that happened:

| step | full | chunk16 / left128 |
|---|---|---|
| 0 | 6.87 | 11.79 |
| 500 | 7.31 | 16.96 |
| 1000–4000 | 6.7–7.2 | 17.4–18.3 |

The streaming score jumped in the first 500 steps and then stayed flat, while the full-utterance score recovered. That
looks like a setting that conflicts with how the shipped model learned to stream. It does not look like slow
forgetting, and more steps would not fix it.

## Suspected causes, most likely first

1. **Frame-level distillation (`--kd 0.5`).** On chunked batches, the full-context EMA model's posteriors teach the
   chunked posteriors frame by frame. A streaming CTC model emits its tokens a few frames later than a full-context
   one, so matching frame by frame pushes the streaming model to emit before it has heard enough. The shipped model
   may never have been trained this way.
2. **Learning rate 1e-4.** That is high for a short fine-tune of a model that is already converged. A big early jump
   is the typical sign of this.
3. **Chunk sizes down to 4 frames** (`--chunks 4,8,12,16,24,32`). The device always runs chunk 32 with left 128.
   Training on tiny chunks spends capacity on a setting we never use.
4. **TORGO weighting.** `torgo_train_x3` repeats TORGO three times, so about a third of every batch is mostly single
   words from 11 speakers. That explains why words improved far more than sentences (54.6 → 32.4 vs 21.4 → 19.0).

## Step 1: short test runs (about 30 min)

Three 400-step runs from the shipped model, each changing one thing, compared on the chunk16 validation score
(starting point 11.79; atyp1 reached 16.96 at step 500):

| run | change |
|---|---|
| `abl_nokd` | `--kd 0` |
| `abl_lowlr` | `--lr 2e-5` |
| `abl_bigchunk` | `--chunks 16,24,32 --lefts 128` |

The script is `G:\cosgc\exp\abl.cmd` on the training PC. It was started on 2026-10-30, but the PC dropped off the
network before the runs finished, so they have to be re-run.

## Step 2: the long run (`atyp2`, about 2 h)

These settings combine whatever the test runs show to help. The expected starting point:

- `--lr 3e-5 --warmup 300`
- `--kd 0` if `abl_nokd` holds streaming, otherwise a smaller weight (0.1)
- `--chunks 16,24,32 --lefts 64,128 --p_full 0.25`: train near what the device runs
- `--val_chunk 32 --val_left 128` so the validation score matches the device
- TORGO repeated 2× instead of 3×, so typical speech dominates more batches
- same augmentation (MUSAN noise and reverb), 4000 steps, batches of 150 s

**Guard:** check the run every 500 steps. If LibriSpeech streaming (validation, chunk 32) rises more than 0.5 points
above the shipped model, stop the run and change the settings.

## Step 3: what counts as better

The new model replaces the shipped one only if all of these hold, measured with `train/eval_manifest.py`:

| test | shipped | atyp1 | target |
|---|---|---|---|
| LibriSpeech test-clean, streaming | 6.2 | 9.7 | **≤ 6.5** |
| LibriSpeech test-clean, full | 4.0 | 4.6 | ≤ 4.3 |
| TORGO dysarthric (F03, M05), streaming | 57.0 / 89.7 | 49.3 / 74.8 | at least as good as atyp1 |
| TORGO controls (FC03, MC04), streaming | 23.6 / 21.7 | 17.8 / 18.5 | at least as good as atyp1 |
| TORGO sentences, streaming | 28.5 | 27.6 | lower than atyp1 |

After that, the model also has to pass:

- export to int8 `.tnm` (`export_nemo.py`) and rerun the TORGO evaluation and the host stress test (`stress_run.py`) on
  the exported file, so quantization does not undo the gain;
- a new row in the model table in DATA.md, marked **academic / non-profit only** because it trained on TORGO.

## Later improvements

- **Per-speaker adaptation:** a short fine-tune on 5–10 minutes of one user's speech. This is where dysarthric
  speakers gain the most in published work. TORGO lets us simulate it: train on half of a held-out speaker's
  sessions, test on the rest.
- **Speed and tempo augmentation** (0.9× / 1.1×): dysarthric speech is often slow, and the stress test found slow
  speech is a weak spot.
- **Language model:** the device's GRU LM was trained on typical text. A small bias toward short commands and a
  word list would help with single words.
- **More data:** apply to the Speech Accessibility Project again later, and request UASpeech (a form, mostly
  isolated words).

## What the training PC needs

Sleep set to "Never" while plugged in, connected to the home network, and no VPN that blocks local network access.
Each step above runs on the PC by itself; the results are in `G:\cosgc\exp\eval\` and the logs in `G:\cosgc\exp\*.log`.
