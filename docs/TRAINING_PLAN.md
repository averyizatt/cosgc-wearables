# Plan for the second atypical-speech fine-tune

Status: draft, 2026-10-30. Follows the first run (`atyp1`, results in [DATA.md](DATA.md#results-so-far)). Nothing in
this plan has been run yet; the training PC went offline before the step-1 test runs finished.

## Summary

atyp1 showed that TORGO data helps: every held-out TORGO speaker improved. But it made streaming on typical speech
50 % worse, so it cannot ship. The plan:

1. Five short test runs (about 45 min) to find which setting caused the streaming regression.
2. One 4000-step run (`atyp2`, about 2 h) with the fixes, plus speed augmentation, then automatic evaluation.
3. Ship it only if it meets the targets in step 3: typical speech no worse than today, TORGO at least as good as
   atyp1.

Everything runs on the training PC from two scripts: [`train/runs/abl.cmd`](../train/runs/abl.cmd) and
[`train/runs/atyp2.cmd`](../train/runs/atyp2.cmd).

## What went wrong with atyp1

| Test (WER %, full / streaming) | shipped | atyp1 |
|---|---|---|
| TORGO F03 (dysarthric) | 49.5 / 57.0 | 38.2 / 49.3 |
| TORGO M05 (dysarthric) | 80.0 / 89.7 | 64.3 / 74.8 |
| TORGO FC03, MC04 (controls) | 17.4, 16.5 / 23.6, 21.7 | 10.6, 12.0 / 17.8, 18.5 |
| TORGO single words | 54.6 / 59.8 | 32.4 / 39.5 |
| TORGO sentences | 21.4 / 28.5 | 19.0 / 27.6 |
| LibriSpeech test-clean | 4.0 / 6.2 | 4.6 / **9.7** |

Validation during training (mixed TORGO + LibriSpeech, chunk 16):

| step | full | chunk16 / left128 |
|---|---|---|
| 0 | 6.87 | 11.79 |
| 500 | 7.31 | 16.96 |
| 1000–4000 | 6.7–7.2 | 17.4–18.3 |

Two things stand out:

- **The streaming score jumped in the first 500 steps, then stayed flat.** The full-utterance score recovered, the
  streaming one never did. That looks like a setting that conflicts with how the shipped model learned to stream.
  It does not look like slow forgetting, which would get worse gradually; more steps would not have fixed it.
- **Most of the TORGO gain is on single words** (−20 points) and little is on sentences (−1 point). TORGO repeats the
  same word list across speakers, so part of the gain is the model learning that list. It is real for the
  held-out speakers, but it is not the same as general dysarthric-speech robustness.

## Suspected causes, most likely first

1. **Frame-level distillation (`--kd 0.5`).** On chunked batches, the full-context model's outputs teach the
   chunked outputs, frame by frame. A streaming CTC model naturally emits each token a few frames later than a
   full-context model, because it has to wait for the audio. Matching frame by frame pushes it to emit before it
   has heard enough, which costs accuracy in exactly the mode the device uses. A jump right at the start fits this,
   because distillation is at full strength from step 1.
2. **Learning rate 1e-4.** That is high for a short fine-tune of a model that is already converged. A large early
   jump is the usual sign. The original training ended at a much lower rate.
3. **Chunk sizes down to 4 frames** (`--chunks 4,8,12,16,24,32`, unlimited left context some of the time). The device
   always runs chunk 32 with left 128. Training on tiny chunks spends capacity on a setting we never use.
4. **TORGO weighting.** `torgo_train_x3` repeats TORGO three times, so roughly a third of every batch is TORGO,
   mostly single words from 11 speakers. This mainly explains the word/sentence gap, not the streaming jump.

There is also a measurement mismatch to fix: training validated at chunk 16, but the device and `eval_manifest.py`
use chunk 32. From now on, validation runs at chunk 32 / left 128 (`--val_chunk 32`), so the number watched during
training is the same kind of number as the final test.

## Step 1: short test runs (about 45 min)

Five 400-step runs from the shipped model, validated every 200 steps at chunk 32 / left 128
([`train/runs/abl.cmd`](../train/runs/abl.cmd)):

| run | kd | lr | chunks / lefts | tests |
|---|---|---|---|---|
| `abl_ref` | 0.5 | 1e-4 | atyp1's | the baseline: does the regression show up at chunk 32 too? |
| `abl_nokd` | 0 | 1e-4 | atyp1's | cause 1 |
| `abl_lowlr` | 0.5 | 2e-5 | atyp1's | cause 2 |
| `abl_bigchunk` | 0.5 | 1e-4 | 16,24,32 / 64,128 | cause 3 |
| `abl_combo` | 0 | 3e-5 | 16,24,32 / 64,128 | all three fixes together |

The step-0 validation score (the shipped model) is the reference.

**How to read the results:**

- A run whose streaming validation stays within about 1 point of step 0 at step 400 has removed the cause.
- If only `abl_nokd` holds, atyp2 uses `--kd 0`; if `abl_lowlr` also holds, a low LR is a free extra safety margin.
- If no single change holds but `abl_combo` does, the causes add up, and atyp2 uses all three.
- If even `abl_combo` regresses, the problem is something else. The next suspects are the EMA (validation scores the
  averaged weights) and the TORGO share. Then run `abl_combo` again with TORGO 1× and with plain LibriSpeech only.
  If LibriSpeech alone regresses, the training code differs from how the shipped model was trained, not the data.
- 400 steps is short. These runs only show which settings avoid the early jump; they are not expected to show
  TORGO gains.

## Step 2: the long run, `atyp2` (about 2 h plus 30 min of tests)

Settings in [`train/runs/atyp2.cmd`](../train/runs/atyp2.cmd); `KD` and `LR` at the top get set from step 1.

| setting | atyp1 | atyp2 | why |
|---|---|---|---|
| learning rate | 1e-4, warmup 200 | 3e-5, warmup 300 | gentler on a converged model |
| distillation | 0.5 | 0 (or 0.1 if step 1 says it is harmless) | cause 1 |
| chunk sizes | 4–32 | 16, 24, 32 | close to the device's 32 |
| left context | 0 (unlimited), 64, 128, 256 | 64, 128 | the device uses 128 |
| validation | chunk 16 | chunk 32 / left 128 | same as the device and the final test |
| TORGO share | 3× | 2× | less word-list memorizing, more typical speech per batch |
| speed augmentation | none | 30 % of clips at 0.9× or 1.1× | slow speech is a weak spot (stress test), and dysarthric speech is often slow |
| steps, batch | 4000 × 150 s | same | |
| noise and reverb | MUSAN, p 0.6 / 0.3 | same | |

Speed augmentation is new (`--p_speed` in `train_nemo_stream.py`, done in `augment.py`). It resamples the clip, so
speed and pitch change together, as in the standard Kaldi recipe.

**Watching the run.** Validation runs every 500 steps. If the streaming validation score rises more than 1 point
above step 0, the run is stopped and settings changed. A checkpoint is saved every 250 steps, so nothing is lost by
stopping.

**After training,** the script evaluates `last.pt` on TORGO's held-out speakers and on LibriSpeech test-clean, with
results in `G:\cosgc\exp\eval\atyp2_*.json`.

## Step 3: what counts as better

The new model replaces the shipped one only if all of these hold (`train/eval_manifest.py`, greedy decoding):

| test (WER %, streaming) | shipped | atyp1 | target for atyp2 |
|---|---|---|---|
| LibriSpeech test-clean | 6.2 | 9.7 | **≤ 6.5** |
| LibriSpeech test-clean, full utterance | 4.0 | 4.6 | ≤ 4.3 |
| TORGO F03 | 57.0 | 49.3 | ≤ 50 |
| TORGO M05 | 89.7 | 74.8 | ≤ 76 |
| TORGO controls FC03 / MC04 | 23.6 / 21.7 | 17.8 / 18.5 | ≤ 19 each |
| TORGO sentences | 28.5 | 27.6 | < 27 |

If atyp2 meets the LibriSpeech target but gives back some TORGO gain, that is still progress. The shipped model must
not get worse for typical speakers, so a smaller TORGO gain is the acceptable trade.

Then, before it goes on a device:

1. Export to int8 `.tnm` with `export_nemo.py`. Rerun the TORGO and LibriSpeech evaluations on the exported file,
   to check that quantization does not undo the gain.
2. Run the host stress test (`esp32/host/stress_run.py`) with the shipped model and atyp2 side by side, especially
   the quiet and slow conditions.
3. Add a row to the model table in [DATA.md](DATA.md), marked **academic / non-profit only** because it trained on
   TORGO.

## Timeline once the PC is back online

| when | what |
|---|---|
| 0:00 | copy the updated `train/` to `G:\cosgc\repo`, start `abl.cmd` |
| 0:45 | read the test runs, set `KD` / `LR` in `atyp2.cmd`, start it |
| 2:45 | training ends, evaluations run on their own |
| 3:15 | results in; if they pass, export, stress test, docs, push |

## After atyp2

In rough order of expected value:

- **Per-speaker adaptation.** A short fine-tune (a few hundred steps) on 5–10 minutes of one user's speech. In
  published work on dysarthric ASR, this gives the biggest gains, often far larger than any general model. TORGO
  can simulate it: for each held-out dysarthric speaker (F03, M05), train on half of their sessions and test on the
  other half. This tells us how much recorded speech a new user would need to provide.
- **A sentence-level test set.** Report sentences and words separately from now on, and judge models mainly on
  sentences, since the device is used for phrases rather than single words.
- **Language model for commands.** The device's GRU LM was trained on typical text. A small LM, or a bias toward a
  fixed list of commands and words, would help most on short utterances, where the acoustic model has least context.
- **More data.** Apply to the Speech Accessibility Project again later. Request UASpeech (form; mostly isolated
  words, cerebral palsy).

## Risks

- **TORGO is small.** 11 training speakers, 4 test speakers. Differences of a couple of points on one speaker can
  be noise. Look at the direction across all four speakers, not one number.
- **No validation-only speakers.** The TORGO validation set during training is drawn from the held-out test
  speakers. Choosing settings by watching it slightly overfits the test set. For now this is acceptable; with more
  data, hold out a separate validation speaker.
- **The PC dropping offline** loses nothing (checkpoints every 250 steps) but restarts need someone at the network.

## What the training PC needs

Sleep set to "Never" while plugged in (Settings → System → Power), connected to the home network, and no VPN that
blocks local network access. Results are written to `G:\cosgc\exp\eval\` and logs to `G:\cosgc\exp\*.log`.
