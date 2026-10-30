# Experiment log: adapting an on-device streaming recognizer to dysarthric speech

This is the research record for the paper. Every training run, test run and evaluation goes here, with exact settings,
numbers and what we concluded, including runs that failed. It is written so a section can be lifted into the
paper's Method and Results with little rewriting. Planning is in [TRAINING_PLAN.md](TRAINING_PLAN.md), the data in
[DATA.md](DATA.md), the device and engine in [MEMO.md](MEMO.md) and [P4.md](P4.md).

**Rules for this log:** record every run, including failed and abandoned ones. Give numbers exactly as the scripts
printed them. Every run names its script and the file its results are in. Use the date the run started. Never edit an
old entry's results; add a correction below it.

## 1. Question

Can a 13 M-parameter streaming speech recognizer that runs on a microcontroller (ESP32-S3 / ESP32-P4) be fine-tuned to
recognize dysarthric speech better, without getting worse on typical speech, using only freely available data?

## 2. Setup common to all experiments

### 2.1 Model

- **Architecture:** NVIDIA NeMo Conformer-CTC small, as ported in the Oído engine: 16 Conformer layers, model width
  176, 4 attention heads, feed-forward 704, depthwise convolution kernel 31, 4× subsampling to 40 ms frames,
  SentencePiece vocabulary of 1024 pieces + CTC blank. About 13 M parameters.
- **Starting point ("shipped model"):** `models/oido_stream.tnm`, the streaming model distributed with Oído
  (trained by its authors on LibriSpeech, Common Voice, VoxPopuli, AMI, VCTK, People's Speech, OpenSLR 70/83, with
  MUSAN noise; see NOTICE). We converted it back to float PyTorch weights with `train/tnm_to_ckpt.py`; re-exporting
  the converted weights reproduces the original file byte for byte (checked with `--verify`), so all fine-tunes
  start from exactly the deployed model.
- **Streaming mode:** chunked self-attention over chunks of C encoder frames with L frames of left context, and a
  depthwise convolution that sees nothing past the end of its own chunk. The device runs C = 32 (1.28 s), L = 128
  (5.1 s). Feature normalization is fixed (global per-mel mean and standard deviation stored in the model), so it
  does not depend on future audio.

### 2.2 Training procedure (`train/train_nemo_stream.py`)

| | |
|---|---|
| Loss | CTC (sum over the batch / number of target tokens); optionally + λ · frame-level KL from a teacher ("in-place distillation", `--kd λ`, see 2.4) |
| Optimizer | AdamW, β = (0.9, 0.98), weight decay 1e-3, gradient clipping at norm 1.0 |
| Schedule | linear warmup, then cosine decay to 5 % of the peak rate |
| Batches | bucketed by length, about 150 s of audio per batch (`--batch_sec 150`), clips up to 20 s |
| Precision | bfloat16 autocast on GPU |
| Weight averaging | EMA of weights, decay min(0.999, (1+t)/(10+t)); all validation and all evaluated checkpoints use the EMA weights |
| Streaming training | per batch: full context with probability `p_full` = 0.25, else a chunk size from `--chunks` and a left context from `--lefts` (0 = unlimited) |
| Feature augmentation | random gain ±12 dB; SpecAugment: 2 frequency masks up to 16 of 80 mels, one time mask up to 25 frames per 200 frames |
| Waveform augmentation (`train/augment.py`) | reverberation with probability 0.3 (400 simulated rooms, pyroomacoustics); additive noise with probability 0.6 at SNR uniform 0–20 dB, drawn 45 % MUSAN noise, 20 % music, 35 % babble of 2–4 MUSAN speakers; from atyp2 on, speed perturbation 0.9× / 1.1× with probability 0.3 (resampling, so pitch changes too) |
| Output head | kept from the shipped model (`--keep_head`), same tokenizer |

### 2.3 Data and splits

Details, sources, licenses and archive hashes: [DATA.md](DATA.md).

| Set | Use | Size |
|---|---|---|
| LibriSpeech train-clean-100 | training (typical speech) | 28,539 clips, 100.6 h |
| TORGO, 11 speakers (F01, F04, M01–M04 dysarthric; FC01, FC02, MC01–MC03 control) | training | 11,334 clips, 9.80 h |
| TORGO, 4 held-out speakers: F03, M05 (dysarthric), FC03, MC04 (control) | test, never trained on | 5,214 clips, 3.87 h |
| LibriSpeech test-clean | test (typical speech) | 2,620 clips, 5.4 h |
| Validation during training | 150 clips from TORGO test + 150 from LibriSpeech test-clean, fixed | 300 clips |

**TORGO preparation (`train/make_manifest_torgo.py`):** each prompt file gives the transcript for both microphones
(head-mounted and array), so each recording is one clip. Transcripts lowercased, punctuation removed. Dropped: 210
instructions or empty prompts (e.g. "[say Ah-P-Eee repeatedly]"), 118 picture descriptions (no transcript), 58
prompts with nothing left after cleaning, 34 clips shorter than 0.3 s; plus 1,223 and 411 prompts with no head or
array recording. All audio is 16 kHz. Prompts with one word are labeled "word", longer ones "sentence".

**Split by speaker.** Each held-out speaker is entirely in the test set, so no voice is in both training and test.
The held-out speakers were chosen to cover one female and one male in each group, and include both a moderate (F03)
and a severe (M05) dysarthric speaker.

**Known weakness of the split:** the validation clips watched during training come from the test speakers. Choosing
settings by watching them biases the test results slightly. See section 6.

### 2.4 Evaluation (`train/eval_manifest.py`)

- Word error rate (jiwer), greedy CTC decoding, no language model. The device adds a small GRU language model, so its
  WER is a little lower; all comparisons here are without LM so they measure only the acoustic model.
- Two modes for every clip: **full** (whole utterance at once) and **stream** (chunk 32, left 128, as on the device).
  Stream is the number that matters for the product; full is reported to separate streaming effects from others.
- Breakdowns by speaker, group (dysarthric / control), microphone and prompt kind (word / sentence).
- Every evaluation writes a summary JSON and per-clip hypotheses (`*_utts.jsonl`) for error analysis.
- Floating-point weights. Results after int8 export are reported separately once a model is exported.

### 2.5 Hardware and software

One NVIDIA RTX 3060 (12 GB), Ryzen 5 1600, 16 GB RAM, Windows 11; Python 3.11.9, PyTorch 2.14.1+cu126. About 1.1 to
1.8 s per 150 s batch, depending on settings and data loading (the CPU is the bottleneck: augmentation runs in 3
loader processes; 5 ran out of memory).

## 3. Baseline: the shipped model (2026-10-29)

Results: `G:\cosgc\exp\eval\base_torgo_test.json`, `base_ls_testclean.json`.

| Test | full | stream |
|---|---|---|
| TORGO all held-out | 30.76 | 37.31 |
| control speakers | 16.98 | 22.75 |
| dysarthric speakers | 60.06 | 68.28 |
| F03 (dysarthric, 1085 clips) | 49.5 | 57.0 |
| M05 (dysarthric, 587 clips) | 80.0 | 89.7 |
| FC03 (control, 1922 clips) | 17.4 | 23.6 |
| MC04 (control, 1620 clips) | 16.4 | 21.7 |
| single words | 54.64 | 59.80 |
| sentences | 21.42 | 28.52 |
| head microphone | 33.44 | 41.45 |
| array microphone | 28.09 | 33.20 |
| LibriSpeech test-clean | 4.01 | 6.21 |

**Observations:**

- Dysarthric speech has about 3.5× the error rate of control speech from the same corpus, and the gap grows with
  severity (F03 vs M05).
- Even control speakers in TORGO are far worse than LibriSpeech (22 % vs 6 %). TORGO is mostly isolated words, which
  have no context for the model to use: single words are worse than sentences for every speaker.
- Streaming costs 5 to 10 points on TORGO, versus 2 on LibriSpeech.

## 4. Experiments

### E1. atyp1: first combined fine-tune (2026-10-29)

**Goal:** a first fine-tune on typical + dysarthric speech.

**Settings:** LibriSpeech train-clean-100 + TORGO training speakers repeated 3× (130 h total, 62,522 clips), 4000
steps, peak LR 1e-4, warmup 200, distillation λ = 0.5, chunks {4, 8, 12, 16, 24, 32}, left contexts {unlimited,
unlimited, 64, 128, 256}, validation at chunk 16 / left 128. Augmentation as in 2.2 without speed perturbation.

The first launch ran at about 4 s per step. It was restarted with 5 loader processes, which crashed (out of memory)
without an error in the log, and then with 3 loader processes at about 1.1 s per step. Only the last launch produced
results. Run directory `G:\cosgc\exp\atyp1`, log `atyp1.log`.

**Validation during training (mixed TORGO + LibriSpeech):**

| step | full | chunk16 / left128 |
|---|---|---|
| 0 | 6.87 | 11.79 |
| 500 | 7.31 | 16.96 |
| 1000 | 7.16 | 18.33 |
| 2000 | 6.99 | 17.85 |
| 3000 | 6.87 | 17.43 |
| 4000 | 6.69 | 17.70 |

**Test results** (`atyp1_torgo_test.json`, `atyp1_ls_testclean.json`):

| Test | baseline full / stream | atyp1 full / stream |
|---|---|---|
| F03 | 49.5 / 57.0 | 38.2 / 49.3 |
| M05 | 80.0 / 89.7 | 64.3 / 74.8 |
| FC03 | 17.4 / 23.6 | 10.6 / 17.8 |
| MC04 | 16.5 / 21.7 | 12.0 / 18.5 |
| TORGO all | 30.8 / 37.3 | 22.7 / 30.9 |
| single words | 54.6 / 59.8 | 32.4 / 39.5 |
| sentences | 21.4 / 28.5 | 19.0 / 27.6 |
| LibriSpeech test-clean | 4.0 / 6.2 | 4.6 / **9.7** |

**Findings:**

1. Fine-tuning on 9.8 h of TORGO lowered streaming WER for every held-out speaker, dysarthric and control: 7.7
   (F03) to 14.9 (M05) points.
2. Most of the gain is on single words (−20 points), very little on sentences (−0.9). TORGO's prompts reuse a fixed
   word list across speakers, so the held-out speakers say words the model saw in training. Part of the gain is
   therefore vocabulary familiarity, not acoustic adaptation to dysarthria. The paper must report words and
   sentences separately.
3. Streaming on typical speech degraded by 56 % (6.2 → 9.7), while full-context WER barely moved. The degradation
   happened within the first 500 steps and did not recover. **Not shippable.**

### E2. Test runs: cause of the streaming degradation (2026-10-30)

**Goal:** find which atyp1 setting caused finding 3. Each run is atyp1's setup for 400 steps (warmup 100), changing
one thing. Script `train/runs/abl.cmd` (an earlier version, without `abl_ref` and `abl_combo`, which were never run).
Validated at step 400 on the same 300 clips, at chunk 16 / left 128, so comparable to E1's log.

| run | change from atyp1 | full | chunk16 / left128 |
|---|---|---|---|
| (step 0, shipped model) | | 6.87 | 11.79 |
| atyp1 at step 500 (E1, for reference) | none | 7.31 | 16.96 |
| `abl_nokd` | distillation off (λ = 0) | 6.51 | **11.01** |
| `abl_lowlr` | LR 2e-5 instead of 1e-4 | 6.69 | 12.18 |
| `abl_bigchunk` | chunks {16, 24, 32}, lefts {128} | 7.25 | 16.78 |

**Findings:**

4. **Frame-level distillation from the full-context model causes the degradation.** Without it, the streaming
   score improves over the shipped model within 400 steps; with it, restricting chunks to the device's sizes does
   not help. Our explanation: a streaming CTC model emits each token later than a full-context one, because it must
   wait for audio from the next chunk. A per-frame KL to the full-context posteriors penalizes this delay and pushes
   the streaming model to emit too early. (This is a hypothesis consistent with the numbers; we have not yet
   measured emission delays directly. To do: compare token emission frames of the two modes on the same clips.)
5. A 5× lower learning rate also mostly avoids the damage while keeping distillation, which fits the explanation:
   distillation still pulls the wrong way, but slowly.
6. Training only on device-like chunk sizes, with distillation on, does not help.

Caveat: one 400-step run per setting, one seed. The difference between `abl_nokd` and `abl_bigchunk` (5.8 points) is
far larger than run-to-run noise in this setup is likely to be, but the smaller differences (11.01 vs 12.18) are not
established.

### E3. atyp2: corrected fine-tune (started 2026-10-30, running)

**Settings** (`train/runs/atyp2.cmd`): distillation off; peak LR 5e-5, warmup 300; chunks {16, 24, 32}, lefts {64,
128}, `p_full` 0.25; validation at chunk 32 / left 128 (as on the device); TORGO repeated 2× instead of 3×; speed
perturbation with probability 0.3; 4000 steps of 150 s. Everything else as atyp1.

Before launch, the speed perturbation was checked with a 30-step run on TORGO with probability 1.0
(`G:\cosgc\exp\speedtest`, completed without errors; not evaluated).

**Validation during training (chunk 32 / left 128, so not comparable to E1's chunk-16 column):**

| step | full | chunk32 / left128 |
|---|---|---|
| 0 | 6.87 | 9.46 |
| 500 | 6.57 | 9.10 |

| 1000 | 6.78 | 8.84 |
| 1500 | 6.60 | 9.01 |
| 2000 | 6.60 | 9.04 |

Test results: to be added when the run finishes. Targets are in [TRAINING_PLAN.md](TRAINING_PLAN.md#step-3-what-counts-as-better).

### E4. atyp3: atyp2 + simulated dysarthria (queued 2026-10-30)

**Question:** does distorting typical speech toward dysarthric traits help on real dysarthric speakers, beyond what
TORGO alone gives? Free and needs no data agreement, so worth knowing for the paper either way.

**Method** (`train/simdys.py`, `--p_dys 0.3`): 30 % of LibriSpeech clips (never TORGO clips) get a random subset of
these distortions on their log-mel features, on the GPU:

| distortion | probability | parameters | dysarthric trait it imitates |
|---|---|---|---|
| uneven slowing | 0.7 | 150–500 ms pieces each stretched 1.0–1.8×, pitch unchanged | slow, irregular rate |
| time smoothing | 0.5 | moving average 30–90 ms, mixed 30–80 % | slurred transitions, imprecise articulation |
| mel smoothing | 0.5 | 3 neighbouring bands, mixed 30–80 % | blurred formants / vowels |
| tremor | 0.4 | loudness modulation 3–8 Hz, depth 10–40 % | vocal tremor, unsteady breath |
| breathiness | 0.4 | envelope-following noise, tilted to high bands, 10–25 dB below speech | breathy voice |

Not modeled: hypernasality, pitch breaks, monotone pitch, repetitions. This is a crude approximation.

**Check before launch** (2026-10-30): 20 random rounds on 4 LibriSpeech clips: distorted clips got longer (e.g.
1044 → 1403 frames), undistorted clips were unchanged, padding stayed zero, all values finite.

**Settings:** identical to atyp2 except `--p_dys 0.3` (`train/runs/atyp3.cmd`), so the difference between the two
measures the simulation alone. One seed each, so a small difference will not be conclusive.

**What to compare:** dysarthric speakers F03 and M05 (should improve if the simulation works), sentences separately
from words, and LibriSpeech (should not get worse, since the distortions only touch training clips).

## 5. Findings so far, for the paper

1. The shipped on-device model fails on dysarthric speech: 57 % and 90 % streaming WER for a moderate and a severe
   speaker, versus 6 % on LibriSpeech.
2. About 10 hours of TORGO lowers WER for unseen dysarthric speakers, by 8 to 15 points in streaming mode.
3. On TORGO, much of that gain is on its repeated word list; sentence-level gains are small.
4. Frame-level in-place distillation, a common technique for streaming models, damaged streaming accuracy when used
   to fine-tune an already-streaming model. Turning it off removed the problem.

## 6. Threats to validity and open issues

- **Few speakers.** 11 training and 4 test speakers, 2 of them dysarthric. Results per speaker are anecdotal; the
  direction across all four matters more than any one number. A second corpus (UASpeech) would allow a test on
  different speakers and recording conditions.
- **Validation on test speakers.** Settings in E2 and E3 were chosen by watching validation clips from the
  held-out speakers. Before the paper, re-run the final configuration with a validation set from a training speaker
  (or held out sessions), and report the test speakers only once.
- **Vocabulary overlap.** TORGO training and test speakers share prompts. Report an overlap-free subset: test clips
  whose transcript never appears in training.
- **One seed per run.** Before the paper, repeat the final configuration with at least 3 seeds and report mean and
  spread.
- **No language model in evaluation.** The deployed system uses one; the final model should also be evaluated with it.
- **Float, not deployed weights.** The device runs int8 (or int4) weights; final numbers must come from the exported
  `.tnm`.

## 7. How to reproduce

1. Download the data with `train/fetch_data.ps1` and check the hashes against [DATA.md](DATA.md).
2. Build manifests: `make_manifest_librispeech.py`, `make_manifest_torgo.py <TORGO dir> <prefix> --heldout
   F03,M05,FC03,MC04`. torgo_train_x3 is torgo_train repeated three times.
3. Convert the shipped model: `tnm_to_ckpt.py models/oido_stream.tnm <dir> --verify`.
4. Build the augmentation bank: `augment.py bank` and `augment.py rirs --n 400`.
5. Run `train/runs/*.cmd` (paths there are the training PC's), then `eval_manifest.py` on each test manifest.

Each run directory keeps `ckpt.pt` (resumable) and `last.pt` (EMA weights + normalization).

## 8. Works to cite

- TORGO: Rudzicz, Namasivayam, Wolff (2012), *Language Resources and Evaluation* 46(4). (Citation required by its terms.)
- LibriSpeech: Panayotov et al., ICASSP 2015.
- MUSAN: Snyder, Chen, Povey, arXiv:1510.08484, 2015.
- Conformer: Gulati et al., Interspeech 2020. CTC: Graves et al., ICML 2006.
- Oído engine and model: see `docs/oido-paper.pdf` and NOTICE.
- SpecAugment: Park et al., Interspeech 2019. Speed perturbation: Ko et al., Interspeech 2015.
- In-place distillation for dual-mode ASR: Yu et al., "Dual-mode ASR", ICLR 2021.
