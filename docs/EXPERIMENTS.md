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
| 2500 | 6.48 | 8.78 |
| 3000 | 6.30 | 8.78 |
| 3500 | 6.33 | 8.60 |
| 4000 | 6.36 | 8.66 |

**Test results** (`atyp2_torgo_test.json`, `atyp2_ls_testclean_train.json`), against the targets in
[TRAINING_PLAN.md](TRAINING_PLAN.md#step-3-what-counts-as-better):

| Test (full / stream) | baseline | atyp1 | atyp2 | target (stream) | met? |
|---|---|---|---|---|---|
| LibriSpeech test-clean | 4.01 / 6.21 | 4.58 / 9.65 | **4.00 / 6.10** | ≤ 6.5 (full ≤ 4.3) | yes |
| F03 | 49.5 / 57.0 | 38.2 / 49.3 | 39.8 / 48.8 | ≤ 50 | yes |
| M05 | 80.0 / 89.7 | 64.3 / 74.8 | 68.9 / 78.3 | ≤ 76 | no (2.3 over) |
| FC03 | 17.4 / 23.6 | 10.6 / 17.8 | 11.4 / 16.2 | ≤ 19 | yes |
| MC04 | 16.5 / 21.7 | 12.0 / 18.5 | 11.9 / 17.0 | ≤ 19 | yes |
| TORGO all | 30.8 / 37.3 | 22.7 / 30.9 | 23.9 / 30.1 | | |
| single words | 54.6 / 59.8 | 32.4 / 39.5 | 35.3 / 41.2 | | |
| sentences | 21.4 / 28.5 | 19.0 / 27.6 | 19.4 / 25.8 | < 27 | yes |

**Findings:**

9. Without distillation, the fine-tune keeps typical-speech accuracy fully (LibriSpeech streaming 6.21 → 6.10, full
   4.01 → 4.00) while keeping most of atyp1's TORGO gains. In streaming mode, atyp2 is better than atyp1 on 3 of 4
   speakers and on sentences; only the severe speaker M05 lost some of the gain (74.8 → 78.3).
10. Streaming sentences improved more than in atyp1 (28.5 → 25.8, vs 27.6), and words a little less (41.2 vs 39.5),
    consistent with TORGO's lower weight (2× instead of 3×).

atyp2 meets 5 of 6 targets. It is the first candidate for export (int8 `.tnm`, stress test) and the reference that
atyp3 is compared against.

**Resources:** 3 loader processes used about 1.7 GB of RAM each and the main process about 4.3 GB, leaving about
1.2 GB of the PC's 16 GB free; that was too little to use the PC at the same time. Run scripts now default to 2 loader
processes and below-normal priority.

### E4. atyp3: atyp2 + simulated dysarthria (queued 2026-10-30)

**Correction (2026-10-30, before the run started):** the queued atyp3 was switched from simulation version 1 (below) to
version 2, calibrated in E5, with `--dys_severity 0.2,1.0`. Version 1 was never trained with. Its description is kept
because E5 explains why it was replaced (its "tremor" pushed the 3–8 Hz rhythm the wrong way, and it lacked pauses).

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

### E5. Calibrating simulated dysarthria against TORGO (2026-10-30)

**Goal:** make the simulation resemble real dysarthric speech by measurement, not by ear, and check it the two ways
a reviewer would ask: does it sound (measure) like dysarthric speech, and does it make the recognizer fail like
dysarthric speech does.

**Measurements** (`train/dys_stats.py`, on log-mel features, leading and trailing silence trimmed, medians over clips):

| measure | definition | expected in dysarthria |
|---|---|---|
| rate | letters per second of speaking time | lower |
| pause | share of the trimmed clip more than 30 dB below the clip's loud frames | higher |
| flux | mean absolute change between neighbouring frames during speech, dB per mel band | lower if articulation is slurred |
| hilo | energy of top 20 mel bands minus bottom 20 during speech, dB | lower (weak consonants) |
| rhythm (`trem` in the output) | share of the loudness contour's 0.5–20 Hz modulation energy at 3–8 Hz | changes if tremor or rhythm changes |

**Real speech** (1500 random clips per set; `G:\cosgc\exp\dysstats_real.json`). Only TORGO **sentences** are compared,
dysarthric against control, because the same prompts were recorded with the same equipment. Comparisons with
LibriSpeech mix in recording differences: TORGO's background noise makes nearly every word-clip frame count as
"speech" (pause ≈ 0), and its microphones give hilo values 10–15 dB apart between sessions.

| TORGO sentences | rate | pause | flux | hilo | rhythm | clips |
|---|---|---|---|---|---|---|
| test speakers, control | 10.23 | 0.019 | 3.69 | 0.60 | 0.414 | 278 |
| test speakers, dysarthric | 7.14 | 0.124 | 3.43 | −0.32 | 0.284 | 125 |
| training speakers, control | 7.62 | 0.000 | 3.48 | 10.02 | 0.423 | 258 |
| training speakers, dysarthric | 7.15 | 0.183 | 3.57 | −3.67 | 0.180 | 115 |
| LibriSpeech test-clean (all) | 15.85 | 0.156 | 4.05 | −3.07 | 0.337 | 1500 |

Dysarthric relative to control, the targets for the simulation: rate × 0.70–0.94; pause +0.12 to +0.18; flux
× 0.93–1.02 (articulation smoothness barely differs); hilo −0.9 to −13.7 dB; rhythm × 0.43–0.69.

**Finding 7:** the 3–8 Hz loudness modulation is *weaker* in dysarthric speech, not stronger. In running speech this
band is dominated by the syllable rhythm (about 4–5 syllables per second), and dysarthric speech has a flatter
rhythm. Version 1's tremor added energy in exactly this band, the opposite of what the data shows, and version 1 had
no pauses, the clearest difference in the table.

**Version 2** (`distort_severity` in `train/simdys.py`): one severity s in (0, 1] scales every trait together, as in
real speakers, where severe dysarthria shows all of them more strongly. In order: pieces of 150–500 ms stretched by
1 + s · U(0.1, 0.8); frame-level variation added back to interpolated frames (interpolation is smoother than real slow
speech); loudness contour pulled toward its 300 ms average by s · U(0.1, 0.3); mel-band blur s · U(0.1, 0.4); high
bands lowered by up to s · U(6, 16) dB (linear across bands); breathy noise with probability 0.5 s; tremor with
probability 0.15 (some speakers have it); pauses of 150–450 ms (copies of the clip's quietest frames) at energy dips,
about 30 % · s of the clip. Named levels: mild 0.3, moderate 0.6, severe 1.0.

Three calibration rounds on 400 LibriSpeech test-clean clips (`G:\cosgc\exp\dysstats_sim2.json` holds the last):

| simulated / original | rate | pause | flux | hilo | rhythm |
|---|---|---|---|---|---|
| target (TORGO dysarthric / control) | × 0.70–0.94 | +0.12 to +0.18 | × 0.93–1.02 | −1 to −14 dB | × 0.43–0.69 |
| round 1: mild / moderate / severe | 0.85 / 0.74 / 0.63 | +0.02 / +0.05 / +0.10 | 0.67 / 0.55 / 0.48 | −1.4 / −2.6 / −3.5 | 0.66 / 0.40 / 0.28 |
| round 2 (no time blur, more pauses, more dulling, flux restored) | 0.85 / 0.75 / 0.64 | +0.05 / +0.12 / +0.21 | 0.78 / 0.71 / 0.65 | −2.1 / −4.3 / −5.3 | 0.61 / 0.43 / 0.32 |
| round 3 (final: more frame variation, less rhythm flattening) | 0.87 / 0.76 / 0.66 | +0.06 / +0.13 / +0.23 | 0.88 / 0.80 / 0.74 | −2.1 / −3.9 / −5.2 | 0.65 / 0.48 / 0.36 |

Round 1 is version 2 before tuning. At moderate severity, rate, pauses, dulling and rhythm fall in the TORGO range.
Remaining gap: the simulated speech is still somewhat smoother frame to frame than real speech (flux × 0.80 vs 0.93–1.02).

**Effect on the shipped model** (`eval_manifest.py --simdys`, every 5th LibriSpeech test-clean clip, 524 clips;
`G:\cosgc\exp\eval\base_ls_sim_*.json`):

| | full | stream | stream / original |
|---|---|---|---|
| original (all clips, from section 3) | 4.01 | 6.21 | 1.0 |
| simulated mild | 7.74 | 11.65 | 1.9 |
| simulated moderate | 12.74 | 19.52 | 3.1 |
| simulated severe | 24.40 | 35.50 | 5.7 |

For comparison, real TORGO speakers against TORGO controls (streaming, section 3): F03 2.5×, M05 3.9×, all dysarthric
3.0×. **Finding 8:** at moderate severity the simulation degrades the shipped model by about as much as real
dysarthria does. That is necessary for the simulation to be useful, but not sufficient: the model could fail on
simulated speech for different reasons than on real speech. Whether training on it helps real speakers is E4.

Caveats: one random draw per clip; measurement medians over a few hundred clips; the measurements are our own
simple proxies, not validated clinical measures (for the paper, cite the literature on acoustic markers of
dysarthria and state this).

**atyp3 progress (2026-10-30):** validation WER at chunk 32 / left 128 was 8.87 at step 2500 and **8.54 at step 3000**,
already below atyp2's final 8.66. Full results when the run ends.

**Interruption (2026-10-30, about 16:45):** atyp3 was stopped at about step 3250 to free RAM on the training PC and
restarted from its last checkpoint (`ckpt.pt`, saved every 250 steps: model, EMA, optimizer and step) a few minutes
later. Batches after the restart do not repeat the exact order of an uninterrupted run (new loader processes; the
sampler restarts the epoch), so this is not bit-identical to an uninterrupted run, but the schedule continues from
the saved step. The restart overwrote `atyp3.log`, so the per-step log before the restart is lost; the validation
numbers above were copied from it beforehand. The run scripts now append to their logs.

**atyp3 results (2026-10-30, 4000 steps; WER %, full / streaming chunk 32 left 128):**

| test | shipped | atyp2 | atyp3 |
|---|---|---|---|
| F03 (dysarthric) | 49.5 / 57.0 | **39.8 / 48.8** | 41.6 / 50.4 |
| M05 (dysarthric) | 80.0 / 89.7 | 68.9 / 78.3 | **67.3 / 77.9** |
| FC03 (control) | 17.4 / 23.6 | **11.4 / 16.2** | 11.5 / 16.6 |
| MC04 (control) | 16.4 / 21.7 | **11.9 / 17.0** | 12.9 / 18.1 |
| sentences | 21.4 / 28.5 | 19.4 / **25.8** | **19.3** / 26.4 |
| single words | 54.6 / 59.8 | **35.3 / 41.1** | 37.4 / 42.5 |
| all dysarthric | 60.1 / 68.3 | **49.9 / 59.0** | 50.5 / 59.9 |
| LibriSpeech test-clean | 4.01 / 6.21 | 4.00 / 6.10 | 4.02 / **6.07** |

**Finding 11:** simulated dysarthria (version 2, 30 % of typical clips) did **not** improve recognition of real
dysarthric speakers: M05 improved by 0.4 points, F03 got worse by 1.6, dysarthric speakers overall by 0.9 (streaming).
The validation subset (150 TORGO + 150 LibriSpeech clips) had suggested a gain (8.54 vs 8.84 at step 3000), which the
full test set does not confirm: a reminder that the small validation set is too noisy to pick between close runs.
Possible reasons: the simulation changes features in ways the real speakers do not (E5 matched only coarse
statistics), TORGO's real dysarthric data already covers what the simulation adds, or 4000 steps are too few for the
extra variety to pay off. One seed each; the differences are within what a second seed could change.

### E6. big1: a deeper model for the ESP32-P4 (queued 2026-10-30)

**Why:** the device moves from the ESP32-S3 to the ESP32-P4 (400 MHz instead of 240, 32 MB flash and 32 MB PSRAM
instead of 16 and 8). Memory then allows a much larger model; compute is the limit. On the S3 the 13 M model needs
1.54 s of compute per second of speech on one core (1.19 on two, README). Estimate for the P4, not measured (no board
yet): 1.7× the clock, the same 128-bit vector width, 2–3× faster PSRAM, but the P4 port lacks the S3's fastest matrix
kernel and runs on one core, so about 0.7–0.9 on one core and 0.55–0.7 on two. Compute grows with depth, so a model
1.5× larger (about 0.8–1.3) is the largest likely to keep up with live streaming. The S3 numbers are measured; the P4
numbers are an estimate to check on the board.

**Method:** depth up-scaling (`train/grow_nemo.py`). The shipped 16-layer model gets 8 new layers, one after every
second old layer (new positions 2, 5, 8, 11, 14, 17, 20, 23 of 24), 13 → 19 M parameters. Each new layer is a copy of the
layer before it with the output projections of its four residual branches (both feed-forward modules, attention,
convolution) set to zero. It then only re-applies the copied layer's final LayerNorm, so the grown model starts close
to, but not exactly at, the old one. Width, heads, kernel and tokenizer are unchanged; the engine and `.tnm` format
already read the depth from the header (`export_nemo.py` used to write 16 and 176 as constants, now it writes the
model's sizes).

**Settings:** identical to atyp2 (`train/runs/big1.cmd`), so the difference to atyp2 measures depth alone. The grown
model is also evaluated before training, to measure what inserting the layers costs. One change that does not affect
the model: the training audio is read from packed files (`train/pack_audio.py`: the same samples: all 5,214 TORGO test clips read back identical to the original files,
stored back to back sorted by duration). With one file per clip, the duration-bucketed batches made the training PC's
hard drive seek constantly (100 % busy at 1–2 MB/s during atyp3), which slowed the PC and likely the training. Compare
big1's time per step with atyp2's, keeping in mind the deeper model also costs about 1.5× compute.

**big1 results (2026-10-31):** the grown model *before training* was broken, not "close to the old one": 98.8 % (F03),
100 % (M05), 91–92 % (controls), 70.3 % (LibriSpeech) streaming WER. Re-applying a LayerNorm with the old layer's
scale and shift to an input that already carries them changes it a lot, and eight such insertions compound. After
4000 steps the model recovered but stayed worse than atyp2 everywhere (streaming: F03 51.5, M05 81.4, FC03 18.0,
MC04 19.4, LibriSpeech 7.06; atyp2: 48.8, 78.3, 16.2, 17.0, 6.10), so the training went into repairing the damage.
The `.tnm` export of the 24-layer model works (20.4 MB at int8).

**Finding 12 (training speed):** big1 took 5517 s for 4000 steps (1.38 s per step) from packed audio, against about
3.8 s per step for atyp3 from one file per clip, although big1's model needs 1.5× the compute. Reading training audio
sequentially made training roughly 3–4× faster on this PC (one hard drive); the data loading, not the GPU, had been
the bottleneck.

**Fix (big2):** `grow_nemo.py` now makes the start exact. The old layer before each new one gets a plain LayerNorm
(weight 1, bias 0), and the new layer's final LayerNorm takes over the old scale and shift. LayerNorm of an already
normalized vector is unchanged, so the new layer (zero branches) applies exactly the scale and shift the old layer
did. Measured: encoder outputs of the grown and the shipped model differ by 1.2 × 10⁻⁵ (relative, random input).
big2 = this start + the atyp2 recipe (`train/runs/big2.cmd`).

**big2 results (2026-10-31):** before training, the grown model scores what the shipped one does (streaming: TORGO
37.31 %, LibriSpeech 6.21 %, every speaker equal to the shipped model's to the second decimal), so the start is exact
in practice too. After 4000 steps (5933 s, 1.48 s per step from packed audio, partly slowed by an engine evaluation
running alongside) big2 beats atyp2 for every test speaker in both modes (WER %, PyTorch float, greedy):

| | atyp2 full | big2 full | atyp2 stream | big2 stream |
|---|---|---|---|---|
| all TORGO | 23.85 | 21.14 | 30.11 | **27.55** |
| F03 | 39.8 | 35.6 | 48.8 | 44.8 |
| M05 | 68.9 | 61.2 | 78.3 | 73.9 |
| FC03 | 11.4 | 9.9 | 16.2 | 14.1 |
| MC04 | 11.9 | 10.5 | 17.0 | 15.4 |
| sentences / words | 19.4 / 35.3 | 17.6 / 30.3 | 25.8 / 41.1 | 24.2 / 36.0 |
| LibriSpeech | 4.00 | 4.02 | 6.10 | 6.01 |

**Finding 15:** with the same data and recipe, 8 more layers (13 → 19 M parameters, initialized so the model starts
as the shipped one) lower streaming WER by 2.6 points overall and 4.0 / 4.4 for the dysarthric speakers, with no loss
on LibriSpeech. The gain holds for sentences and single words. The int8 export is 20.4 MB (fits the P4's 32 MB flash
with room for the language model); whether it runs in real time on the P4 is to be measured. Device-engine numbers in
E8; the quantization-aware version is big2q (`train/runs/big2q.cmd`, the atyp2q recipe on big2).

**What to compare:** big1 vs atyp2 per TORGO speaker and on LibriSpeech, and step time (training cost). On the P4: compute
per second of speech for both models; big1 is only useful if it stays under 1.0.

### E7. atyp4: atyp2 + UASpeech (2026-10-31): invalid, transcript bug

**Plan:** atyp2's recipe with UASpeech blocks B1 and B3 (microphones M5 and M2, 25,939 clips, 18 h) added to the
training data; test on UASpeech block B2 (M5, 7,136 clips, all 28 speakers) as well as TORGO (`train/runs/atyp4.cmd`).

**What happened:** atyp4 got worse on TORGO (streaming 34.36 % vs 30.11 % for atyp2; every speaker worse, F03 53.6,
M05 80.2, FC03 19.8, MC04 22.5; LibriSpeech 6.16 unchanged), and every model scored 94–96 % on UASpeech B2, controls
included. The transcripts were wrong: each one-word clip had the word repeated about 10 times ("paste paste ... paste"),
and some had phone-level labels ("sil silmarker dh their eh r ..."). Cause in `make_manifest_uaspeech.py`: the label
files list each utterance under several microphones and every release folder has a copy, the phone-alignment files
were read as well, and the reader appended all entries for an utterance instead of keeping one. The models' actual
outputs were mostly right ("paste", "golf", "your"). So atyp4 was trained to emit long repetitions for short clips,
which explains its TORGO loss; atyp4 says nothing about UASpeech as training data.

**Fix:** the reader keeps one entry per utterance from the `*_word.mlf` files only, and the manifest report now counts
words per transcript (must be 1 for UASpeech). The earlier note that each utterance is listed under one random
microphone was wrong. Manifests rebuilt; B2 hypotheses already computed are re-scored against the corrected
transcripts below; UASpeech training is re-run on the 24-layer model as big3 (`train/runs/big3.cmd`: big2 + UASpeech,
otherwise identical, so big3 vs big2 measures the data).

**UASpeech B2 re-scored** (corrected transcripts, 765 of 765 block/word pairs agree across speakers; 56 clips have two
words, "x ray" and "re united"; PyTorch float, greedy, WER %). None of these models saw UASpeech in training except
atyp4 (with the broken transcripts), so for the others all 28 speakers and the recording setup are unseen:

| | all, full | all, stream | control | dysarthric | digits | letters | commands | common | uncommon |
|---|---|---|---|---|---|---|---|---|---|
| shipped | 71.1 | 72.8 | 41.3 | 100.0 | 41.8 | 71.7 | 76.0 | 52.8 | 95.3 |
| atyp2 | 62.4 | 66.5 | 37.6 | 91.5 | 36.1 | 63.8 | 65.0 | 49.1 | 87.6 |
| big2 | 63.1 | 67.4 | 39.0 | 92.0 | 42.1 | 63.6 | 66.7 | 52.4 | 85.8 |
| atyp4 (bad labels) | 75.4 | 82.0 | 69.6 | 92.8 | 95.4 | 77.9 | 89.8 | 74.2 | 88.1 |

(stream columns after "all" are streaming.) Single isolated words with no sentence context and an open vocabulary are
hard even for the controls (37–41 %); UASpeech is usually reported with a closed vocabulary, so these numbers are not
comparable to published UASpeech results. Dysarthric WER near 100 % for the shipped model includes insertions.

**Finding 17:** TORGO fine-tuning carries over to an unseen corpus: 6.3 points better than the shipped model on
UASpeech B2 overall, 8.5 for the dysarthric speakers. But big2's advantage over atyp2 does not: on UASpeech they are
equal (67.4 vs 66.5 streaming). big2's 2.6-point TORGO gain (E6) may therefore come partly from fitting TORGO's
shared prompts better rather than from recognizing dysarthric speech better in general. The overlap-free TORGO subset
(46 clips, E8) points the same way as the full set, but is small. Report both corpora.

### E8. The deployed engine: int8 loss and the language model (2026-10-31)

**Why:** all numbers above come from PyTorch in floating point with greedy decoding. The device runs the C engine
with int8 weights and activations and can decode with beam search and the shipped n-gram language model
(`models/nemo_lm.tlm`). Two items under "threats to validity".

**Method:** models exported with `train/export_nemo.py` (int8), run with the host build of the engine
(`esp32/host/tasr_cli`, same arithmetic as the chip; new `--stream` option feeds 20 ms at a time to the streaming
engine with the chunk and left context stored in the model, 32 / 128) through `esp32/host/eval_cli.py` (same text
normalization and corpus-level WER as `eval_manifest.py`). Language model with the engine defaults (beam 4, weight and
bonus as tuned upstream on LibriSpeech dev-clean): nothing was tuned on TORGO.

**Results, TORGO test, streaming (WER %):**

| | shipped, PyTorch | shipped, engine | atyp2, PyTorch | atyp2, engine | atyp2, engine + LM |
|---|---|---|---|---|---|
| all | 37.31 | 37.37 | 30.11 | 33.67 | 30.87 |
| F03 | 57.0 | 57.1 | 48.8 | 53.3 | 49.5 |
| M05 | 89.7 | 89.8 | 78.3 | 83.0 | 78.6 |
| FC03 | 23.6 | 23.6 | 16.2 | 20.1 | 17.4 |
| MC04 | 21.7 | 21.8 | 17.0 | 19.1 | 17.3 |
| sentences / words | 28.5 / 59.8 | 28.5 / 60.1 | 25.8 / 41.1 | 27.1 / 50.6 | 23.6 / 49.5 |

**Finding 13:** the shipped model loses nothing in the int8 engine (37.31 → 37.37), but atyp2 loses 3.6 points
(30.11 → 33.67), most of it on single words (41.1 → 50.6). The float fine-tune gave up the int8 robustness the shipped
model was trained with. PyTorch with int8 fake quantization (`eval_manifest.py --bits 8`, every 5th clip) gives
33.1 %, so the simulation reproduces the loss and quantization-aware training should recover it (E8b, `atyp2q`:
atyp2 + 1000 steps with `--bits 8`). All float results above therefore overstate what the device achieves; final
numbers must come from the engine.

**Finding 14:** the language model, untuned for this data, recovers 2.8 points overall (33.67 → 30.87) and 4.0 for
the dysarthric speakers (63.6 → 59.6), mostly on sentences (27.1 → 23.6) and less on single words (50.6 → 49.5).

**Vocabulary overlap:** only 46 of 5,214 test clips have a transcript that never occurs in TORGO training
(`esp32/host/overlap.py`); their WER (34.8 % greedy, 31.2 % with LM) is close to the rest (33.6 / 30.9), but 46 clips
are too few to rule out that the shared prompts help.

**big2 (24 layers, E6) in the engine, TORGO test, streaming (WER %):**

| | shipped, engine | atyp2, engine + LM | big2, PyTorch | big2, engine | big2, engine + LM |
|---|---|---|---|---|---|
| all | 37.37 | 30.87 | 27.55 | 28.49 | **26.23** |
| dysarthric / control | 68.4 / 22.8 | 59.6 / 15.4 | | 56.9 / 15.2 | **52.9 / 13.7** |
| F03 | 57.1 | 49.5 | 44.8 | 47.1 | **43.4** |
| M05 | 89.8 | 78.6 | 73.9 | 75.4 | **71.0** |
| FC03 | 23.6 | 17.4 | 14.1 | 15.0 | **13.5** |
| MC04 | 21.8 | 17.3 | 15.4 | 15.3 | **14.0** |
| sentences / words | 28.5 / 60.1 | 23.6 / 49.5 | 24.2 / 36.0 | 24.4 / 38.9 | **21.7 / 38.0** |

**Finding 16:** the 24-layer model loses only 0.9 points in the int8 engine (27.55 → 28.49), against 3.6 for atyp2,
without quantization-aware training. With the language model it reaches 26.2 % on the device arithmetic, 11.1 points
(30 % relative) below the shipped model, 15.5 points below it for the dysarthric speakers (68.4 → 52.9). On the 46
clips with a transcript unseen in training: 32.4 % greedy, 27.9 % with LM (rest 28.4 / 26.2).

### E8b. atyp2q: quantization-aware fine-tune, and where the int8 loss comes from (2026-10-31)

**atyp2q:** atyp2 plus 1000 steps with `--bits 8` (`train/runs/atyp2q.cmd`). Validation 6.66 full / 9.07 streaming at the
start, 6.57 / 8.96 at the end. TORGO test, streaming:

| | atyp2, engine | atyp2, engine + LM | atyp2q, PyTorch `--bits 8` | atyp2q, engine | atyp2q, engine + LM |
|---|---|---|---|---|---|
| all | 33.67 | 30.87 | 32.66 | 32.47 | 29.84 |
| dysarthric / control | 63.56 / 19.61 | 59.55 / 17.38 | | 62.02 / 18.57 | 58.15 / 16.52 |
| F03 | 53.27 | 49.51 | 52.46 | 52.01 | 48.35 |
| M05 | 83.02 | 78.56 | 81.49 | 80.96 | 76.70 |
| FC03 | 20.05 | 17.41 | 18.63 | 18.63 | 16.31 |
| MC04 | 19.07 | 17.34 | 18.61 | 18.49 | 16.79 |
| sentences / words | 27.06 / 50.55 | 23.6 / 49.45 | | 26.66 / 47.3 | 23.3 / 46.54 |

PyTorch full context with `--bits 8`: 25.74; LibriSpeech test-clean 6.16 streaming, 4.02 full. The simulation matches the
engine (32.66 against 32.47), so it can be trusted for quantization experiments.

**Which part of int8 causes the loss** (`eval_manifest.py --fq`, atyp2, every 5th TORGO test clip, streaming WER %):

| | none (float) | attention rows only | activations only | weights only | all three |
|---|---|---|---|---|---|
| all | 29.24 | 29.39 | 29.77 | **33.22** | 33.11 |
| F03 / M05 | 44.67 / 76.74 | | | 49.24 / 80.4 | |
| FC03 / MC04 | 15.84 / 16.25 | | | 19.98 / 19.73 | |

atyp2q's float weights on the same subset score 29.8.

**Finding 18:** nearly all of atyp2's int8 loss is weight rounding (29.24 → 33.22); activations and attention rows cost
about 0.5 and 0.15 points. 1000 steps of quantization-aware training recover only 1.2 of the 3.6 points. **Hypothesis, not
tested:** the shipped weights already sit on the int8 grid, so small fine-tune changes are partly rounded away, while big2's
new layers have small weights and therefore a fine per-row scale. To try: a clipped or MSE-optimal per-row scale at export
(`export_nemo.py`, no engine change) and quantization-aware training for the whole run.

**Failure (my error, 2026-10-31):** that diagnosis ran on the GPU next to big2q. Training holds about 11.2 GB, the ProtonVPN
client 2.4 GB and the evaluation 485 MB, on a 12 GB card. Both crawled for about 10 hours: big2q started at 10:23 and had
reached step 400 by 15:58, with no error and a normal-looking log. Stopping the last, redundant variant of the diagnosis
freed it at once (steps 400 to 500 had taken 4.4 hours). Not an accuracy problem, but the wall-clock times of big2q (37274 s)
are not comparable with other runs. Rule since then: no GPU job next to training; check the log's last-write time.

### E9. First run on real hardware: ESP32-P4 (2026-10-31)

**Board:** plugged in over a CH343 USB-serial bridge (VID 1A86 PID 55D3, COM3) to UART0, not the native USB-Serial/JTAG.
ESP32-P4 revision v3.2 (ROM `esp32p4-eco7-20260109`), 400 MHz measured (12,039,016 cycles in 30,097 µs), 32 MB PSRAM (hex,
200 MHz), GigaDevice flash c8/4019, 32 MB. The exact board model is still assumed (Waveshare ESP32-P4-Pico); the codec's
chip id will confirm the pins. Firmware: `oido_p4_bench`, ESP-IDF 5.5.4, one core, model `models/oido_stream.tnm` (shipped,
16 layers, 13 M parameters), LM `models/nemo_lm.tlm`, beam 4, four clips totalling 24.1 s.

**Results:** self-test of the vector kernels against plain C: all four PASS (QACC layout 1). Transcripts are
word-for-word identical to the host engine on all four clips.

| One core, streaming | S3, 240 MHz | P4, 400 MHz, plain C | P4, 400 MHz, vector kernels |
|---|---|---|---|
| wall RTF (compute time / audio time) | 1.54 | 5.25 (corrupted model, see below) | **0.759** |

Mean compute left after the last audio block: 542 ms; final text at most 822 ms after the end of speech in the paced runs. Minimum
free memory: PSRAM 15.4 MB, internal 267 KB. Share of the run: decoder 26.7 % (language model most of it, 200 GRU steps),
`gemm` 56.4 % (of which 704-wide feed-forward layers 19.4 %), attention 12.5 %, depthwise convolution 8.4 %, features 5.6 %.
Log: `docs/results/p4_bench_13m.log`. The plain-C figure comes from a run with part of the model overwritten (bug 3 below);
its transcript was wrong, so it is a timing comparison only: the vector kernels make the whole system about 6.9 times
faster, and `gemm` alone about 8.2 times (33.9 G against 4.12 G cycles).

**Finding 19:** the shipped model streams faster than real time on one P4 core with the same transcripts as the host, about
twice as fast as the S3 (1.54 → 0.759). One core is idle, so there is headroom for a larger model; big2's speed is not measured
yet (it needs bug 3 fixed properly).

**Three bugs only the hardware could reveal** (none is visible in a host build, and each was found by a measurement):

1. **Wrong vector instruction encoding (mine).** `kernels.c` was compiled with `-march=..._xesppie` because I assumed IDF's
   default might lack the vector instructions. IDF's default for this chip is `xesploop_xespv`, and the same mnemonic assembles
   differently (`esp.zero.xacc`: `0x5b` against `0x1b`). The first vector instruction crashed the board with an
   undocumented cause (`MCAUSE 0x1f`, later `0x7` with a nonsense address). Removing my override fixed it. A build system
   detail cost one more round: PlatformIO re-reads CMake only when the top-level `CMakeLists.txt` changes, so the first rebuild
   silently kept the old flag; checking the object file's `Tag_RISCV_arch` showed that.
2. **esptool's stub flasher does not work on this chip** (esptool 4.11 and 5.2.0, every baud rate): "Invalid head of packet
   (0x61)" or "The chip stopped responding". The ROM loader works up to at least 2 Mbaud, so we use `--no-stub`. The ROM on this
   chip is dated 2026; a stub depending on older ROM addresses is a plausible cause, not a confirmed one.
3. **Flash above 16 MB wraps around (a design error of mine).** I laid out the 32 MB flash with the language model at
   `0x1A10000` and the clips at `0x1E10000`. Without 32-bit addressing these writes land at `0xA10000` and `0xE10000`, inside the
   model. Every per-image hash passed because reading back wraps the same way. Symptom: the same nonsense for every clip
   ("the fors to the fors..."), with plain C and vector kernels alike. Found by bisection: C kernels (not the vectors), audio
   hash and features (match the host to 5 decimals), per-layer values (identical up to layer 8, `inf` in layer 9), then a hash of
   every stored parameter, then `verify-flash` of the whole image. **Proof:** `lm.bin` and `audio.bin` match the flash exactly at
   the wrapped addresses, and the damaged model blocks are exactly the offsets predicted (`0x800000`-`0x93FFFF`, `0xC00000`-`0xCBFFFF`).
   Fix for now: all images below 16 MB. A 19 M model does not fit that way.

A lesson for the method: self-tests and per-image hashes passed throughout, and the only reliable check was comparing the
transcript and every intermediate result with the host.

**Other failures on the way:** I killed a PlatformIO run during tool installation, which removed the toolchain; editing
`platformio.ini` while it is open in VS Code starts the PlatformIO extension's `project init`, which rewrites the shared package
folder underneath a running build (it broke one flash attempt); my build watcher read only stdout while failures go to stderr, which
hid a failed build for two hours.

## 5. Findings so far, for the paper

1. The shipped on-device model fails on dysarthric speech: 57 % and 90 % streaming WER for a moderate and a severe
   speaker, versus 6 % on LibriSpeech.
2. About 10 hours of TORGO lowers WER for unseen dysarthric speakers, by 8 to 15 points in streaming mode.
3. On TORGO, much of that gain is on its repeated word list; sentence-level gains are small.
4. Frame-level in-place distillation, a common technique for streaming models, damaged streaming accuracy when used
   to fine-tune an already-streaming model. Turning it off removed the problem.
5. In TORGO, dysarthric speakers differ from controls reading the same sentences mainly by more pauses, slower rate,
   weaker high frequencies and a flatter syllable rhythm, not by smoother articulation (E5).
6. A simulation calibrated to those differences degrades the shipped model by about as much as real dysarthria does
   (E5). Whether training on it helps is pending (E4).
7. Float fine-tunes can lose several points in the device's int8 arithmetic that the shipped model did not (E8);
   results must be measured on the exported model.
8. Growing the model from 16 to 24 layers with an exact identity start, then fine-tuning, beats the same fine-tune at
   16 layers for every test speaker, and is more robust to int8 (E6, E8). On the device arithmetic with the language
   model: 26.2 % WER on TORGO test vs 37.4 % for the shipped model (dysarthric speakers 52.9 vs 68.4). On an unseen
   corpus (UASpeech B2) the 24-layer model is no better than the 16-layer one, so part of the depth gain may be
   TORGO-specific (E7).
9. TORGO fine-tuning improves an unseen corpus too (UASpeech B2: 72.8 → 66.5 % streaming, dysarthric 100 → 91.5).
10. Nearly all of the int8 loss of a float fine-tune comes from weight rounding, not activations (E8b).
11. On real hardware, the shipped model streams at 0.759 times real time on one ESP32-P4 core at 400 MHz, about twice as fast
    as the ESP32-S3, with transcripts identical to the host engine on four clips (E9). The speed of the 24-layer model on the
    chip is not measured yet.

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
