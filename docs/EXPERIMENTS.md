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

**Microphone firmware (`oido_p4`), same night:** the codec answers with chip id `0x8311`, which confirms the I2C pins and
address. The API works in both directions over UART0. In a quiet room the microphone reads -61.3 to -55.8 dBFS, median -57.9 (142 level
events in 45 s), so the capture path delivers a live signal, not constant zeros. Right after boot the voice detector
ran once for the 20 s maximum and produced "the", with nobody speaking; it did not recur in the next 45 s. **Not tested: recognition
of a person speaking at the board.** Nobody was present.

### E9b. Flash above 16 MB, and the 24-layer model on the chip (2026-11-01)

**Correction to E9:** bug 3 has a better fix than "keep everything below 16 MB", and big2's speed is now measured.

**Method:** the flash chip has an Extended Address Register (commands C8h read, C5h write) that selects the 16 MB bank for
3-byte addresses. Test on the unused top of the flash: with the register at 1, the ROM loader wrote and verified a 4 KB pattern, and bank 0 was
unchanged. `esp32/tools/flash_p4.py` splits images at the boundary, switches banks, verifies every image again after the last write
and leaves bank 0 selected. The cache still cannot reach above 16 MB, so the app, LM and clips stay below it, and at boot the P4 firmware
reads the model into PSRAM through the flash driver (`read_partition_to_psram` in `main.c`), which uses 4-byte commands. ESP-IDF's
experimental `CONFIG_BOOTLOADER_CACHE_32BIT_ADDR_QUAD_FLASH` is not used. Model partition at `0x610000`, 24 MB.

**Results,** one core, 400 MHz, same four clips (24.1 s), beam 4 + LM:

| | shipped 13 M, model read from flash | shipped 13 M, whole model in PSRAM | big2, 19 M, whole model in PSRAM |
|---|---|---|---|
| wall RTF | 0.759 | **0.748** | **0.967** |
| per clip | 0.754, 0.754, 0.759, 0.781 | 0.743, 0.743, 0.748, 0.767 | 0.964, 0.962, 0.969, 0.979 |
| mean compute after the last block | 542 ms | | 755 ms |
| final text after end of speech (paced runs) | at most 822 ms | | at most 1120 ms |
| model read at boot | | 1279 ms | 1865 ms (about 11 MB/s) |
| minimum free PSRAM | 15.4 MB | 14.4 MB | 6.5 MB |
| transcripts against the host engine | identical | identical | identical |

Logs: `docs/results/p4_bench_13m_psram.log`, `docs/results/p4_bench_big2.log`. big2's share of the run: `gemm` 64.2 %, decoder 20.9 %, attention
15.0 %, depthwise convolution 9.9 %. The decoder cost is about the same as with the 13 M model (1.94 G cycles against 1.95 G), since it depends on the
language model, not the depth.

**Finding 20:** the 24-layer model, our best on TORGO (26.2 % against 37.4 %), streams in real time on one ESP32-P4 core: 0.967 times
real time. The margin is only 3 %, this benchmark has no microphone capture running, and four clips are few. The second core is idle and is the obvious
source of headroom. Compute grew by 29 % for 50 % more encoder layers, because the decoder and front end do not grow.

**Both cores, and the language model as stored (2026-11-01).** Envs `oido_p4_bench_2c`, `oido_p4_2c` (spin-wait jobs, as on the S3). A
correction to earlier numbers: the firmware re-quantizes the language model to int4 when loading it (`CONFIG_TASR_LM_INT4`, written for the
S3's slower PSRAM), while every host evaluation with the LM used int8. So LM scores in this log described a slightly different LM from the one on
the device. The P4 now keeps int8 (`sdkconfig.p4`), which matches the host and can also be split across the cores. big2, same four clips:

| | one core, int4 LM | two cores, int4 LM | two cores, int8 LM |
|---|---|---|---|
| wall RTF | 0.967 | 0.647 | **0.602** |
| decoder, of which LM (cycles) | 1.94 G, 1.77 G | 1.97 G, 1.79 G | **1.08 G, 0.85 G** |
| final text after end of speech, paced, at most | 1120 ms | | **745 ms** |
| minimum free PSRAM | 6.5 MB | 6.5 MB | 6.0 MB |
| transcripts against the host engine | identical | identical | identical |

Logs: `docs/results/p4_bench_big2_2c.log`, `p4_bench_big2_2c_lm8.log`, and `p4_bench_big2q121_2c.log` (the E10 export, RTF 0.603, again identical to the host).
The microphone firmware on two cores boots ready in 3.2 s and ran 90 s with no crash; the screen code logged "no SSD1306" as designed (none attached). Nobody
has spoken at it yet, and the S3's two-core mode had transcript errors with its original kernels, so transcripts during live use still need checking.

### E10. Most of a fine-tune was lost at export: the int8 grid (2026-11-01)

**Question:** Finding 18 showed the int8 loss comes from weight rounding, with a guess about why. Why would 8 bits hurt a fine-tune and not the shipped model?

**Measurements** (`train/quant_stats.py`, `train/quant_survival.py`; CPU, no audio):

- The clipping idea from E8b is **refuted**: per-row max/rms is only about 3, and the MSE-optimal per-row scale gains 0.00 dB.
- Rounding error against the size of the fine-tune's change is almost exactly 1.0 in nearly all layers, as it would be if changes rounded back to their start.
- The shipped weights lie exactly on the grid `max|w|/127`, and a change under half a step returns to the starting value:

| Fine-tune | grid | share of the change kept | weights whose rounded value changed | SNR of the rounded weights |
|---|---|---|---|---|
| atyp2 | 127 (old) | **11.6 %** | 1.7 % | 52.6 dB |
| atyp2 | 125 | 92.3 % | 8.7 % | 41.7 dB |
| atyp2 | **121** | **99.2 %** | 8.7 % | 40.8 dB |
| atyp2 | 113 | 98.5 % | 8.0 % | 39.9 dB |
| big2 | 127 (old) | **22.5 %** | 14.7 % | 53.1 dB |
| big2 | **121** | **99.3 %** | 20.5 % | 41.1 dB |
| big2q | 127 (old) | 23.2 % | 14.8 % | 52.8 dB |
| big2q | 121 | 99.3 % | 20.8 % | 41.1 dB |

**Fix:** round on a different grid, `max|w|/121`, so rounding no longer cancels small changes. The rounding noise is real now (about 41 dB, 12 dB
worse), but it is unrelated to the change and averages out over a row: dithering, in effect. We chose 121 from the table above, before scoring anything, and did not tune
it on test data. `export_nemo.py` uses it for any fine-tune (`FINE_TUNE_QMAX`); with no checkpoint it stays on 127, so the shipped export does not change. `TASR_FQ_QMAX=121` makes
the PyTorch simulation match.

**Results, TORGO test, streaming, C engine (WER %), full test set:**

| | PyTorch float | engine, old export | engine, **new** | engine + LM, old | engine + LM, **new** |
|---|---|---|---|---|---|
| atyp2 | 30.11 | 33.67 | **30.11** | 30.87 | **27.82** |
| big2 | 27.55 | 28.49 | **27.52** | 26.23 | **25.14** |
| big2q | | 28.26 | 27.66 | 25.96 | 25.19 |
| big3 (E11) | 29.61 | | 29.65 | | 27.55 |

big2, engine + LM, new export, against the shipped model's engine score without LM (37.37; dysarthric 68.39): all 25.14, dysarthric 51.17, control 12.90, F03 41.80, M05 68.91, FC03
12.50, MC04 13.37, sentences 21.57, words 34.26. For the same model on the old export: 26.23, 52.90, 13.69, 43.35, 70.97, 13.45, 13.97, 21.65, 37.95.
Every speaker and slice improved. Every-5th-clip check first (greedy): atyp2 33.63 → 29.23, big2 27.72 → 26.16. Results in `esp32/host`-format JSON on the laptop (`*_q121_*`).

**Other corpora on the new grid** (PyTorch with `TASR_FQ_QMAX=121`, `--bits 8`, greedy, `train/runs/q121eval.cmd`; WER %):

| | LibriSpeech test-clean, full / stream | UASpeech B2, full / stream | UASpeech control / dysarthric (stream) |
|---|---|---|---|
| atyp2 | 4.04 / 6.13 | 62.25 / 66.46 | 37.86 / 91.27 |
| big2 | 4.06 / 6.04 | 63.24 / 67.71 | 39.09 / 92.55 |
| big2q | 4.14 / 6.15 | 65.96 / 69.09 | 41.51 / 93.01 |

The grid costs nothing on typical speech (big2 float 4.02 / 6.01). On UASpeech, atyp2 and big2 match their float scores within 0.4 points (66.5, 67.35). big2q does worse than it did on the old
grid (66.88 streaming), so on this corpus quantization-aware training on the old grid is a net loss once the grid is fixed: one more reason to drop it.

**Finding 21:** the int8 loss was not a limit of 8 bits. It was an artifact of rounding a fine-tune on the grid of the model it started from, which threw away 77 to 88 % of the weight change. A
different grid removes the loss entirely: engine scores now equal float (30.11 = 30.11; 27.52 against 27.55), at no cost in size, speed or training. This should apply to
anyone fine-tuning an already-quantized model. **Consequences:** E8's Finding 13 and E8b's Finding 18 correctly describe the old exporter, not a limit of the device. Quantization-aware
training is not needed on this model: big2q gains nothing over big2 on the new grid (25.19 against 25.14). All engine results before this entry understate the fine-tunes.

### E11. big3: big2 + UASpeech with corrected transcripts (2026-11-01)

**Question:** does adding UASpeech to training help, now that its transcripts are right (E7)? **Run:** `train/runs/big3.cmd`, identical to big2 (start, steps, learning rate, masks) with UASpeech
blocks B1 and B3 added (18 h on top of 120 h; the number of steps stays 4000, so TORGO and LibriSpeech each get a smaller share of the batches). Took 7369 s.

| WER % | big2 | big3 |
|---|---|---|
| TORGO test, engine, new grid, greedy | 27.52 | 29.65 |
| TORGO test, engine, new grid, + LM | **25.14** | 27.55 |
| ... dysarthric / control (+ LM) | 51.17 / 12.90 | 53.55 / 15.32 |
| ... F03 / FC03 / M05 / MC04 (+ LM) | 41.80 / 12.50 / 68.91 / 13.37 | 43.53 / 13.17 / 72.50 / 17.94 |
| LibriSpeech test-clean, float, stream / full | 6.01 / 4.02 | 6.2 / 4.05 |
| UASpeech B2, float, stream / full | 67.35 / 63.11 | **46.54** / 42.89 |
| ... UASpeech control / dysarthric (stream) | 38.97 / 91.98 | **19.69 / 69.83** |

**Reading this.** Every TORGO speaker got worse, by 0.7 to 4.6 points, and MC04 (a control) the most. The large UASpeech gain does **not** show better recognition of new dysarthric speakers: B1, B2 and B3
share all 28 speakers and most of the words, so big3 was tested on people and vocabulary it trained on, only with a different block and microphone. The fair measure on UASpeech would hold out whole
speakers, which we have not run. big3's TORGO loss may be partly a dilution of batches (isolated words from a noisier setup, steps unchanged), which this run cannot separate from a real conflict between corpora.

**Finding 22:** adding a second dysarthric corpus of isolated words made the model worse on the first corpus's unseen speakers (25.14 → 27.55 with the language model). big2 stays the deployed model. Do not
quote big3's UASpeech score as a gain for new speakers.

### E12. Longer context on the device: full-context final text (2026-11-01)

**Question:** big2 scored about 21 % with full context and 27.5 % streaming in PyTorch. A push button tells the device when speech ended, so it could wait for more audio. How much accuracy does each length of context buy on the deployed model, and is full context affordable on the chip?

**Sweep, big2 on the new grid, C engine with the language model, full TORGO test set (5214 clips; chunk in encoder frames of 40 ms; WER %):**

| Context | all | control | dysarthric | sentence | word | F03 | FC03 | M05 | MC04 |
|---|---|---|---|---|---|---|---|---|---|
| chunk 32 = 1.28 s (deployed so far) | 25.14 | 12.90 | 51.17 | 21.57 | 34.26 | 41.80 | 12.50 | 68.91 | 13.37 |
| chunk 48 = 1.92 s | 23.00 | 10.74 | 49.06 | 19.42 | 32.14 | 39.69 | 10.56 | 66.78 | 10.97 |
| chunk 64 = 2.56 s | 21.89 | 10.30 | 46.55 | 18.22 | 31.27 | 36.73 | 9.96 | 65.11 | 10.70 |
| **full utterance** | **19.75** | 9.21 | 42.15 | 15.99 | 29.36 | 33.81 | 8.82 | 57.92 | 9.69 |
| full utterance, no language model | 21.24 | 10.10 | 44.94 | 17.56 | 30.67 | 35.78 | 9.73 | 62.25 | 10.56 |

Every speaker and slice improves at each step. big2 was not trained for this: only a quarter of its batches had full context, and none had chunk 64 (`--chunks 16,24,32`).

**Left context barely matters** (every 5th clip, 1043 clips; all / dysarthric): chunk 32 with 2.56 s of left context 24.54 / 48.21, 5.12 s (deployed) 24.24 / 47.34, 10.24 s 24.02 / 46.69; chunk 64 with the same three 20.99 / 42.67, 20.95 / 42.56, 20.99 / 42.67. Audio *after* a frame is what helps. On this subset chunk 16 gives 27.90 / 53.31 and full context 19.22 / 38.98. Our first "chunk 96" run is not a result: the engine silently caps a chunk at 64 rows (`RB`), so it reproduced chunk 64 exactly.

**On the chip** (`oido_p4_bench_full_2c`, two cores, 400 MHz; `docs/results/p4_bench_big2q121_full_2c.log`). Clips joined to the wanted length, the streaming object still allocated:

| Utterance | 1 s | 2 s | 3 s | 5 s | 8 s | 12 s | 16 s | 20 s |
|---|---|---|---|---|---|---|---|---|
| Time for the full-context pass | 0.68 s | 1.29 s | 1.93 s | 3.17 s | 5.27 s | 8.30 s | 11.77 s | 15.70 s |
| x real time | 0.682 | 0.643 | 0.642 | 0.633 | 0.659 | 0.692 | 0.736 | 0.785 |
| Min free PSRAM (MB) | 4.73 | 4.64 | 4.55 | 4.36 | 4.09 | 3.72 | 3.34 | 2.97 |

Attention is quadratic in length, but it only starts to show beyond about 8 s. TORGO test clips have a median of 1.8 s for single words and 3.9 s for sentences (90 % under 4.5 s, 3 of 5214 over the 20 s buffer), so the usual wait at this rate is about 1 to 3 s.

**A transcript difference, investigated.** On one clip of four the board's full-context pass says "yeah i think of it ..." where the host says "i think of it ...", on every run. This would be a bug if the arithmetic differed, so we compared the two layer by layer. The front end agrees to about 1 part in a million (the C libraries' `logf` differ in the last bit), then the difference grows through the layers to about 1.5 % in the logits' squared sum. Two tests settle it:

1. **The kernels are exact:** the board with plain C kernels gives the same hash as the board with vector kernels at 899 of 899 checkpoints, in full-context mode.
2. **The host does the same when disturbed:** multiplying the host's features by noise of 1 part in a million, with six seeds, moves the logits' squared sum across 4.83 to 4.94 x 10^6 (undisturbed 4.86, board 4.93), and one seed of six produces the same "yeah".

So it is a marginal word tipped by rounding, not a defect. The likely amplifier is the dynamic quantization: a tiny input change flips some int8 rounding, which then acts as noise of its own at each of 24 layers. Earlier "transcripts identical to the host" statements hold for the clips checked, but identity should not be expected on every clip, and a device score can differ slightly from a host score clip by clip.

**Firmware:** `CONFIG_TASR_FINAL_FULL` (on for the P4). Streaming still runs while you speak and gives the partial text; when speech ends the whole utterance goes through the full-context pass for the final text. An utterance beyond the 20 s recording buffer keeps the streaming text. The audio ring (16 s with the voice detector, 32 s with a button) holds more than the longest pass, so nothing is lost meanwhile.

**First speech through the board's own microphone** (`docs/results/p4_mic_full_2c_playback.log`). No person yet: bench clips played on the laptop's speakers at its existing volume, board position not controlled, so this is a check of the path, not an accuracy measurement.

| Clip | Streaming partial text | Final text (full context) | Wait |
|---|---|---|---|
| real_20 | no such thought as he was given for the course of peal of registry remained cosed | no such authority was given for the court of appeal into registries remained closed | 3.75 s for 5.7 s |
| rooms_00 | it is such a noble ambition that it is a pity it has usually such a shallow foundation | the same, correct | 4.46 s for 6.7 s |

The reference for real_20 is "no such authority was given for the court of appeal whose registries remained closed". The third clip, real_26, produced no text. It is recorded about 18 dB quieter than the others (-45.7 dBFS against -28.8 and -26.8), which may be why, but we did not test that. **Problem seen:** seven short bursts (1.3 to 1.8 s) triggered the voice detector and produced empty text, six of them in the last 20 s or so of the 70 s run, with the laptop's fans probably running from other jobs. A button avoids this. The detector's threshold has not been tuned on real rooms.

**Finding 23:** on the deployed 24-layer model, using the whole utterance instead of 1.28 s chunks lowers WER from 25.14 to 19.75 on TORGO test, and from 51.17 to 42.15 for dysarthric speakers, with no training and no change to the model. The ESP32-P4 can afford it: about 0.64 times the utterance's length up to 8 s on two cores. Compared with the shipped model's streaming score without a language model (37.37; dysarthric 68.39), that is close to half the errors overall. For this model the cost of real-time text is larger than the cost of int8, of the model's size, or of the language model.

**Finding 24:** because of dynamic int8 quantization, the engine amplifies a difference of one part in a million in the input to about 1 % in the logits, enough to flip a marginal word. Bit-exact kernels do not imply identical transcripts across platforms unless `logf` matches too.

### E13. Where the errors are, and how much dysarthric speech training sees (2026-11-01)

**big4** (`train/runs/big4.cmd`): big2's data and steps with `--chunks 32,64 --lefts 64,128 --p_full 0.5 --val_chunk 64`, more training at full context because the final text now comes from it. Took 6177 s. Small gain at full context, small loss at chunk 32 (table below).

**Language model grid** (big2_q121, full context, beam 4, TORGO test every 5th clip, 1043 clips; `eval_cli.py --lm_weight w --token_bonus b`, weights 0.15, 0.3, 0.45, 0.6, 0.8, bonuses 0, 0.5, 1, 1.5). WER across all clips:

| weight \ bonus | 0 | 0.5 | 1 | 1.5 |
|---|---|---|---|---|
| 0.15 | 19.33 | 19.14 | 19.14 | 19.4 |
| 0.3 | 19.92 | **19.22 (in use)** | 19.07 | 19.03 |
| 0.45 | 21.14 | 20.58 | 19.84 | 19.59 |
| 0.6 | 23.24 | 21.77 | 21.47 | 20.62 |
| 0.8 | 26.61 | 25.17 | 23.8 | 22.76 |

We chose by tuning on two speakers and checking on the other two, not on the test score. Tuning on F03 + FC03 picked 0.3 / 1, which helped the others (33.58 against 34.05, mean of speakers); tuning on M05 + MC04 picked 0.15 / 1, which hurt the others (19.69 against 19.43). Tuning on F03 + MC04 and on M05 + FC03 split the same way. The optimum depends on the speakers, so the setting stays at 0.3 / 0.5. At most about 0.2 points are left here, and more weight hurts, single words most (43.69 at 0.8 / 0).

**Beam width** (same model and clips, `--beam`):

| beam | all | control | dysarthric | sentence | word |
|---|---|---|---|---|---|
| 2 | 19.62 | 9.3 | 39.63 | 16.21 | 28.22 |
| 4 (in use) | 19.22 | 9.02 | 38.98 | 15.64 | 28.22 |
| 8 | 19.07 | 8.96 | 38.65 | 15.44 | 28.22 |
| 16 | 18.99 | 8.96 | 38.44 | 15.33 | 28.22 |

Beam 32 is not a result: `MAXB 16` in `tinyasr_lm.c` caps it silently, and its per-clip output is identical to beam 16. Cost on the chip has not been measured.

**Where the errors are** (big4_q121, full context + LM, full test set; `sub / del / ins` as a percentage of reference words):

| group | kind | WER | sub / del / ins | clips with no text | clips fully right |
|---|---|---|---|---|---|
| control | sentence | 7.4 | 3.6 / 3.1 / 0.7 | 0.1 % | 73.3 % |
| control | word | 13.7 | 11.3 / 1.7 / 0.7 | 1.7 % | 86.8 % |
| dysarthric | sentence | 33.9 | 20.7 / 11.0 / 2.3 | 0.7 % | 22.3 % |
| dysarthric | word | 60.7 | 46.7 / 6.6 / 7.3 | 6.6 % | 45.9 % |

F03 and M05 cause 68 % of all word errors. Among single-word clips, 17.8 % of those of 0.8 to 1.2 s produce no text, against 2.5 % of those of 1.2 to 2 s. Errors include words cut short ("steer" to "ste", "weep" to "we"), for control speakers too, and homophones ("hear" to "here").

**Silence after the clip** (big4_q121, + LM; the clips are written with zeros appended and passed with `eval_cli.py --wavs`). Hypothesis: cut-off and empty short words are an edge effect. Results on the full test set, 5214 clips:

| padding | all | control | dysarthric | word | F03 / FC03 / M05 / MC04 | word clips with no text |
|---|---|---|---|---|---|---|
| none | 19.51 | 9.15 | 41.55 | 28.97 | 33.78 / 8.53 / 56.26 / 9.91 | 125 |
| 0.5 s after | 19.23 | 8.88 | 41.25 | 28.0 | 33.18 / 8.72 / 56.52 / 9.07 | 82 |
| 0.5 s before and after | 19.18 | 8.72 | 41.41 | 28.08 | 32.93 / 8.68 / 57.46 / 8.75 | 50 |

A paired bootstrap over prompts (1000 resamples) gives a 95 % interval of -0.54 to -0.01 for padding after and -0.79 to +0.13 for both sides. On the 1043-clip subset, where the idea was first tried, padding after improved all four speakers; on the full set FC03 and M05 get slightly worse, so that was noise. The hypothesis is partly right: empty results fall by a third (after) to 60 % (both sides), but dysarthric speakers gain little. The firmware's button path now appends 0.5 s before the full-context pass (`PTT_PAD` in `mic.c`), which costs about 0.3 s per utterance. A real button press may behave differently from TORGO's clips; this has not been measured on the board.

**The training mix.** TORGO's dysarthric training speakers are 3.89 h (F01 0.16, F04 0.61, M01 0.82, M02 0.79, M03 0.71, M04 0.80). With `torgo_train` listed twice next to 100.6 h of LibriSpeech, as in every run from atyp2 to big5, dysarthric speech is 6.5 % of the training audio. 4000 steps at `--batch_sec` 150 are at most 167 h of audio, so each dysarthric clip is seen about 3 times. We had never varied this.

**big6** (`train/runs/big6.cmd`): big4 with those dysarthric clips listed 8 more times (`torgo_dys_x8.jsonl`), 26 % of 150 h. Nothing else changes, so the difference measures the mix. Took about 3350 s for training.

| WER %, TORGO test, engine + LM, new grid | big2 | big4 | big6 |
|---|---|---|---|
| full context, all | 19.75 | 19.51 | **17.74** |
| ... control / dysarthric | 9.21 / 42.15 | 9.15 / 41.55 | **8.89 / 36.56** |
| ... sentence / word | 15.99 / 29.36 | 15.82 / 28.97 | **14.2 / 26.79** |
| ... F03 / FC03 / M05 / MC04 | 33.81 / 8.82 / 57.92 / 9.69 | 33.78 / 8.53 / 56.26 / 9.91 | **29.77 / 8.51 / 49.4 / 9.36** |
| chunk 32, all | 25.14 | 25.76 | **23.86** |
| ... control / dysarthric | 12.9 / 51.17 | 13.51 / 51.8 | 13.01 / **46.92** |

| WER %, float, greedy (`eval_manifest.py`), full / chunk 32 | big4 | big6 |
|---|---|---|
| TORGO test | 20.68 / 28.09 | **19.18 / 26.17** |
| LibriSpeech test-clean | **3.99** / 6.44 | 4.08 / **6.39** |
| UASpeech B2 | **63.58 / 66.98** | 66.25 / 70.76 |
| ... control | **36.85 / 39.0** | 39.57 / 42.71 |
| ... dysarthric | **86.78 / 91.25** | 89.41 / 95.09 |

**Reading this.** On TORGO, every test speaker improves and dysarthric speakers gain 5 points in the engine. LibriSpeech is unchanged. UASpeech gets worse by about 2.6 points at full context, for its control speakers as well. One seed each. Note also that big2 to big6 validation sets contain test speakers, so the choice to keep following this direction used test information; see section 6.

**How much is memorized prompts?** `train/score_unseen.py` and `train/score_vocab.py`. **5168 of the 5214 test clips (99 %) use a prompt that is also in TORGO's training set**, and 10758 of the 11334 training clips use a prompt that is also in the test set (9.33 h of 9.8 h). The 46 test clips with unseen prompts (247 words) point the same way for dysarthric speakers (big4 31.54, big6 28.86, float, full context; 149 words) but are too few to conclude from. Removing shared prompts from training is not possible either: it would leave 0.47 h. On UASpeech, single words let us split by whether the word occurs in TORGO's training prompts, but big6 is worse in both parts, for both groups (in vocabulary, control 28.06 to 31.54, dysarthric 71.45 to 72.94; out of vocabulary, control 47.2 to 49.02, dysarthric 104.87 to 108.83, full context). So UASpeech's loss is not simply TORGO's vocabulary being favored. It could be specialization to TORGO's speakers or recording setup, or the smaller share of LibriSpeech. This run cannot separate them.

**Runs stopped.**
- **big5** (`train/runs/big5.cmd`): big4 with 12000 steps at 1e-4. Paused by us at about step 5300 to free the GPU for the mix experiment; no result. Validation had reached 6.00 / 6.93 at step 4000, against big4's 5.73 / 7.07 at the same step, on a schedule not comparable at that point. It can resume from its checkpoint.
- **big7** (`train/runs/big7.cmd`): big6 with simulated dysarthria (`--p_dys 0.3 --dys_severity 0.2,1.0`). Stalled at step 200: simulation stretches clips up to 1.8 times, the 24-layer model then ran out of GPU memory (allocator warning in the log), and progress stopped after 942 s of training time. Stopped by us; no result about simulation. It needs a smaller `--batch_sec`.

**Finding 25:** the share of dysarthric audio in training is a large lever we had left at 6.5 %. Raising it to 26 %, with the same steps and everything else unchanged, lowers WER in the engine at full context from 19.51 to 17.74, and from 41.55 to 36.56 for dysarthric speakers. It is the largest gain from training since the 24-layer model. On an unseen corpus the same change loses about 2.6 points, so the gain must not be read as better recognition of dysarthric speech in general.

**Finding 26:** TORGO's test and training speakers read almost entirely the same prompts (99 % of test clips). Every TORGO result in this log measures new speakers on familiar prompts. An overlap-free report as planned in section 6 is not possible with this split.

**Finding 27:** the language model's weight and bonus cannot be usefully tuned further, since the optimum changes with the speakers; a beam wider than 4 gains at most 0.23 points; padding silence gains about 0.3. All are small next to the training mix or full context.

### E14. First speech at the board, its noise, and where the wait goes (2026-11-01)

**Live use.** Firmware with the button on GPIO21 and big6_q121. A first run in a room with several people talking at a distance gave poor text, with speech at p90 −38 to −45 dBFS over a noise floor of −57 dBFS. No crash, and the physical button worked. An early build crash-looped because `api_button` ran before `api_start` (null mutex) when the pin read low at startup; fixed, and a pin low at startup is now reported, not treated as a press. A press made while a final pass was running waited for it, so delays stacked (a 2.4 s utterance waited 4.6 s); a waiting press now takes the streaming text.

**Reading at 15 cm.** `esp32/tools/read_test.py` (Harvard sentences, phrases, words; saves the microphone audio before gain control). One typical speaker, one session, stopped after 4 of 35 prompts, so this is not a score: three sentences perfect in both live and final text; on the fourth the words of the prompt were all recognized after what looks like a restart by the reader (not confirmed by listening). Speech blocks p90 −29 to −38 dBFS, noise −58 dBFS, about 20 to 29 dB apart.

**Microphone path.** The ES8311 is already at its maximum analog gain (30 dB); digital gain cannot improve the signal-to-noise ratio. The NS4150B amplifier only drives the speaker. The noise is not white: per FFT bin it is highest at 200 to 800 Hz and about 20 dB lower above 5 kHz.

**Level control** (big6_q121, full context + LM, every 5th TORGO test clip, 1043 clips; speech −42 dBFS, white noise −57 dBFS):

| Level control | all | control | dysarthric |
|---|---|---|---|
| clean recordings | 17.74 | 8.91 | 34.85 |
| none | 24.87 | 12.21 | 49.40 |
| firmware AGC, starting cold | 24.17 | 12.49 | 46.80 |
| firmware AGC, settled | 24.02 | 11.99 | 47.34 |
| whole clip normalized | 24.39 | 12.38 | 47.67 |

**Measured noise and front-end treatments** (`train/room_sim.py`: noise of the spectrum measured on the board's recordings at −58 dBFS, fixed gain; same clips). Fixed before scoring. Spectral subtraction with the noise spectrum known, 32 ms frames: nr1 over-subtraction 1, floor −10 dB; nr2 over-subtraction 2, floor −20 dB. hp: 200 Hz second-order high-pass. Stationary noise of known spectrum is the best case for noise reduction. CI: bootstrap over clips of the paired difference, 95 %.

| Speech level (distance, roughly) | treatment | all | control | dysarthric | difference from none (CI) |
|---|---|---|---|---|---|
| −35 dBFS (reading close) | none | 18.18 | 8.40 | 37.13 | |
| −42 dBFS | none | 20.47 | 10.08 | 40.61 | |
| | nr1 | 20.51 | 9.64 | 41.59 | +0.04 (−0.74..0.82) |
| | nr2 | 21.58 | 10.64 | 42.78 | +1.11 (0.11..2.12) |
| | hp | 21.03 | 10.03 | 42.35 | +0.56 (−0.04..1.14) |
| −50 dBFS | none | 31.08 | 18.43 | 55.59 | |
| | nr1 | 28.31 | 16.19 | 51.79 | −2.77 (−3.94..−1.69) |
| | nr2 | 28.86 | 15.85 | 54.07 | −2.22 (−3.58..−0.98) |
| | hp | 30.97 | 18.66 | 54.83 | −0.11 (−0.94..0.77) |

Chunk 32 at −42 dBFS: 27.01 (control 14.45, dysarthric 51.36), against 23.87 clean.

**Finding 28:** on this microphone, level control barely matters and the signal-to-noise ratio does. Real, low-frequency noise costs far less than the white noise we first simulated (2.7 points at −42 dBFS, not 6 to 7), and little at reading distance (0.4 overall, 2.3 dysarthric).

**Finding 29:** spectral subtraction helps only at low signal-to-noise ratio: 2.8 points at about 8 dB, nothing at about 16 dB, and the stronger setting hurts there. A high-pass filter does not help. Not shipped: this is simulated stationary noise, and no recording at a distance with known prompts exists yet.

**Live chunk length, big6_q121, same 1043 clips, streaming + LM** (`--chunk`, left 128; big6 was trained on chunks 32 and 64 only). CI against chunk 32.

| Chunk | all | control | dysarthric | sentence | word | CI |
|---|---|---|---|---|---|---|
| 16 = 0.64 s | 27.75 | 15.46 | 51.57 | 23.54 | 38.36 | |
| 24 = 0.96 s | 25.98 | 15.97 | 45.39 | 22.15 | 35.63 | |
| 32 = 1.28 s (shipped) | 23.87 | 12.94 | 45.06 | 20.19 | 33.16 | |
| 48 = 1.92 s | 20.69 | 10.70 | 40.07 | 17.76 | 28.09 | −4.42..−2.06 |
| 64 = 2.56 s | 19.77 | 9.64 | 39.41 | 16.47 | 28.09 | −5.17..−3.14 |
| full utterance | 17.74 | 8.91 | 34.85 | 14.04 | 27.05 | |

**Finding 30:** live text can gain 3 to 4 points by updating less often; shorter chunks than 32 lose more than the wait they save. On chip cost, chunks other than 32 are not yet measured.

**Where the wait goes.** The two passes run one after the other, so slowing live text alone does not shorten the final pass. From the chip's profile (`docs/results/p4_bench_big2q121_full_2c.log` minus the streaming run) the front end is about 17 % of a full pass, and streaming has already computed it. `tasr_nemo_stream_keep` and `tasr_nemo_transcribe_from` start the pass after the kept rows; only rows computed from real audio before the stream was finished count, since finishing computes the last rows as if the audio stopped there. Host check (`tasr_cli --reuse`, 60 clips, each with a stream left running and a stream finished): logits differ by exactly 0 and the text is the same in all 120 runs; host CPU 139.5 → 105.8 s (finished), 108.4 s (not finished). Not yet confirmed with speech on the board; a press with only room noise took 6.0 s for 9.4 s of audio, not comparable with speech because the decoder has less to do.

**The last words.** Text used to stop at the last whole chunk until the final pass ended. From 2 s of audio the firmware now finishes the stream first (at most one chunk of work) and shows it. Clip played at the board, 9.5 s of audio: all words on screen 0.16 s after release, final text 7.37 s after release (before the front-end reuse).

**big8** (`train/runs/big8.cmd`, dysarthric audio share 38 %): engine, full + LM 17.33 (big6 17.74; CI −0.80..−0.07), dysarthric 35.43, chunk 32 23.26. The gain is one speaker: M05 49.40 → 45.81 (CI −5.44..−1.79); F03 and both controls do not change significantly. Float: LibriSpeech 4.08 → 4.24, UASpeech 66.25 → 67.51. Not deployed: big6 stays on the board and the mix stops here.

**Finding 31:** the dysarthric share has stopped paying. Beyond 26 % the gain is on one test speaker, on familiar prompts, and unfamiliar material gets worse.

**Reference point.** The Interspeech 2025 challenge winner (Takahashi et al.) fine-tuned Parakeet-TDT 1.1B, about 1.1 billion parameters and about 4 GB, on 290 h of real dysarthric speech, decoded with beam 40 on Test1 and reached 5.97 % on Test1 and 8.11 % on Test2; the baseline, Whisper large v2, reached 14.97 and 17.82. Off the shelf the same model scored 10.40 on Test1. Test sets differ from ours and are not public, so these numbers cannot be set against TORGO's.

**Waveform simulator.** `train/dysim.py` simulates 14 conditions on a WORLD analysis (about 10 x real time per process on the training PC). Profiles are hand-set from clinical descriptions and not yet calibrated. Results follow in E15.

### E15. Calibrating the waveform simulator, and a 100-hour pilot set (2026-11-02)

**Method.** Each round regenerates 4 h of LibriSpeech test-clean (1947 clips, seed 11, severity uniform 0.2–1.0, one condition per simulated speaker), measures it with `train/dys_stats.py` against the E5 targets (dysarthric TORGO speakers against controls), and scores it with big6_q121 + LM in the engine, full context and chunk 32. Test-clean is never used for training, and the clean set scores about 4 % with the same model.

Targets from E5, as a ratio or difference against typical speech: speaking rate × 0.70–0.94, pause share +0.12 to +0.18, spectral flux × 0.93–1.02, high/low energy 1 to 14 dB lower, syllable rhythm strength × 0.43–0.69.

**Round 1** (the profiles as first written). Rate and rhythm were in range. Pauses were short of the target (+0.01 to +0.06). Flux was too low for the slowest conditions (ALS and CP about × 0.67): interpolated frames are smoother than real slow speech. Spastic and ataxic speech came out brighter than the source, not duller. WER, full context: 10.7 % overall (mild 5.8, moderate 10.7, severe 16.9). Flaccid, unilateral upper motor neuron, dysphonia, electrolarynx and whisper scored under 7 % even at severe, so those profiles were too weak.

**Round 2 changes:**
- Frame-level noise on interpolated spectra.
- Milder vowel blur and transition blur.
- For the eight dysarthria types, a pause share of at least s × U(0.3, 0.8) and a further s × U(4, 12) dB high-frequency loss.
- Stronger flaccid and unilateral profiles.

Measured (lsclean = the unprocessed clips):

| Set | rate | pause | flux | hi/lo dB | rhythm |
|---|---|---|---|---|---|
| lsclean | 15.82 | 0.156 | 4.055 | −2.90 | 0.340 |
| ALS | 10.74 | 0.280 | 3.299 | −11.29 | 0.133 |
| CP | 10.75 | 0.335 | 3.330 | −9.74 | 0.123 |
| spastic | 12.96 | 0.323 | 3.588 | −3.20 | 0.167 |
| ataxic | 14.64 | 0.263 | 3.926 | +0.35 | 0.193 |
| flaccid | 13.73 | 0.293 | 3.669 | −12.50 | 0.162 |
| hypokinetic | 15.62 | 0.323 | 3.971 | −8.03 | 0.230 |
| hyperkinetic | 14.03 | 0.372 | 3.846 | −5.72 | 0.138 |
| unilateral UMN | 12.77 | 0.321 | 3.690 | −7.96 | 0.170 |

Pauses are now in range or above it (+0.11 to +0.22). Most types are duller, but ataxic speech is still brighter. Flux has improved but is still low for ALS and CP (× 0.81). ALS and CP are slightly too slow (× 0.68) and too even (× 0.36–0.39).

WER, big6_q121 + LM, 1947 clips:

| | full | chunk 32 |
|---|---|---|
| all | 13.3 | 19.6 |
| mild / moderate / severe | 7.1 / 13.6 / 21.0 | 11.2 / 19.7 / 30.4 |
| CP | 26.9 | 38.5 |
| hyperkinetic | 21.3 | 31.0 |
| ALS | 18.2 | 27.1 |
| spastic, hypokinetic, uumn, stutter, flaccid | 12.8–13.9 | 16.9–21.5 |
| apraxia, ataxic, esophageal | 10.1–10.7 | 15.8–17.1 |
| whisper, electrolarynx, dysphonia | 4.3–6.7 | 6.1–10.4 |

**Finding 32:** a signal-level simulator can match real dysarthric speech on rate, pausing, rhythm and spectral tilt, but at severity 1 it costs our model about 21 %. The real severe test speaker costs about 45–50 %. What is missing is probably the articulation itself: wrong or omitted sounds rather than weakened ones, which a vocoder that keeps the source phones cannot produce. Voice-source conditions (whisper, electrolarynx, dysphonia) barely trouble a model trained with noise augmentation.

**Pilot set.** Rather than calibrate further on statistics, the pilot asks the question that matters: does the simulated audio help on real speakers?
- `dysim.py gen` on LibriSpeech 100, 100 source hours, seed 21, severity 0.2–1.4. Above 1, the speaking rate is held at no less than 0.4 × and retime noise is 0.8.
- `train/runs/big9.cmd` trains on big6's mix plus this set, with the real dysarthric share held near big6's and the same compute.
- It is judged on TORGO test (held-out speakers), UASpeech and LibriSpeech. On TORGO test, 99 % of the prompts are also in training (E13), so UASpeech is the check on unfamiliar words.

The set came out at 166 h of audio (28376 clips) from the 100 source hours. Slowing makes 65 % of the clips longer than the trainer's usual 20 s limit, so big9 uses `--max_dur 32`, which keeps 120 h (24331 clips, 86 %, every condition represented). It also takes big7's batches: 6000 steps of 100 s, the same audio as big6.

**big7** (`train/runs/big7.cmd`: big6 plus the older mel-level simulation, `simdys.py`, on 30 % of LibriSpeech clips, 6000 steps of 100 s).

Float, greedy, against big6:

| Set | big6 | big7 |
|---|---|---|
| TORGO test | 19.18 | 18.46 |
| UASpeech | 66.25 | 65.46 |
| LibriSpeech test-clean | 4.08 | 4.19 |

Engine (int8, LM), TORGO test, paired bootstrap against big6:

| | big6 | big7 | CI of the difference |
|---|---|---|---|
| full context | 17.74 | 17.03 | −1.08..−0.35 |
| chunk 32 | 23.86 | 23.31 | −0.99..−0.07 |
| M05 (severe), full | 49.40 | 46.14 | −5.06..−1.44 |
| FC03 (control), full | 8.51 | 7.74 | −1.23..−0.31 |

F03 and MC04 do not change.

**Finding 33:** at 24 layers, the older mel-level simulation now helps slightly, where at 16 layers (E4) it did not. Unlike a larger real dysarthric share (big8, Finding 31), it does not cost UASpeech. big7 replaces big6 on the board. It was flashed with `flash_p4.py --only model` and verified; it boots and loads (20381216 bytes into PSRAM in 1.9 s, ready 3.2 s after boot). Board text was not checked with speech.

**Round 3** (`sound_errors` in `dysim.py`): wrong sounds rather than weak ones. For the eight dysarthria types and apraxia, three errors were added:
- consonant runs left out with probability s × U(0.05, 0.25);
- fricatives moved in place (envelope scaled by 0.6–1.6) with probability s × U(0.1, 0.45);
- weak voiced stretches made voiceless with probability s × U(0.1, 0.4).

Same 1947 test-clean clips:
- WER, full context: 13.3 → 15.5 % overall; mild 8.3, moderate 16.3, severe 24.1.
- WER, chunk 32: 21.9 % overall, severe 33.3.
- Flux for ALS and CP rose to × 0.88–0.89, close to the target.
- Ataxic speech is still brighter than the source.

**Off-the-shelf Parakeet on our test sets** (`train/eval_parakeet.py`, separate NeMo environment, bf16, model's own greedy decoding, no LM, the same text normalization for reference and output). nvidia/parakeet-tdt-1.1b is the model the Interspeech 2025 Speech Accessibility Project winner fine-tuned. Ours: float, greedy, no LM (eval_manifest.py).

| Model | Parameters | TORGO test | TORGO dysarthric | M05 | UASpeech | UASpeech control / dysarthric | LibriSpeech test-clean |
|---|---|---|---|---|---|---|---|
| parakeet-tdt-1.1b, untuned | 1070 M | 23.14 | 43.21 | 53.99 | 60.80 | 26.31 / 90.73 | 1.52 |
| parakeet-tdt_ctc-110m, untuned | 115 M | 24.40 | 45.72 | 54.59 | 66.62 | 38.52 / 90.99 | 2.62 |
| big7 (ours, fine-tuned on TORGO train) | 19 M | 18.46 | 39.11 | 66.25 | 65.46 | 40.74 / 94.86 | 4.19 |
| big7 in the engine, int8 + LM | 19 M | 17.03 | | 46.14 | | | |

UASpeech is unseen by all three models, so it is the fair column. There the 56 times larger model is 4.7 points better, almost all of it on control speakers (26 vs 41 %). On dysarthric UASpeech speakers every model is at about 91–95 %. Most of the remaining errors on control speakers are homophones of isolated words ("their"/"there", "to"/"two", "write"/"right"), which no model can resolve without context.

On TORGO, our 19 M model beats both because it was fine-tuned on TORGO's prompts (99 % shared with test, E13); the untuned models have not seen them. Parakeet gives empty output for 2.9 % of UASpeech clips (206 of 7136), probably digits that the normalization deletes, so its UASpeech number may be slightly pessimistic.

**Finding 35:** size alone buys little on dysarthric speech. Ten times the parameters (115 M → 1.1 B) gains 1.3 points on TORGO and 0.3 on dysarthric UASpeech speakers. Fine-tuning on in-domain speech, even 19 M parameters on 4 hours, gains more on TORGO than 56 times the size, though there the test prompts were seen in training. On unseen words (UASpeech), the large model stays ahead. The challenge winner's 8 % comes from 290 h of real dysarthric training speech on a 1.1 B model; the model alone, untuned, scores 10.4 % on their Test1.

**Finding 34:** sound errors move simulated speech toward real dysarthria on both the statistics and the model's errors, but severe simulated speech still costs about half what real severe speech costs. The pilot set (sim100) was generated with round 2, before these errors existed.

**big9 result** (`train/runs/big9.cmd`: big6's mix + sim100, 120 h of round-2 simulated speech, 6000 steps of 100 s).

Engine, TORGO test, against big7 (paired bootstrap):

| | big7 | big9 | CI of the difference |
|---|---|---|---|
| final, all | 17.03 | 18.41 | +0.94..+1.84 |
| live (chunk 32), all | 23.31 | 25.09 | +1.28..+2.26 |
| final F03 | 29.77 | 31.81 | +0.82..+3.49 |
| final M05 | 46.14 | 46.14 | −1.76..+1.90 |
| final FC03 / MC04 | 7.74 / 9.14 | 9.16 / 10.53 | both worse |

Float greedy: UASpeech 64.81 (big7 65.46), with its dysarthric speakers at 92.0 against 94.86, the best of any model so far. LibriSpeech 4.21 (4.19).

**Finding 38:** big9 is worse on TORGO, mostly for the control speakers, and better on unseen dysarthric speech (UASpeech). The comparison is confounded. Adding 120 h with the same number of steps halved how often the model saw TORGO's control speakers and LibriSpeech: their share of the 302 h set is about half of what it was in big6's 150 h. The dysarthric clips were listed twice to keep their share, which matches the pattern: controls worse, dysarthric flat, unseen dysarthric better. The next simulated run must hold every real set's exposure fixed by scaling the steps with the set's size. big7 stays on the board.

### E16. The simulator as the main route: sound-level errors from phone alignments (2026-11-02, built, not yet measured)

**Why.** Finding 35: the large model's advantage comes from hundreds of hours of real dysarthric speech, which we cannot get. Finding 34: our simulation is half as hard as real severe speech, and what it lacks is wrong sounds. Without knowing where each sound is, the simulator can only weaken frames. With a phone alignment it can say one sound in place of another.

**Phone alignments.** Lugosch et al. (2019) published Montreal Forced Aligner alignments (words and ARPAbet phones) of all of LibriSpeech, about 623 MB, at zenodo.org/records/2619474. `dysim.py gen --align_dir` finds each clip's TextGrid by utterance name.

**`train/phonesim.py`** applies the errors on the WORLD frames before every other trait, while the alignment still matches. Its rates are per sound at severity 1, set by hand from the clinical literature, and are to be calibrated.

| Error | Examples |
|---|---|
| Substitution | voicing (b → p), place (s → sh, k → t), stops to fricatives (t → s), fricatives to stops (s → t), gliding (r → w), nasal (b → m) |
| Distortion | part of the way toward a substitute (the most frequent kind) |
| Deletion | any consonant; word-final consonants and one consonant of a cluster most often |
| Insertion | schwa inside a cluster |
| Vowels | reduction toward schwa, diphthongs flattened (ay → aa), tense to lax (iy → ih) |

- **How the new sound is made:** from the speaker's own version of it in the same clip when there is one, otherwise by a rule. Rules: move the noise peak for a place change; draw out the burst for a stop that becomes a fricative; close and add a burst for a fricative that becomes a stop; a nasal murmur; a neutral envelope for vowels. Voicing follows the target sound.
- **Conditions:** each condition weights the error kinds. Flaccid: stops leak to fricatives and nasals. Spastic: voicing errors. Ataxic: mostly vowels. Hypokinetic: stops to fricatives, final sounds lost. Apraxia: inconsistent substitutions and inserted vowels.
- **Transcript:** stays the intended words.
- **Testing so far:** only a code-path test on one bench clip with a hand-spread alignment. Every condition runs. Severity 1 gives about 12 errors in 50 sounds, and severity 0.6 fewer and milder ones.

**Calibration without leakage** (`train/calibrate_sim.py`):
- The model is the starting model of every run (base24x), which has heard no dysarthric speech.
- It scores TORGO's dysarthric training speakers (never the test speakers F03 and M05) and a simulated set made from LibriSpeech test-clean.
- The tool reports the severity at which the simulation is as hard as each real speaker, and suggests a training range that spans them.

`eval_manifest.py` now reads packed audio. A simulated manifest's "wav" names the clean source clip, so before this change it would have scored the clean audio. Earlier simulated scores went through the C engine on extracted audio and are not affected.

**Realism work** (all built 2026-11-02; none yet run on real data):
1. *Consistent speakers.* Each simulated speaker gets trouble sounds (lognormal weights per sound) and a habitual substitute for each sound. They are used 70–90 % of the time, 25 % for apraxia, whose errors vary between attempts. Test: one simulated spastic speaker in three clips of the same text repeated r → w, s → z, k → hh and others.
2. *Measured error rates* (`train/real_errors.py`). A phoneme recognizer (wav2vec2-xlsr-53-espeak-cv-ft) is aligned against CMUdict pronunciations for TORGO's training speakers. Control speakers give the recognizer's floor, and each dysarthric speaker's errors above it become a table.
   - With `--real_errors`, phonesim draws substitutions and deletions from one measured speaker's table, scaled by severity. Distortion, which a recognizer cannot count, stays hand-set.
   - TORGO only: UASpeech's test block is spoken by its training speakers, so fitting on UASpeech would leak.
3. *Voice conversion* (`train/knnvc_gen.py`, kNN-VC). to_dys: LibriSpeech sentences in a real dysarthric training speaker's voice. from_dys: real dysarthric training recordings re-voiced by hundreds of LibriSpeech speakers, which keeps the real timing and articulation errors.
4. *Realism meter* (`train/realism.py`, WavLM features). Two measures:
   - A probe trained on control vs dysarthric within TORGO's training speakers (same channel and prompts) rates how dysarthric each simulated set sounds, against its clean source.
   - A discriminator compares real dysarthric speech with simulation made from TORGO's own control speakers (same channel and prompts): 0.5 AUC means indistinguishable.
5. *Details* in dysim: audible breaths in inserted pauses, fatigue (duller and breathier toward the end), hard voice onsets (spastic, CP, ALS, hyperkinetic), and sound errors that spill into the neighbouring frames.

**Plan:**
1. Calibrate on test-clean with real alignments.
2. Generate from all 960 h of LibriSpeech (100 + 360 + other-500), about 1600 h of output.
3. Train longer than 6000 steps, so the set is seen at least once.
4. Judge only on held-out real speakers (TORGO test, UASpeech).

**First runs of version 4** (2026-11-02, `v4home.cmd` on the training PC).

*Measured error tables* (`real_errors.py`, wav2vec2-xlsr-53-espeak-cv-ft against CMUdict, TORGO training speakers). The recognizer's own phone error on the five control speakers is 17.8 % (16.1–25.1). Excess error per dysarthric speaker:

| Speaker | Phone error | Excess over controls | Most frequent errors above the floor |
|---|---|---|---|
| M03 | 17.5 % | −0.3 | none clearly above the floor |
| F04 | 19.7 % | 1.8 | oy → ao, aw → r |
| M02 | 57.0 % | 39.1 | er → ah, aw → ae, dh → d, oy → ay, iy → ih |
| M04 | 62.0 % | 44.2 | dh → d 39 %, oy → ao, p → b 20 %, uh → ah, er → ah |
| M01 | 62.1 % | 44.3 | dh → d 43 %, uh → ah 34 %, f → v 26 %, t → d 23 %, ng → n |
| F01 | 67.3 % | 49.4 | th → t 48 %, sh → s 43 %, ch → k 33 %, jh → g 28 % |

Stopping (dh → d, th → t), voicing (p → b, t → d, f → v), place (sh → s) and vowel reduction (er → ah, uh → ah) are what the hand-set profiles assumed.

*Calibration* (`calibrate_sim.py`). The model is base24x, untrained on dysarthric speech, full context. The real side is TORGO's dysarthric training speakers; the simulated side is simval4: 1947 test-clean clips with phone alignments and the measured tables, severity 0.2–1.4.

The first attempt scored 100 % on everything, because `v4home.cmd` passed the base weights as `--init`. eval_manifest.py loads model weights only from `--ckpt`; rerun with `--ckpt init.pt`.

| | all prompts | sentences only |
|---|---|---|
| real mild: M03, F04 | 15.3, 22.0 | 10.2, 14.6 |
| real severe: F01, M02, M04, M01 | 79.1, 84.1, 89.8, 93.9 | 73.3, 75.6, 85.3, 82.2 |
| simulated, severity 0.2–0.4 / 0.4–0.6 / 0.6–0.8 / 0.8–1.0 / 1.0–1.2 / 1.2–1.4 | 17.1 / 24.3 / 35.4 / 52.7 / 60.4 / 63.0 | (same set) |

By condition: CP 80.1, ALS 64.4, hyperkinetic 61.1, flaccid 57.7, hypokinetic 54.5, UUMN 49.3, spastic 49.3, ataxic 48.1, apraxia 30.1, stutter 21.9, esophageal 15.0, whisper 7.9, electrolarynx 5.4, dysphonia 4.8.

*Realism meter* (`realism.py`, WavLM-base-plus layer 6, mean and spread, logistic regression):
- Probe, control vs dysarthric, leaving one training speaker out: AUC 0.845. The mild M03 and F04 are judged dysarthric 28 % and 25 % of the time, the severe four 98–100 %, the controls 8–45 %.
- TORGO control speech after simulation: judged dysarthric 44.3 %, against 3.4 % for the same clips unprocessed.
- LibriSpeech after simulation: 56.3 %, against 31.2 %.
- Discriminator, real dysarthric against simulated from TORGO controls: AUC 1.000 for every condition. The simulation is trivially told apart; the test of the vocoder alone follows.

*Voice conversion trial:* the first two attempts failed: `torch.hub` could not resolve the default branch, then SSL certificate verification failed on the PC. Fixed by naming the branch (`bshall/knn-vc:master`) and setting `SSL_CERT_FILE` to certifi's bundle. `knnvc_gen.py from_dys --per_source 2 --n_targets 40` then re-voiced TORGO's dysarthric training clips with 40 LibriSpeech voices: 7.7 h in all, 5.6 times faster than real time on the RTX 3060. Its measurement follows below.

**Finding 39** (*corrected by Finding 43:* the bins below averaged in the voice conditions; with the dysarthria conditions alone the simulation reaches 92 % at severity 1.2–1.4 and spans the severe speakers too): with sound-level errors and measured error tables, the simulator spans mild real speakers (severity 0.3–0.44 matches M03 and F04) and reaches 60–63 %. Real severe speakers cost 73–85 % on sentences, so the gap at the severe end is about three quarters closed, from about half (Finding 34). The measured errors agree with the clinical descriptions the profiles were built on. But a linear classifier separates simulated from real speech perfectly, so the simulation carries a strong fingerprint the model could learn as a shortcut.

*The vocoder alone* (`vocoder.cmd`, `voccost.cmd`). 1276 clips from TORGO's control training speakers, passed through WORLD with no traits (severity 0, which turns every trait off; checked in `profile`), set against the same clips unprocessed:

| | original | WORLD analysis + resynthesis |
|---|---|---|
| base24x, full context, WER | 18.73 % | 29.92 % |
| big7, full context, WER | 7.72 % | 17.41 % |
| realism probe, judged dysarthric | 2.0 % | 17.0 % |
| discriminator, original vs resynthesized (speakers held out) | | AUC 1.000 |

Every speaker gets worse (base: FC01 34.1 → 42.9, FC02 14.8 → 21.1, MC01 19.6 → 27.4, MC02 24.8 → 34.9, MC03 16.5 → 38.8). big7 never saw WORLD output (its simulation is the older mel-level one, Finding 33) and loses 9.7 points.

**Finding 40:** the vocoder, not the dysarthria, is a large part of what the simulation contains. Resynthesis with no change at all adds 10–11 points of WER for both models (more than doubling big7's error on clean speech), is told from the original perfectly, and by itself moves the probe a sixth of the way toward "dysarthric". Two consequences: (1) part of the "difficulty" measured in calibration (Finding 39) is vocoder damage, so the severity scale is miscalibrated; (2) a model trained on WORLD output can learn the fingerprint as a shortcut for "dysarthric" (Finding 39's AUC 1.000), and what it learns on it need not carry over to real recordings. This may explain why simulated data has helped so little (Findings 11, 33, 38). (*Limited by Finding 44:* on clean LibriSpeech, the source of every simulated training set, resynthesis costs only 0.4 points.)

*Rendering without the vocoder* (`dysim.py --render transparent`, the new default). The traits are still computed on WORLD frames, but the output is the original recording changed only where the traits change it:
- timing and pitch: pitch-synchronous overlap-add of the original waveform (TD-PSOLA, Moulines & Charpentier 1990). Every change of timing (phone deletions and insertions, rate, pauses, repetitions) is tracked as each output frame's position in the original. Where neither pitch nor rate changes, grains are placed in phase with the original's own pitch marks, so no change gives back the input;
- envelope and voicing: a time-varying filter, the ratio of the target to the original envelope, on the STFT (5 ms hop, the WORLD frame grid), plus noise shaped to the target wherever the target is noisier (WORLD's convention: aperiodic power = envelope × aperiodicity²). This renders breathiness, devoicing, lenition, fatigue and breaths;
- WORLD only where the recording has nothing to offer: frames made voiced that were unvoiced in the source, and the electrolarynx and esophageal voice sources.

At severity 0 the output is 33 dB from the input (WORLD: −3 dB, its phase is its own); the filter alone is exact (104 dB). Compute is about 1.4 times WORLD's (1.0 s against 0.7 s for 10 s of audio, one core).

Sound substitutions are spliced from the recording itself (`phonesim.py`): a full substitution copies the clip's own longest instance of the target phone (at least 3 frames), stretched to fit, and an inserted schwa loops the clip's own AH frames. Only when the clip has no instance of the phone does the frame-level fallback (envelope interpolation toward the target) run.

A bug found on the first large run: the in-phase snapping could move a grain back before the previous one where the timing jumps backward (a repetition), and the loop then cycled forever. One worker spent over 2 hours on one clip (seen with `py-spy dump`: inside `psola`). Fixed by never placing a grain at or before the previous one, which bounds the loop; 144 stress runs over all conditions on an aligned clip all finished.

*Voice conversion trial measured* (`vccheck.cmd`): 7800 clips, TORGO's dysarthric training recordings re-voiced by 40 LibriSpeech speakers, against the same recordings unconverted. Base24x, full context, WER by source speaker:

| | F04 | M03 | F01 | M02 | M04 | M01 | all |
|---|---|---|---|---|---|---|---|
| original | 22.0 | 15.3 | 79.1 | 84.1 | 89.8 | 93.9 | 61.3 |
| kNN-VC, re-voiced | 32.0 | 21.9 | 89.6 | 111.3 | 117.1 | 121.8 | 80.3 |

- Realism probe: judged dysarthric 96.1 %, against 98.1 % for the unconverted sources. The dysarthria survives conversion as far as the probe can tell.
- Discriminator, real dysarthric against re-voiced: AUC 0.983. Easier to mistake for real than WORLD output (1.000), but still separable.
- The mild speakers lose 7–10 points, about what the WORLD vocoder costs clean speech (Finding 40); the severe speakers go above 100 % because the model inserts words into the converted babble.
- big7 on the re-voiced clips: 51.1 % (F04 16.5, M03 9.3, F01 71.0, M02 75.6, M04 78.0, M01 73.7). big7 trained on the unconverted clips (27.4 % on them), so this is not a held-out score; it shows the conversion moves them far from what the model memorised.

**Finding 41:** kNN-VC keeps what makes speech sound dysarthric (the probe rates 96 % of re-voiced clips dysarthric) and makes the real dysarthric recordings available in many new voices, but it adds its own damage, about 7–10 points on mild speakers, similar to the vocoder's. As training data it would carry the same risk as WORLD: a fingerprint the model could learn instead of the dysarthria. It is not used in training until a run shows it helps on held-out speakers.

*WORLD against transparent, measured* (`rendercmp.cmd`, `rendercmp2.cmd`). Base24x, full context.

- Same TORGO control clips, conditions, severities (0.2–1.4) and seeds (3786 clips, 8 dysarthria conditions), only the renderer differs: WORLD 60.63 %, transparent 50.59 %. The 10-point difference is what the vocoder alone cost (Finding 40), so with the new renderer the difficulty is the simulated dysarthria's.
- Recalibration on test-clean (simval4_tr, 1947 clips, alignments, measured tables, now with splicing), by severity bin 0.2–0.4 … 1.2–1.4: 17.7 / 25.0 / 35.1 / 51.3 / 59.0 / 61.1 (WORLD, simval4: 17.1 / 24.3 / 35.4 / 52.7 / 60.4 / 63.0). Mild real speakers match at severity 0.30 (M03) and 0.42 (F04); the severe four (79–94 %) are still beyond the hardest bin.
- Realism meter, same TORGO control clips: judged dysarthric 44.6 % with WORLD, 20.6 % transparent (sources 3.5 %). By condition, WORLD → transparent: UUMN 78.5 → 45.4, ataxic 67.4 → 43.6, ALS 66.8 → 41.1, flaccid 40.7 → 12.8, hyperkinetic 38.0 → 17.2, hypokinetic 31.9 → 15.6. Discriminator, real dysarthric against simulated: AUC 1.000 → 0.990 (0.987–1.000 by condition). Test-clean, transparent: 53.8 % (sources 31.1 %).
- Unexplained so far: on test-clean the renderer hardly changes the difficulty, while on TORGO control speech it removes 10 points. Two things changed together on test-clean, the renderer and the splicing of substitutions from real phones (a real phone in the wrong place is plausibly harder for the model than an interpolated envelope), so their effects may cancel. The vocoder's own cost was measured on TORGO control recordings only; on LibriSpeech it is not measured.

**Finding 42:** about half of what made the WORLD simulation sound dysarthric, and 10 points of what made it hard, was the vocoder. Rendered without it, the same traits are judged dysarthric half as often (20.6 % against 44.6 %) and cost 50.6 % instead of 60.6 %. The simulation is still separable from real dysarthric speech (AUC 0.99). It is no longer the vocoder's fingerprint (severity 0 is 33 dB from the input), so it is in what the traits do: by the probe, the transparent simulation sounds less dysarthric than real severe speakers (judged dysarthric 98–100 %), although by WER it is as hard (Finding 43).

*Calibration corrected* (`sevhi.cmd`, `calibrate_sim.py` fixed). The severity bins in Findings 34 and 39 and in the recalibration above averaged all 14 conditions. The voice conditions (whisper, electrolarynx, dysphonia, esophageal) cost 4–15 % at any severity and stutter and apraxia 22–30 %, so the curve flattened near 61 % and looked "too mild" for the severe speakers. TORGO's speakers are dysarthric; `calibrate_sim.py` now fits the curve on the 8 dysarthria conditions (`--conditions all` for the old behaviour), takes several simulated files, and bins up to the highest severity present. A new set, simval5_hi (566 clips, test-clean, 8 dysarthria conditions, severity 1.4–2.2, transparent), extends the curve. Base24x, full context, dysarthria conditions only:

| severity | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 | 1.0–1.2 | 1.2–1.4 | 1.4–1.6 | 1.6–1.8 | 1.8–2.0 | 2.0–2.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| transparent | 28.0 | 41.0 | 50.2 | 69.5 | 80.6 | 90.4 | 94.4 | 94.6 | 95.6 | 99.8 |
| WORLD (simval4) | 26.6 | 39.4 | 50.7 | 71.1 | 82.0 | 92.0 | | | | |

Matching severity per real training speaker, all prompts / sentences only: M03 and F04 at or below 0.3 (below the lowest bin), F01 1.07 / 0.97, M02 1.17 / 1.01, M04 1.29 / 1.20, M01 1.47 / 1.13.

**Finding 43:** the simulator was never too mild by WER; the calibration was. With only the dysarthria conditions in the curve, severity 0.3 to 1.5 spans every TORGO training speaker, mild and severe, and beyond 1.5 the model fails almost completely (95–100 %), so there is nothing more to learn from there. On LibriSpeech sentences, WORLD and transparent rendering are equally hard at every severity (within 1.6 points), unlike on TORGO's control recordings (10 points); explained by Finding 44. Training range from here: severity 0.1–1.4.

*The vocoder on LibriSpeech* (`voc_ls.cmd`). 481 test-clean clips, severity 0, base24x, full context: original 4.10 %, WORLD 4.47 %, transparent 4.08 %.

**Finding 44:** the vocoder's cost depends on the recording. On clean LibriSpeech it is 0.4 points; on TORGO's recordings (array and head microphones, single words, more noise) it is 11 (Finding 40). WORLD's pitch and aperiodicity analysis is reliable on clean read speech and not on TORGO's channel. This limits Finding 40: every simulated training set so far (sim100, big9) was made from LibriSpeech, where resynthesis did little harm by WER, so the vocoder is unlikely to be why big9 was worse; the halved real exposure (Finding 38) remains the main suspect. What Finding 40 still shows: WORLD output is perfectly separable from the original (on TORGO), and on TORGO-sourced simulation (sim_torgoctl, voice conversion of TORGO) the renderer matters. big10 changes the renderer and restores real exposure together, so it cannot separate the two; it tests whether simulation helps at all when exposure is right.

**big10** (`train/runs/big10.cmd`, started 2026-11-02 21:32 on the training PC). big7's data plus simtr360, a new transparent set: train-clean-360 (921 speakers and texts big7 never trains on; manifest 104014 clips, 363.6 h), 100 source hours of clips up to 16 s (`--max_src`, new; from big9's set, sources up to 16 s keep 87 % of slowed output under 32 s), all 14 conditions, severity 0.1–1.4, phone alignments, measured tables. 9950 steps, so real data is seen as often as in big7 (big9 halved that, Finding 38). Comparisons: big7 (deployed) and big9.

Two differences from big7 besides the new set:
- big10 has no mel-level simulation (big7's `--p_dys 0.3`): the waveform simulation replaces it.
- As in big9, the dysarthric TORGO clips are listed twice, to keep their share of the larger set.

Generation was stopped once, at 3438 clips, to free the PC; it resumed from those clips. Generation ran at about 30 × real time. Training: 9950 steps in 8201 s. Finished 2026-11-03 07:19.

**big10 result.**

Float, greedy (eval_manifest.py):

| Set | big6 | big7 | big9 | big10 |
|---|---|---|---|---|
| TORGO test, full / chunk 64 | 19.18 / 26.17 | 18.46 / 25.75 | 20.02 / 27.54 | 18.28 / 25.14 |
| UASpeech, full | 66.25 | 65.46 | 64.81 | 65.29 |
| UASpeech control / dysarthric, chunk 64 | 42.71 / 95.09 | 40.74 / 94.86 | 41.96 / 92.00 | 42.20 / 92.34 |
| LibriSpeech test-clean, full | 4.08 | 4.19 | 4.21 | 4.27 |

Engine (int8, LM), TORGO test, against big7 (paired bootstrap over clips, 95 %). The speaker groups are pooled: F03 + M05 and FC03 + MC04.

| | big7 | big10 | CI of the difference |
|---|---|---|---|
| final, all | 17.03 | 16.64 | −0.94..+0.33 |
| live (chunk 32), all | 23.31 | 22.80 | −1.04..−0.02 |
| final, dysarthric speakers | 35.43 | 35.41 | −1.47..+1.99 |
| live, dysarthric speakers | 45.99 | 46.04 | −1.01..+1.16 |
| final, control speakers | 8.37 | 7.82 | −0.89..−0.22 |
| live, control speakers | 12.65 | 11.87 | −1.32..−0.26 |
| final M05 / F03 | 46.14 / 29.77 | 44.27 / 30.72 | −3.81..0.00 / −0.93..+3.93 |

Against big9 (engine): dysarthric speakers final 36.76 → 35.41 (−2.57..0.00); control speakers final 9.78 → 7.82 (−2.35..−1.54), live 14.90 → 11.87.

UASpeech (float, paired bootstrap over clips, big10 minus big7):

| | full | CI | chunk 64 | CI |
|---|---|---|---|---|
| dysarthric speakers | 89.90 → 88.26 | −2.86..−0.36 | 94.86 → 92.34 | −3.80..−1.12 |
| control speakers | 37.29 → 38.82 | +0.36..+2.61 | 40.74 → 42.20 | +0.30..+2.69 |

All intervals from `train/paired_ci.py` (engine output, and eval_manifest.py output with `--mode`).

**Finding 45:** with real exposure restored, 100 source hours of the waveform simulation does not change recognition of TORGO's dysarthric test speakers. Final WER is 35.43 against 35.41, CI −1.5..+2.0.

What it does change:
- TORGO control speakers improve by 0.6–0.8 points.
- UASpeech's dysarthric speakers improve by 1.6–2.5 points. These are unseen speakers and words.
- UASpeech's control speakers get 1.5 points worse.

The UASpeech pattern matches big9's (92.0 / 42.0), so the simulated speech causes it, whatever the renderer. TORGO's controls recovering from big9 (9.78 → 7.82) confirms that big9's TORGO loss came from halved real exposure (Finding 38).

The pre-set test was that big10 beats big7 overall with a CI excluding zero. It passes only in live mode (−1.04..−0.02), and only through the control speakers. The simulator therefore does not deliver what it was built for, better recognition of the deployed population's speech. It does shift the model toward unseen dysarthric speech, at a cost on unseen typical speech.

big7 stays on the board. The two are equal on the target speakers, and big10's UASpeech gain is at 92 % WER, where neither model is usable.

### E17. A confidence meter on the device (2026-11-02)

**Question:** can the device tell the user when its text is probably wrong? A wrong transcript shown with confidence is worse than one marked as doubtful.

**Score.** During the full-context pass, the engine takes the log posterior of each token at the frame where the model emits it (greedy CTC peaks), and keeps their mean and their minimum (`tasr_nemo_last_confidence`). This adds one softmax normalization per emitted token, and WER is unchanged (17.03 %). A clip where nothing was emitted has no confidence. The host tool prints the score on stderr; `eval_cli.py` stores it per clip, and `esp32/host/conf_calib.py` calibrates it.

**Calibration.** big7_q121 + LM, TORGO test, 5214 clips, 72.4 % fully right. The measure and the thresholds were chosen on every other clip and reported on the rest. All four test speakers are on both sides: there are only four.

AUC for "the transcript is fully right":

| Measure | AUC |
|---|---|
| mean | 0.873 |
| min | 0.893 |
| (mean + min) / 2 | 0.894 (chosen) |

Levels, sure ≥ −0.120 > unsure ≥ −0.424 > probably wrong:

| Level | Held-out clips | Fully right | WER |
|---|---|---|---|
| sure | 53.6 % | 94.6 % | 3.9 % |
| unsure | 22.7 % | 65.0 % | 14.5 % |
| probably wrong | 23.7 % | 26.7 % | 49.3 % |

Control speakers' clips are 68 % sure and 12 % probably wrong; dysarthric speakers' clips are 25 % sure and 48 % probably wrong.

**Device.**
- `mic.c` marks the screen's history: "text ?" for unsure, "?? text" for probably wrong, nothing for sure.
- The serial "final" message gains "confidence" (sure / unsure / low) and "conf".
- Text from streaming alone (a press while another is waiting) has no level.
- Built; not yet flashed (the board was not connected).

**Finding 36:** the model knows fairly well when it is wrong. Half of all utterances can be shown as sure with 95 % of them fully right, and a quarter flagged as probably wrong, where half the words are wrong. For a user this is the difference between trusting the screen and being misled by it. The thresholds come from TORGO test and should be checked on new speakers and on the device's own microphone.

### E18. Phrase mode: a user's own list of phrases (2026-11-02)

**Question:** with a short list of phrases the user cares about ("I need water", names, places), how often does the device show the right
one, and how often does it put a list phrase over speech that was not on the list? This is a vocabulary list the user keeps, not a model
trained on the user.

**Method.**
- The engine (`tasr_phrase.c`) computes, during the full-context pass, each phrase's CTC log-likelihood (the forward algorithm over all alignments, from the same log posteriors the beam decoder gets). It also keeps the free decoding path's score, the best token at every frame.
- A phrase is shown when its score per frame is within a threshold of free decoding's; otherwise the free transcript is shown.
- Lists are tokenized off the device with the model's SentencePiece model (`esp32/tools/make_phrases.py`).
- Test: big7_q121 + LM, TORGO test, 5214 clips. Half of the 748 distinct test prompts go on the list (398, chosen by a hash of the text; `esp32/host/phrase_eval.py`). Clips of the other half stand for speech the user did not put on the list.
- For a list of N, each clip is scored against N − 1 random list phrases plus its own when it is on the list. The threshold is set so that 5 % of off-list utterances are replaced by a phrase, which is then always wrong.

**Results** (share of on-list utterances whose final screen is fully right):

| List size | all | control | dysarthric | best phrase right (dysarthric) | threshold |
|---|---|---|---|---|---|
| free decoding only | 72.6 | 83.5 | 49.7 | | |
| 10 | 94.4 | 97.1 | **88.5** | 93.8 | −0.163 |
| 25 | 92.2 | 96.2 | **83.7** | 90.6 | −0.119 |
| 50 | 90.1 | 95.3 | **79.0** | 89.0 | −0.089 |
| 100 | 87.5 | 94.3 | **73.3** | 86.3 | −0.067 |
| 398 | 79.3 | 89.3 | 58.3 | 78.2 | −0.018 |

Off-list utterances: 5.0 % replaced by a phrase at every list size (by construction); free decoding is fully right on 72.1 % of them.

**Caveats:**
- The threshold was set on the same off-list clips it is reported on; a device needs one threshold per list size, fitted on other data.
- TORGO's prompts are mostly single words (474 of 748), and longer phrases should be easier to tell apart.
- All of these prompts were seen in training (E13). A user's own phrases may not have been; the scoring itself needs no training.

**Cost:** for each frame, about 2 operations per phrase token. A 100-phrase list of about 5 tokens each is about 1000 operations per 40 ms frame, negligible next to the encoder.

**Finding 37:** a short phrase list turns the device from unreliable to usable for dysarthric speakers. With 25 phrases, 84 % of utterances come out fully right, against 50 % for free decoding, at a cost of 5 % of off-list speech being wrongly matched. The gain shrinks as the list grows (58 % at about 400 phrases), so the feature suits a core vocabulary rather than open dictation.

**On the device** (same day):
- The `phrases` command loads a list, which is kept in NVS (16 KB).
- `esp32/tools/send_phrases.py` tokenizes a plain list and sends it in batches. `esp32/tools/phrases_example.txt` holds 25 everyday phrases.
- Before each full pass the firmware rebuilds the engine's copy if the list changed. Afterwards it shows the best phrase as "> phrase" when it clears the threshold for the list's size (E18's table, interpolated in log size).
- The final event reports `"confidence":"phrase"`, and the console keeps the free text.
- Checked on the board: 25 phrases loaded, still there after a reboot. Not yet checked with speech.

### E19. Rotating the held-out TORGO speakers (queued 2026-11-02)

Question:   do big6's and big7's gains hold when other TORGO speakers are held out, or are they specific to F03 and M05?
Change:     only the TORGO split. Fold A is the official split (F03, M05, FC03, MC04). Fold B holds out F04 (mild), M04 (severe), FC01, MC01; fold C holds out M03 (mild), F01 (severe), FC02, MC02. M01 and M02 are in training in every fold.
Recipe:     `train/runs/fold.cmd <fold> <speakers> <big6|big7>`: makes and packs the fold's manifests, scores the base model on the fold's test set, trains with the named run's settings, and scores TORGO (fold test), LibriSpeech and UASpeech, float greedy. `G:\cosgc\folds.cmd` runs B and C with both recipes after big10.
Data:       no fold's test speakers in its training manifests. Caveat: the mel-level simulation (big7 recipe) was calibrated in E5 on fold A's training speakers, which include fold B's and C's test speakers: a small leak in the simulated arm only. UASpeech is test-only, as always.
Measure:    per fold, WER change from base, and big7 against big6 (paired CI); the paper reports the mean over folds A–C.
Result:     done 2026-11-05 03:54 (third launch, from 2026-11-03 15:10; order B big6, C big6, B big7, C big7). Results below the two void launches.
Void first launch (07:21–14:35):
- `fold.cmd` took the speaker list unquoted, and cmd splits arguments at commas.
- Each fold held out only its first speaker (B: F04, C: M03), and the recipe argument became a speaker name.
- Both runs therefore trained big7's recipe, with three of the four intended test speakers in training.
- Outputs moved to `G:\cosgc\exp\bad_folds\`; not used.
- `fold.cmd` now reads a quoted list (`%~2`), rejects any recipe but big6 or big7, and stops unless the test manifest holds exactly the requested speakers.

Void second launch (14:38–15:09):
- The cleanup had left the first launch's packed training lists behind (a name clash with the files already moved). Their split still had M04 in training.
- `fold.cmd` skipped re-packing because those lists existed.
- Training crashed at its first batch, because the pack they point to had been moved. No model was trained.
- `fold.cmd` now also stops before training if the packed training data holds any held-out speaker or points to a missing pack. The correct fold B list has 4233 dysarthric clips; the stale one had 4899.

Interruption (fold B, big7 recipe):
- Stopped at step 3800 on 2026-11-03 at about 21:40, at the user's request (RAM needed).
- The PC's CPU and motherboard were then replaced (Ryzen 5 1600 → 5500, same GPU and RAM).
- Resumed from the step-3750 checkpoint on 2026-11-05 at 01:01.
- On resume the trainer restarts the batch order of the current epoch, so this run's data order differs from an uninterrupted one.
- Restarting exposed a bug: `last.pt` is rewritten at every save, and the run scripts took it to mean "finished". A plain restart would have skipped the last 2250 steps and scored the partial model. The trainer now writes `out/done` after its final save, and every run script checks for that instead (commit 680b900).
- Speed: on this run, the median 100-step block took 1.33 s/step before the swap and 0.86 s after. That comparison is confounded: before the swap the PC was in use in the evening, and after it the PC sat idle overnight.
- Medians of whole runs show no clear CPU effect:
  - big6 recipe: 0.66 s/step (big6, old CPU), 0.97 and 1.02 (folds B and C, old CPU), 0.99 (big6_s1, new CPU, morning).
  - big7 recipe: 0.76 (simpre's stage 2, new CPU, night), 0.98 (big7_s1, new CPU, morning).
- What else the PC is doing (RAM was near zero free) matters more than the CPU.

Results, float greedy (`train/foldsum.py`). Each fold's models are scored only on that fold's held-out speakers. WER is full context / chunk 64:

| Fold: dysarthric test / control test | base, dysarthric | big6, dysarthric | big7, dysarthric | base, control | big6, control | big7, control |
|---|---|---|---|---|---|---|
| A: F03, M05 / FC03, MC04 | 60.06 | 40.12 / 51.24 | 39.11 / 50.60 | 16.98 | 9.32 / 14.38 | 8.75 / 14.06 |
| B: F04, M04 / FC01, MC01 | 55.55 / 61.30 | 41.02 / 47.79 | 41.19 / 46.77 | 17.93 / 23.46 | 8.82 / 13.29 | 8.03 / 12.04 |
| C: M03, F01 / FC02, MC02 | 29.70 / 35.26 | 18.93 / 24.98 | 19.27 / 24.20 | 17.78 / 22.34 | 9.73 / 13.51 | 8.99 / 12.61 |
| mean of folds, full | 48.44 | 33.36 | 33.19 | 17.56 | 9.29 | 8.59 |

Base → big6, dysarthric speakers, full context:
- A: −19.9 points (33 % relative). B: −14.5 (26 %). C: −10.8 (36 %).
- Per speaker: F04 22.0 → 11.6, M04 89.8 → 71.0, M03 15.3 → 7.2, F01 79.1 → 59.0. Fold A: F03 32.3, M05 54.9 after training.
- Severe speakers stay high in every fold: M05 52.6–54.9, F01 59.0–59.4, M04 71.0–71.4.

Other sets, big6 / big7:
- LibriSpeech test-clean: A 4.08 / 4.19, B 4.16 / 4.11, C 4.07 / 4.23.
- UASpeech dysarthric, full: A 89.41 / 89.90, B 86.73 / 87.54, C 86.21 / 87.04. big7 is 0.5–0.8 worse in every fold.
- UASpeech controls, full: A 39.57 / 37.29, B 40.11 / 40.14, C 41.51 / 39.72.

big7 against big6, paired bootstrap (`train/paired_ci.py`), change in WER points:
- Per fold, dysarthric speakers, full context: A −1.01 (CI −1.95..−0.16), B +0.17 (−0.79..+1.08), C +0.34 (−0.33..+1.01).
- Pooled over all three folds: 13 366 clips, all 12 TORGO speakers, each scored by models that never trained on them.

| Pooled | full: big6 → big7 | CI | chunk 64: big6 → big7 | CI |
|---|---|---|---|---|
| all | 17.17 → 16.61 | −0.78..−0.34 | 22.82 → 22.05 | −1.05..−0.50 |
| control | 9.33 → 8.64 | −0.91..−0.48 | 13.79 → 13.02 | −1.08..−0.47 |
| dysarthric | 35.02 → 34.75 | −0.76..+0.24 | 43.40 → 42.59 | −1.38..−0.22 |

The leak noted under Data favours big7 in folds B and C, which makes their null results on dysarthric final text conservative. One seed per fold and recipe.

**Finding 46:** the fine-tune's gain is not specific to the official split.
- With each of three disjoint sets of dysarthric speakers held out, big6's recipe lowers their WER by 11–20 points (26–36 % relative), and controls' by about half.
- LibriSpeech stays at 4.1–4.2 in every fold.
- Severe speakers stay above 50 % in every fold.

**Finding 47:** the mel-level simulation's gain (big7 over big6) holds across folds for control speakers (final −0.7, CI −0.9..−0.5) and for dysarthric speakers' live text (chunk 64, −0.8, CI −1.4..−0.2).
- Its final-text gain on dysarthric speakers came from fold A alone, mostly M05. Pooled over all 12 speakers it is −0.27, CI −0.76..+0.24.
- On UASpeech's dysarthric speakers, big7 is slightly worse in every fold.

### E20. The simulation as a first stage (queued 2026-11-03)

Question:   does the waveform simulation help when it comes before the real data instead of mixed into it?
Change:     big7's starting point only. Stage 1 trains base24x on simtr360 (E16) and LibriSpeech 100, 4000 steps of 100 s, no TORGO. Stage 2 is big7's recipe exactly, seed 0, started from stage 1.
Recipe:     `train/runs/simpre.cmd` (simpre1, then simpre).
Data:       no TORGO in stage 1; stage 2 as big7. No test speaker or UASpeech in anything fitted.
Measure:    engine int8 + LM, TORGO test live and final, paired CI against big7; UASpeech and LibriSpeech float greedy; stage 1 alone on TORGO test.
Why:        mixed in (big10), the simulation is separable from real speech (AUC 0.99), so the model can learn it as its own kind of speech. A first stage cannot be separated from what comes after.
Result:     done 2026-11-05. Both stages ran after the folds (from 03:54) and finished before 07:22, on the new CPU.
Stage 1 alone (simpre1, no TORGO), TORGO test float greedy: 30.8 full / 37.6 chunk 64; F03 50.7, M05 78.4, FC03 17.0, MC04 16.9.

Engine int8 + LM, TORGO test, big7 → simpre (`~/lmtest/simpre_q121_{full,stream}_lm.json`, `train/paired_ci.py`):

| | final: big7 → simpre | CI | live (c32): big7 → simpre | CI |
|---|---|---|---|---|
| all | 17.03 → 17.19 | −0.20..+0.56 | 23.31 → 23.78 | +0.02..+0.91 |
| control | 8.37 → 8.68 | −0.01..+0.64 | 12.65 → 13.18 | +0.11..+0.99 |
| dysarthric | 35.43 → 35.29 | −1.04..+0.76 | 45.99 → 46.34 | −0.56..+1.21 |
| M05 | 46.14 → 44.74 | −3.00..+0.25 | 58.79 → 59.45 | −1.13..+2.51 |

Float greedy, big7 → simpre:
- TORGO all: 18.46 → 18.65 (CI −0.22..+0.57); dysarthric 39.11 → 38.58 (−1.49..+0.42); control 8.75 → 9.27 (+0.14..+0.88).
- UASpeech dysarthric: 89.90 → 88.18 (−2.75..−0.60); controls 37.29 → 37.44 (−0.93..+1.26).
- LibriSpeech: 4.19 → 4.17.

Decision:   drop; big7 stays on the board.

**Finding 48:** the waveform simulation as a first stage changes nothing on TORGO's dysarthric speakers, as it did when mixed in (F45).
- Final text: 35.43 → 35.29, CI −1.0..+0.8.
- Live text is slightly worse, mostly on controls.
- Like big10, it improves unseen dysarthric speech in UASpeech (−1.7, CI −2.8..−0.6).
- Order does not rescue this simulator for TORGO.

### E21. A Parakeet teacher (queued 2026-11-03)

Question:   does a larger model, fine-tuned on the same data, make the 19 M model better when its posteriors are a second training target?
Teacher:    `train/teacher_ft.py`. nvidia/parakeet-tdt_ctc-110m (109 M parameters, CC-BY-4.0): its encoder and NeMo feature front end, our 1024-piece tokenizer, a new CTC head (blank last, as in the student), fine-tuned on big6's data, 6000 steps of 100 s, learning rate 1e-4, head 10 × faster, SpecAugment and our noise, reverb and speed augmentation. The 1.1 B model does not fit a 12 GB card for training.
Student:    big11 = big7's recipe exactly, plus a frame-level KL to the teacher on full-context batches (`train_nemo_stream.py --teacher`):
- weight 0.5, temperature 2 (scaled by T²);
- the teacher's 80 ms frames are repeated to the student's 40 ms;
- shifted by the offset `teacher_ft.py offset` measures against big7; a shift beyond 4 frames stops the run;
- clips changed by the mel-level simulation are left out (the teacher hears the waveform).
Recipe:     `train/runs/teacher.cmd`.
Data:       as big7; the teacher sees only big6's training data.
Measure:    the teacher on TORGO test, LibriSpeech and UASpeech (float, greedy, full context); big11 as E20, against big7.
Checked:    CPU smoke test on 24 TORGO training clips (laptop, NeMo 3.0.0, torch 2.6.0): teacher training, scoring, offset, and student training with and without the teacher all run.
Result:     ran 2026-11-05 22:26 to 2026-11-06 03:15 on the PC. big11 is worse than big7. Dropped; big7 stays on the board.
- Teacher: 6000 steps in 4683 s (0.78 s/step, 3.8 GB of GPU memory). Offset against big7: +4 teacher frames, the edge of the allowed range. The curve peaks clearly there: agreement 1.000 against 0.769 at +6 and 0.459 at +8.
- big11: 6000 steps in 9290 s (1.55 s/step). The teacher term was about 1.1 on the roughly 50 % of batches that carry it. Validation WER (TORGO test + LibriSpeech, float, mixed) went 6.87 at the start to 7.16 at the end.

Float, greedy, full / live (chunk 64) where both exist:

| Model | TORGO all | TORGO control | TORGO dysarthric | UASpeech all | UASpeech control | UASpeech dysarthric | LibriSpeech test-clean |
|---|---|---|---|---|---|---|---|
| Parakeet 110M, untuned (E15) | 24.40 | | | 66.62 | | 90.99 | 2.62 |
| teacher110 (fine-tuned, our tokenizer) | 20.99 | 14.25 | 35.34 | 69.95 | 48.34 | 88.70 | 9.17 |
| big7 | 18.46 | 8.75 / 14.06 | 39.11 / 50.60 | 65.46 | 37.29 / 40.74 | 89.90 / 94.86 | 4.19 / 6.50 |
| big11 | 18.60 | 9.35 / 15.46 | 38.26 / 51.01 | 64.60 | 40.62 / 43.67 | 85.41 / 98.08 | 5.79 / 7.76 |

Device engine, TORGO test, int8 + LM, big7 → big11, paired CI:

| | Final (full) | Live (chunk 32) |
|---|---|---|
| all | 17.03 → 17.84, +0.81 (+0.38..+1.26) | 23.31 → 24.26, +0.95 (+0.45..+1.47) |
| control | 8.37 → 9.30, +0.93 (+0.54..+1.31) | 12.65 → 13.67, +1.02 (+0.49..+1.58) |
| dysarthric | 35.43 → 35.98, +0.55 (−0.58..+1.65) | 45.99 → 46.78, +0.79 (−0.27..+1.85) |
| F03 / M05 | 29.77 → 29.63 / 46.14 → 48.00 | 39.23 → 39.02 / 58.79 → 61.45 (+0.58..+4.60) |
| FC03 / MC04 | 7.74 → 8.72 / 9.14 → 10.00 | 11.72 → 12.27 / 13.78 → 15.39 |

**Finding 50** (corrected by Finding 51: the loss comes from the LM step, not the acoustic model)**:** distilling from this teacher makes the on-device model worse. On the engine, final WER rises by 0.8 points (CI +0.4..+1.3), about 20 times big7's seed spread (sd 0.04), so the torch-version difference is unlikely to explain it. Dysarthric speakers are not clearly better or worse. The likely cause is the teacher itself. Fitting Parakeet's encoder to our 1024-piece tokenizer with a new head lost most of its typical-speech skill: LibriSpeech went 2.62 → 9.17, and TORGO controls are at 14.25 against big7's 8.75. Only on dysarthric TORGO speakers (35.34 against 39.11) and UASpeech's dysarthric speakers was it better than the student. big11 inherited that profile: in float it is better on unseen dysarthric speech (UASpeech dysarthric 89.90 → 85.41) and worse on typical speech (LibriSpeech 4.19 → 5.79, UASpeech controls +3.3). A useful teacher here would first have to beat the student on typical speech, for instance by keeping Parakeet's own tokenizer and distilling across tokenizers, or by applying the teacher term only to dysarthric clips. Neither is tested.
- First attempt, 2026-11-05 07:24, failed while loading the model.
- After the motherboard swap, Windows Smart App Control is on, and it blocked a pandas DLL in the NeMo environment ("An Application Control policy has blocked this file"). No teacher was trained.
- The queue went on to the seed runs. big11 waits for a working NeMo environment on the PC.
- Fixed 2026-11-05 with Smart App Control left on. The Code Integrity log (Event 3077) names each blocked file. Only two files were blocked in the NeMo environment: pandas 3.1.0's `offsets.pyd`, and the `_torchaudio.pyd` from download.pytorch.org's torchaudio 2.6.0+cu126. Changes: pandas 2.3.3; torchaudio 2.6.0+cpu from PyPI (`--no-deps`; torch stays 2.6.0+cu126, CUDA available; NeMo uses torchaudio only for audio metrics). A scan that maps every .pyd/.dll without running it found no blocks, so only a real import shows them.
- PC smoke tests, all passing: teacher training (30 steps), scoring (`--limit 100`), the offset step, and student training with the teacher (20 steps).
- Queued for 01:00 on 2026-11-06 (`G:\cosgc\teacherq.cmd`, which calls `teacher.cmd`). Paused at the user's request on 2026-11-05 at about 16:30, before it started. Started directly at 22:26 that day when the user said to go ahead.
- Caveat for the comparison: big11 trains in the NeMo environment (torch 2.6.0), big7 in the main one (torch 2.14.1). Kernels and random streams differ, so a difference from big7 smaller than the seed spread (Finding 49, about 0.3 points) means nothing.

### E22. The teacher only on dysarthric clips (started 2026-11-06 04:08)

Question:   Finding 50 says the E21 teacher beat big7 only on dysarthric speech, and passed its weakness on typical speech to big11. Does limiting the teacher term to dysarthric training clips keep the gain and avoid the loss?
Change:     big11 exactly, plus `--teacher_groups dysarthric`. The teacher term is computed only on clips whose manifest group is dysarthric (TORGO training speakers F01, F04, M01–M04; never simulated clips; LibriSpeech has no group). `data.py` now keeps the manifest's group. Batches without such a clip skip the teacher. In the default (all groups), the loss is unchanged.
Recipe:     `train/runs/big12.cmd` (teacher110 and its offset, +4, from E21).
Checked:    PC smoke test, 100 full-context steps on LibriSpeech 100 + TORGO train: the teacher term was active on 21 of 100 batches.
Measure:    as E21, against big7. Same torch-version caveat as E21.
Result:     ran 2026-11-06 04:08–06:49. 6000 steps in 7511 s (1.25 s/step). The teacher term was on about 23–26 of every 100 steps. Validation (mixed TORGO test + LibriSpeech, float): 6.87 at the start, 5.73 at the end (big11: 7.16).

Float, greedy (full / live chunk 64):

| Model | TORGO all | TORGO control | TORGO dysarthric | UASpeech all | UASpeech control | UASpeech dysarthric | LibriSpeech |
|---|---|---|---|---|---|---|---|
| big7 | 18.46 | 8.75 / 14.06 | 39.11 / 50.60 | 65.46 | 37.29 / 40.74 | 89.90 / 94.86 | 4.19 / 6.50 |
| big11 | 18.60 | 9.35 / 15.46 | 38.26 / 51.01 | 64.60 | 40.62 / 43.67 | 85.41 / 98.08 | 5.79 / 7.76 |
| big12 | 18.10 | 8.41 / 13.96 | 38.70 / 50.46 | 62.03 | 39.81 / 42.41 | 81.30 / 92.78 | 4.20 / 6.85 |

Device engine, TORGO test, final text (full context), big7 → model, paired CI:

| | int8, greedy, no LM | int8 + LM (deployed decoding) |
|---|---|---|
| big11, all | 18.62 → 18.59 (−0.50..+0.41) | 17.03 → 17.84 (+0.38..+1.26) |
| big11, control | 8.93 → 9.38 (+0.03..+0.88) | 8.37 → 9.30 (+0.54..+1.31) |
| big11, dysarthric | 39.20 → 38.17 (−2.06..−0.07) | 35.43 → 35.98 (−0.58..+1.65) |
| big12, all | 18.62 → 18.16 (−1.00..+0.03) | 17.03 → 17.96 (+0.43..+1.47) |
| big12, control | 8.93 → 8.35 (−1.01..−0.15) | 8.37 → 8.76 (+0.00..+0.76) |
| big12, dysarthric | 39.20 → 39.02 (−1.44..+1.08) | 35.43 → 37.52 (+0.64..+3.38); M05 46.14 → 51.73 |

big12, live text (chunk 32) + LM: all 23.31 → 23.45 (−0.33..+0.60), dysarthric 45.99 → 46.73 (−0.35..+1.83). Against big7's three-seed pool, final + LM: +0.98 (+0.51..+1.44).

What the LM gains (no LM → LM, final, engine): big7 −1.59 overall and −3.77 on dysarthric speakers; big11 −0.75 and −2.19; big12 −0.20 and −1.50.

**Finding 51:** distillation from the teacher changes how much the language model helps, and that, not the acoustic model, is what made both distilled models worse on the device. On the engine without the LM, big12 is at least as good as big7 (controls −0.6, CI −1.0..−0.2), and big11 equals it overall, with dysarthric speakers −1.0 (CI −2.1..−0.1). With the LM, the LM's gain shrinks from 1.6 points (big7) to 0.75 (big11) and 0.2 (big12), and both end up worse. The LM weight and token bonus (0.30 and 0.50) were set for big2–big7. The teacher's soft targets change how sure the student is of each token, so those values likely no longer fit. Retuning them for a distilled model needs a development set that is not TORGO test; until that exists, neither model can be deployed. In float, big12 is the best model so far on unseen dysarthric speech (UASpeech dysarthric 89.90 → 81.30) without big11's loss on typical speech (LibriSpeech 4.20).
**Correction to Finding 50:** the teacher did not make the acoustic model worse on TORGO. The deployed system got worse through the LM interaction above. big11's typical-speech loss (LibriSpeech 4.19 → 5.79, float) stands, and limiting the teacher to dysarthric clips (big12) removes it.

### E23. Live sessions on the board with the microSD log (prepared 2026-11-06)

Question:   on the board's own microphone, with people speaking, how often is the live and the final text right, and how often does
            phrase mode pick the right phrase or a wrong one?
Device:     big7_q121 + LM, `oido_p4_2c` with the SD log (docs/P4.md, "microSD session log"), the 25 phrases of
            `esp32/tools/phrases_example.txt` loaded (phrase mode).
Sessions:   club members, each under a speaker code, about 15 cm from the board.
            - `prompts_session.txt`: the 25 list phrases and 10 sentences not on the list, mixed.
            - `prompts_imitated.txt`: the same prompts read as imitated dysarthria (slow, pauses, soft endings, breathy; phonetic cues on the screen).
            Imitated speech is a stress test of the device and of phrase mode only. It is never reported as accuracy for dysarthric
            speakers, and never used for training. The rows stay labelled "imitated".
Score:      `esp32/tools/sd_session.py score`.
Host check before flashing:
            - The firmware's phrase rule (25 phrases, threshold −0.119), run on the host engine with big7 + LM.
            - Clips: Windows text-to-speech, two voices.
            - Results: 50 of 50 list clips gave the right phrase; 0 of 20 off-list clips were replaced by a phrase. Free decoding was already right on all 50 list clips and on 16 of the 20 off-list clips, so this checks the pipeline, not the benefit.
            - The same 70 clips through `dysim.py demo` (transparent render), six motor-speech conditions (flaccid, spastic, ataxic,
              hypokinetic, ALS, CP), 420 clips per severity:

| Severity | List clips: right phrase / wrong phrase / free text | Free decoding alone, list clips fully right (WER) | Off-list clips wrongly replaced | Off-list WER |
|---|---|---|---|---|
| 0.8 | 296 / 0 / 4 of 300 | 254 of 300 (6.4 %) | 0 of 120 | 12.7 % |
| 1.2 | 268 / 23 / 9 of 300 | 205 of 300 (18.4 %) | 17 of 120 (14 %) | 27.1 % |

            On simulated speech, phrase mode lifts fully-right list utterances from 85 % to 99 % (severity 0.8) and from 68 % to 89 % (1.2).
            At 1.2, 14 % of off-list utterances are replaced by a wrong phrase, against the 5 % the threshold allows on TORGO test. The
            threshold was set for real dysarthric speech, and severe simulated speech pushes free decoding's score down. Simulated, so
            only a sign of where to look on the board.
            Host tools: `esp32/host/eval_cli.py` (now also takes plain WAV manifests) and the firmware's rule re-implemented in a scratch script.
Result:     first session 2026-11-06, about 11:50, one typical speaker (a club member; the speaker line was left as CODE-HERE),
            normal speech, `prompts_session.txt`, card SD32G (FAT32). The card mounted at the first boot (LDO 4). Session s0001 holds a
            first attempt of 5 utterances; the board then restarted when the laptop opened its serial port, and s0002 is the full run.
            Recordings stay on the card.

s0002, 35 prompts (25 list phrases, 10 sentences not on the list), board engine, big7 + LM, phrase mode with 25 phrases:

| Text | WER | Fully right |
|---|---|---|
| shown (final, phrase mode) | 7.87 % | 29 / 35 |
| final, free decoding | 10.24 % | 27 / 35 |
| live (streaming, finished at the release) | 11.81 % | 26 / 35 |

- Phrase mode replaced the free text 23 times, always with the right phrase. It corrected two free-decoding errors: "play" →
  "please" (gap −0.013), and "i do not under the" → "i do not understand" (−0.022).
- It missed one list phrase, "please wait" (free text "please ease wait", gap −0.261 against the threshold −0.119).
- "i need the bathroom" was read as "i need to use the bathroom" (a reader paraphrase, not a recognition error). It was rightly not replaced (−0.275).
- Off-list sentences: none replaced; the closest was "what time is dinner" (6.5 s held), whose best phrase "yes" came within
  −0.165 of the threshold. Lowering the threshold to catch "please wait" would have turned it into "yes". Threshold unchanged;
  one session is too small to refit it, and it would be fitted on its own test.
- Levels (medians): speech −34.9 dBFS, noise −56.8 dBFS (22 dB apart). Final text 1.77 s after the release (median); for
  holds of 4–6.5 s, 3.0–4.5 s.
- Card writes took 0.18–2.13 s each, after the text was shown. The firmware now writes from a task of its own (`sdlog.c`,
  queue of 4), so the recognizer only copies the audio. Built; not yet on the board.

**Finding 52:** on the board's own microphone, with a person speaking, the device gets 29 of 35 prompts fully right (WER 7.9 %), against 27 without
phrase mode. Phrase mode made no wrong replacement in 35 utterances. One typical speaker and one session, so this checks the
whole device pipeline (button, microphone, full-context pass, phrase mode, SD log) end to end, not accuracy for dysarthric speakers.

### E24. UASpeech in training, whole speakers held out (queued 2026-11-06 for 01:00)

Question:   E11 added UASpeech (all 28 speakers) and could only be tested on the same people. Does UASpeech help recognition of
            dysarthric speakers the model never heard, on UASpeech and on TORGO?
Held out:   chosen before any result, from the UASpeech paper's intelligibility table (doc/is2008_kim.pdf, Table 1), one dysarthric speaker
            per band, two women and two men: F03 (very low, 6 %), M07 (low, 28 %), F04 (mid, 62 %), M09 (high, 86 %). Controls: CF03,
            CF05, CM05, CM10. M08, M10, M12, M14 and M16 had no rating in that table, so they were not candidates.
Data:       big7's list (150 h) plus UASpeech B1 + B3 of the other 20 speakers (`train/split_speakers.py`; 17 850 clips, 12.27 h, of
            which dysarthric 7.93 h; each word is recorded on two microphones, M2 and M5, so about half that is distinct speech).
            Test: UASpeech B2, M5, of the 8 held-out speakers (2036 clips, 1.40 h).
            The held-out speakers' B2 words include the common words, digits, letters and commands that other speakers say in B1/B3,
            so this is new people on partly familiar words, as on TORGO.
Change:     big7's recipe; steps 6000 → 6500 so TORGO and LibriSpeech are seen as often (162 h against 150 h);
            `--dys_sources torgo,uaspeech` so the mel-level simulation never touches UASpeech. Main environment (torch 2.14.1), as big7.
Recipe:     `train/runs/big13.cmd`. Checked: the five-manifest list loads on the PC (100 163 clips, 162 h).
Measure:    big13 against big7, paired: TORGO test (engine + LM, float), LibriSpeech, the held-out UASpeech speakers (float). The full B2
            score is also computed, but it includes 20 speakers big13 trained on: not a result.
Result:     not run yet.

### Seeds (queued 2026-11-03)

big6 and big7 with seeds 1 and 2 (`big6.cmd 1`, `big7.cmd 2`, ...; `train_nemo_stream.py --seed` changes batch order, masks and chunk choices; seed 0 reproduces the original runs). Measure: the spread of engine WER across seeds, and whether big7 against big6 holds for every seed.

Queue on the PC: simpre, teacher, big6 seed 1, big7 seed 1, big6 seed 2, big7 seed 2. Originally `G:\cosgc\queue2.cmd`; since 2026-11-05 `G:\cosgc\resume.cmd`, which finished the folds first. Started 03:54 that day.

Ran 2026-11-05 on the PC with the new CPU, after simpre and the failed teacher. Finish times: big6_s1 08:42, big7_s1 10:53, big6_s2 12:33, big7_s2 14:57 (QUEUEDONE). Each run is the original recipe exactly (6000 steps for big7), only `--seed` differs. Engine scoring on the laptop as for table 1 (`big6eval.sh`, beam 4 + LM, int8 max/121). Float scoring by the run scripts (`G:\cosgc\exp\eval\<model>_{torgo_test,uaspeech_test_b2,ls_testclean_train}.json`).

Engine, TORGO test, WER all / control / dysarthric:

| Model | Seed | Final (full) | Live (chunk 32) |
|---|---|---|---|
| big6 | 0 | 17.74 / 8.89 / 36.56 | 23.86 / 13.01 / 46.92 |
| big6 | 1 | 17.32 / 8.55 / 35.96 | 23.21 / 12.22 / 46.57 |
| big6 | 2 | 17.46 / 8.71 / 36.07 | 23.33 / 12.55 / 46.25 |
| big6 | mean ± sd | 17.51 ± 0.21 / 8.72 ± 0.17 / 36.20 ± 0.32 | 23.47 ± 0.35 / 12.60 ± 0.40 / 46.58 ± 0.33 |
| big7 | 0 (on the board) | 17.03 / 8.37 / 35.43 | 23.31 / 12.65 / 45.99 |
| big7 | 1 | 16.99 / 8.55 / 34.92 | 23.49 / 12.83 / 46.16 |
| big7 | 2 | 16.94 / 8.40 / 35.08 | 23.02 / 12.45 / 45.49 |
| big7 | mean ± sd | 16.98 ± 0.04 / 8.44 ± 0.10 / 35.14 ± 0.26 | 23.27 ± 0.24 / 12.64 ± 0.19 / 45.88 ± 0.35 |

big7 against big6, same seed, engine, paired CI (B − A):

| Seed | Final, all | Final, dysarthric | Live, control | Live, dysarthric |
|---|---|---|---|---|
| 0 | −0.71 | −1.13 (−2.09..−0.24) | −0.36 (−0.84..+0.09) | −0.93 (−1.91..+0.07) |
| 1 | −0.33 | −1.04 (−1.96..−0.22) | +0.61 (+0.12..+1.13) | −0.41 (−1.37..+0.53) |
| 2 | −0.52 | −0.99 (−1.96..−0.04) | −0.10 (−0.51..+0.29) | −0.76 (−1.74..+0.19) |

Pooled over seeds: each clip's edits summed over the three seeds of a model, then paired_ci.py over clips (`train/pool_seeds.py`; this compares the seed-mean models and resamples clips only, not seeds).

| | Final (full) | Live (chunk 32) |
|---|---|---|
| all | 17.51 → 16.98, −0.53 (−0.75..−0.28) | 23.47 → 23.27, −0.20 (−0.46..+0.09) |
| control | 8.72 → 8.44, −0.28 (−0.48..−0.09) | 12.60 → 12.64, +0.04 (−0.23..+0.34) |
| dysarthric | 36.20 → 35.14, −1.06 (−1.65..−0.47) | 46.58 → 45.88, −0.70 (−1.31..−0.09) |
| F03 | 30.17 → 29.63 (−1.25..+0.08) | 39.32 → 39.17 (−0.84..+0.56) |
| M05 | 47.60 → 45.58 (−2.99..−0.99) | 60.32 → 58.57 (−2.77..−0.80) |
| FC03 | 8.22 → 7.87 (−0.62..−0.10) | 12.29 → 11.82 (−0.82..−0.11) |
| MC04 | 9.32 → 9.14 (−0.51..+0.12) | 12.97 → 13.65 (+0.18..+1.18) |

Float, greedy, mean ± sd over the three seeds (big6 → big7): TORGO dysarthric full 39.75 ± 0.40 → 38.94 ± 0.15 (big7 lower for every seed: −1.01, −0.92, −0.48); TORGO control full 9.16 → 8.85; UASpeech B2 control full 39.02 ± 0.48 → 37.85 ± 0.60; UASpeech B2 dysarthric full 88.73 ± 0.64 → 88.80 ± 0.96; LibriSpeech test-clean 4.11 ± 0.04 → 4.15 ± 0.05.

**Finding 49:** retraining with another seed moves engine WER by little: sd 0.04–0.35 points over all clips and 0.26–0.35 on dysarthric speakers. Differences of about 0.5 points between single runs are therefore at the edge of seed noise. big7's gain over big6 on dysarthric speakers' final text holds for every seed (−1.13, −1.04, −0.99; pooled −1.06, CI −1.65..−0.47), so on the official split it is not seed luck. It comes mostly from M05 (−2.0). Folds B and C did not show it (Finding 47). The gain therefore depends on which speakers are held out, not on the seed. Seed 0 of big6 was its worst seed, so table 1's single-seed step from row 5 to row 6 (−0.71 overall) overstates the mean step (−0.53). On live text, big7's control speakers are no better (MC04 is worse in two seeds of three). The deployed big7 (seed 0) sits at its seeds' mean (17.03 against 16.98), so it stays on the board.

## 5. Findings so far, for the paper

The list below covers E1–E14. Findings 17–36 are numbered in their entries. The consolidated claims the paper will make, each with its
evidence and status, are in [PAPER.md](PAPER.md) (section 5); keep that table current rather than extending this list.

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
10. Nearly all of the int8 loss of a float fine-tune comes from weight rounding, not activations (E8b), and the cause is the grid:
    the shipped weights sit exactly on it, so 77 to 88 % of the fine-tune's change rounded away. A different grid keeps 99 % of it and removes
    the loss entirely (E10). On the device arithmetic with the language model, the 24-layer model reaches 25.1 % WER on TORGO test.
11. On real hardware, the shipped model streams at 0.748 times real time on one ESP32-P4 core at 400 MHz, about twice as fast
    as the ESP32-S3, with transcripts identical to the host engine on four clips (E9). The 24-layer model, our most accurate,
    runs at 0.967 on one core and 0.602 on two (E9b). Nobody has yet spoken at the live firmware; clips played on a laptop's speakers were recognized through its microphone (E12).
12. Context matters more than anything else we tried on the device: the same 24-layer model goes from 25.1 % WER with 1.28 s chunks to 19.8 % with the whole utterance on TORGO test (dysarthric 51.2 to 42.2), and the P4 can
    compute that in about 0.64 times the utterance's length (E12). The firmware now streams partial text, then computes the final text with full context.
13. Bit-exact kernels do not make transcripts identical across platforms: a one-in-a-million difference from `logf` grows to about 1 % in the logits through the dynamic int8 quantization, which can flip a marginal word (E12).
14. Dysarthric speech was only 6.5 % of the training audio. Raising it to 26 % lowers WER on the device arithmetic, full context, from 19.5 to 17.7 % (dysarthric 41.6 to 36.6), the largest training gain since the 24-layer model, but it costs about 2.6 points on UASpeech, an unseen corpus (E13).
15. 99 % of TORGO's test clips use prompts also in its training set, so TORGO scores describe new speakers on familiar prompts (E13).
16. On the board's microphone, signal-to-noise ratio matters and level control does not. Measured noise costs 0.4 points at reading distance and 2.7 at about 16 dB; noise reduction helps only near 8 dB, in simulation (E14).

## 6. Threats to validity and open issues

- **Few speakers.** 11 training and 4 test speakers, 2 of them dysarthric. Results per speaker are anecdotal; the
  direction across all four matters more than any one number. A second corpus (UASpeech) would allow a test on
  different speakers and recording conditions.
- **Validation on test speakers.** Settings in E2 and E3 were chosen by watching validation clips from the
  held-out speakers. Before the paper, re-run the final configuration with a validation set from a training speaker
  (or held out sessions), and report the test speakers only once.
- **Vocabulary overlap.** TORGO training and test speakers share prompts. Report an overlap-free subset: test clips
  whose transcript never appears in training. **Correction (E13):** this is not possible with this split. Only 46 of 5214 test clips qualify, and removing the shared
  prompts from training would leave 0.47 h. State the overlap in the paper and use UASpeech and LibriSpeech as the checks on unfamiliar material.
- **One seed per run.** Before the paper, repeat the final configuration with at least 3 seeds and report mean and
  spread. **Done for big6 and big7 (Seeds, Finding 49):** the seed spread is 0.04–0.35 points; other rows remain one seed.
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
