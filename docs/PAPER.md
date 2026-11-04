# Paper skeleton: offline dysarthric speech recognition on a microcontroller

The structure of the paper, the results it reports, and where each number comes from. The detailed record of every run is
[EXPERIMENTS.md](EXPERIMENTS.md) (entries E1–E17); this file only collects, in a fixed format, what the paper will say. Every number here
is copied from a result file named beside it; if a number changes, change it there first.

**Status key:** **solid**: measured on the held-out sets, consistent across speakers or with a confidence interval that excludes zero.
**tentative**: one seed, one speaker, or simulated. **pending**: built, not yet measured.

## 1. One-paragraph summary

A 19 M-parameter streaming Conformer-CTC model runs offline on an ESP32-P4 microcontroller (two RISC-V cores at 400 MHz, 32 MB PSRAM). It
shows live text while the user holds a button, then a full-context transcript, with a confidence mark. Fine-tuning on about 10 h of public
dysarthric speech (TORGO) plus LibriSpeech lowers word error on unseen TORGO speakers from 37.4 % to 17.0 %. Over the same steps,
dysarthric speakers go from 68.4 % to 35.4 %. Several of the steps are engineering rather than data: growing the model, the int8 export
grid, full-context final text. Simulated dysarthric speech is our route to more training data than the public corpora hold; first results
are small but positive. An untuned 1.1 B-parameter model (Parakeet-TDT) is no better on dysarthric speakers than ours. All results use
held-out speakers. TORGO's prompt overlap is reported, not hidden.

Figures: `docs/figures/make_figures.py` regenerates every figure from the numbers in this file. Figure 2 `fig_ladder` (table 1), figure 3
`fig_simulator` (C8), figure 4 `fig_confidence` (C10).

## 2. System (figure 1)

```
 public data ──► waveform simulator ──► training (PyTorch, RTX 3060) ──► int8 export ──► C engine ──► ESP32-P4 firmware
 LibriSpeech      dysim.py + phonesim.py   train_nemo_stream.py           export_nemo.py   tinyasr      push-to-talk, OLED
 TORGO (train)    14 conditions            streaming chunks + full ctx    grid max/121     (bit-exact   live text (chunk 32)
 MUSAN, RIRs      severity 0.2-1.4         noise, reverb, speed           19 M params      host build)  final text (full ctx)
                                                                         20.4 MB                        confidence mark
```

| Component | File(s) | Paper section |
|---|---|---|
| Model: NeMo Conformer-CTC, grown from 16 to 24 layers by identity init, 1024 tokens | `train/nemo_small.py`, `train/grow_nemo.py` (E6) | Method |
| Training: streaming chunks 32/64 and full context, EMA, augmentation | `train/train_nemo_stream.py`, `train/runs/*.cmd` | Method |
| Export: int8 weights on a max/121 grid | `train/export_nemo.py` (E10) | Method |
| Engine and LM: C, int8, beam 4 + token LM | `esp32/components/tinyasr/` (E8) | Method |
| Firmware: push-to-talk, live text, full-context final pass reusing the stream's front end, confidence | `esp32/firmware/main/mic.c` (E12, E14, E17) | System |
| Simulator: 14 conditions, sound-level errors from phone alignments spliced from the clip's own phones, rendered on the original recording (PSOLA + filter; WORLD only for new voicing) | `train/dysim.py`, `train/phonesim.py` (E15, E16) | Method |

## 3. Evaluation protocol (fixed for every result)

| Set | What it measures | Size |
|---|---|---|
| TORGO test: F03 (moderate), M05 (severe), FC03, MC04 (controls) | new speakers. Prompts are familiar: 99 % of test clips use a training prompt (E13) | 5214 clips, 3.87 h |
| UASpeech block 2 (never trained on in reported models) | unfamiliar words, other speakers and room | 7136 clips |
| LibriSpeech test-clean | typical speech must not get worse | 2620 clips |
| Simulated test-clean (dysim, seed 11) | the simulator's difficulty; never a headline result | 1947 clips |

- **Metric:** word error rate, corpus level (jiwer), same text normalization everywhere.
- **Decoding:** headline numbers are from the device engine (int8, host build, bit-exact with the chip's kernels), beam 4 + LM. Float greedy numbers are labeled as such.
- **Modes:** live (chunk 32 = 1.28 s, left 128) and final (full context).
- **Comparisons:** paired bootstrap over clips within speaker, 1000 resamples, 95 % interval (`paired.py`).
- **Leakage rules:**
  - Nothing is ever fitted on TORGO's test speakers.
  - Nothing is fitted on UASpeech at all, because its test block shares speakers with its training blocks.
  - The simulator is calibrated on TORGO training speakers only.
  - Test speakers' validation clips were watched during training up to E3 (see section 8).

## 4. Main results: one change per row (table 1)

TORGO test, device engine, int8 + LM unless noted. Result files: `~/lmtest/<model>_q121_{full,stream}_lm.json` (laptop) and
`G:\cosgc\exp\eval\` (PC).

| # | Model | Change from the row above | Live (c32) all / dysarthric | Final (full) all / dysarthric | Entry | Status |
|---|---|---|---|---|---|---|
| 0 | shipped (Oído, 16 layers) | none | 37.37 / 68.4 (no LM) | | E8 | solid |
| 1 | atyp2 | fine-tune: TORGO train + LibriSpeech 100, augmentation | 30.87 / 59.6 | | E3, E8 | solid |
| 2 | big2 | 24 layers, identity-initialized | 26.23 / 52.9 | | E6, E8 | solid |
| 3 | big2_q121 | export grid max/121 (keeps the fine-tune's weight changes) | 25.14 / 51.17 | 19.75 / 42.15 | E10, E12 | solid |
| 4 | big4 | more training on full context | | 19.51 / 41.55 | E13 | tentative (small) |
| 5 | big6 | dysarthric share of audio 6.5 → 26 % | 23.86 / 46.92 | 17.74 / 36.56 | E13 | solid |
| 6 | big7 (on the board) | + mel-level simulated dysarthria on 30 % of LibriSpeech clips | 23.31 / 45.99 | 17.03 / 35.43 | E15 | solid (CI −1.08..−0.35) |
| 7 | big9 | + 120 h of waveform-simulated speech (round 2), same steps | 25.09 / 46.75 | 18.41 / 36.76 | E15 | worse (CI +0.94..+1.84); confounded: real data seen half as often (F38) |
| 8 | big10 | from row 6: the mel-level simulation replaced by 100 source hours of transparently rendered waveform simulation (round 4), steps scaled to keep real exposure | 22.80 / 46.04 | 16.64 / 35.41 | E16 | no change on dysarthric speakers (CI −1.47..+1.99); controls better; UASpeech dysarthric better, controls worse (F45) |

Per speaker, final, row 6: F03 29.77, M05 46.14, FC03 7.74, MC04 9.14. Row 6 on other sets, float greedy: UASpeech 65.46 (shipped 72.8
streaming), LibriSpeech 4.19 (shipped 4.01).

## 5. Claims (each with its evidence)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The shipped on-device model fails on dysarthric speech: 57 % and 90 % live WER for a moderate and a severe speaker, against 6 % on LibriSpeech. | §3 of the log | solid |
| C2 | About 10 h of public dysarthric speech lowers WER on unseen dysarthric speakers by 8–15 points (live), and carries over to an unseen corpus (UASpeech 72.8 → 66.5). | E3, E7 (F17) | solid |
| C3 | Depth helps more than width at this size: 16 → 24 layers by identity init beats the same fine-tune for every test speaker and is more robust to int8. | E6, E8 (F15, F16) | solid |
| C4 | The int8 loss of fine-tunes came from the export grid, not from 8 bits: rounding on the parent model's grid discarded 77–88 % of the weight change; another grid keeps 99 %. | E8b, E10 (F18, F21) | solid |
| C5 | Context is the largest lever on the device: full-context final text cuts WER from 25.1 to 19.8 % (dysarthric 51.2 → 42.2), and the P4 computes it in about 0.64 × the utterance. | E12 (F23) | solid |
| C6 | The share of dysarthric audio matters up to about 26 %; beyond that the gain is one speaker and unfamiliar words get worse. | E13, E14 (F25, F31) | solid |
| C7 | Simulated dysarthria helps slightly at 24 layers (−0.7 points, CI excludes zero) without the UASpeech cost of more real data; it did not help at 16 layers. Scaling it up to 100 source hours of waveform simulation with real exposure held fixed does not help TORGO's dysarthric speakers (35.43 → 35.41). It moves the model toward unseen dysarthric speech (UASpeech dysarthric −1.6 to −2.5 points) and away from unseen typical speech (UASpeech controls +1.5). | E4, E15, E16 (F11, F33, F38, F45) | tentative (one seed); the large-scale result is solid for this simulator |
| C8 | A signal-level simulator matches real dysarthric speech on rate, pausing, rhythm and spectral tilt; with sound-level errors from phone alignments and measured error tables, its dysarthria conditions span every TORGO training speaker by WER (severity 0.3–1.5). It is still separable from real dysarthric speech (AUC 0.99) and sounds less dysarthric to a probe than real severe speech. | E5, E15, E16 (F32, F39, F43) | solid (measurement); in training it does not help TORGO's dysarthric speakers (big10, F45) |
| C9 | Size alone does not solve dysarthric speech: an untuned 1.1 B model scores 90.7 % on UASpeech dysarthric speakers (ours 94.9) and 23.1 % on TORGO test (ours 18.5, float greedy, prompt-familiar). | E15 (F35) | solid |
| C10 | The model's own token posteriors flag errors: AUC 0.894; utterances marked "sure" (54 %) are 95 % fully right, "probably wrong" (24 %) 27 %. | E17 (F36) | solid on TORGO; device microphone pending |
| C11 | On the board's microphone, signal-to-noise ratio matters and level control does not; noise reduction helps only near 8 dB SNR. | E14 (F28, F29) | tentative (simulated noise) |
| C12 | Bit-exact kernels do not guarantee identical transcripts across platforms: dynamic int8 quantization amplifies 1e-6 input differences to about 1 % in the logits. | E12 (F24) | solid |
| C13 | A user's short phrase list, scored by CTC on the device, raises fully right utterances for dysarthric speakers from 50 % to 84 % (25 phrases) or 89 % (10), with 5 % of off-list speech wrongly matched. | E18 (F37) | tentative (threshold fitted on the test clips; TORGO prompts) |
| C14 | Vocoder resynthesis is a hidden confound in simulated speech: WORLD with no change at all costs 11 points of WER on TORGO's recordings (0.4 on clean LibriSpeech) and is perfectly separable from the original; it supplied about half of what made the simulation sound dysarthric. Editing the recording in place (PSOLA + filtering) removes it. | E16 (F40, F42, F44) | solid |
| C15 | Voice conversion (kNN-VC) keeps the dysarthria of real recordings (probe 96 %) but adds 7–10 points of its own. | E16 (F41) | solid (measurement); not used in training |

**Negative results worth reporting:**
- Frame-level in-place distillation damaged an already-streaming model (E2).
- Adding UASpeech to training made TORGO worse, 25.14 → 27.55 (E11, F22).
- More dysarthric share beyond 26 % gave no broad gain (big8, F31).
- Language-model weight tuning, a wider beam and padding gave at most 0.3 points (F27).
- 100 source hours of the waveform simulator, with real exposure held fixed, left TORGO's dysarthric speakers unchanged (big10, F45).

## 6. Device (table 2)

| Measure | Value | Entry |
|---|---|---|
| Model size (int8) / LM | 20.4 MB / 1.29 MB | E9b |
| Live streaming speed, 24 layers | 0.602 × real time on two cores (0.967 on one) | E9b |
| Final full-context pass | about 0.64 × the utterance (0.785 at 20 s), two cores | E12 |
| Front-end reuse in the final pass | identical logits on host, 24 % less CPU; on chip pending | E14 |
| Last words on screen after release | 0.16 s (stream finished first) | E14 |
| Free PSRAM at a 20 s utterance | 2.97 MB of 32 | E12 |
| Live use by people | **not yet done**: only clips played from a laptop | E12, E14 |

## 7. Reference points

| System | Params | TORGO test | UASpeech (dys) | LibriSpeech | Note |
|---|---|---|---|---|---|
| Ours, big7 (float, greedy) | 19 M | 18.46 | 65.46 (94.86) | 4.19 | fine-tuned on TORGO train |
| Parakeet-TDT 1.1B, untuned | 1070 M | 23.14 | 60.80 (90.73) | 1.52 | E15 |
| Parakeet-TDT-CTC 110M, untuned | 115 M | 24.40 | 66.62 (90.99) | 2.62 | E15 |
| SAP Challenge 2025 winner | about 1.1 B | n/a | n/a | n/a | 290 h of real dysarthric speech; 8.11 % on its own Test2; not comparable |

## 8. Threats to validity (keep current)

- **Two dysarthric test speakers.** Report per speaker and with intervals. Planned: rotate which TORGO speakers are held out (several shorter runs), and report the mean.
- **Prompt overlap.** 99 % of TORGO test clips use training prompts. UASpeech and LibriSpeech carry the unfamiliar-material check.
- **One seed per model.** Before submission, three seeds for rows 5–7.
- **Early validation on test speakers.** E1–E3 chose settings while watching test-speaker validation clips. Rows 2 onward keep those settings; say so.
- **Simulated conditions.** Only CP and ALS (TORGO) have real test speech. The other 12 simulated conditions are "simulated, untested", never "supported".
- **Device microphone.** All accuracy numbers are from recordings, not from the board's microphone with live speakers.

## 9. Repeatable experiment card

Every new experiment gets an entry in EXPERIMENTS.md with these fields, in this order, and a row in table 1 or 5 here when it reaches a
status:

```
### E<n>. <title> (<date started>)
Question:   one sentence.
Change:     exactly one thing against a named parent run (or "none: measurement").
Recipe:     train/runs/<name>.cmd (or the script and its arguments).
Data:       which manifests; confirm no test speaker and no UASpeech in anything fitted.
Measure:    engine int8 + LM, TORGO test live and final, paired CI vs parent; UASpeech and LibriSpeech (float greedy).
Result:     table, numbers as printed, file names.
Decision:   keep / drop / deploy, and why.
Finding:    one sentence for the paper (numbered).
```

**Reproducing a row of table 1:**
1. Train: `cmd /c G:\cosgc\repo3\train\runs\<model>.cmd`. The script trains, then runs `eval_manifest.py` on TORGO test, LibriSpeech and UASpeech, then exports `<model>_int8.tnm`.
2. Device-engine numbers, in WSL, from `esp32/host`:
   ```
   python3 eval_cli.py <model>_q121.tnm torgo_test.jsonl --out <model>_q121_full_lm.json --lm ../../models/nemo_lm.tlm
   ```
   Add `--stream` for the live column.
3. Compare: `python train/paired_ci.py <parent>_q121_full_lm_utts.jsonl <model>_q121_full_lm_utts.jsonl` (engine output). The same script works on eval_manifest.py's `*_utts.jsonl` with `--mode full|stream`. It prints all clips, each group and each speaker.

## 10. Still to do for the paper

| Item | Why | Status |
|---|---|---|
| Rotate held-out TORGO speakers (e.g. 3 folds) for the final recipe | two test speakers are too few | planned |
| 3 seeds for big6, big7 and the best simulated run | one seed per model | queued: seeds 1 and 2 of big6 and big7 |
| Simulation as a first stage, then big7's recipe (E20) | C7: does order rescue the simulator? | queued |
| big10: 120 h of transparently rendered simulation, real exposure kept at big7's | C7, C8 | done (F45): no gain on dysarthric TORGO speakers; big7 stays deployed |
| Rotate held-out TORGO speakers, folds B and C (E19) | two test speakers are too few | running (started 2026-11-03) |
| Listening test of the simulator (club members rate real vs simulated, blind) | C8, C14 | planned |
| Live tests on the board: club members, 15 and 60 cm, quiet and noisy, with known prompts | §6 last row; C10 and C11 on the real microphone | needs the board and volunteers |
| Teacher-student training from a fine-tuned Parakeet (CC-BY-4.0; 110 M, since 1.1 B does not fit a 12 GB card for training) (E21) | more accuracy at 19 M | queued; code smoke-tested |
| Cite: TORGO (Rudzicz et al. 2012), UASpeech (Kim et al. 2008), LibriSpeech, MUSAN, Conformer, CTC, SpecAugment, WORLD (Morise et al. 2016), MFA (McAuliffe et al. 2017) and the LibriSpeech alignments (Lugosch et al. 2019), kNN-VC (Baas et al. 2023), Parakeet (NVIDIA), the SAP Challenge (Interspeech 2025), Oído | | add to log §8 |
