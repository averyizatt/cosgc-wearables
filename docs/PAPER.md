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
grid, full-context final text. The gain holds when every TORGO speaker is held out in turn. Simulated dysarthric speech is our route to more training data than the
public corpora hold. So far it helps a little: on control speakers, and on dysarthric speakers' live text, but not on their final
text. An untuned 1.1 B-parameter model (Parakeet-TDT) is no better on dysarthric speakers than ours. All results use
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
| 5 | big6 | dysarthric share of audio 6.5 → 26 % | 23.86 / 46.92 (3 seeds: 23.47 ± 0.35 / 46.58 ± 0.33) | 17.74 / 36.56 (3 seeds: 17.51 ± 0.21 / 36.20 ± 0.32) | E13, Seeds | solid |
| 6 | big7 (on the board) | + mel-level simulated dysarthria on 30 % of LibriSpeech clips | 23.31 / 45.99 (3 seeds: 23.27 ± 0.24 / 45.88 ± 0.35) | 17.03 / 35.43 (3 seeds: 16.98 ± 0.04 / 35.14 ± 0.26) | E15, Seeds | solid: over 3 seeds, final −0.53 (CI −0.75..−0.28), dysarthric −1.06 (−1.65..−0.47) (F49) |
| 7 | big9 | + 120 h of waveform-simulated speech (round 2), same steps | 25.09 / 46.75 | 18.41 / 36.76 | E15 | worse (CI +0.94..+1.84); confounded: real data seen half as often (F38) |
| 8 | big10 | from row 6: the mel-level simulation replaced by 100 source hours of transparently rendered waveform simulation (round 4), steps scaled to keep real exposure | 22.80 / 46.04 | 16.64 / 35.41 | E16 | no change on dysarthric speakers (CI −1.47..+1.99); controls better; UASpeech dysarthric better, controls worse (F45) |
| 9 | simpre | from row 6: the waveform simulation as a first stage (simtr360 + LibriSpeech, 4000 steps), then row 6's recipe | 23.78 / 46.34 | 17.19 / 35.29 | E20 | no change (final CI −0.20..+0.56; dysarthric −1.04..+0.76); live slightly worse; UASpeech dysarthric better (F48) |
| 10 | big11 | from row 6: + distillation from a fine-tuned Parakeet 110M (weight 0.5, T 2, full-context batches) | 24.26 / 46.78 | 17.84 / 35.98 | E21 | worse with the LM (final CI +0.38..+1.26); equal without it (−0.50..+0.41), so the loss is the LM step (F51); LibriSpeech 4.19 → 5.79 in float (F50) |
| 11 | big12 | from row 10: the teacher term only on dysarthric clips | 23.45 / 46.73 | 17.96 / 37.52 | E22 | worse with the LM (final CI +0.43..+1.47; M05 +5.6); without it better than row 6 (−0.46, CI −1.00..+0.03; controls −0.58); float: UASpeech dysarthric 89.90 → 81.30, LibriSpeech 4.20 (F51). Needs an LM retune on a development set |
| 12 | big13 | from row 6: + UASpeech B1/B3 of 20 speakers, 8 held out whole; 6500 steps | 23.95 / 47.17 | 17.26 / 35.89 | E24 | no change on TORGO final (CI −0.17..+0.62), live slightly worse (+0.18..+1.11); held-out UASpeech speakers 60.09 → 42.25 float, controls as much as dysarthric (F53); not deployed |

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
| C7 | Mel-level simulated dysarthria helps slightly at 24 layers. Over three folds (all 12 TORGO speakers held out in turn), it lowers final WER by 0.6 points (CI −0.8..−0.3). That gain comes from control speakers and from dysarthric speakers' live text (−0.8, CI −1.4..−0.2), not their final text (−0.3, CI −0.8..+0.2; fold A's −1.0 did not replicate in folds B and C, though on fold A it holds for all three seeds, pooled −1.06, CI −1.65..−0.47: it depends on the held-out speakers, not the seed, F49). It did not help at 16 layers. Scaling up to 100 source hours of waveform simulation, real exposure held fixed, does not help TORGO's dysarthric speakers (35.43 → 35.41), and neither does using it as a first stage (35.29, F48). Both move the model toward unseen dysarthric speech (UASpeech dysarthric −1.6 to −2.5 points) and away from unseen typical speech (UASpeech controls +1.5). | E4, E15, E16, E19, E20, Seeds (F11, F33, F38, F45, F47, F48, F49) | solid across folds (one seed per fold) and across 3 seeds on fold A; the large-scale result is solid for this simulator |
| C16 | The fine-tune's gain is not specific to the official split. With each of three disjoint sets of dysarthric speakers held out, it lowers their WER by 11–20 points (26–36 % relative; mean over folds 48.4 → 33.4, float greedy). Controls go 17.6 → 9.3 and LibriSpeech stays at 4.1–4.2. Severe speakers stay above 50 % in every fold. | E19 (F46) | solid (3 folds, one seed each) |
| C8 | A signal-level simulator matches real dysarthric speech on rate, pausing, rhythm and spectral tilt; with sound-level errors from phone alignments and measured error tables, its dysarthria conditions span every TORGO training speaker by WER (severity 0.3–1.5). It is still separable from real dysarthric speech (AUC 0.99) and sounds less dysarthric to a probe than real severe speech. | E5, E15, E16 (F32, F39, F43) | solid (measurement); in training it does not help TORGO's dysarthric speakers (big10, F45) |
| C9 | Size alone does not solve dysarthric speech: an untuned 1.1 B model scores 90.7 % on UASpeech dysarthric speakers (ours 94.9) and 23.1 % on TORGO test (ours 18.5, float greedy, prompt-familiar). | E15 (F35) | solid |
| C10 | The model's own token posteriors flag errors: AUC 0.894; utterances marked "sure" (54 %) are 95 % fully right, "probably wrong" (24 %) 27 %. | E17 (F36) | solid on TORGO; device microphone pending |
| C11 | On the board's microphone, signal-to-noise ratio matters and level control does not; noise reduction helps only near 8 dB SNR. | E14 (F28, F29) | tentative (simulated noise) |
| C12 | Bit-exact kernels do not guarantee identical transcripts across platforms: dynamic int8 quantization amplifies 1e-6 input differences to about 1 % in the logits. | E12 (F24) | solid |
| C13 | A user's short phrase list, scored by CTC on the device, raises fully right utterances for dysarthric speakers from 50 % to 84 % (25 phrases) or 89 % (10), with 5 % of off-list speech wrongly matched. | E18 (F37) | tentative (threshold fitted on the test clips; TORGO prompts) |
| C14 | Vocoder resynthesis is a hidden confound in simulated speech: WORLD with no change at all costs 11 points of WER on TORGO's recordings (0.4 on clean LibriSpeech) and is perfectly separable from the original; it supplied about half of what made the simulation sound dysarthric. Editing the recording in place (PSOLA + filtering) removes it. | E16 (F40, F42, F44) | solid |
| C17 | Training on one corpus's other speakers helps that corpus's unseen speakers a lot (UASpeech held-out 60.1 → 42.3, all 8 better) but controls gain as much as dysarthric speakers and TORGO does not move: the gain is mostly the corpus, not dysarthria. | E24 (F53) | solid (8 speakers, one seed) |
| C15 | Voice conversion (kNN-VC) keeps the dysarthria of real recordings (probe 96 %) but adds 7–10 points of its own. | E16 (F41) | solid (measurement); not used in training |

**Negative results worth reporting:**
- Frame-level in-place distillation damaged an already-streaming model (E2).
- Adding UASpeech to training made TORGO worse, 25.14 → 27.55 (E11, F22).
- More dysarthric share beyond 26 % gave no broad gain (big8, F31).
- Language-model weight tuning, a wider beam and padding gave at most 0.3 points (F27).
- 100 source hours of the waveform simulator, with real exposure held fixed, left TORGO's dysarthric speakers unchanged (big10, F45).
- Putting it first, as a pre-training stage, did not either (simpre, F48).
- Distillation from Parakeet 110M, refitted to our tokenizer, made the deployed system worse (final +0.8, CI +0.4..+1.3), but through the language-model step: without the LM it equals big7, and limited to dysarthric clips (big12) it is better without the LM and best on UASpeech in float (F50, F51).

## 6. Device (table 2)

| Measure | Value | Entry |
|---|---|---|
| Model size (int8) / LM | 20.4 MB / 1.29 MB | E9b |
| Live streaming speed, 24 layers | 0.602 × real time on two cores (0.967 on one) | E9b |
| Final full-context pass | about 0.64 × the utterance (0.785 at 20 s), two cores | E12 |
| Front-end reuse in the final pass | identical logits on host, 24 % less CPU; on chip pending | E14 |
| Last words on screen after release | 0.16 s (stream finished first) | E14 |
| Free PSRAM at a 20 s utterance | 2.97 MB of 32 | E12 |
| Live use by people | one typical speaker, 35 prompts at about 15 cm: WER 7.9 % shown (10.2 % without phrase mode), 29 of 35 fully right, no wrong phrase; final text 1.8 s after release (median). No dysarthric speaker yet | E23 (F52) |

## 7. Reference points

| System | Params | TORGO test | UASpeech (dys) | LibriSpeech | Note |
|---|---|---|---|---|---|
| Ours, big7 (float, greedy) | 19 M | 18.46 | 65.46 (94.86) | 4.19 | fine-tuned on TORGO train |
| Parakeet-TDT 1.1B, untuned | 1070 M | 23.14 | 60.80 (90.73) | 1.52 | E15 |
| Parakeet-TDT-CTC 110M, untuned | 115 M | 24.40 | 66.62 (90.99) | 2.62 | E15 |
| SAP Challenge 2025 winner | about 1.1 B | n/a | n/a | n/a | 290 h of real dysarthric speech; 8.11 % on its own Test2; not comparable |

## 8. Threats to validity (keep current)

- **Two dysarthric test speakers.** Report per speaker and with intervals. Addressed for big6 and big7 by three folds that hold out all 12 TORGO speakers in turn (E19, C16, float greedy). Table 1's device numbers are still fold A only. A small leak: big7's simulation settings were calibrated on speakers that folds B and C hold out.
- **Prompt overlap.** 99 % of TORGO test clips use training prompts. UASpeech and LibriSpeech carry the unfamiliar-material check.
- **One seed per model.** Addressed for rows 5 and 6: three seeds each, engine WER spread (sd) 0.04–0.35 points; big7 beats big6 on final text with every seed (F49). Other rows are one seed, so differences of about 0.5 points between them are within seed noise.
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
| 3 seeds for big6, big7 and the best simulated run | one seed per model | done for big6 and big7 2026-11-05 (F49): seed sd 0.04–0.35 points; big7 > big6 on final text for every seed; no simulated run beat big7, so none needs seeds |
| Simulation as a first stage, then big7's recipe (E20) | C7: does order rescue the simulator? | done (F48): no; big7 stays deployed |
| big10: 120 h of transparently rendered simulation, real exposure kept at big7's | C7, C8 | done (F45): no gain on dysarthric TORGO speakers; big7 stays deployed |
| Rotate held-out TORGO speakers, folds B and C (E19) | two test speakers are too few | done 2026-11-05 (F46, F47): the fine-tune's gain holds in every fold; big7's gain over big6 holds for controls and live text, not dysarthric final text |
| Listening test of the simulator (club members rate real vs simulated, blind) | C8, C14 | planned |
| Live tests on the board: club members, 15 and 60 cm, quiet and noisy, with known prompts | §6 last row; C10 and C11 on the real microphone | needs the board and volunteers |
| Teacher-student training from a fine-tuned Parakeet (CC-BY-4.0; 110 M, since 1.1 B does not fit a 12 GB card for training) (E21) | more accuracy at 19 M | done 2026-11-06 (F50, F51): big11 and big12 lose to big7 only through the LM step; big12 is better without it |
| LM weight and token bonus for distilled models (big12), tuned on a development set that is not TORGO test | F51: the LM's gain shrank from 1.6 to 0.2 points; E13 also tuned on TORGO test | running (E25): dev set = folds B and C held-out speakers; fold big12 models train 2026-11-07 |
| Cite: TORGO (Rudzicz et al. 2012), UASpeech (Kim et al. 2008), LibriSpeech, MUSAN, Conformer, CTC, SpecAugment, WORLD (Morise et al. 2016), MFA (McAuliffe et al. 2017) and the LibriSpeech alignments (Lugosch et al. 2019), kNN-VC (Baas et al. 2023), Parakeet (NVIDIA), the SAP Challenge (Interspeech 2025), Oído | | add to log §8 |
