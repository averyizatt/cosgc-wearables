# Making the recognizer work for atypical speech: options

This is a plan, not a result. **Nothing here has been trained or measured yet.** The current models were trained on typical
read and conversational speech (LibriSpeech, Common Voice, VoxPopuli, AMI, VCTK, People's Speech; see [NOTICE](../NOTICE)), and we
have no error rates for stuttering, dysarthria, aphasia, quiet (hypophonic) voices, deaf speech or strong accents. The only
related number we have is a simulation: clean speech turned down and mixed with the microphone's noise stays accurate down to
40 dB below normal speech (see the README). Real quiet voices differ from turned-down audio.

"Atypical speech" is several different problems, and they need different fixes:

| Speech | What goes wrong for a typical-speech model | What helps most |
|---|---|---|
| Stuttering (repetitions, blocks, prolongations) | repeated syllables become extra words or garbage; long blocks end utterances early | decide what the output should be (see "What should the text say?"); turn-taking by button, not silence; fine-tuning |
| Dysarthria (cerebral palsy, ALS, Parkinson's, stroke) | slurred, slow, variable speech; low volume; imprecise consonants | speaker-adapted fine-tuning; a restricted vocabulary and domain language model; microphone placement and gain |
| Aphasia | word-finding pauses, wrong words, fragments | tolerant language model (a strong LM can *hurt* here), push-to-talk |
| Quiet or breathy voice | low level, noise dominates | microphone closer, gain, noise augmentation in training |
| Accent / non-native | systematic phoneme differences | accented data (L2-ARCTIC, Common Voice accents), fine-tuning |

## Step 0: measure before you build anything

You cannot tell which of the options below is worth the effort without a baseline.

1. Record 50 to 100 short utterances from each of a few volunteers **on the device itself** (same microphone, same placement), with
   known reference text. Phrases the person actually uses beat read sentences. Keep a separate set you never train on.
2. Run the current firmware or the host tool (`esp32/host/tasr_cli`) on them. Report word error rate **per speaker**, and
   separately the error rate on the few words that matter (names, commands). One average hides a model that fails for one person.
3. Look at the errors. If most are *missing* words, it is level or segmentation (microphone and push-to-talk). If they are
   *wrong* words, it is acoustic mismatch or the language model.

Recording needs a consent form and a storage plan before the first clip (see "People and data"). A small firmware feature that
streams raw microphone audio over the device API would let you collect training data through the same microphone the product uses;
it does not exist yet and is a good first piece of work.

## The options, cheapest first

### A. Adjust what is already there (hours, no training)

- **Microphone distance and gain.** The README's advice stands: closer helps more than software. On the P4 board, `CONFIG_TASR_CODEC_MIC_GAIN`.
- **Push-to-talk instead of silence detection.** Already the default. It is the right design for stuttering and aphasia: the person, not a timer, decides when the sentence ends.
- **Language model weight and token bonus** (`--lm_weight`, `--token_bonus`, or `CONFIG_TASR_LM_WEIGHT_X100`). For fragmented or unusual speech a
  weaker LM can be better (on our four clips, weight 1.0 was far worse than 0.3 to 0.6). Sweep these on the Step 0 set.
- **Chunk size.** Shorter chunks (0.64 s) cost accuracy (upstream: 6.1 against 4.9 % on test-clean); if the person is slow and the wait is acceptable, the full-context model is a third more accurate (3.4 against 4.9 %).

Expected gain: small to moderate, free. Do this first and re-measure.

### B. A language model for the person's own words (days)

The recognizer uses a small GRU language model (1.3 M parameters). `train/lm.py` trains one; `train/export_lm.py` packs it.
Training it on the person's own phrases and domain (a caregiver's commands, a name list, a work vocabulary) plus general text
helps with every kind of speech, because it helps the decoder pick plausible words from unclear sounds. Risks: it will
force output toward its training text, so a rare word it has never seen becomes a wrong common word. Keep general text in the mix
and test on sentences that are **not** in the training phrases.

Expected gain: good for a constrained vocabulary, little for open conversation. No acoustic training needed.

### C. Fine-tune the acoustic model (weeks; needs a GPU)

This is where the large gains are, and the repository already has the machinery:

| Step | Tool in `train/` |
|---|---|
| Get NVIDIA's base model | `fetch_nemo_small.py` |
| Manifest of your audio: one JSON line per clip `{"wav": ..., "text": ..., "duration": ..., "source": ...}` | `data.py` reads it; text is lower-cased to `a-z` and apostrophe |
| Noise and reverberation augmentation | `augment.py` (MUSAN noise, simulated rooms) |
| Streaming fine-tune (chunked attention, 128-frame history) | `train_nemo_stream.py --stream --train ... --init <checkpoint> --keep_head`, with `--kd` (full-context teacher for the chunked student) |
| 4-bit quantization-aware training (smaller, faster on-chip) | `train_nemo_qat.py` (or `--bits 4` in the stream trainer) |
| Export to the chip | `export_nemo.py` then flash as the `model` partition |

Things to know before starting:

1. **The repository has exported models, not training checkpoints.** `--init` wants a checkpoint (float weights). Options: (a) start from NVIDIA's
   `stt_en_conformer_ctc_small` and run the streaming recipe (Lokutor used 23,000 steps in two stages: expensive); (b) ask upstream whether the
   float checkpoints exist; (c) write a small tool that dequantizes `oido_stream.tnm` back to floats (the int8 weights are a close
   approximation, and fine-tuning repairs the rounding). Option (c) is now `train/tnm_to_ckpt.py` (verified: re-exporting
   its output reproduces `oido_stream.tnm` byte for byte).
2. **Mix, do not replace.** Training only on atypical speech makes the model forget typical speech (and the model must still work for
   everyone else around the user). Keep a share of LibriSpeech/Common Voice in every batch.
3. **Augmentation matters more with little data.** Add speed perturbation (about 0.9x to 1.1x; not in `augment.py` yet, easy to add),
   volume changes, and noise. Dysarthric speech is slow, so wider tempo changes (0.8x) are reasonable.
4. **Group model or personal model?** A group model (many speakers with one condition) generalizes to a new person but gains less per person. A
   personal model (one speaker, a few hours) gains the most and does not transfer. A practical scheme is a group model fine-tuned briefly
   per person, which needs far less personal data.

Public datasets that exist for this (**check each license and access terms yourself before using; several are research-only or need an application, and we have not verified them**):

| Dataset | What | Access |
|---|---|---|
| Speech Accessibility Project (University of Illinois) | large, growing; Parkinson's, ALS, cerebral palsy, Down syndrome, stroke | application and data-use agreement |
| TORGO | dysarthria, about 15 speakers, with articulatory data | research license |
| UASpeech | dysarthria, 19 speakers, mostly isolated words | research use, on request |
| FluencyBank, SEP-28k, LibriStutter | stuttering (mostly labeled events, little transcription) | varies |
| AphasiaBank (TalkBank) | aphasia, conversation | membership |
| L2-ARCTIC, Common Voice accents | accents | open |

Our existing models are released under CC-BY-SA because of share-alike sources. **Data from people with a medical condition and
data with a research-only license can change what we are allowed to release or ship.** See [LICENSING.md](../LICENSING.md) and ask before shipping a model trained on it.

Expected gain: this is the option that can turn "unusable" into "usable" for some speakers, and the only one that can do it for dysarthria.
The gain is uneven; severe cases improve least. Plan a validation set per speaker.

### D. What should the text say? (a decision, not a technique)

For stuttering in particular, the *target transcript* is a design choice that changes what you train on:

- **Verbatim**: "I w-w-want to go". Honest, awkward to read, and the model must learn to spell the repetitions.
- **Clean**: "I want to go". More readable, but the recognizer must learn to skip repeated fragments, which is harder and needs transcripts written that way.
- **Clean with a marker** (or a toggle in the API): both outputs are possible from one model with a post-process.

Ask the people who will use it. For dysarthria and aphasia the equivalent question is whether the device should guess the intended word or
show what it heard. The device API can carry both (`partial` and `final` text today).

### E. Not recommended now

- **Training a model from scratch**: needs thousands of hours and a cluster.
- **Replacing the engine** with a larger model (Whisper and similar): they do not fit the chip's memory and speed; they are an option only for a phone companion app.
- **Cloud recognition**: removes the main promise of this project (nothing leaves the device).

## Recipe: a combined typical + atypical model, distilled from a larger teacher

The student is what runs on the chip: the same Conformer-CTC small (13 M parameters, 14 MB int8 or 8 MB int4) the engine runs
today. The teacher can be any size, because it only runs on the training machine. Commands are run from `train/` on a machine with an
NVIDIA GPU, PyTorch, `sentencepiece`, `soundfile` and `jiwer` (the training scripts' imports).

1. **Starting weights** (done once; works on this PC too, without PyTorch it writes the `.npz` only):
   ```
   python tnm_to_ckpt.py ../models/oido_stream.tnm ../exp/base --verify
   cp ../models/nemo_small/tokenizer.model ../exp/base/
   ```
   gives `exp/base/state_dict_plain.pt` (the `--nemo` directory) and `exp/base/init.pt` (the `--init` checkpoint, with the
   model's fixed feature normalization).
2. **Manifests**: one JSON line per clip, `{"wav": "/data/x.wav", "text": "...", "duration": 3.2, "source": "torgo"}`, 16 kHz mono.
   Typical speech (LibriSpeech, Common Voice) in one or more files, atypical speech in another, a held-out file per atypical speaker.
3. **Teacher labels** for atypical audio with missing or unreliable transcripts: fine-tune or pick a large model (NVIDIA's Conformer
   large or Parakeet), then `filter_teacher.py --teacher <dir> --lang en --out_dir scored atypical.jsonl` writes its transcript and its
   WER against your reference; `--max_twer` in the trainer drops clips the teacher disagrees with. Where there is no reference, use the
   teacher's text as the label (pseudo-labelling), after listening to a sample of it.
4. **Train the student** (typical speech kept in the mix so it does not forget; the in-place distillation `--kd` keeps the
   streaming student close to its own full-context output):
   ```
   python train_nemo_stream.py --nemo ../exp/base --init ../exp/base/init.pt --keep_head        --train typical.jsonl,atypical.jsonl --val atypical_heldout.jsonl --out ../exp/atyp1        --stream --kd 0.5 --steps 4000 --lr 2e-4 --aug_dir <augment.py banks>
   ```
   Short and gentle first (a few thousand steps, a lower learning rate than a from-scratch run): this is adaptation, not training.
   Check per-speaker WER on the held-out files every `--val_every` steps.
5. **Export and test on the host, then the chip**: `python export_nemo.py ../exp/base model.tnm 8 ../exp/atyp1/last.pt`, then
   `esp32/host/tasr_cli model.tnm --lm ../models/nemo_lm.tlm <wavs>` and `esp32/host/stress_run.py`, then flash it as `custom_model`.

What is **not** in the code yet: distilling from an *external* teacher's frame posteriors (soft targets). Today `--kd` uses the
student's own full-context copy as teacher, and an external teacher only contributes transcripts (step 3). Adding soft targets
needs the teacher and student to share the 1,024-piece tokenizer and about 50 to 100 lines in `train_nemo_stream.py`'s loss.

## People and data

Audio of someone's speech, especially with a health condition, is sensitive. Before collecting anything:

- Get informed written consent that says what is recorded, who can hear it, where it is stored, for how long, and that the person
  can withdraw. If this is part of a university or club project, ask whether it needs ethics (IRB) review.
- Store the audio encrypted, and keep the names and the audio apart.
- Training a model on one person's voice makes a file that can leak how they speak. Treat a personal model as personal data.
- This is not a medical device. If transcripts will be used in care, validate on the people it is for.

## Suggested order

1. A: tune and measure on 3 to 5 volunteers (Step 0). Decide from the per-speaker errors whether the problem is level, language, or sound.
2. Add raw-audio capture to the device API so training data comes from the real microphone.
3. B: a domain language model if the vocabulary is constrained.
4. C: a small fine-tune on one group's data (apply for one dataset now: access can take weeks), keeping typical speech in the mix; compare with the baseline **per speaker**.
5. Only then decide on personal models.
