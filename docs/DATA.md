# Training and evaluation data

What we downloaded for fine-tuning and testing, where it came from, under which terms, and what we may and may not do with it.
None of this data is in the git repository: it lives on the training PC at `G:\cosgc\data` (the 3 TB drive). The download script
is [`train/fetch_data.ps1`](../train/fetch_data.ps1). It records every archive's size and SHA-256 hash in
`G:\cosgc\data\fetch_log.tsv`. The hash table at the end of this file is copied from that log.

**The short version on terms:** LibriSpeech and MUSAN can be used for anything with attribution. **TORGO is free for academic,
non-profit use only, and anyone who publishes with it must cite its papers.** A model trained on TORGO is fine for this club
project, for research and for demos. It must not be sold or shipped in a product without asking the University of Toronto. Keep track of which
models saw TORGO (see "Models and their data" below).

## Datasets

### LibriSpeech (typical read English)

| | |
|---|---|
| What | Read audiobook English from LibriVox, 16 kHz FLAC, with transcripts |
| Source | https://www.openslr.org/12 (OpenSLR resource 12), Panayotov, Chen, Povey, Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015 |
| License | CC-BY-4.0 (the text in `LibriSpeech/LICENSE.TXT`). Use for anything, with attribution. |
| Parts we use | `train-clean-100` (100 h, training: the "typical speech" share of every batch), `dev-clean` (5.4 h, smoke tests and validation), `test-clean` and `test-other` (evaluation; never trained on) |
| Location | `G:\cosgc\data\LibriSpeech\<part>` |
| Why | Keeps the model from forgetting typical speech while it learns atypical speech, and gives a standard test to compare against the published numbers (README: 4.9 % test-clean streaming) |

The `us.openslr.org` mirror served a certificate for a different name on 2026-10-29, so the script uses `www.openslr.org` only.

### TORGO (dysarthric and control speech)

| | |
|---|---|
| What | Speech from 8 speakers with dysarthria (cerebral palsy or ALS: F01, F03, F04, M01–M05) and 7 control speakers without it (FC01–FC03, MC01–MC04): single words, sentences, and non-words, two microphones (head-mounted and array), plus articulatory (EMA) data for some sessions. About 18 GB unpacked. |
| Source | https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html, University of Toronto and Holland-Bloorview Kids Rehabilitation Hospital |
| Terms (quoted) | "Use of this database is free for academic (non-profit) purposes." Provided as-is. |
| Citation required | Anything published with it must cite at least one of: Rudzicz, Namasivayam, Wolff (2012), "The TORGO database of acoustic and articulatory speech from speakers with dysarthria", *Language Resources and Evaluation* 46(4) (the one describing the database); Rudzicz, Hirst, van Lieshout (2012), *JSLHR* 55(4); Rudzicz (2012), *Speech Communication* 54(3). |
| Files | `F.tar.bz2` (females with dysarthria), `FC.tar.bz2` (female controls), `M.tar.bz2` (males with dysarthria), `MC.tar.bz2` (male controls) |
| Location | `G:\cosgc\data\TORGO\<speaker>\Session*\` (F01, F03, F04, FC01–FC03, M01–M05, MC01–MC04) |
| Why | The only free, directly downloadable dysarthric speech corpus we found. Small (8 speakers), so it is best for adaptation and for **evaluation per speaker**, not as the only training data. |
| Caveats | Much of it is single words and short prompts, not conversation. Some prompts are image descriptions without a transcript. Speakers' severity varies widely. Keep each speaker's held-out data separate, and never put the same speaker in both training and test when reporting results. |

### MUSAN (noise, music and speech for augmentation)

| | |
|---|---|
| What | About 109 h of noise, music and speech recordings, used to mix background sound into training audio (`train/augment.py`) |
| Source | https://www.openslr.org/17 (OpenSLR resource 17), Snyder, Chen, Povey, "MUSAN: A Music, Speech, and Noise Corpus", 2015, arXiv:1510.08484 |
| License | CC-BY-4.0 (the README in the archive lists any parts under other terms; check it before redistributing) |
| Location | `G:\cosgc\data\musan` |
| Why | The shipped streaming model was trained with MUSAN noise. Fine-tuning without it would make the model less robust to noise. |

### What we did not download, and why

| Dataset | Why not (yet) |
|---|---|
| Speech Accessibility Project (University of Illinois) | Needs an application and a data-use agreement. The largest and most useful atypical-speech dataset (Parkinson's, ALS, cerebral palsy, Down syndrome, stroke). **Worth applying now.** |
| UASpeech | Request by email to uaspeech-requests@lists.illinois.edu (https://speechtechnology.web.illinois.edu/uaspeech/), academic and government labs only, no commercial use. 16 speakers with dysarthria (cerebral palsy) plus controls, mostly isolated words. |
| FluencyBank, AphasiaBank (TalkBank) | Need registration or membership. Stuttering and aphasia with transcripts. |
| SEP-28k | Labels only, the audio has to be scraped from podcasts. Stutter events without transcripts. |
| L2-ARCTIC | Request form, CC-BY-NC-4.0. Accented English. |
| Common Voice | Needs an account on Mozilla's data portal. CC0. Useful later for more speaker variety. |

## Models and their data

| Model | Trained on | May be shipped? |
|---|---|---|
| `models/oido_stream.tnm` (as shipped) | see NOTICE (LibriSpeech, Common Voice, VoxPopuli, AMI, VCTK, People's Speech, OpenSLR 70/83, MUSAN) | yes, CC-BY-SA-4.0 |
| `G:\cosgc\exp\smoke` (smoke test, 2026-10-29) | the above + LibriSpeech dev-clean (200 steps) | test only, never exported |
| `G:\cosgc\exp\atyp1` (2026-10-29, 4000 steps) | the above + LibriSpeech train-clean-100 + TORGO (all but F03, M05, FC03, MC04) | test only: streaming on typical speech regressed (see below) |
| anything fine-tuned with TORGO | + TORGO | **academic / non-profit only**; cite TORGO |

Add a row for every model you export.

### Results so far

Word error rate in %, greedy decoding without LM (`train/eval_manifest.py`), as full utterance / streaming (chunk 32,
left 128, as on the device). TORGO speakers below were never trained on.

| Test | shipped model | atyp1 |
|---|---|---|
| TORGO F03 (dysarthric) | 49.5 / 57.0 | 38.2 / 49.3 |
| TORGO M05 (dysarthric) | 80.0 / 89.7 | 64.3 / 74.8 |
| TORGO FC03 (control) | 17.4 / 23.6 | 10.6 / 17.8 |
| TORGO MC04 (control) | 16.5 / 21.7 | 12.0 / 18.5 |
| TORGO single words | 54.6 / 59.8 | 32.4 / 39.5 |
| TORGO sentences | 21.4 / 28.5 | 19.0 / 27.6 |
| LibriSpeech test-clean | 4.0 / 6.2 | 4.6 / 9.7 |

atyp1 (lr 1e-4, distillation 0.5, 4000 steps of 150 s batches) helps every TORGO speaker, mostly on single words, but
its streaming error on typical speech rose by half within the first 500 steps and stayed there.

## Archive hashes

Copied from `G:\cosgc\data\fetch_log.tsv`. Re-run `fetch_data.ps1` on another machine and compare to check you have the same data.

All downloaded 2026-10-29, all unpacked without errors. About 35 GB of archives; they are kept next to the unpacked data.

| Archive | Bytes | SHA-256 |
|---|---|---|
| LibriSpeech `train-clean-100.tar.gz` | 6,387,309,499 | `D4DDD1D5A6AB303066F14971D768EE43278A5F2A0AA43DC716B0E64ECBBBF6E2` |
| LibriSpeech `dev-clean.tar.gz` | 337,926,286 | `76F87D090650617FCA0CAC8F88B9416E0EBF80350ACB97B343A85FA903728AB3` |
| LibriSpeech `test-clean.tar.gz` | 346,663,984 | `39FDE525E59672DC6D1551919B1478F724438A95AA55F874B576BE21967E6C23` |
| LibriSpeech `test-other.tar.gz` | 328,757,843 | `D09C181BBA5CF717B3DEE7D4D592AF11A3EE3A09E08AE025C5506F6EBE961C29` |
| TORGO `F.tar.bz2` | 1,139,746,435 | `ED22D12EF3AC4511E99EC15B13DDCD42BAFD6FBAE939A749BDD23D761A8C462E` |
| TORGO `FC.tar.bz2` | 2,532,029,976 | `1DEBF8B214656591152EFB94F465C22DD067B12A6DC41851EB70651B5717EE31` |
| TORGO `M.tar.bz2` | 2,506,634,222 | `FDED3B14C0B54F58803AFDDB53FB98A826F3584CFDA918BFCF25C798D994D1D7` |
| TORGO `MC.tar.bz2` | 3,398,783,713 | `DC11AB0257B82065D7A32C7E816E039107B7B3228633F8E03592F1350CD41565` |
| MUSAN `musan.tar.gz` | 11,086,114,085 | `86D1061C7E15B5C9E906777685C519701DF51BFDE3001E1070DCC9FFAC955EE1` |

`dev-clean` was downloaded by hand before the script existed, so it has no line in `fetch_log.tsv`; its hash was computed afterwards.
The OpenSLR servers sent about 4 to 5 MB/s per connection, so the full set took about two hours on a 1 Gbps line.
