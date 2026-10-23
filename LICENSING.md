# Licensing

This project is **cosgc-wearables**, a modified version of **Oído** by Lokutor (Hashing Works, S.L.L.),
<https://github.com/lokutor-ai/oido>. Everything here builds on their engine and models, so their licenses apply to
ours. This page is a plain-language summary for the people working on the project. It is **not legal advice**; read
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and [`COMMERCIAL.md`](COMMERCIAL.md) for the actual terms, and ask a lawyer
before shipping a product.

## What is licensed how

| Part | License | Where |
|---|---|---|
| Engine, firmware, tools, and our own code (display, LED, button, API, benchmarks) | **GNU GPL v3** | [`LICENSE`](LICENSE) |
| `models/nemo8.tnm`, `models/nemo_lm.tlm` | CC-BY-4.0 | [`NOTICE`](NOTICE) |
| `models/oido_es.tnm`, `models/oido_es.tlm` | CC-BY-4.0 | [`NOTICE`](NOTICE) |
| `models/nemo4.tnm`, `models/oido_stream.tnm` (the model the firmware uses by default) | **CC-BY-SA-4.0** (share-alike) | [`NOTICE`](NOTICE) |
| The acoustic model underneath all of them | NVIDIA `stt_en_conformer_ctc_small`, CC-BY-4.0 | [`NOTICE`](NOTICE) |
| ESP-IDF and the libraries it brings (FreeRTOS, NimBLE, cJSON, ...) | their own licenses (mostly Apache-2.0 and MIT) | check each before shipping |

## What that means in practice

**Using it, modifying it, building devices for yourselves or a lab:** fine under the GPL. Keep the license and
copyright notices, and mark what you changed (see "Our changes" below).

**Giving or selling firmware or devices to other people:**
- You must provide the corresponding source code (the GPL). `COMMERCIAL.md` adds that for consumer devices this includes
  *the information needed to install modified firmware*, so locking the firmware down is not allowed under the free license.
- Our own API, Bluetooth and display code is linked into the same firmware, so it is under the GPL as well.
- If those terms do not fit the product, Lokutor sells a **commercial license** without GPL obligations
  (`COMMERCIAL.md`, contact@lokutor.com). Their prices and terms are theirs to set; we have not asked.

**Models:**
- Credit NVIDIA and Lokutor as `NOTICE` says (CC-BY).
- A model fine-tuned from `oido_stream.tnm` or `nemo4.tnm` has to be released under CC-BY-SA-4.0 too. A model fine-tuned
  from `nemo8.tnm` or the NVIDIA checkpoint is CC-BY (credit required), unless the data you train it on brings its own
  share-alike terms.
- Training on recordings of people needs their consent, and public datasets for atypical speech (Speech Accessibility
  Project, TORGO, UASpeech, FluencyBank, AphasiaBank) come with their own data-use agreements. Read them first.

**Medical use:** nothing in this repository is a medical device, and the licenses say nothing about regulation. If
transcripts are used in care, the error rates need validating on the people it is for, and medical-software rules
(for example FDA or EU MDR, depending on the claim and the country) may apply.

## Our changes

The GPL asks modified files to say so. [`NOTICE`](NOTICE) lists, with dates, what we changed in Oído's files and which files
are new. New files carry a copyright line for "the cosgc-wearables contributors" (replace this with the legal name of the
copyright holder once there is one). Oído's own copyright lines stay untouched.

## Before shipping anything, check

- [ ] The license path is decided: GPL (source and installable firmware for every user) or a commercial license.
- [ ] `NOTICE` and the model credits ship with the product (a printed or on-screen "open source licenses" page).
- [ ] The third-party components in the final firmware are listed with their licenses (including the Bluetooth stack).
- [ ] Any model trained on people's recordings: consent on file, dataset agreements followed, nothing identifying in the repo.
- [ ] The privacy story is true: transcripts are health-adjacent data. Bluetooth pairing is encrypted, nothing is logged
      to flash, and nothing leaves the device unless the user's phone asks for it.
