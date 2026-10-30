# cosgc-wearables

**Offline speech-to-text for a wearable.** Hold a button, speak, and the text appears on a small screen and goes out over
USB (and Bluetooth) as JSON. It all runs on one ESP32-S3 chip: no phone, no cloud, nothing leaves the device.

This project builds on **[Oído](https://github.com/lokutor-ai/oido) by Lokutor**, the speech engine and models that make
on-chip recognition possible (their original README is kept as [README.upstream.md](README.upstream.md)). Everything
around it is ours: the push-to-talk firmware, the display and LED, a lot of speed work on the engine, and the device API.

## What it does

- **Push-to-talk.** Words appear while you speak; the final text comes about **2 seconds after you let go** on two cores
  (4 to 5 on one core), with no end-of-speech pause to wait for. Long sentences work (we tried 30 seconds).
- **Screen, LED and level bar.** A 128×64 OLED shows the transcript and a live microphone level bar, so you can see
  at once whether the mic is close enough. A status LED shows what the device is doing.
- **Device API.** One JSON line per message: partial and final text, button, state, microphone level. Over USB today;
  Bluetooth LE carries the same lines ([docs/API.md](docs/API.md)).
- **Private by design.** The speech model runs on the chip. Nothing is stored, and nothing is sent anywhere unless a
  client you connected asks for it.
- **English** with the streaming model (NVIDIA's Conformer, fine-tuned by Lokutor, plus a small on-chip language model).
  The engine also supports Oído's Spanish model; we have not tested it in our firmware.

## Hardware

| Part | Notes | Pins |
|---|---|---|
| ESP32-S3 board, **N16R8** (16 MB flash, 8 MB octal PSRAM) | ESP32-S3-DevKitC-1 or a clone. Other variants do not have enough memory | |
| INMP441 I2S microphone | left channel (L/R to GND) | SCK 4, WS 5, SD 6 |
| SSD1306 128×64 I2C OLED | | SDA 8, SCL 9 |
| Push button | to GND, internal pull-up | GPIO 7 |
| On-board RGB LED (WS2812) | | GPIO 48 (38 on DevKitC-1 v1.1) |

Details and troubleshooting are in [WIRING.md](WIRING.md). Pins live in
[`esp32/firmware/sdkconfig.myboard`](esp32/firmware/sdkconfig.myboard).

## Build and flash

You need [PlatformIO](https://platformio.org/) (VS Code extension or CLI). **The project path must not contain spaces**
(ESP-IDF refuses it; on Windows map a drive letter with `subst`, see the top of `platformio.ini`).

```
pio run -e oido_s3_2c -t upload     # flash app + model + language model (about 3 minutes, the model is 14 MB)
pio device monitor                   # console log, 115200 baud
```

| Environment | What it is |
|---|---|
| `oido_s3` | one core, streaming model: the safe default |
| `oido_s3_2c` | **both cores: about 1.3× faster**, experimental (see Status) |
| `oido_s3_accurate` | most accurate English model, text appears after you stop (utterance mode) |
| `oido_s3_ble`, `oido_s3_ble_dev` | two cores plus the Bluetooth LE service; `_dev` pairs without a PIN (for development only) |
| `oido_s3_bench*` | speed benchmarks with test clips ([WIRING.md](WIRING.md#environments)) |

## Using it

1. Power it on. The LED is dim blue while the model loads (about 3 seconds), then dim green: ready.
2. **Hold the button and speak.** The LED is blue, the screen shows `hearing...`, and the words appear as you go.
3. **Let go.** The LED turns amber while the last words are computed, then flashes green and the text is final.
   A red flash means nothing was recognized or the press was too short.

The **level bar** in the top-right corner is the microphone level. Keep it past the small tick mark while you speak: that is
about the quietest speech that still recognizes well (about 5 % errors in our simulation; 10 dB quieter than that,
about 9 %). If the bar stays short of the tick, move the microphone closer to your mouth. That helps more than
anything in software. A purple LED means the chip fell behind the speaker and dropped audio.

## Speed

Measured on our board, same four test clips (24 seconds of speech), streaming model with the language model:

| | Oído as released | This project, one core | This project, two cores |
|---|---|---|---|
| Compute time per second of speech (lower is better, 1.0 = real time) | 1.93 | 1.54 | **1.19** |
| Final text after a 7 s sentence ends | 7.2 s | 4.5 s | **2.2 s** |

Transcripts are identical in all three, so the speed-up costs no accuracy. The chip is still slower than real time,
so very long stretches of continuous speech build up a delay. The 32-second audio buffer absorbs about a minute of
non-stop talking on one core and several minutes on two; beyond that, audio is dropped (purple LED). Where the speed came from: fused output stages and a faster int8 kernel for short rows in the
engine, attention split across the two cores, data-cache prefetch of the weights, an int4 language model, and
`-O3`. Details are in the git history of the `engine-speedups` branch.

## Accuracy

LibriSpeech with the same engine run on a laptop (500 utterances per set), streaming model with the language model:
**4.9 % word error rate on test-clean, 10.7 % on test-other** (3.2 % and 7.0 % in the slower utterance mode,
`oido_s3_accurate`). Oído's README has the full tables and a comparison with other systems.

**Quiet speech** (simulation: clean speech turned down and mixed with the microphone's rated noise): no loss down to
40 dB below normal speech, 8.5 % errors at 50 dB below, unusable at 60 dB. Real quiet voices differ from turned-down
audio, so treat this as a guide only.

**Not measured yet:** speech of people who stutter or have dysarthria, accented speech, noisy rooms. The model is trained
on typical read speech; for atypical speech the next step is fine-tuning on recordings of the people it is for.

## ESP32-P4 (in progress)

A port to the Waveshare **ESP32-P4-Pico** (32 MB PSRAM and flash, onboard microphone) builds but **has not run on a board**: the
firmware environment `oido_p4`, the codec microphone driver, and vector kernels with a boot self-test that falls back to plain C. See
[docs/P4.md](docs/P4.md). How the extra memory could be used, and options for models for atypical speech (stuttering, dysarthria):
[docs/P4.md](docs/P4.md#what-the-extra-memory-can-buy) and [docs/ATYPICAL_SPEECH.md](docs/ATYPICAL_SPEECH.md).

## Device API

```
→ {"cmd":"talk","on":true}            the virtual button (the physical one works too)
← {"type":"state","state":"listening","utt":1}
← {"type":"partial","utt":1,"text":"turn on the kitchen"}
→ {"cmd":"talk","on":false}
← {"type":"final","utt":1,"text":"turn on the kitchen lights","audio_s":2.4,"delay_ms":1180}
```

```
python esp32/tools/oido_client.py COM17 --hello      # watch the events
```

The full protocol is in [docs/API.md](docs/API.md). Bluetooth: the `oido_s3_ble` firmware advertises as `cosgc-oido` and
carries the same lines over a GATT service in the Nordic UART layout, so apps such as nRF Connect can talk to it.
Pairing asks for a 6-digit PIN shown on the screen. **Not yet tested with a phone:** the device advertises and the speed
and memory cost are measured, but nobody has connected to it and read a transcript over Bluetooth yet (the development PC
has no Bluetooth adapter). Step-by-step instructions for trying it with nRF Connect are in the API document.

## Status and limits

- **Working prototype**, not a product. Two-core mode (`oido_s3_2c`) has run through repeated benchmark runs and live use
  without a crash since the final fixes, but has not had a multi-hour soak test. The one-core firmware is the conservative choice.
- **Bluetooth costs speed and memory.** The radio takes about 90 KB of the chip's fast RAM, so some of the recognizer's
  working buffers move to slower memory: the final text comes about 0.5 s later after a 10-second hold (2.9 s instead of
  2.4 s), and the app needs a 1 MB flash partition. If you do not need Bluetooth, use `oido_s3_2c`.
- This is **not a medical device**. If transcripts will be used in care, the error rates must be validated on the people
  it is for.

## Repository

```
esp32/components/tinyasr/   the speech engine (C, ESP32-S3 vector instructions); from Oído, with our speed work
esp32/firmware/             the ESP-IDF app: microphone, button, display, LED, device API, Bluetooth
esp32/host/                 the engine built for a laptop (test and tools); esp32/tools/ flashing, benchmarks, API client
docs/API.md                 the device protocol          WIRING.md  pins and troubleshooting
models/                     the speech models (see LICENSING.md)
train/, eval/, results/     Oído's training and evaluation code and results
```

## Credits and license

The speech engine, the models and the training pipeline are the work of **Lokutor (Hashing Works, S.L.L.)** and
**NVIDIA** (the acoustic model). The code is under the **GNU GPL v3**, the models under CC-BY or CC-BY-SA, as
[LICENSING.md](LICENSING.md) explains in plain language (with a checklist for shipping a product). [NOTICE](NOTICE) lists
exactly what we changed. Lokutor also sells commercial licenses ([COMMERCIAL.md](COMMERCIAL.md)).
