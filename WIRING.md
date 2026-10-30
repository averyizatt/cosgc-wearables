# My build: ESP32-S3 + INMP441 + 0.96" OLED

Live speech-to-text on the chip. Each sentence prints over serial and shows on the OLED.

## Board

This only runs on an **ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (N16R8)**, such as the ESP32-S3-DevKitC-1 N16R8.
A classic ESP32 (ESP32-WROOM-32 / "ESP32 DevKit V1") does not work: it lacks the S3's vector instructions, and it does not have enough flash or PSRAM for the 14 MB model.

## Wiring

| Module | Module pin | ESP32-S3 pin |
|---|---|---|
| INMP441 mic | VDD | 3V3 |
| | GND | GND |
| | L/R | GND (selects the left channel, which the firmware reads) |
| | SCK | GPIO4 |
| | WS | GPIO5 |
| | SD | GPIO6 |
| SSD1306 OLED (I2C, 128x64) | VCC | 3V3 |
| | GND | GND |
| | SDA | GPIO8 |
| | SCL | GPIO9 |
| Push-to-talk button | one side | GPIO7 |
| | other side | GND (internal pull-up; hold while speaking) |
| Status LED | on board (WS2812) | GPIO48 (DevKitC-1 v1.0 and most clones; GPIO38 on v1.1) |

- Power both modules from **3V3**, not 5V.
- Do not leave L/R floating. If it is floating, the mic may output on the right channel, and you get silence.
- Pins to avoid on an N16R8 board: GPIO26–32 (flash), 33–37 (octal PSRAM), 0/3/45/46 (strapping), 19/20 (USB),
  43/44 (serial console).
- With the button, an utterance is what you say while holding it (plus 0.3 s from before the press); the final text
  comes as soon as you let go, with no end-of-speech pause, and room noise is never transcribed. Set
  `CONFIG_TASR_PTT_GPIO=-1` to go back to voice-activity detection.
- LED: dim blue loading, dim green ready, blue hearing you, amber computing, green flash text ready, red flash nothing
  recognized, purple audio dropped (recognition is behind).
- To change pins or the OLED address (0x3C or 0x3D), edit [`esp32/firmware/sdkconfig.myboard`](esp32/firmware/sdkconfig.myboard).

## Build and flash (PlatformIO)

1. In VS Code with the PlatformIO extension, open **`oido.code-workspace`** (File > Open Workspace from File...), or open the
   folder **`esp32/firmware`** directly. PlatformIO only finds `platformio.ini` at the root of a workspace folder, so opening
   the repo root alone shows no PlatformIO tasks.
2. Plug the board into the USB port labeled **UART** (or COM). On boards with two ports, the other one is native USB.
3. Click **Upload** in the PlatformIO toolbar, or run `pio run -t upload`. This flashes three things: the app, the 14 MB model and the
   language model. Writing the model takes about 2 minutes. The first build also downloads ESP-IDF, which takes a while.
4. Open **Serial Monitor** (`pio device monitor`, 115200 baud) and talk.

Serial output looks like this:

```
ready (streaming): speak, partial text follows you, the final line prints after each pause
... turn on the
turn on the kitchen lights   [2.1 s audio, final text 1180 ms after the pause was detected]
```

The OLED shows a status line at the top (`* listening`, `* hearing you...`) and the latest sentences below it.

## Environments

| Env | Model | Behavior |
|---|---|---|
| `oido_s3` (default) | `oido_stream.tnm` + `nemo_lm.tlm` | Words appear while you speak. The final line comes about 4 s after you stop. |
| `oido_s3_accurate` | `nemo8.tnm` + `nemo_lm.tlm` | The most accurate model. Text appears about 7 s after a short sentence ends. |
| `oido_s3_bench` | `oido_stream.tnm` + `nemo_lm.tlm` | No mic. Transcribes 4 sample clips from flash and prints the real-time factor (`UTT`, `STREAM`, `PACED`), cycles per stage (`PROF`) and a kernel memory benchmark (`KBENCH`). Run `python esp32/tools/bench_clips.py` once first (needs `soundfile`). |

The chip runs at about 2x real time on one core, so longer sentences take longer to finish. See the main [README](README.md).

## Troubleshooting

- **The OLED stays blank, and serial shows `no SSD1306 at 0x3c`.** Check the SDA and SCL wires. If they are right, set `CONFIG_TASR_OLED_ADDR=0x3D`.
- **`ready` prints, but no text appears when you talk.** Check that L/R is tied to GND and that SD is on GPIO6.
- **Serial shows `audio buffer full: recognition is behind`.** The chip runs at about 1.8x real time, so during long stretches of
  speech (or loud noise the VAD takes for speech) it falls behind; after about 16 s of backlog it drops audio, which costs words.
  The next line says how much was lost. Pause between sentences, or keep the mic away from constant noise.
- **Nothing appears on serial.** Check that you are on the UART port. On the native USB port, follow the notes in the README.

## ESP32-P4 (Waveshare ESP32-P4-Pico)

Not tested on hardware. No wiring is needed for the microphone: it is on the board (ES8311 codec, I2C SDA 7 / SCL 8, I2S MCLK 13 /
SCLK 12 / LRCK 10 / data 11). Build with `pio run -e oido_p4`. A push-to-talk button (to GND) and an I2C OLED (SDA 7, SCL 8, address
0x3C, shared with the codec) can be added on the 40-pin header; set `CONFIG_TASR_PTT_GPIO` and `CONFIG_TASR_OLED*` in
`esp32/firmware/sdkconfig.p4`. Full notes, the bring-up checklist and what is unverified: [docs/P4.md](docs/P4.md).
