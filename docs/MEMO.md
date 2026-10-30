MEMO: Offline Speech-to-Text Wearable (cosgc-wearables)

Subject: Testing, verification and tuning results
Branch: engine-speedups

1. SUMMARY

We took Oído (Lokutor's open speech recognizer) and built a push-to-talk wearable around an ESP32-S3 (16 MB flash, 8 MB PSRAM), an INMP441 microphone, a 128x64 OLED, a button and an LED. All recognition runs on the chip. Nothing is sent to the cloud. We tuned the engine for speed, added a screen, LED and microphone level bar, a JSON device API over USB, and a Bluetooth LE transport. The result is a working prototype, not a product.

2. WHAT WE TESTED AND FOUND

Speed (same four test clips, 24 s of speech, streaming model with language model)
- Oído as released: 1.93 seconds of compute per second of speech. Final text came 7.2 s after a 7 s sentence ended.
- Ours, one core: 1.54. Final text after 4.5 s.
- Ours, two cores: 1.19. Final text after about 2.2 s.
- Transcripts were identical in all three, so the speedup cost no accuracy.
- Where the gain came from: fused output stages and a faster int8 kernel for short rows, attention split across both cores, data-cache prefetch of the weights, an int4 language model, and -O3.
- The language model turned out to be limited by memory bandwidth (PSRAM), not by arithmetic, so halving its weights to int4 helped more than kernel tweaks.

Accuracy (LibriSpeech, laptop run of the same engine, 500 utterances per set)
- Streaming mode: 4.9% word error rate (test-clean), 10.7% (test-other).
- Slower utterance mode: 3.2% and 7.0%.
- The int4 language model changed error rates by almost nothing (4.80 vs 4.87).
- Quiet speech, simulated: no loss down to 40 dB below normal speech, 8.5% errors at 50 dB below, unusable at 60 dB. This drove the level bar's tick mark.

Live use on the board
- Ten push-to-talk utterances on two cores with no crash. Short phrases finalized 0.7 to 1.6 s after button release.
- A 29-second sentence was transcribed correctly after we removed a 20 s cutoff and enlarged the audio buffer to 32 s.
- Repeated benchmark runs on two cores were clean after the final fixes.

Device API and Bluetooth
- USB: hello, status, ping, level and a virtual talk button all worked and returned events in the correct order. JSON escaping (quotes, accents, control characters) passed host tests.
- Bluetooth: the device advertises, and the build, memory and speed were measured. Bluetooth costs about 0.5 s of extra delay (2.42 s to 2.92 s on a 10 s hold) and about 90 KB of fast RAM. It also needs a larger flash partition.

3. DOWNFALLS AND LIMITS

- The chip is still slower than real time (1.19x). Long non-stop speech builds a delay. After several minutes on two cores (about a minute on one) the buffer overflows and audio is dropped.
- Two-core mode is the fastest but is experimental. Early versions crashed (cache errors); the current spin-wait version has not had a multi-hour soak test. One core is the safe choice.
- Not verified: connecting to Bluetooth from a phone (the development PC has no adapter). A transcript with real text has not yet been seen through the API, because the laptop speaker was too quiet for the microphone.
- Not measured: stuttering or dysarthric speech, accented speech, noisy rooms, Spanish on our firmware. The model was trained on typical read speech, and the accuracy numbers come from LibriSpeech and simulation, not from real users.
- Quiet-speech results are simulated (turned-down audio), and real quiet voices will differ.
- Bluetooth makes the device slower and uses memory. Two-core mode also keeps the second core busy, which uses more power, a concern for a battery wearable.
- English only in practice. Needs the N16R8 board; other variants lack memory.
- Licensing: GPL v3 code and CC-BY / CC-BY-SA models. Shipping a product needs the checklist in LICENSING.md, or a commercial license from Lokutor. The copyright holder named in our file headers is still a placeholder.
- Not a medical device. It must not be used in care without validation on the people it is for.

4. OPPORTUNITIES

- Fine-tune the model on recordings of the target users (for example people who stutter or have dysarthria). This is the biggest likely accuracy gain and the main reason to do this project.
- Real-user testing: accents, noisy rooms, quiet voices. This would replace the simulation with real numbers.
- Test the phone connection over Bluetooth, then build a small companion app (captions, commands, logging).
- Soak-test two-core mode for several hours, then make it the default.
- Power: measure battery draw, and let the second core sleep when idle.
- Use the same API for voice commands, such as the "turn on the kitchen lights" example.
- Hardware: a better microphone or placement. The level bar showed mic distance matters more than any software change.
- Spanish support exists in the engine and could be tried.
- A fully offline, private design is a strong selling point for accessibility and health use.

5. RECOMMENDED NEXT STEPS

1. Verify a text transcript end to end over USB with a real voice.
2. Test the Bluetooth connection with nRF Connect on a phone.
3. Run a multi-hour soak test of two-core mode.
4. Collect a small set of real recordings from intended users and measure error rates.
5. Decide on licensing (GPL compliance or commercial) before sharing the hardware.
