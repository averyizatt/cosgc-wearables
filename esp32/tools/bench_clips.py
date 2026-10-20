"""Make the benchmark clips for the PlatformIO `bench` environments: decodes audio samples from docs/samples (16 kHz
mono PCM16 WAV) into esp32/firmware/bench_clips/, as many as fit the 768 KB 'audio' partition, and writes clips.json
([[wav, reference], ...], paths relative to the repo root) for esp32/firmware/pio_images.py.

usage: python esp32/tools/bench_clips.py        (needs numpy and soundfile >= 0.12, which reads MP3)
"""
import json
import os

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "esp32", "firmware", "bench_clips")
AUDIO_SIZE = 0xC0000  # 'audio' partition in partitions_nemo16lm.csv
# a quiet room, a car at 5 dB SNR, real meetings: different conditions, mostly 4-8 s
ORDER = ["rooms_00", "rooms_01", "real_20", "real_22", "real_24", "real_26", "real_21", "real_23", "real_25"]


def main():
    meta = {os.path.splitext(os.path.basename(r["mp3"]))[0]: r for r in
            json.load(open(os.path.join(ROOT, "docs", "samples", "results.json")))}
    os.makedirs(OUT, exist_ok=True)
    used, clips = 8, []
    for name in ORDER:
        r = meta[name]
        x, sr = sf.read(os.path.join(ROOT, "docs", r["mp3"]), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            x = np.interp(np.arange(len(x) * 16000 // sr) * (sr / 16000), np.arange(len(x)), x)
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        ref = r["ref"].encode()
        need = 8 + len(ref) + (-len(ref)) % 4 + 2 * (len(pcm) + len(pcm) % 2)
        if used + need > AUDIO_SIZE:
            continue
        used += need
        path = os.path.join(OUT, name + ".wav")
        sf.write(path, pcm, 16000, subtype="PCM_16")
        clips.append([os.path.relpath(path, ROOT).replace(os.sep, "/"), r["ref"]])
        print(f"{name}: {len(pcm) / 16000:.1f} s  {r['label']}")
    json.dump(clips, open(os.path.join(OUT, "clips.json"), "w"), indent=1)
    print(f"{len(clips)} clips, {used} of {AUDIO_SIZE} bytes -> {os.path.relpath(OUT, ROOT)}/clips.json")


if __name__ == "__main__":
    main()
