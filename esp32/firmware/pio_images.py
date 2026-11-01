# PlatformIO pre-script: writes the model and LM partition images ([u32 size][12 zero bytes][blob], as tools/mkimages.py
# does) and adds them to the upload, so `pio run -t upload` flashes app, model and LM in one go.
import os
import struct

Import("env")  # noqa: F821 (SCons)

proj = env.subst("$PROJECT_DIR")
board = env.BoardConfig()

# the sdkconfig.* defaults are only read when sdkconfig.<env> is created: start it over when one of them changed since
sdk = os.path.join(proj, "sdkconfig." + env.subst("$PIOENV"))
defaults = []
for arg in board.get("build.cmake_extra_args", "").split():
    if arg.startswith("-DSDKCONFIG_DEFAULTS="):
        defaults = [os.path.join(proj, f) for f in arg.split("=", 1)[1].split(";")]
if os.path.isfile(sdk) and any(os.path.getmtime(f) > os.path.getmtime(sdk) for f in defaults):
    print("sdkconfig defaults changed: regenerating", os.path.basename(sdk))
    os.remove(sdk)

parts = {}
for line in open(os.path.join(proj, board.get("build.partitions"))):
    f = [x.strip() for x in line.split("#")[0].split(",")]
    if len(f) >= 5:
        parts[f[0]] = (int(f[3], 0), int(f[4], 0))


def image(src, part):
    off, size = parts[part]
    blob = open(os.path.join(proj, src), "rb").read() if src else b""  # an empty LM image means greedy decoding
    img = struct.pack("<I", len(blob)) + b"\0" * 12 + blob
    if len(img) > size:
        raise SystemExit("%s (%d bytes) does not fit partition '%s' (%d)" % (src, len(blob), part, size))
    out = os.path.join(env.subst("$BUILD_DIR"), part + ".bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(img)
    if off + len(img) > 1 << 24:  # esptool's ROM loader would wrap around onto what is below 16 MB (docs/P4.md)
        print("%s crosses 16 MB: pio upload skips it, flash it with esp32/tools/flash_p4.py" % part)
        return
    env.Append(FLASH_EXTRA_IMAGES=[(hex(off), out)])


def audio(clips_json, part="audio"):
    # benchmark clips for TASR_MODE_FILE builds, the layout tools/mkimages.py writes: "AUD0", n, then per clip
    # [u32 samples][u32 ref bytes][ref, padded to 4][PCM16, padded to an even sample count]
    import json
    import wave
    off, size = parts[part]
    path = os.path.join(proj, clips_json)
    if not os.path.isfile(path):
        raise SystemExit("%s missing: run  python esp32/tools/bench_clips.py  first" % clips_json)
    clips = json.load(open(path))
    img = bytearray(b"AUD0" + struct.pack("<I", len(clips)))
    for wav, ref in clips:
        with wave.open(os.path.join(proj, wav), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
                raise SystemExit("%s: need 16 kHz mono PCM16" % wav)
            pcm = w.readframes(w.getnframes())
        rb = ref.encode()
        ns = len(pcm) // 2
        img += struct.pack("<II", ns, len(rb)) + rb + b"\0" * ((-len(rb)) % 4) + pcm + (b"\0\0" if ns % 2 else b"")
    if len(img) > size:
        raise SystemExit("benchmark clips (%d bytes) do not fit partition '%s' (%d)" % (len(img), part, size))
    out = os.path.join(env.subst("$BUILD_DIR"), part + ".bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(bytes(img))
    env.Append(FLASH_EXTRA_IMAGES=[(hex(off), out)])


image(env.GetProjectOption("custom_model"), "model")
image(env.GetProjectOption("custom_lm", ""), "lm")
if env.GetProjectOption("custom_clips", ""):
    audio(env.GetProjectOption("custom_clips"))
