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
    env.Append(FLASH_EXTRA_IMAGES=[(hex(off), out)])


image(env.GetProjectOption("custom_model"), "model")
image(env.GetProjectOption("custom_lm", ""), "lm")
