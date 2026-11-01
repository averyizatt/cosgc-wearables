#!/usr/bin/env python3
"""Flash an ESP32-P4 board, including above 16 MB, through the chip's ROM loader.

esptool's stub flasher does not run on our chip, and the ROM loader sends 3-byte addresses, so a plain write above 16 MB wraps
around and lands on top of whatever is at the same address below 16 MB, with a hash check that still passes. This tool uses the
flash chip's Extended Address Register (commands C8h / C5h) to pick the 16 MB bank, splits images at the boundary and, once
everything is written, verifies every image again, which is what catches an overlap. See docs/P4.md.

  python esp32/tools/flash_p4.py COM3 --env oido_p4_bench                 everything from that PlatformIO build
  python esp32/tools/flash_p4.py COM3 --env oido_p4 --only app            just bootloader, partition table and app
  python esp32/tools/flash_p4.py COM3 --env oido_p4 --only model --model big2_int8.tnm      just another model
"""
import argparse
import csv
import hashlib
import os
import struct
import sys
import time

sys.path.insert(0, os.path.expanduser("~/.platformio/tools/tool-esptoolpy"))
from esptool.cmds import attach_flash, detect_chip  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANK = 1 << 24
FLASH_SIZE = 32 << 20


def partitions():
    parts = {}
    with open(os.path.join(REPO, "esp32", "firmware", "partitions_p4_32.csv")) as f:
        for row in csv.reader(line for line in f if not line.startswith("#")):
            if len(row) >= 5:
                parts[row[0].strip()] = (int(row[3], 0), int(row[4], 0))
    return parts


def model_image(path):
    # the layout pio_images.py and mkimages.py write: [u32 size][12 zero bytes][blob]
    blob = open(path, "rb").read()
    return struct.pack("<I", len(blob)) + b"\0" * 12 + blob


class Flash:
    def __init__(self, port, baud):
        self.esp = detect_chip(port, 115200, "default-reset", False, 3)
        self.esp.change_baud(baud)
        attach_flash(self.esp)
        self.esp.flash_set_parameters(FLASH_SIZE)
        self.bank = None
        self.set_bank(0)

    def set_bank(self, bank):
        if bank == self.bank:
            return
        self.esp.run_spiflash_command(0x06)
        self.esp.run_spiflash_command(0xC5, bytes([bank]))
        got = self.esp.run_spiflash_command(0xC8, b"", 8) & 0xFF
        if got != bank:
            raise SystemExit("flash did not accept bank %d (reads %d): this chip may have no extended address register" % (bank, got))
        self.bank = bank

    def pieces(self, addr, data):
        # (bank, address inside the bank, bytes) for each side of a 16 MB boundary
        o = 0
        while o < len(data):
            a = addr + o
            n = min(len(data) - o, BANK - (a % BANK))
            yield a // BANK, a % BANK, data[o:o + n]
            o += n

    def write(self, addr, data, name):
        esp, step, t0, done = self.esp, self.esp.FLASH_WRITE_SIZE, time.time(), 0
        for bank, a, piece in self.pieces(addr, data):
            self.set_bank(bank)
            esp.flash_begin(len(piece), a)
            for seq, o in enumerate(range(0, len(piece), step)):
                esp.flash_block(piece[o:o + step].ljust(step, b"\xff"), seq)
                done += step
                if seq % 1024 == 1023:
                    print("  %s %.1f of %.1f MB, %.0f s" % (name, done / 1e6, len(data) / 1e6, time.time() - t0), flush=True)
        print("  %s written at 0x%x, %d bytes, %.0f s" % (name, addr, len(data), time.time() - t0), flush=True)

    def verify(self, addr, data):
        for bank, a, piece in self.pieces(addr, data):
            self.set_bank(bank)
            m = self.esp.flash_md5sum(a, len(piece))
            if (m.hex() if isinstance(m, bytes) else m) != hashlib.md5(piece).hexdigest():
                return False
        return True

    def close(self):
        # a board reset does not reset the flash chip: leave bank 0 selected or the chip will not boot
        self.set_bank(0)
        self.esp.hard_reset()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("--env", default="oido_p4_bench", help="PlatformIO environment whose build to flash")
    ap.add_argument("--only", choices=["app", "model"], help="write only the app (with bootloader and table), or only the model")
    ap.add_argument("--model", help="a .tnm to flash instead of the build's model.bin")
    ap.add_argument("--baud", type=int, default=921600)
    a = ap.parse_args()

    build, parts = os.path.join(REPO, ".pio", "build", a.env), partitions()
    read = lambda name: open(os.path.join(build, name), "rb").read()
    images = []
    if a.only != "model":
        images += [("bootloader", 0x2000, read("bootloader.bin")), ("partitions", 0x8000, read("partitions.bin")),
                   ("app", parts["factory"][0], read("firmware.bin"))]
    if a.only is None:
        images.append(("lm", parts["lm"][0], read("lm.bin")))
        if os.path.exists(os.path.join(build, "audio.bin")):
            images.append(("audio", parts["audio"][0], read("audio.bin")))
    if a.only != "app":
        images.append(("model", parts["model"][0], model_image(a.model) if a.model else read("model.bin")))

    for name, addr, data in images:
        part = parts.get("factory" if name == "app" else name)
        if part and len(data) > part[1]:
            raise SystemExit("%s (%d bytes) does not fit its partition (%d)" % (name, len(data), part[1]))
        if name not in ("model",) and addr + len(data) > BANK:
            raise SystemExit("%s crosses 16 MB, where the cache cannot read it" % name)

    f = Flash(a.port, a.baud)
    try:
        for name, addr, data in images:
            f.write(addr, data, name)
        bad = [name for name, addr, data in images if not f.verify(addr, data)]
    finally:
        f.close()
    if bad:
        raise SystemExit("VERIFY FAILED after writing everything: %s" % ", ".join(bad))
    print("all images verified after the last write:", ", ".join(n for n, _, _ in images))


if __name__ == "__main__":
    main()
