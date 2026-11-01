# PlatformIO post-script for the ESP32-P4: flash through the ROM loader. esptool's stub flasher starts on this chip but
# then fails with "Invalid head of packet (0x61)" or "The chip stopped responding" at any baud rate; the ROM loader works.
Import("env")  # noqa: F821 (SCons)

env.Prepend(UPLOADERFLAGS=["--no-stub"])
