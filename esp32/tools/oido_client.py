#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Talk to the device over its USB serial port with the JSON API (docs/API.md).

  python esp32/tools/oido_client.py COM17                     print every event (transcripts, state, button)
  python esp32/tools/oido_client.py COM17 --hello             also ask for the hello / status messages
  python esp32/tools/oido_client.py COM17 --talk 5            hold the virtual button for 5 s (the device listens)
  python esp32/tools/oido_client.py COM17 --level             stream microphone levels
  python esp32/tools/oido_client.py COM17 --cmd '{"cmd":"ping","id":1}'

As a library:

  from oido_client import Oido
  with Oido("COM17") as dev:
      print(dev.request({"cmd": "hello"}))
      for ev in dev.events():            # JSON events as dicts, until you stop iterating
          if ev["type"] == "final":
              print(ev["text"])

The console log shares the port; lines that do not start with '{' are skipped. The port is opened with DTR and RTS low so
the native USB port does not reset the chip into download mode.
"""
import argparse
import json
import queue
import threading
import time

import serial


class Oido:
    def __init__(self, port, baud=115200):
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.2
        self.ser.dtr = self.ser.rts = False
        self.ser.open()
        self.q = queue.Queue()
        self._stop = False
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self._stop = True
        self._t.join(timeout=1)
        self.ser.close()

    def _read(self):
        buf = b""
        while not self._stop:
            try:
                buf += self.ser.read(4096)
            except serial.SerialException:
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                i = line.find(b"{")  # a console log fragment can precede the JSON on the same line
                if i < 0:
                    continue
                try:
                    self.q.put(json.loads(line[i:].decode("utf-8", "replace")))
                except ValueError:
                    pass

    def send(self, msg):
        self.ser.write((json.dumps(msg) + "\n").encode())

    def events(self, timeout=None):
        """Yield events as they arrive; stops after `timeout` seconds without one (None: wait forever)."""
        while True:
            try:
                yield self.q.get(timeout=timeout)
            except queue.Empty:
                return

    def request(self, msg, timeout=3.0):
        """Send a command with an id and return (the ack, the other messages that came before it)."""
        msg = dict(msg, id=int(time.time() * 1000) % 1000000)
        self.send(msg)
        seen = []
        end = time.time() + timeout
        while time.time() < end:
            try:
                ev = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if ev.get("type") == "ack" and ev.get("id") == msg["id"]:
                return ev, seen
            seen.append(ev)
        raise TimeoutError("no answer to %s" % msg["cmd"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("--hello", action="store_true", help="ask for hello and status first")
    ap.add_argument("--talk", type=float, metavar="SECONDS", help="hold the virtual button for this long, then print until the final text")
    ap.add_argument("--level", action="store_true", help="ask for microphone level events")
    ap.add_argument("--cmd", help="send this JSON line")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (default: run until Ctrl-C, or the final text with --talk)")
    a = ap.parse_args()
    with Oido(a.port) as dev:
        if a.hello:
            dev.send({"cmd": "hello"})
            dev.send({"cmd": "status"})
        if a.level:
            dev.send({"cmd": "set", "level": True})
        if a.cmd:
            dev.send(json.loads(a.cmd))
        t0 = time.time()
        if a.talk:
            dev.send({"cmd": "talk", "on": True})
        released = False
        try:
            while True:
                try:
                    ev = dev.q.get(timeout=0.2)
                except queue.Empty:
                    ev = None
                now = time.time()
                if a.talk and not released and now - t0 >= a.talk:
                    dev.send({"cmd": "talk", "on": False})
                    released = True
                if ev is not None:
                    print(json.dumps(ev, ensure_ascii=False), flush=True)
                    if a.talk and released and ev.get("type") == "final":
                        break
                if a.seconds and now - t0 > a.seconds:
                    break
                if a.talk and released and now - t0 > a.talk + 60:  # the final text did not come
                    break
        except KeyboardInterrupt:
            pass
        if a.talk and not released:
            dev.send({"cmd": "talk", "on": False})


if __name__ == "__main__":
    main()
