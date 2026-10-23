# Device API

The device speaks one protocol over every transport: **one JSON object per line**, in both directions. Today the
transport is the USB serial port; Bluetooth carries the same lines (see [Bluetooth](#bluetooth)). Version: `api` = 1.

The console log (human-readable text) shares the USB serial port, so a client ignores every line that does not start with
`{`. Text is UTF-8. Lines are at most about 6000 bytes.

Try it without writing code:

```
pip install pyserial
python esp32/tools/oido_client.py COM17 --hello        # print everything the device says
python esp32/tools/oido_client.py COM17 --talk 5       # hold the virtual button for 5 s, print the transcript
```

## Events (device → client)

| `type` | Fields | When |
|---|---|---|
| `hello` | `api`, `fw` (firmware version), `mode` (`streaming` or `utterance`), `lm` (language model on), `ptt` (button mode), `cores`, `uptime_s`, `psram_free`, `internal_free` | at boot, and on the `hello` command |
| `state` | `state` (`idle`, `listening`, `processing`), `utt` | the recognizer starts hearing an utterance (`utt` counts up), starts computing the final text, or is done |
| `partial` | `utt`, `text` | the text so far of the current utterance (streaming models), sent when it changes; never empty |
| `final` | `utt`, `text`, `audio_s`, `delay_ms` | the final text (empty if nothing was recognized). `delay_ms` is the time from the button release (or the end-of-speech pause) to this message |
| `button` | `down` | the push-to-talk button changed (the physical one or the virtual one), in real time |
| `level` | `db` | microphone level in dBFS, at most 5 a second, only after `{"cmd":"set","level":true}` |
| `warn` | `code`, `value` | `audio_dropped`: the recognizer fell behind and `value` ms of audio were lost |
| `status` | `state`, `utt`, `last_delay_ms`, `uptime_s`, `psram_free`, `internal_free` | answer to `status` |
| `ack` | `cmd`, `ok`, optional `id`, optional `error` | the result of every command |

`state` and `final` come from the recognizer, so they can lag the real world by seconds when the chip is behind (long
utterances). `button` is real time. For a long sentence a phone sees `button down=false` first and the `final` several
seconds later.

## Commands (client → device)

Every command may carry a number `"id"`, which comes back in its `ack`.

| Command | Meaning |
|---|---|
| `{"cmd":"hello"}` | send the `hello` event |
| `{"cmd":"ping"}` | `ack` only (is the device alive) |
| `{"cmd":"status"}` | send a `status` event |
| `{"cmd":"talk","on":true}` / `{"cmd":"talk","on":false}` | the virtual push-to-talk button, OR-ed with the physical one. Press, wait, release: the device transcribes what it heard in between |
| `{"cmd":"set","level":true}` | start (or stop, with `false`) the `level` events |
| `{"cmd":"reboot"}` | restart the firmware |

Unknown commands and bad JSON get an `ack` with `"ok":false` and an `error`.

## Example session

```
→ {"cmd":"talk","on":true,"id":1}
← {"type":"ack","cmd":"talk","ok":true,"id":1}
← {"type":"button","down":true}
← {"type":"state","state":"listening","utt":1}
← {"type":"partial","utt":1,"text":"turn on the"}
← {"type":"partial","utt":1,"text":"turn on the kitchen lights"}
→ {"cmd":"talk","on":false,"id":2}
← {"type":"button","down":false}
← {"type":"state","state":"processing","utt":1}
← {"type":"final","utt":1,"text":"turn on the kitchen lights","audio_s":2.4,"delay_ms":1180}
← {"type":"state","state":"idle","utt":1}
```

## Bluetooth

Not yet. See the project README for the current status; the lines will be the same as above, carried over a GATT service.

## Compatibility

`api` is bumped when a field changes meaning or disappears. New events, new fields and new commands may appear in the same
version, so a client should ignore what it does not know. Planned: per-word confidence in `final`, settings for the
language model weight, and a device information request.

Privacy: transcripts are the user's speech. Nothing leaves the device except what a connected client reads from these
lines, and nothing is stored on it.
