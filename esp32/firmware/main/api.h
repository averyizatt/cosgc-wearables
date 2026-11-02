// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// The device's own API (docs/API.md): events out, commands in, one line of JSON per message, over every transport (the USB
// serial port today). Lines of the human-readable console log never start with '{', so a client keeps only the JSON lines.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#define API_VERSION 1

typedef struct {
    const char *mode;  // "streaming" or "utterance"
    bool lm;           // language model beam search on
    bool ptt;          // push-to-talk button
    bool ble;          // the Bluetooth service is built in
} api_info_t;

void api_start(const api_info_t *info);  // starts the command reader; announces itself with a "hello" event

// events (any task)
void api_state(const char *state);                   // "idle" | "listening" | "processing"; "listening" starts a new utterance
void api_partial(const char *text);                  // text so far of the current utterance (streaming models)
void api_final(const char *text, double audio_s, double delay_ms);  // final text; delay: from the button release (or the pause)
void api_button(bool down);                          // the physical button changed (real time, before the recognizer gets there)
void api_warn(const char *code, int value);          // e.g. "audio_dropped" (value: ms of audio lost)
void api_level(float dbfs);                          // microphone level of a 20 ms block (sent only when a client asked for it)
void api_human_partial(void);                        // the console log just printed an unfinished line: JSON starts a fresh one

// virtual button: a client can press and release it ("talk" command); true while it is held
bool api_talk_held(void);

// the microphone audio of each utterance, before gain control, for measuring (only after {"cmd":"set","dump":true})
bool api_dump_on(void);
void api_audio(const int16_t *pcm, int n);

// one line of text from a client (a transport calls this with each complete line)
void api_handle_line(const char *line);

void api_ble(const char *event);                      // "connected" | "secure" | "disconnected" (Bluetooth link changes)
void api_ack(const char *cmd, int id, bool ok, const char *err);  // for commands handled elsewhere; id < 0: none

// commands a transport adds (for example "ble_unpair"): return true if it handled cmd (and acked it)
typedef bool (*api_ext_t)(const char *cmd, int id);
void api_set_ext_handler(api_ext_t fn);

// transports: called with every outgoing line (without the newline); register one with api_add_sink
typedef void (*api_sink_t)(const char *line, int len);
void api_add_sink(api_sink_t sink);
