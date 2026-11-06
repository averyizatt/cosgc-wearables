// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// Session log on the microSD card (CONFIG_TASR_SDLOG): every push-to-talk utterance's audio, text and scores, and prompted
// recording from a prompts.txt on the card. All functions are no-ops when the option is off or no card was found.
#pragma once
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const int16_t *audio;   // what the model heard (after gain control), n_audio samples at 16 kHz
    int n_audio;
    const int16_t *raw;     // the microphone before gain control (NULL: not kept), n_raw samples
    int n_raw;
    const int16_t *raw2;    // the other microphone before gain control (CONFIG_TASR_MIC2; NULL: none), n_raw2 samples, same timing
    int n_raw2;
    int mic;                // the microphone the recognizer heard (1 built in, 2 the second)
    int pre;                // samples of pre-roll (from before the button went down) at the start of each
    const char *text;       // what the screen showed
    const char *free_text;  // the free decoding's text (differs from text when a list phrase replaced it)
    const char *stream_text;  // the live (streaming) text at the end of the utterance; NULL if none
    const char *phrase;     // the closest list phrase, shown or not; NULL without a list
    float phrase_gap;       // its score per frame minus free decoding's (shown when above phrase_thr)
    float phrase_thr;
    int phrase_n;           // phrases on the list
    int conf_level;         // as api_final: 3 phrase, 2 sure, 1 unsure, 0 probably wrong, -1 not measured
    float conf;
    bool full;              // a full-context pass made the final text
    double delay_ms;        // final text after the button was released
    double press_s;         // when the button went down, seconds since boot
} sdlog_utt_t;

void sdlog_start(const void *model, uint32_t model_bytes);  // mount the card, open a session folder, read prompts.txt
bool sdlog_on(void);
const char *sdlog_prompt(void);  // the words of the next prompt (scored against), or NULL (no prompts.txt, or all done)
const char *sdlog_cue(void);     // what the screen shows for it: a "text | cue" line's cue (e.g. a phonetic spelling), else the text
int sdlog_prompt_pos(int *count);  // 1-based position of the current prompt; *count: how many there are
void sdlog_prompt_back(void);    // a tap: the previous prompt again
bool sdlog_next_session(void);   // a tap after the last prompt: a new session folder with the next prompt file
void sdlog_utt(const sdlog_utt_t *u);  // write the utterance, then move to the next prompt
