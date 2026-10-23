// Optional SSD1306 128x64 I2C display showing the live transcript (menuconfig -> tinyasr -> OLED display).
// All functions are no-ops when CONFIG_TASR_OLED is off, so callers need no #ifdefs.
#pragma once
#include <stdint.h>

#define OLED_W 128
#define OLED_H 64

// core for the mic capture and display tasks: the otherwise idle second core in single-core builds (recognition runs
// on core 0), core 0 when core 1 runs the engine's worker
#ifdef TASR_SINGLE_CORE
#define TASR_AUX_CORE 1
#else
#define TASR_AUX_CORE 0
#endif

void oled_start(void);                  // I2C + panel init and the render task (once)
void oled_status(const char *s);        // top line, e.g. "listening" / "transcribing..."
void oled_push(const char *text);       // append a finished utterance to the history
void oled_partial(const char *text);    // replace the in-progress utterance (streaming model)
void oled_level(float dbfs);            // microphone level (rms of a 20 ms block) for the bar in the status line

// level bar in the status line: LEVEL_DB_MIN..LEVEL_DB_MAX dBFS over LEVEL_PX pixels; the marker is the quietest
// level at which recognition still worked well in simulation (speech 40 dB below the test set's, about 5 % errors)
#define LEVEL_DB_MIN (-80.f)
#define LEVEL_DB_MAX (-20.f)
#define LEVEL_PX 40
#define LEVEL_MARK_DB (-55.f)

// portable: draw status + '\n'-separated history into a 1024-byte SSD1306 page buffer; level_px < 0: no bar
void oled_render(uint8_t *fb, const char *status, const char *hist, int level_px);
