// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// Live microphone buffers (PSRAM), shared by mic.c (allocates them) and main.c (keeps that much PSRAM free).
#pragma once
#include "sdkconfig.h"

// Audio ring between the capture task and the recognizer: the backlog the recognizer can fall behind by (it runs
// slower than real time) before audio is dropped. With the push-to-talk button and a streaming model there is no
// recording buffer, so the ring gets that memory: 32 s, enough for ~50 s of continuous speech at 1.6x real time.
#if CONFIG_TASR_PTT_GPIO >= 0
#define MIC_RING_SAMPLES (16000 * 32)
#else
#define MIC_RING_SAMPLES (16000 * 16)
#endif
#define MIC_UTT_SAMPLES (16000 * 20)  // recording buffer: the VAD segmenter, or push-to-talk with an utterance model
// PSRAM for the ring and, when used, the recording buffer
#define MIC_PSRAM(streaming)                                                                                             \
    ((size_t)MIC_RING_SAMPLES * 2 + ((CONFIG_TASR_PTT_GPIO >= 0 && (streaming)) ? 0 : (size_t)MIC_UTT_SAMPLES * 2))
