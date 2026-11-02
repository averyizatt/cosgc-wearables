// Live microphone mode: I2S MEMS mic -> capture task -> PSRAM ring buffer -> AGC + energy VAD -> streaming ASR.
// The capture task runs at high priority so no audio is lost while the ASR task is busy with an encoder chunk.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "tinyasr.h"
#include "tasr_nemo.h"
#include "tasr_seg.h"
#include "api.h"
#include "codec.h"
#include "led.h"
#include "mic.h"
#include "oled.h"

#ifdef CONFIG_TASR_MODE_MIC
#define RING_SAMPLES MIC_RING_SAMPLES  // mic.h
#ifdef CONFIG_TASR_BLE
#define HAS_BLE true
#else
#define HAS_BLE false
#endif
static StreamBufferHandle_t g_ring;
static i2s_chan_handle_t g_rx;

#if CONFIG_TASR_PTT_GPIO >= 0
// Push-to-talk: the capture task (real time) debounces the button and queues each press / release with the index of
// the audio sample it happened at; the recognizer, which runs behind real time, acts on it when it reaches that sample
// in the audio, so releasing the button never cuts off the last words still waiting in the ring.
typedef struct {
    uint32_t at;   // sample index in the ring's stream (samples actually queued)
    uint8_t down;
    int64_t t_us;  // when the capture task saw it (esp_timer)
} ptt_ev_t;
static QueueHandle_t g_ptt;
#endif

static void capture_task(void *arg)
{
    (void)arg;
    static int32_t raw[320];
    static int16_t pcm[320];
    float dc = 0.f;
    int dropped = 0;  // samples lost in the current overflow episode
#if CONFIG_TASR_PTT_GPIO >= 0
    uint32_t queued = 0;           // samples queued so far
    int state = gpio_get_level(CONFIG_TASR_PTT_GPIO) == 0, seen = 0;  // debounced button state (a pin low at startup is not a press); blocks in a row that read the other state
    uint32_t seen_at = 0;
#endif
    for (;;) {
        size_t got = 0;
        i2s_channel_read(g_rx, raw, sizeof(raw), &got, portMAX_DELAY);
        int n = got / 4;
        float energy = 0.f;
        for (int i = 0; i < n; i++) {  // 24-bit left-justified -> 16-bit with DC removal
            float x = (float)(raw[i] >> 8) / 256.0f;
            dc += 0.001f * (x - dc);
            float y = x - dc;
            energy += y * y;
            pcm[i] = (int16_t)(y > 32767.f ? 32767.f : (y < -32768.f ? -32768.f : y));
        }
        if (n) {
            const float db = 10.f * log10f(energy / n / (32768.f * 32768.f) + 1e-12f);  // dBFS of this 20 ms block
            oled_level(db);
            api_level(db);
        }
#if CONFIG_TASR_PTT_GPIO >= 0
        const int down = gpio_get_level(CONFIG_TASR_PTT_GPIO) == 0 || api_talk_held();  // button to GND (pull-up), or a client's virtual button
        if (down != state) {
            if (!seen++) seen_at = queued;  // the first block that read the new state
            if (seen >= 2) {                // stable for 40 ms
                state = down;
                seen = 0;
                api_button(down);
                const ptt_ev_t ev = {seen_at, (uint8_t)down, esp_timer_get_time()};
                xQueueSend(g_ptt, &ev, 0);
                if (!down) {  // released: say so now; the recognizer may still be seconds behind (long utterances)
                    led_set(LED_BUSY);
                    oled_status("finishing...");
                }
            }
        } else {
            seen = 0;
        }
#endif
        const int sent = (int)(xStreamBufferSend(g_ring, pcm, n * 2, 0) / 2);
#if CONFIG_TASR_PTT_GPIO >= 0
        queued += sent;
#endif
        if (sent < n) {
            if (!dropped) {
                ESP_LOGW("mic", "audio buffer full: recognition is behind, dropping audio");
                led_set(LED_BEHIND);
            }
            dropped += n - sent;
        } else if (dropped) {
            ESP_LOGW("mic", "audio buffer has room again: %d ms of audio were lost", dropped / 16);
            api_warn("audio_dropped", dropped / 16);
            led_set(LED_BUSY);
            dropped = 0;
        }
    }
}
#endif

#ifdef CONFIG_TASR_MODE_MIC
static void mic_start(void)
{
    static int started;
    if (started) return;
    started = 1;
    static const char *TAG = "mic";
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = CONFIG_TASR_I2S_DMA_DESC;
    cc.dma_frame_num = 320;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &g_rx));
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
#ifdef CONFIG_TASR_MIC_CODEC
                     .mclk = CONFIG_TASR_I2S_MCLK,
#else
                     .mclk = I2S_GPIO_UNUSED,
#endif
                     .bclk = CONFIG_TASR_I2S_BCLK, .ws = CONFIG_TASR_I2S_WS,
                     .dout = I2S_GPIO_UNUSED, .din = CONFIG_TASR_I2S_DIN},
    };
    sc.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  // INMP441 with L/R tied to GND; the ES8311 puts the microphone on the left too
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_rx, &sc));
    ESP_ERROR_CHECK(i2s_channel_enable(g_rx));
#ifdef CONFIG_TASR_MIC_CODEC
    codec_start();  // needs MCLK running, so after the I2S clocks start; an error is logged and the level bar shows silence
#endif
    uint8_t *storage = heap_caps_malloc(RING_SAMPLES * 2 + 1, MALLOC_CAP_SPIRAM);
    static StaticStreamBuffer_t sb;
    g_ring = xStreamBufferCreateStatic(RING_SAMPLES * 2, 2, storage, &sb);
#if CONFIG_TASR_PTT_GPIO >= 0
    const gpio_config_t io = {.pin_bit_mask = 1ULL << CONFIG_TASR_PTT_GPIO, .mode = GPIO_MODE_INPUT,
                              .pull_up_en = GPIO_PULLUP_ENABLE, .pull_down_en = GPIO_PULLDOWN_DISABLE,
                              .intr_type = GPIO_INTR_DISABLE};
    ESP_ERROR_CHECK(gpio_config(&io));
    g_ptt = xQueueCreate(16, sizeof(ptt_ev_t));
    ESP_LOGI(TAG, "push-to-talk button on GPIO%d (to GND): hold it while you speak", CONFIG_TASR_PTT_GPIO);
    vTaskDelay(pdMS_TO_TICKS(10));  // let the pull-up settle
    if (gpio_get_level(CONFIG_TASR_PTT_GPIO) == 0)
        ESP_LOGW(TAG, "GPIO%d reads low with nothing pressed: check the button's wiring (ignored until it reads high)", CONFIG_TASR_PTT_GPIO);
#endif
    xTaskCreatePinnedToCore(capture_task, "mic_cap", 4096, NULL, configMAX_PRIORITIES - 1, NULL, TASR_AUX_CORE);
    ESP_LOGI(TAG, "listening (BCLK %d WS %d DIN %d)", CONFIG_TASR_I2S_BCLK, CONFIG_TASR_I2S_WS, CONFIG_TASR_I2S_DIN);
}

#endif

#if defined(CONFIG_TASR_MODE_MIC) && CONFIG_TASR_PTT_GPIO >= 0
#define PTT_PRE 4800  // 0.3 s of audio from before the press (a word started as the button goes down)
#define PTT_MIN 2400  // presses shorter than 0.15 s are ignored
#define PTT_CAP MIC_UTT_SAMPLES  // utterance mode: the recording buffer (streaming models have no length limit)
#define PTT_PAD 8000  // 0.5 s of silence added before the final full-context pass
typedef struct {
    const tasr_nemo_t *m;
    struct tasr_decoder *dec;
    tasr_nemo_stream_t *st;
    int16_t *utt;
    int n_utt, pre_used;
    int16_t *raw;  // the same audio before gain control, then its pre-roll; only after a client asks (api_dump_on)
    int n_raw;
    char *text;
    int text_len;
    char idle[32];
} ptt_t;
// the utterance ends here: final text (released_us: when the button went up, or 0 at the utterance-mode limit)
static void ptt_end(ptt_t *p, int64_t released_us)
{
    if (p->n_utt - p->pre_used < PTT_MIN) {  // a tap
        oled_status(p->idle);
        led_set(LED_NOTHING);
        api_state("idle");
        return;
    }
    led_set(LED_BUSY);
    api_state("processing");
    const int64_t t0 = esp_timer_get_time();
    // another press is already waiting: take the streaming text now rather than making that utterance wait for a full-context pass
    const int next_waiting = uxQueueMessagesWaiting(g_ptt) > 0;
    if (p->st && (next_waiting || !(MIC_FINAL_FULL && p->n_utt <= PTT_CAP))) tasr_nemo_stream_finish(p->st, p->text, p->text_len);
    else {
        int n = p->n_utt;
        if (n + PTT_PAD <= PTT_CAP) {  // silence after the release: fewer cut-off and empty short words (docs/EXPERIMENTS.md, E13)
            memset(p->utt + n, 0, sizeof(int16_t) * PTT_PAD);
            n += PTT_PAD;
        }
        tasr_nemo_transcribe(p->m, p->utt, n, p->dec, p->text, p->text_len, NULL, 0, NULL);
    }
    const int64_t t1 = esp_timer_get_time();
    const double after = (t1 - (released_us ? released_us : t0)) / 1e3;
    printf("\r%s   [%.1f s audio, final text %.0f ms after %s]\n", p->text, p->n_utt / 16000.0, after,
           released_us ? "the button was released" : "the 20 s recording limit (still holding: next part follows)");
    oled_push(p->text[0] ? p->text : "(nothing recognized)");
    led_set(p->text[0] ? LED_DONE : LED_NOTHING);
    api_final(p->text, p->n_utt / 16000.0, after);
    if (p->raw && api_dump_on() && p->n_raw) api_audio(p->raw, p->n_raw);
    api_state("idle");
    snprintf(p->idle, sizeof(p->idle), "hold to talk");  // (the delay is on the serial line; the status line has room for 13 characters)
    oled_status(p->idle);
}
// Push-to-talk: an utterance is the audio between a press and a release (plus the pre-roll). Streaming models are fed
// while the button is held, for as long as it is held; the final text comes as soon as the recognizer reaches the
// release, with no pause to wait. Utterance-mode models record up to 20 s, then transcribe and go on recording.
// With TASR_FINAL_FULL a streaming model does both: partial text while held, then a full-context pass after the release
// (an utterance longer than the recording buffer keeps the streaming text).
static void run_mic_ptt(const tasr_nemo_t *m, struct tasr_decoder *dec)
{
    static int16_t in[320], blk[320], pre[PTT_PRE];
    static char text[4096];
    static tasr_seg_t seg;  // only its AGC
    tasr_seg_init(&seg, NULL, 0);
    ptt_t p = {m, dec, tasr_nemo_stream_new(m, 0, 0, dec), NULL, 0, 0, NULL, 0, text, sizeof(text), "hold to talk"};
    p.utt = heap_caps_malloc(sizeof(int16_t) * (p.st && !MIC_FINAL_FULL ? PTT_PRE : PTT_CAP), MALLOC_CAP_SPIRAM);  // pre-roll / recording
    oled_start();
    oled_status(p.idle);
    api_start(&(api_info_t){p.st ? "streaming" : "utterance", dec != NULL, true, HAS_BLE});
    printf("ready (%s, push-to-talk on GPIO%d, %.2f s after boot): hold the button while you speak\n",
           p.st ? "streaming" : "utterance", CONFIG_TASR_PTT_GPIO, esp_timer_get_time() / 1e6);
    uint32_t rd = 0;  // samples taken from the ring
    int active = 0, npre = 0, pw = 0;
    led_set(LED_IDLE);
    for (;;) {
        const int n = (int)(xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY) / 2);
        ptt_ev_t ev;
        while (xQueuePeek(g_ptt, &ev, 0) == pdTRUE && ev.at <= rd) {  // button events up to this sample
            xQueueReceive(g_ptt, &ev, 0);
            if (ev.down && !active) {  // press: start with the pre-roll, oldest sample first
                active = 1;
                for (int i = 0; i < npre; i++) p.utt[i] = pre[(pw - npre + i + PTT_PRE) % PTT_PRE];
                p.n_utt = p.pre_used = npre;
                p.n_raw = 0;
                if (p.raw) {
                    for (int i = 0; i < npre; i++) p.raw[i] = p.raw[PTT_CAP + (pw - npre + i + PTT_PRE) % PTT_PRE];
                    p.n_raw = npre;
                }
                if (p.st) {
                    tasr_nemo_stream_reset(p.st);
                    tasr_nemo_stream_feed(p.st, p.utt, p.n_utt);
                }
                oled_status("hearing...");
                led_set(LED_LISTEN);
                api_state("listening");
            } else if (!ev.down && active) {  // release: the utterance ends at this sample
                active = 0;
                npre = 0;
                ptt_end(&p, ev.t_us);
            } else if (!ev.down) {  // released without a press we acted on (the pin was low at startup)
                oled_status(p.idle);
                led_set(LED_IDLE);
            }
        }
        if (active && !p.st && p.n_utt + n > PTT_CAP) {  // utterance mode, recording full while still held
            ptt_end(&p, 0);
            p.n_utt = p.pre_used = 0;  // and the next part starts right here
            led_set(LED_LISTEN);
            api_state("listening");
        }
        tasr_seg_vad(&seg, in, blk, n);  // AGC only (the button, not the energy VAD, decides what is speech)
        rd += n;
        if (api_dump_on() && !p.raw) p.raw = heap_caps_calloc(PTT_CAP + PTT_PRE, sizeof(int16_t), MALLOC_CAP_SPIRAM);
        if (p.raw && active && p.n_raw + n <= PTT_CAP) {
            memcpy(p.raw + p.n_raw, in, sizeof(int16_t) * n);
            p.n_raw += n;
        }
        if (!active) {  // idle: keep the last 0.3 s as pre-roll
            if (p.raw) for (int i = 0; i < n; i++) p.raw[PTT_CAP + (pw + i) % PTT_PRE] = in[i];
            for (int i = 0; i < n; i++) pre[(pw + i) % PTT_PRE] = blk[i];
            pw = (pw + n) % PTT_PRE;
            npre = npre + n < PTT_PRE ? npre + n : PTT_PRE;
            continue;
        }
        if (p.st) {
            const int before = tasr_nemo_stream_text(p.st, NULL, 0);
            if (tasr_nemo_stream_feed(p.st, blk, n) > before) {
                tasr_nemo_stream_text(p.st, text, sizeof(text));
                printf("\r... %s", text);
                fflush(stdout);
                oled_partial(text);
                api_human_partial();
                api_partial(text);
            }
        }
        if ((!p.st || MIC_FINAL_FULL) && p.n_utt + n <= PTT_CAP) memcpy(p.utt + p.n_utt, blk, sizeof(int16_t) * n);
        p.n_utt += n;
    }
}
#endif

// Utterance mode for the NeMo engine: record while speech is detected (max 20 s), transcribe after 0.6 s of silence.
void run_mic_nemo(const tasr_nemo_t *m, struct tasr_decoder *dec)
{
#ifdef CONFIG_TASR_MODE_MIC
    mic_start();
#if CONFIG_TASR_PTT_GPIO >= 0
    run_mic_ptt(m, dec);  // does not return
#endif
    const int cap = MIC_UTT_SAMPLES;
    int16_t *utt = heap_caps_malloc(sizeof(int16_t) * cap, MALLOC_CAP_SPIRAM);
    static int16_t in[320];
    static char text[4096];
    static tasr_seg_t seg;
    tasr_seg_init(&seg, utt, cap);
    seg.hang = CONFIG_TASR_SEG_HANG_MS / 20;  // 320-sample blocks
    oled_start();
    char idle[32] = "listening";
    oled_status(idle);
    int hearing = 0;
    tasr_nemo_stream_t *st = tasr_nemo_stream_new(m, 0, 0, dec);
    api_start(&(api_info_t){st ? "streaming" : "utterance", dec != NULL, false, HAS_BLE});
    if (st) {  // streaming model: the encoder runs while you speak; only the last chunk is left when you stop
        printf("ready (streaming, %.2f s after boot): speak, partial text follows you, the final line prints after each pause\n",
               esp_timer_get_time() / 1e6);
        int fed = 0;
        led_set(LED_IDLE);
        for (;;) {
            size_t got = xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY);
            const int n = tasr_seg_feed(&seg, in, (int)(got / 2));
            if (hearing && !seg.speech && !n) {  // a blip the segmenter discarded: forget what the stream heard
                tasr_nemo_stream_reset(st);
                fed = 0;
                oled_status(idle);
                led_set(LED_IDLE);
            }
            if (seg.speech && !hearing) { led_set(LED_LISTEN); api_state("listening"); }
            hearing = seg.speech;
            if (!seg.speech && !n) continue;
            const int have = n ? n : seg.n;
            if (have > fed) {
                const int before = tasr_nemo_stream_text(st, NULL, 0);
                if (tasr_nemo_stream_feed(st, utt + fed, have - fed) > before) {
                    tasr_nemo_stream_text(st, text, sizeof(text));
                    printf("\r... %s", text);
                    fflush(stdout);
                    oled_partial(text);
                    api_human_partial();
                    api_partial(text);
                }
                fed = have;
            }
            if (!n) continue;
            led_set(LED_BUSY);
            api_state("processing");
            int64_t t0 = esp_timer_get_time();
#ifdef CONFIG_TASR_FINAL_FULL
            oled_status("working...");
            tasr_nemo_transcribe(m, utt, n, dec, text, sizeof(text), NULL, 0, NULL);
#else
            tasr_nemo_stream_finish(st, text, sizeof(text));
#endif
            const double dt = (esp_timer_get_time() - t0) / 1e6;
            printf("\r%s   [%.1f s audio, final text %.0f ms after the pause was detected]\n", text, n / 16000.0, dt * 1000);
            oled_push(text[0] ? text : "(nothing recognized)");
            led_set(text[0] ? LED_DONE : LED_NOTHING);
            api_final(text, n / 16000.0, dt * 1000);
            api_state("idle");
            oled_status(idle);
            tasr_nemo_stream_reset(st);
            fed = 0;
            hearing = 0;
            tasr_seg_next(&seg);
        }
    }
    printf("ready (%.2f s after boot): speak, the transcript prints after each pause\n", esp_timer_get_time() / 1e6);
    led_set(LED_IDLE);
    for (;;) {
        size_t got = xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY);
        const int n = tasr_seg_feed(&seg, in, (int)(got / 2));
        if (seg.speech != hearing) {  // speech started, or a blip was discarded
            hearing = seg.speech;
            oled_status(hearing ? "hearing..." : idle);
            led_set(hearing ? LED_LISTEN : LED_IDLE);
            api_state(hearing ? "listening" : "idle");
        }
        if (!n) continue;
        oled_status("working...");
        led_set(LED_BUSY);
        api_state("processing");
        int64_t t0 = esp_timer_get_time();
        tasr_nemo_transcribe(m, utt, n, dec, text, sizeof(text), NULL, 0, NULL);
        const double dt = (esp_timer_get_time() - t0) / 1e6;
        printf("%s   [%.1f s audio, %.2f s compute, RTF %.2f]\n", text, n / 16000.0, dt, dt / (n / 16000.0));
        oled_push(text[0] ? text : "(nothing recognized)");
        led_set(text[0] ? LED_DONE : LED_NOTHING);
        api_final(text, n / 16000.0, dt * 1000);
        api_state("idle");
        oled_status(idle);
        hearing = 0;
        tasr_seg_next(&seg);
    }
#else
    (void)m; (void)dec;
#endif
}

void run_mic(tasr_stream_t *s)
{
#ifdef CONFIG_TASR_MODE_MIC
    mic_start();

    static int16_t in[320], pcm[320];
    static tasr_seg_t vad;  // shared VAD/AGC; the streaming model consumes audio as it arrives
    tasr_seg_init(&vad, NULL, 0);
    oled_start();
    oled_status("listening");
    for (;;) {
        size_t got = xStreamBufferReceive(g_ring, in, sizeof(in), portMAX_DELAY);
        const int n = (int)(got / 2);
        const int voiced = tasr_seg_vad(&vad, in, pcm, n);
        if (voiced) { vad.speech = 1; vad.silence = 0; }
        else vad.silence++;
        if (!vad.speech) continue;
        if (tasr_stream_feed(s, pcm, n) > 0) {
            printf("\r%s", tasr_stream_text(s));
            fflush(stdout);
            oled_partial(tasr_stream_text(s));
        }
        if (vad.silence > TASR_SEG_HANG_BLOCKS) {  // 0.8 s of silence ends the utterance
            tasr_stream_finish(s);
            printf("\r%s\n", tasr_stream_text(s));
            oled_push(tasr_stream_text(s));
            tasr_stream_reset(s);
            vad.speech = 0;
        }
    }
#else
    (void)s;
#endif
}
