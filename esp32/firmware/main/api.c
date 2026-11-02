// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// The device API: see api.h and docs/API.md.
#include "api.h"
#include <stdio.h>
#include <string.h>
#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "hal/usb_serial_jtag_ll.h"
#ifdef CONFIG_IDF_TARGET_ESP32P4
#include "driver/uart.h"
#endif
#include "cJSON.h"
#include "oled.h"  // TASR_AUX_CORE
#include "sdkconfig.h"

#define MAX_SINKS 3
#define LINE_MAX_BYTES 6144  // longest outgoing line (a minute or more of speech); incoming lines are shorter
#define RX_MAX_BYTES 512

static SemaphoreHandle_t s_mux;
static char *s_out, *s_line, *s_esc, *s_prev;  // line buffers and the last partial text (PSRAM)
static api_sink_t s_sinks[MAX_SINKS];
static int s_nsinks;
static api_info_t s_info;
static volatile bool s_talk;       // virtual button
static volatile bool s_level_on;   // a client asked for level events
static volatile int s_pending;     // the console log has an unfinished line on screen
static int s_utt;                  // number of the current utterance (starts at 1 with the first "listening")
static const char *s_state = "idle";
static double s_last_delay_ms;
static int64_t s_level_t;
static api_ext_t s_ext;

// least stack ever left free (bytes) of the named task, or 0 if there is no such task
static unsigned stack_min(const char *name)
{
    TaskHandle_t t = xTaskGetHandle(name);
    return t ? (unsigned)uxTaskGetStackHighWaterMark(t) : 0;
}

void api_set_ext_handler(api_ext_t fn) { s_ext = fn; }

void api_add_sink(api_sink_t sink)
{
    if (s_nsinks < MAX_SINKS) s_sinks[s_nsinks++] = sink;
}
bool api_talk_held(void) { return s_talk; }
void api_human_partial(void) { s_pending = 1; }

// JSON string escaping into dst (UTF-8 passes through); returns bytes written, always terminated
static int esc(char *dst, int max, const char *s)
{
    int n = 0;
    for (; *s && n < max - 7; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { dst[n++] = '\\'; dst[n++] = (char)c; }
        else if (c == '\n') { dst[n++] = '\\'; dst[n++] = 'n'; }
        else if (c == '\r') { dst[n++] = '\\'; dst[n++] = 'r'; }
        else if (c == '\t') { dst[n++] = '\\'; dst[n++] = 't'; }
        else if (c < 0x20) n += snprintf(dst + n, 7, "\\u%04x", c);
        else dst[n++] = (char)c;
    }
    dst[n] = 0;
    return n;
}

// one outgoing line: the USB serial port (stdout, in a single write so console log lines cannot get inside it) and every
// registered transport
static void send_line(const char *line, int len)
{
    xSemaphoreTake(s_mux, portMAX_DELAY);
    char *out = s_out;
    int n = 0;
    if (s_pending) { out[n++] = '\n'; s_pending = 0; }  // finish the console log's unfinished "\r... text" line first
    memcpy(out + n, line, len);
    n += len;
    out[n++] = '\n';
    fwrite(out, 1, n, stdout);
    fflush(stdout);
    for (int i = 0; i < s_nsinks; i++) s_sinks[i](line, len);
    xSemaphoreGive(s_mux);
}

static void send_fmt_text(const char *head, const char *text, const char *tail)
{
    char *line = s_line, *esc_text = s_esc;  // shared buffers: the caller holds s_fmt_mux
    esc(esc_text, LINE_MAX_BYTES / 2, text ? text : "");
    int n = snprintf(line, LINE_MAX_BYTES, "%s\"%s\"%s", head, esc_text, tail);
    if (n > LINE_MAX_BYTES - 1) n = LINE_MAX_BYTES - 1;
    send_line(line, n);
}

static SemaphoreHandle_t s_fmt_mux;  // serializes the static buffers of the event builders

void api_state(const char *state)
{
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    if (!strcmp(state, "listening")) { s_utt++; s_prev[0] = 0; }
    s_state = state;
    char line[96];
    const int n = snprintf(line, sizeof(line), "{\"type\":\"state\",\"state\":\"%s\",\"utt\":%d}", state, s_utt);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

void api_partial(const char *text)
{
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    if (!text[0] || !strncmp(text, s_prev, LINE_MAX_BYTES / 2 - 1)) {  // nothing new: clients hear changes only
        xSemaphoreGive(s_fmt_mux);
        return;
    }
    snprintf(s_prev, LINE_MAX_BYTES / 2, "%s", text);
    char head[64];
    snprintf(head, sizeof(head), "{\"type\":\"partial\",\"utt\":%d,\"text\":", s_utt);
    send_fmt_text(head, text, "}");
    xSemaphoreGive(s_fmt_mux);
}

void api_final(const char *text, double audio_s, double delay_ms)
{
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    s_last_delay_ms = delay_ms;
    char head[64], tail[96];
    snprintf(head, sizeof(head), "{\"type\":\"final\",\"utt\":%d,\"text\":", s_utt);
    snprintf(tail, sizeof(tail), ",\"audio_s\":%.1f,\"delay_ms\":%.0f}", audio_s, delay_ms);
    send_fmt_text(head, text, tail);
    xSemaphoreGive(s_fmt_mux);
}

void api_button(bool down)
{
    if (!s_mux) return;  // a press while the model is still loading: api_start has not run
    char line[64];
    const int n = snprintf(line, sizeof(line), "{\"type\":\"button\",\"down\":%s}", down ? "true" : "false");
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

void api_warn(const char *code, int value)
{
    if (!s_mux) return;
    char line[96];
    const int n = snprintf(line, sizeof(line), "{\"type\":\"warn\",\"code\":\"%s\",\"value\":%d}", code, value);
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

void api_ble(const char *event)
{
    char line[64];
    const int n = snprintf(line, sizeof(line), "{\"type\":\"ble\",\"event\":\"%s\"}", event);
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

void api_level(float dbfs)
{
    if (!s_level_on || !s_mux) return;
    const int64_t now = esp_timer_get_time();
    if (now - s_level_t < 200000) return;  // at most 5 a second
    s_level_t = now;
    char line[64];
    const int n = snprintf(line, sizeof(line), "{\"type\":\"level\",\"db\":%.1f}", dbfs);
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

static void send_hello(void)
{
    const esp_app_desc_t *d = esp_app_get_description();
    char line[320];
    const int n = snprintf(line, sizeof(line),
                           "{\"type\":\"hello\",\"api\":%d,\"fw\":\"%s\",\"mode\":\"%s\",\"lm\":%s,\"ptt\":%s,\"ble\":%s,\"cores\":%d,"
                           "\"uptime_s\":%d,\"psram_free\":%u,\"internal_free\":%u}",
                           API_VERSION, d ? d->version : "?", s_info.mode, s_info.lm ? "true" : "false",
                           s_info.ptt ? "true" : "false", s_info.ble ? "true" : "false",
#ifdef TASR_SINGLE_CORE
                           1,
#else
                           2,
#endif
                           (int)(esp_timer_get_time() / 1000000), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

void api_ack(const char *cmd, int idv, bool ok, const char *err)
{
    char line[160];
    char idbuf[32] = "";
    if (idv >= 0) snprintf(idbuf, sizeof(idbuf), ",\"id\":%d", idv);
    int n = snprintf(line, sizeof(line), "{\"type\":\"ack\",\"cmd\":\"%.20s\",\"ok\":%s%s", cmd, ok ? "true" : "false", idbuf);
    if (err) n += snprintf(line + n, sizeof(line) - n, ",\"error\":\"%.40s\"", err);
    n += snprintf(line + n, sizeof(line) - n, "}");
    xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
    send_line(line, n);
    xSemaphoreGive(s_fmt_mux);
}

static void reply(const cJSON *id, const char *cmd, bool ok, const char *err)
{
    api_ack(cmd, cJSON_IsNumber(id) ? (int)id->valuedouble : -1, ok, err);
}

void api_handle_line(const char *text)
{
    while (*text == ' ' || *text == '\t') text++;
    if (*text != '{') return;  // not JSON (a terminal's stray keystroke, or a console log echo)
    cJSON *j = cJSON_Parse(text);
    const cJSON *c = j ? cJSON_GetObjectItemCaseSensitive(j, "cmd") : NULL;
    if (!cJSON_IsString(c)) {
        reply(NULL, "?", false, "bad json: need {\"cmd\":...}");
        cJSON_Delete(j);
        return;
    }
    const char *cmd = c->valuestring;
    const cJSON *id = cJSON_GetObjectItemCaseSensitive(j, "id");
    if (!strcmp(cmd, "hello")) {
        send_hello();
        reply(id, cmd, true, NULL);
    } else if (!strcmp(cmd, "ping")) {
        reply(id, cmd, true, NULL);
    } else if (!strcmp(cmd, "talk")) {  // the virtual button: {"cmd":"talk","on":true}
        const cJSON *on = cJSON_GetObjectItemCaseSensitive(j, "on");
        if (cJSON_IsBool(on)) { s_talk = cJSON_IsTrue(on); reply(id, cmd, true, NULL); }
        else reply(id, cmd, false, "need on:true|false");
    } else if (!strcmp(cmd, "set")) {  // {"cmd":"set","level":true}
        const cJSON *lv = cJSON_GetObjectItemCaseSensitive(j, "level");
        if (cJSON_IsBool(lv)) { s_level_on = cJSON_IsTrue(lv); reply(id, cmd, true, NULL); }
        else reply(id, cmd, false, "unknown setting");
    } else if (!strcmp(cmd, "status")) {
        char line[320];
        const int n = snprintf(line, sizeof(line),
                               "{\"type\":\"status\",\"state\":\"%s\",\"utt\":%d,\"last_delay_ms\":%.0f,\"uptime_s\":%d,"
                               "\"psram_free\":%u,\"internal_free\":%u,\"internal_min\":%u,\"stack_free_main\":%u,\"stack_free_worker\":%u}",
                               s_state, s_utt, s_last_delay_ms, (int)(esp_timer_get_time() / 1000000),
                               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                               (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL), stack_min("main"), stack_min("asr_w1"));
        xSemaphoreTake(s_fmt_mux, portMAX_DELAY);
        send_line(line, n);
        xSemaphoreGive(s_fmt_mux);
        reply(id, cmd, true, NULL);
    } else if (!strcmp(cmd, "reboot")) {
        reply(id, cmd, true, NULL);
        vTaskDelay(pdMS_TO_TICKS(100));
        esp_restart();
    } else if (s_ext && s_ext(cmd, cJSON_IsNumber(id) ? (int)id->valuedouble : -1)) {
        // handled (and acked) by a transport
    } else {
        reply(id, cmd, false, "unknown command");
    }
    cJSON_Delete(j);
}

// USB serial port reader. S3: the console reads UART, so the USB-Serial/JTAG receive FIFO is polled directly.
// P4 board: the USB port is a CH343 bridge to the console UART, read through the driver (its FIFO alone overflows on long lines).
static void rx_task(void *arg)
{
    (void)arg;
    static char line[RX_MAX_BYTES];
    int n = 0;
#ifdef CONFIG_IDF_TARGET_ESP32P4
    uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 2048, 0, 0, NULL, 0);
#endif
    for (;;) {
        uint8_t b[64];
        int got = 0;
#ifdef CONFIG_IDF_TARGET_ESP32P4
        got = uart_read_bytes(CONFIG_ESP_CONSOLE_UART_NUM, b, sizeof(b), pdMS_TO_TICKS(20));
        if (got < 0) got = 0;
#else
        while (usb_serial_jtag_ll_rxfifo_data_available() && got < (int)sizeof(b)) {
            got += usb_serial_jtag_ll_read_rxfifo(b + got, sizeof(b) - got);
        }
#endif
        for (int i = 0; i < got; i++) {
            if (b[i] == '\n' || b[i] == '\r') {
                if (n) { line[n] = 0; api_handle_line(line); n = 0; }
            } else if (n < RX_MAX_BYTES - 1) {
                line[n++] = (char)b[i];
            } else {
                n = 0;  // too long: drop it
            }
        }
        vTaskDelay(pdMS_TO_TICKS(got ? 1 : 20));
    }
}

void api_start(const api_info_t *info)
{
    if (s_mux) return;
    s_info = *info;
    s_mux = xSemaphoreCreateMutex();
    s_fmt_mux = xSemaphoreCreateMutex();
    s_out = heap_caps_malloc(LINE_MAX_BYTES + 4, MALLOC_CAP_SPIRAM);
    s_line = heap_caps_malloc(LINE_MAX_BYTES, MALLOC_CAP_SPIRAM);
    s_esc = heap_caps_malloc(LINE_MAX_BYTES / 2, MALLOC_CAP_SPIRAM);
    s_prev = heap_caps_calloc(LINE_MAX_BYTES / 2, 1, MALLOC_CAP_SPIRAM);
    xTaskCreatePinnedToCore(rx_task, "api_rx", 4096, NULL, 3, NULL, TASR_AUX_CORE);  // not on the spinning worker's core
    send_hello();
}
