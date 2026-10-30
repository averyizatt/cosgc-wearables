// SSD1306 driver + render task. The ASR code only copies strings under a mutex and notifies the task, so a display
// update (1 KB over I2C at 400 kHz, ~25 ms) never blocks recognition.
#include "oled.h"
#include "codec.h"
#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"

#ifdef CONFIG_TASR_OLED
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char *TAG = "oled";
static i2c_master_dev_handle_t s_dev;
static SemaphoreHandle_t s_mux;
static TaskHandle_t s_task;
static char s_status[32] = "starting";
static char s_hist[640];   // finished utterances, '\n'-separated, oldest trimmed first
static char s_part[256];   // in-progress utterance (streaming)

static esp_err_t cmd(const uint8_t *c, int n)
{
    uint8_t b[32];
    b[0] = 0x00;  // control byte: command stream
    memcpy(b + 1, c, n);
    return i2c_master_transmit(s_dev, b, n + 1, 50);
}

#define N_FULL 1u   // text changed: redraw the whole screen
#define N_LEVEL 2u  // only the level bar changed: redraw the status line (page 0, 128 bytes, ~3 ms)
static volatile int s_level_px = -1;  // -1: no level yet (no bar)
static float s_level_db = -120.f;     // smoothed level
static int64_t s_level_t;

static void render_task(void *arg)
{
    (void)arg;
    static uint8_t fb[1 + OLED_W * OLED_H / 8];
    static char st[32], hist[sizeof(s_hist) + sizeof(s_part) + 2];
    for (;;) {
        uint32_t bits = 0;
        xTaskNotifyWait(0, UINT32_MAX, &bits, portMAX_DELAY);
        xSemaphoreTake(s_mux, portMAX_DELAY);
        strcpy(st, s_status);
        snprintf(hist, sizeof(hist), "%s%s%s", s_hist, (s_hist[0] && s_part[0]) ? "\n" : "", s_part);
        xSemaphoreGive(s_mux);
        oled_render(fb + 1, st, hist, s_level_px);
        fb[0] = 0x40;  // control byte: data stream
        if (bits & N_FULL) {
            static const uint8_t win[] = {0x21, 0, 127, 0x22, 0, 7};  // full-screen column/page window
            cmd(win, sizeof(win));
            i2c_master_transmit(s_dev, fb, sizeof(fb), 100);
        } else {
            static const uint8_t win[] = {0x21, 0, 127, 0x22, 0, 0};  // page 0 only
            cmd(win, sizeof(win));
            i2c_master_transmit(s_dev, fb, 1 + OLED_W, 100);
        }
    }
}

// fast attack, release 40 dB/s (called per 20 ms block); redraws only when the bar changes, at most 12 times a second
void oled_level(float dbfs)
{
    if (!s_task) return;
    s_level_db = dbfs > s_level_db ? dbfs : s_level_db - 0.8f;
    float f = (s_level_db - LEVEL_DB_MIN) / (LEVEL_DB_MAX - LEVEL_DB_MIN);
    f = f < 0.f ? 0.f : f > 1.f ? 1.f : f;
    const int px = (int)(f * LEVEL_PX + 0.5f);
    const int64_t now = esp_timer_get_time();
    if (px == s_level_px || now - s_level_t < 80000) return;
    s_level_t = now;
    s_level_px = px;
    xTaskNotify(s_task, N_LEVEL, eSetBits);
}

void oled_start(void)
{
    if (s_task) return;
    i2c_master_bus_handle_t bus = tasr_i2c_bus(CONFIG_TASR_OLED_SDA, CONFIG_TASR_OLED_SCL);  // shared with the codec, if any
    if (!bus) { ESP_LOGE(TAG, "I2C bus init failed"); return; }
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = CONFIG_TASR_OLED_ADDR,
                              .scl_speed_hz = 400000};
    if (i2c_master_bus_add_device(bus, &dc, &s_dev) != ESP_OK) { ESP_LOGE(TAG, "I2C device add failed"); return; }
    static const uint8_t init[] = {0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00, 0xA1,
                                   0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF};
    if (cmd(init, sizeof(init)) != ESP_OK) {
        ESP_LOGW(TAG, "no SSD1306 at 0x%02x (SDA %d, SCL %d); transcript on serial only", CONFIG_TASR_OLED_ADDR,
                 CONFIG_TASR_OLED_SDA, CONFIG_TASR_OLED_SCL);
        return;
    }
    s_mux = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(render_task, "oled", 4096, NULL, 5, &s_task, TASR_AUX_CORE);
    ESP_LOGI(TAG, "SSD1306 ready (SDA %d, SCL %d)", CONFIG_TASR_OLED_SDA, CONFIG_TASR_OLED_SCL);
    xTaskNotify(s_task, N_FULL, eSetBits);
}

void oled_status(const char *s)
{
    if (!s_task) return;
    xSemaphoreTake(s_mux, portMAX_DELAY);
    snprintf(s_status, sizeof(s_status), "%s", s);
    xSemaphoreGive(s_mux);
    xTaskNotify(s_task, N_FULL, eSetBits);
}

void oled_push(const char *text)
{
    if (!s_task) return;
    xSemaphoreTake(s_mux, portMAX_DELAY);
    s_part[0] = 0;
    const size_t need = strlen(text) + 2;
    while (strlen(s_hist) + need >= sizeof(s_hist)) {  // drop the oldest utterance
        char *nl = strchr(s_hist, '\n');
        if (!nl) { s_hist[0] = 0; break; }
        memmove(s_hist, nl + 1, strlen(nl + 1) + 1);
    }
    if (s_hist[0]) strcat(s_hist, "\n");
    strncat(s_hist, text, sizeof(s_hist) - strlen(s_hist) - 1);
    xSemaphoreGive(s_mux);
    xTaskNotify(s_task, N_FULL, eSetBits);
}

void oled_partial(const char *text)
{
    if (!s_task) return;
    xSemaphoreTake(s_mux, portMAX_DELAY);
    snprintf(s_part, sizeof(s_part), "%s", text);
    xSemaphoreGive(s_mux);
    xTaskNotify(s_task, N_FULL, eSetBits);
}
#else
void oled_start(void) {}
void oled_status(const char *s) { (void)s; }
void oled_push(const char *text) { (void)text; }
void oled_partial(const char *text) { (void)text; }
void oled_level(float dbfs) { (void)dbfs; }
#endif
