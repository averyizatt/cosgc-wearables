// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// WS2812 status LED on the RMT peripheral (no external component): 24 bits GRB, 0.3 / 0.9 us high times at 10 MHz.
#include "led.h"
#include <stdint.h>
#include "sdkconfig.h"

#if CONFIG_TASR_LED_GPIO >= 0
#include "driver/rmt_tx.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static rmt_channel_handle_t s_ch;
static rmt_encoder_handle_t s_enc;
static SemaphoreHandle_t s_mux;
static esp_timer_handle_t s_flash;  // returns a flash to idle
static led_state_t s_state;

static void put(uint8_t r, uint8_t g, uint8_t b)
{
    static uint8_t px[3];
    const rmt_transmit_config_t tc = {.loop_count = 0};
    px[0] = g; px[1] = r; px[2] = b;
    rmt_transmit(s_ch, s_enc, px, sizeof(px), &tc);
    rmt_tx_wait_all_done(s_ch, 10);
}

static void show(led_state_t s)
{
    switch (s) {  // kept dim: a wearable's LED should not dominate the battery
    case LED_BOOT: put(0, 0, 6); break;
    case LED_IDLE: put(0, 4, 0); break;
    case LED_LISTEN: put(0, 0, 40); break;
    case LED_BUSY: put(32, 12, 0); break;
    case LED_DONE: put(0, 40, 0); break;
    case LED_NOTHING: put(12, 0, 0); break;
    case LED_BEHIND: put(24, 0, 24); break;
    default: put(0, 0, 0); break;
    }
}

static void flash_end(void *arg)
{
    (void)arg;
    xSemaphoreTake(s_mux, portMAX_DELAY);
    if (s_state == LED_DONE || s_state == LED_NOTHING) {
        s_state = LED_IDLE;
        show(s_state);
    }
    xSemaphoreGive(s_mux);
}

void led_start(void)
{
    if (s_ch) return;
    const rmt_tx_channel_config_t cfg = {.gpio_num = CONFIG_TASR_LED_GPIO, .clk_src = RMT_CLK_SRC_DEFAULT,
                                         .resolution_hz = 10000000, .mem_block_symbols = 64, .trans_queue_depth = 4};
    if (rmt_new_tx_channel(&cfg, &s_ch) != ESP_OK) { s_ch = NULL; ESP_LOGW("led", "no RMT channel for the LED"); return; }
    const rmt_bytes_encoder_config_t bc = {.bit0 = {.level0 = 1, .duration0 = 3, .level1 = 0, .duration1 = 9},
                                           .bit1 = {.level0 = 1, .duration0 = 9, .level1 = 0, .duration1 = 3},
                                           .flags.msb_first = 1};
    rmt_new_bytes_encoder(&bc, &s_enc);
    rmt_enable(s_ch);
    s_mux = xSemaphoreCreateMutex();
    const esp_timer_create_args_t ta = {.callback = flash_end, .name = "led"};
    esp_timer_create(&ta, &s_flash);
    led_set(LED_BOOT);
}

void led_set(led_state_t s)
{
    if (!s_ch) return;
    xSemaphoreTake(s_mux, portMAX_DELAY);
    if (s != s_state) {
        s_state = s;
        show(s);
    }
    xSemaphoreGive(s_mux);
    if (s == LED_DONE || s == LED_NOTHING) {
        esp_timer_stop(s_flash);
        esp_timer_start_once(s_flash, s == LED_DONE ? 400000 : 250000);
    }
}
#else
void led_start(void) {}
void led_set(led_state_t s) { (void)s; }
#endif
