// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// ES8311 codec as a microphone input (Waveshare ESP32-P4-Pico), on the new I2C master driver: the register sequence
// follows Espressif's es8311 driver (Apache-2.0, espressif/esp-bsp), fixed to 16 kHz, MCLK 256 x fs (4.096 MHz),
// 32-bit I2S slots, I2S slave, analog microphone. NOT TESTED ON HARDWARE.
#include "codec.h"
#include "sdkconfig.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "codec";

i2c_master_bus_handle_t tasr_i2c_bus(int sda, int scl)
{
    i2c_master_bus_handle_t bus = NULL;
    if (i2c_master_get_bus_handle(I2C_NUM_0, &bus) == ESP_OK && bus) return bus;
    i2c_master_bus_config_t bc = {
        .i2c_port = I2C_NUM_0, .sda_io_num = sda, .scl_io_num = scl,
        .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7, .flags.enable_internal_pullup = true,
    };
    if (i2c_new_master_bus(&bc, &bus) != ESP_OK) return NULL;
    return bus;
}

#ifdef CONFIG_TASR_MIC_CODEC

static i2c_master_dev_handle_t s_dev;

static esp_err_t wr(uint8_t reg, uint8_t v)
{
    const uint8_t b[2] = {reg, v};
    return i2c_master_transmit(s_dev, b, 2, 100);
}

static esp_err_t rd(uint8_t reg, uint8_t *v)
{
    return i2c_master_transmit_receive(s_dev, &reg, 1, v, 1, 100);
}

#define TRY(x) do { esp_err_t e_ = (x); if (e_ != ESP_OK) { ESP_LOGE(TAG, "I2C error 0x%x at line %d", e_, __LINE__); return e_; } } while (0)

esp_err_t codec_start(void)
{
#if CONFIG_TASR_PA_GPIO >= 0
    gpio_set_direction(CONFIG_TASR_PA_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(CONFIG_TASR_PA_GPIO, 0);  // speaker amplifier off: we only listen
#endif
    i2c_master_bus_handle_t bus = tasr_i2c_bus(CONFIG_TASR_CODEC_I2C_SDA, CONFIG_TASR_CODEC_I2C_SCL);
    if (!bus) { ESP_LOGE(TAG, "I2C bus init failed"); return ESP_FAIL; }
    const i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = CONFIG_TASR_CODEC_ADDR,
                                    .scl_speed_hz = 400000};
    TRY(i2c_master_bus_add_device(bus, &dc, &s_dev));
    uint8_t v;
    if (rd(0xFD, &v) != ESP_OK) { ESP_LOGE(TAG, "no codec at 0x%02x (SDA %d, SCL %d)", CONFIG_TASR_CODEC_ADDR, CONFIG_TASR_CODEC_I2C_SDA, CONFIG_TASR_CODEC_I2C_SCL); return ESP_FAIL; }
    uint8_t id2 = 0, ver = 0;
    rd(0xFE, &id2);
    rd(0xFF, &ver);
    ESP_LOGI(TAG, "chip id 0x%02x%02x version 0x%02x (an ES8311 reads 83 11)", v, id2, ver);

    TRY(wr(0x00, 0x1F));  // reset
    vTaskDelay(pdMS_TO_TICKS(20));
    TRY(wr(0x00, 0x00));
    TRY(wr(0x00, 0x80));  // power on, slave serial port
    TRY(wr(0x01, 0x3F));  // all clocks on, MCLK from the pin
    TRY(rd(0x06, &v));
    TRY(wr(0x06, (uint8_t)((v & 0xC0) | (4 - 1))));  // BCLK divider 4: 4.096 MHz / 4 = 64 x fs; BCLK not inverted
    // coefficients for MCLK 4096000, 16 kHz: pre_div 1, pre_multi 1x, adc/dac_div 1, single speed, LRCK divider 0x00ff, OSR 0x10
    TRY(rd(0x02, &v));
    TRY(wr(0x02, (uint8_t)((v & 0x07) | ((1 - 1) << 5) | (0 << 3))));
    TRY(wr(0x03, 0x10));
    TRY(wr(0x04, 0x10));
    TRY(wr(0x05, 0x00));
    TRY(rd(0x07, &v));
    TRY(wr(0x07, (uint8_t)(v & 0xC0)));
    TRY(wr(0x08, 0xFF));
    TRY(rd(0x00, &v));
    TRY(wr(0x00, (uint8_t)(v & 0xBF)));  // serial port slave
    TRY(wr(0x09, 4 << 2));  // 32-bit I2S in
    TRY(wr(0x0A, 4 << 2));  // 32-bit I2S out (the microphone data)
    TRY(wr(0x0D, 0x01));    // analog circuits on
    TRY(wr(0x0E, 0x02));    // PGA and ADC modulator on
    TRY(wr(0x12, 0x00));
    TRY(wr(0x13, 0x10));
    TRY(wr(0x1C, 0x6A));    // ADC equalizer bypass, DC offset cancelled
    TRY(wr(0x37, 0x08));
    TRY(wr(0x17, 0xC8));    // ADC digital volume
    TRY(wr(0x14, 0x1A));    // analog microphone, PGA gain
    TRY(wr(0x16, CONFIG_TASR_CODEC_MIC_GAIN));  // ADC gain scale-up, 6 dB per step
    ESP_LOGI(TAG, "ES8311 set up: 16 kHz, mic gain %d dB", 6 * CONFIG_TASR_CODEC_MIC_GAIN);
    return ESP_OK;
}
#endif
