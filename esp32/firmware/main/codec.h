// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// ES8311 microphone codec (CONFIG_TASR_MIC_CODEC) and the I2C bus it shares with the OLED.
#pragma once
#include "driver/i2c_master.h"
#include "esp_err.h"

// the I2C bus on port 0, created on first use (the codec and the OLED share it)
i2c_master_bus_handle_t tasr_i2c_bus(int sda, int scl);
// reset and configure the codec for 16 kHz microphone capture; call before the I2S clocks start
esp_err_t codec_start(void);
