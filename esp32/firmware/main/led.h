// Optional on-board addressable RGB LED (WS2812, e.g. GPIO48 on ESP32-S3-DevKitC-1 v1.0, GPIO38 on v1.1) showing what
// the recognizer is doing. All functions are no-ops when CONFIG_TASR_LED_GPIO is -1.
#pragma once

typedef enum {
    LED_OFF,
    LED_BOOT,     // dim blue: loading the model
    LED_IDLE,     // dim green: ready, waiting for speech / the button
    LED_LISTEN,   // blue: hearing you
    LED_BUSY,     // amber: computing the final text
    LED_DONE,     // green flash, then idle: text ready
    LED_NOTHING,  // dim red flash, then idle: nothing recognized / too short
    LED_BEHIND,   // purple: the audio buffer is full and audio is being dropped
} led_state_t;

void led_start(void);
void led_set(led_state_t s);  // any task; flashes return to idle by themselves
