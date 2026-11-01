# ESP32-P4 board (ESP32-P4NRW32): the benchmark runs on real hardware, the microphone path not yet. Used instead of sdkconfig.defaults (that file is S3).
CONFIG_IDF_TARGET="esp32p4"
CONFIG_ESPTOOLPY_FLASHSIZE_32MB=y
CONFIG_ESPTOOLPY_FLASHMODE_QIO=y
CONFIG_ESPTOOLPY_FLASHFREQ_80M=y
CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="esp32/firmware/partitions_p4_32.csv"
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_HEX=y
CONFIG_SPIRAM_SPEED_200M=y
CONFIG_SPIRAM_USE_CAPS_ALLOC=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_400=y
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768
CONFIG_CACHE_L2_CACHE_256KB=y
CONFIG_CACHE_L2_CACHE_LINE_128B=y
CONFIG_ESP_TASK_WDT_EN=n
CONFIG_ESP_INT_WDT=n
CONFIG_FREERTOS_HZ=1000
# this board's USB port is a CH343 bridge to UART0 (it shows up as a COM port; the ROM log appears there), not the chip's
# USB-Serial/JTAG; the device API reads and writes the same UART
CONFIG_ESP_CONSOLE_UART_DEFAULT=y
CONFIG_ESP_CONSOLE_UART_BAUDRATE=115200

CONFIG_TASR_LM_WEIGHT_X100=30
CONFIG_TASR_TOKEN_BONUS_X100=50
# the language model as stored (int8), as in the host evaluations; int4 is an S3 trick for its slower PSRAM, and int8 rows can use both cores
# CONFIG_TASR_LM_INT4 is not set
# no S3 cache prefetch registers on the P4
CONFIG_TASR_EXP_DCACHE_AUTOLOAD=-1
# vector kernels, self-checked at boot against plain C (set to n if the board crashes before the P4SELFTEST lines)
CONFIG_TASR_P4_PIE=y

# Waveshare ESP32-P4-Pico: onboard microphone on the ES8311 (I2C SDA 7 / SCL 8, I2S MCLK 13, SCLK 12, LRCK 10, ASDOUT 11)
CONFIG_TASR_MODE_MIC=y
CONFIG_TASR_MIC_CODEC=y
CONFIG_TASR_I2S_BCLK=12
CONFIG_TASR_I2S_WS=10
CONFIG_TASR_I2S_DIN=11
CONFIG_TASR_I2S_MCLK=13
CONFIG_TASR_CODEC_I2C_SDA=7
CONFIG_TASR_CODEC_I2C_SCL=8
CONFIG_TASR_PA_GPIO=53
# SSD1306 128x64 OLED on the codec's I2C bus (0x3C, VCC to 3V3): safe to leave on without a screen, the firmware logs a warning
CONFIG_TASR_OLED=y
CONFIG_TASR_OLED_SDA=7
CONFIG_TASR_OLED_SCL=8
CONFIG_TASR_OLED_ADDR=0x3C
# no LED on the board; a push button on a header pin can be added with CONFIG_TASR_PTT_GPIO (-1: the VAD ends utterances)
CONFIG_TASR_SEG_HANG_MS=800
CONFIG_TASR_PTT_GPIO=-1
CONFIG_TASR_LED_GPIO=-1
