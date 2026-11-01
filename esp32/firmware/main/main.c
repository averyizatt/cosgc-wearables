// tinyasr firmware for ESP32-S3 (N16R8): transcribes the utterances stored in the "audio" partition
// and reports transcript + real-time factor measured with the CPU cycle counter.
#include <stdio.h>
#include <string.h>
#include "esp_cpu.h"
#include "kernels.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "sdkconfig.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"
#include "tasr_nemo.h"
#include "ble.h"
#include "led.h"
#include "mic.h"

static const char *TAG = "tinyasr";

// ---- dual-core parallel-for: worker task pinned to core 1 runs the upper half of each job
static struct {
    tasr_job_fn fn;
    void *ctx;
    int b, e;
} g_job;
static SemaphoreHandle_t g_start, g_done;
#ifdef CONFIG_TASR_EXP_SPIN_PARFOR
// experiment: no task is ever blocked or switched while a job is running; the second core spins until the next job
static volatile uint32_t g_seq_go, g_seq_done;
static void worker_task(void *arg)
{
    (void)arg;
    uint32_t seen = 0;
    for (;;) {
        while (g_seq_go == seen) { }
        seen = g_seq_go;
        __sync_synchronize();
#ifdef CONFIG_TASR_EXP_NOINT
        const unsigned ps = portSET_INTERRUPT_MASK_FROM_ISR();
#endif
        g_job.fn(g_job.ctx, g_job.b, g_job.e, 1);
#ifdef CONFIG_TASR_EXP_NOINT
        portCLEAR_INTERRUPT_MASK_FROM_ISR(ps);
#endif
        __sync_synchronize();
        g_seq_done = seen;
    }
}
static void par_for(tasr_job_fn fn, void *ctx, int n)
{
    if (n < 2) { fn(ctx, 0, n, 0); return; }
    int mid = n / 2;
    g_job.fn = fn; g_job.ctx = ctx; g_job.b = mid; g_job.e = n;
    const uint32_t s = g_seq_go + 1;
    __sync_synchronize();
    g_seq_go = s;
#ifdef CONFIG_TASR_EXP_NOINT
    const unsigned ps = portSET_INTERRUPT_MASK_FROM_ISR();
#endif
    fn(ctx, 0, mid, 0);
#ifdef CONFIG_TASR_EXP_NOINT
    portCLEAR_INTERRUPT_MASK_FROM_ISR(ps);
#endif
    while (g_seq_done != s) { }
    __sync_synchronize();
}
#else
static void worker_task(void *arg)
{
    (void)arg;
    for (;;) {
        xSemaphoreTake(g_start, portMAX_DELAY);
        g_job.fn(g_job.ctx, g_job.b, g_job.e, 1);
        xSemaphoreGive(g_done);
    }
}
static void par_for(tasr_job_fn fn, void *ctx, int n)
{
    if (n < 2) { fn(ctx, 0, n, 0); return; }
    int mid = n / 2;
    g_job.fn = fn; g_job.ctx = ctx; g_job.b = mid; g_job.e = n;
    xSemaphoreGive(g_start);
    fn(ctx, 0, mid, 0);
    xSemaphoreTake(g_done, portMAX_DELAY);
}
#endif
#define CPU_HZ (CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1e6)

// internal RAM kept free when the engine asks for a hot work buffer: I2S DMA, the mic / OLED task stacks and their
// buffers are allocated before the engine's buffers, so this only needs to cover later runtime allocations. A Wi-Fi or
// BLE stack needs much more (raise this to ~72 KB, which costs speed: the utterance path's 45 KB activation tile then
// lands in PSRAM, measured 4 % slower).
#define INTERNAL_RESERVE (CONFIG_TASR_INTERNAL_RESERVE_KB * 1024)
static uintptr_t g_ps_lo = UINTPTR_MAX, g_ps_hi;  // span of the PSRAM blocks handed out (prefetch window)
static void *fw_alloc(size_t n, int kind)
{
    void *p = NULL;
    if (kind == 1 && n <= 48 * 1024 &&
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL) > n + INTERNAL_RESERVE)
        p = heap_caps_aligned_alloc(16, n ? n : 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (kind == 1 && !p) {  // a hot buffer the engine wanted in internal RAM: slower in PSRAM, say so (first few times)
        static int warned;
        if (warned < 8) {
            warned++;
            ESP_LOGW(TAG, "%u-byte work buffer in PSRAM (internal RAM free %u, reserve %u)", (unsigned)n,
                     (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)INTERNAL_RESERVE);
        }
    }
    if (!p) p = heap_caps_aligned_alloc(16, n ? n : 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p && esp_ptr_external_ram(p)) {
        if ((uintptr_t)p < g_ps_lo) g_ps_lo = (uintptr_t)p;
        if ((uintptr_t)p + n > g_ps_hi) g_ps_hi = (uintptr_t)p + n;
    }
    if (p) memset(p, 0, n);
    else ESP_LOGE(TAG, "alloc %u kind %d failed", (unsigned)n, kind);
    return p;
}
static void fw_free(void *p) { heap_caps_free(p); }

static const void *map_partition(uint8_t subtype, const char *name, size_t *size)
{
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, subtype, name);
    if (!p) return NULL;
    const void *ptr;
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &ptr, &h) != ESP_OK) return NULL;
    *size = p->size;
    return ptr;
}

#ifdef CONFIG_IDF_TARGET_ESP32P4
// The P4's model may cross 16 MB, which the cache cannot reach here (docs/P4.md): read the image [u32 size][pad to 16][blob]
// into PSRAM through the flash driver, which uses 4-byte commands. This also leaves all the weights in PSRAM from the start.
static const void *read_partition_to_psram(uint8_t subtype, const char *name, size_t *size)
{
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, subtype, name);
    uint32_t bytes = 0;
    if (!p || esp_partition_read(p, 0, &bytes, 4) != ESP_OK || bytes == 0 || bytes + 16 > p->size) {
        ESP_LOGE(TAG, "partition %s: no image (size word %u)", name, (unsigned)bytes);
        return NULL;
    }
    void *buf = heap_caps_aligned_alloc(16, bytes + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) { ESP_LOGE(TAG, "partition %s: no PSRAM for %u bytes", name, (unsigned)bytes); return NULL; }
    const int64_t t0 = esp_timer_get_time();
    const esp_err_t e = esp_partition_read(p, 0, buf, bytes + 16);
    if (e != ESP_OK) { ESP_LOGE(TAG, "partition %s: read failed: %s", name, esp_err_to_name(e)); heap_caps_free(buf); return NULL; }
    ESP_LOGI(TAG, "%s: %u bytes read from flash at 0x%x to PSRAM in %lld ms", name, (unsigned)bytes, (unsigned)p->address,
             (esp_timer_get_time() - t0) / 1000);
    *size = bytes + 16;
    return buf;
}
#endif

static void calibrate(void)
{
    // 4M single-cycle NOPs (+ loop setup): on silicon cycles ~= 4.0M; under QEMU -icount this gives cycles/instruction
    uint32_t c0 = esp_cpu_get_cycle_count();
    int64_t t0 = esp_timer_get_time();
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    __asm__ volatile("movi a8, 1000000\n loop a8, 1f\n nop\n nop\n nop\n nop\n 1:\n" ::: "a8");
#else  // RISC-V (ESP32-P4): 1M iterations of 4 nops; not cycle-exact like the S3 zero-overhead loop
    for (volatile int i = 0; i < 1000000; i++) __asm__ volatile("nop\n nop\n nop\n nop\n");
#endif
    uint32_t dc = esp_cpu_get_cycle_count() - c0;
    printf("CALIB 4M nops: cycles %u, time %lld us\n", (unsigned)dc, esp_timer_get_time() - t0);
}

// ---- NeMo Conformer-CTC small (utterance mode): transcribe each stored utterance, report RTF
static void run_nemo(const uint8_t *blob, uint32_t blob_size)
{
    tasr_nemo_t *m = tasr_nemo_load(blob, blob_size);
    if (!m) { ESP_LOGE(TAG, "nemo model load failed"); return; }
    tasr_decoder_t *dec = NULL;
    size_t lsize = 0;
    const uint8_t *lmp = map_partition(0x42, "lm", &lsize);
    uint32_t lbytes = 0;
    if (lmp) memcpy(&lbytes, lmp, 4);
    if (lmp && lbytes > 0 && lbytes < lsize) {
        tasr_lm_t *lm = tasr_lm_load(lmp + 16, lbytes);
        if (lm) {
#ifdef CONFIG_TASR_LM_INT4
            extern int tasr_lm_int4;
            tasr_lm_int4 = 1;  // GRU and output matrices as int4 rows: half the PSRAM traffic per LM step
#endif
            size_t mv = tasr_lm_to_ram(lm);
            dec = tasr_decoder_create(lm, 1025, CONFIG_TASR_BEAM, 6, tasr_lm_weight(lm, CONFIG_TASR_LM_WEIGHT_X100 / 100.0f),
                                      tasr_lm_bonus(lm, CONFIG_TASR_TOKEN_BONUS_X100 / 100.0f));
            ESP_LOGI(TAG, "LM %u bytes in PSRAM, beam %d", (unsigned)mv, CONFIG_TASR_BEAM);
        }
    }
    size_t free_ps = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#ifdef CONFIG_TASR_MODE_MIC
    const size_t reserve = 2500000 + MIC_PSRAM(0);  // mic ring and recording buffer (mic.h) + 20 s utterance buffers
#else
    const size_t reserve = 2700000;  // per-utterance buffers for clips up to 20 s (1.9 MB measured at 15 s)
#endif
    const int streaming = tasr_nemo_stream_supported(m);
    // a streaming model also keeps its attention / conv caches and positional rows (~2 MB at chunk 16, left 128)
    size_t keep = reserve + (streaming ? 2200000 : 0);
#ifdef CONFIG_TASR_MODE_MIC
    if (streaming) {  // live streaming never allocates the utterance buffers: keep the stream's measured size instead
        const size_t f0 = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        tasr_nemo_stream_t *probe = tasr_nemo_stream_new(m, 0, 0, dec);
        if (probe) {
            const size_t sb = f0 - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            tasr_nemo_stream_free(probe);
            keep = MIC_PSRAM(1) + sb + 300000;  // mic ring (+ recording buffer without the button), the stream, headroom
            ESP_LOGI(TAG, "stream needs %u bytes of PSRAM; the rest caches weights", (unsigned)sb);
        }
    }
#endif
    size_t budget = free_ps > keep ? free_ps - keep : 0;
#if defined(CONFIG_TASR_EXP_NO_PSRAM_WEIGHTS) || defined(CONFIG_IDF_TARGET_ESP32P4)
    budget = 0;  // on the P4 the whole model is already in PSRAM
#endif
    size_t moved = tasr_nemo_place_weights(m, budget);
#if defined(CONFIG_TASR_EXP_DCACHE_AUTOLOAD) && CONFIG_TASR_EXP_DCACHE_AUTOLOAD >= 0 && defined(CONFIG_IDF_TARGET_ESP32S3)  // S3 cache registers
    extern void dcache_autoload_psram(uintptr_t lo, uintptr_t hi);
    if (moved && g_ps_hi > g_ps_lo) dcache_autoload_psram(g_ps_lo, g_ps_hi);  // the weights copied to PSRAM
#endif
    ESP_LOGI(TAG, "NeMo conformer-ctc-small: %u weight bytes, %u moved to PSRAM | free PSRAM %u internal %u",
             (unsigned)tasr_nemo_weight_bytes(m), (unsigned)moved, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#ifdef CONFIG_TASR_MODE_MIC
    extern void run_mic_nemo(const tasr_nemo_t *m, struct tasr_decoder *dec);
    run_mic_nemo(m, dec);
#endif
    size_t asize = 0;
    const uint8_t *aud = map_partition(0x41, "audio", &asize);
    if (!aud || memcmp(aud, "AUD0", 4)) { ESP_LOGE(TAG, "no audio partition"); return; }
    uint32_t n_utt;
    memcpy(&n_utt, aud + 4, 4);
    const uint8_t *p = aud + 8;
    double tot_audio = 0, tot_cycles = 0, tot_fin = 0, tot_wall = 0, tot_fin_wall = 0;
    static char text[4096];
    tasr_nemo_stream_t *st = streaming ? tasr_nemo_stream_new(m, 0, 0, dec) : NULL;
    if (streaming) printf("STREAM model: chunk/left defaults, stream %s\n", st ? "ok" : "ALLOCATION FAILED");
    for (uint32_t u = 0; u < n_utt; u++) {
        uint32_t ns, tl;
        memcpy(&ns, p, 4); memcpy(&tl, p + 4, 4);
        const char *ref = (const char *)(p + 8);
        const int16_t *pcm = (const int16_t *)(p + 8 + ((tl + 3) & ~3u));
        p = (const uint8_t *)(pcm + ns + (ns & 1));
        double sec = ns / 16000.0;
        tot_audio += sec;
#ifdef CONFIG_TASR_BENCH_STREAM_ONLY
        printf("REF: %.*s\n", (int)tl, ref);  // streaming passes only; TOTAL is the streaming RTF
#else
        uint32_t c0 = esp_cpu_get_cycle_count();
        int64_t w0 = esp_timer_get_time();
        tasr_nemo_transcribe(m, pcm, ns, dec, text, sizeof(text), NULL, 0, NULL);
        const int64_t wall = esp_timer_get_time() - w0;  // 64-bit microseconds: the 32-bit cycle counter wraps every 17.9 s at 240 MHz
        uint32_t cyc = esp_cpu_get_cycle_count() - c0;
        tot_cycles += cyc;
        tot_wall += wall;
        printf("UTT %u | audio %.2fs | cycles %u | RTF %.3f | wall %lld us | RTF_wall %.3f\nREF: %.*s\nHYP: %s\n", (unsigned)u, sec,
               (unsigned)cyc, cyc / CPU_HZ / sec, (long long)wall, wall / 1e6 / sec, (int)tl, ref, text);
#endif
        if (st) {  // the same audio arriving in 20 ms blocks; what matters is the compute left after the last block
            tasr_nemo_stream_reset(st);
            uint32_t s0 = esp_cpu_get_cycle_count();
            int64_t sw0 = esp_timer_get_time();
            for (uint32_t o = 0; o < ns; o += 320) tasr_nemo_stream_feed(st, pcm + o, ns - o < 320 ? ns - o : 320);
            uint32_t s1 = esp_cpu_get_cycle_count();
            int64_t sw1 = esp_timer_get_time();
            tasr_nemo_stream_finish(st, text, sizeof(text));
            uint32_t s2 = esp_cpu_get_cycle_count();
            int64_t sw2 = esp_timer_get_time();
            tot_fin += s2 - s1;
            tot_fin_wall += sw2 - sw1;
#ifdef CONFIG_TASR_BENCH_STREAM_ONLY
            tot_cycles += (double)(sw2 - sw0) * (CPU_HZ / 1e6);  // wall time: feed + finish can exceed the 32-bit counter
            tot_wall += sw2 - sw0;
#endif
            printf("STREAM %u | feed cycles %u | finish cycles %u (%.0f ms @240MHz) | RTF %.3f | feed wall %lld us | finish wall %lld us | RTF_wall %.3f\nSHYP: %s\n",
                   (unsigned)u, (unsigned)(s1 - s0), (unsigned)(s2 - s1), (s2 - s1) / CPU_HZ * 1000.0, (s2 - s0) / CPU_HZ / sec,
                   (long long)(sw1 - sw0), (long long)(sw2 - sw1), (sw2 - sw0) / 1e6 / sec, text);
#ifdef CONFIG_TASR_BENCH_PACED
            tasr_nemo_stream_reset(st);
            const int64_t p0 = esp_timer_get_time();
            for (uint32_t o = 0; o < ns; o += 320) {
                const uint32_t nb = ns - o < 320 ? ns - o : 320;
                const int64_t due = p0 + (int64_t)(o + nb) * 1000000 / 16000;  // a block is complete when its last sample has been spoken
                for (int64_t now = esp_timer_get_time(); now < due; now = esp_timer_get_time())
                    if (due - now > 12000) vTaskDelay(1);
                tasr_nemo_stream_feed(st, pcm + o, nb);
            }
            const int64_t p1 = esp_timer_get_time();
            tasr_nemo_stream_finish(st, text, sizeof(text));
            const int64_t p2 = esp_timer_get_time();
            const int64_t eos = (int64_t)ns * 1000000 / 16000;  // end of speech, relative to p0
            printf("PACED %u | audio %.2fs | last block fed %lld ms after the end of speech | final text %lld ms after the end of speech\nPHYP: %s\n",
                   (unsigned)u, sec, (long long)((p1 - p0 - eos) / 1000), (long long)((p2 - p0 - eos) / 1000), text);
#endif
        }
    }
    if (st) printf("STREAM mean finish %.0f ms @240MHz, %.0f ms wall\n", tot_fin / n_utt / CPU_HZ * 1000.0, tot_fin_wall / n_utt / 1e3);
    printf("TOTAL audio %.1fs cycles %.0f RTF@240MHz %.3f wall RTF %.3f\n", tot_audio, tot_cycles, tot_cycles / CPU_HZ / tot_audio,
           tot_wall / 1e6 / tot_audio);
#if defined(CONFIG_IDF_TARGET_ESP32P4) && !defined(CONFIG_TASR_BENCH_STREAM_ONLY)
    {  // how long a full-context pass takes against the utterance's length: the clips joined end to end, the streaming object still allocated
        const uint32_t cap = 16000 * 20;
        int16_t *join = heap_caps_malloc(sizeof(int16_t) * cap, MALLOC_CAP_SPIRAM);
        uint32_t have = 0;
        p = aud + 8;
        for (uint32_t u = 0; u < n_utt && join; u++) {
            uint32_t ns, tl;
            memcpy(&ns, p, 4); memcpy(&tl, p + 4, 4);
            const int16_t *pcm = (const int16_t *)(p + 8 + ((tl + 3) & ~3u));
            p = (const uint8_t *)(pcm + ns + (ns & 1));
            const uint32_t nc = ns < cap - have ? ns : cap - have;
            memcpy(join + have, pcm, sizeof(int16_t) * nc);
            have += nc;
        }
        static const int lens[] = {1, 2, 3, 5, 8, 12, 16, 20};
        for (int i = 0; join && i < (int)(sizeof(lens) / sizeof(lens[0])); i++) {
            const uint32_t ns = (uint32_t)lens[i] * 16000;
            if (ns > have) break;
            const int64_t w0 = esp_timer_get_time();
            tasr_nemo_transcribe(m, join, ns, dec, text, sizeof(text), NULL, 0, NULL);
            const int64_t wall = esp_timer_get_time() - w0;
            printf("LEN %2d s | full-context wall %lld ms | RTF %.3f | min free PSRAM %u\n", lens[i], (long long)(wall / 1000), wall / 1e6 / lens[i],
                   (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
        }
        heap_caps_free(join);
    }
#endif
    printf("MEM min free PSRAM %u internal %u\n", (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
    for (int i = 0; tasr_nemo_profile_name(i); i++)
        printf("PROF %-10s %12llu cycles %5.1f%%\n", tasr_nemo_profile_name(i), (unsigned long long)tasr_nemo_profile_value(i),
               100.0 * tasr_nemo_profile_value(i) / tot_cycles);
    {
        extern uint64_t tasr_lm_prof_cycles, tasr_lm_prof_calls, tasr_lm_prof_rows;  // part of "(decoder)"
        printf("PROF (lm)      %12llu cycles in %llu GRU steps, %llu rows\n", (unsigned long long)tasr_lm_prof_cycles,
               (unsigned long long)tasr_lm_prof_calls, (unsigned long long)tasr_lm_prof_rows);
    }
    printf("DONE\n");
}

#if defined(CONFIG_TASR_EXP_DCACHE_AUTOLOAD) && CONFIG_TASR_EXP_DCACHE_AUTOLOAD >= 0 && defined(CONFIG_IDF_TARGET_ESP32S3)  // S3 cache registers
#include "soc/extmem_reg.h"
#include "soc/soc.h"
// Experiment: the data cache's hardware prefetch ("autoload", off after boot) over the model's flash mapping: on a
// miss the cache also fetches the next blocks, so weights read in order arrive while the kernel works. Only timing
// changes, never data. (Over the whole external-memory window it faults at once: it reaches unmapped pages.)
// second prefetch window: the PSRAM blocks allocated so far (weights moved there; all of it mapped heap)
void dcache_autoload_psram(uintptr_t lo, uintptr_t hi)
{
    REG_WRITE(EXTMEM_DCACHE_AUTOLOAD_SCT1_ADDR_REG, (uint32_t)lo);
    REG_WRITE(EXTMEM_DCACHE_AUTOLOAD_SCT1_SIZE_REG, (uint32_t)(hi - lo));
    REG_SET_BIT(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG, EXTMEM_DCACHE_AUTOLOAD_SCT1_ENA);
    ESP_LOGI(TAG, "prefetch also over PSRAM 0x%08x + %u", (unsigned)lo, (unsigned)(hi - lo));
}
static void dcache_autoload(const void *addr, size_t size)
{
    const uint32_t before = REG_READ(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG);
    REG_WRITE(EXTMEM_DCACHE_AUTOLOAD_SCT0_ADDR_REG, (uint32_t)addr);
    REG_WRITE(EXTMEM_DCACHE_AUTOLOAD_SCT0_SIZE_REG, (uint32_t)size);
    uint32_t v = before & ~(EXTMEM_DCACHE_AUTOLOAD_SIZE_M | EXTMEM_DCACHE_AUTOLOAD_RQST_M | EXTMEM_DCACHE_AUTOLOAD_ORDER_M);
    v |= ((uint32_t)CONFIG_TASR_EXP_DCACHE_AUTOLOAD << EXTMEM_DCACHE_AUTOLOAD_SIZE_S)  // blocks per autoload
         | (0u << EXTMEM_DCACHE_AUTOLOAD_RQST_S)                                      // trigger: cache miss
         | EXTMEM_DCACHE_AUTOLOAD_SCT0_ENA | EXTMEM_DCACHE_AUTOLOAD_ENA;
    REG_WRITE(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG, v);
    printf("DCACHE autoload: ctrl 0x%08x -> 0x%08x, window 0x%08x + 0x%x\n", (unsigned)before,
           (unsigned)REG_READ(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG), (unsigned)(uint32_t)addr, (unsigned)size);
}
#endif

void app_main(void)
{
#ifdef CONFIG_TASR_KERNEL_BENCH
    vTaskDelay(pdMS_TO_TICKS(3000));  // benchmark builds: on the native USB console, give the monitor time to attach
#endif
    ble_start();  // first: the radio needs internal RAM, and the engine takes what is left
    led_start();  // dim blue while the model loads
    calibrate();
    tasr_alloc = fw_alloc;
#ifdef CONFIG_TASR_P4_PIE
    tasr_p4_selftest();
#endif
    tasr_free = fw_free;
#ifndef TASR_SINGLE_CORE
    g_start = xSemaphoreCreateBinary();
    g_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker_task, "asr_w1", CONFIG_TASR_WORKER_STACK, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    tasr_parallel = par_for;
    ESP_LOGI(TAG, "dual-core enabled");
#endif
    size_t msize = 0, asize = 0;
#ifdef CONFIG_IDF_TARGET_ESP32P4
    const uint8_t *mflash = read_partition_to_psram(0x40, "model", &msize);
#else
    const uint8_t *mflash = map_partition(0x40, "model", &msize);
#endif
    const uint8_t *aud = map_partition(0x41, "audio", &asize);
    if (!mflash) { ESP_LOGE(TAG, "model partition missing"); return; }
#if defined(CONFIG_TASR_EXP_DCACHE_AUTOLOAD) && CONFIG_TASR_EXP_DCACHE_AUTOLOAD >= 0 && defined(CONFIG_IDF_TARGET_ESP32S3)
    dcache_autoload(mflash, msize);
#endif
#ifdef CONFIG_TASR_KERNEL_BENCH
    extern void run_kernel_bench(const uint8_t *flash, size_t flash_bytes);
    run_kernel_bench(mflash, msize);
#endif
    uint32_t blob_size;
    memcpy(&blob_size, mflash, 4);  // image = [u32 size][pad to 16][TASR blob]
    const uint8_t *blob = mflash + 16;
    ESP_LOGI(TAG, "model blob %u bytes; free PSRAM %u, internal %u", (unsigned)blob_size,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    // weights stay memory-mapped in flash; hot matrices are moved to PSRAM after the stream is allocated
    if (!memcmp(blob, "TNM1", 4)) { run_nemo(blob, blob_size); return; }
    tasr_model_t *m = tasr_model_load(blob, blob_size);
    if (!m) { ESP_LOGE(TAG, "model load failed"); return; }
    tasr_stream_t *s = tasr_stream_create(m, CONFIG_TASR_CHUNK, CONFIG_TASR_LEFT);
    if (!s) { ESP_LOGE(TAG, "stream create failed"); return; }
    {   // optional LM + beam search (partition "lm": [u32 size][pad16][TLM1 blob])
        size_t lsize = 0;
        const uint8_t *lmp = map_partition(0x42, "lm", &lsize);
        uint32_t lbytes = 0;
        if (lmp) memcpy(&lbytes, lmp, 4);
        if (lmp && lbytes > 0 && lbytes < lsize) {
            tasr_lm_t *lm = tasr_lm_load(lmp + 16, lbytes);
            if (lm) {
                size_t mv = tasr_lm_to_ram(lm);  // LM weights are re-read every emitting frame: keep them in PSRAM
                ESP_LOGI(TAG, "LM weights in PSRAM: %u bytes", (unsigned)mv);
                tasr_decoder_t *dec = tasr_decoder_create(lm, 257, CONFIG_TASR_BEAM, 6, CONFIG_TASR_LM_WEIGHT_X100 / 100.0f,
                                                          CONFIG_TASR_TOKEN_BONUS_X100 / 100.0f);
                tasr_stream_set_decoder(s, dec);
                ESP_LOGI(TAG, "LM beam search: beam %d, lm %u bytes", CONFIG_TASR_BEAM,
                         (unsigned)lbytes);
            }
        }
    }
#ifndef TASR_WEIGHTS_IN_FLASH
    {
        size_t free_ps = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        size_t budget = free_ps > 600000 ? free_ps - 600000 : 0;  // keep headroom for the logit sink / app
        int64_t t0 = esp_timer_get_time();
        size_t moved = tasr_model_place_weights(m, budget);
        ESP_LOGI(TAG, "moved %u of %u weight bytes to PSRAM in %lld ms", (unsigned)moved,
                 (unsigned)tasr_model_weight_bytes(m), (esp_timer_get_time() - t0) / 1000);
    }
#endif
#ifdef CONFIG_TASR_MODE_MIC
    extern void run_mic(tasr_stream_t *s);
    run_mic(s);
#endif
    const int max_frames = 400, V1 = 257;
    float *sink = heap_caps_malloc(sizeof(float) * max_frames * V1, MALLOC_CAP_SPIRAM);
    int n_frames = 0;

    ESP_LOGI(TAG, "weights %u bytes | after alloc: free PSRAM %u, internal %u", (unsigned)tasr_model_weight_bytes(m),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    if (memcmp(aud, "AUD0", 4)) { ESP_LOGE(TAG, "no audio"); return; }
    uint32_t n_utt;
    memcpy(&n_utt, aud + 4, 4);
    const uint8_t *p = aud + 8;
    double tot_audio = 0, tot_cycles = 0;
    uint32_t worst_block = 0;
    for (uint32_t u = 0; u < n_utt; u++) {
        uint32_t ns, tl;
        memcpy(&ns, p, 4); memcpy(&tl, p + 4, 4);
        const char *ref = (const char *)(p + 8);
        const int16_t *pcm = (const int16_t *)(p + 8 + ((tl + 3) & ~3u));
        p = (const uint8_t *)(pcm + ns + (ns & 1));
        tasr_stream_reset(s);
        if (sink) tasr_stream_set_logit_sink(s, sink, max_frames, &n_frames);
        uint64_t cyc = 0;
        for (uint32_t i = 0; i < ns; i += 320) {  // 20 ms blocks as from an I2S mic
            int n = ns - i < 320 ? ns - i : 320;
            uint32_t c0 = esp_cpu_get_cycle_count();
            tasr_stream_feed(s, pcm + i, n);
            uint32_t dc = esp_cpu_get_cycle_count() - c0;
            cyc += dc;
            if (dc > worst_block) worst_block = dc;
        }
        uint32_t c0 = esp_cpu_get_cycle_count();
        tasr_stream_finish(s);
        cyc += esp_cpu_get_cycle_count() - c0;
        double sec = ns / 16000.0;
        tot_audio += sec;
        tot_cycles += cyc;
        double cs1 = 0, cs2 = 0;
        for (int i = 0; i < n_frames * V1; i++) { cs1 += sink[i]; cs2 += (double)sink[i] * sink[i]; }
        printf("UTT %u | audio %.2fs | cycles %llu | RTF %.3f | frames %d logit-sum %.4f logit-sq %.4f\nREF: %.*s\nHYP: %s\n",
               (unsigned)u, sec, (unsigned long long)cyc, cyc / CPU_HZ / sec, n_frames, cs1, cs2, (int)tl, ref,
               tasr_stream_text(s));
    }
    printf("TOTAL audio %.1fs cycles %.0f RTF@240MHz %.3f worst-block %.1f ms\n", tot_audio, tot_cycles,
           tot_cycles / CPU_HZ / tot_audio, worst_block / CPU_HZ * 1000);
    double ptot = 0;
    for (int i = 0; tasr_profile_name(i); i++) ptot += tasr_profile_value(i);
    for (int i = 0; tasr_profile_name(i); i++)
        printf("PROF %-10s %12llu cycles %5.1f%%\n", tasr_profile_name(i), (unsigned long long)tasr_profile_value(i),
               100.0 * tasr_profile_value(i) / (ptot > 0 ? ptot : 1));
    printf("DONE\n");
}
