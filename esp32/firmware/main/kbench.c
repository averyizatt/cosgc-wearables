// Kernel memory benchmark (CONFIG_TASR_KERNEL_BENCH): times the engine's two main int8 GEMM shapes (64 rows, as in
// the encoder) with the weights in internal RAM, PSRAM and flash and the activations in internal RAM or PSRAM.
// Internal weights cost no memory stalls, so the gap to PSRAM / flash is the time spent waiting for weights; if the
// internal case is still ~2 cycles per instruction, the stalls are in the kernels themselves.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#include "kernels.h"

#ifdef CONFIG_TASR_KERNEL_BENCH
static uint32_t fnv(const float *x, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t k = 0; k < n; k++) { uint32_t b; memcpy(&b, &x[k], 4); h = (h ^ b) * 16777619u; }
    return h;
}

void run_kernel_bench(const uint8_t *flash, size_t flash_bytes)
{
    static const struct { int n, k; const char *name; } shp[] = {{176, 704, "K=704 x 176 (ff*_2)"}, {704, 176, "K=176 x 704 (ff*_1)"}};
    const int T = 64;
    printf("KBENCH int8 GEMM, %d rows, min of 3 runs at 240 MHz\n", T);
    for (size_t si = 0; si < sizeof(shp) / sizeof(shp[0]); si++) {
        const int n = shp[si].n, k = shp[si].k;
        const size_t wb = (size_t)n * k, macs = wb * T;
        const int8_t *wf = (const int8_t *)(((uintptr_t)flash + 4096 + 15) & ~(uintptr_t)15);  // model bytes: any int8 data
        if (flash_bytes < wb + 8192) { printf("KBENCH model partition too small\n"); return; }
        int8_t *wi = heap_caps_aligned_alloc(16, wb, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        int8_t *wp = heap_caps_aligned_alloc(16, wb, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        int8_t *xi = heap_caps_aligned_alloc(16, (size_t)T * k, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        int8_t *xp = heap_caps_aligned_alloc(16, (size_t)T * k, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        float *xs = heap_caps_malloc(sizeof(float) * T, MALLOC_CAP_INTERNAL);
        float *sw = heap_caps_malloc(sizeof(float) * n, MALLOC_CAP_INTERNAL);
        float *bb = heap_caps_malloc(sizeof(float) * n, MALLOC_CAP_INTERNAL);
        int32_t *acc = heap_caps_aligned_alloc(16, sizeof(int32_t) * 64 * 16, MALLOC_CAP_INTERNAL);
        float *y = heap_caps_aligned_alloc(16, sizeof(float) * T * n, MALLOC_CAP_SPIRAM);  // like the engine's h2 / hb
        if (!wp || !xi || !xp || !xs || !sw || !bb || !acc || !y) { printf("KBENCH allocation failed\n"); return; }
        if (wi) memcpy(wi, wf, wb);
        memcpy(wp, wf, wb);
        uint32_t r = 12345;
        for (int i = 0; i < T * k; i++) { r = r * 1103515245u + 12345u; xi[i] = (int8_t)(r >> 24); }
        memcpy(xp, xi, (size_t)T * k);
        for (int t = 0; t < T; t++) xs[t] = 1e-3f;
        for (int j = 0; j < n; j++) { sw[j] = 1e-2f; bb[j] = 0.f; }
        const struct { const int8_t *w, *x; const char *wn, *xn; } cs[] = {
            {wi, xi, "internal", "internal"}, {wp, xi, "PSRAM", "internal"}, {wf, xi, "flash", "internal"}, {wp, xp, "PSRAM", "PSRAM"}};
        // rows of <= 256 bytes normally use the x-stationary kernel; time the row kernel on them too (same results)
        for (int kv = 0; kv < (k <= 256 ? 2 : 1); kv++)
        for (size_t c = 0; c < sizeof(cs) / sizeof(cs[0]); c++) {
            tasr_xr_max_kp = kv ? 0 : 256;
            const char *kn = k > 256 || kv ? "row" : "x-stat";
            if (!cs[c].w) { printf("KBENCH %s | weights %-8s: no room in internal RAM\n", shp[si].name, cs[c].wn); continue; }
            tasr_qlin_t L = {.n = n, .k = k, .kp = k, .bits = 8, .blocked = 0, .w = cs[c].w, .s = sw, .b = bb};
            uint32_t best = UINT32_MAX;
            for (int rep = 0; rep < 3; rep++) {
                const uint32_t c0 = esp_cpu_get_cycle_count();
                tasr_qlin_range(&L, cs[c].x, xs, T, k, y, n, NULL, acc, 0, n);
                const uint32_t dc = esp_cpu_get_cycle_count() - c0;
                if (dc < best) best = dc;
            }
            printf("KBENCH %s | %-6s kernel | weights %-8s act %-8s | %8u cycles | %.3f cycles/MAC | weights %.1f MB/s | y %08x\n",
                   shp[si].name, kn, cs[c].wn, cs[c].xn, (unsigned)best, (double)best / macs, wb / (best / 240e6) / 1e6,
                   (unsigned)fnv(y, (size_t)T * n));
        }
        tasr_xr_max_kp = 256;
        heap_caps_free(wi); heap_caps_free(wp); heap_caps_free(xi); heap_caps_free(xp); heap_caps_free(xs);
        heap_caps_free(sw); heap_caps_free(bb); heap_caps_free(acc); heap_caps_free(y);
    }
    printf("KBENCH same y hash on every line of a shape = same results; cycles/MAC near the internal line = not memory-bound\n");
}
#else
void run_kernel_bench(const uint8_t *flash, size_t flash_bytes) { (void)flash; (void)flash_bytes; }
#endif
