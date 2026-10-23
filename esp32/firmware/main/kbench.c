// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
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

// PIE pipeline timings: cycles per instruction of short unrolled sequences (operands in registers / internal RAM)
static uint8_t pbuf[512] __attribute__((aligned(16)));
#define PIE_N 1000
#define PIE_T(name, ninstr, body)                                                                                   \
    do {                                                                                                            \
        int n = PIE_N, r;                                                                                           \
        uint8_t *p = pbuf;                                                                                          \
        const uint32_t c0 = esp_cpu_get_cycle_count();                                                              \
        __asm__ volatile("loopnez %[n], 1f\n" body "1:\n" : [n] "+r"(n), [p] "+r"(p), [r] "=&r"(r) : : "memory"); \
        const uint32_t dc = esp_cpu_get_cycle_count() - c0;                                                         \
        (void)r;                                                                                                    \
        printf("PIE %-46s %.2f cycles/instr\n", name, (double)dc / (PIE_N * (ninstr)));                             \
    } while (0)
static void pie_bench(void)
{
    __asm__ volatile("ee.vld.128.ip q0, %0, 0\n ee.vld.128.ip q1, %0, 0\n ee.vld.128.ip q2, %0, 0\n ee.vld.128.ip q3, %0, 0\n"
                     ::"r"(pbuf) : "memory");
    PIE_T("vmulas.accx chain", 8, ".rept 8\n ee.vmulas.s8.accx q0, q1\n .endr\n");
    PIE_T("vmulas.accx.ld (load unused) chain", 8, ".rept 8\n ee.vmulas.s8.accx.ld.ip q2, %[p], 0, q0, q1\n .endr\n");
    PIE_T("vld + vmulas.accx (independent)", 8, ".rept 4\n ee.vld.128.ip q2, %[p], 0\n ee.vmulas.s8.accx q0, q1\n .endr\n");
    PIE_T("vld q0 + vmulas.accx q0 (load-use)", 8, ".rept 4\n ee.vld.128.ip q0, %[p], 0\n ee.vmulas.s8.accx q0, q1\n .endr\n");
    PIE_T("vmulas.accx + rur.accx_0", 8, ".rept 4\n ee.vmulas.s8.accx q0, q1\n rur.accx_0 %[r]\n .endr\n");
    PIE_T("zero.accx + vmulas.accx", 8, ".rept 4\n ee.zero.accx\n ee.vmulas.s8.accx q0, q1\n .endr\n");
    PIE_T("vmulas.qacc chain", 8, ".rept 8\n ee.vmulas.s8.qacc q0, q1\n .endr\n");
    PIE_T("vmulas.accx / vmulas.qacc alternating", 8, ".rept 4\n ee.vmulas.s8.accx q0, q1\n ee.vmulas.s8.qacc q2, q3\n .endr\n");
    PIE_T("vsmulas.qacc chain", 8, ".rept 8\n ee.vsmulas.s8.qacc q0, q1, 0\n .endr\n");
    PIE_T("row kernel body (vld, vmulas.ld x2)", 8,
          ".rept 2\n ee.vld.128.ip q2, %[p], 0\n ee.vmulas.s8.accx.ld.ip q3, %[p], 0, q0, q1\n"
          " ee.vld.128.ip q0, %[p], 0\n ee.vmulas.s8.accx.ld.ip q1, %[p], 0, q2, q3\n .endr\n");
    PIE_T("l32i + dependent add", 8, ".rept 4\n l32i %[r], %[p], 0\n add %[r], %[r], %[r]\n .endr\n");
    PIE_T("nop (loop overhead reference)", 8, ".rept 8\n nop\n .endr\n");
    PIE_T("blk16 body: vld + 16 vsmulas.ld.incp (q0/q1)", 18,
          " ee.vld.128.ip q7, %[p], 0\n ee.vld.128.ip q0, %[p], 16\n"
          " .rept 8\n ee.vsmulas.s8.qacc.ld.incp q1, %[p], q0, q7, 0\n ee.vsmulas.s8.qacc.ld.incp q0, %[p], q1, q7, 1\n .endr\n"
          " addi %[p], %[p], -144\n");
    PIE_T("same with q0/q1/q2 rotation (load 2 ahead)", 18,
          " ee.vld.128.ip q7, %[p], 0\n ee.vld.128.ip q0, %[p], 16\n ee.vld.128.ip q1, %[p], 16\n"
          " .rept 5\n ee.vsmulas.s8.qacc.ld.incp q2, %[p], q0, q7, 0\n ee.vsmulas.s8.qacc.ld.incp q0, %[p], q1, q7, 1\n"
          " ee.vsmulas.s8.qacc.ld.incp q1, %[p], q2, q7, 2\n .endr\n addi %[p], %[p], -272\n");
    PIE_T("st.qacc_l/h stores (x4) after vsmulas", 5,
          " ee.vsmulas.s8.qacc q0, q7, 0\n ee.st.qacc_l.l.128.ip %[p], 16\n ee.st.qacc_l.h.32.ip %[p], 16\n"
          " ee.st.qacc_h.l.128.ip %[p], 16\n ee.st.qacc_h.h.32.ip %[p], -48\n");
    {   // the whole broadcast kernel per weight row: kp = 16 (1 step) and 176 (11 steps), 16 weight rows
        static int8_t tile[176 * 16 + 64] __attribute__((aligned(16))), wr[16 * 176] __attribute__((aligned(16)));
        static int32_t out[16 * 16];
        uint32_t c[2];
        for (int v = 0; v < 2; v++) {
            const int kp = v ? 176 : 16;
            const uint32_t c0 = esp_cpu_get_cycle_count();
            for (int rep = 0; rep < 100; rep++) tasr_gemm_blk16(tile, kp, wr, kp, 16, out);
            c[v] = esp_cpu_get_cycle_count() - c0;
        }
        printf("PIE blk16 per weight row: kp 16: %.1f cycles, kp 176: %.1f cycles (%.1f per 16-k step, %.1f fixed)\n",
               c[0] / 1600.0, c[1] / 1600.0, (c[1] - c[0]) / 1600.0 / 10, c[0] / 1600.0 - (c[1] - c[0]) / 1600.0 / 10);
    }
}

// one GEMM shape with model-like int8 data (bell shaped, mostly |v| < 64): broadcast kernel on / off, weights in
// internal RAM (when they fit) and PSRAM, outputs in PSRAM
static void bench_ml(int n, int k, int T, const char *name)
{
    const size_t wb = (size_t)n * k;
    const double macs = (double)wb * T;
    int8_t *w[2] = {wb <= 130000 ? heap_caps_aligned_alloc(16, wb, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : NULL,
                    heap_caps_aligned_alloc(16, wb, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)};
    int8_t *x = heap_caps_aligned_alloc(16, (size_t)T * k, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *xs = heap_caps_malloc(sizeof(float) * T, MALLOC_CAP_INTERNAL), *sw = heap_caps_malloc(sizeof(float) * n, MALLOC_CAP_SPIRAM);
    float *bb = heap_caps_malloc(sizeof(float) * n, MALLOC_CAP_SPIRAM);
    int32_t *acc = heap_caps_aligned_alloc(16, sizeof(int32_t) * 64 * 16, MALLOC_CAP_INTERNAL);
    float *y = heap_caps_aligned_alloc(16, sizeof(float) * T * n, MALLOC_CAP_SPIRAM);
    if (!w[1] || !x || !xs || !sw || !bb || !acc || !y) { printf("KBENCH %s: allocation failed\n", name); goto out; }
    uint32_t r = 777;
    for (size_t i = 0; i < wb; i++) {
        int s = 0;
        for (int q = 0; q < 4; q++) { r = r * 1103515245u + 12345u; s += (int)(r >> 24) - 128; }
        w[1][i] = (int8_t)(s / 8);
    }
    if (w[0]) memcpy(w[0], w[1], wb);
    for (int i = 0; i < T * k; i++) {
        int s = 0;
        for (int q = 0; q < 4; q++) { r = r * 1103515245u + 12345u; s += (int)(r >> 24) - 128; }
        x[i] = (int8_t)(s / 8);
    }
    for (int t = 0; t < T; t++) xs[t] = 1e-3f;
    for (int j = 0; j < n; j++) { sw[j] = 1e-2f; bb[j] = 0.f; }
    for (int pl = 0; pl < 2; pl++) {
        if (!w[pl]) continue;
        tasr_qlin_t L = {.n = n, .k = k, .kp = k, .bits = 8, .blocked = 0, .w = w[pl], .s = sw, .b = bb};
        L.wn2 = tasr_qlin_norms(&L);
        for (int e = 0; e < 2; e++) {
            tasr_xt_enable = e;
            uint32_t best = UINT32_MAX;
            for (int rep = 0; rep < 3; rep++) {
                const uint32_t c0 = esp_cpu_get_cycle_count();
                tasr_qlin_range(&L, x, xs, T, k, y, n, NULL, acc, 0, n);
                const uint32_t dc = esp_cpu_get_cycle_count() - c0;
                if (dc < best) best = dc;
            }
            printf("KBENCH %-22s %2d rows | %-9s | weights %-8s | %9u cycles | %.3f cycles/MAC | y %08x\n", name, T,
                   e ? "broadcast" : "old", pl ? "PSRAM" : "internal", (unsigned)best, best / macs, (unsigned)fnv(y, (size_t)T * n));
        }
        tasr_xt_enable = 1;
        heap_caps_free((void *)L.wn2);
    }
out:
    heap_caps_free(w[0]); heap_caps_free(w[1]); heap_caps_free(x); heap_caps_free(xs); heap_caps_free(sw);
    heap_caps_free(bb); heap_caps_free(acc); heap_caps_free(y);
}

static void bench_blk(void)  // tasr_gemm_blk16 alone on a 256-byte chunk: weight row stride and placement
{
    static int8_t tile[256 * 16 + 64] __attribute__((aligned(16)));
    static int32_t out[16 * 16];
    int8_t *wi = heap_caps_aligned_alloc(16, 16 * 704, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *wp = heap_caps_aligned_alloc(16, 16 * 704 * 64, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!wi || !wp) return;
    memset(wi, 1, 16 * 704);
    memset(wp, 1, 16 * 704 * 64);
    const struct { int8_t *w; int ld; const char *n; } cs[] = {{wi, 256, "internal, stride 256"}, {wi, 704, "internal, stride 704"},
                                                              {wp, 256, "PSRAM, stride 256"}, {wp, 704, "PSRAM, stride 704"}};
    for (size_t c = 0; c < 4; c++) {
        uint32_t best = UINT32_MAX;
        for (int rep = 0; rep < 3; rep++) {
            const uint32_t c0 = esp_cpu_get_cycle_count();
            for (int i = 0; i < 16; i++) tasr_gemm_blk16(tile, 256, cs[c].w + (cs[c].w == wp ? (size_t)i * 16 * 704 : 0), cs[c].ld, 16, out);
            const uint32_t dc = esp_cpu_get_cycle_count() - c0;
            if (dc < best) best = dc;
        }
        printf("PIE blk16 kn=256, 16 weight rows (%s): %.1f cycles per weight row\n", cs[c].n, best / 256.0);
    }
    heap_caps_free(wi);
    heap_caps_free(wp);
}

void run_kernel_bench(const uint8_t *flash, size_t flash_bytes)
{
    pie_bench();
    bench_blk();
    bench_ml(704, 176, 64, "K=176 x 704 (ff*_1)");
    bench_ml(176, 176, 64, "K=176 x 176 (out)");
    bench_ml(176, 704, 64, "K=704 x 176 (ff*_2)");
    bench_ml(176, 704, 32, "K=704 x 176 (ff*_2)");
    bench_ml(176, 1584, 60, "K=1584 x 176 (conv2)");
    bench_ml(176, 3520, 32, "K=3520 x 176 (sub)");
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
        const int xr_saved = tasr_xr_max_kp;
        for (int kv = 0; kv < (k <= 256 ? 2 : 1); kv++)
        for (size_t c = 0; c < sizeof(cs) / sizeof(cs[0]); c++) {
            tasr_xr_max_kp = kv ? 0 : (k <= 256 ? 256 : 0);
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
        tasr_xr_max_kp = xr_saved;
        heap_caps_free(wi); heap_caps_free(wp); heap_caps_free(xi); heap_caps_free(xp); heap_caps_free(xs);
        heap_caps_free(sw); heap_caps_free(bb); heap_caps_free(acc); heap_caps_free(y);
    }
    {   // broadcast kernel (tasr_xt_enable) against the x-stationary one on K=176, with model-like int8 values (bell
        // shaped, |w|, |x| mostly < 64) so that the overflow bound passes as it does on real activations
        const int n = 704, k = 176;
        const size_t wb = (size_t)n * k, macs = wb * T;
        int8_t *w[2] = {heap_caps_aligned_alloc(16, wb, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                        heap_caps_aligned_alloc(16, wb, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)};
        int8_t *x = heap_caps_aligned_alloc(16, (size_t)T * k, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        float *xs = heap_caps_malloc(sizeof(float) * T, MALLOC_CAP_INTERNAL), *sw = heap_caps_malloc(sizeof(float) * n, MALLOC_CAP_INTERNAL);
        float *bb = heap_caps_malloc(sizeof(float) * n, MALLOC_CAP_INTERNAL);
        int32_t *acc = heap_caps_aligned_alloc(16, sizeof(int32_t) * 64 * 16, MALLOC_CAP_INTERNAL);
        float *y = heap_caps_aligned_alloc(16, sizeof(float) * T * n, MALLOC_CAP_SPIRAM);
        if (!w[1] || !x || !xs || !sw || !bb || !acc || !y) { printf("KBENCH allocation failed\n"); return; }
        uint32_t r = 777;
        for (size_t i = 0; i < wb; i++) {
            int s = 0;
            for (int q = 0; q < 4; q++) { r = r * 1103515245u + 12345u; s += (int)(r >> 24) - 128; }
            w[1][i] = (int8_t)(s / 8);
        }
        if (w[0]) memcpy(w[0], w[1], wb);
        for (int i = 0; i < T * k; i++) {
            int s = 0;
            for (int q = 0; q < 4; q++) { r = r * 1103515245u + 12345u; s += (int)(r >> 24) - 128; }
            x[i] = (int8_t)(s / 8);
        }
        for (int t = 0; t < T; t++) xs[t] = 1e-3f;
        for (int j = 0; j < n; j++) { sw[j] = 1e-2f; bb[j] = 0.f; }
        static const char *wn[2] = {"internal", "PSRAM"};
        for (int pl = 0; pl < 2; pl++) {
            if (!w[pl]) continue;
            tasr_qlin_t L = {.n = n, .k = k, .kp = k, .bits = 8, .blocked = 0, .w = w[pl], .s = sw, .b = bb};
            L.wn2 = tasr_qlin_norms(&L);
            for (int e = 0; e < 2; e++) {
                tasr_xt_enable = e;
                uint32_t best = UINT32_MAX;
                for (int rep = 0; rep < 3; rep++) {
                    const uint32_t c0 = esp_cpu_get_cycle_count();
                    tasr_qlin_range(&L, x, xs, T, k, y, n, NULL, acc, 0, n);
                    const uint32_t dc = esp_cpu_get_cycle_count() - c0;
                    if (dc < best) best = dc;
                }
                printf("KBENCH K=176 x 704 model-like | %-9s kernel | weights %-8s | %8u cycles | %.3f cycles/MAC | y %08x\n",
                       e ? "broadcast" : "x-stat", wn[pl], (unsigned)best, (double)best / macs, (unsigned)fnv(y, (size_t)T * n));
            }
            tasr_xt_enable = 1;
            heap_caps_free((void *)L.wn2);
        }
        // where the outputs go: 176 outputs (45 KB of y) into PSRAM (as the engine's h2) or internal RAM
        float *yi = heap_caps_aligned_alloc(16, sizeof(float) * T * 176, MALLOC_CAP_INTERNAL);
        if (yi && w[0]) {
            tasr_qlin_t L = {.n = 176, .k = k, .kp = k, .bits = 8, .blocked = 0, .w = w[0], .s = sw, .b = bb};
            L.wn2 = tasr_qlin_norms(&L);
            for (int yl = 0; yl < 2; yl++)
                for (int e = 0; e < 2; e++) {
                    tasr_xt_enable = e;
                    uint32_t best = UINT32_MAX;
                    for (int rep = 0; rep < 3; rep++) {
                        const uint32_t c0 = esp_cpu_get_cycle_count();
                        tasr_qlin_range(&L, x, xs, T, k, yl ? yi : y, 176, NULL, acc, 0, 176);
                        const uint32_t dc = esp_cpu_get_cycle_count() - c0;
                        if (dc < best) best = dc;
                    }
                    printf("KBENCH K=176 x 176 model-like | %-9s kernel | weights internal y %-8s | %8u cycles | %.3f cycles/MAC\n",
                           e ? "broadcast" : "x-stat", yl ? "internal" : "PSRAM", (unsigned)best, (double)best / ((double)T * 176 * k));
                }
            tasr_xt_enable = 1;
            heap_caps_free((void *)L.wn2);
        }
        heap_caps_free(yi);
        heap_caps_free(w[0]); heap_caps_free(w[1]); heap_caps_free(x); heap_caps_free(xs); heap_caps_free(sw);
        heap_caps_free(bb); heap_caps_free(acc); heap_caps_free(y);
    }
    printf("KBENCH same y hash on every line of a shape = same results; cycles/MAC near the internal line = not memory-bound\n");
}
#else
void run_kernel_bench(const uint8_t *flash, size_t flash_bytes) { (void)flash; (void)flash_bytes; }
#endif
