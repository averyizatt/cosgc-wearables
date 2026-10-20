#include "kernels.h"
#include <math.h>
#include <string.h>
#include "tinyasr.h"
#if defined(ESP_PLATFORM)
#include "esp_cpu.h"
#endif
#if defined(CONFIG_TASR_EXP_DOT_IRAM)
#include "esp_attr.h"
#define TASR_DOT_ATTR IRAM_ATTR
#else
#define TASR_DOT_ATTR
#endif

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(TASR_NO_SIMD)
#define TASR_PIE 1
#endif

// round-to-nearest-even via the 1.5*2^23 trick (matches torch.round), valid for |x| < 2^22
static inline int rne(float x)
{
    float y = x + 12582912.0f;  // not folded without -ffast-math
    return (int)(y - 12582912.0f);
}

#ifdef TASR_PIE
// ESP32-S3 PIE: 16 int8 MACs per instruction into the 40-bit ACCX accumulator.
// The asm uses the zero-overhead loop registers; the enclosing C loop must not become a hardware loop,
// hence no-branch-count-reg (disables GCC's doloop / zero-overhead loop generation for this function).
#if defined(CONFIG_TASR_EXP_DOT_DUP)
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8_core0(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8_core1(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}

#if defined(CONFIG_TASR_EXP_DOT_DUP)
// experiment: each core runs its own copy of the kernel (different instruction addresses)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    if (esp_cpu_get_core_id()) tasr_dot_rows_s8_core1(w, x, ldq, T, kp, out);
    else tasr_dot_rows_s8_core0(w, x, ldq, T, kp, out);
}
#endif
#else
TASR_DOT_ATTR __attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}
#endif
#else
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t acc = 0;
        for (int i = 0; i < kp; i++) acc += (int32_t)a[i] * (int32_t)w[i];
        out[t] = acc;
    }
}
#endif

int32_t tasr_dot_s8(const int8_t *a, const int8_t *b, int kp)
{
    int32_t r;
    tasr_dot_rows_s8(b, a, kp, 1, kp, &r);
    return r;
}

#ifdef TASR_PIE
static const uint8_t k_mask0f[16] __attribute__((aligned(16))) = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
static const int8_t k_eight[16] __attribute__((aligned(16))) = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
// 32 weights per iteration: lo nibbles -> w[j], hi nibbles -> w[j+16]; offset-binary (+8) removed with a saturating sub
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    int n = kp >> 5;
    const uint8_t *mk = k_mask0f;
    const int8_t *e8 = k_eight;
    __asm__ volatile(
        "ee.vld.128.ip q6, %3, 0\n"
        "ee.vld.128.ip q7, %4, 0\n"
        "ssai 4\n"
        "loopnez %2, 1f\n"
        "  ee.vld.128.ip q0, %0, 16\n"
        "  ee.andq q1, q0, q6\n"
        "  ee.vsr.32 q2, q0\n"
        "  ee.andq q2, q2, q6\n"
        "  ee.vsubs.s8 q1, q1, q7\n"
        "  ee.vsubs.s8 q2, q2, q7\n"
        "  ee.vst.128.ip q1, %1, 16\n"
        "  ee.vst.128.ip q2, %1, 16\n"
        "1:\n"
        : "+r"(src), "+r"(dst), "+r"(n)
        : "r"(mk), "r"(e8)
        : "memory");
}
#else
void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    for (int blk = 0; blk < kp; blk += 32) {
        for (int j = 0; j < 16; j++) {
            uint8_t v = src[j];
            dst[blk + j] = (int8_t)((v & 15) - 8);
            dst[blk + j + 16] = (int8_t)((v >> 4) - 8);
        }
        src += 16;
    }
}
#endif

// max |r[k]|: four running maxima (independent compare chains for the in-order FPU); the max of a set does not depend
// on the grouping, so the result is the same as one sequential pass
static inline float row_absmax(const float *r, int K)
{
    float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const float a = fabsf(r[k]), b = fabsf(r[k + 1]), c = fabsf(r[k + 2]), d = fabsf(r[k + 3]);
        m0 = a > m0 ? a : m0; m1 = b > m1 ? b : m1; m2 = c > m2 ? c : m2; m3 = d > m3 ? d : m3;
    }
    for (; k < K; k++) { const float a = fabsf(r[k]); m0 = a > m0 ? a : m0; }
    m0 = m1 > m0 ? m1 : m0;
    m2 = m3 > m2 ? m3 : m2;
    return m2 > m0 ? m2 : m0;
}
static inline void quant_row(const float *r, int K, int8_t *q, int kp, float m, float *xs)
{
    if (m < 1e-30f) m = 1e-30f;
    const float inv = 127.0f / m;
    int k = 0;
    for (; k + 4 <= K; k += 4) {  // |r*inv| <= 127 by construction; four independent conversions per pass
        const int a = rne(r[k] * inv), b = rne(r[k + 1] * inv), c = rne(r[k + 2] * inv), d = rne(r[k + 3] * inv);
        q[k] = (int8_t)a; q[k + 1] = (int8_t)b; q[k + 2] = (int8_t)c; q[k + 3] = (int8_t)d;
    }
    for (; k < K; k++) q[k] = (int8_t)rne(r[k] * inv);
    for (k = K; k < kp; k++) q[k] = 0;
    *xs = m / 127.0f;
}
void tasr_quant_rows(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs)
{
    for (int t = 0; t < T; t++) {
        const float *r = x + (size_t)t * ldx;
        quant_row(r, K, xq + (size_t)t * ldq, kp, row_absmax(r, K), xs + t);
    }
}
void tasr_quant_rows_mx(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs, const float *mx)
{
    for (int t = 0; t < T; t++) quant_row(x + (size_t)t * ldx, K, xq + (size_t)t * ldq, kp, mx[t], xs + t);
}

int tasr_xr_max_kp = 256;
int tasr_xt_enable = 1;

// ---- broadcast int8 kernel: tasr_gemm_blk16 with the roles swapped. The 16 QACC lanes hold 16 activation rows
// (transposed tile xT[k][16]), and each weight byte is broadcast: one instruction per 16 MACs, against ~2 for the
// the x-stationary kernel, for int8 rows of kp <= XT_KC. The lanes are 20 bits wide, so a 16-row x 16-output block is
// only run this way when Cauchy-Schwarz proves that no partial sum can leave [-2^19, 2^19):
// max |x_t|^2 * max |w_j|^2 < (2^19 - 1)^2. Then every sum is exact and identical to the x-stationary kernel's; blocks
// that fail the test (1-11 %, most in nemo8) use that kernel. Longer rows stay on the row kernel: cut into chunks they
// were 2x slower on the board, because weight rows read 256 bytes at a time at the row stride defeat the PSRAM / flash
// streaming that the row kernel gets from reading each row end to end (a chunk-major weight layout would fix that).
#define XT_LIM (524287ull * 524287ull)
#define XT_KC 256
const uint16_t *tasr_qlin_norms(const tasr_qlin_t *L)
{
    if (L->bits != 8 || L->blocked || L->kp > XT_KC) return NULL;
    uint16_t *q = (uint16_t *)tasr_alloc(sizeof(uint16_t) * L->n, 0);
    if (!q) return NULL;
    for (int j = 0; j < L->n; j++) {
        const int8_t *w = L->w + (size_t)j * L->kp;
        const uint32_t s = (uint32_t)tasr_dot_s8(w, w, L->kp);  // <= 256 * 128^2: no overflow
        q[j] = (uint16_t)((s + 127) / 128);  // rounded up (the bound stays safe); <= 32768
    }
    return q;
}
typedef uint32_t __attribute__((may_alias, aligned(4))) tasr_u32a;  // aligned word access to int8 rows (l32i / s32i)
static inline uint32_t ld32(const int8_t *p) { return *(const tasr_u32a *)p; }
static inline void st32(int8_t *p, uint32_t v) { *(tasr_u32a *)p = v; }
static const int8_t k_zero_row[XT_KC] __attribute__((aligned(16)));
// columns [0, kn) of x rows [0, tn) -> 16-row groups xt[g * kn + k * 16 + r] = x[g + r][k] (rows past tn zero), as
// 4 x 4 byte blocks with 32-bit loads and stores; xn[t] = |x_t|^2 over those columns (PIE dot product)
static void build_xt(const int8_t *x, int ldq, int tn, int kn, int8_t *xt, int32_t *xn)
{
    for (int g = 0; g < tn; g += 16) {
        int8_t *o = xt + (size_t)g * kn;
        for (int r0 = 0; r0 < 16; r0 += 4) {
            const int8_t *p[4];
            for (int i = 0; i < 4; i++) {
                const int t = g + r0 + i;
                p[i] = t < tn ? x + (size_t)t * ldq : k_zero_row;
                if (t < tn) xn[t] = tasr_dot_s8(p[i], p[i], kn);
            }
            for (int k = 0; k < kn; k += 4) {  // a = a0 a1 a2 a3 (bytes, low first) ... -> k: a0 b0 c0 d0, k+1: a1 b1 c1 d1, ...
                const uint32_t a = ld32(p[0] + k), b = ld32(p[1] + k), c = ld32(p[2] + k), d = ld32(p[3] + k);
                const uint32_t ab02 = (a & 0x00ff00ffu) | ((b << 8) & 0xff00ff00u), ab13 = ((a >> 8) & 0x00ff00ffu) | (b & 0xff00ff00u);
                const uint32_t cd02 = (c & 0x00ff00ffu) | ((d << 8) & 0xff00ff00u), cd13 = ((c >> 8) & 0x00ff00ffu) | (d & 0xff00ff00u);
                int8_t *q = o + (size_t)k * 16 + r0;
                st32(q, (ab02 & 0xffffu) | (cd02 << 16));
                st32(q + 16, (ab13 & 0xffffu) | (cd13 << 16));
                st32(q + 32, (ab02 >> 16) | (cd02 & 0xffff0000u));
                st32(q + 48, (ab13 >> 16) | (cd13 & 0xffff0000u));
            }
        }
    }
}
// transposed-activation scratch (64 rows x kp <= 256), one per core / thread, allocated on first use (internal RAM)
#define XT_BYTES (64 * 256 + 64)  // + tasr_gemm_blk16 read-ahead
#if defined(ESP_PLATFORM)
static int8_t *s_xt[2];
static int8_t *xt_scratch(void)
{
    const int c = esp_cpu_get_core_id();
    if (!s_xt[c]) s_xt[c] = (int8_t *)tasr_alloc(XT_BYTES, 1);
    return s_xt[c];
}
#else
static _Thread_local int8_t *s_xt;
static int8_t *xt_scratch(void)
{
    if (!s_xt) s_xt = (int8_t *)tasr_alloc(XT_BYTES, 1);
    return s_xt;
}
#endif

// SiLU sigmoid, the same table lookup as the engines' nsig()
static inline float ep_sig(float x, const float *tab, const float *slope)
{
    const float u = (x + 12.0f) * 16.0f;
    if (u <= 0.f) return 0.f;
    if (u >= 384.f) return 1.f;
    const int i = (int)u;
    return tab[i] + (u - (float)i) * slope[i];
}
// output stage of one tile: y[t*ldy + j] = acc[t*lda + j*ldc] * xs[t] * sw[j] + bb[j] for t < tn, j < nb; then
// optionally SiLU (sg != NULL) and the running row max |y| (rm != NULL). Without either it is the plain loop.
static inline __attribute__((always_inline)) void qout(const int32_t *acc, int lda, int ldc, int tn, int nb, const float *xs,
                                                       const float *sw, const float *bb, float *y, int ldy,
                                                       const float *sg, const float *ss, float *rm)
{
    if (!sg && !rm) {
        for (int t = 0; t < tn; t++) {
            float *yo = y + (size_t)t * ldy;
            const int32_t *a = acc + (size_t)t * lda;
            const float xst = xs[t];
            for (int j = 0; j < nb; j++) yo[j] = (float)a[j * ldc] * xst * sw[j] + bb[j];
        }
        return;
    }
    for (int t = 0; t < tn; t++) {
        float *yo = y + (size_t)t * ldy;
        const int32_t *a = acc + (size_t)t * lda;
        const float xst = xs[t];
        float m = rm ? rm[t] : 0.f;
        for (int j = 0; j < nb; j++) {
            float v = (float)a[j * ldc] * xst * sw[j] + bb[j];
            if (sg) v = v * ep_sig(v, sg, ss);
            yo[j] = v;
            const float av = fabsf(v);
            m = av > m ? av : m;
        }
        if (rm) rm[t] = m;
    }
}

static inline __attribute__((always_inline)) void qlin_body(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T,
                                                            int ldq, float *y, int ldy, int8_t *wtmp, int32_t *acc, int n0,
                                                            int n1, const float *sg, const float *ss, float *rm)
{
    const int kp = L->kp;
    if (L->blocked) {
        for (int nb = n0; nb < n1; nb += 16) {
            tasr_unpack_s4((const uint8_t *)L->w + (size_t)nb * (kp >> 1), wtmp, kp * 16);
            const float *sw = L->s + nb, *bb = L->b + nb;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                const int8_t *x0 = xq + (size_t)t0 * ldq;
                tasr_gemm_blk16(wtmp, kp < 512 ? kp : 512, x0, ldq, tn, acc);
                for (int ks = 512; ks < kp; ks += 512) {  // K > 512: split to keep 20-bit lanes overflow-free
                    int32_t part[64 * 16];
                    const int kn = kp - ks < 512 ? kp - ks : 512;
                    tasr_gemm_blk16(wtmp + (size_t)ks * 16, kn, x0 + ks, ldq, tn, part);
                    for (int i = 0; i < tn * 16; i++) acc[i] += part[i];
                }
                qout(acc, 16, 1, tn, 16, xs + t0, sw, bb, y + (size_t)t0 * ldy + nb, ldy, sg, ss, rm ? rm + t0 : NULL);
            }
        }
        return;
    }
    int8_t *xt = L->bits == 8 && L->wn2 && tasr_xt_enable && kp <= XT_KC && T >= 8 ? xt_scratch() : NULL;
    if (xt) {  // broadcast kernel (see tasr_qlin_norms) on 16-row groups x 16-output blocks, x-stationary elsewhere
        int32_t xn[64];
        for (int t0 = 0; t0 < T; t0 += 64) {
            const int tn = T - t0 < 64 ? T - t0 : 64;
            const int8_t *x0 = xq + (size_t)t0 * ldq;
            build_xt(x0, ldq, tn, kp, xt, xn);
            for (int nb0 = n0; nb0 < n1; nb0 += 16) {  // the block's 16 weight rows stay in cache over the groups
                const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
                const float *sw = L->s + nb0, *bb = L->b + nb0;
                const int8_t *wb = L->w + (size_t)nb0 * kp;
                uint32_t wm = 0;
                for (int j = 0; j < nb; j++) wm = L->wn2[nb0 + j] > wm ? L->wn2[nb0 + j] : wm;
                for (int g = 0; g < tn; g += 16) {
                    const int rn = tn - g < 16 ? tn - g : 16;
                    uint32_t xm = 0;
                    for (int r = 0; r < rn; r++) xm = (uint32_t)xn[g + r] > xm ? (uint32_t)xn[g + r] : xm;
                    float *yo = y + (size_t)(t0 + g) * ldy + nb0, *rmg = rm ? rm + t0 + g : NULL;
                    if (rn >= 8 && (uint64_t)xm * wm * 128u < XT_LIM) {
                        tasr_gemm_blk16(xt + (size_t)g * kp, kp, wb, kp, nb, acc);  // acc[j * 16 + r]
                        qout(acc, 1, 16, rn, nb, xs + t0 + g, sw, bb, yo, ldy, sg, ss, rmg);
                    } else {  // the bound does not hold: the x-stationary kernel, same sums
                        tasr_gemm_s8_xr(wb, kp, nb, x0 + (size_t)g * ldq, ldq, rn, acc);  // acc[r * nb + j]
                        qout(acc, nb, 1, rn, nb, xs + t0 + g, sw, bb, yo, ldy, sg, ss, rmg);
                    }
                }
            }
        }
        return;
    }
    if (L->bits == 8 && kp <= tasr_xr_max_kp) {
        // x-stationary kernel (short rows), 16 outputs x 64 rows per tile
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                tasr_gemm_s8_xr(L->w + (size_t)nb0 * kp, kp, nb, xq + (size_t)t0 * ldq, ldq, tn, acc);
                qout(acc, nb, 1, tn, nb, xs + t0, sw, bb, y + (size_t)t0 * ldy + nb0, ldy, sg, ss, rm ? rm + t0 : NULL);
            }
        }
        return;
    }
    if (kp > 1024) {
        // long rows (front-end projections; activations live in PSRAM): split K so a 64 x kc activation sub-tile stays
        // cache resident, and accumulate 16 outputs x 64 rows in acc, so every weight byte is fetched once per 64 rows
        const int nk = (kp + 511) / 512;
        const int kc = ((kp + nk - 1) / nk + 31) & ~31;  // multiple of 32 (int4 unpack granularity)
        int32_t part[64];
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                memset(acc, 0, sizeof(int32_t) * tn * nb);
                for (int k0 = 0; k0 < kp; k0 += kc) {
                    const int kn = kp - k0 < kc ? kp - k0 : kc;
                    for (int j = 0; j < nb; j++) {
                        const int8_t *w;
                        if (L->bits == 4) {
                            tasr_unpack_s4((const uint8_t *)L->w + ((size_t)(nb0 + j) * kp + k0) / 2, wtmp, kn);
                            w = wtmp;
                        } else {
                            w = L->w + (size_t)(nb0 + j) * kp + k0;
                        }
                        tasr_dot_rows_s8(w, xq + (size_t)t0 * ldq + k0, ldq, tn, kn, part);
                        for (int t = 0; t < tn; t++) acc[t * nb + j] += part[t];
                    }
                }
                qout(acc, nb, 1, tn, nb, xs + t0, sw, bb, y + (size_t)t0 * ldy + nb0, ldy, sg, ss, rm ? rm + t0 : NULL);
            }
        }
        return;
    }
    // row layout, K <= 1024: activation tiles of up to 64 rows (<= 64 KB, internal RAM in both engines); each weight
    // row is fetched once per tile
    const int tb = 64;
    for (int t0 = 0; t0 < T; t0 += tb) {
        const int tn = T - t0 < tb ? T - t0 : tb;
        for (int n = n0; n < n1; n++) {
            const int8_t *w;
            if (L->bits == 4) {
                tasr_unpack_s4((const uint8_t *)L->w + (size_t)n * (kp >> 1), wtmp, kp);
                w = wtmp;
            } else {
                w = L->w + (size_t)n * kp;
            }
            tasr_dot_rows_s8(w, xq + (size_t)t0 * ldq, ldq, tn, kp, acc);
            const float sw = L->s[n], b = L->b[n];
            float *yo = y + (size_t)t0 * ldy + n;
            const float *xst = xs + t0;
            if (!sg && !rm) {
                for (int t = 0; t < tn; t++) yo[(size_t)t * ldy] = (float)acc[t] * xst[t] * sw + b;
                continue;
            }
            float *rmt = rm ? rm + t0 : NULL;
            for (int t = 0; t < tn; t++) {
                float v = (float)acc[t] * xst[t] * sw + b;
                if (sg) v = v * ep_sig(v, sg, ss);
                yo[(size_t)t * ldy] = v;
                if (rmt) { const float av = fabsf(v); rmt[t] = av > rmt[t] ? av : rmt[t]; }
            }
        }
    }
}
void tasr_qlin_range(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                     int8_t *wtmp, int32_t *acc, int n0, int n1)
{
    qlin_body(L, xq, xs, T, ldq, y, ldy, wtmp, acc, n0, n1, NULL, NULL, NULL);
}
void tasr_qlin_range_ep(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                        int8_t *wtmp, int32_t *acc, int n0, int n1, const tasr_ep_t *ep)
{
    if (!ep) { qlin_body(L, xq, xs, T, ldq, y, ldy, wtmp, acc, n0, n1, NULL, NULL, NULL); return; }
    qlin_body(L, xq, xs, T, ldq, y, ldy, wtmp, acc, n0, n1, ep->sig_tab, ep->sig_slope, ep->rmax);
}

void tasr_qlin(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy, int8_t *wtmp)
{
    static int32_t acc[64 * 16];
    tasr_qlin_range(L, xq, xs, T, ldq, y, ldy, wtmp, acc, 0, L->n);
}

// ---------------------------------------------------------------------------------------------------------------
// int4 "blocked" GEMM: tile = [kp][16] int8 weights for 16 consecutive outputs (unpacked from the blocked int4
// layout), x rows int8. out[t*16 + j] = sum_k x[t][k] * tile[k][j].  kp multiple of 16 and <= 512 per call
// (20-bit QACC lanes: 512 * 127 * 7 < 2^19, so no overflow is possible with int4 weights).
#ifdef TASR_PIE
static inline void qacc_extract16(int32_t *o)
{
    // QACC_L -> buf[0..19], QACC_H -> buf[32..51] (128-bit stores must stay 16-byte aligned)
    uint8_t buf[64] __attribute__((aligned(16)));
    uint8_t *p = buf;
    __asm__ volatile(
        "ee.st.qacc_l.l.128.ip %0, 16\n"
        "ee.st.qacc_l.h.32.ip %0, 16\n"
        "ee.st.qacc_h.l.128.ip %0, 16\n"
        "ee.st.qacc_h.h.32.ip %0, 16\n"
        : "+r"(p) : : "memory");
    for (int h = 0; h < 2; h++) {
        const uint32_t *w = (const uint32_t *)(buf + 32 * h);
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
        int32_t *d = o + 8 * h;
        // 8 x 20-bit two's complement lanes packed little-endian in 160 bits
        d[0] = ((int32_t)(w0 << 12)) >> 12;
        d[1] = ((int32_t)(((w0 >> 20) | (w1 << 12)) << 12)) >> 12;
        d[2] = ((int32_t)((w1 >> 8) << 12)) >> 12;
        d[3] = ((int32_t)(((w1 >> 28) | (w2 << 4)) << 12)) >> 12;
        d[4] = ((int32_t)(((w2 >> 16) | (w3 << 16)) << 12)) >> 12;
        d[5] = ((int32_t)((w3 >> 4) << 12)) >> 12;
        d[6] = ((int32_t)(((w3 >> 24) | (w4 << 8)) << 12)) >> 12;
        d[7] = ((int32_t)((w4 >> 12) << 12)) >> 12;
    }
}

__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x,
                                                                                int ldq, int T, int32_t *out)
{
    const int nkb = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        const int8_t *pw = tile;
        __asm__ volatile("ee.zero.qacc\n" ::: "memory");
        int n = nkb;
        // tile rows rotate through q0..q3, each loaded three instructions before its use: a load used by the very
        // next instruction stalls (measured on the board: 2.0 cycles per instruction with two registers, 1.15 with
        // the load two ahead). The last step loads three rows past the tile (callers pad tiles by 64 bytes).
        __asm__ volatile(
            "ee.vld.128.ip q0, %1, 16\n"
            "ee.vld.128.ip q1, %1, 16\n"
            "ee.vld.128.ip q2, %1, 16\n"
            "loopnez %2, 1f\n"
            "  ee.vld.128.ip q7, %0, 16\n"          // 16 activations
            "  ee.vsmulas.s8.qacc.ld.incp q3, %1, q0, q7, 0\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 1\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q2, q7, 2\n"
            "  ee.vsmulas.s8.qacc.ld.incp q2, %1, q3, q7, 3\n"
            "  ee.vsmulas.s8.qacc.ld.incp q3, %1, q0, q7, 4\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 5\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q2, q7, 6\n"
            "  ee.vsmulas.s8.qacc.ld.incp q2, %1, q3, q7, 7\n"
            "  ee.vsmulas.s8.qacc.ld.incp q3, %1, q0, q7, 8\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 9\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q2, q7, 10\n"
            "  ee.vsmulas.s8.qacc.ld.incp q2, %1, q3, q7, 11\n"
            "  ee.vsmulas.s8.qacc.ld.incp q3, %1, q0, q7, 12\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 13\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q2, q7, 14\n"
            "  ee.vsmulas.s8.qacc.ld.incp q2, %1, q3, q7, 15\n"
            "1:\n"
            : "+r"(px), "+r"(pw), "+r"(n) : : "memory");
        qacc_extract16(out + (size_t)t * 16);
    }
}
#else
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        int32_t acc[16] = {0};
        for (int k = 0; k < kp; k++) {
            const int32_t xv = px[k];
            const int8_t *w = tile + (size_t)k * 16;
            for (int j = 0; j < 16; j++) acc[j] += xv * (int32_t)w[j];
        }
        for (int j = 0; j < 16; j++) out[(size_t)t * 16 + j] = acc[j];
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// Specialized 48-byte dot products (attention head dim 44 padded to 48): the 48-byte query stays in q0..q2 for all
// T rows. out[t] = dot(q, x + t*ldx).
#ifdef TASR_PIE
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx,
                                                                                int T, int32_t *out)
{
    const int8_t *pq = q;
    __asm__ volatile(
        "ee.vld.128.ip q0, %0, 16\n"
        "ee.vld.128.ip q1, %0, 16\n"
        "ee.vld.128.ip q2, %0, 16\n"
        : "+r"(pq) : : "memory");
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldx;
        int32_t r;
        __asm__ volatile(
            "ee.zero.accx\n"
            "ee.vld.128.ip q3, %1, 16\n"
            "ee.vld.128.ip q4, %1, 16\n"
            "ee.vmulas.s8.accx.ld.ip q5, %1, 16, q3, q0\n"
            "ee.vmulas.s8.accx q4, q1\n"
            "ee.vmulas.s8.accx q5, q2\n"
            "rur.accx_0 %0\n"
            : "=r"(r), "+r"(px) : : "memory");
        out[t] = r;
    }
}
#else
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int32_t acc = 0;
        for (int i = 0; i < 48; i++) acc += (int32_t)a[i] * (int32_t)q[i];
        out[t] = acc;
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// int8 row-layout GEMM, x-stationary: acc[t * nb + j] = sum_k x[t][k] * W[j][k] for j < nb (<= 16), t < T.
// For each row t, up to 6 16-byte chunks of x live in q2..q7 while the nb weight rows stream through q0/q1, so each
// 16 MACs cost one instruction (vs. two when both operands are loaded). kp % 16 == 0; W, x 16-byte aligned.
#ifdef TASR_PIE
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb,
                                                                                const int8_t *x, int ldq, int T,
                                                                                int32_t *acc)
{
    memset(acc, 0, sizeof(int32_t) * (size_t)T * nb);
    for (int t = 0; t < T; t++) {
        for (int k0 = 0; k0 < kp; k0 += 96) {
            int c = (kp - k0) >> 4;
            if (c > 6) c = 6;
            const int8_t *px = x + (size_t)t * ldq + k0;
            const int8_t *pw = W + k0;
            int32_t *pa = acc + (size_t)t * nb;
            int cnt = nb, r, tmp;
            const int rs = kp - 16 * c;
            switch (c) {
        case 1:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx q2, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 2:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx q3, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 3:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx q4, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 4:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx q5, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 5:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "ee.vld.128.ip q6, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q5, q1\n"
                "  ee.vmulas.s8.accx q6, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 6:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "ee.vld.128.ip q6, %[px], 16\n"
                "ee.vld.128.ip q7, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q5, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q6, q0\n"
                "  ee.vmulas.s8.accx q7, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
            }
        }
    }
}
#else
void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    for (int t = 0; t < T; t++)
        for (int j = 0; j < nb; j++) {
            const int8_t *a = x + (size_t)t * ldq, *w = W + (size_t)j * kp;
            int32_t s = 0;
            for (int i = 0; i < kp; i++) s += (int32_t)a[i] * (int32_t)w[i];
            acc[(size_t)t * nb + j] = s;
        }
}
#endif
