#pragma once
#include <stdint.h>

typedef struct {
    int n, k, kp, bits, blocked;
    const int8_t *w;  // int8: [n][kp]; int4: packed [n][kp/2] (per 32-block: byte j = w[j] | w[j+16] << 4, offset +8)
    const float *s;   // [n] per-row weight scale
    const float *b;   // [n] bias
    const uint16_t *wn2;  // [n] ceil(|w_j|^2 / 128) for int8 rows of kp <= 256 (tasr_qlin_norms), or NULL
} tasr_qlin_t;

// per-output squared norms for the broadcast int8 kernel's overflow bound (NULL when the layer cannot use it)
const uint16_t *tasr_qlin_norms(const tasr_qlin_t *L);
// 1: int8 layers with kp <= 256 use the broadcast kernel wherever the bound proves its 20-bit lanes cannot overflow
// (same results as the x-stationary kernel); 0: always the x-stationary kernel (benchmark hook)
extern int tasr_xt_enable;

// out[t] = dot(x + t*ldq, w) for t < T (kp multiple of 16, 16-byte aligned rows)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out);
// int8 dot product; kp multiple of 16, both pointers 16-byte aligned.
int32_t tasr_dot_s8(const int8_t *a, const int8_t *b, int kp);
// unpack one int4 row (kp multiple of 32) to int8
void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp);
// dynamic symmetric per-row int8 quantization: xq rows are ldq bytes (>= kp, zero padded to kp), xs = row scale
void tasr_quant_rows(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs);
// the same with each row's max |x| already known (mx[t], e.g. from a tasr_ep_t output stage): one pass over x
void tasr_quant_rows_mx(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs, const float *mx);

// Optional output stage of tasr_qlin_range_ep, applied to each output while it is still in a register:
// sig_tab/sig_slope != NULL: y = y * sigmoid(y) (SiLU) with the caller's 385-entry table over [-12, 12] in 1/16 steps;
// rmax != NULL: rmax[t] = max(rmax[t], |y[t][n]|) over the outputs written (rows relative to the call's xq / y).
typedef struct {
    const float *sig_tab, *sig_slope;
    float *rmax;
} tasr_ep_t;
// y[t][n] = sum_k xq[t][k] * w[n][k] * xs[t] * s[n] + b[n]
void tasr_qlin(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy, int8_t *wtmp);
// outputs [n0, n1) (multiples of 16 for blocked layers). wtmp >= 16*kp+16 bytes (blocked) or kp bytes; acc >= 64*16
void tasr_qlin_range(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                     int8_t *wtmp, int32_t *acc, int n0, int n1);
// int8 layers with kp <= this use the x-stationary kernel, longer rows the row kernel (same results: integer sums)
extern int tasr_xr_max_kp;
void tasr_qlin_range_ep(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                        int8_t *wtmp, int32_t *acc, int n0, int n1, const tasr_ep_t *ep);
// int4-blocked GEMM on an unpacked [kp][16] tile (kp multiple of 16, <= 512): out[t][j] for 16 outputs
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out);
// out[t] = dot48(q, x + t*ldx) (both 16-byte aligned)
void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc);
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out);
