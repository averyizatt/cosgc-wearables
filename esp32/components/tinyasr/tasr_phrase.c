// Phrase mode: CTC forward scores of a phrase list (see include/tasr_phrase.h).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tasr_phrase.h"

#define NEG (-1e30f)

struct tasr_phrases {
    int n;
    char **text;
    int **ids;     // tokens of each phrase
    int *len;      // number of tokens
    float **alpha; // forward variables over blank, l1, blank, l2, ..., blank (2 len + 1)
    float free_sum;
    int frames;
};

static inline float logadd(float a, float b)
{
    if (a < b) { float t = a; a = b; b = t; }
    if (b <= NEG) return a;
    return a + log1pf(expf(b - a));
}

tasr_phrases_t *tasr_phrases_parse(const char *buf, size_t n, int vocab)
{
    tasr_phrases_t *p = (tasr_phrases_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    int cap = 0;
    size_t i = 0;
    while (i < n) {
        size_t e = i;
        while (e < n && buf[e] != '\n') e++;
        size_t tab = i;
        while (tab < e && buf[tab] != '\t') tab++;
        if (tab < e && tab > i) {
            int ids[256], k = 0;
            const char *q = buf + tab + 1, *end = buf + e;
            while (q < end && k < 256) {
                while (q < end && (*q == ' ' || *q == '\r')) q++;
                if (q >= end) break;
                int v = 0, any = 0;
                while (q < end && *q >= '0' && *q <= '9') { v = v * 10 + (*q - '0'); q++; any = 1; }
                if (!any) { q++; continue; }
                if (v < vocab) ids[k++] = v;
            }
            if (k > 0) {
                if (p->n == cap) {
                    cap = cap ? 2 * cap : 64;
                    p->text = (char **)realloc(p->text, cap * sizeof(char *));
                    p->ids = (int **)realloc(p->ids, cap * sizeof(int *));
                    p->len = (int *)realloc(p->len, cap * sizeof(int));
                    p->alpha = (float **)realloc(p->alpha, cap * sizeof(float *));
                }
                const size_t tl = tab - i;
                p->text[p->n] = (char *)malloc(tl + 1);
                memcpy(p->text[p->n], buf + i, tl);
                p->text[p->n][tl] = 0;
                p->ids[p->n] = (int *)malloc(k * sizeof(int));
                memcpy(p->ids[p->n], ids, k * sizeof(int));
                p->len[p->n] = k;
                p->alpha[p->n] = (float *)malloc((2 * k + 1) * sizeof(float));
                p->n++;
            }
        }
        i = e + 1;
    }
    if (!p->n) { tasr_phrases_free(p); return NULL; }
    tasr_phrases_reset(p);
    return p;
}

void tasr_phrases_free(tasr_phrases_t *p)
{
    if (!p) return;
    for (int i = 0; i < p->n; i++) { free(p->text[i]); free(p->ids[i]); free(p->alpha[i]); }
    free(p->text); free(p->ids); free(p->len); free(p->alpha); free(p);
}

int tasr_phrases_count(const tasr_phrases_t *p) { return p ? p->n : 0; }
const char *tasr_phrases_text(const tasr_phrases_t *p, int i) { return p && i >= 0 && i < p->n ? p->text[i] : ""; }
int tasr_phrases_frames(const tasr_phrases_t *p) { return p ? p->frames : 0; }
float tasr_phrases_free_score(const tasr_phrases_t *p) { return p ? p->free_sum : 0.f; }

void tasr_phrases_reset(tasr_phrases_t *p)
{
    if (!p) return;
    p->frames = 0;
    p->free_sum = 0.f;
    for (int i = 0; i < p->n; i++)
        for (int s = 0; s < 2 * p->len[i] + 1; s++) p->alpha[i][s] = NEG;
}

void tasr_phrases_step(tasr_phrases_t *p, const float *lp, float best_lp)
{
    if (!p) return;
    const float lb = lp[0];
    for (int i = 0; i < p->n; i++) {
        float *a = p->alpha[i];
        const int *l = p->ids[i];
        const int S = 2 * p->len[i] + 1;
        if (p->frames == 0) {
            a[0] = lb;
            a[1] = lp[l[0] + 1];
            continue;
        }
        for (int s = S - 1; s >= 0; s--) {  // in place, from the end: a[s-1] and a[s-2] still hold the previous frame
            float v = a[s];
            if (s >= 1) v = logadd(v, a[s - 1]);
            const int blank = !(s & 1);
            if (!blank && s >= 2 && l[(s - 1) / 2] != l[(s - 3) / 2]) v = logadd(v, a[s - 2]);  // skip a blank between different tokens
            a[s] = v <= NEG ? NEG : v + (blank ? lb : lp[l[(s - 1) / 2] + 1]);
        }
    }
    p->free_sum += best_lp;  // the free decoding's path: the best token at every frame
    p->frames++;
}

float tasr_phrases_score(const tasr_phrases_t *p, int i)
{
    if (!p || i < 0 || i >= p->n || !p->frames) return NEG;
    const int S = 2 * p->len[i] + 1;
    return logadd(p->alpha[i][S - 1], p->alpha[i][S - 2]);
}

int tasr_phrases_best(const tasr_phrases_t *p, float *best, float *second)
{
    int bi = -1;
    float b = NEG, s2 = NEG;
    for (int i = 0; p && i < p->n; i++) {
        const float v = tasr_phrases_score(p, i);
        if (v > b) { s2 = b; b = v; bi = i; }
        else if (v > s2) s2 = v;
    }
    if (best) *best = b;
    if (second) *second = s2;
    return bi;
}
