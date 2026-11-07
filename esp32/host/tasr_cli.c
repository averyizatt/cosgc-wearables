// tasr_cli: transcribe 16 kHz mono PCM16 WAV files with the exact ESP32 engine arithmetic.
// usage: tasr_cli model.{tasr,tnm} [--lm lm.tlm] [--beam 4] [--lm_weight w] [--token_bonus b] [--chunk 32] [--left 4]
//                 [--stream] file1.wav [file2.wav ...]
// .tasr = streaming tinyasr model, .tnm = NeMo Conformer-CTC small (utterance mode; --stream: the streaming engine with
// the chunk and left context stored in the model, fed 20 ms at a time as on the device)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tinyasr.h"
#include "tinyasr_lm.h"
#include "tasr_nemo.h"
#include "tasr_phrase.h"

static uint8_t *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 16, sz + 16)) { fclose(f); return NULL; }
    *n = fread(buf, 1, sz, f);
    fclose(f);
    return buf;
}

// minimal RIFF/WAVE reader: PCM16, mono, 16 kHz
static int16_t *read_wav(const char *path, int *ns)
{
    size_t n;
    uint8_t *b = read_file(path, &n);
    if (!b || n < 44 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { free(b); return NULL; }
    size_t p = 12;
    int ch = 0, rate = 0, bits = 0;
    while (p + 8 <= n) {
        uint32_t len;
        memcpy(&len, b + p + 4, 4);
        if (!memcmp(b + p, "fmt ", 4)) {
            ch = b[p + 10] | (b[p + 11] << 8);
            rate = b[p + 12] | (b[p + 13] << 8) | (b[p + 14] << 16) | (b[p + 15] << 24);
            bits = b[p + 22] | (b[p + 23] << 8);
        } else if (!memcmp(b + p, "data", 4)) {
            if (ch != 1 || rate != 16000 || bits != 16) {
                fprintf(stderr, "%s: need 16 kHz mono PCM16 (got %d ch, %d Hz, %d bit)\n", path, ch, rate, bits);
                free(b);
                return NULL;
            }
            if (p + 8 + len > n) len = (uint32_t)(n - p - 8);
            int16_t *pcm = malloc(len);
            memcpy(pcm, b + p + 8, len);
            *ns = len / 2;
            free(b);
            return pcm;
        }
        p += 8 + len + (len & 1);
    }
    free(b);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.{tasr,tnm} [--lm lm.tlm] [--beam 4] [--lm_weight w] [--token_bonus b] file.wav ...\n", argv[0]);
        return 1;
    }
    const char *lm_path = NULL, *phr_path = NULL, *grid = NULL;
    int chunk = 32, left = 4, first = 2, beam = 4, stream = 0, nchunk = 0, nleft = 0, reuse = 0;
    float lw = -1.f, tb = -1.f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--stream")) { stream = 1; first = i + 1; }
        if (!strcmp(argv[i], "--reuse")) { stream = reuse = 1; first = i + 1; }  // check tasr_nemo_transcribe_from against tasr_nemo_transcribe (the push-to-talk sequence)
        if (!strcmp(argv[i], "--nchunk") && i + 1 < argc) { nchunk = atoi(argv[++i]); first = i + 1; }  // .tnm streaming; 0 = stored in the model
        if (!strcmp(argv[i], "--nleft") && i + 1 < argc) { nleft = atoi(argv[++i]); first = i + 1; }
        if (!strcmp(argv[i], "--phrases") && i + 1 < argc) { phr_path = argv[++i]; first = i + 1; }  // phrase mode (make_phrases.py output)
        if (!strcmp(argv[i], "--lm") && i + 1 < argc) { lm_path = argv[++i]; first = i + 1; }
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) { chunk = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--left") && i + 1 < argc) { left = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--beam") && i + 1 < argc) { beam = atoi(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--lm_weight") && i + 1 < argc) { lw = (float)atof(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--token_bonus") && i + 1 < argc) { tb = (float)atof(argv[++i]); first = i + 1; }
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc) { grid = argv[++i]; first = i + 1; }  // "w:b,w:b,...": full pass, LM
    }
    size_t msz;
    uint8_t *mb = read_file(argv[1], &msz);
    if (!mb) { fprintf(stderr, "cannot read model %s\n", argv[1]); return 1; }
    const int nemo = msz >= 4 && !memcmp(mb, "TNM1", 4);
    tasr_lm_t *lm = NULL;
    if (lm_path) {
        size_t lsz;
        uint8_t *lb = read_file(lm_path, &lsz);
        lm = lb ? tasr_lm_load(lb, lsz) : NULL;
        if (!lm) { fprintf(stderr, "cannot load LM %s\n", lm_path); return 1; }
    }
    tasr_nemo_t *nm = NULL;
    tasr_stream_t *s = NULL;
    tasr_decoder_t *dec = NULL;
    if (nemo) {  // defaults tuned on dev-clean: beam 4, LM weight 0.3, token bonus 0.5
        nm = tasr_nemo_load(mb, msz);
        if (!nm) { fprintf(stderr, "cannot load NeMo model %s\n", argv[1]); return 1; }
        if (lm) dec = tasr_decoder_create(lm, 1025, beam, 6, lw < 0 ? tasr_lm_weight(lm, 0.3f) : lw, tb < 0 ? tasr_lm_bonus(lm, 0.5f) : tb);
        if (stream && !tasr_nemo_stream_supported(nm)) { fprintf(stderr, "model has no streaming support\n"); return 1; }
    } else {
        tasr_model_t *m = tasr_model_load(mb, msz);
        if (!m) { fprintf(stderr, "cannot load model %s\n", argv[1]); return 1; }
        s = tasr_stream_create(m, chunk, left);
        if (lm) tasr_stream_set_decoder(s, tasr_decoder_create(lm, 257, beam, 6, lw < 0 ? 0.5f : lw, tb < 0 ? 2.0f : tb));
    }
    tasr_phrases_t *phr = NULL;
    if (phr_path && nemo) {
        size_t psz;
        uint8_t *pb = read_file(phr_path, &psz);
        phr = pb ? tasr_phrases_parse((const char *)pb, psz, 1024) : NULL;
        if (!phr) { fprintf(stderr, "cannot load phrases %s\n", phr_path); return 1; }
        tasr_nemo_set_phrases(phr);
        fprintf(stderr, "%d phrases\n", tasr_phrases_count(phr));
    }
    // --grid: each listed LM weight and token bonus decodes the same full pass's logits again (one encoder run per clip); every
    // setting's text goes to stderr as "GRID\t<k>\t<file>\t<text>", k counting from 0 in the order given
    enum { GMAX = 64, GFR = 4000 };
    tasr_decoder_t *gdec[GMAX];
    int ng = 0;
    float *glog = NULL;
    if (grid) {
        if (!nemo || !lm || stream) { fprintf(stderr, "--grid needs a .tnm model, --lm and a full pass\n"); return 1; }
        for (const char *p = grid; *p && ng < GMAX;) {
            char *e;
            const float w = strtof(p, &e);
            if (*e != ':') { fprintf(stderr, "--grid: expected w:b at %s\n", p); return 1; }
            const float b = strtof(e + 1, &e);
            gdec[ng++] = tasr_decoder_create(lm, 1025, beam, 6, w, b);
            p = *e == ',' ? e + 1 : e;
        }
        glog = (float *)malloc(sizeof(float) * GFR * 1025);
        if (!glog) { fprintf(stderr, "--grid: no memory\n"); return 1; }
    }
    static char text[16384];
    tasr_nemo_stream_t *ns_ =nemo && stream ? tasr_nemo_stream_new(nm, nchunk, nleft, dec) : NULL;
    for (int i = first; i < argc; i++) {
        int ns = 0;
        int16_t *pcm = read_wav(argv[i], &ns);
        if (!pcm) { fprintf(stderr, "skip %s\n", argv[i]); continue; }
        clock_t t0 = clock();
        if (ns_ && reuse) {  // two sequences, each with 0.5 s of silence after the audio: release after a stream that was finished, and one that was not
            static char text2[16384];
            enum { PAD = 8000, MAXF = 4000 };
            static float a[MAXF * 1025], b[MAXF * 1025];
            int16_t *p2 = (int16_t *)calloc((size_t)ns + PAD, sizeof(int16_t));
            memcpy(p2, pcm, sizeof(int16_t) * (size_t)ns);
            int fa = 0, fb = 0, rows = 0;
            tasr_nemo_stream_keep(ns_, 16000 * 120);
            clock_t ta = clock();
            tasr_nemo_transcribe(nm, p2, ns + PAD, dec, text, sizeof(text), a, MAXF, &fa);
            const double ca = (double)(clock() - ta) / CLOCKS_PER_SEC;
            double cb = 0.;
            for (int fin = 0; fin < 2; fin++) {
                tasr_nemo_stream_reset(ns_);
                for (int j = 0; j < ns; j += 320) tasr_nemo_stream_feed(ns_, pcm + j, ns - j < 320 ? ns - j : 320);
                if (fin) tasr_nemo_stream_finish(ns_, text2, sizeof(text2));
                const float *kx = tasr_nemo_stream_kept(ns_, &rows);
                ta = clock();
                tasr_nemo_transcribe_from(nm, p2, ns + PAD, dec, text2, sizeof(text2), b, MAXF, &fb, kx, rows);
                cb = (double)(clock() - ta) / CLOCKS_PER_SEC;
                float worst = 0.f;
                if (fa == fb) for (long k = 0; k < (long)fa * 1025; k++) { const float e = fabsf(a[k] - b[k]); worst = e > worst ? e : worst; }
                printf("REUSE %s finished %d: rows %d of %d, frames %d %d, largest logit difference %g, text %s, CPU %.3f -> %.3f s\n", argv[i], fin, rows, fa, fa, fb, worst,
                       strcmp(text, text2) ? "DIFFERS" : "same", ca, cb);
            }
            free(p2);
            free(pcm);
            continue;
        }
        if (ns_) {
            tasr_nemo_stream_reset(ns_);
            for (int j = 0; j < ns; j += 320) tasr_nemo_stream_feed(ns_, pcm + j, ns - j < 320 ? ns - j : 320);
            tasr_nemo_stream_finish(ns_, text, sizeof(text));
        } else if (nemo) {
            int nf = 0;
            tasr_nemo_transcribe(nm, pcm, ns, dec, text, sizeof(text), glog, glog ? GFR : 0, &nf);
            if (glog && nf > GFR) fprintf(stderr, "--grid: %s has %d frames, more than %d: not redecoded\n", argv[i], nf, GFR);
            else for (int k = 0; k < ng; k++) {
                static char gt[16384];
                tasr_nemo_decode_logits(nm, glog, nf, gdec[k], gt, sizeof(gt));
                fprintf(stderr, "GRID\t%d\t%s\t%s\n", k, argv[i], gt);
            }
        } else {
            tasr_stream_reset(s);
            for (int j = 0; j < ns; j += 320) tasr_stream_feed(s, pcm + j, ns - j < 320 ? ns - j : 320);
            tasr_stream_finish(s);
            snprintf(text, sizeof(text), "%s", tasr_stream_text(s));
        }
        double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("%s\t%s\n", argv[i], text);
        if (nemo && !ns_) {  // full pass: the model's confidence, on stderr so stdout keeps its format
            tasr_nemo_conf_t c;
            tasr_nemo_last_confidence(&c);
            fprintf(stderr, "CONF\t%s\t%.4f\t%.4f\t%d\n", argv[i], c.mean_logp, c.min_logp, c.tokens);
            if (phr) {  // phrase mode: every phrase's log-likelihood, then the free path's score and the frame count
                fprintf(stderr, "PHR\t%s\t%.3f\t%d\t", argv[i], tasr_phrases_free_score(phr), tasr_phrases_frames(phr));
                for (int k = 0; k < tasr_phrases_count(phr); k++) fprintf(stderr, k ? " %.3f" : "%.3f", tasr_phrases_score(phr, k));
                fprintf(stderr, "\n");
            }
        }
        fprintf(stderr, "  %.2f s audio, %.3f s CPU (host)\n", ns / 16000.0, sec);
        free(pcm);
    }
    return 0;
}
