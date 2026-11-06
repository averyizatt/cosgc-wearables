// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// Session log on the microSD card (sdlog.h). Layout on the card:
//   prompts.txt            optional: one prompt per line, or "text | cue" to score against text while the screen shows cue (for example
//                          a phonetic spelling); lines starting with '#' are notes copied into the session (speaker code, style, ...)
//   s0001/session.json     the model's size and fingerprint, the firmware build, the decoder settings, the notes, the prompts
//   s0001/log.jsonl        one line per utterance (fields: sdlog_utt_t, plus levels and file names)
//   s0001/u0001.wav        what the model heard (after gain control), 16 kHz mono 16-bit
//   s0001/u0001r.wav       the microphone before gain control (for levels and noise)
// The log is appended and synced after every utterance, so pulling the power loses at most the utterance being written.
#include "sdlog.h"
#include "sdkconfig.h"

#ifdef CONFIG_TASR_SDLOG
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cJSON.h"
#include "driver/sdmmc_host.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"

#define ROOT "/sdcard"
#define MAX_PROMPTS 1000
static const char *TAG = "sd";
static bool s_on;
static char s_dir[24];
static int s_utt;
static char *s_ptext;              // prompts.txt, lines split in place
static const char *s_prompt[MAX_PROMPTS];  // the words to score against
static const char *s_cue[MAX_PROMPTS];     // what the screen shows ("text | cue" lines; otherwise the text itself)
static int s_np, s_pi;             // prompts, the current one (0-based)

static uint32_t fnv1a(const uint8_t *p, uint32_t n)
{
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; i++) h = (h ^ p[i]) * 16777619u;
    return h;
}

static bool write_wav(const char *path, const int16_t *pcm, int n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t data = (uint32_t)n * 2, riff = 36 + data, fmt = 16, rate = 16000, bps = 32000;
    const uint16_t pcm_fmt = 1, ch = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
    fwrite(&pcm_fmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&bps, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    const bool ok = fwrite(pcm, 2, n, f) == (size_t)n;
    return fclose(f) == 0 && ok;
}

static int cmp_float(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}
// dBFS of 20 ms blocks: the speech level (90th percentile after the pre-roll), the noise (median of the pre-roll, or the
// 10th percentile of everything without one) and the peak sample
static void levels(const int16_t *x, int n, int pre, float *speech, float *noise, float *peak)
{
    const int B = 320, nb = n / B;
    *speech = *noise = *peak = -120.f;
    if (nb < 1) return;
    float *db = malloc(sizeof(float) * nb);
    if (!db) return;
    int mx = 0;
    for (int b = 0; b < nb; b++) {
        double e = 0;
        for (int i = b * B; i < (b + 1) * B; i++) {
            e += (double)x[i] * x[i];
            if (abs(x[i]) > mx) mx = abs(x[i]);
        }
        db[b] = 10.f * log10f((float)(e / B) / (32768.f * 32768.f) + 1e-12f);
    }
    *peak = 20.f * log10f(mx / 32768.f + 1e-9f);
    const int np = pre / B < nb ? pre / B : 0;
    if (np >= 3) {
        qsort(db, np, sizeof(float), cmp_float);
        *noise = db[np / 2];
    }
    qsort(db + np, nb - np, sizeof(float), cmp_float);
    *speech = db[np + (int)(0.9f * (nb - np - 1))];
    if (np < 3) *noise = db[np + (int)(0.1f * (nb - np - 1))];
    free(db);
}

static void read_prompts(cJSON *notes, cJSON *prompts)
{
    FILE *f = fopen(ROOT "/prompts.txt", "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 256 * 1024 || !(s_ptext = heap_caps_malloc(sz + 1, MALLOC_CAP_SPIRAM))) { fclose(f); return; }
    const size_t got = fread(s_ptext, 1, sz, f);
    fclose(f);
    s_ptext[got] = 0;
    for (char *line = strtok(s_ptext, "\r\n"); line; line = strtok(NULL, "\r\n")) {
        while (*line == ' ' || *line == '\t') line++;
        if (!*line) continue;
        if (*line == '#') { cJSON_AddItemToArray(notes, cJSON_CreateString(line + 1 + (line[1] == ' '))); continue; }
        if (s_np < MAX_PROMPTS) {
            char *bar = strchr(line, '|');  // "i need water | i nee wa-uh": score against the left, show the right
            const char *cue = line;
            if (bar) {
                cue = bar + 1;
                while (*cue == ' ') cue++;
                for (char *e = bar; e > line && (e[-1] == ' ' || e[-1] == '\t'); e--) e[-1] = 0;
                *bar = 0;
                if (!*cue) cue = line;
            }
            s_prompt[s_np] = line;
            s_cue[s_np++] = cue;
            cJSON *pj = cJSON_CreateObject();
            cJSON_AddStringToObject(pj, "text", line);
            if (cue != line) cJSON_AddStringToObject(pj, "cue", cue);
            cJSON_AddItemToArray(prompts, pj);
        }
    }
}

void sdlog_start(const void *model, uint32_t model_bytes)
{
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
#if CONFIG_TASR_SD_LDO >= 0
    sd_pwr_ctrl_ldo_config_t ldo = {.ldo_chan_id = CONFIG_TASR_SD_LDO};
    sd_pwr_ctrl_handle_t pwr = NULL;
    if (sd_pwr_ctrl_new_on_chip_ldo(&ldo, &pwr) != ESP_OK) { ESP_LOGW(TAG, "LDO %d for the card did not start", CONFIG_TASR_SD_LDO); return; }
    host.pwr_ctrl_handle = pwr;
#endif
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 4;
    slot.clk = CONFIG_TASR_SD_CLK;
    slot.cmd = CONFIG_TASR_SD_CMD;
    slot.d0 = CONFIG_TASR_SD_D0;
    slot.d1 = CONFIG_TASR_SD_D0 + 1;
    slot.d2 = CONFIG_TASR_SD_D0 + 2;
    slot.d3 = CONFIG_TASR_SD_D0 + 3;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    const esp_vfs_fat_sdmmc_mount_config_t mc = {.format_if_mount_failed = false, .max_files = 4, .allocation_unit_size = 16 * 1024};
    sdmmc_card_t *card;
    const esp_err_t e = esp_vfs_fat_sdmmc_mount(ROOT, &host, &slot, &mc, &card);
    if (e != ESP_OK) {
        ESP_LOGW(TAG, "no microSD card (%s): utterances are not logged", esp_err_to_name(e));
        return;
    }
    struct stat st;
    int k = 1;
    for (; k < 10000; k++) {
        snprintf(s_dir, sizeof(s_dir), ROOT "/s%04d", k);
        if (stat(s_dir, &st) != 0) break;
    }
    if (k == 10000 || mkdir(s_dir, 0777) != 0) { ESP_LOGW(TAG, "cannot make a session folder on the card"); return; }
    cJSON *j = cJSON_CreateObject(), *notes = cJSON_CreateArray(), *prompts = cJSON_CreateArray();
    read_prompts(notes, prompts);
    char hex[12];
    snprintf(hex, sizeof(hex), "%08lx", (unsigned long)fnv1a(model, model_bytes));
    cJSON_AddNumberToObject(j, "session", k);
    cJSON_AddNumberToObject(j, "model_bytes", model_bytes);
    cJSON_AddStringToObject(j, "model_fnv1a", hex);  // esp32/tools/sd_session.py models: the same hash of each .tnm
    cJSON_AddStringToObject(j, "firmware", __DATE__ " " __TIME__);
    cJSON_AddStringToObject(j, "idf", esp_get_idf_version());
    cJSON_AddNumberToObject(j, "beam", CONFIG_TASR_BEAM);
    cJSON_AddNumberToObject(j, "lm_weight_x100", CONFIG_TASR_LM_WEIGHT_X100);
    cJSON_AddNumberToObject(j, "token_bonus_x100", CONFIG_TASR_TOKEN_BONUS_X100);
    cJSON_AddItemToObject(j, "notes", notes);
    cJSON_AddItemToObject(j, "prompts", prompts);
    char path[48];
    snprintf(path, sizeof(path), "%s/session.json", s_dir);
    FILE *f = fopen(path, "w");
    char *txt = cJSON_Print(j);
    if (f && txt) fputs(txt, f);
    if (f) fclose(f);
    free(txt);
    cJSON_Delete(j);
    s_on = f != NULL;
    printf("sd: card %s, %llu MB, logging to %s (model %s), %d prompts\n", card->cid.name,
           ((unsigned long long)card->csd.capacity) * card->csd.sector_size / (1024 * 1024), s_dir, hex, s_np);
}

bool sdlog_on(void) { return s_on; }
const char *sdlog_prompt(void) { return s_on && s_pi < s_np ? s_prompt[s_pi] : NULL; }
const char *sdlog_cue(void) { return s_on && s_pi < s_np ? s_cue[s_pi] : NULL; }
int sdlog_prompt_pos(int *count)
{
    if (count) *count = s_np;
    return s_pi + 1;
}
void sdlog_prompt_back(void)
{
    if (s_pi > 0) s_pi--;
}

void sdlog_utt(const sdlog_utt_t *u)
{
    if (!s_on) return;
    const int64_t t0 = esp_timer_get_time();
    s_utt++;
    char wav[48], raw[48], name[16], rname[16];
    snprintf(name, sizeof(name), "u%04d.wav", s_utt);
    snprintf(rname, sizeof(rname), "u%04dr.wav", s_utt);
    snprintf(wav, sizeof(wav), "%s/%s", s_dir, name);
    snprintf(raw, sizeof(raw), "%s/%s", s_dir, rname);
    const bool ok_wav = write_wav(wav, u->audio, u->n_audio);
    const bool ok_raw = u->raw && u->n_raw > 0 && write_wav(raw, u->raw, u->n_raw);
    float speech, noise, peak;
    if (u->raw && u->n_raw > 0) levels(u->raw, u->n_raw, u->pre, &speech, &noise, &peak);
    else levels(u->audio, u->n_audio, u->pre, &speech, &noise, &peak);
    cJSON *j = cJSON_CreateObject();
    cJSON_AddNumberToObject(j, "utt", s_utt);
    const char *pr = sdlog_prompt();
    if (pr) {
        cJSON_AddNumberToObject(j, "prompt_idx", s_pi + 1);
        cJSON_AddStringToObject(j, "prompt", pr);
        if (s_cue[s_pi] != pr) cJSON_AddStringToObject(j, "cue", s_cue[s_pi]);
    }
    cJSON_AddStringToObject(j, "text", u->text);
    cJSON_AddStringToObject(j, "free", u->free_text);
    if (u->stream_text) cJSON_AddStringToObject(j, "stream", u->stream_text);
    if (u->phrase) {
        cJSON_AddStringToObject(j, "phrase", u->phrase);
        cJSON_AddNumberToObject(j, "phrase_gap", roundf(u->phrase_gap * 1000) / 1000);
        cJSON_AddNumberToObject(j, "phrase_thr", roundf(u->phrase_thr * 1000) / 1000);
        cJSON_AddNumberToObject(j, "phrase_n", u->phrase_n);
    }
    cJSON_AddNumberToObject(j, "conf_level", u->conf_level);
    cJSON_AddNumberToObject(j, "conf", roundf(u->conf * 1000) / 1000);
    cJSON_AddBoolToObject(j, "full", u->full);
    cJSON_AddNumberToObject(j, "audio_s", roundf(u->n_audio / 16.f) / 1000);
    cJSON_AddNumberToObject(j, "pre_s", roundf(u->pre / 16.f) / 1000);
    cJSON_AddNumberToObject(j, "delay_ms", round(u->delay_ms));
    cJSON_AddNumberToObject(j, "press_s", round(u->press_s * 1000) / 1000);
    cJSON_AddNumberToObject(j, "speech_db", roundf(speech * 10) / 10);
    cJSON_AddNumberToObject(j, "noise_db", roundf(noise * 10) / 10);
    cJSON_AddNumberToObject(j, "peak_db", roundf(peak * 10) / 10);
    if (ok_wav) cJSON_AddStringToObject(j, "wav", name);
    if (ok_raw) cJSON_AddStringToObject(j, "raw", rname);
    char *line = cJSON_PrintUnformatted(j);
    cJSON_Delete(j);
    char path[48];
    snprintf(path, sizeof(path), "%s/log.jsonl", s_dir);
    FILE *f = fopen(path, "a");
    bool ok = f && line && fputs(line, f) >= 0 && fputc('\n', f) != EOF;
    if (f) {
        fflush(f);
        fsync(fileno(f));
        ok = fclose(f) == 0 && ok;
    }
    free(line);
    if (!ok || !ok_wav) ESP_LOGW(TAG, "writing utterance %d to the card failed (card full or removed?)", s_utt);
    else printf("sd: %s/%s written in %lld ms\n", s_dir + sizeof(ROOT), name, (esp_timer_get_time() - t0) / 1000);
    if (s_pi < s_np) s_pi++;
}

#else
void sdlog_start(const void *model, uint32_t model_bytes) { (void)model; (void)model_bytes; }
bool sdlog_on(void) { return false; }
const char *sdlog_prompt(void) { return NULL; }
const char *sdlog_cue(void) { return NULL; }
int sdlog_prompt_pos(int *count) { if (count) *count = 0; return 0; }
void sdlog_prompt_back(void) {}
void sdlog_utt(const sdlog_utt_t *u) { (void)u; }
#endif
