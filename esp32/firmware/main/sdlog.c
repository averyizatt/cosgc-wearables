// Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
// Session log on the microSD card (sdlog.h). Layout on the card:
//   prompts.txt            optional: one prompt per line, or "text | cue" to score against text while the screen shows cue (for example
//                          a phonetic spelling); lines starting with '#' are notes copied into the session (speaker code, style, ...)
//   prompts1.txt ... prompts9.txt   instead of prompts.txt: one per condition (distance, place, effort). The first boot session uses
//                          the first; after the last prompt, a tap opens a new session with the next file (wrapping round)
//   s0001/session.json     the model's size and fingerprint, the firmware build, the decoder settings, the notes, the prompts
//   s0001/log.jsonl        one line per utterance (fields: sdlog_utt_t, plus levels and file names)
//   s0001/u0001.wav        what the model heard (after gain control), 16 kHz mono 16-bit
//   s0001/u0001r.wav       the microphone before gain control (for levels and noise)
//   s0001/u0001b.wav       with a second microphone (CONFIG_TASR_MIC2): the one the recognizer did not hear, same timing ("mic" in
//                          the log line says which one it heard)
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
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
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
// The card is written by a task of its own: a write takes 0.2 to 2 s (the card's own housekeeping), and the recognizer must not
// wait for it. sdlog_utt copies the audio and hands over the log line; the writer adds the levels and file names.
typedef struct {
    cJSON *j;          // the log line without levels and file names
    int16_t *pcm;      // audio, then raw, in one PSRAM block (NULL: copy failed, the line is still written)
    int n_audio, n_raw, n_raw2, pre, utt;  // pcm holds audio, raw, then raw2 (the other microphone)
    int64_t t0;
    char dir[24];      // the session folder it belongs to (a tap can open the next session while this one is written)
} job_t;
static QueueHandle_t s_jobs;
static void writer_task(void *arg);
// prompt files: prompts1.txt ... prompts9.txt in turn (a tap after the last prompt opens the next session with the next file), or
// prompts.txt alone
static char s_files[9][16];
static int s_nfiles, s_file;
static char s_hex[12];
static uint32_t s_model_bytes;

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

static void read_prompts(const char *name, cJSON *notes, cJSON *prompts)
{
    heap_caps_free(s_ptext);
    s_ptext = NULL;
    s_np = s_pi = 0;
    if (!name) return;
    char path[40];
    snprintf(path, sizeof(path), ROOT "/%s", name);
    FILE *f = fopen(path, "rb");
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

// a new session folder (s0001, s0002, ...) with the current prompt file; false if the card refused
static bool open_session(void)
{
    struct stat st;
    int k = 1;
    char dir[24];
    for (; k < 10000; k++) {
        snprintf(dir, sizeof(dir), ROOT "/s%04d", k);
        if (stat(dir, &st) != 0) break;
    }
    if (k == 10000 || mkdir(dir, 0777) != 0) { ESP_LOGW(TAG, "cannot make a session folder on the card"); return false; }
    snprintf(s_dir, sizeof(s_dir), "%s", dir);
    s_utt = 0;
    cJSON *j = cJSON_CreateObject(), *notes = cJSON_CreateArray(), *prompts = cJSON_CreateArray();
    read_prompts(s_nfiles ? s_files[s_file] : NULL, notes, prompts);
    cJSON_AddNumberToObject(j, "session", k);
    cJSON_AddNumberToObject(j, "model_bytes", s_model_bytes);
    cJSON_AddStringToObject(j, "model_fnv1a", s_hex);  // esp32/tools/sd_session.py models: the same hash of each .tnm
    cJSON_AddStringToObject(j, "firmware", __DATE__ " " __TIME__);
    cJSON_AddStringToObject(j, "idf", esp_get_idf_version());
    cJSON_AddNumberToObject(j, "beam", CONFIG_TASR_BEAM);
    cJSON_AddNumberToObject(j, "lm_weight_x100", CONFIG_TASR_LM_WEIGHT_X100);
    cJSON_AddNumberToObject(j, "token_bonus_x100", CONFIG_TASR_TOKEN_BONUS_X100);
    if (s_nfiles) cJSON_AddStringToObject(j, "prompt_file", s_files[s_file]);
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
    return f != NULL;
}

bool sdlog_next_session(void)
{
    if (!s_on || !s_nfiles) return false;
    s_file = (s_file + 1) % s_nfiles;
    const bool ok = open_session();
    printf("sd: new session %s, %d prompts from %s\n", s_dir + sizeof(ROOT), s_np, s_files[s_file]);
    return ok;
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
    char path[40];
    s_nfiles = 0;
    for (int i = 1; i <= 9; i++) {
        snprintf(path, sizeof(path), ROOT "/prompts%d.txt", i);
        if (stat(path, &st) == 0) snprintf(s_files[s_nfiles++], sizeof(s_files[0]), "prompts%d.txt", i);
    }
    if (!s_nfiles && stat(ROOT "/prompts.txt", &st) == 0) snprintf(s_files[s_nfiles++], sizeof(s_files[0]), "prompts.txt");
    snprintf(s_hex, sizeof(s_hex), "%08lx", (unsigned long)fnv1a(model, model_bytes));
    s_model_bytes = model_bytes;
    s_file = 0;
    const bool f = open_session();
    s_jobs = xQueueCreate(4, sizeof(job_t));
    s_on = f && s_jobs && xTaskCreate(writer_task, "sd_write", 6144, NULL, 2, NULL) == pdPASS;
    printf("sd: card %s, %llu MB, logging to %s (model %s), %d prompts from %s\n", card->cid.name,
           ((unsigned long long)card->csd.capacity) * card->csd.sector_size / (1024 * 1024), s_dir, s_hex, s_np,
           s_nfiles ? s_files[s_file] : "no prompt file");
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

static void writer_task(void *arg)
{
    (void)arg;
    for (;;) {
        job_t jb;
        xQueueReceive(s_jobs, &jb, portMAX_DELAY);
        char wav[48], raw[48], name[16], rname[16];
        snprintf(name, sizeof(name), "u%04d.wav", jb.utt);
        snprintf(rname, sizeof(rname), "u%04dr.wav", jb.utt);
        snprintf(wav, sizeof(wav), "%s/%s", jb.dir, name);
        snprintf(raw, sizeof(raw), "%s/%s", jb.dir, rname);
        const int16_t *audio = jb.pcm, *rawp = jb.pcm ? jb.pcm + jb.n_audio : NULL;
        const bool ok_wav = audio && write_wav(wav, audio, jb.n_audio);
        const bool ok_raw = rawp && jb.n_raw > 0 && write_wav(raw, rawp, jb.n_raw);
        float speech = -120.f, noise = -120.f, peak = -120.f;
        if (rawp && jb.n_raw > 0) levels(rawp, jb.n_raw, jb.pre, &speech, &noise, &peak);
        else if (audio) levels(audio, jb.n_audio, jb.pre, &speech, &noise, &peak);
        // the other microphone (CONFIG_TASR_MIC2): uNNNNb.wav and its own levels
        const int16_t *raw2 = jb.pcm && jb.n_raw2 > 0 ? jb.pcm + jb.n_audio + jb.n_raw : NULL;
        char bname[16];
        snprintf(bname, sizeof(bname), "u%04db.wav", jb.utt);
        snprintf(raw, sizeof(raw), "%s/%s", jb.dir, bname);
        const bool ok_b = raw2 && write_wav(raw, raw2, jb.n_raw2);
        float speech2 = -120.f, noise2 = -120.f, peak2 = -120.f;
        if (raw2) levels(raw2, jb.n_raw2, jb.pre, &speech2, &noise2, &peak2);
        heap_caps_free(jb.pcm);
        cJSON *j = jb.j;
        cJSON_AddNumberToObject(j, "speech_db", roundf(speech * 10) / 10);
        cJSON_AddNumberToObject(j, "noise_db", roundf(noise * 10) / 10);
        cJSON_AddNumberToObject(j, "peak_db", roundf(peak * 10) / 10);
        if (ok_wav) cJSON_AddStringToObject(j, "wav", name);
        if (ok_raw) cJSON_AddStringToObject(j, "raw", rname);
        if (raw2) {
            cJSON_AddNumberToObject(j, "speech_db_b", roundf(speech2 * 10) / 10);
            cJSON_AddNumberToObject(j, "noise_db_b", roundf(noise2 * 10) / 10);
            cJSON_AddNumberToObject(j, "peak_db_b", roundf(peak2 * 10) / 10);
            if (ok_b) cJSON_AddStringToObject(j, "raw_b", bname);  // the microphone the recognizer did not hear
        }
        char *line = cJSON_PrintUnformatted(j);
        cJSON_Delete(j);
        char path[48];
        snprintf(path, sizeof(path), "%s/log.jsonl", jb.dir);
        FILE *f = fopen(path, "a");
        bool ok = f && line && fputs(line, f) >= 0 && fputc('\n', f) != EOF;
        if (f) {
            fflush(f);
            fsync(fileno(f));
            ok = fclose(f) == 0 && ok;
        }
        free(line);
        if (!ok || !ok_wav) ESP_LOGW(TAG, "writing utterance %d to the card failed (card full or removed?)", jb.utt);
        else printf("sd: %s/%s written in %lld ms\n", jb.dir + sizeof(ROOT), name, (esp_timer_get_time() - jb.t0) / 1000);
    }
}

void sdlog_utt(const sdlog_utt_t *u)
{
    if (!s_on) return;
    job_t jb = {.utt = ++s_utt, .t0 = esp_timer_get_time(), .pre = u->pre, .n_audio = u->n_audio,
                .n_raw = u->raw ? u->n_raw : 0, .n_raw2 = u->raw2 ? u->n_raw2 : 0};
    snprintf(jb.dir, sizeof(jb.dir), "%s", s_dir);
    jb.pcm = heap_caps_malloc(sizeof(int16_t) * (size_t)(jb.n_audio + jb.n_raw + jb.n_raw2 + 1), MALLOC_CAP_SPIRAM);
    if (jb.pcm) {
        memcpy(jb.pcm, u->audio, sizeof(int16_t) * jb.n_audio);
        if (jb.n_raw) memcpy(jb.pcm + jb.n_audio, u->raw, sizeof(int16_t) * jb.n_raw);
        if (jb.n_raw2) memcpy(jb.pcm + jb.n_audio + jb.n_raw, u->raw2, sizeof(int16_t) * jb.n_raw2);
    } else {
        ESP_LOGW(TAG, "no PSRAM to copy utterance %d: only its log line is written", jb.utt);
    }
    cJSON *j = cJSON_CreateObject();
    jb.j = j;
    cJSON_AddNumberToObject(j, "utt", jb.utt);
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
    cJSON_AddNumberToObject(j, "mic", u->mic ? u->mic : 1);
    cJSON_AddNumberToObject(j, "audio_s", roundf(u->n_audio / 16.f) / 1000);
    cJSON_AddNumberToObject(j, "pre_s", roundf(u->pre / 16.f) / 1000);
    cJSON_AddNumberToObject(j, "delay_ms", round(u->delay_ms));
    cJSON_AddNumberToObject(j, "press_s", round(u->press_s * 1000) / 1000);
    if (xQueueSend(s_jobs, &jb, 0) != pdTRUE) {  // the writer is four utterances behind: drop this one rather than make the user wait
        ESP_LOGW(TAG, "card writes are behind: utterance %d not saved", jb.utt);
        cJSON_Delete(j);
        heap_caps_free(jb.pcm);
    }
    if (s_pi < s_np) s_pi++;
}

#else
void sdlog_start(const void *model, uint32_t model_bytes) { (void)model; (void)model_bytes; }
bool sdlog_on(void) { return false; }
const char *sdlog_prompt(void) { return NULL; }
const char *sdlog_cue(void) { return NULL; }
int sdlog_prompt_pos(int *count) { if (count) *count = 0; return 0; }
void sdlog_prompt_back(void) {}
bool sdlog_next_session(void) { return false; }
void sdlog_utt(const sdlog_utt_t *u) { (void)u; }
#endif
