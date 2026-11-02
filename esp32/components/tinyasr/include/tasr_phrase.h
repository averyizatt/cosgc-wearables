// Phrase mode: score a user's list of phrases against the model's output, alongside free decoding.
// For every phrase, the CTC log-likelihood that the audio says exactly that phrase (forward algorithm over the frames, all
// alignments). The firmware shows the best phrase when it clearly beats free decoding, otherwise the free text.
// The list is prepared off the device (esp32/tools/make_phrases.py): one line per phrase, "text<TAB>token ids separated by spaces",
// tokenized with the model's own SentencePiece model.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tasr_phrases tasr_phrases_t;

tasr_phrases_t *tasr_phrases_parse(const char *buf, size_t n, int vocab);  // NULL if nothing valid; ids >= vocab are skipped
void tasr_phrases_free(tasr_phrases_t *p);
int tasr_phrases_count(const tasr_phrases_t *p);
const char *tasr_phrases_text(const tasr_phrases_t *p, int i);
void tasr_phrases_reset(tasr_phrases_t *p);
// one frame: lp = log posteriors, blank first then the vocab's tokens (the beam decoder's layout); best_lp = the largest of them
void tasr_phrases_step(tasr_phrases_t *p, const float *lp, float best_lp);
float tasr_phrases_score(const tasr_phrases_t *p, int i);  // log-likelihood of phrase i so far
float tasr_phrases_free_score(const tasr_phrases_t *p);    // sum over frames of the best token's log posterior (free decoding's path)
int tasr_phrases_frames(const tasr_phrases_t *p);
int tasr_phrases_best(const tasr_phrases_t *p, float *best, float *second);  // index of the best phrase, or -1

#ifdef __cplusplus
}
#endif
