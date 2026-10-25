/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Models: any provider's model, opened by path or id and run by name. A graph
 * saved as an .rlxp package runs anywhere; a plugin's models (a downstream
 * collection's LLMs, or the reference plugin's demo:count) load through
 * RLX_MODEL_PLUGINS or rlx_load_model_plugin. Where nothing loads at run
 * time (iOS, wasm), the plugin is linked into a static librlx_capi instead:
 * build with -DRLX_LINKED_PLUGINS. The Swift, Python and JS examples of the
 * same name do the same steps.
 *
 *     RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib ./models
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rlx.h"

#define OK(call)                                                              \
    do {                                                                      \
        int32_t rc_ = (call);                                                 \
        if (rc_ != RLX_OK) {                                                  \
            fprintf(stderr, "FAIL %s -> %d: %s\n", #call, rc_, rlx_last_error()); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* A string field of a flat JSON object (enough for `info`'s top level). */
static const char *field(const char *json, const char *key, char *buf, size_t cap) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":\"", key);
    const char *at = strstr(json, pat);
    if (!at) return "?";
    at += strlen(pat);
    size_t n = strcspn(at, "\"");
    if (n >= cap) n = cap - 1;
    memcpy(buf, at, n);
    buf[n] = '\0';
    return buf;
}

static void print_info(const char *label, RlxModel *m, int with_device) {
    char *info = NULL, a[64], b[64], c[64];
    OK(rlx_model_info(m, &info));
    printf("%s: %s (%s)", label, field(info, "family", a, sizeof a), field(info, "task", b, sizeof b));
    if (with_device) printf(" on %s", field(info, "device", c, sizeof c));
    printf("\n");
    rlx_string_free(info);
}

int main(void) {
#ifdef RLX_LINKED_PLUGINS
    if (rlx_link_model_plugins() < 1) return 1;
#endif
    RlxGraph *g = NULL;
    uint32_t x, w, y;
    size_t xs[2] = {1, 2}, ws[2] = {2, 2};
    OK(rlx_graph_new("affine", &g));
    OK(rlx_graph_input(g, "x", xs, 2, "f32", &x));
    OK(rlx_graph_param(g, "w", ws, 2, "f32", &w));
    OK(rlx_graph_matmul(g, x, w, &y));
    OK(rlx_graph_set_outputs(g, &y, 1));

    const char *tmp = getenv("TMPDIR");
    char path[512];
    snprintf(path, sizeof path, "%s/rlx-example-c.rlxp", tmp && *tmp ? tmp : "/tmp");
    const char *wname[1] = {"w"};
    const float wdata[4] = {2, 0, 0, 3};
    const float *wptr[1] = {wdata};
    const size_t wlen[1] = {4};
    OK(rlx_save_model(g, path, wname, wptr, wlen, 1,
                      "{\"task\": \"graph\", \"family\": \"affine\", \"outputs\": [\"y\"]}"));
    rlx_graph_free(g);

    RlxModel *m = NULL;
    OK(rlx_model_open(path, "{\"device\": \"cpu\"}", &m));
    print_info("model", m, 1);
    const char *xname[1] = {"x"};
    const float xdata[2] = {1, 1};
    const float *xptr[1] = {xdata};
    const size_t xlen[1] = {2};
    RlxOutputs *outs = NULL;
    OK(rlx_model_run(m, xname, xptr, xlen, 1, NULL, &outs));
    const float *out;
    size_t n;
    OK(rlx_outputs_get(outs, 0, &out, &n));
    printf("model run: %.3f %.3f\n", out[0], out[1]);
    rlx_outputs_free(outs);
    rlx_model_free(m);
    remove(path);

    RlxModel *lm = NULL;
    OK(rlx_model_open("demo:count", NULL, &lm));
    print_info("plugin model", lm, 0);
    char *text = NULL;
    OK(rlx_model_generate_text(lm, "c", "{\"max_tokens\": 4}", &text));
    printf("model text: %s\n", text);
    rlx_string_free(text);
    const uint32_t prompt[1] = {2};
    RlxU32s ids = {0};
    OK(rlx_model_generate(lm, prompt, 1, 3, NULL, &ids));
    printf("model ids:");
    for (size_t i = 0; i < ids.len; i++) printf(" %u", ids.ptr[i]);
    printf("\n");
    rlx_u32s_free(ids);

    /* By task: a chat, speech out and back in, embeddings. */
    OK(rlx_model_chat(lm, "[{\"role\": \"user\", \"content\": \"c\"}]", "{\"max_tokens\": 4}", &text));
    printf("chat: %s\n", text);
    rlx_string_free(text);
    rlx_model_free(lm);

    RlxModel *voice = NULL, *ear = NULL, *embedder = NULL;
    OK(rlx_model_open("demo:voice", NULL, &voice));
    RlxF32s audio = {0};
    uint32_t rate = 0;
    OK(rlx_model_synthesize(voice, "hello", NULL, &audio, &rate));
    printf("voice: %zu samples at %u Hz\n", audio.len, rate);
    OK(rlx_model_open("demo:speech", NULL, &ear));
    OK(rlx_model_transcribe(ear, audio.ptr, audio.len, rate, NULL, &text));
    printf("heard: %s\n", text);
    rlx_string_free(text);
    rlx_f32s_free(audio);

    OK(rlx_model_open("demo:embed", NULL, &embedder));
    const char *texts[2] = {"hello", "sky"};
    char *rows = NULL;
    OK(rlx_model_embed(embedder, texts, 2, NULL, &rows));
    printf("embed:");  /* rows is JSON: [[5.0,2.0],[3.0,0.0]] */
    for (const char *p = rows; *p;) {
        char *end;
        double v = strtod(p, &end);
        if (end != p) {
            printf(" %d", (int)v);
            p = end;
        } else {
            p++;
        }
    }
    printf("\n");
    rlx_string_free(rows);
    rlx_model_free(voice);
    rlx_model_free(ear);
    rlx_model_free(embedder);
    return 0;
}
