/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Drives the C ABI end to end: build a loss graph with the generated
 * builders, compile and run it, differentiate it, train it, and check the
 * error path. Prints one line per check; any failure exits non-zero.
 */
#include <math.h>
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

/* loss = mean((x·w - t)²) over x [1, 2], w [2, 2], t [1, 2]. */
static RlxGraph *loss_graph(uint32_t *w_out) {
    RlxGraph *g = NULL;
    OK(rlx_graph_new("mse", &g));
    size_t shape[2] = {1, 2}, wshape[2] = {2, 2};
    uint32_t x, t, w, y, d, sq, loss;
    OK(rlx_graph_input(g, "x", shape, 2, NULL, &x));
    OK(rlx_graph_input(g, "t", shape, 2, NULL, &t));
    OK(rlx_graph_param(g, "w", wshape, 2, "f32", &w));
    OK(rlx_graph_matmul(g, x, w, &y));
    OK(rlx_graph_sub(g, y, t, &d));
    OK(rlx_graph_mul(g, d, d, &sq));
    size_t axes[2] = {0, 1};
    OK(rlx_graph_mean(g, sq, axes, 2, false, &loss));
    OK(rlx_graph_set_outputs(g, &loss, 1));
    *w_out = w;
    return g;
}

static int batch_fn(void *user, uint64_t step, RlxBatch *batch) {
    (void)user;
    (void)step;
    static const float x[2] = {1, 1}, t[2] = {1, -1};
    if (rlx_batch_set(batch, "x", x, 2) != RLX_OK) return -1;
    if (rlx_batch_set(batch, "t", t, 2) != RLX_OK) return -1;
    return 1;
}

static int on_step(void *user, uint64_t step, float loss) {
    (void)step;
    *(float *)user = loss;
    return 1;
}

int main(void) {
    uint32_t w;
    RlxGraph *g = loss_graph(&w);

    /* Shape inference reads back through the generated builder. */
    char *shape_json = NULL;
    OK(rlx_graph_shape_of(g, w, &shape_json));
    printf("shape_of w: %s\n", shape_json);
    rlx_string_free(shape_json);

    /* Forward: compile a copy (compile consumes), run on CPU. */
    RlxGraph *copy = NULL;
    OK(rlx_graph_clone(g, &copy));
    RlxSession *s = NULL;
    OK(rlx_session_new("cpu", NULL, NULL, &s));
    RlxCompiled *c = NULL;
    OK(rlx_session_compile(s, copy, NULL, &c));
    const float wv[4] = {1, 0, 0, 1};
    OK(rlx_compiled_set_param(c, "w", wv, 4));
    const float x[2] = {1, 2}, t[2] = {0, 0};
    const char *names[2] = {"x", "t"};
    const float *data[2] = {x, t};
    size_t lens[2] = {2, 2};
    RlxOutputs *outs = NULL;
    OK(rlx_compiled_run(c, names, data, lens, 2, &outs));
    const float *loss;
    size_t loss_len;
    OK(rlx_outputs_get(outs, 0, &loss, &loss_len));
    printf("forward loss: %.4f\n", loss[0]); /* mean(1², 2²) = 2.5 */
    rlx_outputs_free(outs);
    rlx_compiled_free(c);

    /* Reverse mode: [loss, dloss/dw], seeded through `d_output`. */
    RlxGraph *bwd = NULL;
    OK(rlx_grad(g, &w, 1, &bwd));
    OK(rlx_session_compile(s, bwd, NULL, &c));
    OK(rlx_compiled_set_param(c, "w", wv, 4));
    const float seed[1] = {1};
    const char *bnames[3] = {"x", "t", "d_output"};
    const float *bdata[3] = {x, t, seed};
    size_t blens[3] = {2, 2, 1};
    OK(rlx_compiled_run(c, bnames, bdata, blens, 3, &outs));
    const float *dw;
    size_t dw_len;
    OK(rlx_outputs_get(outs, 1, &dw, &dw_len));
    /* d mean(d²)/dw = x^T · d = [[1],[2]]·[1, 2] = [[1, 2], [2, 4]]. */
    printf("grad w: %.1f %.1f %.1f %.1f\n", dw[0], dw[1], dw[2], dw[3]);
    rlx_outputs_free(outs);
    rlx_compiled_free(c);
    rlx_graph_free(bwd);

    /* Errors carry a status and a message, never a crash. */
    uint32_t bad;
    int32_t rc = rlx_graph_matmul(g, 999, w, &bad);
    printf("bad node: %s %s\n", rc == RLX_ERR_RANGE ? "RLX_ERR_RANGE" : "WRONG",
           strstr(rlx_last_error(), "does not exist") ? "named" : "unnamed");

    /* Static diagnostics. */
    char *report = NULL;
    OK(rlx_check(g, "{\"backends\": [\"cpu\"]}", &report));
    printf("check: %s\n", strstr(report, "\"errors\":0") ? "no errors" : report);
    rlx_string_free(report);

    /* Training: the graph is consumed; init gives w its starting value. */
    const char *wrt[1] = {"w"};
    const char *init_names[1] = {"w"};
    const float zeros[4] = {0, 0, 0, 0};
    const float *init_data[1] = {zeros};
    size_t init_lens[1] = {4};
    RlxTrainer *tr = NULL;
    OK(rlx_trainer_new(g, wrt, 1, init_names, init_data, init_lens, 1,
                       "{\"optimizer\": {\"kind\": \"adamw\", \"lr\": 0.1}}", &tr));
    float last = NAN;
    char *summary = NULL;
    OK(rlx_trainer_run(tr, 200, batch_fn, NULL, on_step, 1, &last, &summary));
    printf("trained: %s (summary has steps=%s)\n", last < 1e-4f ? "converged" : "NOT converged",
           strstr(summary, "\"steps\":200") ? "200" : summary);
    rlx_string_free(summary);
    uint64_t steps;
    OK(rlx_trainer_steps(tr, &steps));
    printf("steps: %llu\n", (unsigned long long)steps);
    rlx_trainer_free(tr);

    /* The graph handle survived as an empty shell. */
    printf("consumed: %s\n", rlx_graph_is_consumed(g) ? "yes" : "no");
    rlx_graph_free(g);
    rlx_session_free(s);
    printf("version: %s\n", rlx_version());
    return 0;
}
