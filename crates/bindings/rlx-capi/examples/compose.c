/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Combining and splitting graphs through the C ABI: splice two layers into
 * one graph, split it in two, run the parts in order (the boundary tensor
 * handed from one to the next by name), merge them back, and check the
 * error path. Prints one line per check; any failure exits non-zero.
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

/* y = x * w over [3]. */
static RlxGraph *layer(void) {
    RlxGraph *g = NULL;
    OK(rlx_graph_new("layer", &g));
    size_t shape[1] = {3};
    uint32_t x, w, y;
    OK(rlx_graph_input(g, "x", shape, 1, NULL, &x));
    OK(rlx_graph_param(g, "w", shape, 1, "f32", &w));
    OK(rlx_graph_mul(g, x, w, &y));
    OK(rlx_graph_set_outputs(g, &y, 1));
    return g;
}

static const float W0[3] = {2, 2, 2}, W1[3] = {1, 2, 3}, X[3] = {1, 2, 3};

/* Compile a copy of `g` on CPU, bind whichever of l0.w / l1.w it has, feed
 * `n` named inputs and copy output `k` into `out`. */
static void run(RlxSession *s, RlxGraph *g, const char **names, const float **data,
                size_t n, size_t k, float out[3]) {
    RlxGraph *copy = NULL;
    OK(rlx_graph_clone(g, &copy));
    RlxCompiled *c = NULL;
    OK(rlx_session_compile(s, copy, NULL, &c));
    char *params = NULL;
    OK(rlx_graph_param_names(g, &params));
    if (strstr(params, "\"l0.w\"")) OK(rlx_compiled_set_param(c, "l0.w", W0, 3));
    if (strstr(params, "\"l1.w\"")) OK(rlx_compiled_set_param(c, "l1.w", W1, 3));
    rlx_string_free(params);
    size_t lens[4] = {3, 3, 3, 3};
    RlxOutputs *outs = NULL;
    OK(rlx_compiled_run(c, names, data, lens, n, &outs));
    const float *y;
    size_t len;
    OK(rlx_outputs_get(outs, k, &y, &len));
    memcpy(out, y, 3 * sizeof(float));
    rlx_outputs_free(outs);
    rlx_compiled_free(c);
}

/* The first string of a JSON string array, into `buf`. */
static const char *first_name(const char *json, char *buf, size_t cap) {
    const char *a = strchr(json, '"'), *b = a ? strchr(a + 1, '"') : NULL;
    size_t n = (a && b) ? (size_t)(b - a - 1) : 0;
    if (n >= cap) n = cap - 1;
    memcpy(buf, a ? a + 1 : "", n);
    buf[n] = 0;
    return buf;
}

int main(void) {
    RlxSession *s = NULL;
    OK(rlx_session_new("cpu", NULL, NULL, &s));

    /* Two layers with their own weights: l0.w, l1.w. */
    RlxGraph *g = NULL, *l = layer();
    OK(rlx_graph_new("stack", &g));
    RlxU32s h0, h1;
    OK(rlx_splice(g, l, "{\"prefix\": \"l0.\"}", &h0));
    char opts[96];
    snprintf(opts, sizeof opts, "{\"prefix\": \"l1.\", \"bind\": {\"x\": %u}}", h0.ptr[0]);
    OK(rlx_splice(g, l, opts, &h1));
    OK(rlx_graph_set_outputs(g, h1.ptr, 1));

    const char *in_names[1] = {"l0.x"};
    const float *in_data[1] = {X};
    float whole[3];
    run(s, g, in_names, in_data, 1, 0, whole);
    printf("stacked: %.0f %.0f %.0f\n", whole[0], whole[1], whole[2]); /* 2 8 18 */

    /* Split after the first layer; run the parts in order. */
    RlxPartition *p = NULL;
    OK(rlx_partition_split_at(g, h0.ptr, 1, &p));
    size_t parts;
    OK(rlx_partition_num_parts(p, &parts));
    RlxGraph *head = NULL, *tail = NULL;
    OK(rlx_partition_graph(p, 0, &head));
    OK(rlx_partition_graph(p, 1, &tail));
    char *exports = NULL, *imports = NULL, boundary[64];
    OK(rlx_partition_exports(p, 0, &exports));
    OK(rlx_partition_imports(p, 1, &imports));
    printf("boundary: %s\n", strcmp(exports, imports) == 0 ? "matches" : imports);
    float mid[3], split[3];
    run(s, head, in_names, in_data, 1, 0, mid);
    const char *tail_names[1] = {first_name(imports, boundary, sizeof boundary)};
    const float *tail_data[1] = {mid};
    run(s, tail, tail_names, tail_data, 1, 0, split);
    printf("parts: %zu, split: %.0f %.0f %.0f\n", parts, split[0], split[1], split[2]);
    rlx_string_free(exports);
    rlx_string_free(imports);

    RlxGraph *merged = NULL;
    float again[3];
    OK(rlx_partition_merge(p, &merged));
    run(s, merged, in_names, in_data, 1, 0, again);
    printf("merged: %.0f %.0f %.0f\n", again[0], again[1], again[2]);

    /* The same layer twice, sharing `w`: x * w * w. */
    RlxGraph *twice = NULL;
    OK(rlx_sequential(l, l, NULL, &twice));
    char *shared = NULL;
    OK(rlx_graph_param_names(twice, &shared));
    printf("sequential params: %s\n", shared);
    rlx_string_free(shared);

    /* Errors name the problem and leave the target unchanged. */
    size_t before, after;
    OK(rlx_graph_node_count(g, &before));
    int32_t rc = rlx_splice(g, l, "{\"bind\": {\"nope\": 0}}", &h1);
    OK(rlx_graph_node_count(g, &after));
    printf("bad bind: %s %s %s\n", rc == RLX_ERR_TYPE ? "RLX_ERR_TYPE" : "WRONG",
           strstr(rlx_last_error(), "\"nope\"") ? "named" : "unnamed",
           before == after ? "unchanged" : "CHANGED");

    rlx_u32s_free(h0);
    rlx_u32s_free(h1);
    rlx_graph_free(twice);
    rlx_graph_free(merged);
    rlx_graph_free(head);
    rlx_graph_free(tail);
    rlx_partition_free(p);
    rlx_graph_free(l);
    rlx_graph_free(g);
    rlx_session_free(s);
    return 0;
}
