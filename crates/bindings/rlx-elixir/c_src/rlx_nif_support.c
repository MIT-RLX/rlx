/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The hand-written half of the Elixir NIF: term <-> C conversion, results,
 * and Optimizer.step.
 */
#include <string.h>

#include "rlx_nif.h"

/* The optimizer's resource type, opened by rlx_nif.c's load. */
extern ErlNifResourceType* RES_optimizer;

int rlx_ex_load(ErlNifEnv* env) {
    (void)env;
    return 0;
}

ERL_NIF_TERM rlx_ex_ok(ErlNifEnv* env, ERL_NIF_TERM t) {
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), t);
}

ERL_NIF_TERM rlx_ex_error(ErlNifEnv* env, int32_t rc) {
    static const char* kinds[] = {"type", "range", "runtime", "panic", "not_found"};
    const char* kind = rc <= -1 && rc >= -5 ? kinds[-rc - 1] : "other";
    return enif_make_tuple3(env, enif_make_atom(env, "error"), enif_make_atom(env, kind),
                            rlx_ex_cstring(env, rlx_last_error()));
}

ERL_NIF_TERM rlx_ex_badarg(ErlNifEnv* env, int index, const char* what) {
    char msg[128];
    snprintf(msg, sizeof msg, "argument %d must be %s", index + 1, what);
    return enif_make_tuple3(env, enif_make_atom(env, "error"), enif_make_atom(env, "type"), rlx_ex_cstring(env, msg));
}

/* ── arguments ───────────────────────────────────────────────── */

static bool is_nil(ErlNifEnv* env, ERL_NIF_TERM t) {
    char a[8];
    return enif_get_atom(env, t, a, sizeof a, ERL_NIF_LATIN1) && strcmp(a, "nil") == 0;
}

RlxExStr rlx_ex_cstr(ErlNifEnv* env, ERL_NIF_TERM t) {
    RlxExStr r = {NULL, true};
    if (is_nil(env, t)) return r;
    ErlNifBinary b;
    if (!enif_inspect_iolist_as_binary(env, t, &b)) { r.ok = false; return r; }
    r.s = (char*)malloc(b.size + 1);
    memcpy(r.s, b.data, b.size);
    r.s[b.size] = 0;
    return r;
}

bool rlx_ex_bool(ErlNifEnv* env, ERL_NIF_TERM t) {
    char a[8];
    return enif_get_atom(env, t, a, sizeof a, ERL_NIF_LATIN1) && strcmp(a, "true") == 0;
}

double rlx_ex_double(ErlNifEnv* env, ERL_NIF_TERM t) {
    double d;
    ErlNifSInt64 i;
    if (enif_get_double(env, t, &d)) return d;
    if (enif_get_int64(env, t, &i)) return (double)i;
    return 0;
}

int64_t rlx_ex_i64(ErlNifEnv* env, ERL_NIF_TERM t) {
    ErlNifSInt64 i = 0;
    double d;
    if (enif_get_int64(env, t, &i)) return i;
    if (enif_get_double(env, t, &d)) return (int64_t)d;
    return 0;
}

uint64_t rlx_ex_u64(ErlNifEnv* env, ERL_NIF_TERM t) {
    ErlNifUInt64 u = 0;
    if (enif_get_uint64(env, t, &u)) return u;
    return (uint64_t)rlx_ex_i64(env, t);
}

#define LIST_OF(T, name, get)                                               \
    RlxExArr name(ErlNifEnv* env, ERL_NIF_TERM t) {                         \
        RlxExArr r = {NULL, true};                                          \
        unsigned n;                                                         \
        if (is_nil(env, t)) return r;                                       \
        if (!enif_get_list_length(env, t, &n)) { r.ok = false; return r; }  \
        if (n == 0) return r;                                               \
        T* out = (T*)malloc(sizeof(T) * n);                                 \
        ERL_NIF_TERM h, tail = t;                                           \
        for (unsigned k = 0; enif_get_list_cell(env, tail, &h, &tail); k++) \
            out[k] = (T)get(env, h);                                        \
        r.p = out;                                                          \
        return r;                                                           \
    }
LIST_OF(float, rlx_ex_floats, rlx_ex_double)
LIST_OF(uint32_t, rlx_ex_u32s_in, rlx_ex_u64)
LIST_OF(int64_t, rlx_ex_i64s_in, rlx_ex_i64)
LIST_OF(uint64_t, rlx_ex_u64s_in, rlx_ex_u64)

static RlxExList list_of(ErlNifEnv* env, ERL_NIF_TERM t, void* (*each)(ErlNifEnv*, ERL_NIF_TERM, bool*)) {
    RlxExList l = {NULL, 0, true};
    if (is_nil(env, t)) return l;
    if (!enif_get_list_length(env, t, &l.n)) { l.ok = false; return l; }
    if (l.n == 0) return l;
    l.ptrs = (void**)calloc(l.n, sizeof(void*));
    ERL_NIF_TERM h, tail = t;
    for (unsigned k = 0; enif_get_list_cell(env, tail, &h, &tail); k++) {
        bool ok = true;
        l.ptrs[k] = each(env, h, &ok);
        if (!ok) l.ok = false;
    }
    return l;
}

static void* each_string(ErlNifEnv* env, ERL_NIF_TERM t, bool* ok) {
    RlxExStr s = rlx_ex_cstr(env, t);
    *ok = s.ok && s.s;
    return s.s;
}

static void* each_floats(ErlNifEnv* env, ERL_NIF_TERM t, bool* ok) {
    RlxExArr a = rlx_ex_floats(env, t);
    *ok = a.ok;
    return a.p;
}

static void* each_binary(ErlNifEnv* env, ERL_NIF_TERM t, bool* ok) {
    ErlNifBinary b;
    if (!enif_inspect_binary(env, t, &b)) { *ok = false; return NULL; }
    void* p = malloc(b.size ? b.size : 1);
    memcpy(p, b.data, b.size);
    return p;
}

RlxExList rlx_ex_strings(ErlNifEnv* env, ERL_NIF_TERM t) { return list_of(env, t, each_string); }
RlxExList rlx_ex_float_lists(ErlNifEnv* env, ERL_NIF_TERM t) { return list_of(env, t, each_floats); }
RlxExList rlx_ex_binaries(ErlNifEnv* env, ERL_NIF_TERM t) { return list_of(env, t, each_binary); }

void rlx_ex_list_free(RlxExList* l) {
    for (unsigned k = 0; k < l->n; k++) free(l->ptrs[k]);
    free(l->ptrs);
}

bool rlx_ex_ptr(ErlNifEnv* env, ErlNifResourceType* type, ERL_NIF_TERM t, void** out) {
    if (is_nil(env, t)) { *out = NULL; return true; }
    void** p = NULL;
    if (!enif_get_resource(env, t, type, (void**)&p) || !*p) return false;
    *out = *p;
    return true;
}

/* ── results ─────────────────────────────────────────────────── */

ERL_NIF_TERM rlx_ex_handle(ErlNifEnv* env, ErlNifResourceType* type, void* p) {
    void** obj = (void**)enif_alloc_resource(type, sizeof(void*));
    *obj = p;
    ERL_NIF_TERM t = enif_make_resource(env, obj);
    enif_release_resource(obj);
    return t;
}

ERL_NIF_TERM rlx_ex_cstring(ErlNifEnv* env, const char* s) {
    ERL_NIF_TERM t;
    size_t n = strlen(s);
    memcpy(enif_make_new_binary(env, n, &t), s, n);
    return t;
}

ERL_NIF_TERM rlx_ex_string(ErlNifEnv* env, char* s) {
    if (!s) return enif_make_atom(env, "nil");
    ERL_NIF_TERM t = rlx_ex_cstring(env, s);
    rlx_string_free(s);
    return t;
}

static ERL_NIF_TERM float_list(ErlNifEnv* env, const float* d, size_t n) {
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (size_t k = n; k > 0; k--) list = enif_make_list_cell(env, enif_make_double(env, d[k - 1]), list);
    return list;
}

ERL_NIF_TERM rlx_ex_f32s(ErlNifEnv* env, RlxF32s v) {
    ERL_NIF_TERM t = float_list(env, v.ptr, v.len);
    rlx_f32s_free(v);
    return t;
}

ERL_NIF_TERM rlx_ex_bytes(ErlNifEnv* env, RlxBytes v) {
    ERL_NIF_TERM t;
    if (v.len) memcpy(enif_make_new_binary(env, v.len, &t), v.ptr, v.len);
    else enif_make_new_binary(env, 0, &t);
    rlx_bytes_free(v);
    return t;
}

ERL_NIF_TERM rlx_ex_u32s(ErlNifEnv* env, RlxU32s v) {
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (size_t k = v.len; k > 0; k--) list = enif_make_list_cell(env, enif_make_uint(env, v.ptr[k - 1]), list);
    rlx_u32s_free(v);
    return list;
}

ERL_NIF_TERM rlx_ex_outputs(ErlNifEnv* env, RlxOutputs* o) {
    size_t n = rlx_outputs_len(o);
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (size_t i = n; i > 0; i--) {
        const float* d = NULL;
        size_t len = 0;
        rlx_outputs_get(o, i - 1, &d, &len);
        list = enif_make_list_cell(env, float_list(env, d, len), list);
    }
    rlx_outputs_free(o);
    return list;
}

ERL_NIF_TERM rlx_ex_typed_outputs(ErlNifEnv* env, RlxTypedOutputs* o) {
    size_t n = rlx_typed_outputs_len(o);
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (size_t i = n; i > 0; i--) {
        const uint8_t* d = NULL;
        size_t len = 0;
        const char* dt = NULL;
        rlx_typed_outputs_get(o, i - 1, &d, &len, &dt);
        ERL_NIF_TERM data;
        unsigned char* buf = enif_make_new_binary(env, len, &data);
        if (len) memcpy(buf, d, len);
        list = enif_make_list_cell(env, enif_make_tuple2(env, data, rlx_ex_cstring(env, dt)), list);
    }
    rlx_typed_outputs_free(o);
    return list;
}

ERL_NIF_TERM rlx_ex_tensors(ErlNifEnv* env, RlxTensorMap* m) {
    size_t n = rlx_tensor_map_len(m);
    ERL_NIF_TERM map = enif_make_new_map(env);
    for (size_t i = 0; i < n; i++) {
        const char* name = NULL;
        const float* d = NULL;
        size_t len = 0;
        rlx_tensor_map_get(m, i, &name, &d, &len);
        enif_make_map_put(env, map, rlx_ex_cstring(env, name), float_list(env, d, len), &map);
    }
    rlx_tensor_map_free(m);
    return map;
}

/* optimizer_step(optimizer, names, params, grads) → {:ok, updated param lists}. */
ERL_NIF_TERM rlx_ex_optimizer_step(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    void* h = NULL;
    if (!rlx_ex_ptr(env, RES_optimizer, argv[0], &h) || !h) return rlx_ex_badarg(env, 0, "an rlx Optimizer");
    RlxExList names = rlx_ex_strings(env, argv[1]);
    RlxExList ps = rlx_ex_float_lists(env, argv[2]);
    RlxExList gs = rlx_ex_float_lists(env, argv[3]);
    unsigned n = names.n;
    size_t* lens = (size_t*)calloc(n ? n : 1, sizeof(size_t));
    ERL_NIF_TERM hd, tail = argv[2];
    for (unsigned k = 0; enif_get_list_cell(env, tail, &hd, &tail); k++) {
        unsigned len = 0;
        enif_get_list_length(env, hd, &len);
        lens[k] = len;
    }
    int32_t rc = rlx_optimizer_step((RlxOptimizer*)h, (const char* const*)names.ptrs, (float* const*)ps.ptrs,
                                    (const float* const*)gs.ptrs, lens, NULL, NULL, n);
    ERL_NIF_TERM out = enif_make_list(env, 0);
    if (rc == RLX_OK)
        for (unsigned k = n; k > 0; k--) out = enif_make_list_cell(env, float_list(env, (float*)ps.ptrs[k - 1], lens[k - 1]), out);
    free(lens);
    rlx_ex_list_free(&names);
    rlx_ex_list_free(&ps);
    rlx_ex_list_free(&gs);
    return rc == RLX_OK ? rlx_ex_ok(env, out) : rlx_ex_error(env, rc);
}
