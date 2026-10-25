/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The hand-written half of the OCaml stubs: value <-> C conversion, errors,
 * and Optimizer.step.
 */
#include <string.h>

#include <caml/callback.h>

#include "rlx_ml.h"

/* Raise Rlx.Error (kind, message) (registered by rlx.ml as "rlx_error"): kind
 * is the constant constructor Type_error … Other_error, by index. */
void rlx_ml_raise(int32_t rc, const char* msg) {
    const value* exn = caml_named_value("rlx_error");
    if (!exn) caml_failwith(msg);
    value args[2];
    args[0] = Val_int(rc <= -1 && rc >= -5 ? -rc - 1 : 5);
    args[1] = caml_copy_string(msg);
    caml_raise_with_args(*exn, 2, args);
}

void rlx_ml_check(int32_t rc) {
    if (rc != RLX_OK) rlx_ml_raise(rc, rlx_last_error());
}

/* A handle, or NULL for None. */
void* rlx_ml_ptr(value opt) {
    if (Is_none(opt)) return NULL;
    void* p = *(void**)Data_custom_val(Some_val(opt));
    if (!p) rlx_ml_raise(RLX_ERR_RUNTIME, "this rlx object was closed");
    return p;
}

const char* rlx_ml_cstr(value opt) {
    return Is_none(opt) ? NULL : String_val(Some_val(opt));
}

float* rlx_ml_floats(value arr) {
    mlsize_t n = Wosize_val(arr) / Double_wosize;
    if (n == 0) return NULL;
    float* out = (float*)malloc(sizeof(float) * n);
    for (mlsize_t k = 0; k < n; k++) out[k] = (float)Double_flat_field(arr, k);
    return out;
}

const uint8_t* rlx_ml_bytes_in(value s) {
    return caml_string_length(s) ? (const uint8_t*)String_val(s) : NULL;
}

#define INTS(T, name)                                                       \
    T* name(value arr) {                                                    \
        mlsize_t n = Wosize_val(arr);                                       \
        if (n == 0) return NULL;                                            \
        T* out = (T*)malloc(sizeof(T) * n);                                 \
        for (mlsize_t k = 0; k < n; k++) out[k] = (T)Long_val(Field(arr, k)); \
        return out;                                                         \
    }
INTS(uint32_t, rlx_ml_u32s_in)
INTS(int64_t, rlx_ml_i64s_in)
INTS(uint64_t, rlx_ml_u64s_in)

const char** rlx_ml_strings(value arr) {
    mlsize_t n = Wosize_val(arr);
    if (n == 0) return NULL;
    const char** out = (const char**)malloc(sizeof(char*) * n);
    for (mlsize_t k = 0; k < n; k++) out[k] = String_val(Field(arr, k));
    return out;
}

const uint8_t** rlx_ml_byte_arrays(value arr) {
    return (const uint8_t**)rlx_ml_strings(arr);
}

RlxMlArrays rlx_ml_float_arrays(value arr) {
    RlxMlArrays a = {NULL, Wosize_val(arr)};
    if (a.n == 0) return a;
    a.ptrs = (float**)calloc(a.n, sizeof(float*));
    for (size_t k = 0; k < a.n; k++) a.ptrs[k] = rlx_ml_floats(Field(arr, k));
    return a;
}

void rlx_ml_arrays_free(RlxMlArrays* a) {
    for (size_t k = 0; k < a->n; k++) free(a->ptrs[k]);
    free(a->ptrs);
}

/* ── results ─────────────────────────────────────────────────── */

value rlx_ml_handle(void* p, struct custom_operations* ops) {
    if (!p) rlx_ml_raise(RLX_ERR_RUNTIME, "rlx returned NULL");
    value v = caml_alloc_custom(ops, sizeof(void*), 0, 1);
    *(void**)Data_custom_val(v) = p;
    return v;
}

value rlx_ml_string_opt(char* s) {
    CAMLparam0();
    CAMLlocal1(str);
    if (!s) CAMLreturn(Val_none);
    str = caml_copy_string(s);
    rlx_string_free(s);
    CAMLreturn(caml_alloc_some(str));
}

static value doubles(const float* d, size_t n) {
    value arr = caml_alloc(n * Double_wosize, Double_array_tag);
    for (size_t k = 0; k < n; k++) Store_double_flat_field(arr, k, d[k]);
    return n ? arr : Atom(0);
}

value rlx_ml_f32s(RlxF32s v) {
    value r = doubles(v.ptr, v.len);
    rlx_f32s_free(v);
    return r;
}

value rlx_ml_bytes(RlxBytes v) {
    value r = caml_alloc_initialized_string(v.len, (const char*)v.ptr);
    rlx_bytes_free(v);
    return r;
}

value rlx_ml_u32s(RlxU32s v) {
    CAMLparam0();
    CAMLlocal1(arr);
    arr = v.len ? caml_alloc(v.len, 0) : Atom(0);
    for (size_t k = 0; k < v.len; k++) Store_field(arr, k, Val_long(v.ptr[k]));
    rlx_u32s_free(v);
    CAMLreturn(arr);
}

value rlx_ml_outputs(RlxOutputs* o) {
    CAMLparam0();
    CAMLlocal2(arr, x);
    size_t n = rlx_outputs_len(o);
    arr = n ? caml_alloc(n, 0) : Atom(0);
    for (size_t i = 0; i < n; i++) {
        const float* d = NULL;
        size_t len = 0;
        rlx_outputs_get(o, i, &d, &len);
        x = doubles(d, len);
        Store_field(arr, i, x);
    }
    rlx_outputs_free(o);
    CAMLreturn(arr);
}

value rlx_ml_typed_outputs(RlxTypedOutputs* o) {
    CAMLparam0();
    CAMLlocal4(arr, t, d, s);
    size_t n = rlx_typed_outputs_len(o);
    arr = n ? caml_alloc(n, 0) : Atom(0);
    for (size_t i = 0; i < n; i++) {
        const uint8_t* data = NULL;
        size_t len = 0;
        const char* dt = NULL;
        rlx_typed_outputs_get(o, i, &data, &len, &dt);
        d = caml_alloc_initialized_string(len, (const char*)data);
        s = caml_copy_string(dt);
        t = caml_alloc_tuple(2);
        Store_field(t, 0, d);
        Store_field(t, 1, s);
        Store_field(arr, i, t);
    }
    rlx_typed_outputs_free(o);
    CAMLreturn(arr);
}

value rlx_ml_tensors(RlxTensorMap* m) {
    CAMLparam0();
    CAMLlocal4(arr, t, nm, x);
    size_t n = rlx_tensor_map_len(m);
    arr = n ? caml_alloc(n, 0) : Atom(0);
    for (size_t i = 0; i < n; i++) {
        const char* name = NULL;
        const float* d = NULL;
        size_t len = 0;
        rlx_tensor_map_get(m, i, &name, &d, &len);
        nm = caml_copy_string(name);
        x = doubles(d, len);
        t = caml_alloc_tuple(2);
        Store_field(t, 0, nm);
        Store_field(t, 1, x);
        Store_field(arr, i, t);
    }
    rlx_tensor_map_free(m);
    CAMLreturn(arr);
}

/* optimizer_step(handle, names, params, grads) → the updated params. */
value caml_rlx_support_optimizer_step(value h, value names, value params, value grads) {
    CAMLparam4(h, names, params, grads);
    CAMLlocal2(out, x);
    size_t n = Wosize_val(names);
    const char** cn = rlx_ml_strings(names);
    RlxMlArrays ps = rlx_ml_float_arrays(params);
    RlxMlArrays gs = rlx_ml_float_arrays(grads);
    size_t* lens = (size_t*)calloc(n ? n : 1, sizeof(size_t));
    for (size_t k = 0; k < n; k++) lens[k] = Wosize_val(Field(params, k)) / Double_wosize;
    int32_t rc = rlx_optimizer_step((RlxOptimizer*)rlx_ml_ptr(h), cn, ps.ptrs, (const float* const*)gs.ptrs,
                                    lens, NULL, NULL, n);
    if (rc == RLX_OK) {
        out = n ? caml_alloc(n, 0) : Atom(0);
        for (size_t k = 0; k < n; k++) {
            x = doubles(ps.ptrs[k], lens[k]);
            Store_field(out, k, x);
        }
    }
    free(lens);
    free((void*)cn);
    rlx_ml_arrays_free(&ps);
    rlx_ml_arrays_free(&gs);
    rlx_ml_check(rc);
    CAMLreturn(out);
}
