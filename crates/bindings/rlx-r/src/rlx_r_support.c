/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The hand-written half of the rlx R package's C code: SEXP <-> C conversion,
 * and the support entries the R side calls — the last error, reading run
 * outputs and tensor maps, and Optimizer$step.
 */
#include <string.h>

#include "rlx_r.h"

/* ── arguments ───────────────────────────────────────────────── */

static R_xlen_t len(SEXP x) {
    return Rf_isNull(x) ? 0 : XLENGTH(x);
}

static double elem(SEXP x, R_xlen_t k) {
    switch (TYPEOF(x)) {
    case REALSXP: return REAL(x)[k];
    case INTSXP: return INTEGER(x)[k];
    case LGLSXP: return LOGICAL(x)[k];
    case RAWSXP: return RAW(x)[k];
    default: Rf_error("rlx: expected numbers, got a %s", Rf_type2char(TYPEOF(x)));
    }
    return 0;
}

double rlx_r_number(SEXP x) {
    if (len(x) != 1) Rf_error("rlx: expected a single number");
    return elem(x, 0);
}

void* rlx_r_ptr(SEXP x) {
    if (Rf_isNull(x)) return NULL;
    if (TYPEOF(x) != EXTPTRSXP) Rf_error("rlx: expected an rlx object");
    void* p = R_ExternalPtrAddr(x);
    if (!p) Rf_error("rlx: this object was closed");
    return p;
}

const char* rlx_r_string(SEXP x) {
    if (Rf_isNull(x)) return NULL;
    if (!Rf_isString(x) || len(x) != 1) Rf_error("rlx: expected a single string");
    return Rf_translateCharUTF8(STRING_ELT(x, 0));
}

#define ARRAY(T, name)                                                      \
    const T* name(SEXP x) {                                                 \
        R_xlen_t n = len(x);                                                \
        if (n == 0) return NULL;                                            \
        T* out = (T*)R_alloc(n, sizeof(T));                                 \
        for (R_xlen_t k = 0; k < n; k++) out[k] = (T)elem(x, k);            \
        return out;                                                         \
    }
ARRAY(float, rlx_r_floats)
ARRAY(uint32_t, rlx_r_u32s)
ARRAY(int64_t, rlx_r_i64s)
ARRAY(uint64_t, rlx_r_u64s)

const uint8_t* rlx_r_bytes(SEXP x) {
    if (len(x) == 0) return NULL;
    if (TYPEOF(x) == RAWSXP) return RAW(x);
    R_xlen_t n = len(x);
    uint8_t* out = (uint8_t*)R_alloc(n, 1);
    for (R_xlen_t k = 0; k < n; k++) out[k] = (uint8_t)elem(x, k);
    return out;
}

const char* const* rlx_r_strings(SEXP x) {
    R_xlen_t n = len(x);
    if (n == 0) return NULL;
    if (!Rf_isString(x)) Rf_error("rlx: expected a character vector");
    const char** out = (const char**)R_alloc(n, sizeof(char*));
    for (R_xlen_t k = 0; k < n; k++) out[k] = Rf_translateCharUTF8(STRING_ELT(x, k));
    return out;
}

const float* const* rlx_r_float_arrays(SEXP x) {
    R_xlen_t n = len(x);
    if (n == 0) return NULL;
    if (TYPEOF(x) != VECSXP) Rf_error("rlx: expected a list of numeric vectors");
    const float** out = (const float**)R_alloc(n, sizeof(float*));
    for (R_xlen_t k = 0; k < n; k++) out[k] = rlx_r_floats(VECTOR_ELT(x, k));
    return out;
}

const uint8_t* const* rlx_r_byte_arrays(SEXP x) {
    R_xlen_t n = len(x);
    if (n == 0) return NULL;
    if (TYPEOF(x) != VECSXP) Rf_error("rlx: expected a list of raw vectors");
    const uint8_t** out = (const uint8_t**)R_alloc(n, sizeof(uint8_t*));
    for (R_xlen_t k = 0; k < n; k++) out[k] = rlx_r_bytes(VECTOR_ELT(x, k));
    return out;
}

/* ── results ─────────────────────────────────────────────────── */

static SEXP reals(const float* d, size_t n) {
    SEXP v = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
    for (size_t k = 0; k < n; k++) REAL(v)[k] = d[k];
    UNPROTECT(1);
    return v;
}

SEXP rlx_r_from_f32s(RlxF32s v) {
    SEXP r = reals(v.ptr, v.len);
    rlx_f32s_free(v);
    return r;
}

SEXP rlx_r_from_bytes(RlxBytes v) {
    SEXP r = Rf_allocVector(RAWSXP, (R_xlen_t)v.len);
    if (v.len) memcpy(RAW(r), v.ptr, v.len);
    rlx_bytes_free(v);
    return r;
}

SEXP rlx_r_from_u32s(RlxU32s v) {
    SEXP r = Rf_allocVector(REALSXP, (R_xlen_t)v.len);
    for (size_t k = 0; k < v.len; k++) REAL(r)[k] = v.ptr[k];
    rlx_u32s_free(v);
    return r;
}

SEXP rlx_r_from_string(char* s) {
    if (!s) return R_NilValue;
    SEXP r = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(r, 0, Rf_mkCharCE(s, CE_UTF8));
    rlx_string_free(s);
    UNPROTECT(1);
    return r;
}

SEXP rlx_r_handle(void* p, R_CFinalizer_t fin) {
    SEXP x = PROTECT(R_MakeExternalPtr(p, R_NilValue, R_NilValue));
    if (fin) R_RegisterCFinalizerEx(x, fin, TRUE);
    UNPROTECT(1);
    return x;
}

/* ── support entries (.Call(R_support_…)) ────────────────────── */

SEXP R_support_last_error(void) {
    return Rf_mkString(rlx_last_error());
}

/* take_outputs(ptr) → a list of numeric vectors; frees the outputs. */
SEXP R_support_take_outputs(SEXP p) {
    RlxOutputs* o = (RlxOutputs*)rlx_r_ptr(p);
    size_t n = rlx_outputs_len(o);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    for (size_t i = 0; i < n; i++) {
        const float* d = NULL;
        size_t k = 0;
        rlx_outputs_get(o, i, &d, &k);
        SET_VECTOR_ELT(out, (R_xlen_t)i, reals(d, k));
    }
    rlx_outputs_free(o);
    R_ClearExternalPtr(p);
    UNPROTECT(1);
    return out;
}

/* take_typed_outputs(ptr) → a list of list(data = raw, dtype = chr). */
SEXP R_support_take_typed_outputs(SEXP p) {
    RlxTypedOutputs* o = (RlxTypedOutputs*)rlx_r_ptr(p);
    size_t n = rlx_typed_outputs_len(o);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    for (size_t i = 0; i < n; i++) {
        const uint8_t* d = NULL;
        size_t k = 0;
        const char* dt = NULL;
        rlx_typed_outputs_get(o, i, &d, &k, &dt);
        SEXP t = PROTECT(Rf_allocVector(VECSXP, 2));
        SEXP data = Rf_allocVector(RAWSXP, (R_xlen_t)k);
        SET_VECTOR_ELT(t, 0, data);
        if (k) memcpy(RAW(data), d, k);
        SET_VECTOR_ELT(t, 1, Rf_mkString(dt));
        SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
        SET_STRING_ELT(names, 0, Rf_mkChar("data"));
        SET_STRING_ELT(names, 1, Rf_mkChar("dtype"));
        Rf_setAttrib(t, R_NamesSymbol, names);
        SET_VECTOR_ELT(out, (R_xlen_t)i, t);
        UNPROTECT(2);
    }
    rlx_typed_outputs_free(o);
    R_ClearExternalPtr(p);
    UNPROTECT(1);
    return out;
}

/* take_tensors(ptr) → a named list of numeric vectors. */
SEXP R_support_take_tensors(SEXP p) {
    RlxTensorMap* m = (RlxTensorMap*)rlx_r_ptr(p);
    size_t n = rlx_tensor_map_len(m);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)n));
    for (size_t i = 0; i < n; i++) {
        const char* name = NULL;
        const float* d = NULL;
        size_t k = 0;
        rlx_tensor_map_get(m, i, &name, &d, &k);
        SET_STRING_ELT(names, (R_xlen_t)i, Rf_mkCharCE(name, CE_UTF8));
        SET_VECTOR_ELT(out, (R_xlen_t)i, reals(d, k));
    }
    Rf_setAttrib(out, R_NamesSymbol, names);
    rlx_tensor_map_free(m);
    R_ClearExternalPtr(p);
    UNPROTECT(2);
    return out;
}

/* optimizer_step(handle, names, params, grads) → list(rc, updated params). */
SEXP R_support_optimizer_step(SEXP h, SEXP names, SEXP params, SEXP grads) {
    RlxOptimizer* o = (RlxOptimizer*)rlx_r_ptr(h);
    R_xlen_t n = len(names);
    if (len(params) != n || len(grads) != n) Rf_error("rlx: names, params and grads differ in length");
    float** ps = (float**)R_alloc(n ? n : 1, sizeof(float*));
    size_t* lens = (size_t*)R_alloc(n ? n : 1, sizeof(size_t));
    for (R_xlen_t k = 0; k < n; k++) {
        lens[k] = (size_t)len(VECTOR_ELT(params, k));
        if ((size_t)len(VECTOR_ELT(grads, k)) != lens[k])
            Rf_error("rlx: '%s' has %d values but its gradient %d", Rf_translateCharUTF8(STRING_ELT(names, k)),
                     (int)lens[k], (int)len(VECTOR_ELT(grads, k)));
        ps[k] = (float*)rlx_r_floats(VECTOR_ELT(params, k));
    }
    int32_t rc = rlx_optimizer_step(o, rlx_r_strings(names), ps, rlx_r_float_arrays(grads), lens, NULL, NULL,
                                    (size_t)n);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, Rf_ScalarInteger(rc));
    SEXP updated = Rf_allocVector(VECSXP, n);
    SET_VECTOR_ELT(out, 1, updated);
    for (R_xlen_t k = 0; k < n; k++) SET_VECTOR_ELT(updated, k, reals(ps[k], lens[k]));
    UNPROTECT(1);
    return out;
}
