/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The hand-written half of the rlx_mex gateway: mexFunction (finds a function
 * by name), mxArray <-> C conversion, and the support functions the +rlx code
 * calls — the last error, reading run outputs and tensor maps, and
 * Optimizer.step.
 */
#include <stdlib.h>
#include <string.h>

#include "rlx_mex.h"

/* ── arguments ───────────────────────────────────────────────── */

void rlx_mx_arity(const char* name, int nlhs, int outs, int nrhs, int ins) {
    if (nrhs != ins)
        mexErrMsgIdAndTxt("rlx:type", "rlx_mex %s takes %d arguments, got %d", name, ins, nrhs);
    if (outs > (nlhs > 1 ? nlhs : 1) && outs > 0)
        mexErrMsgIdAndTxt("rlx:type", "rlx_mex %s returns %d values; ask for all of them", name, outs);
}

void rlx_mx_empty(mxArray** plhs, int outs) {
    for (int i = 1; i < outs; i++) plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

static double elem(const mxArray* m, size_t k) {
    const void* d = mxGetData(m);
    switch (mxGetClassID(m)) {
    case mxDOUBLE_CLASS: return ((const double*)d)[k];
    case mxSINGLE_CLASS: return ((const float*)d)[k];
    case mxINT8_CLASS: return ((const int8_t*)d)[k];
    case mxUINT8_CLASS: return ((const uint8_t*)d)[k];
    case mxINT16_CLASS: return ((const int16_t*)d)[k];
    case mxUINT16_CLASS: return ((const uint16_t*)d)[k];
    case mxINT32_CLASS: return ((const int32_t*)d)[k];
    case mxUINT32_CLASS: return ((const uint32_t*)d)[k];
    case mxINT64_CLASS: return (double)((const int64_t*)d)[k];
    case mxUINT64_CLASS: return (double)((const uint64_t*)d)[k];
    case mxLOGICAL_CLASS: return ((const mxLogical*)d)[k] ? 1.0 : 0.0;
    default: mexErrMsgIdAndTxt("rlx:type", "expected a number, got a %s", mxGetClassName(m));
    }
    return 0;
}

static size_t count(const mxArray* m) {
    return m ? mxGetNumberOfElements(m) : 0;
}

double rlx_mx_double(const mxArray* m) {
    if (count(m) != 1) mexErrMsgIdAndTxt("rlx:type", "expected a scalar");
    return elem(m, 0);
}

uint64_t rlx_mx_u64(const mxArray* m) {
    if (count(m) != 1) mexErrMsgIdAndTxt("rlx:type", "expected a scalar");
    if (mxIsUint64(m)) return *(const uint64_t*)mxGetData(m);
    double v = elem(m, 0);
    if (v < 0) mexErrMsgIdAndTxt("rlx:range", "expected a non-negative integer, got %g", v);
    return (uint64_t)v;
}

void* rlx_mx_ptr(const mxArray* m) {
    if (count(m) == 0) return NULL;
    if (!mxIsUint64(m)) mexErrMsgIdAndTxt("rlx:type", "expected an rlx handle");
    return (void*)(uintptr_t)(*(const uint64_t*)mxGetData(m));
}

char* rlx_mx_string(const mxArray* m) {
    if (!m || !mxIsChar(m)) return NULL; /* [] → none */
    return mxArrayToString(m);
}

#define ARRAY(T, name)                                                      \
    T* name(const mxArray* m) {                                             \
        size_t n = count(m);                                                \
        if (n == 0) return NULL;                                            \
        T* out = (T*)mxMalloc(sizeof(T) * n);                               \
        for (size_t k = 0; k < n; k++) out[k] = (T)elem(m, k);              \
        return out;                                                         \
    }
ARRAY(uint32_t, rlx_mx_u32s)
ARRAY(int64_t, rlx_mx_i64s)
ARRAY(uint64_t, rlx_mx_u64s)

float* rlx_mx_floats(const mxArray* m) {
    size_t n = count(m);
    if (n == 0) return NULL;
    float* out = (float*)mxMalloc(sizeof(float) * n);
    if (mxIsSingle(m)) {
        memcpy(out, mxGetData(m), sizeof(float) * n);
    } else if (mxIsDouble(m)) {
        const double* d = (const double*)mxGetData(m);
        for (size_t k = 0; k < n; k++) out[k] = (float)d[k];
    } else {
        for (size_t k = 0; k < n; k++) out[k] = (float)elem(m, k);
    }
    return out;
}

uint8_t* rlx_mx_bytes(const mxArray* m) {
    size_t n = count(m);
    if (n == 0) return NULL;
    uint8_t* out = (uint8_t*)mxMalloc(n);
    if (mxIsUint8(m) || mxIsInt8(m)) memcpy(out, mxGetData(m), n);
    else for (size_t k = 0; k < n; k++) out[k] = (uint8_t)elem(m, k);
    return out;
}

static RlxMxList list(const mxArray* m, void* (*each)(const mxArray*)) {
    RlxMxList l = {NULL, 0};
    if (count(m) == 0) return l;
    if (!mxIsCell(m)) {
        l.n = 1;
        l.ptrs = (void**)mxCalloc(1, sizeof(void*));
        l.ptrs[0] = each(m);
        return l;
    }
    l.n = mxGetNumberOfElements(m);
    l.ptrs = (void**)mxCalloc(l.n, sizeof(void*));
    for (size_t k = 0; k < l.n; k++) l.ptrs[k] = each(mxGetCell(m, k));
    return l;
}

static void* each_string(const mxArray* m) {
    if (!m || !mxIsChar(m)) mexErrMsgIdAndTxt("rlx:type", "expected a cell array of strings");
    return mxArrayToString(m);
}
static void* each_floats(const mxArray* m) { return rlx_mx_floats(m); }
static void* each_bytes(const mxArray* m) { return rlx_mx_bytes(m); }

RlxMxList rlx_mx_strings(const mxArray* m) { return list(m, each_string); }
RlxMxList rlx_mx_float_arrays(const mxArray* m) { return list(m, each_floats); }
RlxMxList rlx_mx_byte_arrays(const mxArray* m) { return list(m, each_bytes); }

void rlx_mx_list_free(RlxMxList* l) {
    for (size_t k = 0; k < l->n; k++) mxFree(l->ptrs[k]);
    mxFree(l->ptrs);
}

/* ── results ─────────────────────────────────────────────────── */

static mxArray* singles(const float* d, size_t n) {
    mxArray* m = mxCreateNumericMatrix(1, n, mxSINGLE_CLASS, mxREAL);
    if (n) memcpy(mxGetData(m), d, sizeof(float) * n);
    return m;
}

mxArray* rlx_mx_from_f32s(RlxF32s v) {
    mxArray* m = singles(v.ptr, v.len);
    rlx_f32s_free(v);
    return m;
}

mxArray* rlx_mx_from_bytes(RlxBytes v) {
    mxArray* m = mxCreateNumericMatrix(1, v.len, mxUINT8_CLASS, mxREAL);
    if (v.len) memcpy(mxGetData(m), v.ptr, v.len);
    rlx_bytes_free(v);
    return m;
}

mxArray* rlx_mx_from_u32s(RlxU32s v) {
    mxArray* m = mxCreateDoubleMatrix(1, v.len, mxREAL);
    double* d = (double*)mxGetData(m);
    for (size_t k = 0; k < v.len; k++) d[k] = v.ptr[k];
    rlx_u32s_free(v);
    return m;
}

mxArray* rlx_mx_from_string(char* s) {
    if (!s) return mxCreateDoubleMatrix(0, 0, mxREAL);
    mxArray* m = mxCreateString(s);
    rlx_string_free(s);
    return m;
}

mxArray* rlx_mx_from_ptr(void* p) {
    mxArray* m = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *(uint64_t*)mxGetData(m) = (uint64_t)(uintptr_t)p;
    return m;
}

/* ── support functions (rlx_mex('support_…', …)) ─────────────── */

static void s_last_error(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    (void)prhs;
    rlx_mx_arity("support_last_error", nlhs, 1, nrhs, 0);
    plhs[0] = mxCreateString(rlx_last_error());
}

/* take_outputs(ptr) → a cell row of single row vectors; frees the outputs. */
static void s_take_outputs(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    rlx_mx_arity("support_take_outputs", nlhs, 1, nrhs, 1);
    RlxOutputs* o = (RlxOutputs*)rlx_mx_ptr(prhs[0]);
    size_t n = rlx_outputs_len(o);
    mxArray* cell = mxCreateCellMatrix(1, n);
    for (size_t i = 0; i < n; i++) {
        const float* d = NULL;
        size_t len = 0;
        rlx_outputs_get(o, i, &d, &len);
        mxSetCell(cell, i, singles(d, len));
    }
    rlx_outputs_free(o);
    plhs[0] = cell;
}

/* take_typed_outputs(ptr) → a struct row with fields data (uint8) and dtype. */
static void s_take_typed_outputs(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    rlx_mx_arity("support_take_typed_outputs", nlhs, 1, nrhs, 1);
    RlxTypedOutputs* o = (RlxTypedOutputs*)rlx_mx_ptr(prhs[0]);
    size_t n = rlx_typed_outputs_len(o);
    const char* fields[] = {"data", "dtype"};
    mxArray* s = mxCreateStructMatrix(1, n, 2, fields);
    for (size_t i = 0; i < n; i++) {
        const uint8_t* d = NULL;
        size_t len = 0;
        const char* dt = NULL;
        rlx_typed_outputs_get(o, i, &d, &len, &dt);
        mxArray* data = mxCreateNumericMatrix(1, len, mxUINT8_CLASS, mxREAL);
        if (len) memcpy(mxGetData(data), d, len);
        mxSetField(s, i, "data", data);
        mxSetField(s, i, "dtype", mxCreateString(dt));
    }
    rlx_typed_outputs_free(o);
    plhs[0] = s;
}

/* take_tensors(ptr) → [names (cell row), values (cell row of singles)]. */
static void s_take_tensors(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    rlx_mx_arity("support_take_tensors", nlhs, 2, nrhs, 1);
    RlxTensorMap* m = (RlxTensorMap*)rlx_mx_ptr(prhs[0]);
    size_t n = rlx_tensor_map_len(m);
    mxArray* names = mxCreateCellMatrix(1, n);
    mxArray* values = mxCreateCellMatrix(1, n);
    for (size_t i = 0; i < n; i++) {
        const char* name = NULL;
        const float* d = NULL;
        size_t len = 0;
        rlx_tensor_map_get(m, i, &name, &d, &len);
        mxSetCell(names, i, mxCreateString(name));
        mxSetCell(values, i, singles(d, len));
    }
    rlx_tensor_map_free(m);
    plhs[0] = names;
    plhs[1] = values;
}

/* optimizer_step(handle, names, params, grads) → [rc, updated params (cell row)]. */
static void s_optimizer_step(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    rlx_mx_arity("support_optimizer_step", nlhs, 2, nrhs, 4);
    RlxOptimizer* o = (RlxOptimizer*)rlx_mx_ptr(prhs[0]);
    RlxMxList names = rlx_mx_strings(prhs[1]);
    RlxMxList ps = rlx_mx_float_arrays(prhs[2]);
    RlxMxList gs = rlx_mx_float_arrays(prhs[3]);
    size_t n = names.n;
    if (ps.n != n || gs.n != n) mexErrMsgIdAndTxt("rlx:type", "optimizer step: names, params and grads differ in length");
    size_t* lens = (size_t*)mxCalloc(n ? n : 1, sizeof(size_t));
    for (size_t k = 0; k < n; k++) {
        lens[k] = mxGetNumberOfElements(mxGetCell(prhs[2], k));
        if (mxGetNumberOfElements(mxGetCell(prhs[3], k)) != lens[k])
            mexErrMsgIdAndTxt("rlx:type", "optimizer step: '%s' has %zu values but its gradient %zu",
                              (const char*)names.ptrs[k], lens[k], mxGetNumberOfElements(mxGetCell(prhs[3], k)));
    }
    int32_t rc = rlx_optimizer_step(o, (const char* const*)names.ptrs, (float* const*)ps.ptrs,
                                    (const float* const*)gs.ptrs, lens, NULL, NULL, n);
    plhs[0] = mxCreateDoubleScalar(rc);
    mxArray* out = mxCreateCellMatrix(1, n);
    for (size_t k = 0; k < n; k++) mxSetCell(out, k, singles((const float*)ps.ptrs[k], lens[k]));
    plhs[1] = out;
    mxFree(lens);
    rlx_mx_list_free(&names);
    rlx_mx_list_free(&ps);
    rlx_mx_list_free(&gs);
}

static const RlxMexFn SUPPORT[] = {
    {"support_last_error", s_last_error},
    {"support_optimizer_step", s_optimizer_step},
    {"support_take_outputs", s_take_outputs},
    {"support_take_tensors", s_take_tensors},
    {"support_take_typed_outputs", s_take_typed_outputs},
};

static int by_name(const void* key, const void* row) {
    return strcmp((const char*)key, ((const RlxMexFn*)row)->name);
}

/* rlx_mex(name, args…): run the rlx-capi function `name`. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 1 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt("rlx:type", "usage: rlx_mex(name, args...)");
    char name[128];
    mxGetString(prhs[0], name, sizeof(name));
    const RlxMexFn* f = (const RlxMexFn*)bsearch(name, RLX_MEX_FNS, RLX_MEX_NFNS, sizeof(RlxMexFn), by_name);
    if (!f) f = (const RlxMexFn*)bsearch(name, SUPPORT, sizeof(SUPPORT) / sizeof(SUPPORT[0]), sizeof(RlxMexFn), by_name);
    if (!f) mexErrMsgIdAndTxt("rlx:not_found", "rlx_mex: no function '%s' in this build", name);
    f->fn(nlhs, plhs, nrhs - 1, prhs + 1);
}
