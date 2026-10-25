/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * What the generated rlx_mex.c stands on: mxArray <-> C conversion
 * (rlx_mex_support.c). Temporary buffers are mxMalloc'd, so MATLAB / Octave
 * reclaim them even when an argument error unwinds the call.
 */
#ifndef RLX_MEX_H
#define RLX_MEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mex.h"
#include "rlx.h"

typedef void (*RlxMexBody)(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
typedef struct {
    const char* name;
    RlxMexBody fn;
} RlxMexFn;

/* The generated table, sorted by name. */
extern const RlxMexFn RLX_MEX_FNS[];
extern const size_t RLX_MEX_NFNS;

/* A list of C pointers built from a cell array; freed with rlx_mx_list_free. */
typedef struct {
    void** ptrs;
    size_t n;
} RlxMxList;

void rlx_mx_arity(const char* name, int nlhs, int outs, int nrhs, int ins);
void rlx_mx_empty(mxArray** plhs, int outs);

double rlx_mx_double(const mxArray* m);
uint64_t rlx_mx_u64(const mxArray* m);
void* rlx_mx_ptr(const mxArray* m);
char* rlx_mx_string(const mxArray* m);
float* rlx_mx_floats(const mxArray* m);
uint8_t* rlx_mx_bytes(const mxArray* m);
uint32_t* rlx_mx_u32s(const mxArray* m);
int64_t* rlx_mx_i64s(const mxArray* m);
uint64_t* rlx_mx_u64s(const mxArray* m);
RlxMxList rlx_mx_strings(const mxArray* m);
RlxMxList rlx_mx_float_arrays(const mxArray* m);
RlxMxList rlx_mx_byte_arrays(const mxArray* m);
void rlx_mx_list_free(RlxMxList* l);

mxArray* rlx_mx_from_f32s(RlxF32s v);
mxArray* rlx_mx_from_bytes(RlxBytes v);
mxArray* rlx_mx_from_u32s(RlxU32s v);
mxArray* rlx_mx_from_string(char* s);
mxArray* rlx_mx_from_ptr(void* p);

#endif
