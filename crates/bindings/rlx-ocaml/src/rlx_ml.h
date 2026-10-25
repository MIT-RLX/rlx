/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * What the generated rlx_stubs.c stands on: OCaml value <-> C conversion
 * (rlx_ml_support.c).
 */
#ifndef RLX_ML_H
#define RLX_ML_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include "rlx.h"

typedef struct {
    float** ptrs;
    size_t n;
} RlxMlArrays;

void rlx_ml_check(int32_t rc);
void rlx_ml_raise(int32_t rc, const char* msg);
void* rlx_ml_ptr(value opt);
const char* rlx_ml_cstr(value opt);
float* rlx_ml_floats(value arr);
const uint8_t* rlx_ml_bytes_in(value s);
uint32_t* rlx_ml_u32s_in(value arr);
int64_t* rlx_ml_i64s_in(value arr);
uint64_t* rlx_ml_u64s_in(value arr);
const char** rlx_ml_strings(value arr);
RlxMlArrays rlx_ml_float_arrays(value arr);
const uint8_t** rlx_ml_byte_arrays(value arr);
void rlx_ml_arrays_free(RlxMlArrays* a);

value rlx_ml_handle(void* p, struct custom_operations* ops);
value rlx_ml_string_opt(char* s);
value rlx_ml_f32s(RlxF32s v);
value rlx_ml_bytes(RlxBytes v);
value rlx_ml_u32s(RlxU32s v);
value rlx_ml_outputs(RlxOutputs* o);
value rlx_ml_typed_outputs(RlxTypedOutputs* o);
value rlx_ml_tensors(RlxTensorMap* m);

#endif
