/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * What the generated rlx_r.c stands on: SEXP <-> C conversion
 * (rlx_r_support.c). Temporaries are R_alloc'd, so R reclaims them at the end
 * of the .Call, even when an error unwinds it.
 */
#ifndef RLX_R_H
#define RLX_R_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "rlx.h"

/* The R API names the generated code uses without R_NO_REMAP's Rf_ prefix. */
#define allocVector Rf_allocVector
#define ScalarInteger Rf_ScalarInteger
#define ScalarLogical Rf_ScalarLogical
#define ScalarReal Rf_ScalarReal
#define mkString Rf_mkString
#define asLogical Rf_asLogical
#define asReal Rf_asReal

double rlx_r_number(SEXP x);
void* rlx_r_ptr(SEXP x);
const char* rlx_r_string(SEXP x);
const float* rlx_r_floats(SEXP x);
const uint8_t* rlx_r_bytes(SEXP x);
const uint32_t* rlx_r_u32s(SEXP x);
const int64_t* rlx_r_i64s(SEXP x);
const uint64_t* rlx_r_u64s(SEXP x);
const char* const* rlx_r_strings(SEXP x);
const float* const* rlx_r_float_arrays(SEXP x);
const uint8_t* const* rlx_r_byte_arrays(SEXP x);

SEXP rlx_r_from_f32s(RlxF32s v);
SEXP rlx_r_from_bytes(RlxBytes v);
SEXP rlx_r_from_u32s(RlxU32s v);
SEXP rlx_r_from_string(char* s);
SEXP rlx_r_handle(void* p, R_CFinalizer_t fin);

SEXP R_support_last_error(void);
SEXP R_support_take_outputs(SEXP p);
SEXP R_support_take_typed_outputs(SEXP p);
SEXP R_support_take_tensors(SEXP p);
SEXP R_support_optimizer_step(SEXP h, SEXP names, SEXP params, SEXP grads);

#define RLX_R_SUPPORT_CALLS                                                        \
    {"R_support_last_error", (DL_FUNC)&R_support_last_error, 0},                   \
    {"R_support_take_outputs", (DL_FUNC)&R_support_take_outputs, 1},               \
    {"R_support_take_typed_outputs", (DL_FUNC)&R_support_take_typed_outputs, 1},   \
    {"R_support_take_tensors", (DL_FUNC)&R_support_take_tensors, 1},               \
    {"R_support_optimizer_step", (DL_FUNC)&R_support_optimizer_step, 4},

#endif
