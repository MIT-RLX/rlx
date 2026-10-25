/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * What the generated rlx_nif.c stands on: term <-> C conversion
 * (rlx_nif_support.c).
 */
#ifndef RLX_NIF_H
#define RLX_NIF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <erl_nif.h>

#include "rlx.h"

typedef struct { char* s; bool ok; } RlxExStr;
typedef struct { void* p; bool ok; } RlxExArr;
typedef struct { void** ptrs; unsigned n; bool ok; } RlxExList;

int rlx_ex_load(ErlNifEnv* env);
ERL_NIF_TERM rlx_ex_ok(ErlNifEnv* env, ERL_NIF_TERM t);
ERL_NIF_TERM rlx_ex_error(ErlNifEnv* env, int32_t rc);
ERL_NIF_TERM rlx_ex_badarg(ErlNifEnv* env, int index, const char* what);

RlxExStr rlx_ex_cstr(ErlNifEnv* env, ERL_NIF_TERM t);
bool rlx_ex_bool(ErlNifEnv* env, ERL_NIF_TERM t);
double rlx_ex_double(ErlNifEnv* env, ERL_NIF_TERM t);
int64_t rlx_ex_i64(ErlNifEnv* env, ERL_NIF_TERM t);
uint64_t rlx_ex_u64(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExArr rlx_ex_floats(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExArr rlx_ex_u32s_in(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExArr rlx_ex_i64s_in(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExArr rlx_ex_u64s_in(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExList rlx_ex_strings(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExList rlx_ex_float_lists(ErlNifEnv* env, ERL_NIF_TERM t);
RlxExList rlx_ex_binaries(ErlNifEnv* env, ERL_NIF_TERM t);
void rlx_ex_list_free(RlxExList* l);
bool rlx_ex_ptr(ErlNifEnv* env, ErlNifResourceType* type, ERL_NIF_TERM t, void** out);

ERL_NIF_TERM rlx_ex_handle(ErlNifEnv* env, ErlNifResourceType* type, void* p);
ERL_NIF_TERM rlx_ex_cstring(ErlNifEnv* env, const char* s);
ERL_NIF_TERM rlx_ex_string(ErlNifEnv* env, char* s);
ERL_NIF_TERM rlx_ex_f32s(ErlNifEnv* env, RlxF32s v);
ERL_NIF_TERM rlx_ex_bytes(ErlNifEnv* env, RlxBytes v);
ERL_NIF_TERM rlx_ex_u32s(ErlNifEnv* env, RlxU32s v);
ERL_NIF_TERM rlx_ex_outputs(ErlNifEnv* env, RlxOutputs* o);
ERL_NIF_TERM rlx_ex_typed_outputs(ErlNifEnv* env, RlxTypedOutputs* o);
ERL_NIF_TERM rlx_ex_tensors(ErlNifEnv* env, RlxTensorMap* m);

ERL_NIF_TERM rlx_ex_optimizer_step(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
#define RLX_EX_SUPPORT_FUNCS \
    {"support_optimizer_step", 4, rlx_ex_optimizer_step, ERL_NIF_DIRTY_JOB_CPU_BOUND},

#endif
