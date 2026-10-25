/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The helpers the generated rlx_core.c stands on: Lua tables as the C arrays
 * rlx-capi takes, and pushing what it returns.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <lauxlib.h>
#include <lua.h>

#include "rlx.h"

typedef struct {
    float** ptrs;
    int n;
} RlxLuaArrays;

void* rlx_lua_ptr(lua_State* L, int i);
void rlx_lua_push_ptr(lua_State* L, void* p);
void rlx_lua_push_string(lua_State* L, char* s);
void rlx_lua_push_f32s(lua_State* L, RlxF32s v);
void rlx_lua_push_bytes(lua_State* L, RlxBytes v);
void rlx_lua_push_u32s(lua_State* L, RlxU32s v);
float* rlx_lua_floats(lua_State* L, int i);
uint32_t* rlx_lua_u32s(lua_State* L, int i);
int64_t* rlx_lua_i64s(lua_State* L, int i);
size_t* rlx_lua_sizes(lua_State* L, int i);
const char** rlx_lua_strings(lua_State* L, int i);
RlxLuaArrays rlx_lua_float_arrays(lua_State* L, int i);
void rlx_lua_arrays_free(RlxLuaArrays* a);
void rlx_lua_register_support(lua_State* L);
