/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The hand-written half of the rlx_core Lua module: table <-> C array
 * conversion, and the support functions the Lua side calls — the last error,
 * reading run outputs and tensor maps, and Optimizer.step (in place).
 */

#include <string.h>

#include "rlx_lua.h"

void* rlx_lua_ptr(lua_State* L, int i) {
    return lua_isnoneornil(L, i) ? NULL : lua_touserdata(L, i);
}

void rlx_lua_push_ptr(lua_State* L, void* p) {
    if (p) lua_pushlightuserdata(L, p);
    else lua_pushnil(L);
}

void rlx_lua_push_string(lua_State* L, char* s) {
    if (!s) { lua_pushnil(L); return; }
    lua_pushstring(L, s);
    rlx_string_free(s);
}

void rlx_lua_push_f32s(lua_State* L, RlxF32s v) {
    lua_createtable(L, (int)v.len, 0);
    for (size_t i = 0; i < v.len; i++) { lua_pushnumber(L, v.ptr[i]); lua_rawseti(L, -2, (lua_Integer)i + 1); }
    rlx_f32s_free(v);
}

void rlx_lua_push_bytes(lua_State* L, RlxBytes v) {
    lua_pushlstring(L, (const char*)v.ptr, v.len);
    rlx_bytes_free(v);
}

void rlx_lua_push_u32s(lua_State* L, RlxU32s v) {
    lua_createtable(L, (int)v.len, 0);
    for (size_t i = 0; i < v.len; i++) { lua_pushinteger(L, v.ptr[i]); lua_rawseti(L, -2, (lua_Integer)i + 1); }
    rlx_u32s_free(v);
}

#define TABLE_TO(T, name, get)                                              \
    T* name(lua_State* L, int i) {                                          \
        if (lua_isnoneornil(L, i)) return NULL;                             \
        luaL_checktype(L, i, LUA_TTABLE);                                   \
        lua_Integer n = luaL_len(L, i);                                     \
        T* out = (T*)malloc(sizeof(T) * (n > 0 ? n : 1));                   \
        for (lua_Integer k = 1; k <= n; k++) {                              \
            lua_rawgeti(L, i, k);                                           \
            out[k - 1] = (T)get(L, -1);                                     \
            lua_pop(L, 1);                                                  \
        }                                                                   \
        return out;                                                         \
    }

TABLE_TO(float, rlx_lua_floats, luaL_checknumber)
TABLE_TO(uint32_t, rlx_lua_u32s, luaL_checkinteger)
TABLE_TO(int64_t, rlx_lua_i64s, luaL_checkinteger)
TABLE_TO(size_t, rlx_lua_sizes, luaL_checkinteger)

const char** rlx_lua_strings(lua_State* L, int i) {
    if (lua_isnoneornil(L, i)) return NULL;
    luaL_checktype(L, i, LUA_TTABLE);
    lua_Integer n = luaL_len(L, i);
    const char** out = (const char**)malloc(sizeof(char*) * (n > 0 ? n : 1));
    for (lua_Integer k = 1; k <= n; k++) {
        lua_rawgeti(L, i, k);
        out[k - 1] = lua_tostring(L, -1); /* the table keeps the string alive */
        lua_pop(L, 1);
    }
    return out;
}

RlxLuaArrays rlx_lua_float_arrays(lua_State* L, int i) {
    RlxLuaArrays a = {NULL, 0};
    if (lua_isnoneornil(L, i)) return a;
    luaL_checktype(L, i, LUA_TTABLE);
    a.n = (int)luaL_len(L, i);
    a.ptrs = (float**)calloc(a.n > 0 ? a.n : 1, sizeof(float*));
    for (int k = 1; k <= a.n; k++) {
        lua_rawgeti(L, i, k);
        a.ptrs[k - 1] = rlx_lua_floats(L, lua_gettop(L));
        lua_pop(L, 1);
    }
    return a;
}

void rlx_lua_arrays_free(RlxLuaArrays* a) {
    for (int k = 0; k < a->n; k++) free(a->ptrs[k]);
    free(a->ptrs);
}

/* ── support functions (rlx_core.support_*) ─────────────────── */

static int l_last_error(lua_State* L) {
    lua_pushstring(L, rlx_last_error());
    return 1;
}

static int l_take_outputs(lua_State* L) {
    RlxOutputs* o = (RlxOutputs*)rlx_lua_ptr(L, 1);
    size_t n = rlx_outputs_len(o);
    lua_createtable(L, (int)n, 0);
    for (size_t i = 0; i < n; i++) {
        const float* d = NULL;
        size_t len = 0;
        rlx_outputs_get(o, i, &d, &len);
        lua_createtable(L, (int)len, 0);
        for (size_t k = 0; k < len; k++) { lua_pushnumber(L, d[k]); lua_rawseti(L, -2, (lua_Integer)k + 1); }
        lua_rawseti(L, -2, (lua_Integer)i + 1);
    }
    rlx_outputs_free(o);
    return 1;
}

static int l_take_typed_outputs(lua_State* L) {
    RlxTypedOutputs* o = (RlxTypedOutputs*)rlx_lua_ptr(L, 1);
    size_t n = rlx_typed_outputs_len(o);
    lua_createtable(L, (int)n, 0);
    for (size_t i = 0; i < n; i++) {
        const uint8_t* d = NULL;
        size_t len = 0;
        const char* dt = NULL;
        rlx_typed_outputs_get(o, i, &d, &len, &dt);
        lua_createtable(L, 0, 2);
        lua_pushlstring(L, (const char*)d, len);
        lua_setfield(L, -2, "data");
        lua_pushstring(L, dt);
        lua_setfield(L, -2, "dtype");
        lua_rawseti(L, -2, (lua_Integer)i + 1);
    }
    rlx_typed_outputs_free(o);
    return 1;
}

static int l_take_tensors(lua_State* L) {
    RlxTensorMap* m = (RlxTensorMap*)rlx_lua_ptr(L, 1);
    size_t n = rlx_tensor_map_len(m);
    lua_createtable(L, 0, (int)n);
    for (size_t i = 0; i < n; i++) {
        const char* name = NULL;
        const float* d = NULL;
        size_t len = 0;
        rlx_tensor_map_get(m, i, &name, &d, &len);
        lua_createtable(L, (int)len, 0);
        for (size_t k = 0; k < len; k++) { lua_pushnumber(L, d[k]); lua_rawseti(L, -2, (lua_Integer)k + 1); }
        lua_setfield(L, -2, name);
    }
    rlx_tensor_map_free(m);
    return 1;
}

/* optimizer_step(handle, names, params(table of tables), grads) -> rc; params updated in place. */
static int l_optimizer_step(lua_State* L) {
    RlxOptimizer* o = (RlxOptimizer*)rlx_lua_ptr(L, 1);
    const char** names = rlx_lua_strings(L, 2);
    int n = (int)luaL_len(L, 2);
    RlxLuaArrays ps = rlx_lua_float_arrays(L, 3);
    RlxLuaArrays gs = rlx_lua_float_arrays(L, 4);
    size_t* lens = (size_t*)calloc(n > 0 ? n : 1, sizeof(size_t));
    for (int k = 1; k <= n; k++) {
        lua_rawgeti(L, 3, k);
        lens[k - 1] = (size_t)luaL_len(L, -1);
        lua_pop(L, 1);
    }
    int32_t rc = rlx_optimizer_step(o, names, ps.ptrs, (const float* const*)gs.ptrs, lens, NULL, NULL, (size_t)n);
    for (int k = 1; rc == RLX_OK && k <= n; k++) {
        lua_rawgeti(L, 3, k);
        for (size_t j = 0; j < lens[k - 1]; j++) { lua_pushnumber(L, ps.ptrs[k - 1][j]); lua_rawseti(L, -2, (lua_Integer)j + 1); }
        lua_pop(L, 1);
    }
    free(lens);
    free((void*)names);
    rlx_lua_arrays_free(&ps);
    rlx_lua_arrays_free(&gs);
    lua_pushinteger(L, rc);
    return 1;
}

void rlx_lua_register_support(lua_State* L) {
    static const luaL_Reg support[] = {
        {"support_last_error", l_last_error},
        {"support_take_outputs", l_take_outputs},
        {"support_take_typed_outputs", l_take_typed_outputs},
        {"support_take_tensors", l_take_tensors},
        {"support_optimizer_step", l_optimizer_step},
        {NULL, NULL},
    };
    luaL_setfuncs(L, support, 0);
}
