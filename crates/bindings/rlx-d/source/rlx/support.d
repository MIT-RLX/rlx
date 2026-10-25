// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// What the generated rlx.generated stands on: errors, the handle base, and
/// moving values to and from C.
module rlx.support;

import std.algorithm : sort;
import std.array : array;
import std.conv : to;
import std.json : JSONType, JSONValue, parseJSON;
import std.math : isNaN;
import std.string : fromStringz, toStringz;

/// "No node", for an optional node argument.
enum uint NONE = 0xFFFF_FFFF;

/// What kind of failure an `RlxException` is (the `RLX_ERR_*` status).
enum RlxErrorKind { type, range, runtime, panic, notFound, other }

/// A failed rlx call.
class RlxException : Exception
{
    RlxErrorKind kind;
    this(RlxErrorKind kind, string msg, string file = __FILE__, size_t line = __LINE__)
    {
        super(msg, file, line);
        this.kind = kind;
    }
}

/// A tensor of any dtype: its little-endian bytes and dtype name.
struct TypedTensor
{
    const(ubyte)[] data;
    string dtype;
}

/// A handle's C free function.
package alias FreeFn = extern (C) void function(void*) nothrow @nogc;

package struct Adopt {}
package enum adopt = Adopt();

/// The base of every handle class: freed by the destructor, or now with `close()`.
abstract class Handle
{
    package void* h;
    private FreeFn release;

    package this(void* h, FreeFn release)
    {
        if (h is null)
            throw new RlxException(RlxErrorKind.runtime, "rlx returned NULL");
        this.h = h;
        this.release = release;
    }

    /// Free the handle now (it is unusable afterwards).
    final void close() nothrow @nogc
    {
        if (h !is null)
        {
            release(h);
            h = null;
        }
    }

    ~this()
    {
        close();
    }
}

extern (C) nothrow @nogc
{
    private const(char)* rlx_last_error();
    private void rlx_string_free(void* s);
    private void rlx_f32s_release(RlxSlice* v);
    private void rlx_bytes_release(RlxSlice* v);
    private void rlx_u32s_release(RlxSlice* v);
    private size_t rlx_outputs_len(void* o);
    private int rlx_outputs_get(void* o, size_t i, const(float)** d, size_t* n);
    private void rlx_outputs_free(void* o);
    private size_t rlx_typed_outputs_len(void* o);
    private int rlx_typed_outputs_get(void* o, size_t i, const(ubyte)** d, size_t* n, const(char)** dt);
    private void rlx_typed_outputs_free(void* o);
    private size_t rlx_tensor_map_len(void* m);
    private int rlx_tensor_map_get(void* m, size_t i, const(char)** name, const(float)** d, size_t* n);
    private void rlx_tensor_map_free(void* m);
    package int rlx_optimizer_step(void* o, const(char*)* names, float** params, const(float*)* grads,
        const(size_t)* lens, const(size_t*)* shapes, const(size_t)* ndims, size_t n);
}

package void checkRc(int rc)
{
    if (rc == 0)
        return;
    immutable kind = rc == -1 ? RlxErrorKind.type : rc == -2 ? RlxErrorKind.range
        : rc == -3 ? RlxErrorKind.runtime : rc == -4 ? RlxErrorKind.panic
        : rc == -5 ? RlxErrorKind.notFound : RlxErrorKind.other;
    throw new RlxException(kind, rlx_last_error().fromStringz.idup);
}

// ── arguments ──────────────────────────────────────────────────

package const(char)* jsonArg(JSONValue v)
{
    return v.type == JSONType.null_ ? null : v.toString.toStringz;
}

package T* box(T)(T v, T none)
{
    return v == none ? null : new T(v);
}

package float* boxF(float v)
{
    return v.isNaN ? null : new float(v);
}

package const(char)*[] cstrs(const(string)[] v)
{
    auto out_ = new const(char)*[v.length];
    foreach (i, s; v)
        out_[i] = s.toStringz;
    return out_;
}

package size_t[] rows(const(size_t[])[] v, size_t w, bool splat)
{
    size_t[] out_;
    foreach (i, r; v)
    {
        if (r.length == w)
            out_ ~= r;
        else if (splat && r.length == 1)
            foreach (_; 0 .. w)
                out_ ~= r[0];
        else
            throw new RlxException(RlxErrorKind.type,
                "row " ~ i.to!string ~ " needs " ~ w.to!string ~ " values, got " ~ r.length.to!string);
    }
    return out_;
}

package float[] frows(const(float[])[] v, size_t w)
{
    float[] out_;
    foreach (i, r; v)
    {
        if (r.length != w)
            throw new RlxException(RlxErrorKind.type,
                "row " ~ i.to!string ~ " needs " ~ w.to!string ~ " values, got " ~ r.length.to!string);
        out_ ~= r;
    }
    return out_;
}

/// Named f32 tensors as the C ABI takes them.
package struct Inputs
{
    const(char)*[] names;
    const(float)*[] data;
    size_t[] lens;

    this(const(float[])[string] m)
    {
        auto keys = m.keys.dup;
        keys.sort();
        foreach (k; keys)
        {
            names ~= k.toStringz;
            data ~= m[k].ptr;
            lens ~= m[k].length;
        }
    }
}

/// Named typed tensors as the C ABI takes them.
package struct TypedInputs
{
    const(char)*[] names;
    const(ubyte)*[] data;
    size_t[] lens;
    const(char)*[] dtypes;

    this(const(TypedTensor)[string] m)
    {
        auto keys = m.keys.dup;
        keys.sort();
        foreach (k; keys)
        {
            names ~= k.toStringz;
            data ~= m[k].data.ptr;
            lens ~= m[k].data.length;
            dtypes ~= m[k].dtype.toStringz;
        }
    }
}

// ── results ────────────────────────────────────────────────────

/// `{ptr, len}` as rlx returns arrays.
package struct RlxSlice
{
    void* ptr;
    size_t len;
}

package string takeStringOrNull(void* p)
{
    if (p is null)
        return null;
    auto s = (cast(const(char)*) p).fromStringz.idup;
    rlx_string_free(p);
    return s;
}

package string takeString(void* p)
{
    auto s = takeStringOrNull(p);
    return s is null ? "" : s;
}

package T fromJson(T)(string s)
{
    return convert!T(parseJSON(s));
}

private T convert(T)(JSONValue v)
{
    static if (is(T == string))
        return v.str;
    else static if (is(T : E[], E))
        return v.array.map!(e => convert!E(e)).array;
    else
        return cast(T) v.integer;
}

import std.algorithm : map;

package float[] takeF32s(ref RlxSlice o)
{
    auto v = (cast(float*) o.ptr)[0 .. o.len].dup;
    rlx_f32s_release(&o);
    return v;
}

package ubyte[] takeBytes(ref RlxSlice o)
{
    auto v = (cast(ubyte*) o.ptr)[0 .. o.len].dup;
    rlx_bytes_release(&o);
    return v;
}

package uint[] takeU32s(ref RlxSlice o)
{
    auto v = (cast(uint*) o.ptr)[0 .. o.len].dup;
    rlx_u32s_release(&o);
    return v;
}

package float[][] takeOutputs(void* o)
{
    float[][] out_;
    foreach (i; 0 .. rlx_outputs_len(o))
    {
        const(float)* d;
        size_t n;
        rlx_outputs_get(o, i, &d, &n);
        out_ ~= d[0 .. n].dup;
    }
    rlx_outputs_free(o);
    return out_;
}

package TypedTensor[] takeTypedOutputs(void* o)
{
    TypedTensor[] out_;
    foreach (i; 0 .. rlx_typed_outputs_len(o))
    {
        const(ubyte)* d;
        size_t n;
        const(char)* dt;
        rlx_typed_outputs_get(o, i, &d, &n, &dt);
        out_ ~= TypedTensor(d[0 .. n].dup, dt.fromStringz.idup);
    }
    rlx_typed_outputs_free(o);
    return out_;
}

package float[][string] takeTensors(void* m)
{
    float[][string] out_;
    foreach (i; 0 .. rlx_tensor_map_len(m))
    {
        const(char)* name;
        const(float)* d;
        size_t n;
        rlx_tensor_map_get(m, i, &name, &d, &n);
        out_[name.fromStringz.idup] = d[0 .. n].dup;
    }
    rlx_tensor_map_free(m);
    return out_;
}
