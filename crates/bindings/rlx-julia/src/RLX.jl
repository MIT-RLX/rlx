# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Not precompiled: `librlx` below is read from `RLX_CAPI_LIB` when the module
# loads, and a precompile cache would freeze the path of the first build (a
# release library kept loading after switching to a debug one).
__precompile__(false)

"""
    RLX

Julia bindings for rlx over its C ABI (`crates/bindings/rlx-capi`). Every C
function is a Julia method (`generated.jl`, from rlx-bind's tables): handle
types are `mutable struct`s freed by finalizers, functions dispatch on them
(`matmul(g, a, b)`, `run(compiled, inputs)`), statics on the type
(`from_env(Runner, g)`), optional parameters are keywords.

The library is `ENV["RLX_CAPI_LIB"]` when set, else the one bundled with the
package (`native/<os>-<arch>/`), else `librlx_capi` from the workspace's
`target/debug` (a checkout).
"""
module RLX

const librlx = get(ENV, "RLX_CAPI_LIB") do
    file = Sys.isapple() ? "librlx_capi.dylib" : Sys.iswindows() ? "rlx_capi.dll" : "librlx_capi.so"
    os = Sys.isapple() ? "macos" : Sys.iswindows() ? "windows" : "linux"
    arch = Sys.ARCH === :aarch64 ? "arm64" : "x64"
    bundled = joinpath(@__DIR__, "..", "native", "$os-$arch", file)
    isfile(bundled) ? bundled : joinpath(@__DIR__, "..", "..", "..", "..", "target", "debug", file)
end

"""A node of a graph — the id every builder returns."""
const Node = UInt32
""""No node", for an optional node argument."""
const NONE = typemax(UInt32)

struct RlxF32s
    ptr::Ptr{Cfloat}
    len::Csize_t
end
struct RlxBytes
    ptr::Ptr{UInt8}
    len::Csize_t
end
struct RlxU32s
    ptr::Ptr{UInt32}
    len::Csize_t
end
_free(v::RlxF32s) = ccall((:rlx_f32s_free, librlx), Cvoid, (RlxF32s,), v)
_free(v::RlxBytes) = ccall((:rlx_bytes_free, librlx), Cvoid, (RlxBytes,), v)
_free(v::RlxU32s) = ccall((:rlx_u32s_free, librlx), Cvoid, (RlxU32s,), v)

"""A failed rlx call: `kind` is `:type`, `:range`, `:runtime`, `:panic` or `:not_found`."""
struct RLXError <: Exception
    kind::Symbol
    message::String
end
Base.showerror(io::IO, e::RLXError) = print(io, "RLXError(", e.kind, "): ", e.message)

const _KINDS = Dict(-1 => :type, -2 => :range, -3 => :runtime, -4 => :panic, -5 => :not_found)

function _check(rc::Int32)
    rc == 0 && return nothing
    msg = unsafe_string(ccall((:rlx_last_error, librlx), Ptr{UInt8}, ()))
    throw(RLXError(get(_KINDS, Int(rc), :unknown), msg))
end

# ── arguments ───────────────────────────────────────────────

_cstr(s::AbstractString) = Vector{UInt8}(string(s, "\0"))
_cstr(s::Symbol) = _cstr(String(s))
_cstr_or_null(s) = s === nothing ? C_NULL : _cstr(s)
_json_or_null(v) = v === nothing ? C_NULL : _cstr(_json(v))
_ref_or_null(::Type{T}, v) where {T} = v === nothing ? C_NULL : Ref{T}(v)

"""Rows (`[[a, b], …]`) flattened; with `splat`, a one-value row stands for `w` copies."""
function _rows(::Type{T}, rows, w::Int, splat::Bool) where {T}
    out = T[]
    for r in rows
        if length(r) == w
            append!(out, T.(r))
        elseif splat && length(r) == 1
            append!(out, fill(T(r[1]), w))
        else
            throw(RLXError(:type, "row needs $w values, got $(length(r))"))
        end
    end
    out
end

function _cstrs(xs)
    xs === nothing && return (Vector{UInt8}[], Ptr{UInt8}[])
    bufs = [_cstr(x) for x in xs]
    (bufs, [pointer(b) for b in bufs])
end

_sorted_pairs(d::AbstractDict) = sort!(collect(pairs(d)); by = p -> string(first(p)))
_sorted_pairs(ps) = collect(ps)

"""Named f32 inputs as parallel C arrays, with what they point into."""
function _inputs(d)
    ps = _sorted_pairs(d)
    names = [_cstr(string(first(p))) for p in ps]
    data = [collect(Cfloat, vec(last(p))) for p in ps]
    (names = [pointer(n) for n in names], data = [pointer(x) for x in data],
     lens = Csize_t[length(x) for x in data], keep = (names, data))
end

"""A typed tensor: raw bytes and a dtype name (`"f32"`, `"i64"`, …)."""
struct TypedTensor
    bytes::Vector{UInt8}
    dtype::String
end
TypedTensor(a::AbstractArray{Float32}) = TypedTensor(collect(reinterpret(UInt8, vec(a))), "f32")

function _typed_inputs(d)
    ps = _sorted_pairs(d)
    names = [_cstr(string(first(p))) for p in ps]
    bytes = [last(p).bytes for p in ps]
    dtypes = [_cstr(last(p).dtype) for p in ps]
    (names = [pointer(n) for n in names], data = [pointer(b) for b in bytes],
     lens = Csize_t[length(b) for b in bytes], dtypes = [pointer(t) for t in dtypes],
     keep = (names, bytes, dtypes))
end

# ── results ─────────────────────────────────────────────────

function _take_string(p::Ptr{UInt8})
    p == C_NULL && return ""
    s = unsafe_string(p)
    ccall((:rlx_string_free, librlx), Cvoid, (Ptr{UInt8},), p)
    s
end

function _take_array(v)
    out = v.ptr == C_NULL ? eltype(v.ptr)[] : copy(unsafe_wrap(Array, v.ptr, Int(v.len)))
    _free(v)
    out
end

function _take_outputs(o::Ptr{Cvoid})
    n = ccall((:rlx_outputs_len, librlx), Csize_t, (Ptr{Cvoid},), o)
    out = Vector{Vector{Float32}}(undef, n)
    for i in 1:n
        p = Ref{Ptr{Cfloat}}(C_NULL); len = Ref{Csize_t}(0)
        _check(ccall((:rlx_outputs_get, librlx), Int32, (Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cfloat}}, Ptr{Csize_t}), o, i - 1, p, len))
        out[i] = copy(unsafe_wrap(Array, p[], Int(len[])))
    end
    ccall((:rlx_outputs_free, librlx), Cvoid, (Ptr{Cvoid},), o)
    out
end

function _take_typed_outputs(o::Ptr{Cvoid})
    n = ccall((:rlx_typed_outputs_len, librlx), Csize_t, (Ptr{Cvoid},), o)
    out = Vector{TypedTensor}(undef, n)
    for i in 1:n
        p = Ref{Ptr{UInt8}}(C_NULL); len = Ref{Csize_t}(0); dt = Ref{Ptr{UInt8}}(C_NULL)
        _check(ccall((:rlx_typed_outputs_get, librlx), Int32, (Ptr{Cvoid}, Csize_t, Ptr{Ptr{UInt8}}, Ptr{Csize_t}, Ptr{Ptr{UInt8}}), o, i - 1, p, len, dt))
        out[i] = TypedTensor(copy(unsafe_wrap(Array, p[], Int(len[]))), unsafe_string(dt[]))
    end
    ccall((:rlx_typed_outputs_free, librlx), Cvoid, (Ptr{Cvoid},), o)
    out
end

function _take_tensors(m::Ptr{Cvoid})
    n = ccall((:rlx_tensor_map_len, librlx), Csize_t, (Ptr{Cvoid},), m)
    out = Dict{String,Vector{Float32}}()
    for i in 1:n
        nm = Ref{Ptr{UInt8}}(C_NULL); p = Ref{Ptr{Cfloat}}(C_NULL); len = Ref{Csize_t}(0)
        _check(ccall((:rlx_tensor_map_get, librlx), Int32, (Ptr{Cvoid}, Csize_t, Ptr{Ptr{UInt8}}, Ptr{Ptr{Cfloat}}, Ptr{Csize_t}), m, i - 1, nm, p, len))
        out[unsafe_string(nm[])] = copy(unsafe_wrap(Array, p[], Int(len[])))
    end
    ccall((:rlx_tensor_map_free, librlx), Cvoid, (Ptr{Cvoid},), m)
    out
end

# ── JSON (options in, reports out) ──────────────────────────

_json(v) = (io = IOBuffer(); _write_json(io, v); String(take!(io)))
_write_json(io, ::Nothing) = print(io, "null")
_write_json(io, b::Bool) = print(io, b ? "true" : "false")
_write_json(io, x::Integer) = print(io, x)
_write_json(io, x::AbstractFloat) = isfinite(x) ? print(io, x) : print(io, "null")
_write_json(io, s::Union{AbstractString,Symbol}) = (print(io, '"'); _escape(io, string(s)); print(io, '"'))
function _escape(io, s)
    for ch in s
        ch == '"' ? print(io, "\\\"") : ch == '\\' ? print(io, "\\\\") :
        ch == '\n' ? print(io, "\\n") : ch < ' ' ? print(io, "\\u", string(UInt16(ch), base = 16, pad = 4)) : print(io, ch)
    end
end
function _write_json(io, d::Union{AbstractDict,NamedTuple})
    print(io, '{')
    for (i, (k, v)) in enumerate(pairs(d))
        i > 1 && print(io, ',')
        _write_json(io, string(k)); print(io, ':'); _write_json(io, v)
    end
    print(io, '}')
end
function _write_json(io, xs::Union{AbstractVector,Tuple})
    print(io, '[')
    for (i, x) in enumerate(xs)
        i > 1 && print(io, ',')
        _write_json(io, x)
    end
    print(io, ']')
end

"""JSON text → `Dict{String,Any}` / `Vector{Any}` / numbers / strings / `nothing`."""
function _parse_json(s::AbstractString)
    v, i = _jvalue(s, _skip(s, firstindex(s)))
    v
end
_skip(s, i) = (while i <= ncodeunits(s) && isspace(s[i]); i = nextind(s, i); end; i)
function _jvalue(s, i)
    c = s[i]
    if c == '{'
        d = Dict{String,Any}(); i = _skip(s, i + 1)
        s[i] == '}' && return d, i + 1
        while true
            k, i = _jstring(s, _skip(s, i)); i = _skip(s, i) + 1
            v, i = _jvalue(s, _skip(s, i)); d[k] = v; i = _skip(s, i)
            s[i] == '}' && return d, i + 1
            i = i + 1
        end
    elseif c == '['
        a = Any[]; i = _skip(s, i + 1)
        s[i] == ']' && return a, i + 1
        while true
            v, i = _jvalue(s, _skip(s, i)); push!(a, v); i = _skip(s, i)
            s[i] == ']' && return a, i + 1
            i = i + 1
        end
    elseif c == '"'
        return _jstring(s, i)
    elseif startswith(SubString(s, i), "true")
        return true, i + 4
    elseif startswith(SubString(s, i), "false")
        return false, i + 5
    elseif startswith(SubString(s, i), "null")
        return nothing, i + 4
    else
        j = i
        while j <= ncodeunits(s) && (isdigit(s[j]) || s[j] in "+-.eE"); j += 1; end
        t = s[i:j-1]
        return (occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)), j
    end
end
function _jstring(s, i)
    io = IOBuffer(); i = nextind(s, i)
    while s[i] != '"'
        if s[i] == '\\'
            i = nextind(s, i); c = s[i]
            if c == 'u'
                Base.write(io, Char(parse(UInt16, s[i+1:i+4], base = 16))); i += 4
            else
                Base.write(io, c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == 'b' ? '\b' : c == 'f' ? '\f' : c)
            end
        else
            Base.write(io, s[i])
        end
        i = nextind(s, i)
    end
    String(take!(io)), nextind(s, i)
end

include("generated.jl")

# ── the manual rows ─────────────────────────────────────────

"""
    step!(opt::Optimizer, params::AbstractDict, grads::AbstractDict)

Update each `params[name]` (a `Vector{Float32}`) in place from `grads[name]`.
"""
function step!(opt::Optimizer, params::AbstractDict, grads::AbstractDict)
    names = sort!(collect(keys(params)); by = string)
    bufs = [_cstr(string(n)) for n in names]
    ps = [params[n]::Vector{Float32} for n in names]
    gs = [collect(Cfloat, grads[n]) for n in names]
    GC.@preserve bufs ps gs begin
        _check(ccall((:rlx_optimizer_step, librlx), Int32,
            (Ptr{Ptr{UInt8}}, Ptr{Ptr{UInt8}}, Ptr{Ptr{Cfloat}}, Ptr{Ptr{Cfloat}}, Ptr{Csize_t}, Ptr{Ptr{Csize_t}}, Ptr{Csize_t}, Csize_t),
            opt.handle, [pointer(b) for b in bufs], [pointer(p) for p in ps], [pointer(g) for g in gs],
            Csize_t[length(p) for p in ps], C_NULL, C_NULL, length(ps)))
    end
    params
end

"""
    run(trainer::Trainer, steps; batch, lr=nothing, on_step=nothing, every=1)

Train `steps` steps: `batch(step)` returns the inputs (or `nothing` when the
data is exhausted), `lr(step)` sets each step's learning rate, `on_step(step,
loss)` returns `false` to stop. Returns the run summary.
"""
function run(t::Trainer, steps::Integer; batch, lr = nothing, on_step = nothing, every::Integer = 1)
    losses = Float32[]
    stopped = false
    t0 = time()
    start = Int(RLX.steps(t))
    for i in 0:steps-1
        s = start + i
        b = batch(s)
        b === nothing && break
        lr === nothing || set_lr(t, lr(s))
        loss = RLX.step(t, b)
        push!(losses, loss)
        if on_step !== nothing && (i + 1) % every == 0 && on_step(s, loss) == false
            stopped = true
            break
        end
    end
    n = length(losses)
    Dict{String,Any}("steps" => n, "stopped" => stopped,
        "first_loss" => n > 0 ? losses[1] : nothing, "last_loss" => n > 0 ? losses[end] : nothing,
        "mean_loss" => n > 0 ? Base.sum(Float64.(losses)) / n : nothing,
        "steps_per_second" => n / max(time() - t0, 1e-9), "grad_norm" => grad_norm(t))
end

end # module
