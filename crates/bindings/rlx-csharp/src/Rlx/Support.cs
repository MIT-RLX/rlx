// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Rlx;

/// <summary>A failed rlx call, with its <c>RLX_ERR_*</c> kind.</summary>
public sealed class RlxException : Exception
{
    /// <summary>What went wrong, by status.</summary>
    public enum ErrorKind { Type = -1, Range = -2, Runtime = -3, Panic = -4, NotFound = -5, Other = 0 }

    public ErrorKind Kind { get; }

    public RlxException(ErrorKind kind, string message) : base(message) => Kind = kind;
}

/// <summary>A node's static dims (-1 for a dynamic axis) and dtype.</summary>
public sealed record Shape(long[] Dims, string Dtype);

/// <summary>A tensor of any dtype as raw bytes ("f32", "i64", …).</summary>
public sealed record TypedTensor(byte[] Bytes, string Dtype);

/// <summary>An array rlx allocated (<c>RlxF32s</c> / <c>RlxU32s</c> / <c>RlxBytes</c>).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct RlxArray
{
    public IntPtr Ptr;
    public nuint Len;
}

/// <summary>The base of every handle type: owns the C pointer; <c>Dispose</c> or the finalizer frees it.</summary>
public abstract class RlxHandle : IDisposable
{
    internal IntPtr H;

    internal RlxHandle(IntPtr h)
    {
        if (h == IntPtr.Zero) throw new RlxException(RlxException.ErrorKind.Runtime, "rlx returned NULL");
        H = h;
    }

    protected abstract void Free(IntPtr h);

    public void Dispose()
    {
        if (H != IntPtr.Zero) { Free(H); H = IntPtr.Zero; }
        GC.SuppressFinalize(this);
    }

    ~RlxHandle()
    {
        if (H != IntPtr.Zero) Free(H);
    }
}

public static partial class RLX
{
    internal static partial class Native
    {
        internal const string Lib = "rlx_capi";

        static Native()
        {
            // RLX_CAPI_LIB (a full path) wins; otherwise the usual probing, which
            // finds runtimes/<rid>/native/ from an Rlx.Native.<rid> package.
            NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, (name, asm, path) =>
            {
                if (name != Lib) return IntPtr.Zero;
                var explicitPath = Environment.GetEnvironmentVariable("RLX_CAPI_LIB");
                if (explicitPath is not null) return NativeLibrary.Load(explicitPath);
                if (NativeLibrary.TryLoad(name, asm, path, out var handle)) return handle;
                throw new DllNotFoundException(
                    $"librlx_capi not found: add the Rlx.Native.{RuntimeInformation.RuntimeIdentifier} package " +
                    "(or Rlx.Native.<rid>.Cuda), or set RLX_CAPI_LIB to its path.");
            });
        }

        [DllImport(Lib, EntryPoint = "rlx_last_error")] internal static extern IntPtr rlx_last_error();
        [DllImport(Lib, EntryPoint = "rlx_string_free")] internal static extern void rlx_string_free(IntPtr s);
        [DllImport(Lib, EntryPoint = "rlx_f32s_release")] internal static extern void rlx_f32s_release(IntPtr v);
        [DllImport(Lib, EntryPoint = "rlx_u32s_release")] internal static extern void rlx_u32s_release(IntPtr v);
        [DllImport(Lib, EntryPoint = "rlx_bytes_release")] internal static extern void rlx_bytes_release(IntPtr v);
        [DllImport(Lib, EntryPoint = "rlx_outputs_len")] internal static extern nuint rlx_outputs_len(IntPtr o);
        [DllImport(Lib, EntryPoint = "rlx_outputs_get")] internal static extern int rlx_outputs_get(IntPtr o, nuint i, IntPtr data, IntPtr len);
        [DllImport(Lib, EntryPoint = "rlx_outputs_free")] internal static extern void rlx_outputs_free(IntPtr o);
        [DllImport(Lib, EntryPoint = "rlx_typed_outputs_len")] internal static extern nuint rlx_typed_outputs_len(IntPtr o);
        [DllImport(Lib, EntryPoint = "rlx_typed_outputs_get")] internal static extern int rlx_typed_outputs_get(IntPtr o, nuint i, IntPtr data, IntPtr len, IntPtr dtype);
        [DllImport(Lib, EntryPoint = "rlx_typed_outputs_free")] internal static extern void rlx_typed_outputs_free(IntPtr o);
        [DllImport(Lib, EntryPoint = "rlx_tensor_map_len")] internal static extern nuint rlx_tensor_map_len(IntPtr m);
        [DllImport(Lib, EntryPoint = "rlx_tensor_map_get")] internal static extern int rlx_tensor_map_get(IntPtr m, nuint i, IntPtr name, IntPtr data, IntPtr len);
        [DllImport(Lib, EntryPoint = "rlx_tensor_map_free")] internal static extern void rlx_tensor_map_free(IntPtr m);
        [DllImport(Lib, EntryPoint = "rlx_optimizer_step")] internal static extern int rlx_optimizer_step_raw(IntPtr o, IntPtr names, IntPtr ps, IntPtr gs, IntPtr lens, IntPtr shapes, IntPtr ndims, nuint n);
    }

    public sealed partial class Optimizer
    {
        /// <summary>Update each <c>parameters[name]</c> in place from <c>grads[name]</c>.</summary>
        public unsafe void Step(IDictionary<string, float[]> parameters, IDictionary<string, float[]> grads)
        {
            var names = parameters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
            using var cn = new CStrings(names);
            var handles = new List<GCHandle>();
            try
            {
                var ps = new IntPtr[names.Length];
                var gs = new IntPtr[names.Length];
                var ls = new nuint[names.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    var p = parameters[names[i]];
                    if (!grads.TryGetValue(names[i], out var g) || g.Length != p.Length)
                        throw new RlxException(RlxException.ErrorKind.Type, $"Optimizer.Step: no matching gradient for '{names[i]}'");
                    var hp = GCHandle.Alloc(p, GCHandleType.Pinned);
                    var hg = GCHandle.Alloc(g, GCHandleType.Pinned);
                    handles.Add(hp);
                    handles.Add(hg);
                    ps[i] = hp.AddrOfPinnedObject();
                    gs[i] = hg.AddrOfPinnedObject();
                    ls[i] = (nuint)p.Length;
                }
                fixed (IntPtr* pp = ps)
                fixed (IntPtr* gp = gs)
                fixed (nuint* lp = ls)
                {
                    Support.Check(Native.rlx_optimizer_step_raw(H, cn.Ptr, (IntPtr)pp, (IntPtr)gp, (IntPtr)lp, IntPtr.Zero, IntPtr.Zero, (nuint)names.Length));
                }
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }
        }
    }

    public sealed partial class Trainer
    {
        /// <summary>
        /// Train <paramref name="steps"/> steps, pulling inputs from <paramref name="batch"/>
        /// (null when the data is exhausted); <paramref name="lr"/> sets each step's rate,
        /// <paramref name="onStep"/> returns false to stop. Returns the run summary.
        /// </summary>
        public Dictionary<string, object?> Run(int steps, Func<ulong, IDictionary<string, float[]>?> batch,
            Func<ulong, float>? lr = null, Func<ulong, float, bool>? onStep = null, int every = 1)
        {
            every = Math.Max(every, 1);
            ulong start = Steps();
            var losses = new List<float>();
            bool stopped = false;
            var t0 = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < steps; i++)
            {
                ulong s = start + (ulong)i;
                var b = batch(s);
                if (b is null) break;
                if (lr is not null) SetLr(lr(s));
                float loss = Step(b);
                losses.Add(loss);
                if (onStep is not null && (i + 1) % every == 0 && !onStep(s, loss)) { stopped = true; break; }
            }
            return new Dictionary<string, object?>
            {
                ["steps"] = losses.Count,
                ["stopped"] = stopped,
                ["first_loss"] = losses.Count > 0 ? losses[0] : null,
                ["last_loss"] = losses.Count > 0 ? losses[^1] : null,
                ["mean_loss"] = losses.Count > 0 ? losses.Average() : null,
                ["steps_per_second"] = losses.Count / Math.Max(t0.Elapsed.TotalSeconds, 1e-9),
                ["grad_norm"] = GradNorm(),
            };
        }
    }
}

/// <summary>A list of strings as a C <c>const char* const*</c>, in native memory.</summary>
internal sealed class CStrings : IDisposable
{
    public IntPtr Ptr { get; }
    public nuint Count { get; }
    private readonly IntPtr[] _strs;

    public CStrings(IEnumerable<string>? list)
    {
        if (list is null) { _strs = Array.Empty<IntPtr>(); return; }
        _strs = list.Select(Marshal.StringToCoTaskMemUTF8).ToArray();
        Count = (nuint)_strs.Length;
        Ptr = Marshal.AllocHGlobal(IntPtr.Size * Math.Max(_strs.Length, 1));
        Marshal.Copy(_strs, 0, Ptr, _strs.Length);
    }

    public void Dispose()
    {
        foreach (var s in _strs) Marshal.FreeCoTaskMem(s);
        if (Ptr != IntPtr.Zero) Marshal.FreeHGlobal(Ptr);
    }
}

/// <summary>Named tensors as the parallel C arrays a run takes, copied into native memory.</summary>
internal sealed class CInputs : IDisposable
{
    public IntPtr Names => _names.Ptr;
    public IntPtr Data { get; }
    public IntPtr Lens { get; }
    public IntPtr Dtypes => _dtypes?.Ptr ?? IntPtr.Zero;
    public nuint Count { get; }
    private readonly CStrings _names;
    private readonly CStrings? _dtypes;
    private readonly List<IntPtr> _bufs = new();

    public CInputs(IDictionary<string, float[]> m)
        : this(m.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray(),
               k => MemoryMarshal.AsBytes(m[k].AsSpan()).ToArray(), k => m[k].Length, null) { }

    public CInputs(IDictionary<string, TypedTensor> m)
        : this(m.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray(),
               k => m[k].Bytes, k => m[k].Bytes.Length, k => m[k].Dtype) { }

    private CInputs(string[] keys, Func<string, byte[]> bytes, Func<string, int> len, Func<string, string>? dtype)
    {
        _names = new CStrings(keys);
        if (dtype is not null) _dtypes = new CStrings(keys.Select(dtype));
        Count = (nuint)keys.Length;
        Data = Marshal.AllocHGlobal(IntPtr.Size * Math.Max(keys.Length, 1));
        Lens = Marshal.AllocHGlobal(IntPtr.Size * Math.Max(keys.Length, 1));
        for (int i = 0; i < keys.Length; i++)
        {
            var b = bytes(keys[i]);
            var p = Marshal.AllocHGlobal(Math.Max(b.Length, 1));
            Marshal.Copy(b, 0, p, b.Length);
            _bufs.Add(p);
            Marshal.WriteIntPtr(Data, i * IntPtr.Size, p);
            Marshal.WriteIntPtr(Lens, i * IntPtr.Size, (IntPtr)len(keys[i]));
        }
    }

    public void Dispose()
    {
        foreach (var p in _bufs) Marshal.FreeHGlobal(p);
        Marshal.FreeHGlobal(Data);
        Marshal.FreeHGlobal(Lens);
        _names.Dispose();
        _dtypes?.Dispose();
    }
}

internal static class Support
{
    internal static void Check(int rc)
    {
        if (rc == 0) return;
        var msg = Marshal.PtrToStringUTF8(RLX.Native.rlx_last_error()) ?? "";
        var kind = Enum.IsDefined(typeof(RlxException.ErrorKind), rc) ? (RlxException.ErrorKind)rc : RlxException.ErrorKind.Other;
        throw new RlxException(kind, msg);
    }

    internal static byte[]? Utf8(string? s) => s is null ? null : Encoding.UTF8.GetBytes(s + "\0");

    internal static byte[]? Json(object? v) => v is null ? null : Utf8(JsonSerializer.Serialize(v));

    internal static long[] Rows(long[][] rows, int w, bool splat)
    {
        var flat = new List<long>();
        for (int i = 0; i < rows.Length; i++)
        {
            var r = rows[i];
            if (r.Length == w) flat.AddRange(r);
            else if (splat && r.Length == 1) flat.AddRange(Enumerable.Repeat(r[0], w));
            else throw new RlxException(RlxException.ErrorKind.Type, $"row {i} needs {w} values, got {r.Length}");
        }
        flat.Add(0);
        return flat.ToArray();
    }

    internal static float[] FRows(float[][] rows, int w)
    {
        var flat = new List<float>();
        for (int i = 0; i < rows.Length; i++)
        {
            if (rows[i].Length != w) throw new RlxException(RlxException.ErrorKind.Type, $"row {i} needs {w} values");
            flat.AddRange(rows[i]);
        }
        flat.Add(0);
        return flat.ToArray();
    }

    internal static string TakeString(IntPtr p)
    {
        if (p == IntPtr.Zero) return "";
        var s = Marshal.PtrToStringUTF8(p) ?? "";
        RLX.Native.rlx_string_free(p);
        return s;
    }

    internal static string? TakeStringOrNull(IntPtr p) => p == IntPtr.Zero ? null : TakeString(p);

    internal static unsafe float[] TakeFloats(ref RlxArray a)
    {
        var v = new float[(int)a.Len];
        if (a.Ptr != IntPtr.Zero) Marshal.Copy(a.Ptr, v, 0, v.Length);
        fixed (RlxArray* p = &a) RLX.Native.rlx_f32s_release((IntPtr)p);
        return v;
    }

    internal static unsafe byte[] TakeBytes(ref RlxArray a)
    {
        var v = new byte[(int)a.Len];
        if (a.Ptr != IntPtr.Zero) Marshal.Copy(a.Ptr, v, 0, v.Length);
        fixed (RlxArray* p = &a) RLX.Native.rlx_bytes_release((IntPtr)p);
        return v;
    }

    internal static unsafe uint[] TakeUInts(ref RlxArray a)
    {
        var v = new uint[(int)a.Len];
        if (a.Ptr != IntPtr.Zero) new ReadOnlySpan<uint>((void*)a.Ptr, v.Length).CopyTo(v);
        fixed (RlxArray* p = &a) RLX.Native.rlx_u32s_release((IntPtr)p);
        return v;
    }

    internal static unsafe float[][] TakeOutputs(IntPtr o)
    {
        try
        {
            int n = (int)RLX.Native.rlx_outputs_len(o);
            var r = new float[n][];
            for (int i = 0; i < n; i++)
            {
                IntPtr d = IntPtr.Zero;
                nuint l = 0;
                Check(RLX.Native.rlx_outputs_get(o, (nuint)i, (IntPtr)(&d), (IntPtr)(&l)));
                r[i] = new float[(int)l];
                if (l > 0) Marshal.Copy(d, r[i], 0, (int)l);
            }
            return r;
        }
        finally { RLX.Native.rlx_outputs_free(o); }
    }

    internal static unsafe List<TypedTensor> TakeTypedOutputs(IntPtr o)
    {
        try
        {
            int n = (int)RLX.Native.rlx_typed_outputs_len(o);
            var r = new List<TypedTensor>();
            for (int i = 0; i < n; i++)
            {
                IntPtr d = IntPtr.Zero, t = IntPtr.Zero;
                nuint l = 0;
                Check(RLX.Native.rlx_typed_outputs_get(o, (nuint)i, (IntPtr)(&d), (IntPtr)(&l), (IntPtr)(&t)));
                var b = new byte[(int)l];
                if (l > 0) Marshal.Copy(d, b, 0, (int)l);
                r.Add(new TypedTensor(b, Marshal.PtrToStringUTF8(t) ?? ""));
            }
            return r;
        }
        finally { RLX.Native.rlx_typed_outputs_free(o); }
    }

    internal static unsafe Dictionary<string, float[]> TakeTensors(IntPtr m)
    {
        try
        {
            int n = (int)RLX.Native.rlx_tensor_map_len(m);
            var r = new Dictionary<string, float[]>();
            for (int i = 0; i < n; i++)
            {
                IntPtr name = IntPtr.Zero, d = IntPtr.Zero;
                nuint l = 0;
                Check(RLX.Native.rlx_tensor_map_get(m, (nuint)i, (IntPtr)(&name), (IntPtr)(&d), (IntPtr)(&l)));
                var v = new float[(int)l];
                if (l > 0) Marshal.Copy(d, v, 0, (int)l);
                r[Marshal.PtrToStringUTF8(name) ?? ""] = v;
            }
            return r;
        }
        finally { RLX.Native.rlx_tensor_map_free(m); }
    }

    internal static long[][] Shapes(string json) =>
        JsonNode.Parse(json)!.AsArray().Select(s => s!.AsArray().Select(d => d!.GetValue<long>()).ToArray()).ToArray();

    internal static Shape ShapeOf(string json)
    {
        var o = JsonNode.Parse(json)!.AsObject();
        return new Shape(o["dims"]!.AsArray().Select(d => d!.GetValue<long>()).ToArray(), o["dtype"]!.GetValue<string>());
    }

    internal static List<string> Strings(string json) =>
        JsonNode.Parse(json)!.AsArray().Select(s => s!.GetValue<string>()).ToList();
}
