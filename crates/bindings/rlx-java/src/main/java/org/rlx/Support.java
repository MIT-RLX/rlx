// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import static java.lang.foreign.ValueLayout.*;

/**
 * What the generated {@link RLX} stands on: loading librlx_capi, calling a
 * function by name, the argument and result converters, and JSON for options
 * and reports. The library is {@code -Drlx.capi.lib=…} or {@code RLX_CAPI_LIB}
 * when set, else the one bundled in the jar ({@code native/<os>-<arch>/},
 * extracted to a temporary file), else librlx_capi from the workspace's
 * target/debug (a checkout).
 */
final class Support {
    private Support() {}

    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup LIB = SymbolLookup.libraryLookup(libraryPath(), Arena.global());
    private static final Map<String, MethodHandle> HANDLES = new ConcurrentHashMap<>();
    static final Cleaner CLEANER = Cleaner.create();

    private static Path libraryPath() {
        String p = System.getProperty("rlx.capi.lib", System.getenv("RLX_CAPI_LIB"));
        if (p != null) return Path.of(p);
        String os = System.getProperty("os.name").toLowerCase();
        String file = os.contains("mac") ? "librlx_capi.dylib" : os.contains("win") ? "rlx_capi.dll" : "librlx_capi.so";
        String arch = System.getProperty("os.arch").matches("aarch64|arm64") ? "arm64" : "x64";
        String platform = (os.contains("mac") ? "macos" : os.contains("win") ? "windows" : "linux") + "-" + arch;
        try (InputStream in = Support.class.getResourceAsStream("/native/" + platform + "/" + file)) {
            if (in != null) {
                Path dir = Files.createTempDirectory("rlx-capi");
                Path lib = dir.resolve(file);
                Files.copy(in, lib);
                lib.toFile().deleteOnExit();
                dir.toFile().deleteOnExit();
                return lib;
            }
        } catch (IOException e) {
            throw new UncheckedIOException("extracting the bundled " + file, e);
        }
        return Path.of(System.getProperty("rlx.workspace", "../../.."), "target", "debug", file);
    }

    /** Call C function {@code name}, looked up (and its handle built) on first use. */
    static Object call(String name, FunctionDescriptor fd, Object... args) {
        MethodHandle h = HANDLES.computeIfAbsent(name, n -> LINKER.downcallHandle(
                LIB.find(n).orElseThrow(() -> new RLXException(RLXException.Kind.RUNTIME,
                        n + " is not in this rlx-capi build (an optional feature?)")), fd));
        try {
            return h.invokeWithArguments(args);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    static void check(int rc) {
        if (rc == 0) return;
        MemorySegment msg = (MemorySegment) call("rlx_last_error", FunctionDescriptor.of(ADDRESS));
        throw new RLXException(RLXException.Kind.of(rc), msg.reinterpret(Long.MAX_VALUE).getString(0));
    }

    /** The base of every handle class: owns the C pointer, frees it on close or collection. */
    abstract static class Handle implements AutoCloseable {
        final MemorySegment h;
        private final Cleaner.Cleanable cleanable;

        Handle(MemorySegment h, String free) {
            if (h.equals(MemorySegment.NULL)) throw new RLXException(RLXException.Kind.RUNTIME, "rlx returned NULL");
            this.h = h;
            MemorySegment raw = h;
            this.cleanable = CLEANER.register(this, () -> call(free, FunctionDescriptor.ofVoid(ADDRESS), raw));
        }

        @Override
        public void close() {
            cleanable.clean();
        }
    }

    // ── arguments ───────────────────────────────────────────

    static MemorySegment cstrOrNull(Arena a, String s) {
        return s == null ? MemorySegment.NULL : a.allocateFrom(s);
    }

    static MemorySegment json(Arena a, Object v) {
        return v == null ? MemorySegment.NULL : a.allocateFrom(toJson(v));
    }

    static MemorySegment boxLong(Arena a, Long v) {
        return v == null ? MemorySegment.NULL : a.allocateFrom(JAVA_LONG, v);
    }

    static MemorySegment boxFloat(Arena a, Float v) {
        return v == null ? MemorySegment.NULL : a.allocateFrom(JAVA_FLOAT, v);
    }

    static MemorySegment floats(Arena a, float[] v) {
        return v.length == 0 ? MemorySegment.NULL : a.allocateFrom(JAVA_FLOAT, v);
    }

    static MemorySegment bytes(Arena a, byte[] v) {
        return v.length == 0 ? MemorySegment.NULL : a.allocateFrom(JAVA_BYTE, v);
    }

    static MemorySegment ints(Arena a, int[] v) {
        return v.length == 0 ? MemorySegment.NULL : a.allocateFrom(JAVA_INT, v);
    }

    static MemorySegment longs(Arena a, long[] v) {
        return a.allocateFrom(JAVA_LONG, v.length == 0 ? new long[] {0} : v);
    }

    static long[] rows(long[][] rows, int w, boolean splat) {
        long[] out = new long[rows.length * w];
        for (int i = 0; i < rows.length; i++) {
            long[] r = rows[i];
            if (r.length == w) System.arraycopy(r, 0, out, i * w, w);
            else if (splat && r.length == 1) java.util.Arrays.fill(out, i * w, i * w + w, r[0]);
            else throw new RLXException(RLXException.Kind.TYPE, "row " + i + " needs " + w + " values, got " + r.length);
        }
        return out;
    }

    static float[] frows(float[][] rows, int w) {
        float[] out = new float[rows.length * w];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i].length != w) throw new RLXException(RLXException.Kind.TYPE, "row " + i + " needs " + w + " values");
            System.arraycopy(rows[i], 0, out, i * w, w);
        }
        return out;
    }

    static MemorySegment cstrs(Arena a, String[] v) {
        if (v == null) return MemorySegment.NULL;
        MemorySegment arr = a.allocate(ADDRESS, Math.max(v.length, 1));
        for (int i = 0; i < v.length; i++) arr.setAtIndex(ADDRESS, i, a.allocateFrom(v[i]));
        return arr;
    }

    record CInputs(MemorySegment names, MemorySegment data, MemorySegment lens, MemorySegment dtypes, long n) {}

    static CInputs inputs(Arena a, Map<String, float[]> m) {
        var sorted = new TreeMap<>(m);
        int n = sorted.size();
        MemorySegment names = a.allocate(ADDRESS, Math.max(n, 1));
        MemorySegment data = a.allocate(ADDRESS, Math.max(n, 1));
        MemorySegment lens = a.allocate(JAVA_LONG, Math.max(n, 1));
        int i = 0;
        for (var e : sorted.entrySet()) {
            names.setAtIndex(ADDRESS, i, a.allocateFrom(e.getKey()));
            data.setAtIndex(ADDRESS, i, floats(a, e.getValue()));
            lens.setAtIndex(JAVA_LONG, i, e.getValue().length);
            i++;
        }
        return new CInputs(names, data, lens, MemorySegment.NULL, n);
    }

    static CInputs typedInputs(Arena a, Map<String, TypedTensor> m) {
        var sorted = new TreeMap<>(m);
        int n = sorted.size();
        MemorySegment names = a.allocate(ADDRESS, Math.max(n, 1));
        MemorySegment data = a.allocate(ADDRESS, Math.max(n, 1));
        MemorySegment lens = a.allocate(JAVA_LONG, Math.max(n, 1));
        MemorySegment dtypes = a.allocate(ADDRESS, Math.max(n, 1));
        int i = 0;
        for (var e : sorted.entrySet()) {
            names.setAtIndex(ADDRESS, i, a.allocateFrom(e.getKey()));
            data.setAtIndex(ADDRESS, i, bytes(a, e.getValue().bytes()));
            lens.setAtIndex(JAVA_LONG, i, e.getValue().bytes().length);
            dtypes.setAtIndex(ADDRESS, i, a.allocateFrom(e.getValue().dtype()));
            i++;
        }
        return new CInputs(names, data, lens, dtypes, n);
    }

    // ── results ─────────────────────────────────────────────

    static String takeString(MemorySegment out) {
        MemorySegment p = out.get(ADDRESS, 0);
        if (p.equals(MemorySegment.NULL)) return "";
        String s = p.reinterpret(Long.MAX_VALUE).getString(0);
        call("rlx_string_free", FunctionDescriptor.ofVoid(ADDRESS), p);
        return s;
    }

    static String takeStringOrNull(MemorySegment out) {
        return out.get(ADDRESS, 0).equals(MemorySegment.NULL) ? null : takeString(out);
    }

    private static MemorySegment arrayAt(MemorySegment out, long elem) {
        MemorySegment p = out.get(ADDRESS, 0);
        long len = out.get(JAVA_LONG, 8);
        return p.equals(MemorySegment.NULL) ? MemorySegment.NULL : p.reinterpret(len * elem);
    }

    static float[] takeFloats(MemorySegment out) {
        MemorySegment p = arrayAt(out, 4);
        float[] v = p.equals(MemorySegment.NULL) ? new float[0] : p.toArray(JAVA_FLOAT);
        call("rlx_f32s_release", FunctionDescriptor.ofVoid(ADDRESS), out);
        return v;
    }

    static byte[] takeBytes(MemorySegment out) {
        MemorySegment p = arrayAt(out, 1);
        byte[] v = p.equals(MemorySegment.NULL) ? new byte[0] : p.toArray(JAVA_BYTE);
        call("rlx_bytes_release", FunctionDescriptor.ofVoid(ADDRESS), out);
        return v;
    }

    static int[] takeInts(MemorySegment out) {
        MemorySegment p = arrayAt(out, 4);
        int[] v = p.equals(MemorySegment.NULL) ? new int[0] : p.toArray(JAVA_INT);
        call("rlx_u32s_release", FunctionDescriptor.ofVoid(ADDRESS), out);
        return v;
    }

    static float[][] takeOutputs(MemorySegment out) {
        MemorySegment o = out.get(ADDRESS, 0);
        int n = (int) (long) call("rlx_outputs_len", FunctionDescriptor.of(JAVA_LONG, ADDRESS), o);
        float[][] result = new float[n][];
        try (Arena a = Arena.ofConfined()) {
            for (int i = 0; i < n; i++) {
                MemorySegment d = a.allocate(ADDRESS), l = a.allocate(JAVA_LONG);
                check((int) call("rlx_outputs_get", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS), o, (long) i, d, l));
                long len = l.get(JAVA_LONG, 0);
                result[i] = len == 0 ? new float[0] : d.get(ADDRESS, 0).reinterpret(len * 4).toArray(JAVA_FLOAT);
            }
        } finally {
            call("rlx_outputs_free", FunctionDescriptor.ofVoid(ADDRESS), o);
        }
        return result;
    }

    static List<TypedTensor> takeTypedOutputs(MemorySegment out) {
        MemorySegment o = out.get(ADDRESS, 0);
        int n = (int) (long) call("rlx_typed_outputs_len", FunctionDescriptor.of(JAVA_LONG, ADDRESS), o);
        List<TypedTensor> result = new ArrayList<>();
        try (Arena a = Arena.ofConfined()) {
            for (int i = 0; i < n; i++) {
                MemorySegment d = a.allocate(ADDRESS), l = a.allocate(JAVA_LONG), t = a.allocate(ADDRESS);
                check((int) call("rlx_typed_outputs_get", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS), o, (long) i, d, l, t));
                long len = l.get(JAVA_LONG, 0);
                byte[] b = len == 0 ? new byte[0] : d.get(ADDRESS, 0).reinterpret(len).toArray(JAVA_BYTE);
                result.add(new TypedTensor(b, t.get(ADDRESS, 0).reinterpret(Long.MAX_VALUE).getString(0)));
            }
        } finally {
            call("rlx_typed_outputs_free", FunctionDescriptor.ofVoid(ADDRESS), o);
        }
        return result;
    }

    static Map<String, float[]> takeTensors(MemorySegment out) {
        MemorySegment m = out.get(ADDRESS, 0);
        int n = (int) (long) call("rlx_tensor_map_len", FunctionDescriptor.of(JAVA_LONG, ADDRESS), m);
        Map<String, float[]> result = new LinkedHashMap<>();
        try (Arena a = Arena.ofConfined()) {
            for (int i = 0; i < n; i++) {
                MemorySegment nm = a.allocate(ADDRESS), d = a.allocate(ADDRESS), l = a.allocate(JAVA_LONG);
                check((int) call("rlx_tensor_map_get", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS), m, (long) i, nm, d, l));
                long len = l.get(JAVA_LONG, 0);
                result.put(nm.get(ADDRESS, 0).reinterpret(Long.MAX_VALUE).getString(0),
                        len == 0 ? new float[0] : d.get(ADDRESS, 0).reinterpret(len * 4).toArray(JAVA_FLOAT));
            }
        } finally {
            call("rlx_tensor_map_free", FunctionDescriptor.ofVoid(ADDRESS), m);
        }
        return result;
    }

    // ── JSON ────────────────────────────────────────────────

    static String toJson(Object v) {
        StringBuilder sb = new StringBuilder();
        writeJson(sb, v);
        return sb.toString();
    }

    private static void writeJson(StringBuilder sb, Object v) {
        if (v == null) sb.append("null");
        else if (v instanceof Boolean || v instanceof Integer || v instanceof Long || v instanceof Short) sb.append(v);
        else if (v instanceof Number n) {
            double d = n.doubleValue();
            sb.append(Double.isFinite(d) ? String.valueOf(d) : "null");
        } else if (v instanceof CharSequence || v instanceof Character) {
            sb.append('"');
            for (char c : v.toString().toCharArray()) {
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    default -> {
                        if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                        else sb.append(c);
                    }
                }
            }
            sb.append('"');
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (var e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                writeJson(sb, String.valueOf(e.getKey()));
                sb.append(':');
                writeJson(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof Iterable<?> it) {
            sb.append('[');
            boolean first = true;
            for (Object x : it) {
                if (!first) sb.append(',');
                first = false;
                writeJson(sb, x);
            }
            sb.append(']');
        } else if (v.getClass().isArray()) {
            int n = java.lang.reflect.Array.getLength(v);
            List<Object> list = new ArrayList<>(n);
            for (int i = 0; i < n; i++) list.add(java.lang.reflect.Array.get(v, i));
            writeJson(sb, list);
        } else {
            throw new RLXException(RLXException.Kind.TYPE, "options are not JSON-encodable: " + v.getClass());
        }
    }

    /** JSON text → {@code Map<String, Object>} / {@code List<Object>} / Long / Double / String / Boolean / null. */
    static Object parse(String s) {
        int[] i = {0};
        Object v = value(s, i);
        return v;
    }

    private static void skip(String s, int[] i) {
        while (i[0] < s.length() && Character.isWhitespace(s.charAt(i[0]))) i[0]++;
    }

    private static Object value(String s, int[] i) {
        skip(s, i);
        char c = s.charAt(i[0]);
        if (c == '{') {
            Map<String, Object> m = new LinkedHashMap<>();
            i[0]++;
            skip(s, i);
            if (s.charAt(i[0]) == '}') { i[0]++; return m; }
            while (true) {
                skip(s, i);
                String k = string(s, i);
                skip(s, i);
                i[0]++; // ':'
                m.put(k, value(s, i));
                skip(s, i);
                if (s.charAt(i[0]++) == '}') return m;
            }
        }
        if (c == '[') {
            List<Object> l = new ArrayList<>();
            i[0]++;
            skip(s, i);
            if (s.charAt(i[0]) == ']') { i[0]++; return l; }
            while (true) {
                l.add(value(s, i));
                skip(s, i);
                if (s.charAt(i[0]++) == ']') return l;
            }
        }
        if (c == '"') return string(s, i);
        if (s.startsWith("true", i[0])) { i[0] += 4; return Boolean.TRUE; }
        if (s.startsWith("false", i[0])) { i[0] += 5; return Boolean.FALSE; }
        if (s.startsWith("null", i[0])) { i[0] += 4; return null; }
        int start = i[0];
        while (i[0] < s.length() && "+-.eE0123456789".indexOf(s.charAt(i[0])) >= 0) i[0]++;
        String num = s.substring(start, i[0]);
        if (num.contains(".") || num.contains("e") || num.contains("E")) return Double.parseDouble(num);
        return Long.parseLong(num);
    }

    private static String string(String s, int[] i) {
        StringBuilder sb = new StringBuilder();
        i[0]++;
        while (s.charAt(i[0]) != '"') {
            char c = s.charAt(i[0]++);
            if (c == '\\') {
                char e = s.charAt(i[0]++);
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case 'r' -> sb.append('\r');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> { sb.append((char) Integer.parseInt(s.substring(i[0], i[0] + 4), 16)); i[0] += 4; }
                    default -> sb.append(e);
                }
            } else sb.append(c);
        }
        i[0]++;
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> object(String s) {
        return (Map<String, Object>) parse(s);
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> rowsOf(String s) {
        return (List<Map<String, Object>>) parse(s);
    }

    @SuppressWarnings("unchecked")
    static List<String> strings(String s) {
        return (List<String>) parse(s);
    }

    @SuppressWarnings("unchecked")
    static long[][] shapes(String s) {
        List<Object> outer = (List<Object>) parse(s);
        long[][] r = new long[outer.size()][];
        for (int i = 0; i < r.length; i++) {
            List<Object> dims = (List<Object>) outer.get(i);
            r[i] = dims.stream().mapToLong(x -> ((Number) x).longValue()).toArray();
        }
        return r;
    }

    @SuppressWarnings("unchecked")
    static Shape shape(String s) {
        Map<String, Object> m = object(s);
        List<Object> dims = (List<Object>) m.get("dims");
        return new Shape(dims.stream().mapToLong(x -> ((Number) x).longValue()).toArray(), (String) m.get("dtype"));
    }
}
