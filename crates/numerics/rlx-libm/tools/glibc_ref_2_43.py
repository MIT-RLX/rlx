# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Reference for tests/vectors/glibc_{x86_64,aarch64}_2_43.txt: glibc 2.43's libm (Ubuntu 26.04's
# 2.43-2ubuntu2.4) called through ctypes, on Linux x86-64 with FMA (the `*_fma` variants the root
# ports model) and on Linux aarch64 (the `rlx_libm::aarch64` ports where that build differs):
#   python3 -I tools/glibc_ref_2_43.py tests/vectors/glibc_x86_64_2_43.txt      # on x86-64 Linux
#   docker run --rm --platform linux/arm64 -v $PWD:/w <ubuntu:26.04 + python3> \
#       python3 -I /w/tools/glibc_ref_2_43.py /w/tests/vectors/glibc_aarch64_2_43.txt
# One line per input set: "<fn> <kind> <a> <b> <count> <fnv of output bits>".
#   kind "stride": f32 bit patterns a, a+b, a+2b, ... (count of them)
#   kind "bits":   `count` xorshift64 draws (seed a) taken as raw bit patterns
#   kind "exp":    `count` draws with the biased exponent forced into [a, b] (sign, mantissa random)
# Two- and three-argument functions take two / three draws per point; `sincos` / `sincosf` hash
# sin then cos per point. NaN outputs hash as the canonical quiet NaN.
# Every symbol rlx-libm exports must have a line (tests/glibc_2_43_vectors.rs checks that).
# FNV-1a over little-endian output bits. Inputs come from integer ops only, so the Rust test
# (tests/glibc_2_43_vectors.rs) regenerates them exactly.
import ctypes, struct, sys, platform

m = ctypes.CDLL("libm.so.6")
M64 = (1 << 64) - 1


def xorshift(seed):
    s = seed
    while True:
        s ^= (s << 13) & M64
        s ^= s >> 7
        s ^= (s << 17) & M64
        yield s


def fnv(words, width):
    h = 0xCBF29CE484222325
    for w in words:
        for byte in w.to_bytes(width, "little"):
            h = ((h ^ byte) * 0x100000001B3) & M64
    return "%016x" % h


F32, F64 = (ctypes.c_float, "<f", "<I", 4, 23, 0xFF), (ctypes.c_double, "<d", "<Q", 8, 52, 0x7FF)


def bits_of(t, x):
    v = struct.unpack(t[2], struct.pack(t[1], x))[0]
    if x != x:  # NaN
        v = 0x7FC00000 if t is F32 else 0x7FF8000000000000
    return v


def val(t, b):
    return struct.unpack(t[1], struct.pack(t[2], b))[0]


def draw(t, kind, g, a, b):
    raw = next(g)
    if t is F32:
        raw &= 0xFFFFFFFF
    if kind == "bits":
        return raw
    mbits, emask = t[4], t[5]
    sign = raw >> (t[3] * 8 - 1)
    mant = raw & ((1 << mbits) - 1)
    e = a + (raw >> mbits & 0xFFFF) % (b - a + 1)
    return (sign << (t[3] * 8 - 1)) | (e << mbits) | mant


def line(name, t, arity, kind, a, b, count):
    if name in ("sincos", "sincosf"):
        fn = getattr(m, name)
        fn.restype = None
        fn.argtypes = [t[0], ctypes.POINTER(t[0]), ctypes.POINTER(t[0])]
        sv, cv = t[0](), t[0]()

        def call(x):
            fn(x, ctypes.byref(sv), ctypes.byref(cv))
            return [sv.value, cv.value]
    else:
        fn = getattr(m, name)
        fn.restype = t[0]
        fn.argtypes = [t[0]] * arity

        def call(*args):
            return [fn(*args)]
    out = []
    if kind == "stride":
        for i in range(count):
            out += [bits_of(t, v) for v in call(val(t, a + i * b))]
    else:
        g = xorshift(a if kind == "bits" else (a * 0x9E3779B97F4A7C15 + b) & M64 or 1)
        for _ in range(count):
            args = [val(t, draw(t, kind, g, a, b)) for _ in range(arity)]
            out += [bits_of(t, v) for v in call(*args)]
    return f"{name} {kind} {a} {b} {count} {fnv(out, t[3])}"


lines = [f"# {platform.machine()} glibc {platform.libc_ver()[1]}"]
unary32 = (
    "acosf asinf atanf acoshf asinhf atanhf cbrtf coshf sinhf tanf expm1f log10f erff erfcf lgammaf "
    "tgammaf exp2f exp10f log2f expf logf log1pf tanhf sinf cosf sincosf "
    "floorf ceilf truncf roundf rintf roundevenf"
).split()
for n in unary32:
    lines.append(line(n, F32, 1, "stride", 0, 4099, (1 << 32) // 4099))
    lines.append(line(n, F32, 1, "exp", 100, 160, 200000))
for n in "powf atan2f hypotf fmodf fdimf".split():
    lines.append(line(n, F32, 2, "bits", 0x9E3779B97F4A7C15, 0, 1000000))
    lines.append(line(n, F32, 2, "exp", 110, 140, 1000000))
lines.append(line("fmaf", F32, 3, "bits", 0x9E3779B97F4A7C15, 0, 1000000))
lines.append(line("fmaf", F32, 3, "exp", 100, 155, 1000000))
unary64 = (
    "tanh cbrt cosh sinh log10 exp2 exp10 erf atan asin acos atanh asinh acosh "
    "exp log log2 expm1 log1p erfc sin cos tan sincos floor ceil trunc round rint roundeven"
).split()
for n in unary64:
    lines.append(line(n, F64, 1, "bits", 0x2545F4914F6CDD1D, 0, 1000000))
    lines.append(line(n, F64, 1, "exp", 960, 1080, 1000000))
# the trig argument reduction: every exponent up to 2^1023 (Payne–Hanek arm)
for n in "sin cos tan sincos".split():
    lines.append(line(n, F64, 1, "exp", 1040, 2046, 1000000))
for n in ["tgamma", "lgamma"]:
    lines.append(line(n, F64, 1, "bits", 0x2545F4914F6CDD1D, 0, 1000000))
    lines.append(line(n, F64, 1, "exp", 1010, 1031, 2000000))
lines.append(line("atan2", F64, 2, "bits", 0x5851F42D4C957F2D, 0, 1000000))
lines.append(line("atan2", F64, 2, "exp", 900, 1140, 1000000))
for n in "pow hypot fmod fdim".split():
    lines.append(line(n, F64, 2, "bits", 0x5851F42D4C957F2D, 0, 1000000))
    lines.append(line(n, F64, 2, "exp", 900, 1140, 1000000))
# pow where neither overflows nor underflows: |x| near 1, moderate |y|
lines.append(line("pow", F64, 2, "exp", 1015, 1030, 2000000))
lines.append(line("fma", F64, 3, "bits", 0x5851F42D4C957F2D, 0, 1000000))
lines.append(line("fma", F64, 3, "exp", 960, 1080, 1000000))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
