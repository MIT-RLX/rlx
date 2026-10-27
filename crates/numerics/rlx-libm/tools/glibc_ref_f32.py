# Reference for tests/vectors/glibc_aarch64_f32.txt: glibc's logf / sincosf on Linux aarch64 (the torch
# reference: Docker image exg-ref/fnirsatlas-linux-aarch64, glibc of python:3.12-slim), called through ctypes:
#   docker run --rm -v $PWD:/w <image> /opt/fnirsatlas-ref/.venv/bin/python -I /w/tools/glibc_ref_f32.py /w/tests/vectors/glibc_aarch64_f32.txt
# One line per strided range of input bit patterns: "<fn> <start> <end> <step> <count> <fnv of output bits>..."
# (sincosf: the sin hash then the cos hash). FNV-1a over the little-endian u32 bits.
import ctypes, struct, sys, platform
m = ctypes.CDLL("libm.so.6")
m.logf.restype = ctypes.c_float; m.logf.argtypes = [ctypes.c_float]
m.sincosf.restype = None; m.sincosf.argtypes = [ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
f = lambda u: struct.unpack("<f", struct.pack("<I", u))[0]
u = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
M = (1 << 64) - 1
def fnv(vals):
    h = 0xcbf29ce484222325
    for v in vals:
        for b in struct.pack("<I", v):
            h = ((h ^ b) * 0x100000001b3) & M
    return "%016x" % h
lines = [f"# {platform.machine()} glibc {platform.libc_ver()[1]}"]
for (a, b, s) in [(1, 0x7f800000, 4099), (0x3f000000, 0x3f800000, 61), (0x00000001, 0x00800000, 997)]:
    ins = range(a, b, s)
    lines.append(f"logf {a} {b} {s} {len(ins)} " + fnv(u(m.logf(f(i))) for i in ins))
sv, cv = ctypes.c_float(), ctypes.c_float()
def sc(i):
    m.sincosf(f(i), ctypes.byref(sv), ctypes.byref(cv)); return u(sv.value), u(cv.value)
for (a, b, s) in [(0, u(6.2831855), 509), (0, 0x7f800000, 4099), (0x80000000, 0xff800000, 8191), (u(100.0), u(140.0), 13)]:
    ins = range(a, b, s)
    r = [sc(i) for i in ins]
    lines.append(f"sincosf {a} {b} {s} {len(ins)} " + fnv(x for x, _ in r) + " " + fnv(y for _, y in r))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
