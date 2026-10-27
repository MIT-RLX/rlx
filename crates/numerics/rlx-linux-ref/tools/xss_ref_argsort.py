# Reference for tests/fixtures/xss_argsort.txt: run on the Linux reference (msi: NumPy 2.3.5,
# AVX2, so np.argsort dispatches to x86-simd-sort's avx2_argsort):
#   python -I tools/xss_ref_argsort.py out.txt
# Inputs from the xorshift stream: ints in [0, r) (r small → many ties) or doubles rounded to 1/q.
# One line: "dtype n r <idx…>" (over 64 indices: fnv of the little-endian u64s).
import numpy as np, struct, sys
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53))
    return np.array(out)
def fnv(vals):
    h = 0xcbf29ce484222325
    for v in vals:
        for b in struct.pack("<Q", int(v)):
            h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h
def out(idx):
    idx = list(idx)
    return " ".join(str(v) for v in idx) if len(idx) <= 64 else "fnv:%016x" % fnv(idx)
L = []
sizes = list(range(1, 70)) + [100, 127, 128, 129, 200, 255, 256, 257, 300, 400, 513, 700, 900]
for n in sizes:
    for r in (1, 2, 3, 7, 50, 10**9):
        u = stream(0x9e37 + n * 31 + r, n)
        v = np.floor(u * r).astype(np.int64)
        L.append(f"i64 {n} {r} " + out(np.argsort(v)))
        f = np.floor(u * r) / 4.0 - 1.0
        L.append(f"f64 {n} {r} " + out(np.argsort(f)))
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print(len(L), "cases")
