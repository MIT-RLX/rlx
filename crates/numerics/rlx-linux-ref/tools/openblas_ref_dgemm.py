# Reference for tests/fixtures/openblas_haswell_dgemm.txt: run on the Linux reference
# (msi: numpy 2.3.5, OpenBLAS 0.3.30 Haswell): python -I tools/openblas_ref_dgemm.py out.txt
# NumPy routes a 2-D @ 2-D product (m, n > 1, distinct buffers) to cblas_dgemm(RowMajor, …);
# a transposed view is passed as Trans. One line per case: "m n k ta tb <C hex, row-major>".
import numpy as np, struct, sys
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
    return np.array(out)
def hx(v): return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
def fnv(vals):
    h = 0xcbf29ce484222325
    for v in vals:
        for b in struct.pack("<d", float(v)):
            h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h
def out(vals):  # over 64 values: an FNV-1a hash of the f64 bits (little-endian)
    vals = list(vals)
    return " ".join(hx(v) for v in vals) if len(vals) <= 64 else "fnv:%016x" % fnv(vals)
shapes = [(m, n, k) for m in (2, 3, 5, 8) for n in (2, 3, 7, 9) for k in (1, 2, 3, 8, 17, 255, 256, 257, 300)]
shapes += [(2, 1000, 2), (2, 2, 511), (2, 2, 512), (2, 2, 513), (3, 2, 600), (2, 3, 767), (2, 2, 1000), (2, 2, 3001), (40, 40, 100),
           (3, 1000, 9), (5, 600, 20), (7, 1100, 30), (1001, 3, 9), (3, 3, 5),
           (55, 256, 23), (23, 256, 55), (60, 90, 90), (91, 89, 63)]  # 262144 < mnk < 524288: still 1 thread
shapes += [(146, 256, 27), (27, 256, 146), (100, 100, 100), (200, 150, 90), (301, 300, 77), (37, 251, 61), (251, 37, 61),
           (129, 513, 9), (64, 64, 257), (7, 3001, 30), (3001, 7, 30), (513, 513, 5), (45, 45, 600), (99, 1001, 13)]  # threaded
lines = []
for m, n, k in shapes:
    for ta in (0, 1):
        for tb in (0, 1):
            u = stream(0x3131 + m * 7 + n * 131 + k * 1031 + ta * 2 + tb, m * k + k * n)
            A = u[:m * k].reshape((k, m) if ta else (m, k)); B = u[m * k:].reshape((n, k) if tb else (k, n))
            C = (A.T if ta else A) @ (B.T if tb else B)
            lines.append(f"{m} {n} {k} {ta} {tb} " + out(C.ravel()))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
