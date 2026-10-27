# Reference for tests/fixtures/openblas_haswell_dgemv.txt: run on the Linux reference
# (msi: numpy 2.3.5, OpenBLAS 0.3.30 Haswell): python -I tools/openblas_ref_dgemv.py out.txt
# NumPy routes 2-D @ 1-D to cblas_dgemv(RowMajor, NoTrans) and Aᵀ @ x (a transposed view) to
# Trans; a strided x (x[::2]) is passed as incx = 2. One line: "m n trans incx <y hex>" (over 64 values: "fnv:<FNV-1a of the bits>").
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
dims = [2, 3, 4, 5, 6, 7, 8, 9, 13, 31]
shapes = [(m, n) for m in dims for n in dims] + [(2, 2047), (3, 2048), (2, 2049), (4, 2050), (2, 4100), (2047, 2), (2049, 3), (4101, 2), (90, 100), (100, 92),
          (55, 256), (256, 55), (400, 1000), (1001, 399), (3, 5000), (5001, 3), (99, 4100), (4100, 99), (2048, 2048 - 4)]
# threaded (m·n ≥ 460800): column / row splits, odd per-thread widths, the split-x reduction (m·20 ≤ 1024)
shapes += [(54, 9996), (9996, 54), (18, 9996), (9996, 18), (85, 9996), (9996, 85), (20, 30000), (30000, 20), (51, 10000),
           (10000, 51), (52, 10000), (10000, 52), (3, 160000), (160000, 3), (701, 701), (123, 4001), (4001, 123)]
lines = []
for m, n in shapes:
    for trans in (0, 1):
        for incx in (1, 2):
            lx = n if trans == 0 else m
            u = stream(0x4242 + m * 3 + n * 101 + trans * 7 + incx, m * n + lx * incx)
            A = u[:m * n].reshape(m, n); x = u[m * n:][::incx]
            y = A @ x if trans == 0 else A.T @ x   # Aᵀ: a transposed view → Trans, lda = n
            lines.append(f"{m} {n} {trans} {incx} " + out(y))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
