# Reference for tests/fixtures/openblas_haswell_dgesdd.txt: run on the Linux reference
# (msi: scipy 1.17.1, OpenBLAS 0.3.30 + its reference LAPACK): python -I tools/openblas_ref_dgesdd.py out.txt
# scipy.linalg.svd(A, full_matrices=False) → dgesdd('S'). One line per case:
# "m n kind | <U col-major> | <s> | <Vt col-major>" (fnv:<hash> past 64 values); kind 0 random, 1 rank-deficient
# (last column = first), 2 graded (column j scaled by 10^-j).
import numpy as np, struct, sys
from scipy.linalg import svd
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
    return np.array(out)
def hx(v): return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
def fnv(vals):
    x = 0xcbf29ce484222325
    for v in vals:
        for b in struct.pack("<d", float(v)):
            x = ((x ^ b) * 0x100000001b3) & M
    return x
def enc(vals):
    vals = list(vals)
    return " ".join(hx(v) for v in vals) if len(vals) <= 64 else "fnv:%016x" % fnv(vals)
lines = []
for (m, n) in [(1, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (5, 5), (8, 6), (6, 8), (10, 10), (12, 7), (7, 12),
               (20, 20), (24, 25), (25, 25), (25, 30), (30, 25), (40, 25)]:
    for kind in (0, 1, 2):
        A = stream(0x5d5d + m * 101 + n * 7 + kind, m * n).reshape(m, n)
        if kind == 1 and n > 1:
            A[:, -1] = A[:, 0]
        if kind == 2:
            A = A * (10.0 ** -np.arange(n))[None, :]
        U, s, Vt = svd(A, full_matrices=False)
        lines.append(f"{m} {n} {kind} | " + " | ".join(enc(x) for x in (U.ravel(order="F"), s, Vt.ravel(order="F"))))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
