# Reference for tests/fixtures/openblas_haswell_dgelsd_big.txt (blocked DGEBRD/DORMQR/DGEQRF, divide and conquer): run on the Linux reference
# (msi: numpy 2.3.5, OpenBLAS 0.3.30 + its reference LAPACK): python -I tools/openblas_ref_dgelsd.py out.txt
# np.linalg.lstsq(A, B, rcond) → dgelsd. One line per case:
# "m n nrhs kind rc | <x hex (row-major n×nrhs)> | rank | <s hex>"; kind 0 random, 1 rank-deficient
# (column 0 duplicated into the last column), 2 Vandermonde (polyfit-shaped, column-scaled); rc 0 = default rcond, 1 = 1e-3.
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
def out(vals):
    vals = list(vals)
    return " ".join(hx(v) for v in vals) if len(vals) <= 64 else "fnv:%016x" % fnv(vals)
def case(m, n, nrhs, kind):
    u = stream(0x5a5a + m * 977 + n * 131 + nrhs * 7 + kind, m * n + m * nrhs)
    A = u[:m * n].reshape(m, n).copy(); B = u[m * n:].reshape(m, nrhs).copy()
    if kind == 1 and n > 1:
        A[:, -1] = A[:, 0]
    if kind == 2:
        x = np.arange(1, m + 1, dtype=float)
        A = np.vander(x, n)
        A /= np.sqrt((A * A).sum(axis=0))
    return A, B
shapes = [(n, n) for n in (33, 40, 48, 64, 80, 100, 129, 160, 240)] + [(300, 150), (400, 140), (100, 120), (90, 100), (60, 200), (100, 170)]
lines = []
for m, n in shapes:
    for nrhs in (1, 2, 3):
        for kind in (0, 1):
            for rc in (0, 1):
                A, B = case(m, n, nrhs, kind)
                x, res, rank, s = np.linalg.lstsq(A, B, rcond=(None if rc == 0 else 1e-3))
                lines.append(f"{m} {n} {nrhs} {kind} {rc} | " + out(x.ravel()) + f" | {rank} | " + out(s))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
