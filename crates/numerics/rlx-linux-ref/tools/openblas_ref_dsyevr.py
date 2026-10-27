# Reference for tests/fixtures/openblas_haswell_dsyevr.txt: run on the Linux reference
# (msi: scipy 1.17.1, OpenBLAS 0.3.30 + its reference LAPACK): python -I tools/openblas_ref_dsyevr.py out.txt
# scipy.linalg.eigh(A) → dsyevr('V', 'A', 'L', lwork = 33n), and scipy.linalg.lapack.dsytrd(A, lower=1,
# lwork=28n) (the reduction as dsyevr runs it). One line per case:
# "n kind | <w hex> | <v hex, column-major> | <dsytrd: A' col-major> | <d> | <e> | <tau>" (fnv:<hash> past 64 values).
# kind 0: symmetric (U + Uᵀ)/2; kind 1: Gram X·Xᵀ + 1 with X n×(n/2+1) (a cluster of ~0 eigenvalues, plain
# sequential dot products); kind 2: Gram with X n×2n; kind 3: rounded to 1/4 (repeated eigenvalues);
# kind 4: Wilkinson W⁺ (diag |(n-1)/2 − i|, off-diagonal 1: near-equal pairs → MRRR clusters); kind 5: ones + 2·I
# (an (n−1)-fold eigenvalue); kind 6: Q·diag·Qᵀ, Q a Householder reflector, eigenvalues in tight groups
# (1 + i·1e-9, 2 + i·1e-13, 3 + i); kind 7: kind 0 scaled by 2^-500 (DSYEVR's ISCALE path).
import numpy as np, struct, sys
from scipy.linalg import eigh
from scipy.linalg.lapack import dsytrd
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
    return out
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
def matrix(n, kind):
    if kind == 7:
        return matrix(n, 0) * 2.0 ** -500
    if kind == 4:
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = abs((n - 1) / 2.0 - i)
            if i + 1 < n:
                A[i, i + 1] = A[i + 1, i] = 1.0
        return A
    if kind == 5:
        return np.ones((n, n)) + 2.0 * np.eye(n)
    if kind == 6:
        v = stream(0x7c11 + n, n)
        vtv = 0.0
        for x in v:
            vtv += x * x
        Q = [[(1.0 if i == j else 0.0) - (2.0 * v[i] * v[j]) / vtv for j in range(n)] for i in range(n)]
        d = [(1.0 + i * 1e-9) if i % 3 == 0 else (2.0 + i * 1e-13) if i % 3 == 1 else (3.0 + i) for i in range(n)]
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = 0.0
                for k in range(n):
                    s += Q[i][k] * d[k] * Q[j][k]
                A[i][j] = A[j][i] = s
        return np.array(A)
    if kind in (0, 3):
        u = stream(0x51e7 + n * 31 + kind, n * n)
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = (u[i * n + j] + u[j * n + i]) * 0.5
                if kind == 3:
                    A[i][j] = float(round(A[i][j] * 4.0)) / 4.0
        return np.array(A)
    f = n // 2 + 1 if kind == 1 else 2 * n
    u = stream(0x6a3d + n * 17 + kind, n * f)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for l in range(f):
                s += u[i * f + l] * u[j * f + l]
            A[i][j] = A[j][i] = s + 1.0
    return np.array(A)
lines = []
for n in (2, 3, 4, 5, 6, 8, 9, 13, 16, 20, 25, 31, 32, 33, 34, 40, 47, 54, 60, 63, 64, 65, 72, 90, 100, 128, 135, 136, 150, 199, 200, 201, 230, 257, 300):
    for kind in range(8):
        A = matrix(n, kind)
        w, v = eigh(A)
        c, d, e, tau, info = dsytrd(np.asfortranarray(A), lower=1, lwork=28 * n)
        assert info == 0
        lines.append(f"{n} {kind} | " + " | ".join(enc(x) for x in (w, v.ravel(order="F"), c.ravel(order="F"), d, e, tau)))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
