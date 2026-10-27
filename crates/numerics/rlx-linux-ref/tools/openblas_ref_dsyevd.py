# Reference for tests/fixtures/openblas_haswell_dsyevd.txt: run on the Linux reference
# (msi: numpy 2.3.5 / scipy 1.17.1, both scipy-openblas 0.3.30 = OpenBLAS + its reference LAPACK):
#   python -I tools/openblas_ref_dsyevd.py out.txt
# numpy.linalg.eigh(A) → LAPACK dsyevd('V', 'L') with numpy's workspace query (lwork = LOPT);
# scipy.linalg.lapack.dstevd(d, e) → dstedc('I') alone, on A's own tridiagonal form (dsytrd lower).
# One line per case: "n kind | <w> | <v col-major> | <d> | <e> | <dstevd w> | <dstevd z col-major>"
# (values as IEEE-754 hex, or fnv:<hash> past 64 values). Kinds 0-7 as the dsyevr fixture;
# kind 8: diagonal (every off-diagonal zero → dstedc splits into 1x1 blocks); kind 9: block-diagonal
# with blocks of 20 and 40 (splits below and above the 25-row dsteqr leaf size).
import numpy as np, struct, sys
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

def matrix2(n, kind):
    if kind == 8:
        return np.diag(np.array(stream(0x9d1a + n, n)) * 3.0)
    if kind == 9:
        A = np.zeros((n, n))
        start, sizes, s = 0, [20, 40], 0
        while start < n:
            b = min(sizes[s % 2], n - start); s += 1
            u = stream(0x5b10 + n + start, b * b)
            for i in range(b):
                for j in range(b):
                    A[start + i, start + j] = (u[i * b + j] + u[j * b + i]) * 0.5
            start += b
        return A
    return matrix(n, kind)
from scipy.linalg.lapack import dsytrd, dstevd
lines = []
for n in (2, 3, 4, 5, 6, 8, 9, 13, 16, 20, 25, 26, 27, 31, 32, 33, 40, 47, 50, 51, 52, 60, 64, 65, 72, 90, 100, 101, 128, 150, 200, 201, 257, 300):
    for kind in range(10):
        A = matrix2(n, kind)
        w, v = np.linalg.eigh(A)
        c, d, e, tau, info = dsytrd(np.asfortranarray(A), lower=1)
        assert info == 0
        wt, zt, info = dstevd(d, e, compute_v=1)
        assert info == 0
        lines.append(f"{n} {kind} | " + " | ".join(enc(x) for x in (w, v.ravel(order="F"), d, e, wt, zt.ravel(order="F"))))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
