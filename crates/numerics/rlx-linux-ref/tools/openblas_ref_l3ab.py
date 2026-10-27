# Reference for tests/fixtures/openblas_haswell_l3ab.txt: run on the Linux reference
# (msi: SciPy 1.17.1 on OpenBLAS 0.3.30 Haswell): python -I tools/openblas_ref_l3ab.py out.txt
# scipy.linalg.blas dtrmm (every side/uplo/trans/diag) and dgemm/dgemv with general alpha/beta,
# Fortran-order inputs from the xorshift stream. Lines (over 64 outputs: fnv of the bits):
#   "trmm m n side lower trans unit ai | <B' col-major>"  (A is k×k, k = m (left) or n (right))
#   "gemm m n k ta tb ai bi | <C' col-major>"   "gemv m n trans ai bi | <y'>"
import numpy as np, struct, sys
import scipy.linalg.blas as B
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
ALPHA = [1.0, -1.0, 0.7]
BETA = [0.0, 1.0, -0.3]
F = lambda a, r, c: np.asfortranarray(a.reshape(c, r).T)   # r×c Fortran matrix from a flat stream
L = []
dims = [1, 2, 3, 4, 5, 7, 8, 9, 12, 17, 32, 33]
for m in dims + [229, 256, 299]:
    for n in [1, 2, 3, 5, 8, 9, 12, 16, 31, 40, 250]:
        for side in (0, 1):
            k = m if side == 0 else n
            if k > 64: continue
            for lower in (0, 1):
                for trans in (0, 1):
                    for unit in (0, 1):
                        for ai in range(3):
                            seed = 0x6161 + m * 101 + n * 7 + side * 3 + lower * 5 + trans * 11 + unit * 13 + ai * 17
                            u = stream(seed, k * k + m * n)
                            a = F(u[:k * k], k, k); b = F(u[k * k:], m, n)
                            r = B.dtrmm(ALPHA[ai], a, b, side=side, lower=lower, trans_a=trans, diag=unit)
                            L.append(f"trmm {m} {n} {side} {lower} {trans} {unit} {ai} | " + out(r.T.ravel()))
for (m, n, k) in [(1, 1, 1), (3, 5, 7), (8, 8, 8), (9, 4, 33), (33, 31, 2), (64, 3, 40), (240, 32, 33), (40, 40, 600),
                  (229, 12, 217), (217, 229, 12), (256, 14, 243), (243, 256, 14), (299, 18, 280), (280, 299, 18)]:  # DORMQR's DLARFB
    for ta in (0, 1):
        for tb in (0, 1):
            for ai in range(3):
                for bi in range(3):
                    seed = 0x7272 + m * 3 + n * 5 + k * 7 + ta + tb * 2 + ai * 9 + bi * 27
                    u = stream(seed, m * k + k * n + m * n)
                    a = F(u[:m * k], k, m) if ta else F(u[:m * k], m, k)
                    b = F(u[m * k:m * k + k * n], n, k) if tb else F(u[m * k:m * k + k * n], k, n)
                    c = F(u[m * k + k * n:], m, n)
                    r = B.dgemm(ALPHA[ai], a, b, BETA[bi], c, trans_a=ta, trans_b=tb)
                    L.append(f"gemm {m} {n} {k} {ta} {tb} {ai} {bi} | " + out(r.T.ravel()))
for (m, n) in [(1, 1), (3, 2), (7, 9), (33, 5), (5, 33), (240, 31), (31, 240), (250, 250)]:
    for trans in (0, 1):
        for ai in range(3):
            for bi in range(3):
                seed = 0x8383 + m * 3 + n * 5 + trans + ai * 9 + bi * 27
                lx, ly = (n, m) if trans == 0 else (m, n)
                u = stream(seed, m * n + lx + ly)
                a = F(u[:m * n], m, n); x = u[m * n:m * n + lx]; y = u[m * n + lx:]
                r = B.dgemv(ALPHA[ai], a, x, BETA[bi], y, trans=trans)
                L.append(f"gemv {m} {n} {trans} {ai} {bi} | " + out(r))
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print(len(L), "cases")
