# Reference for tests/fixtures/openblas_haswell_dsyrk.txt: run on the Linux reference
# (msi: numpy 2.3.5, OpenBLAS 0.3.30 Haswell, 20 threads): python -I tools/openblas_ref_dsyrk.py out.txt
# NumPy routes `A @ A.T` / `A.T @ A` (same buffer) to cblas_dsyrk(RowMajor, Upper, NoTrans/Trans).
# One line per case: "n k trans <upper-triangle hex, row-major i<=j>". Inputs: the ddot xorshift stream.
import numpy as np, struct, sys
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
    return np.array(out)
def hx(v): return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
shapes = [(n, k) for n in (1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 17, 29) for k in (1, 2, 3, 7, 8, 16, 17, 255, 256, 257)]
shapes += [(2, k) for k in (511, 512, 513, 600, 767, 768, 769, 1000, 1023, 2500, 3000, 4097)]
shapes += [(9, 513), (13, 1000), (40, 300), (64, 120), (100, 50), (120, 40)]  # last ones above the thread threshold
lines = []
for n, k in shapes:
    for trans in (0, 1):
        a = stream(0x5151 + n * 131 + k * 7 + trans, n * k)
        if trans == 0:
            A = a.reshape(n, k); C = A @ A.T     # NoTrans: op(A) = A (n×k)
        else:
            A = a.reshape(k, n); C = A.T @ A     # Trans:   op(A) = Aᵀ, A is k×n
        up = [hx(C[i, j]) for i in range(n) for j in range(i, n)]
        lines.append(f"{n} {k} {trans} " + " ".join(up))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases")
