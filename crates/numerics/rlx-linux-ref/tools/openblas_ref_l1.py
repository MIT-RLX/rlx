# Reference for tests/fixtures/openblas_haswell_l1.txt: run on the Linux reference
# (msi: SciPy 1.17.1 on OpenBLAS 0.3.30 Haswell): python -I tools/openblas_ref_l1.py out.txt
# scipy.linalg.blas dnrm2 / drot on xorshift inputs scaled by 2^k. One line per case:
# "nrm2 n inc k <hex>" or "rot n k <x' hex…> | <y' hex…>" (over 64 values: fnv of the bits).
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
L = []
for n in list(range(0, 40)) + [63, 64, 65, 100, 257, 1000, 4097]:
    for inc in (1, 3):
        for k in (0, 300, -300):
            x = stream(0x1717 + n * 13 + inc + k, max(n * inc, 1)) * 2.0 ** k
            r = B.dnrm2(x, n=n, incx=inc) if n else 0.0
            L.append(f"nrm2 {n} {inc} {k} {hx(r)}")
for n in list(range(1, 40)) + [64, 100]:
    x = stream(0x2323 + n, n); y = stream(0x4545 + n, n)
    c, s = 0.6, 0.8 + 1e-9
    xr, yr = B.drot(x.copy(), y.copy(), c, s)
    L.append(f"rot {n} " + " ".join(hx(v) for v in xr) + " | " + " ".join(hx(v) for v in yr))
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print(len(L), "cases")
