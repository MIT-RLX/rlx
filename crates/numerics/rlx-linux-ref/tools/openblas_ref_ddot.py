# Reference for tests/fixtures/openblas_haswell_ddot.txt: run on the Linux reference
# (msi: numpy 2.3.5, OpenBLAS 0.3.30 Haswell, 20 threads): python -I tools/openblas_ref_ddot.py out.txt
# np.dot on 1-D float64 vectors (OpenBLAS 0.3.30 Haswell ddot) for many lengths/strides.
# Inputs: xorshift64 stream u = (s >> 11) / 2^53 * 2 - 1, reproduced exactly in Rust.
import numpy as np, scipy.linalg.blas as B, struct, sys
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
    return np.array(out)
def hx(v): return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
lens = list(range(0, 260)) + [300, 511, 512, 513, 1000, 1023, 1024, 1025, 4096, 9999, 10000, 10001, 20000, 65537]
lines = []; agree = 0; tot = 0
for n in lens:
    for inc in (1, 2, 3):
        x = stream(0x1234 + n * 7 + inc, n * inc); y = stream(0x9876 + n * 11 + inc, n * inc)
        xs, ys = x[::inc], y[::inc]
        d = np.dot(xs, ys)
        s = B.ddot(xs, ys) if n else 0.0
        tot += 1; agree += hx(d) == hx(s)
        lines.append(f"{n} {inc} {hx(d)}")
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("numpy == scipy ddot:", agree, "/", tot)
