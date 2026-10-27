# Reference for tests/fixtures/openblas_haswell_dgesv.txt: run on the Linux reference
# (msi: numpy 2.3.5, OpenBLAS 0.3.30 Haswell): python -I tools/openblas_ref_dgesv.py out.txt
# np.linalg.solve(A, b) (b 1-D → dgesv, nrhs = 1). One line per case:
# "n kind seed <x hex…>" or "n kind seed singular". Inputs: the ddot xorshift stream;
# kind 0 = uniform, 1 = small integers (pivot ties), 2 = graded rows (1e-8 … 1).
import numpy as np, struct, sys
M = (1 << 64) - 1
def stream(seed, n):
    s = seed; out = []
    for _ in range(n):
        s ^= (s << 13) & M; s ^= s >> 7; s ^= (s << 17) & M
        out.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
    return np.array(out)
def hx(v): return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
def case(n, kind, seed):
    u = stream(0x7777 + n * 1009 + kind * 31 + seed, n * n + n)
    A, b = u[:n * n].reshape(n, n).copy(), u[n * n:].copy()
    if kind == 1:
        A = np.round(A * 3); b = np.round(b * 5)
    if kind == 2:
        A = A * np.logspace(-8, 0, n)[:, None]
    return A, b
lines = []
for n in range(1, 34):
    for kind in (0, 1, 2):
        for seed in range(4):
            A, b = case(n, kind, seed)
            try:
                x = np.linalg.solve(A, b)
                lines.append(f"{n} {kind} {seed} " + " ".join(hx(v) for v in x))
            except np.linalg.LinAlgError:
                lines.append(f"{n} {kind} {seed} singular")
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(len(lines), "cases,", sum(l.endswith("singular") for l in lines), "singular")
