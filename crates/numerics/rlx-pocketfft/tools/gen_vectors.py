# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Reference vectors for rlx-pocketfft: scipy.fft (SciPy 1.17.1) outputs as hex
IEEE-754 bits (outputs over 256 doubles as an FNV-1a hash of those bits). Inputs come from an LCG both sides reproduce exactly
(k-th value = ((k * 2654435761 + seed) mod 2^32) / 2^32 - 0.5, exact in f64).

    /Users/Shared/fnirsatlas-ref/.venv/bin/python -I tools/gen_vectors.py
"""
import pathlib, struct
import numpy as np, scipy, scipy.fft as F
assert scipy.__version__ == "1.17.1", scipy.__version__

def lcg(n, seed):
    k = np.arange(n, dtype=np.uint64)
    return ((k * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(2**32)).astype(np.float64) / 2**32 - 0.5
def fnv(a):
    """FNV-1a (64-bit) over the little-endian bytes of the f64 bit patterns."""
    h = 0xcbf29ce484222325
    for b in np.ascontiguousarray(np.asarray(a).ravel().view(np.float64)).tobytes():
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h
def hx(a):
    a = np.asarray(a).ravel().view(np.float64)
    if a.size > 256:  # long outputs travel as a hash; regenerate locally to diagnose
        return "fnv %016x %d" % (fnv(a), a.size)
    return " ".join("%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in np.asarray(a).ravel().view(np.float64))

sizes = sorted(set(list(range(1, 65)) + [2**k for k in range(7, 15)] + [96, 100, 120, 125, 128 * 3, 243, 625, 1000, 1080, 1536, 2187, 3125, 4320, 6000, 10000]
                   + [97, 101, 127, 251, 509, 1009, 2003, 4099, 7919] + [2 * 1009, 3 * 509, 49, 77, 121, 143, 169, 289, 361, 529, 841, 961]))
lines = []
for n in sizes:
    x = lcg(n, n)
    for norm in [None, "ortho", "forward"]:
        nn = norm or "backward"
        X = F.rfft(x, norm=norm)
        lines.append(f"rfft {n} {n} {nn} {hx(X)}")
        y = F.irfft(X, n=n, norm=norm)
        lines.append(f"irfft {n} {n} {nn} {hx(y)}")  # input: this case's rfft output
        z = lcg(2 * n, 7 * n).view(np.complex128)
        lines.append(f"fft {n} {n} {nn} {hx(F.fft(z, norm=norm))}")
        lines.append(f"ifft {n} {n} {nn} {hx(F.ifft(z, norm=norm))}")
# n differing from the input length: pad and truncate
for n, m in [(100, 128), (128, 100), (37, 64), (1000, 1024), (513, 300)]:
    x = lcg(n, n)
    X = F.rfft(x, n=m)
    lines.append(f"rfft {n} {m} backward {hx(X)}")
    lines.append(f"irfft_n {n} {m} backward {hx(F.irfft(F.rfft(x), n=m))}")
lines.append("good_real " + " ".join(f"{n}:{F.next_fast_len(n, real=True)}" for n in list(range(1, 300)) + [1009, 4099, 10007, 65537]))
lines.append("good_cmplx " + " ".join(f"{n}:{F.next_fast_len(n, real=False)}" for n in list(range(1, 300)) + [1009, 4099, 10007, 65537]))
out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "vectors" / "scipy_fft.txt"
out.write_text("\n".join(lines) + "\n")
print("wrote", out, out.stat().st_size, "bytes,", len(lines), "lines")
