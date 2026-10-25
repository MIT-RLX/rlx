# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Reference vectors for rlx-signal from SciPy 1.17.1 / NumPy 2.3.5 (macOS arm64).

Format, one case per line:  op ; arg=hex,hex,… ; … ; out=hex,…  (floats as IEEE-754
hex bits, ints as decimal; arrays over 64 values as `fnv:<FNV-1a of the bits>:<len>`). Signals come from an LCG both sides reproduce exactly.

    /Users/Shared/fnirsatlas-ref/.venv/bin/python -I tools/gen_vectors.py
"""
import pathlib, struct
import numpy as np, scipy
from scipy import signal as S
assert scipy.__version__ == "1.17.1" and np.__version__ == "2.3.5", (scipy.__version__, np.__version__)

def h(v): return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
def fnv(a):
    hh = 0xcbf29ce484222325
    for b in np.ascontiguousarray(a).tobytes():
        hh = ((hh ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return hh
def hs(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size > 64:  # long outputs as an FNV-1a hash of their bits (+ length)
        return "fnv:%016x:%d" % (fnv(a), a.size)
    return ",".join(h(v) for v in a)
def cs(a): a = np.asarray(a, dtype=np.complex128).ravel(); return ",".join(h(v.real) + ":" + h(v.imag) for v in a)
def lcg(n, seed):
    k = np.arange(n, dtype=np.uint64)
    return ((k * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(2**32)).astype(np.float64) / 2**32 - 0.5
def sig(n, seed):
    """A drifting signal with structure (cumulated LCG noise + a sinusoid)."""
    t = np.arange(n)
    return np.cumsum(lcg(n, seed)) * 0.1 + np.sin(0.05 * t) + lcg(n, seed + 1)

L = []
for n in range(1, 11):
    z, p, k = S.buttap(n)
    L.append(f"buttap;n={n};p={cs(p)};k={h(k)}")
rng = np.random.default_rng(3)
designs = []
for sf in [5.0, 7.8125, 10.0, 10.172526041666666, 12.5, 25.0, 50.0, 100.0, 250.0]:
    designs.append((3, [0.01 / (sf / 2), 0.1 / (sf / 2)], "bandpass"))
    designs.append((3, [(0.5 * 2 / sf)], "lowpass")) if 0.5 * 2 / sf < 1 else None
    designs.append((3, [0.7 / (sf / 2), 1.5 / (sf / 2)], "bandpass")) if 1.5 / (sf / 2) < 1 else None
for _ in range(40):
    n = int(rng.integers(1, 9)); a, b = sorted(rng.uniform(0.001, 0.999, 2))
    designs.append((n, [a, b], "bandpass"))
    designs.append((n, [a], "lowpass")); designs.append((n, [b], "highpass"))
for n, wn, bt in designs:
    w = wn if bt == "bandpass" else wn[0]
    z, p, k = S.butter(n, w, btype=bt, output="zpk")
    b, a = S.butter(n, w, btype=bt, output="ba")
    sos = S.butter(n, w, btype=bt, output="sos")
    L.append(f"butter;n={n};wn={hs(wn)};btype={bt};z={cs(z)};p={cs(p)};k={h(k)};b={hs(b)};a={hs(a)};sos={hs(sos)}")
    x = sig(400, n * 31 + len(L))
    zi = S.lfilter_zi(b, a); L.append(f"lfilter_zi;b={hs(b)};a={hs(a)};zi={hs(zi)}")
    y, zf = S.lfilter(b, a, x, zi=zi * x[0]); L.append(f"lfilter;b={hs(b)};a={hs(a)};seed={n * 31 + len(L) - 1};zi={hs(zi * x[0])};y={hs(y)};zf={hs(zf)}")
    szi = S.sosfilt_zi(sos); L.append(f"sosfilt_zi;sos={hs(sos)};zi={hs(szi)}")
    seed = n * 31 + len(L); x = sig(400, seed)
    y, zf = S.sosfilt(sos, x, zi=szi * x[0]); L.append(f"sosfilt;sos={hs(sos)};seed={seed};zi={hs(szi * x[0])};y={hs(y)};zf={hs(zf)}")
    for pt, pl in [("odd", None), ("odd", 0), ("even", 20), ("constant", 7), (None, None)]:
        try:
            yf = S.filtfilt(b, a, x, padtype=pt, padlen=pl); L.append(f"filtfilt;b={hs(b)};a={hs(a)};seed={seed};padtype={pt};padlen={pl};y={hs(yf)}")
        except ValueError as e:
            L.append(f"filtfilt;b={hs(b)};a={hs(a)};seed={seed};padtype={pt};padlen={pl};err=1")
        yf = S.sosfiltfilt(sos, x, padtype=pt, padlen=pl); L.append(f"sosfiltfilt;sos={hs(sos)};seed={seed};padtype={pt};padlen={pl};y={hs(yf)}")
for win in ["hann", "hamming", "blackman", "boxcar"]:
    for m in [1, 2, 3, 4, 5, 10, 63, 64, 251, 1001, 4097]:
        L.append(f"window;name={win};m={m};w={hs(S.get_window(win, m, fftbins=False))}")
for win in ["hann", "hamming", "blackman"]:
    for numtaps in [3, 11, 51, 257, 1651, 8251]:
        for c in [0.01, 0.05, 0.123456789, 0.5, 0.9]:
            L.append(f"firwin;numtaps={numtaps};cutoff={hs([c])};window={win};pass_zero=1;fs=none;h={hs(S.firwin(numtaps, c, window=win))}")
        # MNE's call: cutoff in units of fs = freq[-1]*2 = 2
        c = (0.0016 + 0.0024) / 2.0
        L.append(f"firwin;numtaps={numtaps};cutoff={hs([c])};window={win};pass_zero=1;fs={h(2.0)};h={hs(S.firwin(numtaps, c, window=win, pass_zero=True, fs=2.0))}")
    L.append(f"firwin;numtaps=101;cutoff={hs([0.2, 0.4])};window={win};pass_zero=0;fs=none;h={hs(S.firwin(101, [0.2, 0.4], window=win, pass_zero=False))}")
# macOS arm64 → tests/vectors/; the Linux x86-64 reference (msi) → tests/vectors/linux-x86_64/
import sys
sub = () if sys.platform == "darwin" else ("linux-x86_64",)
out = pathlib.Path(__file__).resolve().parent.parent.joinpath("tests", "vectors", *sub, "scipy_signal.txt")
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(L) + "\n")
print("wrote", out, out.stat().st_size, "bytes,", len(L), "cases")
