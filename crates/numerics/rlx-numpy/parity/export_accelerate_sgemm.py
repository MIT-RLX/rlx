# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Reference for `rlx_numpy::accelerate::sgemm_nn`: Apple Accelerate's `sgemm_` itself (the
BLAS torch 2.10's macOS wheel links), called through ctypes — macOS only. The Rust side
is a pure-Rust model; this fixture is what it is tested against.

    /Users/Shared/fnirsatlas-ref/.venv/bin/python -I parity/export_accelerate_sgemm.py

Covers every measured regime (k = 9): full tiles, leftover rows / columns, small-call
corners on both sides of the size threshold (m·n 324 vs 342), m = 1 with even and odd
n, n = 1, beta 0 and 1 (into a zeroed C), and zero-heavy inputs for signed zeros.
"""
import ctypes, json, os, platform
import numpy as np

assert platform.system() == "Darwin" and platform.machine() == "arm64"
acc = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Accelerate.framework/Accelerate")
f = acc.sgemm_; P = ctypes.POINTER
f.argtypes = [ctypes.c_char_p, ctypes.c_char_p] + [P(ctypes.c_int)] * 3 + [P(ctypes.c_float), P(ctypes.c_float), P(ctypes.c_int), P(ctypes.c_float), P(ctypes.c_int), P(ctypes.c_float), P(ctypes.c_float), P(ctypes.c_int)]

def sgemm(A, B, beta):
    m, k = A.shape; n = B.shape[1]
    i = lambda v: ctypes.byref(ctypes.c_int(v)); fl = lambda v: ctypes.byref(ctypes.c_float(v))
    A = np.asfortranarray(A, dtype=np.float32); B = np.asfortranarray(B, dtype=np.float32)
    C = np.zeros((m, n), np.float32, order="F")
    f(b"N", b"N", i(m), i(n), i(k), fl(1.0), A.ctypes.data_as(P(ctypes.c_float)), i(m), B.ctypes.data_as(P(ctypes.c_float)), i(k), fl(beta), C.ctypes.data_as(P(ctypes.c_float)), i(m))
    return C

rng = np.random.default_rng(20261025)
shapes = [(1, n) for n in (1, 2, 3, 4, 7, 8, 33, 65, 256)] + [(m, 1) for m in (2, 5, 17)] + \
    [(m, n) for m in (2, 3, 5, 7, 13, 15) for n in (2, 3, 5, 6, 7, 9, 15, 17, 31)] + \
    [(17, 19), (18, 18), (19, 17), (18, 19), (19, 18), (21, 17), (22, 17), (23, 23), (47, 33), (60, 8), (76, 8), (75, 256), (59, 9)]
cases = []
for (m, n) in shapes:
    for beta in (0.0, 1.0):
        A = rng.standard_normal((m, 9)).astype(np.float32); B = rng.standard_normal((9, n)).astype(np.float32)
        if (m + n) % 3 == 0:  # zeros and negative zeros in A (padding columns of im2col)
            A[rng.random(A.shape) < 0.3] = 0.0
            A[rng.random(A.shape) < 0.1] = -0.0
        C = sgemm(A, B, beta)
        assert np.abs(C - A.astype(np.float64) @ B.astype(np.float64)).max() < 1e-3
        cases.append({"m": m, "n": n, "beta": beta, "a": A.T.ravel().view(np.uint32).tolist(), "b": B.T.ravel().view(np.uint32).tolist(), "c": C.T.ravel().view(np.uint32).tolist()})
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "accelerate_sgemm.json")
json.dump({"library": "Accelerate.framework sgemm_ (" + platform.mac_ver()[0] + ")", "k": 9, "cases": cases}, open(p, "w"))
print(len(cases), "calls ->", p, os.path.getsize(p), "bytes")
