#!/usr/bin/env python3
"""Fixture for tests/resample.rs: SciPy's own resample_poly / upfirdn / kaiser / special.i0.

    python3 tools/gen_resample_vectors.py > tests/vectors/scipy_resample.txt

One record per line: `name|args|hex,hex,...` (IEEE-754 bits of float64 outputs).
"""
import numpy as np
import scipy
from scipy import signal, special


def hx(a):
    return ",".join(f"{np.float64(v).view(np.uint64):016x}" for v in np.asarray(a, np.float64).ravel())


print(f"# scipy {scipy.__version__} numpy {np.__version__}")
xs = np.concatenate([np.linspace(0, 8, 33), [8.0000001, 9.5, 12.25, 20.0, 31.7, 50.0, 0.001, 5.0]])
print(f"i0|{hx(xs)}|{hx(special.i0(xs))}")
for m, beta in [(41, 5.0), (201, 5.0), (641, 5.0), (12, 8.6)]:
    print(f"kaiser|{m},{beta!r}|{hx(signal.windows.kaiser(m, beta))}")
rng = np.random.default_rng(3)
for n, up, down in [(300, 25, 32), (300, 4, 5), (190, 25, 16), (500, 1, 5), (257, 1, 2), (64, 3, 1)]:
    x = rng.standard_normal(n) * 37.0
    print(f"resample_poly|{up},{down},{hx(x)}|{hx(signal.resample_poly(x, up, down))}")
for nh, n, up, down in [(7, 50, 3, 2), (12, 40, 1, 3), (5, 33, 4, 1)]:
    h = rng.standard_normal(nh)
    x = rng.standard_normal(n)
    print(f"upfirdn|{up},{down},{nh},{hx(h)},{hx(x)}|{hx(signal.upfirdn(h, x, up, down))}")
