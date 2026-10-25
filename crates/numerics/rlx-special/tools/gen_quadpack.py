# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Generate tests/vectors/quadpack.txt: scipy.stats.studentized_range cdf/sf at
df=inf (QUADPACK dqagie over the Cython integrand) and the C library erfc
(Python's math.erfc) the integrand calls. Run with SciPy 1.17.
Lines: "<fn> <arg bits...> <result bits>"."""
import math, pathlib, struct, numpy as np, scipy, scipy.stats as st
h = lambda v: "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
rng = np.random.default_rng(77)
rows = []
for _ in range(400):
    k = float(rng.integers(2, 16)); q = float(rng.uniform(0, 8)) if rng.uniform() < .8 else float(10 ** rng.uniform(-3, 1.2))
    rows.append(("sr_cdf", (q, k), st.studentized_range.cdf(q, k, np.inf)))
    rows.append(("sr_sf", (q, k), st.studentized_range.sf(q, k, np.inf)))
for _ in range(20000):
    z = float(rng.normal(0, 4)) if rng.uniform() < .7 else float(rng.uniform(-30, 30))
    rows.append(("c_erfc", (z,), math.erfc(z)))
out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "vectors" / "quadpack.txt"
out.write_text(f"# SciPy {scipy.__version__} studentized_range (df=inf) + C erfc; tools/gen_quadpack.py\n" + "".join(f + " " + " ".join(h(v) for v in a) + " " + h(r) + "\n" for f, a, r in rows))
print("wrote", out, len(rows))
