# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Generate tests/vectors/{boost,scipy}.txt for rlx-special.

  python tools/gen_vectors.py <boost_oracle binary> <xsf_oracle binary>

Run in an environment with SciPy 1.17 (the reference). Every value is written as
its IEEE-754 bit pattern so nothing is lost to decimal formatting.

* boost.txt — every ported function, against `tools/boost_oracle.cpp` built with
  clang `-O2 -ffp-contract=on` (how SciPy builds Boost).
* cephes.txt — the cephes port (incl. internals: zeta, cephes' own log1p/expm1/
  log1pmx, lgam1p), against `tools/xsf_oracle.cpp` built the same way.
* scipy.txt — the SciPy entry points both ports cover, against SciPy itself:
  Boost (betainc, betaincc, stdtr, stats.t.sf) and cephes (erf, erfc, ndtr,
  ndtri, gammainc, gammaincc, gammaln, gamma, stats.chi2.sf).

The arguments are aimed at branches, not just sampled: small/large/integer/half-
integer shape parameters, unit parameters, x near 0, 1/2 and 1, the student-t
shapes (0.5, df/2), and the incomplete-gamma regions (x < 0.5, < 1.1, Temme,
x > 1000, integer and half-integer a).
"""
import pathlib, struct, subprocess, sys
import numpy as np
import scipy, scipy.special as sc, scipy.stats as st

oracle = sys.argv[1]
xsf_oracle = sys.argv[2]
out_dir = pathlib.Path(__file__).resolve().parent.parent / "tests" / "vectors"
h = lambda v: "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]
rng = np.random.default_rng(20261024)
U = lambda lo, hi: float(rng.uniform(lo, hi))
L = lambda lo, hi: float(10 ** rng.uniform(lo, hi))

rows = []  # (fn, args)
def add(fn, *a): rows.append((fn, tuple(float(v) for v in a)))

# --- ibeta / ibetac: every region of ibeta_imp
for _ in range(2500):
    add("ibeta", L(-3, 3), L(-3, 3), U(0, 1)); add("ibetac", L(-3, 3), L(-3, 3), U(0, 1))
for _ in range(800):  # both <= 1, x small/large
    a, b = U(0, 1), U(0, 1); x = L(-12, 0)
    add("ibeta", a, b, x); add("ibetac", a, b, 1 - x)
for _ in range(800):  # min <= 1 < max (series / a_step / small-b-large-a)
    a, b = U(0.01, 1), L(0, 3); x = U(0, 1)
    add("ibeta", a, b, x); add("ibetac", b, a, x); add("ibeta", b, a, L(-8, 0))
for _ in range(800):  # integer a, b < 40 (binomial_ccdf) and b < 40 non-integer
    a, b = float(rng.integers(2, 80)), float(rng.integers(2, 40)); x = U(0, 1)
    add("ibeta", a, b, x); add("ibetac", a, b, x)
    add("ibeta", a + U(0, 1), b + U(0, 1), x)
for _ in range(600):  # both large (fraction2)
    a, b = L(1.6, 4), L(1.6, 4); m = a / (a + b)
    add("ibeta", a, b, min(1, max(0, m + U(-0.05, 0.05)))); add("ibetac", a, b, U(0, 1))
for a in [0.5, 1.0, 2.0]:
    for b in [0.5, 1.0, 3.0]:
        for x in [0.0, 1e-300, 1e-10, 0.25, 0.5, 0.75, 1 - 1e-10, 1.0]:
            add("ibeta", a, b, x); add("ibetac", a, b, x)
for _ in range(1500):  # the student-t shapes
    df = float(rng.choice([1, 2, 3, 4, 5, 6, 9, 14, 19, 24, 29, 49, 99, 499, 4999])) if rng.uniform() < 0.7 else L(-1, 4)
    t = L(-3, 2.5); x2 = t * t
    add("ibetac", 0.5, df / 2, x2 / (df + x2)); add("ibeta", df / 2, 0.5, df / (df + x2))
    add("students_t_cdf", df, t); add("students_t_cdf", df, -t)

# --- beta, gamma family (Boost's own — what ibeta calls)
for _ in range(1500):
    add("beta", L(-3, 3), L(-3, 3))
for _ in range(3000):
    a, x = L(-3, 3.2), L(-4, 3.6)
    add("gamma_p", a, x); add("gamma_q", a, x)
for _ in range(600):  # integer / half-integer a, Temme band, large x
    a = float(rng.integers(1, 30)); x = U(0, 60)
    add("gamma_q", a, x); add("gamma_q", a + 0.5, x)
    a = L(1.31, 3.2); add("gamma_p", a, a * U(0.62, 1.38)); add("gamma_q", a, a * U(0.62, 1.38))
    a = L(-1, 3); add("gamma_q", a, L(3, 4.5))
for _ in range(2500):
    z = L(-4, 2.3); add("tgamma", z); add("lgamma", z); add("tgamma", -z + 1e-3 * U(0.1, 0.9))
    add("tgamma1pm1", U(-0.99, 3)); add("tgamma1pm1", L(-12, -0.5) * rng.choice([-1, 1]))
    add("tgamma_delta_ratio", L(-3, 3), U(-0.99, 1) * L(-3, 1.5))
for z in [1.0, 2.0, 3.0, 10.0, 50.0, 170.0, 171.0, 0.5, 1.5, 2.5]:
    add("tgamma", z); add("lgamma", z)
for _ in range(3000):
    z = U(-30, 30) if rng.uniform() < 0.5 else L(-12, 1.5) * rng.choice([-1, 1])
    add("erf", z); add("erfc", z)
for _ in range(1500):
    add("log1pmx", U(-0.99, 2)); add("log1pmx", L(-12, 0) * rng.choice([-1, 1]))
    add("powm1", L(-2, 2), U(-3, 3)); add("powm1", 1 + L(-10, -1) * rng.choice([-1, 1]), L(-2, 2))
for _ in range(1000):
    for fn in ["lanczos_sum", "lanczos_sum_exp_g_scaled"]:
        add(fn, L(-3, 3))
    add("lanczos_sum_near_1", U(-0.5, 0.5)); add("lanczos_sum_near_2", U(-0.5, 0.5))

inp = "".join(fn + " " + " ".join("0x" + h(v) for v in a) + "\n" for fn, a in rows)
res = subprocess.run([oracle], input=inp, capture_output=True, text=True, check=True).stdout
lines = res.strip().split("\n")
assert len(lines) == len(rows), (len(lines), len(rows))
hdr = "# rlx-special Boost oracle vectors: <fn> <arg bits...> <result bits>\n# @generated by tools/gen_vectors.py (boost_oracle, clang -O2 -ffp-contract=on)\n"
(out_dir / "boost.txt").write_text(hdr + "\n".join(lines) + "\n")

# --- cephes (xsf)
crows = []
def cadd(fn, *a): crows.append((fn, tuple(float(v) for v in a)))
for _ in range(4000):
    z = U(-40, 40) if rng.uniform() < 0.5 else L(-12, 1.5) * rng.choice([-1, 1])
    cadd("erf", z); cadd("erfc", z); cadd("ndtr", z * 1.4)
    p = L(-300, 0) if rng.uniform() < 0.3 else U(0, 1)
    cadd("ndtri", p); cadd("ndtri", 1 - p)
for _ in range(4000):
    a, x = L(-3, 3.3), L(-4, 3.6)
    cadd("igam", a, x); cadd("igamc", a, x); cadd("chdtrc", 2 * a, 2 * x)
for _ in range(1500):  # the asymptotic band and integer chi2 dfs (Friedman: k-1)
    a = L(1.31, 3.5); cadd("igam", a, a * U(0.7, 1.3)); cadd("igamc", a, a * U(0.7, 1.3))
    k = float(rng.integers(1, 30)); cadd("chdtrc", k, L(-2, 2.3))
for _ in range(3000):
    g = L(-4, 2.4) * (rng.choice([-1, 1]) if rng.uniform() < 0.3 else 1)
    cadd("lgam", g); cadd("Gamma", g); cadd("lgam", -L(1.5, 2.5) - 0.3)
    cadd("zeta", float(rng.integers(2, 42)), 1.0); cadd("zeta", U(1.01, 30), L(-2, 9))
    cadd("c_log1p", U(-0.5, 1)); cadd("c_expm1", U(-1, 1)); cadd("c_log1pmx", U(-0.9, 2)); cadd("lgam1p", U(-0.9, 3))
inp = "".join(fn + " " + " ".join("0x" + h(v) for v in a) + "\n" for fn, a in crows)
res = subprocess.run([xsf_oracle], input=inp, capture_output=True, text=True, check=True).stdout
clines = res.strip().split("\n")
assert len(clines) == len(crows)
hdr = "# rlx-special xsf/cephes oracle vectors: <fn> <arg bits...> <result bits>\n# @generated by tools/gen_vectors.py (xsf_oracle, clang -O2 -ffp-contract=on)\n"
(out_dir / "cephes.txt").write_text(hdr + "\n".join(clines) + "\n")

# --- SciPy entry points covered by the ports
srows = []
cmap = {"erf": sc.erf, "erfc": sc.erfc, "ndtr": sc.ndtr, "ndtri": sc.ndtri, "igam": sc.gammainc, "igamc": sc.gammaincc,
        "lgam": sc.gammaln, "Gamma": sc.gamma, "chdtrc": lambda k, x: st.chi2.sf(x, k)}
for fn, a in crows:
    if fn in cmap:
        srows.append(("scipy_" + fn, a, cmap[fn](*a)))
for fn, a in rows:
    if fn == "ibeta": srows.append(("betainc", a, sc.betainc(*a)))
    elif fn == "ibetac": srows.append(("betaincc", a, sc.betaincc(*a)))
    elif fn == "students_t_cdf":
        srows.append(("stdtr", a, sc.stdtr(*a)))
        srows.append(("t_sf", a, st.t.sf(a[1], a[0])))
hdr = f"# SciPy {scipy.__version__} reference vectors: <fn> <arg bits...> <result bits>\n# @generated by tools/gen_vectors.py\n"
(out_dir / "scipy.txt").write_text(hdr + "".join(fn + " " + " ".join(h(v) for v in a) + " " + h(r) + "\n" for fn, a, r in srows))
print("boost rows", len(lines), "cephes rows", len(clines), "scipy rows", len(srows))
