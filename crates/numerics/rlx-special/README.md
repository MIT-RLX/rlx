# rlx-special

SciPy's special functions for `f64`, reproduced **bit-for-bit** — the Boost.Math
ones (`betainc`, `stdtr`) and the cephes ones (`gammainc`, `ndtri`, `chi2.sf`, …) —
so a Rust port of a SciPy statistic gets SciPy's p-values exactly, not to a tolerance.

| function | Boost | SciPy entry point it matches |
|---|---|---|
| `ibeta`, `ibetac` | `ibeta`, `ibetac` | `special.betainc`, `special.betaincc` |
| `students_t_cdf`, `students_t_sf` | `cdf(students_t)` | `special.stdtr`, `stats.t.cdf`, `stats.t.sf` |
| `beta`, `gamma_p`, `gamma_q`, `tgamma`, `lgamma`, `tgamma1pm1`, `tgamma_delta_ratio`, `erf`, `erfc`, `log1pmx`, `powm1` | same names | — (Boost's own; what `ibeta` calls) |
| `cephes::{erf, erfc, ndtr, ndtri}` | xsf cephes | `special.erf`, `erfc`, `ndtr`, `ndtri` (`stats.norm.cdf`/`ppf`) |
| `cephes::{igam, igamc, chdtrc}` | xsf cephes | `special.gammainc`, `gammaincc`, `stats.chi2.sf` |
| `cephes::{lgam, gamma, zeta, lgam1p}` | xsf cephes | `special.gammaln`, `gamma`, `zeta(x, q)` |
| `normal_cdf` | `cdf(normal_distribution(0, 1))` | — (`erfc(-x/√2)/2`) |
| `quadpack::{dqagie, studentized_range_cdf_inf, studentized_range_sf_inf}` | SciPy's QUADPACK C translation | `integrate.quad` on infinite intervals, `stats.studentized_range` (`df = inf`) |

## Which SciPy functions are Boost — measured

SciPy 1.17.1 vs a clang `-O2 -ffp-contract=on` build of the Boost headers,
3000 random arguments each, compared as bit patterns:

| SciPy | mismatches vs Boost |
|---|---|
| `betainc`, `betaincc`, `stdtr` | **0** |
| `gammainc`, `gammaincc`, `stats.chi2.sf`, `erf`, `erfc`, `gammaln`, `gamma` | 10–84 % — these are SciPy's cephes-derived `xsf`, not Boost |

Those are ported separately, from `scipy/xsf`'s `include/xsf/cephes/*.h`, as
[`cephes`]: the same headers built with clang `-ffp-contract=on` reproduce SciPy
on 36 000 of 36 000 values (2–31 % differ without contraction). Use the Boost
`gamma_*`/`erf*` only where *Boost* is the reference, and `cephes::*` where SciPy is.

## What it takes to be exact

* **Same algorithms and thresholds**: `ibeta_imp`'s region dispatch,
  `gamma_incomplete_imp_final`'s eight methods, Lanczos-13m53, the 53-bit `erf`
  and 64-bit-tag `lgamma_small` rationals, the Temme expansion.
* **Same evaluation order**: Boost's second-order Horner scheme
  (`BOOST_MATH_POLY_METHOD 3`) for every fixed-size polynomial/rational.
* **Same FMAs**: clang's default `-ffp-contract=on` fuses `a*b + c` within one
  C++ expression. A non-contracted build of the same headers differs from SciPy
  on 212 of 9300 `betainc` values; the contracted build on 0. Every fused site is
  an `f64::mul_add` here.
* **libm where Boost uses libm**: Boost's `double` `log1p`/`expm1` forward to the C
  library, as `f64::ln_1p` / `f64::exp_m1` do.

## Tests

`tests/vectors.rs` — zero-tolerance comparison against
`tests/vectors/boost.txt` (60 264 values, 22 functions, from
`tools/boost_oracle.cpp`), `tests/vectors/cephes.txt` (63 500 values, 14
functions incl. cephes' own `log1p`/`expm1`/`log1pmx`/`zeta`, from
`tools/xsf_oracle.cpp`) and `tests/vectors/scipy.txt` (67 244 SciPy values).
Regenerate with

```sh
clang++ -std=c++17 -O2 -ffp-contract=on -I<boost> tools/boost_oracle.cpp -o boost_oracle
clang++ -std=c++17 -O2 -ffp-contract=on -I<xsf>/include tools/xsf_oracle.cpp -o xsf_oracle
python tools/gen_vectors.py ./boost_oracle ./xsf_oracle   # in a SciPy 1.17 environment
python3 tools/gen_cephes_tables.py <xsf>/include/xsf/cephes
```

Mutation-checked: un-fusing the Horner first step, the `ibeta_power_terms`
exponent, the small-b series sum, the `lgamma_small` tail, cephes' `polevl`, or
the `igamc` continued fraction each fails the suite. (Un-fusing `0.5 + 0.5*erf`
in `ndtr` or `s -= 0.5*b` in `zeta` survives — correctly: a product by 0.5 is
exact, so those mutants are equivalent.)

Only the regularized (`normalised = true`) incomplete functions are ported.
The non-regularized branches differ only by a `tgamma(a)` / `beta(a,b)` factor and
are not exposed by SciPy.

Ported from Boost.Math (Boost Software License 1.0) and SciPy's xsf / Cephes (BSD-3-Clause).
## License

MIT OR Apache-2.0.
