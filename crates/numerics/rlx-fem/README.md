# rlx-fem

Nonlinear scalar finite elements on P1 (linear) triangles. The solver core has no
rlx dependencies and builds under `--no-default-features`.

## What it solves

One scalar unknown per node, over

```text
-div( kappa(|grad u|^2) * grad u  -  s )  =  f
```

with the weak form

```text
integral kappa * grad(u) . grad(w)  =  integral f * w  +  integral s . grad(w)
```

Three pieces, each supplied per element by the caller through [`Constitutive`]:

| Symbol | Name here | Magnetostatics | Heat conduction | Electrostatics |
|---|---|---|---|---|
| `u` | potential | vector potential `A_z` | temperature | electric potential |
| `kappa` | coefficient | reluctivity `nu(|B|^2)` | conductivity `k(|grad T|^2)` | permittivity |
| `f` | source | current density `J_z` | volumetric heating | charge density |
| `s` | flux source | remanence `nu * R(B_r)` | prescribed heat flux | polarisation |

The coefficient takes `|grad u|^2` rather than `|grad u|`, because the caller
always has the square already and the square root is measurable when it happens
per element per Newton step.

## What it provides

- **Geometry** — `Mesh` of nodes and triangles: area, centroid, and the shape
  gradient coefficients that are the whole of a constant-gradient triangle.
- **Layered meshing** — `layered::build` produces a conforming structured mesh
  over a stack of bands, each band a row of tagged segments. Every material
  boundary becomes a grid line, so no element straddles two materials.
- **Tied degrees of freedom** — `DofMap` eliminates fixed nodes and ties node
  pairs with a signed factor. Periodic and anti-periodic boundaries are the
  factor `+1` and `-1` cases. Elimination rather than penalty or Lagrange keeps
  the reduced system symmetric positive definite.
- **Damped Newton** — the tangent is the stiffness plus a rank-one update per
  element, symmetric and positive semi-definite wherever the coefficient rises
  with `|grad u|`, so the same conjugate gradient solves it.
- **Sparse solve** — CSR with a Jacobi-preconditioned conjugate gradient.

## Verification

`tests/analytical.rs` checks against problems whose exact solution lies inside
the P1 space, so the discretisation error is zero and the tolerances are 1e-10
rather than a chosen percentage — a tolerance that loose does not detect a sign
error in the flux-source term. Second-order convergence is checked separately on
a case that is *not* exactly representable.

```sh
cargo test -p rlx-fem --no-default-features
```
## License

MIT OR Apache-2.0.
