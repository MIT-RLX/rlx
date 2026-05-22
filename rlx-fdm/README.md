# rlx-fdm

Force density method (FDM) for pin-jointed structures in RLX, with logic ported from [jax_fdm](https://github.com/arpastrana/jax_fdm).

## Borrowed from jax_fdm

| jax_fdm module | rlx-fdm |
|----------------|---------|
| `datastructures` / `FDNetwork` | [`network::Network`](src/network.rs) |
| `equilibrium/structures` (connectivity, free/fixed split) | [`structure::Structure`](src/structure.rs) |
| `equilibrium/models` (`stiffness_matrix`, `load_matrix`, `nodes_equilibrium`) | [`equilibrium`](src/equilibrium.rs) |
| `equilibrium/fdm` (`fdm`, validation) | [`reference::fdm`](src/reference.rs) |
| `equilibrium/states` | [`state::EquilibriumState`](src/state.rs) |
| `goals/edge/length`, `goals/network/loadpath` | [`goals`](src/goals.rs), [`objective::Goal`](src/objective.rs) |
| `constraints` + `constrained_fdm` | [`constraints`](src/constraints.rs), [`optimize::constrained_fdm`](src/optimize.rs) |
| `data/json/*.json` | [`io::json`](src/io/json.rs) (feature `io`) |
| `equilibrium/sparse` + custom VJP | Planned: `rlx-sparse` + symmetric solve (K is SPD for physical q) |
| `constrained_fdm` + optimizers | [`constrained_fdm`](src/optimize.rs) (GD or L-BFGS on `q`, optional L2 on `q`) |
| Visualization (COMPAS / compas_view2) | Out of scope |

## Quick start

```rust
use rlx_fdm::{Network, fdm};

let network = Network::arch_chain(5.0, 10, -1.0, -0.2);
let eq = fdm(&network).expect("equilibrium");
// Free-node z should sag negative under vertical load.
assert!(eq.xyz[[1, 2]] < 0.0);
```

## Features

| Feature | Description |
|---------|-------------|
| `reference` (default) | Dense CPU FDM solve |
| `io` | Load jax_fdm-style JSON networks |
| `ir` | Emit `Graph` with `Op::DenseSolve` for RLX autodiff |

## Sparse + nonlinear

```rust
use rlx_fdm::{Network, fdm_with_options, FdmOptions};

let mut net = Network::arch_chain(5.0, 10, -1.0, 0.0);
net.edges_load_uniform([0.0, 0.0, -0.05]);

let opts = FdmOptions::nonlinear(50, 1e-6, true);
let eq = fdm_with_options(&net, &opts)?;
```

- **`sparse`:** `SparseStiffness` CSR + PCG (`sparse.rs`, `solve.rs`)
- **`nonlinear`:** fixed-point `x ← K⁻¹ P(x)` with edge loads × length (`iterative.rs`, `loads.rs`)

## Deferred features (implemented)

| jax_fdm | rlx-fdm |
|---------|---------|
| `EquilibriumStructureSparse` / `index_array` | [`SparseStiffnessFast`](src/sparse_fast.rs) — O(nnz) assembly |
| Face tributary loads | [`loads`](src/loads.rs) + [`mesh::MeshStructure`](src/mesh.rs) |
| `solver_fixedpoint_implicit` | [`grad_loss_wrt_q_fixedpoint`](src/implicit.rs) — unrolled adjoint (edge + face loads, local LCS) |
| `LBFGSB` | [`lbfgs::Lbfgs`](src/lbfgs.rs) + [`OptimizerKind::Lbfgs`](src/optimize.rs) |
| Mesh goals | [`MeshArea`](src/objective.rs), [`MeshPlanarity`](src/objective.rs), [`MeshFaceRectangular`](src/objective.rs), [`MeshLaplacian`](src/objective.rs) |
| PCG adjoint | [`solve_adjoint_columns`](src/implicit.rs) + [`grad_loss_wrt_q_linear_with_solver`](src/implicit.rs) |
| Support design params | [`grad_loss_wrt_xyz_fixed_linear`](src/implicit.rs) (`dL/dX_fixed`) |
| `rlx_sparse` PCG | [`rlx_op`](src/rlx_op.rs) (`feature rlx-sparse`) |

## Inverse design

```bash
cargo run -p rlx-fdm --example inverse_design
cargo run -p rlx-fdm --example constrained_arch
```

- **`inverse_design`:** manual GD on `q` for one edge-length target.
- **`constrained_arch`:** [`constrained_fdm`](src/optimize.rs) with `NetworkLoadpath` + `EdgeLength` goals and `EdgeQ` / length caps.

```rust
use rlx_fdm::{constrained_fdm, Constraint, Goal, Network, OptimizeConfig};

let net = Network::arch_chain(5.0, 10, -1.0, -0.2);
let goals = vec![Goal::network_loadpath(12.0, 1.0)];
let constraints = vec![Constraint::all_edge_q(-50.0, -0.5, 0.01)];
let cfg = OptimizeConfig::arch_form_finding(); // L-BFGS + linear FDM + sparse PCG
let res = constrained_fdm(&net, &goals, &constraints, &cfg)?;
```

End-to-end loop (one iteration): equilibrium solve → goal + constraint grads → implicit `dL/dq` using
[`EquilibriumModel::pack_xyz_free`](src/equilibrium.rs) (no extra fixed-point re-solve). Topology is cached in
[`OptWorkspace`](src/optimize.rs); L-BFGS line search trials only vary `q`.

**Fused MIR (`feature ir`, `fuse_mir`):** [`FusedEquilibriumRunner`](src/fuse.rs) compiles `q → x_f` once per
topology and reuses it inside [`constrained_fdm`](src/optimize.rs) (enabled by default in
[`OptimizeConfig::arch_form_finding`](src/optimize.rs)). With `feature fuse`,
[`FusedAutodiffFormFinding`](src/fuse.rs) adds `loss + dL/dq` through sparse PCG in one compiled graph.

Chain rule: `goals_grad_xyz_free` + constraint grads → `grad_loss_wrt_q` (implicit adjoint).

### Design parameters, losses, optimizers (jax_fdm 1–5)

- [`DesignParam`](src/parameters.rs) / [`DesignVector`](src/parameters.rs): all-edge `q`, single `EdgeQ`, support XYZ, free-node loads.
- [`Loss`](src/losses.rs) + [`ErrorKind`](src/losses.rs): squared / RMS / absolute / prediction / log-max aggregation (pass via [`OptimizeConfig::losses`](src/optimize.rs) or default from `goals`).
- [`OptimizerKind::Slsqp`](src/optimize.rs) / [`Ipopt`](src/optimize.rs): penalty + L-BFGS on box `x` with FD on nonlinear inequalities ([`Constraint::edge_angle`](src/constraints.rs), node tangent / normal).
- [`OptimizeConfig::force_dense_with_constraints`](src/optimize.rs): dense FDM when constraints are active (jax_fdm policy).

## Goals (jax_fdm subset)

| Goal | Description |
|------|-------------|
| `NetworkLoadpath` | Total load path energy |
| `EdgeLength` / `MeanEdgeLength` | Edge length targets |
| `EdgeForce` / `MeanEdgeForce` | Axial force targets |
| `NodeCoord` / `MinFreeZ` | Node position / sag |
| `Residual` | Equilibrium misfit penalty |
| `MeshArea` / `MeshPlanarity` | Total face area / mean planarity (needs mesh on `Network`) |
| `MeshFaceRectangular` / `MeshLaplacian` | Quad corner orthogonality / edge spring smoothing |

## Mesh I/O (`feature io`)

```rust
use rlx_fdm::io::{from_json_path, mesh_from_json_str, merge_mesh, to_json_str, MeshDocument};

let mut net = from_json_path("data/quad_mesh.json")?;
let mesh = mesh_from_json_str(r#"{"faces":[[0,1,2]],"faces_load":[[0,0,-1]]}"#)?;
merge_mesh(&mut net, &mesh);
let json = to_json_str(&net)?;
```

## MIR optimizer hook (`feature ir` + `rlx-sparse`)

With `features = ["ir", "rlx-sparse"]`, [`FdmMirOptimizer::build_equilibrium_graph`](src/mir_opt.rs) wires `fdm_q → fdm.assemble_csr_values → rlx_sparse.pcg_solve` (×3) with param `fdm_P`. Autodiff through the graph matches the host PCG adjoint in [`solve_adjoint_columns`](src/implicit.rs).

## MIR optimizer hook (`feature ir`)

```rust
use rlx_fdm::mir_opt::{FdmMirOptimizer, FdmGradMode, goals_grad_wrt_q};
use rlx_fdm::graph::{fdm_dense_graph, build_grad_q_custom_fn, FdmGradQSignature};

let opt = FdmMirOptimizer { grad_mode: FdmGradMode::FixedPoint(cfg), ..Default::default() };
let gq = goals_grad_wrt_q(&opt, &network, &goals, None)?;
```

`build_grad_q_custom_fn` emits `fdm_dq` param; fill with [`set_grad_q_param`](src/mir_opt.rs) before `Session::run` for host `dL/dq` in fused graphs.

## RLX integration path

1. **Today:** nonlinear fixed-point adjoint (edge loads), full goal library, mesh JSON, `FdmMirOptimizer` + `constrained_fdm`.
2. **Next:** IPOPT interface (external); tighter PCG-VJP vs host linear adjoint.

**Anderson acceleration:** set [`IterativeConfig::anderson_depth`](src/iterative.rs) > 0 on nonlinear solves. Implicit adjoint unrolls a full `tmax` trajectory (`eta = 0`) so it matches central-difference `dL/dq`.
