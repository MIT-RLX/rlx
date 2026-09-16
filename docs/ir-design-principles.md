# rlx IR design principles

A written set of principles a proposed `Op` or IR change is checked against,
adapted from CAKE's Appendix B.1 (arXiv:2608.12629). Conventions already existed
in this tree — spread across crate docs, review habits and `just new-op` — but
they were not stated anywhere a proposal could be checked *against*, and an
unstated principle cannot be violated on the record.

Each principle below says what it means here, and — where one exists — names the
**mechanical gate** that enforces it. A principle with no gate is a review
convention; that is honest and fine, but the distinction matters, so it is
marked.

---

## P1 — Ergonomic

Building a graph should read like the math. Prefer builder methods
(`g.matmul`, `g.rope_n_styled`) over hand-assembled `add_node` calls, and avoid
destination-passing or grid bookkeeping in the IR surface.

*Gate:* none (review). The `rlx!` DSL and `rlx_ir::infer::GraphExt` exist to make
the ergonomic path the easy one.

## P2 — Performance-transparent

Decisions that determine performance must be visible in the IR, not buried in a
backend. If a backend's routing depends on a property, that property belongs on
the op or in a table someone can read.

*Gate:* partial — `rlx-gpu-dispatch` makes schedule selection a data lookup with
a per-route report (`dispatch::explain`), so "why is this shape on this schedule"
has an answer. Fusion decisions are reported by `rlx-compile`'s
`dispatch_report`.

## P3 — Canonical

One canonical spelling per operation. Two ops that mean the same thing will drift
apart: a rewrite will be taught about one and not the other, and only one will get
the bug fix.

*Gate:* none (review). rlx knowingly carries several near-synonyms (`MatMul` vs
`FusedMatMulBiasAct` vs `DotGeneral`; `Rope` vs `AxialRope2d`). They are justified
by distinct lowering needs, but nothing prevents a *new* redundant spelling.

## P4 — Statically type-checked

Ill-typed programs should be rejected at construction, not at run time. Builders
assert their preconditions (`rope_n_styled` rejects odd or oversized `n_rot`);
`rlx_ir::verify` and `verify_shapes` reject structurally or dimensionally
inconsistent graphs.

*Gate:* `rlx_ir::verify`, `verify_shapes`, and builder asserts.

## P5 — Analysis-friendly

An op must expose the information the analyses need. If a check cannot be written
because the fact it needs is implicit, the op is under-specified. The RoPE
cos/sin table is the cautionary case: its *stride* was implicit in the tensor's
trailing extent, so nothing could compare it against what a kernel assumed until
the extent became something the check could read.

*Gate:* `rlx_ir::repr_check` — and its `checked_kinds` field is deliberately
part of the report, so an op with no rule shows up as *uncovered* rather than
passing.

## P6 — Test-gated

An IR change is evaluated against the existing test corpus, not just its own new
test. Cross-backend behaviour is the property most likely to break, so the
device-sweeping gates are the ones that matter:

- `rlx-runtime/tests/fd_backward_gate.rs` — every case × every available backend,
  each against **its own** forward via central differences.
- `rlx-cuda/tests/launch_arity.rs` — launch argument counts vs kernel signatures.
- `rlx-gpu-dispatch` — defaults reproduce historical routing exactly.

*Gate:* the above, plus `just lint`. Note the known blind spots: `just lint`
skips feature-gated and arch-gated code, so a change touching CUDA/ROCm or
x86-only paths needs a cross-lint.

## P7 — Analysis-consistent

**Changes to the IR data model must be accompanied by the corresponding analysis
updates.** Adding a field to an op is not complete until every analysis that
should read it does.

This is the principle rlx has broken most expensively. `Op::Rope` gained a
`style` field; `vjp_rope` destructured `Op::Rope { head_dim, n_rot, .. }`; the
`..` swallowed it, and every GptJ (GGUF) rotation received a NeoX adjoint — wrong
gradients on every backend, while the forward stayed correct so nothing
complained.

*Gate:* `rlx-autodiff/tests/vjp_field_consistency.rs`. Every VJP that discards an
op field must carry a written reason. New elisions fail the build until someone
answers "could this field change the gradient?", and the `UNVERIFIED` count is
ratcheted at **zero** — a waiver without a derivation fails the suite.

Prose rots, so where the claim is testable it is tested:
`tests/scaled_quant_vjp_field_invariance.rs` pins the `ScaledQuantize` /
`ScaledQuantScale` pair by asserting the gradient is invariant across FP8/FP6,
OCP/FNUZ and per-tensor/block/NVFP4 — the exact property their waivers claim. Note
that finite differences would be the *wrong* instrument there: a straight-through
estimator is a deliberate substitute for the derivative, not an approximation of
it, so an FD check would fail by design. Match the instrument to the claim.

## P8 — Hardware-grounded

Document the intended hardware behaviour of each op — what it is expected to
lower to, and on which targets. An op whose intended lowering is undocumented
gets a different implementation per backend.

*Gate:* partial — `supported_ops` per backend records the op claim, and
`backend_partial_op_support` discipline requires a backend claiming an op to lower
it itself. Prose documentation is review-enforced.

---

## Using this list

For a new or changed op, the honest questions are:

1. **P3** — does an existing op already mean this?
2. **P5** — can the checks that should cover it actually read what they need?
3. **P7** — which analyses read this field, and have they all been updated?
   (`vjp_field_consistency` will catch the AD half; lowering is on you.)
4. **P6** — does the FD gate cover it on every backend that will run it?
5. **P8** — is the intended lowering written down per target?

`just new-op NAME` scaffolds the files. It does not answer these.

## A note on where this list came from

CAKE derived its IR **bottom-up** — collect a corpus of production kernels,
extract recurring patterns into candidate abstractions, check each against the
principles, then grow by porting more kernels (Appendix A). rlx adds ops
top-down: someone needs an op, `just new-op` scaffolds it. Neither is wrong, but
it explains the difference in what the two IRs *contain*: theirs covers warp
roles, barrier choreography and pipeline staging, because that is what the corpus
kept showing; rlx's covers mathematics, because that is what models kept needing.
## License

MIT OR Apache-2.0.
