# Architecture decision records

One file per decision that shapes how rlx is built and that a reader could not
recover from the code alone: why something is *not* in core, why a default is
off, why a seam exists. Small, numbered, never renumbered.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-empty-prelude.md) | The `rlx` prelude has no default features | Accepted |
| [0002](0002-optimizations-are-opt-in.md) | Optimizations ship behind opt-in features, default off | Accepted |
| [0003](0003-core-is-jax-shaped.md) | Core ships JAX-shaped primitives; domain composites live downstream | Accepted |
| [0004](0004-models-live-out-of-tree.md) | Model crates live in `rlx-models` and pin published rlx | Accepted |
| [0005](0005-known-failures-as-data.md) | Expected test failures are data files linked to issues | Proposed |

## Writing one

Copy the template, take the next number, add a row above. An ADR records a
decision; when the decision changes, write a new ADR that supersedes the old
one and set the old one's status to `Superseded by NNNN` — do not rewrite it.

```markdown
# ADR NNNN — <decision, as a sentence>

## Status

Proposed | Accepted | Superseded by NNNN — <date>

## Context

What forces the decision: the constraint, the incident, the measurement.

## Decision

What we do, specifically enough to check a change against it.

## Alternatives

What else was considered and why it lost.

## Consequences

What gets easier, what gets harder, and what has to be maintained.
```
## License

MIT OR Apache-2.0.
