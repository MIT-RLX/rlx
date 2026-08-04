<!--
Thanks for contributing to RLX. Read CONTRIBUTING.md first.
Keep the change focused — unrelated refactors belong in a separate PR.
-->

## Summary

<!-- What does this change do, and why? Which plan item / issue does it address? -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New `Op` / op coverage
- [ ] New or extended backend path
- [ ] Performance (hot-path / kernel)
- [ ] Docs / tooling / CI
- [ ] Other:

## Backends & hardware touched

<!-- Which backends does this affect, and what hardware did you actually run on?
     e.g. Metal on M4 Pro, CUDA on RTX 3080 Ti (msi), CPU only.
     Delete this whole section for changes that touch no backend (docs / tooling / CI). -->

| Backend | Built | Ran tests | Hardware |
|---------|:-----:|:---------:|----------|
| CPU     |       |           |          |
| Metal   |       |           |          |
| CUDA    |       |           |          |
| ROCm    |       |           |          |
| wgpu / Vulkan |  |           |          |
| Other   |       |           |          |

## Performance

<!-- Per-backend peak perf is the north star. Record the before/after bench
     delta for anything on a hot path — "within noise" is still data. Gate
     benches on `just throttle` first. Delete this section only for pure
     docs/tooling changes. -->

| Config | Before | After | Δ |
|--------|--------|-------|---|
|        |        |       |   |

## Checklist

<!-- Strike through (~~…~~) any row that doesn't apply — see CONTRIBUTING.md.
     `just ci` spans GPU/ROCm/wasm; run what your hardware supports, CI covers the rest. -->

- [ ] Ran what my hardware supports locally (at least `just fmt`, `just lint`, `just test`).
- [ ] The change is focused; no unrelated refactors.
- [ ] Style, comment density, and idioms match the surrounding code.
- [ ] Tests: added / updated meaningful parity or integration coverage
      (not trivial asserts).
- [ ] New op / backend work lands **native** under `RLX_DISPATCH_REPORT=1`
      (not `common-ir`) and has a cross-backend parity test vs the CPU reference.
- [ ] `just gen-op-coverage` run if op coverage changed.
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for user-visible changes.
- [ ] Docs / crate `README.md` updated where behavior or public surface changed.
- [ ] I have the right to contribute this under MIT/Apache-2.0, including any
      AI-assisted content (see `CONTRIBUTING.md` → Licensing).

## Notes for reviewers

<!-- Anything hardware you could not test on, follow-ups, or context that helps review. -->
