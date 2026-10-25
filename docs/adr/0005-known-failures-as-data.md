# ADR 0005 — Expected test failures are data files linked to issues

## Status

Proposed — 2026-10-22.

## Context

Cross-backend harnesses find defects that cannot all be fixed at once (the
wgpu-on-Vulkan DSP divergences, for one). The options were a skip (the case
never runs again and nobody learns when it is fixed), `#[ignore]` (the same,
for a whole test), or a red suite (which hides every new failure behind the
known one). The Torq NPU compiler handles this with per-test expected-failure
files whose entries name an issue; that design carries over.

## Decision

Expected failures are lines in `tests/known_failures/<test>.txt` next to the
test binary: `case  device[@os]  issue  [note]`. The issue column is required.
`rlx_ir::known_failures` loads them, and harnesses route each case through
`KnownFailures::record` (case matrices) or `KnownFailures::check` (single
tests):

- a listed case still runs; its failure is reported as known;
- a listed case that **passes** fails the test until the entry is removed;
- an unlisted failure fails as before.

`RLX_RUN_KNOWN_FAILURES`, `RLX_KNOWN_FAILURES_ISSUE`,
`RLX_KNOWN_FAILURES_REPORT` and `RLX_KNOWN_FAILURES_DIR` (rig-local lists)
control it; `scripts/known-failures.py` (`just known-failures`) applies a
report to the files.

## Alternatives

- **Skips gated on device and OS in code.** What existed; the case never runs
  where it is skipped, so fixes go unnoticed and the reason lives in a comment.
- **`#[ignore]` with a reason string.** Per-test, not per-case or per-device,
  and nothing reruns it.

## Consequences

- A fix is noticed the first time the suite runs on the affected device.
- Adopting a harness is a one-place change (`gpu_filters_parity`'s `parity`,
  `fd_backward_gate`'s `gate`); other harnesses still skip in code.
- Entries need real issues, so filing one is part of listing a failure.
## License

MIT OR Apache-2.0.
