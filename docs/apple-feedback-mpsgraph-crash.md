# Feedback Assistant draft — MPSGraphExecutable null deref in MPSGraphOSLog

Not filed. Submit via Feedback Assistant (macOS → Metal / MetalPerformanceShadersGraph)
and attach the `.ips` reports from `~/Library/Logs/DiagnosticReports/`.

## Summary

`-[MPSGraph compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:]`
with a **nil** compilation descriptor crashes at a measured **~2% per executable
initialization**. The fault is a null dereference inside
MetalPerformanceShadersGraph itself, on its own `MPSGraphExecutable_queue`, with
no caller frames on the faulting thread.

## Environment

- macOS 26.x, Apple Silicon (M4 Pro), arm64
- Crash observed across many unrelated client processes; the client is a Rust
  binary calling MPSGraph through the ObjC runtime.

## Crash signature (identical in every instance, N > 8)

```
EXC_BAD_ACCESS  KERN_INVALID_ADDRESS at 0x10 / 0x18
vmRegionInfo:   "0x18 is not in any region"        → null + field offset
faulting queue: MPSGraphExecutable_queue

  MetalPerformanceShadersGraph  MPSGraphOSLog(MPSGraphLogCategory, bool, int, int)
  MetalPerformanceShadersGraph  GPURegionRuntime::initializeOps()
  MetalPerformanceShadersGraph  GPURegionRuntime::GPURegionRuntime(MPSGraphDevice*, ...)
  MetalPerformanceShadersGraph  -[MPSGraphExecutable getNewRuntimeForDevice:specializedModule:
                                   shapedEntryPoints:compilationDescriptor:]
  MetalPerformanceShadersGraph  -[MPSGraphExecutable specializedModuleWithDevice:...]
  libdispatch                   _dispatch_lane_serial_drain ...
```

Registers at fault: `x0=0x3` (a `MPSGraphLogCategory`), `x1=0x1`, `x5=0x0`,
`x9=0x0` — a null global read at `+0x18`.

## Analysis

`MPSGraphOSLog` guards a `std::call_once` lazy initialization:

```
adrp x8, …; add x8, x8, #0xb10   ; _MergedGlobals        ← init-done flag byte
ldaprb w8, [x8]; tbz w8, #0x0, …                          ← unset ⇒ slow path
adrp x8, …; add x8, x8, #0xb18   ; _MergedGlobals + 8     ← std::once_flag
cmn x8, #0x1; b.eq …                                      ← == -1 ⇒ already run
… __call_once_proxy<std::tuple<MPSGraphOSLog(…)::$_0&&>>
```

Consistent with the one-shot initialization occasionally producing a null handle
that is then cached permanently; the next logging call dereferences it. This
matches the observed behaviour: the failure rate is flat **per process**, not per
call, and does not depend on client thread count.

## Reproduction characteristics (measured)

| condition | executable inits | crashes |
|---|---|---|
| single process, nothing else running | 60 | 1 |
| 5 processes × 20 rounds | ~100 | 2–4 |
| staggered launches (0.4 s apart) | ~75 | 1 |

Ruled out by measurement, i.e. these are **not** the cause:

- client-side threading — the crash report shows a single thread in MPSGraph
  (3–4 threads total in the process);
- client memory pressure — `vmSummary` shows `written=561K`; the process dies at
  first runtime initialization;
- client-side serialization of the compile call — the work is dispatched
  asynchronously by the framework, so caller locks have no effect;
- process launch timing / contention — staggering matches burst rates;
- `OS_ACTIVITY_MODE=disable` — 4 failures in 40 rounds, no improvement.

## Workaround

Passing an explicit `MPSGraphCompilationDescriptor` with
`waitForCompilationCompletion = YES` avoids the deferred path entirely:
**0 crashes in ~260 executable initializations** (vs ≈5 expected at the 2%
baseline, p ≈ 0.005), with no measurable compile-latency regression.

## Re-measured on macOS 26.4.1 — the baseline no longer reproduces

Before filing, the workaround was re-tested against its own control by
restoring the nil-descriptor path (`RLX_MPSGRAPH_NO_SYNC_COMPILE=1`) and
re-running the high-init workload (`examples/mpsgraph_prefill_bench`,
28 executable inits per process):

| condition | executable inits | crashes |
|---|---|---|
| workaround **disabled** (nil descriptor, i.e. the original bug path) | ~700 | **0** |
| workaround enabled | ~700 | 0 |

At the 2% baseline, 700 inits predicts ≈14 crashes; observing zero has
p ≈ 6e-7. So on **macOS 26.4.1 (25E253), M4 Pro** the fault does not reproduce
*with or without* the workaround.

Two consequences:

1. The workaround's efficacy is **unproven** on this OS — there is no longer a
   baseline for it to improve on. It is retained because it costs nothing
   measurable, not because it is demonstrated to fix anything.
2. The original measurements above were taken on an earlier macOS 26.x point
   release. If this is filed, it should say so explicitly and ask whether the
   null dereference was addressed between those releases — otherwise the report
   describes a fault Apple may already have fixed.

Re-run either arm with `crates/backends/rlx-metal/scripts/mpsgraph-soak.sh`.

## Expected behaviour

`compileWithDevice:…compilationDescriptor:nil` should not crash. If the logging
handle cannot be created, the logging path should tolerate a null handle rather
than dereference it.
