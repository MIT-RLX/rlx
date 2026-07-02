# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""pyrlx vs PyTorch vs JAX — same workloads, every backend each
framework can reach on this machine.

Usage
-----
    # everything that's installed + every backend each framework can see
    python bench_vs_torch_jax.py

    # subset
    python bench_vs_torch_jax.py --workloads mlp,attention --frameworks pyrlx,pytorch
    python bench_vs_torch_jax.py --shape 256,1024 --iters 200

Install (heavy — pick what you have hardware for)
-------------------------------------------------
    uv pip install torch              # PyTorch (cpu + cuda + mps)
    uv pip install -U jax             # JAX cpu
    uv pip install -U "jax[cuda12]"   # JAX cuda
    # (jax-metal on Apple Silicon is experimental — see jax docs)

The script prints version + hardware info up top, runs each
(framework, device, workload) cell, and ends with a wide table plus
parity vs the reference cell (`pyrlx/cpu` by default).
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ── Workload spec ───────────────────────────────────────────────────

@dataclass
class Workload:
    name: str
    description: str
    shape: tuple[int, ...]            # primary tensor shape (used by adapters)
    # Each adapter returns a closure that runs the workload once,
    # *synchronously* — caller can time `fn()` directly.
    pyrlx:    Optional[Callable[..., Callable[[], np.ndarray]]] = None
    pytorch:  Optional[Callable[..., Callable[[], np.ndarray]]] = None
    jax:      Optional[Callable[..., Callable[[], np.ndarray]]] = None
    tinygrad: Optional[Callable[..., Callable[[], np.ndarray]]] = None


WORKLOADS: dict[str, Workload] = {}


def register(w: Workload) -> Workload:
    WORKLOADS[w.name] = w
    return w


# ── Workload 1: MLP block (matmul + bias + gelu) ───────────────────

def _mlp_inputs(shape: tuple[int, int], rng: np.random.Generator):
    B, D = shape
    x = rng.standard_normal((B, D)).astype(np.float32)
    w = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    b = np.zeros(D, dtype=np.float32)
    return x, w, b


def _mlp_pyrlx(shape, dev, rng):
    import pyrlx as rlx
    x, w, b = _mlp_inputs(shape, rng)

    g = rlx.Graph("mlp")
    xi = g.input("x", list(x.shape), "f32")
    wi = g.param("w", list(w.shape), "f32")
    bi = g.param("b", list(b.shape), "f32")
    g.set_outputs([g.gelu(g.add(g.matmul(xi, wi), bi))])

    c = rlx.Session(device=dev).compile(g)
    c.set_param("w", w)
    c.set_param("b", b)
    inputs = {"x": x}

    def fn():
        [out] = c.run(inputs)
        return out
    return fn


def _mlp_pytorch(shape, dev, rng):
    import torch
    x, w, b = _mlp_inputs(shape, rng)
    xt = torch.from_numpy(x).to(dev)
    wt = torch.from_numpy(w).to(dev)
    bt = torch.from_numpy(b).to(dev)

    sync = _torch_sync(dev)

    def fn():
        with torch.inference_mode():
            out = torch.nn.functional.gelu(xt @ wt + bt)
        sync()
        return out.detach().cpu().numpy() if dev != "cpu" else out.detach().numpy()
    return fn


def _mlp_jax(shape, dev, rng):
    import jax
    import jax.numpy as jnp
    x, w, b = _mlp_inputs(shape, rng)
    device = _jax_device(dev)
    xt = jax.device_put(jnp.asarray(x), device)
    wt = jax.device_put(jnp.asarray(w), device)
    bt = jax.device_put(jnp.asarray(b), device)

    @jax.jit
    def step(x, w, b):
        # `approximate=False` -> erf-based gelu, matching pyrlx
        # (which uses the exact erf form) and torch's default.
        return jax.nn.gelu(x @ w + b, approximate=False)

    # Trigger compilation outside the timed loop.
    _ = step(xt, wt, bt).block_until_ready()

    def fn():
        return np.asarray(step(xt, wt, bt).block_until_ready())
    return fn


def _mlp_tinygrad(shape, dev, rng):
    from tinygrad import Tensor, TinyJit                        # type: ignore
    x, w, b = _mlp_inputs(shape, rng)
    tg = _LABEL_TO_TG[dev]
    xt = Tensor(x, device=tg).realize()
    wt = Tensor(w, device=tg).realize()
    bt = Tensor(b, device=tg).realize()

    @TinyJit
    def step(x, w, b):
        return ((x @ w + b).gelu()).realize()

    def fn():
        return step(xt, wt, bt).numpy()
    return fn


register(Workload(
    name="mlp",
    description="x[B, D] @ W[D, D] + b -> gelu  (transformer FFN-ish)",
    shape=(128, 768),
    pyrlx=_mlp_pyrlx,
    pytorch=_mlp_pytorch,
    jax=_mlp_jax,
    tinygrad=_mlp_tinygrad,
))


# ── Workload 2: LayerNorm ──────────────────────────────────────────

def _ln_inputs(shape, rng):
    B, S, D = shape
    x = rng.standard_normal((B, S, D)).astype(np.float32)
    g = np.ones (D, dtype=np.float32)
    b = np.zeros(D, dtype=np.float32)
    return x, g, b


def _ln_pyrlx(shape, dev, rng):
    import pyrlx as rlx
    x, gam, bet = _ln_inputs(shape, rng)

    g = rlx.Graph("ln")
    xi = g.input("x", list(x.shape),   "f32")
    gi = g.param("g", list(gam.shape), "f32")
    bi = g.param("b", list(bet.shape), "f32")
    g.set_outputs([g.layer_norm(xi, gi, bi)])

    c = rlx.Session(device=dev).compile(g)
    c.set_param("g", gam); c.set_param("b", bet)
    inputs = {"x": x}

    def fn():
        [out] = c.run(inputs)
        return out
    return fn


def _ln_pytorch(shape, dev, rng):
    import torch
    x, gam, bet = _ln_inputs(shape, rng)
    xt = torch.from_numpy(x).to(dev)
    gt = torch.from_numpy(gam).to(dev)
    bt = torch.from_numpy(bet).to(dev)
    D  = xt.shape[-1]
    sync = _torch_sync(dev)

    def fn():
        with torch.inference_mode():
            out = torch.nn.functional.layer_norm(xt, (D,), gt, bt, eps=1e-5)
        sync()
        return out.detach().cpu().numpy() if dev != "cpu" else out.detach().numpy()
    return fn


def _ln_jax(shape, dev, rng):
    import jax
    import jax.numpy as jnp
    x, gam, bet = _ln_inputs(shape, rng)
    device = _jax_device(dev)
    xt = jax.device_put(jnp.asarray(x), device)
    gt = jax.device_put(jnp.asarray(gam), device)
    bt = jax.device_put(jnp.asarray(bet), device)

    @jax.jit
    def step(x, g, b):
        m = x.mean(-1, keepdims=True)
        v = x.var (-1, keepdims=True)
        return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b

    _ = step(xt, gt, bt).block_until_ready()

    def fn():
        return np.asarray(step(xt, gt, bt).block_until_ready())
    return fn


def _ln_tinygrad(shape, dev, rng):
    from tinygrad import Tensor, TinyJit                        # type: ignore
    x, gam, bet = _ln_inputs(shape, rng)
    tg = _LABEL_TO_TG[dev]
    xt = Tensor(x,   device=tg).realize()
    gt = Tensor(gam, device=tg).realize()
    bt = Tensor(bet, device=tg).realize()

    @TinyJit
    def step(x, g, b):
        # Tensor.layernorm yields the standardized tensor; affine is manual.
        return (x.layernorm(axis=-1, eps=1e-5) * g + b).realize()

    def fn():
        return step(xt, gt, bt).numpy()
    return fn


register(Workload(
    name="layernorm",
    description="LayerNorm over last dim of [B, S, D]",
    shape=(8, 128, 768),
    pyrlx=_ln_pyrlx, pytorch=_ln_pytorch, jax=_ln_jax,
    tinygrad=_ln_tinygrad,
))


# ── Workload 3: Attention (SDPA, no mask) ──────────────────────────

def _attn_inputs(shape, rng):
    """Build Q/K/V as `[B, H, S, D]` — the layout torch + JAX SDPA
    use natively. pyrlx's attention op uses `[B, S, H, D]` internally
    (heads-inside-the-row); the pyrlx adapter transposes on the way in
    and the output on the way out so the parity check sees the same
    axis order across all three frameworks.

    Tensors are unscaled — every SDPA implementation applies
    `1/sqrt(head_dim)` internally, so pre-scaling here would double-
    count on pyrlx and torch.
    """
    B, H, S, D = shape
    q = rng.standard_normal((B, H, S, D)).astype(np.float32)
    k = rng.standard_normal((B, H, S, D)).astype(np.float32)
    v = rng.standard_normal((B, H, S, D)).astype(np.float32)
    return q, k, v


def _attn_pyrlx(shape, dev, rng):
    import pyrlx as rlx
    q_bhsd, k_bhsd, v_bhsd = _attn_inputs(shape, rng)
    B, H, S, D = shape
    HD = H * D

    # rlx attention uses `[B, S, H*D]` (heads inside the row);
    # transpose [B, H, S, D] -> [B, S, H, D] -> flatten last two.
    def to_bshd_flat(x):
        return np.ascontiguousarray(
            np.transpose(x, (0, 2, 1, 3)).reshape(B, S, HD))
    q = to_bshd_flat(q_bhsd); k = to_bshd_flat(k_bhsd); v = to_bshd_flat(v_bhsd)

    # rlx attention: mask values >= 0.5 mean "attend"; ones across
    # the board = no masking. Portable across backends (Metal SDPA's
    # synthetic-None path isn't implemented yet, so we always use a
    # Custom mask). Per-key, shape [B, S].
    mask = np.ones((B, S), dtype=np.float32)

    g = rlx.Graph("attn")
    qi = g.input("q",    [B, S, HD], "f32")
    ki = g.input("k",    [B, S, HD], "f32")
    vi = g.input("v",    [B, S, HD], "f32")
    mi = g.input("mask", [B, S],     "f32")
    g.set_outputs([g.attention(qi, ki, vi, mi, num_heads=H, head_dim=D)])

    c = rlx.Session(device=dev).compile(g)
    inputs = {"q": q, "k": k, "v": v, "mask": mask}

    def fn():
        [out_flat] = c.run(inputs)
        # [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D] for parity
        out = out_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        return np.ascontiguousarray(out)
    return fn


def _attn_pytorch(shape, dev, rng):
    import torch
    q, k, v = _attn_inputs(shape, rng)
    qt = torch.from_numpy(q).to(dev)
    kt = torch.from_numpy(k).to(dev)
    vt = torch.from_numpy(v).to(dev)
    sync = _torch_sync(dev)

    def fn():
        with torch.inference_mode():
            out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
        sync()
        return out.detach().cpu().numpy() if dev != "cpu" else out.detach().numpy()
    return fn


def _attn_jax(shape, dev, rng):
    import jax
    import jax.numpy as jnp
    q, k, v = _attn_inputs(shape, rng)
    B, H, S, D = shape
    device = _jax_device(dev)
    qt = jax.device_put(jnp.asarray(q), device)
    kt = jax.device_put(jnp.asarray(k), device)
    vt = jax.device_put(jnp.asarray(v), device)

    @jax.jit
    def step(q, k, v):
        scores = jnp.einsum("bhsd,bhtd->bhst", q, k) / jnp.sqrt(D).astype(q.dtype)
        attn   = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhst,bhtd->bhsd", attn, v)

    _ = step(qt, kt, vt).block_until_ready()

    def fn():
        return np.asarray(step(qt, kt, vt).block_until_ready())
    return fn


def _attn_tinygrad(shape, dev, rng):
    from tinygrad import Tensor, TinyJit                        # type: ignore
    q, k, v = _attn_inputs(shape, rng)
    tg = _LABEL_TO_TG[dev]
    qt = Tensor(q, device=tg).realize()
    kt = Tensor(k, device=tg).realize()
    vt = Tensor(v, device=tg).realize()

    @TinyJit
    def step(q, k, v):
        # tinygrad's SDPA expects [B, H, S, D] like torch/jax — same input
        # layout as the rest of the bench. is_causal=False = no mask.
        return q.scaled_dot_product_attention(k, v, is_causal=False).realize()

    def fn():
        return step(qt, kt, vt).numpy()
    return fn


register(Workload(
    name="attention",
    description="SDPA on [B, H, S, D] -- softmax(QK^T/sqrt(d)) @ V, no mask",
    shape=(1, 12, 128, 64),
    pyrlx=_attn_pyrlx, pytorch=_attn_pytorch, jax=_attn_jax,
    tinygrad=_attn_tinygrad,
))


# ── Sync / device helpers ──────────────────────────────────────────

def _torch_sync(dev: str) -> Callable[[], None]:
    import torch
    if dev == "cuda":
        return torch.cuda.synchronize
    if dev == "mps":
        return torch.mps.synchronize
    return lambda: None


def _torch_devices() -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def _jax_device(name: str):
    import jax
    # Map "cpu"/"gpu"/"metal" -> JAX device.
    backend_map = {"cpu": "cpu", "cuda": "gpu", "gpu": "gpu", "metal": "METAL"}
    backend = backend_map.get(name, name)
    try:
        return jax.devices(backend)[0]
    except Exception:
        return jax.devices()[0]


def _jax_devices() -> list[str]:
    try:
        import jax
    except ImportError:
        return []
    out = []
    for plat in ("cpu", "gpu", "METAL"):
        try:
            ds = jax.devices(plat)
            if ds:
                out.append({"cpu": "cpu", "gpu": "cuda", "METAL": "metal"}[plat])
        except Exception:
            pass
    return out


# tinygrad ↔ bench-column mapping (tinygrad 0.10+ renamed CLANG→CPU,
# GPU→CL, added NV alongside CUDA). tinygrad's CL backend is OpenCL
# (different API from pyrlx's `gpu`/wgpu) so it gets its own column —
# `WEBGPU` maps to the same `gpu` column as pyrlx since both are wgpu.
# `PYTHON` is the bytecode interpreter; included only as a last-resort
# fallback when no compiled CPU backend is available.
_TG_TO_LABEL: dict[str, str] = {
    "CPU":    "cpu",
    "PYTHON": "cpu",
    "METAL":  "metal",
    "CUDA":   "cuda",
    "NV":     "cuda",       # alternative cuda runtime in tinygrad
    "HIP":    "rocm",
    "AMD":    "rocm",       # newer alias for HIP in tinygrad
    "CL":     "opencl",
    "WEBGPU": "gpu",
}
# Reverse: bench column → preferred tinygrad device. Iteration order
# of `_TG_TO_LABEL` controls which tinygrad backend wins for each label
# (e.g. CLANG > LLVM > CPU for the "cpu" column).
_LABEL_TO_TG: dict[str, str] = {}
for _tg, _label in _TG_TO_LABEL.items():
    _LABEL_TO_TG.setdefault(_label, _tg)


def _tinygrad_devices() -> list[str]:
    """Probe tinygrad backends by trying a one-element realize on each.
    Returns bench-column labels in the order they were discovered."""
    try:
        from tinygrad import Tensor                            # type: ignore
    except ImportError:
        return []
    seen: list[str] = []
    for tg_name, label in _TG_TO_LABEL.items():
        if label in seen:
            continue
        try:
            _ = Tensor([1.0, 2.0, 3.0], device=tg_name).realize().numpy()
        except Exception:                                       # noqa: BLE001
            continue
        # Lock the bench column to the tinygrad backend that worked.
        _LABEL_TO_TG[label] = tg_name
        seen.append(label)
    return seen


def _pyrlx_devices() -> list[str]:
    try:
        import pyrlx
    except ImportError:
        return []
    return list(pyrlx.available_devices())


# ── Timing core ────────────────────────────────────────────────────

@dataclass
class CellResult:
    framework: str
    device: str
    workload: str
    median_ms: float
    p99_ms:    float
    iters:     int
    output:    np.ndarray = field(repr=False)
    error:     Optional[str] = None


def _bench(fn, iters: int, warmup: int) -> tuple[float, float]:
    """Return (median_ms, p99_ms) over `iters` post-warmup runs."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return statistics.median(samples), samples[max(0, int(0.99 * len(samples)) - 1)]


def run_cell(framework: str, device: str, workload: Workload,
             iters: int, warmup: int, seed: int) -> CellResult:
    rng = np.random.default_rng(seed)
    builder = getattr(workload, framework)
    if builder is None:
        return CellResult(framework, device, workload.name, 0, 0, 0,
                          np.empty(0), error="no adapter")
    # PyO3 turns Rust panics into BaseException-rooted PanicException;
    # catch BaseException so a panic in one cell doesn't kill the
    # whole bench. We exclude KeyboardInterrupt / SystemExit to keep
    # ^C and exit() honest.
    try:
        fn = builder(workload.shape, device, rng)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:                                  # noqa: BLE001
        return CellResult(framework, device, workload.name, 0, 0, 0,
                          np.empty(0), error=f"setup failed: {_short(e)}")
    try:
        out = fn()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:                                  # noqa: BLE001
        return CellResult(framework, device, workload.name, 0, 0, 0,
                          np.empty(0), error=f"first run failed: {_short(e)}")
    try:
        med, p99 = _bench(fn, iters=iters, warmup=warmup)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:                                  # noqa: BLE001
        return CellResult(framework, device, workload.name, 0, 0, 0,
                          np.empty(0), error=f"timing failed: {_short(e)}")
    return CellResult(framework, device, workload.name, med, p99, iters, out)


def _short(e: BaseException) -> str:
    """Trim long panic messages to a single line for the error block."""
    return str(e).strip().splitlines()[0][:200]


# ── Reporting ──────────────────────────────────────────────────────

def system_info() -> str:
    parts = [
        f"Python {platform.python_version()}  ({platform.platform()})",
        f"CPU cores: {platform.machine()}, {_cpu_count()}",
    ]
    try:
        import pyrlx; parts.append(f"pyrlx   {pyrlx.__version__}  devices={pyrlx.available_devices()}")
    except ImportError:
        parts.append("pyrlx   not installed")
    try:
        import torch; parts.append(f"torch   {torch.__version__}  devices={_torch_devices()}")
    except ImportError:
        parts.append("torch   not installed")
    try:
        import jax;   parts.append(f"jax     {jax.__version__}  devices={_jax_devices()}")
    except ImportError:
        parts.append("jax     not installed")
    try:
        import tinygrad
        ver = getattr(tinygrad, "__version__", "?")
        parts.append(f"tinygrad{ver:>7}  devices={_tinygrad_devices()}")
    except ImportError:
        parts.append("tinygrad not installed")
    return "\n".join(parts)


def _cpu_count():
    import os
    return os.cpu_count()


def _format_table(rows: list[CellResult], frameworks: list[str], all_devices: list[str]):
    # Per workload, one row per framework, one column per device.
    by_workload: dict[str, list[CellResult]] = {}
    for r in rows:
        by_workload.setdefault(r.workload, []).append(r)

    out = []
    col_w = 12
    name_w = max(len(f) for f in frameworks) + 2
    for wl_name, wl_rows in by_workload.items():
        wl = WORKLOADS[wl_name]
        out.append("")
        out.append(f"== {wl.name}  shape={wl.shape}  ({wl.description}) ==")
        header = " " * name_w + "".join(f"{d:>{col_w}}" for d in all_devices)
        out.append(header)
        for fw in frameworks:
            cells = []
            for dev in all_devices:
                hit = next((r for r in wl_rows if r.framework == fw and r.device == dev), None)
                if hit is None:
                    cells.append("-")
                elif hit.error:
                    cells.append("err")
                else:
                    cells.append(f"{hit.median_ms:.3f} ms")
            out.append(f"{fw:<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cells))
    return "\n".join(out)


def _format_parity(rows: list[CellResult], reference: tuple[str, str]) -> str:
    fw_ref, dev_ref = reference
    by_workload: dict[str, list[CellResult]] = {}
    for r in rows:
        by_workload.setdefault(r.workload, []).append(r)

    lines = ["", f"parity (max|Δ| vs {fw_ref}/{dev_ref}):"]
    for wl_name, wl_rows in by_workload.items():
        ref = next((r for r in wl_rows if r.framework == fw_ref and r.device == dev_ref
                    and not r.error), None)
        if ref is None or ref.output.size == 0:
            lines.append(f"  {wl_name}: no reference")
            continue
        for r in wl_rows:
            if r is ref or r.error or r.output.size == 0:
                continue
            try:
                diff = float(np.max(np.abs(r.output.astype(np.float32)
                                           - ref.output.astype(np.float32))))
            except Exception as e:                                      # noqa: BLE001
                diff = float("nan")
                err = f" ({e})"
            else:
                err = ""
            lines.append(f"  {wl_name:<10} {r.framework}/{r.device}: {diff:.3e}{err}")
    return "\n".join(lines)


def _format_errors(rows: list[CellResult]) -> str:
    errs = [r for r in rows if r.error and r.error != "no adapter"]
    if not errs:
        return ""
    return "\nerrors:\n" + "\n".join(
        f"  {r.framework}/{r.device}/{r.workload}: {r.error}" for r in errs)


def _format_speedups(rows: list[CellResult], _reference: tuple[str, str]) -> str:
    """One line per workload: who won, and pyrlx's headline number
    (best pyrlx cell vs best non-pyrlx cell). Lets you read the table
    at a glance: did pyrlx win, and on which device?"""
    by_workload: dict[str, list[CellResult]] = {}
    for r in rows:
        by_workload.setdefault(r.workload, []).append(r)

    lines = ["", "headline (lower median = faster):"]
    for wl_name, wl_rows in by_workload.items():
        ok = [r for r in wl_rows if not r.error and r.median_ms > 0]
        if not ok:
            continue
        winner = min(ok, key=lambda r: r.median_ms)

        pyrlx_best = min((r for r in ok if r.framework == "pyrlx"),
                         key=lambda r: r.median_ms, default=None)
        other_best = min((r for r in ok if r.framework != "pyrlx"),
                         key=lambda r: r.median_ms, default=None)

        bits = [f"winner: {winner.framework}/{winner.device} "
                f"({winner.median_ms:.3f} ms)"]
        if pyrlx_best is not None and other_best is not None:
            if pyrlx_best.median_ms <= other_best.median_ms:
                ratio = other_best.median_ms / pyrlx_best.median_ms
                verdict = "faster"
            else:
                ratio = pyrlx_best.median_ms / other_best.median_ms
                verdict = "slower"
            bits.append(
                f"pyrlx/{pyrlx_best.device} {ratio:.2f}× {verdict} "
                f"than {other_best.framework}/{other_best.device}")
        lines.append(f"  {wl_name:<10} " + " | ".join(bits))
    return "\n".join(lines)


def _to_json(rows: list[CellResult], reference: tuple[str, str]) -> str:
    """Machine-readable dump — suitable for CI dashboards / regression
    tracking. One JSON document, no trailing newline."""
    import json
    fw_ref, dev_ref = reference
    payload = {
        "schema": 1,
        "system": {
            "python":   platform.python_version(),
            "platform": platform.platform(),
            "machine":  platform.machine(),
            "cpu_count": _cpu_count(),
        },
        "frameworks": _framework_versions(),
        "reference": f"{fw_ref}/{dev_ref}",
        "results": [
            {
                "framework": r.framework,
                "device":    r.device,
                "workload":  r.workload,
                "median_ms": r.median_ms,
                "p99_ms":    r.p99_ms,
                "iters":     r.iters,
                "error":     r.error,
            } for r in rows
        ],
    }
    return json.dumps(payload, indent=2)


def _framework_versions() -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        import pyrlx
        out["pyrlx"] = {"version": pyrlx.__version__,
                        "devices": list(pyrlx.available_devices())}
    except ImportError:
        pass
    try:
        import torch
        out["pytorch"] = {"version": torch.__version__, "devices": _torch_devices()}
    except ImportError:
        pass
    try:
        import jax
        out["jax"] = {"version": jax.__version__, "devices": _jax_devices()}
    except ImportError:
        pass
    try:
        import tinygrad
        out["tinygrad"] = {
            "version": getattr(tinygrad, "__version__", "?"),
            "devices": _tinygrad_devices(),
        }
    except ImportError:
        pass
    return out


# ── CLI / driver ───────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frameworks", default="pyrlx,pytorch,jax,tinygrad",
                   help="comma list (default: all installed)")
    p.add_argument("--workloads",  default=",".join(WORKLOADS.keys()),
                   help=f"comma list (default: {','.join(WORKLOADS.keys())})")
    p.add_argument("--shape", default=None,
                   help="comma-separated ints; overrides workload's default")
    p.add_argument("--iters",  type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--reference", default="pyrlx/cpu",
                   help="cell to compare parity against (framework/device)")
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON to stdout instead of the table")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the per-cell progress line")
    args = p.parse_args()

    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    workloads  = [w.strip() for w in args.workloads.split(",")  if w.strip()]

    fw_devices: dict[str, list[str]] = {
        "pyrlx":    _pyrlx_devices(),
        "pytorch":  _torch_devices(),
        "jax":      _jax_devices(),
        "tinygrad": _tinygrad_devices(),
    }

    selected_workloads = []
    for w in workloads:
        if w not in WORKLOADS:
            print(f"unknown workload: {w}", file=sys.stderr); sys.exit(2)
        wl = WORKLOADS[w]
        if args.shape:
            wl.shape = tuple(int(s) for s in args.shape.split(","))
        selected_workloads.append(wl)

    # Build the union of devices for column ordering.
    seen, all_devices = set(), []
    for fw in frameworks:
        for d in fw_devices.get(fw, []):
            if d not in seen:
                seen.add(d); all_devices.append(d)

    if not args.json:
        print(system_info())

    rows: list[CellResult] = []
    for wl in selected_workloads:
        for fw in frameworks:
            devs = fw_devices.get(fw, [])
            if not devs:
                rows.append(CellResult(fw, "-", wl.name, 0, 0, 0, np.empty(0),
                                       error="not installed"))
                continue
            for dev in devs:
                if not args.quiet and not args.json:
                    print(f"... {fw:<8} {dev:<6} {wl.name}",
                          file=sys.stderr, flush=True)
                rows.append(run_cell(fw, dev, wl, args.iters, args.warmup, args.seed))

    fw_ref, dev_ref = args.reference.split("/", 1)

    if args.json:
        print(_to_json(rows, (fw_ref, dev_ref)))
        return

    print(_format_table(rows, frameworks, all_devices))
    print(_format_speedups(rows, (fw_ref, dev_ref)))
    print(_format_parity(rows, (fw_ref, dev_ref)))

    err_block = _format_errors(rows)
    if err_block:
        print(err_block)


if __name__ == "__main__":
    main()
