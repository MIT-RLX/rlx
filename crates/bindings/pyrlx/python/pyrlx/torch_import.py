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
"""PyTorch → RLX model converter (front-end).

This is the Python half of the ``pytorch code → rlx model`` tool. It takes a
live ``torch.nn.Module`` plus example inputs, runs ``torch.export`` +
``run_decompositions()`` to lower it to **Core ATen IR** (a small, stable op
set), then serialises that graph into a language-neutral intermediate document
``torch-ir.json`` alongside ``weights.safetensors``. The Rust crate
``rlx-torch-import`` consumes those two files, maps each ATen op onto an RLX
``Graph`` builder call, and emits a runnable bundle and/or a generated RLX crate.

Why ``torch.export``: every FX node carries ``node.meta['val']`` (a FakeTensor
with concrete shape + dtype), which supplies the explicit output shapes RLX's
graph builders require, and ``graph_signature`` cleanly separates user inputs
from parameters/buffers. High-level ops (``scaled_dot_product_attention``,
``native_layer_norm``, ``convolution``) can be *preserved* as single nodes so
they map straight onto ``Op::Attention`` / ``Op::LayerNorm`` / ``Op::Conv``.

Entry points
------------
- :func:`export_torch_ir` — module → ``torch-ir.json`` + ``weights.safetensors``
  (+ optional ``reference.npz`` golden I/O for parity). Pure front-end.
- :func:`from_torch` — the full one-call tool: front-end, then invoke the Rust
  ``rlx-torch-import`` binary to build the bundle / crate and verify parity.

The intermediate schema is documented in ``crates/io/rlx-torch-import`` and in
:data:`IR_VERSION` / the ``_build_ir`` docstring below.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

IR_VERSION = 1
IR_FORMAT = "rlx-torch-ir"

# ── dtype mapping ────────────────────────────────────────────────────────────
# torch dtype string (as reported by ``str(torch.dtype)`` w/o the ``torch.``
# prefix) → RLX ``DType`` token used by the Rust side.
_DTYPE_MAP = {
    "float32": "f32",
    "float": "f32",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "float64": "f64",
    "double": "f64",
    "int64": "i64",
    "long": "i64",
    "int32": "i32",
    "int": "i32",
    "int16": "i16",
    "short": "i16",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
    "complex64": "c64",
}


def _torch_dtype_to_rlx(dt: Any) -> str:
    s = str(dt).replace("torch.", "")
    if s not in _DTYPE_MAP:
        raise NotImplementedError(f"unsupported torch dtype: {dt!r}")
    return _DTYPE_MAP[s]


# ── op-preservation ──────────────────────────────────────────────────────────
def _preserved_ops(torch_mod) -> list:
    """Core-aten decompositions we *skip*, keeping the high-level op as one node.

    Preserving these lets the Rust side map them 1:1 onto fused RLX ops instead
    of reconstructing them from a soup of matmul/softmax/mul. Anything not
    available in the running torch build is silently ignored.
    """
    aten = torch_mod.ops.aten
    names = [
        "scaled_dot_product_attention",
        "native_layer_norm",
        "layer_norm",
        "_softmax",
        "softmax",
        "gelu",
        "silu",
        "convolution",
        "native_group_norm",
        "group_norm",
        "embedding",
        "rms_norm",
        "native_batch_norm",
        "_native_batch_norm_legit_no_training",
    ]
    out = []
    for n in names:
        op = getattr(aten, n, None)
        if op is None:
            continue
        # default_decompositions().pop() only accepts a concrete OpOverload
        # (has a `.default`); overload *packets* would raise. Skip those.
        default = getattr(op, "default", None)
        if default is not None:
            out.append(default)
    return out


def _base_name(name: str) -> str:
    """``aten.roll.default`` → ``aten.roll`` (drop the overload)."""
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else name


def _full_decomp_registry(torch_mod):
    """torch's *full* op→decomposition registry (larger than the export default);
    used to force-decompose ops the RLX registry doesn't cover yet."""
    return getattr(getattr(torch_mod, "_decomp", None), "decomposition_table", None) or {}


def _parse_unsupported_ops(text: str) -> set:
    """Pull the ``aten.*`` names out of the Rust importer's "registry does not yet
    support …" error, so the front-end can decompose them and retry."""
    m = re.search(r"does not yet support.*?op\(s\):(.*?)(?:Extend crates|\Z)", text, re.S)
    if not m:
        return set()
    return set(re.findall(r"aten\.[A-Za-z0-9_.]+", m.group(1)))


def _decomposable_ops(torch_mod, names: set) -> set:
    """Subset of `names` that torch actually *has* a decomposition for — the only
    ones worth force-decomposing (others would loop without progress)."""
    reg = _full_decomp_registry(torch_mod)
    have, have_base = set(), set()
    for op in reg:
        s = str(op)
        have.add(s)
        have_base.add(_base_name(s))
    return {n for n in names if n in have or _base_name(n) in have_base}


def _augment_decomp_table(torch_mod, table: dict, force_decompose: set) -> dict:
    """Add decompositions for `force_decompose` ops (from the full registry) so
    ``run_decompositions`` breaks them into smaller ops the RLX registry covers."""
    if not force_decompose:
        return table
    bases = {_base_name(n) for n in force_decompose}
    for op, fn in _full_decomp_registry(torch_mod).items():
        s = str(op)
        if s in force_decompose or _base_name(s) in bases:
            table[op] = fn
    return table


def _accept_decompositions(export_fn, base: set, candidates: set) -> set:
    """Return the largest superset of `base` (⊆ `base ∪ candidates`) that still
    exports. Some decompositions emit ``prims.*`` ops that break torch's
    functionalization; those are dropped one at a time (bisect only on failure)
    so a single bad op can't crash the whole import. `base` must already export.
    """
    combined = base | candidates
    try:
        export_fn(combined)
        return combined
    except Exception:
        pass
    accepted = set(base)
    for op in sorted(candidates):
        try:
            export_fn(accepted | {op})
            accepted.add(op)
        except Exception:
            print(
                f"[from_torch] {op}: decomposition breaks torch.export; leaving unsupported",
                file=sys.stderr,
            )
    return accepted


def _decomp_table(torch_mod, preserve_high_level: bool, force_decompose: Optional[set] = None):
    """Return a decomposition table (or ``None`` for full core-aten decomp).

    ``preserve_high_level`` keeps sdpa/conv/norm/… as single fused ops. Any op in
    ``force_decompose`` is instead broken down (via torch's full registry) — the
    auto-fallback path uses this to shrink the unsupported-op tail.
    """
    export_mod = torch_mod.export
    getter = getattr(export_mod, "default_decompositions", None)
    if getter is None:
        # Older torch: no way to filter — fall back to full decomposition.
        return None
    if not preserve_high_level and not force_decompose:
        return None  # plain full core-aten decomposition
    table = getter()
    if preserve_high_level:
        for op in _preserved_ops(torch_mod):
            table.pop(op, None)
    _augment_decomp_table(torch_mod, table, force_decompose or set())
    return table


# ── FX value encoding ────────────────────────────────────────────────────────
def _normalize_target(target: Any) -> str:
    """``torch.ops.aten.convolution.default`` → ``aten.convolution.default``."""
    import operator

    if target is operator.getitem:
        return "_getitem"
    s = str(target)
    # OpOverload prints as e.g. "aten.convolution.default"; OpOverloadPacket as
    # "aten.convolution". builtins print as "<built-in function add>".
    return s


def _dim_example(d: Any) -> int:
    """Concrete extent of a (possibly symbolic) dim. `int(SymInt)` already returns
    its example hint in recent torch, so this is just `int(d)` with a fallback."""
    try:
        return int(d)
    except Exception:
        hint = getattr(getattr(d, "node", None), "hint", None)
        return int(hint) if hint is not None else 1


def _input_shape_dynamic(val: Any, sym_map: dict) -> tuple[list, list]:
    """`(concrete example shape, per-axis dynamic markers)` for an input
    FakeTensor. Marker `>= 0` = a dynamic dim's symbol id (stable per distinct
    SymInt via `sym_map`), `-1` = static. A `SymInt` axis (detected by its `.node`
    — plain ints have none) is dynamic; its example extent is `int(d)` (the hint)."""
    shape, dyn = [], []
    for d in val.shape:
        node = getattr(d, "node", None)  # SymInt carries `.node`; a plain int does not
        if node is None:
            shape.append(int(d))
            dyn.append(-1)  # static
        else:
            sym = sym_map.setdefault(str(node), len(sym_map))
            shape.append(_dim_example(d))
            dyn.append(sym)
    return shape, dyn


def _shape_dtype_of(val: Any):
    """Extract (shape, dtype) — or a list thereof for multi-output nodes."""
    if isinstance(val, (list, tuple)):
        return [_shape_dtype_of(v) for v in val]
    if val is None:
        return None
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        # A SymInt (dynamic) dim resolves to its example hint here: intermediate
        # node shapes only need concrete example extents (the compile pass
        # re-infers the symbolic shape from the dynamic inputs). Dynamic-ness is
        # marked on the graph *inputs* (see `_input_shape_dynamic`).
        shape = [_dim_example(d) for d in val.shape]
        return {"shape": shape, "dtype": _torch_dtype_to_rlx(val.dtype)}
    # A SymInt or python scalar produced by a shape op.
    if isinstance(val, bool):
        return {"scalar": {"bool": val}}
    if isinstance(val, int):
        return {"scalar": {"int": int(val)}}
    if isinstance(val, float):
        return {"scalar": {"float": float(val)}}
    return {"scalar": {"str": str(val)}}


def _encode_arg(a: Any, torch_mod) -> Any:
    """Encode an FX arg into the tagged-union used by ``torch-ir.json``."""
    fx = torch_mod.fx
    if isinstance(a, fx.Node):
        return {"ref": a.name}
    if a is None:
        return {"none": True}
    if isinstance(a, bool):  # bool is a subclass of int — test first
        return {"bool": a}
    if isinstance(a, int):
        return {"int": int(a)}
    if isinstance(a, float):
        return {"float": float(a)}
    if isinstance(a, str):
        return {"str": a}
    if isinstance(a, (list, tuple)):
        return {"list": [_encode_arg(x, torch_mod) for x in a]}
    if isinstance(a, torch_mod.dtype):
        return {"dtype": _torch_dtype_to_rlx(a)}
    # device / memory_format / layout / etc. — keep as a string tag.
    return {"str": str(a)}


# ── IR build ─────────────────────────────────────────────────────────────────
def _build_ir(ep, torch_mod, model_name: str) -> tuple[dict, dict]:
    """Walk an ExportedProgram → (ir_dict, weights: name→tensor).

    ``torch-ir.json`` shape::

        {
          "format": "rlx-torch-ir", "version": 1,
          "model_name": str,
          "producer": "pyrlx / torch <ver>",
          "inputs":  [{"id": fx_name, "shape": [...], "dtype": "f32"}, ...],
          "weights": [{"id": fx_name, "key": state_dict_fqn,
                       "shape": [...], "dtype": "f32", "kind": "param|buffer|const"}],
          "nodes":   [{"id": fx_name, "op": "aten...", "args": [<arg>...],
                       "kwargs": {k: <arg>}, "out": <shapedtype>|[<shapedtype>]}],
          "outputs": [{"ref": fx_name, "shape": [...], "dtype": "f32"}, ...]
        }

    ``<arg>`` is one of: {"ref":name} {"int":i} {"float":f} {"bool":b}
    {"str":s} {"none":true} {"dtype":tok} {"list":[<arg>...]}.
    ``<shapedtype>`` is {"shape":[...], "dtype":tok} (or {"scalar":{...}} for a
    non-tensor value). A multi-output node has ``out`` = list; its results are
    reached via ``_getitem`` nodes.
    """
    sig = ep.graph_signature
    state_dict = dict(ep.state_dict)
    constants = dict(getattr(ep, "constants", {}) or {})

    # fx placeholder name → (kind, fqn)
    in_to_param = dict(getattr(sig, "inputs_to_parameters", {}) or {})
    in_to_buffer = dict(getattr(sig, "inputs_to_buffers", {}) or {})
    in_to_const = dict(getattr(sig, "inputs_to_lifted_tensor_constants", {}) or {})
    user_inputs = set(getattr(sig, "user_inputs", []) or [])

    inputs: list[dict] = []
    weights: list[dict] = []
    weight_tensors: dict[str, Any] = {}
    nodes: list[dict] = []
    outputs: list[dict] = []
    sym_map: dict[str, int] = {}  # SymInt expr → dynamic symbol id (stable per graph)

    graph = ep.graph_module.graph

    for node in graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val", None)
            if node.name in in_to_param:
                kind, fqn = "param", in_to_param[node.name]
            elif node.name in in_to_buffer:
                kind, fqn = "buffer", in_to_buffer[node.name]
            elif node.name in in_to_const:
                kind, fqn = "const", in_to_const[node.name]
            else:
                # user input — SymInt-aware so `dynamic_shapes` exports work.
                if val is None or not (hasattr(val, "shape") and hasattr(val, "dtype")):
                    raise NotImplementedError(
                        f"input {node.name!r} has no tensor shape metadata"
                    )
                shape, dyn = _input_shape_dynamic(val, sym_map)
                entry = {
                    "id": node.name,
                    "shape": shape,
                    "dtype": _torch_dtype_to_rlx(val.dtype),
                }
                if any(d >= 0 for d in dyn):
                    entry["dynamic"] = dyn
                inputs.append(entry)
                continue
            tensor = state_dict.get(fqn, None)
            if tensor is None:
                tensor = constants.get(fqn, None)
            if tensor is None:
                raise KeyError(f"weight {fqn!r} ({kind}) not found in state_dict/constants")
            weight_tensors[fqn] = tensor
            weights.append(
                {
                    "id": node.name,
                    "key": fqn,
                    "shape": [int(d) for d in tensor.shape],
                    "dtype": _torch_dtype_to_rlx(tensor.dtype),
                    "kind": kind,
                }
            )
        elif node.op == "call_function":
            out = _shape_dtype_of(node.meta.get("val", None))
            nodes.append(
                {
                    "id": node.name,
                    "op": _normalize_target(node.target),
                    "args": [_encode_arg(a, torch_mod) for a in node.args],
                    "kwargs": {k: _encode_arg(v, torch_mod) for k, v in node.kwargs.items()},
                    "out": out,
                }
            )
        elif node.op == "get_attr":
            # A tensor attribute referenced directly (rare after export). Treat
            # like a constant weight.
            target = node.target
            tensor = getattr(ep.graph_module, target, None)
            if tensor is None:
                raise NotImplementedError(f"get_attr {target!r} could not be resolved")
            weight_tensors[target] = tensor
            weights.append(
                {
                    "id": node.name,
                    "key": target,
                    "shape": [int(d) for d in tensor.shape],
                    "dtype": _torch_dtype_to_rlx(tensor.dtype),
                    "kind": "const",
                }
            )
        elif node.op == "output":
            flat = node.args[0]
            if not isinstance(flat, (list, tuple)):
                flat = (flat,)
            for item in flat:
                if item is None:
                    continue
                if hasattr(item, "name"):  # fx.Node
                    sd = _shape_dtype_of(item.meta.get("val", None))
                    entry = {"ref": item.name}
                    if isinstance(sd, dict) and "shape" in sd:
                        entry["shape"] = sd["shape"]
                        entry["dtype"] = sd["dtype"]
                    outputs.append(entry)
                else:
                    outputs.append({"const": _encode_arg(item, torch_mod)})
        # 'call_method' / 'call_module' should not appear post-export.

    ir = {
        "format": IR_FORMAT,
        "version": IR_VERSION,
        "model_name": model_name,
        "producer": f"pyrlx / torch {torch_mod.__version__}",
        "inputs": inputs,
        "weights": weights,
        "nodes": nodes,
        "outputs": outputs,
    }
    return ir, weight_tensors


# ── public: front-end only ───────────────────────────────────────────────────
def export_torch_ir(
    model: Any,
    example_inputs: Sequence[Any],
    out_dir: Union[str, Path],
    *,
    model_name: Optional[str] = None,
    decomposition: str = "high",
    preserve_high_level: Optional[bool] = None,
    write_reference: bool = True,
    strict: bool = False,
    force_decompose: Optional[set] = None,
    dynamic_shapes: Any = None,
) -> dict:
    """Export a ``torch.nn.Module`` to ``torch-ir.json`` + ``weights.safetensors``.

    ``decomposition`` controls how much the graph is lowered — a trade-off
    between staying close to the original module (easier to reconstruct) and
    landing on primitives the RLX registry covers:

    - ``"aten"`` — no ``run_decompositions``; keep the raw exported aten graph
      (highest level, closest to the source; some ops may be unsupported).
    - ``"high"`` — decompose to Core ATen but **preserve** high-level ops
      (sdpa/layer_norm/conv/…) so they map 1:1 onto fused RLX ops. *(default)*
    - ``"core"`` — full Core ATen decomposition (most primitive).

    The emitted ``torch-ir.json`` records every op with its args/shapes/dtypes,
    and the generated crate annotates each RLX op with its aten provenance, so a
    model can be traced/reconstructed regardless of level.

    Returns a summary dict (paths, op counts). Also writes ``reference.safetensors``
    (golden inputs+outputs) when ``write_reference`` so the Rust side can verify
    numeric parity without importing torch.
    """
    import torch  # local import: front-end only needs torch here

    # Back-compat: `preserve_high_level=False` maps to full core decomposition.
    if preserve_high_level is not None and decomposition == "high":
        decomposition = "high" if preserve_high_level else "core"
    if decomposition not in ("aten", "high", "core"):
        raise ValueError(f"decomposition must be aten|high|core, got {decomposition!r}")

    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    example_inputs = tuple(example_inputs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or type(model).__name__.lower()

    model = model.eval()
    with torch.no_grad():
        ep = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict
        )
        if decomposition == "high":
            ep = ep.run_decompositions(_decomp_table(torch, True, force_decompose))
        elif decomposition == "core":
            ep = ep.run_decompositions(_decomp_table(torch, False, force_decompose))
        # "aten": keep the exported graph as-is (least decomposed).

    ir, weight_tensors = _build_ir(ep, torch, model_name)
    ir["decomposition"] = decomposition

    ir_path = out_dir / "torch-ir.json"
    ir_path.write_text(json.dumps(ir, indent=1))

    # weights.safetensors keyed by state_dict FQN (HF-canonical for HF models).
    from safetensors.torch import save_file

    st = {}
    for k, t in weight_tensors.items():
        tt = t.detach().cpu().contiguous()
        if tt.dtype in (torch.float64,):
            tt = tt.to(torch.float32)
        st[k] = tt
    weights_path = out_dir / "weights.safetensors"
    if st:
        save_file(st, str(weights_path))
    else:
        weights_path.write_bytes(b"")  # placeholder for weightless models

    ref_path = None
    if write_reference:
        with torch.no_grad():
            ref_out = model(*example_inputs)
        if isinstance(ref_out, torch.Tensor):
            ref_out = (ref_out,)
        elif isinstance(ref_out, dict):
            ref_out = tuple(ref_out.values())
        ref: dict[str, Any] = {}
        for inp, meta in zip(example_inputs, ir["inputs"]):
            # Preserve the input's native dtype (e.g. int64 token ids) so the
            # Rust side can feed integer inputs via run_typed.
            ref[f"in::{meta['id']}"] = inp.detach().cpu().contiguous()
        for i, o in enumerate(ref_out):
            if isinstance(o, torch.Tensor):
                ref[f"out::{i}"] = o.detach().cpu().float().contiguous()
        ref_path = out_dir / "reference.safetensors"
        save_file(ref, str(ref_path))

    summary = {
        "model_name": model_name,
        "ir_path": str(ir_path),
        "weights_path": str(weights_path),
        "reference_path": str(ref_path) if ref_path else None,
        "num_nodes": len(ir["nodes"]),
        "num_inputs": len(ir["inputs"]),
        "num_weights": len(ir["weights"]),
        "num_outputs": len(ir["outputs"]),
        "op_histogram": _op_histogram(ir["nodes"]),
    }
    return summary


def _op_histogram(nodes: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for n in nodes:
        hist[n["op"]] = hist.get(n["op"], 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))


# ── locating the Rust binary ─────────────────────────────────────────────────
def _find_rlx_bin() -> Optional[list[str]]:
    # NB: we deliberately do NOT `shutil.which("rlx-torch-import")` — that is the
    # name of *this* Python console-script, so it would recurse. The Rust worker
    # is located via an explicit env var or the dev cargo fallback.
    env = os.environ.get("RLX_TORCH_IMPORT_BIN")
    if env:
        return [env]
    # Dev fallback: run through cargo from the workspace this package lives in.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").exists() and (parent / "crates").exists():
            return [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(parent / "Cargo.toml"),
                "-p",
                "rlx-torch-import",
                "--",
            ]
    return None


# ── public: full tool ────────────────────────────────────────────────────────
def from_torch(
    model: Any,
    example_inputs: Sequence[Any],
    out_dir: Union[str, Path],
    *,
    model_name: Optional[str] = None,
    emit: Sequence[str] = ("bundle", "crate"),
    emit_style: str = "graph",
    verify: bool = True,
    decomposition: str = "high",
    run_rlx: Union[bool, str] = "auto",
    strict: bool = False,
    auto_decompose: bool = True,
    max_decompose_rounds: int = 4,
    dynamic_shapes: Any = None,
) -> dict:
    """Convert a live PyTorch module to an RLX model — the full tool.

    Runs the front-end (:func:`export_torch_ir`), then invokes the Rust
    ``rlx-torch-import`` binary to map ATen → RLX and emit the requested
    artifacts (``bundle`` and/or ``crate``) under ``out_dir``, verifying numeric
    parity against PyTorch when ``verify``.

    ``decomposition`` (``aten`` | ``high`` | ``core``) selects how much the graph
    is lowered — see :func:`export_torch_ir`.

    ``run_rlx``: ``True`` requires the binary, ``False`` skips it (front-end
    only), ``"auto"`` runs it if found and otherwise warns.

    ``auto_decompose`` (default): if the Rust importer reports ops the RLX registry
    doesn't cover, re-export with those ops **decomposed** (via torch's full
    decomposition registry) and retry, up to ``max_decompose_rounds`` — so an
    unsupported *high-level* op is automatically broken into primitives RLX does
    cover, shrinking the "unsupported op" tail without hand-adding a lowering. The
    ops that were decomposed are reported in ``summary["auto_decomposed_ops"]``.
    """
    import torch  # for the auto-decompose helpers

    want_rlx = run_rlx if isinstance(run_rlx, bool) else True
    bin_cmd = _find_rlx_bin()

    def _export(force):
        return export_torch_ir(
            model,
            example_inputs,
            out_dir,
            model_name=model_name,
            decomposition=decomposition,
            write_reference=verify,
            strict=strict,
            force_decompose=(force or None),
            dynamic_shapes=dynamic_shapes,
        )

    # Front-end-only cases: export once, skip the Rust build.
    if not want_rlx or (run_rlx == "auto" and bin_cmd is None):
        summary = _export(set())
        if run_rlx == "auto" and bin_cmd is None:
            print(
                "[from_torch] rlx-torch-import binary not found; wrote front-end IR "
                "only. Build it (cargo build -p rlx-torch-import) or set "
                "RLX_TORCH_IMPORT_BIN to emit the bundle/crate.",
                file=sys.stderr,
            )
        summary["rlx_ran"] = False
        return summary
    if bin_cmd is None:
        raise RuntimeError(
            "rlx-torch-import binary not found (set RLX_TORCH_IMPORT_BIN or build "
            "the crate)."
        )

    cmd = bin_cmd + [
        "build",
        str(Path(out_dir)),
        "--emit",
        ",".join(emit),
        "--emit-style",
        emit_style,
    ]
    if verify:
        cmd.append("--verify")

    force_decompose: set = set()
    rounds = max_decompose_rounds if auto_decompose else 0
    proc = None
    for attempt in range(rounds + 1):
        summary = _export(force_decompose)
        print(f"[from_torch] $ {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary["rlx_ran"] = True
        summary["rlx_stdout"] = proc.stdout
        summary["rlx_stderr"] = proc.stderr
        summary["rlx_returncode"] = proc.returncode
        summary["decompose_rounds"] = attempt
        if force_decompose:
            summary["auto_decomposed_ops"] = sorted(force_decompose)
        if proc.returncode == 0:
            result_path = Path(out_dir) / "rlx-import-result.json"
            if result_path.exists():
                summary["rlx_result"] = json.loads(result_path.read_text())
            return summary
        # Failure — try to decompose the offending ops and retry, skipping any
        # whose decomposition breaks torch.export (prims/functionalization).
        unsupported = _parse_unsupported_ops(proc.stdout + proc.stderr)
        newly = _decomposable_ops(torch, unsupported - force_decompose)
        if not newly:
            break  # nothing left we can decompose — a genuine coverage gap
        accepted = _accept_decompositions(_export, force_decompose, newly)
        if accepted == force_decompose:
            break  # none could be cleanly decomposed
        print(
            f"[from_torch] decomposing {sorted(accepted - force_decompose)} and "
            f"retrying (round {attempt + 1})…",
            file=sys.stderr,
        )
        force_decompose = accepted

    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(f"rlx-torch-import failed (exit {proc.returncode})")


# ── CLI: rlx-torch-import ────────────────────────────────────────────────────
def _load_user_model(path: str):
    """Import a user .py and pull out (model, example_inputs).

    The file must expose either module-level ``model`` + ``example_inputs``, or
    a ``get_model()`` / ``build()`` returning ``(model, example_inputs)``.
    """
    import importlib.util

    path_p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path_p.stem, str(path_p))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for factory in ("get_model", "build"):
        fn = getattr(mod, factory, None)
        if callable(fn):
            model, example_inputs = fn()
            return model, example_inputs
    model = getattr(mod, "model", None)
    example_inputs = getattr(mod, "example_inputs", None)
    if model is None or example_inputs is None:
        raise AttributeError(
            f"{path}: expected module-level `model` + `example_inputs`, or a "
            "`get_model()`/`build()` returning (model, example_inputs)."
        )
    return model, example_inputs


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="rlx-torch-import",
        description="Convert a PyTorch nn.Module to an RLX model (bundle + crate).",
    )
    p.add_argument("model_py", help="Python file exposing `model` + `example_inputs`.")
    p.add_argument("-o", "--out", required=True, help="Output directory.")
    p.add_argument("--name", default=None, help="Model name (default: class name).")
    p.add_argument(
        "--emit",
        default="bundle,crate",
        help="Comma list of artifacts: bundle,crate (default: both).",
    )
    p.add_argument(
        "--emit-style",
        choices=["graph", "tensor", "flow"],
        default="graph",
        help="Authoring layer the generated crate targets: graph (HIR builder, "
        "default), tensor (PyTorch-like Tensor DSL), flow (ModelFlow blocks).",
    )
    p.add_argument("--no-verify", action="store_true", help="Skip parity check.")
    p.add_argument(
        "--decomposition",
        choices=["aten", "high", "core"],
        default="high",
        help="How much to lower the graph: aten (raw/highest), high (preserve "
        "high-level ops, default), core (most primitive).",
    )
    p.add_argument(
        "--front-end-only",
        action="store_true",
        help="Only emit torch-ir.json + weights (skip the Rust build).",
    )
    args = p.parse_args(argv)

    model, example_inputs = _load_user_model(args.model_py)
    summary = from_torch(
        model,
        example_inputs,
        args.out,
        model_name=args.name,
        emit=tuple(s.strip() for s in args.emit.split(",") if s.strip()),
        emit_style=args.emit_style,
        verify=not args.no_verify,
        decomposition=args.decomposition,
        run_rlx=False if args.front_end_only else "auto",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
