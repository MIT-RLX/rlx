#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Inspect rlx's JIT'd GPU kernels: turn each compiled translation unit into
disassembly / occupancy / register reports on the target GPU. Two backends,
one tool — the `.cu` kernel sources are shared (rlx-gpu-kernels):
  * CUDA  — NVRTC → PTX → (ptxas) SASS, registers / smem / spills
  * ROCm  — hipRTC → code object → (llvm-objdump) GCN ISA, VGPR / SGPR / LDS

One canonical command builds the backend, exercises the kernels, and reports:

    kinspect.py run                 # auto-detect backend; build + dump + analyze
    kinspect.py run --target rocm   # force ROCm (on the AMD rig)

Or split the two phases (the dump comes from running rlx-cuda with
`RLX_DUMP_KERNELS=<dir>` set — `run` just wires that up):

    <dir>/cu/<entry>.cu        exact source NVRTC compiled (post gelu/codegen assembly)
    <dir>/ptx/<entry>.ptx      rlx's own compiled PTX
    <dir>/manifest.jsonl       {entry, src_hash, src_bytes, ptx_bytes} per kernel

For each translation unit (deduped by src_hash) we:
  * ptxas -arch=sm_XX -v      -> registers / stack / spills / smem / cmem per fn
  * cuobjdump -sass           -> real SASS -> opcode histogram + capability flags

and emit report.json (machine-readable, diffable), report.md (human), and
per-kernel <out>/sass/<entry>.sass.

    kinspect.py analyze <dump_dir> [-o out] [--arch sm_86] [--cuda /usr/local/cuda]
    kinspect.py diff old/report.json new/report.json    # before/after a kernel edit

Pure stdlib; runs anywhere cargo + ptxas + cuobjdump are on PATH (i.e. the
Linux CUDA rig). `analyze` / `diff` need no GPU — only the CUDA binutils.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# --- SASS opcode taxonomy: which mnemonics signal what capability ----------
# Matched against the *base* opcode (mnemonic up to the first '.').
FLAG_OPS = {
    "tensor_core": {"HMMA", "IMMA", "BMMA", "DMMA", "OMMA", "HGMMA", "IGMMA"},
    "fp64": {"DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DADD32I"},
    "fp16x2": {"HADD2", "HMUL2", "HFMA2", "HSETP2", "HMNMX2"},
    "global_mem": {"LDG", "STG", "LDGSTS", "RED", "ATOM", "ATOMG", "MEMBAR"},
    "shared_mem": {"LDS", "STS", "LDSM", "ATOMS"},
    "local_mem": {"LDL", "STL"},  # register spills / stack traffic
    "barrier": {"BAR", "BARWARP", "DEPBAR"},
    "transcendental": {"MUFU"},  # sin/cos/rcp/rsqrt/ex2/lg2
    "branch": {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "BSSY", "BSYNC", "BREAK"},
}
# Base opcodes we treat as "math throughput" for a rough arithmetic-intensity feel.
MATH_OPS = {"FFMA", "FADD", "FMUL", "FMNMX", "IMAD", "IADD3", "IMUL", "HFMA2",
            "FSETP", "MUFU", "DFMA", "DADD", "DMUL", "HMMA", "IMMA"}

# --- SASS instruction classes: bucket every base opcode into a datapath so the
# report shows *where the instructions go* (tensor vs fp32 vs memory vs the
# Ampere uniform datapath), not just a top-N list. Unknowns fall to "misc".
SASS_CLASS = {
    "tensor": {"HMMA", "IMMA", "BMMA", "DMMA", "OMMA", "HGMMA", "IGMMA", "GMMA"},
    "fp32": {"FADD", "FMUL", "FFMA", "FSET", "FSETP", "FMNMX", "FCHK", "FSEL",
             "FSWZADD", "MUFU", "FADD32I", "FMUL32I", "FFMA32I"},
    "fp16": {"HADD2", "HMUL2", "HFMA2", "HSET2", "HSETP2", "HMNMX2", "HADD",
             "HFMA", "HMUL"},
    "fp64": {"DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DADD32I"},
    "int": {"IADD3", "IADD", "IADD32I", "IMAD", "IMUL", "IMNMX", "ISETP", "LOP3",
            "LOP", "LOP32I", "SHF", "SHL", "SHR", "LEA", "BMSK", "POPC", "FLO",
            "BREV", "IABS", "ICMP", "VABSDIFF", "DP4A", "DP2A", "IDP", "BFE",
            "BFI", "PRMT"},
    "mem_global": {"LDG", "STG", "LDGSTS", "RED", "ATOM", "ATOMG", "MEMBAR",
                   "CCTL", "ERRBAR", "LD", "ST", "MATCH"},
    "mem_shared": {"LDS", "STS", "LDSM", "ATOMS", "SHFL"},
    "mem_local": {"LDL", "STL"},           # register spills / stack traffic
    "mem_const": {"LDC"},
    "movement": {"MOV", "MOV32I", "SEL", "CS2R", "S2R", "R2P", "P2R", "I2I",
                 "F2F", "I2F", "F2I", "F2FP", "I2FP", "I2IP", "FRND", "SGXT",
                 "PLOP3", "PSETP", "CSET", "CSETP", "P2UR", "R2B", "B2R"},
    "control": {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY",
                "BSYNC", "BREAK", "BPT", "WARPSYNC", "BAR", "DEPBAR", "NOP",
                "YIELD", "KILL", "CONT", "PBK", "SSY", "BRK", "RTT", "VOTE",
                "RPCMOV"},
}
_OP2CLASS = {op: cls for cls, ops in SASS_CLASS.items() for op in ops}


def sass_class(op: str) -> str:
    c = _OP2CLASS.get(op)
    if c:
        return c
    # Ampere uniform datapath: ULDC/UMOV/UIADD3/UISETP/ULEA/USHF/UFLO/UPRMT/…
    if op.startswith("U") or op in ("VOTEU", "R2UR", "S2UR"):
        return "uniform"
    return "misc"

# --- Ampere GA10x / compute 8.6 hardware limits (for occupancy bounds) ------
SM_LIMITS = {
    "sm_86": dict(regs_per_sm=65536, max_regs_per_thread=255, max_warps=48,
                  warp_alloc_gran=256, smem_per_sm=102400, max_blocks=16),
    "sm_89": dict(regs_per_sm=65536, max_regs_per_thread=255, max_warps=48,
                  warp_alloc_gran=256, smem_per_sm=102400, max_blocks=24),
    "sm_80": dict(regs_per_sm=65536, max_regs_per_thread=255, max_warps=64,
                  warp_alloc_gran=256, smem_per_sm=167936, max_blocks=32),
    "sm_90": dict(regs_per_sm=65536, max_regs_per_thread=255, max_warps=64,
                  warp_alloc_gran=256, smem_per_sm=233472, max_blocks=32),
}

INSTR_RE = re.compile(r"/\*[0-9a-fA-F]+\*/\s*(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")
FUNC_RE = re.compile(r"\bFunction\s*:\s*(\S+)")

# --- AMDGPU (ROCm) side ----------------------------------------------------
# Per-SIMD occupancy limits. `max_waves` from rocminfo (Max Waves Per CU /
# SIMDs per CU). vgpr_pool/gran are the register-file size + allocation
# granularity; SGPR rarely binds. gfx908 (MI100) is exact — the rig's GPU;
# the RDNA3 rows are best-effort (occupancy there is a rougher estimate).
GFX_LIMITS = {
    "gfx908": dict(vgpr_pool=256, vgpr_gran=4, sgpr_pool=800, sgpr_gran=16, max_waves=10),   # MI100 CDNA1
    "gfx90a": dict(vgpr_pool=512, vgpr_gran=8, sgpr_pool=800, sgpr_gran=16, max_waves=8),     # MI200 CDNA2
    "gfx942": dict(vgpr_pool=512, vgpr_gran=8, sgpr_pool=800, sgpr_gran=16, max_waves=8),     # MI300 CDNA3
    "gfx1100": dict(vgpr_pool=768, vgpr_gran=16, sgpr_pool=800, sgpr_gran=16, max_waves=16),  # RDNA3 (est.)
    "gfx1103": dict(vgpr_pool=768, vgpr_gran=16, sgpr_pool=800, sgpr_gran=16, max_waves=16),  # 780M (est.)
}

# GCN/CDNA/RDNA mnemonics are prefix-structured, so classify by prefix.
AMD_ISA_FUNC_RE = re.compile(r"^[0-9a-fA-F]+ <([A-Za-z_]\w*)>:")
AMD_INSTR_RE = re.compile(r"^\s+([a-z][a-z0-9_]+)\b")
AMD_META_KEYS = {
    ".group_segment_fixed_size", ".private_segment_fixed_size", ".sgpr_count",
    ".sgpr_spill_count", ".vgpr_count", ".vgpr_spill_count", ".name",
    ".max_flat_workgroup_size", ".wavefront_size",
}


def find_tool(name: str, cuda_home: str | None) -> str:
    if cuda_home:
        cand = Path(cuda_home) / "bin" / name
        if cand.exists():
            return str(cand)
    p = shutil.which(name)
    if p:
        return p
    for base in ("/usr/local/cuda", "/opt/cuda"):
        cand = Path(base) / "bin" / name
        if cand.exists():
            return str(cand)
    sys.exit(f"error: could not find `{name}` (pass --cuda <CUDA_HOME> or add to PATH)")


def detect_arch() -> str:
    """Query the live GPU's compute capability -> sm_XY (fallback sm_86)."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip().splitlines()
            if out:
                cc = out[0].strip().replace(".", "")
                if cc.isdigit():
                    return f"sm_{cc}"
        except Exception:
            pass
    return "sm_86"


# --- ptxas -v parsing -------------------------------------------------------
def parse_ptxas(stderr: str) -> dict[str, dict]:
    """Map function-name -> {registers, stack, spill_stores, spill_loads,
    smem, cmem} from `ptxas -v` info lines."""
    fns: dict[str, dict] = {}
    cur = None
    for line in stderr.splitlines():
        m = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        if m:
            cur = m.group(1)
            fns.setdefault(cur, dict(registers=0, stack=0, spill_stores=0,
                                     spill_loads=0, smem=0, cmem=0))
            continue
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            cur = m.group(1)
            fns.setdefault(cur, dict(registers=0, stack=0, spill_stores=0,
                                     spill_loads=0, smem=0, cmem=0))
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                      r"(\d+) bytes spill loads", line)
        if m:
            fns[cur]["stack"] = int(m.group(1))
            fns[cur]["spill_stores"] = int(m.group(2))
            fns[cur]["spill_loads"] = int(m.group(3))
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m:
            fns[cur]["registers"] = int(m.group(1))
            sm = re.search(r"(\d+) bytes smem", line)
            if sm:
                fns[cur]["smem"] = int(sm.group(1))
            cm = re.search(r"(\d+) bytes cmem\[0\]", line)
            if cm:
                fns[cur]["cmem"] = int(cm.group(1))
    return fns


# --- SASS parsing -----------------------------------------------------------
def parse_sass(text: str) -> dict[str, dict]:
    """Map function-name -> {instr_count, opcodes(Counter), lines[list]}."""
    fns: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        fm = FUNC_RE.search(line)
        if fm:
            name = fm.group(1)
            if name.startswith(".text."):
                name = name[len(".text."):]
            cur = name
            fns.setdefault(cur, dict(instr=0, opcodes=Counter(), sass=[]))
            continue
        if cur is None:
            continue
        im = INSTR_RE.search(line)
        if im:
            base = im.group(1).split(".")[0]
            fns[cur]["opcodes"][base] += 1
            fns[cur]["instr"] += 1
            fns[cur]["sass"].append(line.rstrip())
    return fns


def caps_from_opcodes(opcodes: Counter) -> dict:
    seen = set(opcodes)
    flags = {name: bool(seen & ops) for name, ops in FLAG_OPS.items()}
    total = sum(opcodes.values()) or 1
    branch = sum(opcodes[o] for o in FLAG_OPS["branch"])
    gmem = sum(opcodes[o] for o in FLAG_OPS["global_mem"])
    smem = sum(opcodes[o] for o in FLAG_OPS["shared_mem"])
    math = sum(opcodes[o] for o in MATH_OPS)
    # Datapath breakdown: bucket every instruction into a class + percentages.
    classes: Counter = Counter()
    for op, c in opcodes.items():
        classes[sass_class(op)] += c
    class_pct = {cls: round(100 * n / total, 1) for cls, n in classes.most_common()}
    tensor_ops = sum(opcodes[o] for o in SASS_CLASS["tensor"])
    async_ops = opcodes.get("LDGSTS", 0)  # cp.async — software-pipelined loads
    uniform_ops = sum(c for op, c in opcodes.items() if sass_class(op) == "uniform")
    flags.update(
        branch_pct=round(100 * branch / total, 1),
        gmem_ops=gmem,
        smem_ops=smem,
        math_ops=math,
        math_pct=round(100 * math / total, 1),
        tensor_ops=tensor_ops,
        async_ops=async_ops,
        uniform_ops=uniform_ops,
        class_counts=dict(classes),
        class_pct=class_pct,
    )
    return flags


def occupancy(regs: int, smem: int, arch: str, block: int) -> dict:
    """Theoretical occupancy for a given block size, matching the CUDA
    occupancy calculator's register/smem/block limits."""
    lim = SM_LIMITS.get(arch, SM_LIMITS["sm_86"])
    warps_per_block = max(1, (block + 31) // 32)
    # Register limit: regs allocated per-warp, rounded to warp_alloc_gran.
    if regs > 0:
        regs_per_warp = ((regs * 32 + lim["warp_alloc_gran"] - 1)
                         // lim["warp_alloc_gran"]) * lim["warp_alloc_gran"]
        warps_by_reg = lim["regs_per_sm"] // max(1, regs_per_warp)
    else:
        warps_by_reg = lim["max_warps"]
    blocks_by_reg = warps_by_reg // warps_per_block
    # Shared-memory limit.
    blocks_by_smem = (lim["smem_per_sm"] // smem) if smem > 0 else lim["max_blocks"]
    # Warp / block ceilings.
    blocks_by_warp = lim["max_warps"] // warps_per_block
    blocks = min(blocks_by_reg, blocks_by_smem, blocks_by_warp, lim["max_blocks"])
    active_warps = blocks * warps_per_block
    return dict(block=block, blocks_per_sm=blocks, active_warps=active_warps,
                occupancy_pct=round(100 * active_warps / lim["max_warps"], 1),
                limiter=_limiter(blocks_by_reg, blocks_by_smem, blocks_by_warp,
                                 lim["max_blocks"]))


def _limiter(reg, smem, warp, block_cap) -> str:
    m = min(reg, smem, warp, block_cap)
    if m == reg:
        return "registers"
    if m == smem:
        return "shared_mem"
    if m == warp:
        return "warps"
    return "blocks"


# --- AMDGPU parsing (llvm-readobj metadata + llvm-objdump ISA) --------------
def parse_amd_metadata(text: str) -> dict[str, dict]:
    """kernel-name -> {vgpr, sgpr, vgpr_spill, sgpr_spill, lds, scratch} from
    the NT_AMDGPU_METADATA note (llvm-readobj --notes prints it as YAML). Keys
    within a kernel appear once, so a repeated key marks the next kernel."""
    recs: dict[str, dict] = {}
    cur: dict[str, str] = {}

    def flush(d):
        if ".name" in d:
            recs[d[".name"]] = dict(
                vgpr=int(d.get(".vgpr_count", 0)),
                sgpr=int(d.get(".sgpr_count", 0)),
                vgpr_spill=int(d.get(".vgpr_spill_count", 0)),
                sgpr_spill=int(d.get(".sgpr_spill_count", 0)),
                lds=int(d.get(".group_segment_fixed_size", 0)),
                scratch=int(d.get(".private_segment_fixed_size", 0)),
            )

    for line in text.splitlines():
        m = re.match(r"^\s*(\.[a-z_]+):\s*(.+?)\s*$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k not in AMD_META_KEYS:
            continue
        if k in cur:
            flush(cur)
            cur = {}
        cur[k] = v
    flush(cur)
    return recs


def parse_amd_isa(text: str) -> dict[str, dict]:
    """kernel-name -> {instr, opcodes(Counter), isa[list]} from `llvm-objdump
    -d` AMDGCN disassembly."""
    fns: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        fm = AMD_ISA_FUNC_RE.match(line)
        if fm:
            cur = fm.group(1)
            fns.setdefault(cur, dict(instr=0, opcodes=Counter(), isa=[]))
            continue
        if cur is None or "//" not in line:  # real instr lines carry a `// addr: enc` comment
            continue
        im = AMD_INSTR_RE.match(line)
        if im:
            op = im.group(1)
            fns[cur]["opcodes"][op] += 1
            fns[cur]["instr"] += 1
            fns[cur]["isa"].append(line.rstrip())
    return fns


def caps_from_amd_opcodes(opcodes: Counter) -> dict:
    total = sum(opcodes.values()) or 1

    def cnt(pred):
        return sum(c for op, c in opcodes.items() if pred(op))

    matrix = cnt(lambda o: o.startswith(("v_mfma", "v_smfmac", "v_wmma")))
    fp64 = cnt(lambda o: "_f64" in o)
    fp16 = cnt(lambda o: "_f16" in o)
    lds = cnt(lambda o: o.startswith("ds_"))
    scratch = cnt(lambda o: o.startswith("scratch_"))  # register-spill traffic
    gmem = cnt(lambda o: o.startswith(("global_", "buffer_", "flat_")))
    trans = cnt(lambda o: o.startswith(("v_rcp", "v_rsq", "v_exp", "v_log",
                                        "v_sin", "v_cos", "v_sqrt")))
    branch = cnt(lambda o: o.startswith(("s_branch", "s_cbranch", "s_setpc",
                                         "s_swappc", "s_call")))
    math = cnt(lambda o: o.startswith(("v_fma", "v_mac", "v_mad", "v_mul_f",
                                       "v_add_f", "v_dot", "v_mfma", "v_pk_fma")))
    return dict(matrix_core=bool(matrix), fp64=bool(fp64), fp16=bool(fp16),
                scratch_ops=scratch, transcendental=bool(trans),
                gmem_ops=gmem, lds_ops=lds, math_ops=math,
                branch_pct=round(100 * branch / total, 1),
                math_pct=round(100 * math / total, 1))


def occupancy_rocm(vgpr: int, sgpr: int, arch: str) -> dict:
    """VGPR/SGPR-limited waves per SIMD (theoretical). LDS/scratch are reported
    raw — they bound per-workgroup occupancy, which needs the launch geometry."""
    lim = GFX_LIMITS.get(arch)
    if not lim:
        return dict(waves_per_simd=None, max_waves=None, occupancy_pct=None, limiter="unknown")

    def waves(pool, gran, used):
        if used <= 0:
            return lim["max_waves"]
        return pool // (((used + gran - 1) // gran) * gran)

    w_v = waves(lim["vgpr_pool"], lim["vgpr_gran"], vgpr)
    w_s = waves(lim["sgpr_pool"], lim["sgpr_gran"], sgpr)
    cand = [(w_v, "vgpr"), (w_s, "sgpr"), (lim["max_waves"], "wave-cap")]
    w, limiter = min(cand, key=lambda t: t[0])
    return dict(waves_per_simd=w, max_waves=lim["max_waves"],
                occupancy_pct=round(100 * w / lim["max_waves"], 1), limiter=limiter)


def load_manifest(dump: Path) -> list[dict]:
    mf = dump / "manifest.jsonl"
    if not mf.exists():
        # Fall back to scanning ptx/ if no manifest.
        return [{"entry": p.stem, "src_hash": p.stem} for p in
                sorted((dump / "ptx").glob("*.ptx"))]
    rows = []
    for line in mf.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def find_amd_tool(name: str) -> str:
    for base in ("/opt/rocm/llvm/bin", "/opt/rocm/bin"):
        cand = Path(base) / name
        if cand.exists():
            return str(cand)
    p = shutil.which(name)
    if p:
        return p
    sys.exit(f"error: could not find `{name}` (ROCm LLVM tools — add /opt/rocm/llvm/bin to PATH)")


def _dedup_tus(rows: list[dict]):
    """One representative entry per translation unit (by src_hash); a TU's
    compiled artifact carries every kernel it defines."""
    by_hash: dict[str, dict] = {}
    entries_by_hash: dict[str, list[str]] = {}
    for r in rows:
        h = r.get("src_hash", r["entry"])
        entries_by_hash.setdefault(h, [])
        if r["entry"] not in entries_by_hash[h]:
            entries_by_hash[h].append(r["entry"])
        by_hash.setdefault(h, r)
    return by_hash, entries_by_hash


def _finish(out: Path, report: dict, subdir: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=list))
    write_markdown(out / "report.md", report)
    print(f"[inspect] wrote {out / 'report.json'}")
    print(f"[inspect] wrote {out / 'report.md'}")
    print(f"[inspect] per-kernel disasm in {out / subdir}/")
    if report["errors"]:
        print(f"[inspect] {len(report['errors'])} kernel(s) had errors (see report.json)")
    _print_headline(report)
    return 0


def run_analyze(args) -> int:
    dump = Path(args.dump_dir)
    is_rocm = (dump / "codeobj").is_dir()
    is_cuda = (dump / "ptx").is_dir()
    if not (is_rocm or is_cuda):
        sys.exit(f"error: neither {dump}/ptx (CUDA) nor {dump}/codeobj (ROCm) found — "
                 f"run `kinspect.py run` (or set RLX_DUMP_KERNELS) first")
    out = Path(args.out or (dump / "report"))
    rows = load_manifest(dump)
    by_hash, entries_by_hash = _dedup_tus(rows)
    if is_rocm:
        return analyze_rocm(args, dump, out, rows, by_hash, entries_by_hash)
    return analyze_cuda(args, dump, out, rows, by_hash, entries_by_hash)


def analyze_cuda(args, dump, out, rows, by_hash, entries_by_hash) -> int:
    (out / "sass").mkdir(parents=True, exist_ok=True)
    arch = args.arch or detect_arch()
    ptxas = find_tool("ptxas", args.cuda)
    cuobjdump = find_tool("cuobjdump", args.cuda)
    print(f"[inspect] target=cuda  arch={arch}  ptxas={ptxas}")
    print(f"[inspect] {len(rows)} kernel entries -> {len(by_hash)} translation units")

    kernels: dict[str, dict] = {}
    errors: list[str] = []
    for h, r in sorted(by_hash.items(), key=lambda kv: kv[1]["entry"]):
        rep_entry = r["entry"]
        ptx_path = dump / "ptx" / f"{rep_entry}.ptx"
        if not ptx_path.exists():
            errors.append(f"{rep_entry}: missing ptx")
            continue
        with tempfile.TemporaryDirectory() as td:
            cubin = Path(td) / "k.cubin"
            pv = subprocess.run(
                [ptxas, f"-arch={arch}", "-v", "-o", str(cubin), str(ptx_path)],
                capture_output=True, text=True)
            if pv.returncode != 0:
                errors.append(f"{rep_entry}: ptxas failed: {pv.stderr.strip().splitlines()[-1:] }")
                continue
            fn_stats = parse_ptxas(pv.stderr)
            sass = subprocess.run([cuobjdump, "-sass", str(cubin)],
                                  capture_output=True, text=True)
            fn_sass = parse_sass(sass.stdout) if sass.returncode == 0 else {}

        # Merge per-function ptxas + SASS. Keys should match (extern "C" names).
        names = set(fn_stats) | set(fn_sass)
        for name in sorted(names):
            st = fn_stats.get(name, dict(registers=0, stack=0, spill_stores=0,
                                         spill_loads=0, smem=0, cmem=0))
            sa = fn_sass.get(name, dict(instr=0, opcodes=Counter(), sass=[]))
            caps = caps_from_opcodes(sa["opcodes"])
            occ = occupancy(st["registers"], st["smem"], arch, 256)
            top = sa["opcodes"].most_common(12)
            kernels[name] = dict(
                tu=rep_entry, src_hash=h,
                shared_by=entries_by_hash[h],
                registers=st["registers"], stack=st["stack"],
                spill_stores=st["spill_stores"], spill_loads=st["spill_loads"],
                smem=st["smem"], cmem=st["cmem"],
                instr=sa["instr"], top_opcodes=top, **caps,
                occ256=occ,
            )
            if sa["sass"]:
                (out / "sass" / f"{name}.sass").write_text("\n".join(sa["sass"]) + "\n")

    report = dict(target="cuda", arch=arch, tools=dict(ptxas=ptxas, cuobjdump=cuobjdump),
                  n_entries=len(rows), n_tus=len(by_hash),
                  n_kernels=len(kernels), errors=errors, kernels=kernels)
    return _finish(out, report, "sass")


def analyze_rocm(args, dump, out, rows, by_hash, entries_by_hash) -> int:
    (out / "isa").mkdir(parents=True, exist_ok=True)
    readobj = find_amd_tool("llvm-readobj")
    objdump = find_amd_tool("llvm-objdump")
    print(f"[inspect] target=rocm  objdump={objdump}")
    print(f"[inspect] {len(rows)} kernel entries -> {len(by_hash)} translation units")

    kernels: dict[str, dict] = {}
    errors: list[str] = []
    archs: set[str] = set()
    for h, r in sorted(by_hash.items(), key=lambda kv: kv[1]["entry"]):
        rep_entry = r["entry"]
        arch = r.get("arch", "")
        archs.add(arch)
        co = dump / "codeobj" / f"{rep_entry}.hsaco"
        if not co.exists():
            errors.append(f"{rep_entry}: missing codeobj")
            continue
        meta = parse_amd_metadata(subprocess.run(
            [readobj, "--elf-output-style=GNU", "--notes", str(co)],
            capture_output=True, text=True).stdout)
        isa = parse_amd_isa(subprocess.run(
            [objdump, "-d", str(co)], capture_output=True, text=True).stdout)

        names = set(meta) | set(isa)
        for name in sorted(names):
            st = meta.get(name, dict(vgpr=0, sgpr=0, vgpr_spill=0, sgpr_spill=0, lds=0, scratch=0))
            sa = isa.get(name, dict(instr=0, opcodes=Counter(), isa=[]))
            caps = caps_from_amd_opcodes(sa["opcodes"])
            occ = occupancy_rocm(st["vgpr"], st["sgpr"], arch)
            kernels[name] = dict(
                tu=rep_entry, src_hash=h, shared_by=entries_by_hash[h], arch=arch,
                vgpr=st["vgpr"], sgpr=st["sgpr"],
                vgpr_spill=st["vgpr_spill"], sgpr_spill=st["sgpr_spill"],
                lds=st["lds"], scratch=st["scratch"],
                instr=sa["instr"], top_opcodes=sa["opcodes"].most_common(12), **caps,
                occ=occ,
            )
            if sa["isa"]:
                (out / "isa" / f"{name}.isa").write_text("\n".join(sa["isa"]) + "\n")

    report = dict(target="rocm", arch=",".join(sorted(a for a in archs if a)) or "unknown",
                  tools=dict(llvm_objdump=objdump, llvm_readobj=readobj),
                  n_entries=len(rows), n_tus=len(by_hash),
                  n_kernels=len(kernels), errors=errors, kernels=kernels)
    return _finish(out, report, "isa")


def _flagstr(k: dict, target: str = "cuda") -> str:
    tags = []
    if target == "rocm":
        if k.get("vgpr_spill") or k.get("sgpr_spill") or k.get("scratch"):
            tags.append("SPILL")
        if k.get("matrix_core"):
            tags.append("MFMA")
        if k.get("fp64"):
            tags.append("f64")
        if k.get("fp16"):
            tags.append("f16")
        if k.get("transcendental"):
            tags.append("trans")
        return ",".join(tags)
    if k["spill_stores"] or k["spill_loads"]:
        tags.append("SPILL")
    if k.get("local_mem"):
        tags.append("local")
    if k.get("tensor_core"):
        tags.append(f"TC×{k.get('tensor_ops', 0)}")
    if k.get("async_ops"):
        tags.append(f"async×{k['async_ops']}")  # cp.async / LDGSTS pipelining
    if k.get("fp64"):
        tags.append("f64")
    if k.get("fp16x2"):
        tags.append("f16x2")
    if k.get("transcendental"):
        tags.append("mufu")
    return ",".join(tags)


def _print_headline(report: dict):
    ks = report["kernels"]
    print("\n[inspect] === headline ===")
    if report.get("target") == "rocm":
        spill = [n for n, k in ks.items()
                 if k.get("vgpr_spill") or k.get("sgpr_spill") or k.get("scratch")]
        hot = sorted(ks.items(), key=lambda kv: -kv[1]["vgpr"])[:5]
        print(f"[inspect] kernels analyzed: {len(ks)}  spilling: {len(spill)}")
        if spill:
            print(f"[inspect] ⚠ spills/scratch: {', '.join(sorted(spill))}")
        print("[inspect] highest VGPR pressure:")
        for n, k in hot:
            occ = k["occ"]
            print(f"[inspect]   {n:<30} {k['vgpr']:>3} vgpr {k['sgpr']:>3} sgpr  "
                  f"occ={occ.get('occupancy_pct')}% ({occ.get('limiter')}-limited)  "
                  f"waves/SIMD={occ.get('waves_per_simd')}")
        return
    spill = [n for n, k in ks.items() if k["spill_stores"] or k["spill_loads"]]
    hot = sorted(ks.items(), key=lambda kv: -kv[1]["registers"])[:5]
    print(f"[inspect] kernels analyzed: {len(ks)}  spilling: {len(spill)}")
    if spill:
        print(f"[inspect] ⚠ spills: {', '.join(sorted(spill))}")
    print("[inspect] highest register pressure:")
    for n, k in hot:
        print(f"[inspect]   {n:<32} {k['registers']:>3} regs  occ@256={k['occ256']['occupancy_pct']}%"
              f"  ({k['occ256']['limiter']}-limited)")


def write_markdown(path: Path, report: dict):
    lines = _md_rocm(report) if report.get("target") == "rocm" else _md_cuda(report)
    path.write_text("\n".join(lines) + "\n")


def _md_cuda(report: dict) -> list[str]:
    ks = report["kernels"]
    lines = [
        "# rlx-cuda kernel inspection report",
        "",
        f"- target: `cuda`  arch: `{report['arch']}`",
        f"- entries: {report['n_entries']}  translation units: {report['n_tus']}  "
        f"kernels: {report['n_kernels']}",
        f"- ptxas: `{report['tools']['ptxas']}`",
        "",
        "Occupancy is a theoretical upper bound at block=256, from register/smem "
        "pressure (real launch block sizes live in the Rust launch config).",
        "",
        "## Summary (sorted by register pressure)",
        "",
        "| kernel | regs | stack | spillST | spillLD | smem | cmem0 | instr | occ@256 | limiter | flags |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|",
    ]
    for n, k in sorted(ks.items(), key=lambda kv: -kv[1]["registers"]):
        lines.append(
            f"| `{n}` | {k['registers']} | {k['stack']} | {k['spill_stores']} | "
            f"{k['spill_loads']} | {k['smem']} | {k['cmem']} | {k['instr']} | "
            f"{k['occ256']['occupancy_pct']}% | {k['occ256']['limiter']} | {_flagstr(k)} |")

    spill = {n: k for n, k in ks.items() if k["spill_stores"] or k["spill_loads"]}
    if spill:
        lines += ["", "## ⚠ Register spills (perf risk — consider tiling/`__launch_bounds__`)", ""]
        for n, k in sorted(spill.items(), key=lambda kv: -(kv[1]["spill_stores"] + kv[1]["spill_loads"])):
            lines.append(f"- `{n}`: {k['registers']} regs, "
                         f"{k['spill_stores']}B spill stores / {k['spill_loads']}B loads "
                         f"(stack {k['stack']}B)")

    tc = [n for n, k in ks.items() if k.get("tensor_core")]
    if tc:
        lines += ["", "## Tensor-core kernels (HMMA/IMMA present)", "", ", ".join(f"`{n}`" for n in sorted(tc))]

    lines += ["", "## Per-kernel opcode profile", ""]
    for n, k in sorted(ks.items()):
        top = ", ".join(f"{op}×{c}" for op, c in k["top_opcodes"][:8])
        shared = k["shared_by"]
        share = f" (shared TU: {', '.join(shared)})" if len(shared) > 1 else ""
        classes = ", ".join(f"{cls} {pct}%" for cls, pct in
                            list(k.get("class_pct", {}).items())[:8])
        lines.append(f"### `{n}`{share}")
        lines.append(f"- {k['registers']} regs · {k['instr']} SASS instr · "
                     f"{k['math_pct']}% math · {k['branch_pct']}% branch · "
                     f"{k['gmem_ops']} gmem / {k['smem_ops']} smem ops · flags: {_flagstr(k) or '—'}")
        lines.append(f"- datapath: {classes}")
        lines.append(f"- top opcodes: {top}")
        lines.append("")
    return lines


def _md_rocm(report: dict) -> list[str]:
    ks = report["kernels"]
    lines = [
        "# rlx-rocm kernel inspection report",
        "",
        f"- target: `rocm`  arch: `{report['arch']}`",
        f"- entries: {report['n_entries']}  translation units: {report['n_tus']}  "
        f"kernels: {report['n_kernels']}",
        f"- llvm-objdump: `{report['tools']['llvm_objdump']}`",
        "",
        "VGPR/SGPR/LDS/scratch are exact (code-object metadata). Occupancy is a "
        "theoretical waves/SIMD bound from VGPR/SGPR pressure (LDS/scratch bound "
        "per-workgroup occupancy, which needs the launch geometry).",
        "",
        "## Summary (sorted by VGPR pressure)",
        "",
        "| kernel | vgpr | sgpr | vgprSpill | sgprSpill | lds | scratch | instr | occ | waves/SIMD | limiter | flags |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|",
    ]
    for n, k in sorted(ks.items(), key=lambda kv: -kv[1]["vgpr"]):
        occ = k["occ"]
        lines.append(
            f"| `{n}` | {k['vgpr']} | {k['sgpr']} | {k['vgpr_spill']} | "
            f"{k['sgpr_spill']} | {k['lds']} | {k['scratch']} | {k['instr']} | "
            f"{occ.get('occupancy_pct')}% | {occ.get('waves_per_simd')} | "
            f"{occ.get('limiter')} | {_flagstr(k, 'rocm')} |")

    spill = {n: k for n, k in ks.items()
             if k["vgpr_spill"] or k["sgpr_spill"] or k["scratch"]}
    if spill:
        lines += ["", "## ⚠ Register spills / scratch (perf risk — VGPR pressure)", ""]
        for n, k in sorted(spill.items(), key=lambda kv: -(kv[1]["scratch"] + kv[1]["vgpr_spill"])):
            lines.append(f"- `{n}`: {k['vgpr']} vgpr / {k['sgpr']} sgpr, "
                         f"{k['vgpr_spill']} vgpr-spill / {k['sgpr_spill']} sgpr-spill, "
                         f"{k['scratch']}B scratch")

    mfma = [n for n, k in ks.items() if k.get("matrix_core")]
    if mfma:
        lines += ["", "## Matrix-core kernels (MFMA/WMMA present)", "", ", ".join(f"`{n}`" for n in sorted(mfma))]

    lines += ["", "## Per-kernel opcode profile", ""]
    for n, k in sorted(ks.items()):
        top = ", ".join(f"{op}×{c}" for op, c in k["top_opcodes"][:8])
        shared = k["shared_by"]
        share = f" (shared TU: {', '.join(shared)})" if len(shared) > 1 else ""
        lines.append(f"### `{n}`{share}")
        lines.append(f"- {k['vgpr']} vgpr / {k['sgpr']} sgpr · {k['instr']} GCN instr · "
                     f"{k['math_pct']}% math · {k['branch_pct']}% branch · "
                     f"{k['gmem_ops']} gmem / {k['lds_ops']} lds ops · flags: {_flagstr(k, 'rocm') or '—'}")
        lines.append(f"- top opcodes: {top}")
        lines.append("")
    return lines


def _workspace_root() -> Path:
    """Locate the rlx workspace root from this script's path (…/crates/
    backends/rlx-cuda/tools/kernel-inspect/kinspect.py)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "Cargo.toml").exists() and (p / "crates").is_dir():
            return p
    return here.parents[5]


def _detect_run_target(explicit: str) -> str:
    if explicit and explicit != "auto":
        return explicit
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return "rocm"
    return "cuda"


def run_run(args) -> int:
    """Build the backend, run its tests with RLX_DUMP_KERNELS set to snapshot
    every kernel it JIT-compiles (NVRTC→PTX for CUDA, hipRTC→code-object for
    ROCm), then analyze the dump."""
    root = Path(args.root) if args.root else _workspace_root()
    target = _detect_run_target(args.target)
    pkg = "rlx-rocm" if target == "rocm" else "rlx-cuda"
    artifact = "codeobj" if target == "rocm" else "ptx"
    dump = Path(args.dump or (root / "target" / "kernel-inspect" / f"dump-{target}"))
    if dump.exists() and not args.keep:
        shutil.rmtree(dump)
    dump.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["RLX_DUMP_KERNELS"] = str(dump)
    if target == "cuda":
        # Make CUDA binutils reachable for the test run (nvrtc via driver).
        cuda = args.cuda or env.get("CUDA_HOME") or "/usr/local/cuda"
        if (Path(cuda) / "bin").is_dir():
            env["PATH"] = f"{Path(cuda) / 'bin'}:{env.get('PATH', '')}"

    cargo = ["cargo", "test", "-p", pkg]
    if args.release:
        cargo.append("--release")
    if args.test_args:
        cargo += args.test_args
    # tests write into a shared dump dir; keep them serial to avoid thrash and
    # make coverage deterministic.
    cargo += ["--", "--test-threads=1"]
    print(f"[inspect] target   : {target}  ({pkg})")
    print(f"[inspect] workspace: {root}")
    print(f"[inspect] dump dir : {dump}")
    print(f"[inspect] running  : {' '.join(cargo)}  (RLX_DUMP_KERNELS set)")
    rc = subprocess.run(cargo, cwd=root, env=env).returncode
    if rc != 0:
        print(f"[inspect] warning: `cargo test` exited {rc} — analyzing whatever "
              f"kernels were captured before the failure", file=sys.stderr)
    if not (dump / artifact).is_dir() or not any((dump / artifact).iterdir()):
        sys.exit(f"[inspect] no kernels captured in {dump}/{artifact} — did the GPU "
                 f"tests actually run? (check the build/test output above)")

    ns = argparse.Namespace(dump_dir=str(dump), out=args.out, arch=args.arch,
                            cuda=args.cuda)
    return run_analyze(ns)


def run_diff(args) -> int:
    ro = json.loads(Path(args.old).read_text())
    rn = json.loads(Path(args.new).read_text())
    old, new = ro["kernels"], rn["kernels"]
    rocm = "rocm" in (ro.get("target"), rn.get("target"))
    rkey = "vgpr" if rocm else "registers"
    label = "vgpr" if rocm else "regs"

    def spill_of(k):
        if rocm:
            return k.get("vgpr_spill", 0) + k.get("sgpr_spill", 0) + k.get("scratch", 0)
        return k.get("spill_stores", 0) + k.get("spill_loads", 0)

    names = sorted(set(old) | set(new))
    changed = False
    print(f"{'kernel':<34} {label:>12} {'spill':>14} {'instr':>14}")
    for n in names:
        o, m = old.get(n), new.get(n)
        if o is None:
            print(f"{n:<34} {'NEW':>12}")
            changed = True
            continue
        if m is None:
            print(f"{n:<34} {'REMOVED':>12}")
            changed = True
            continue
        dr = m[rkey] - o[rkey]
        osp, nsp = spill_of(o), spill_of(m)
        di = m["instr"] - o["instr"]
        if dr or (nsp - osp) or di:
            changed = True
            rs = f"{o[rkey]}->{m[rkey]}({dr:+d})" if dr else f"{m[rkey]}"
            ss = f"{osp}->{nsp}({nsp-osp:+d})" if nsp != osp else f"{nsp}"
            is_ = f"{o['instr']}->{m['instr']}({di:+d})" if di else f"{m['instr']}"
            print(f"{n:<34} {rs:>12} {ss:>14} {is_:>14}")
    if not changed:
        print(f"no {label}/spill/instruction-count changes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="build + dump (cargo test) + analyze — the one-shot")
    r.add_argument("--target", choices=["cuda", "rocm", "auto"], default="auto",
                   help="backend to inspect (default: auto — nvidia-smi→cuda, rocminfo→rocm)")
    r.add_argument("--root", default=None, help="workspace root (default: auto)")
    r.add_argument("--dump", default=None, help="dump dir (default target/kernel-inspect/dump-<target>)")
    r.add_argument("-o", "--out", default=None, help="report output dir (default <dump>/report)")
    r.add_argument("--arch", default=None, help="target arch (CUDA SM; auto-detect if omitted)")
    r.add_argument("--cuda", default=os.environ.get("CUDA_HOME"), help="CUDA toolkit home")
    r.add_argument("--release", action="store_true", help="cargo test --release")
    r.add_argument("--keep", action="store_true", help="keep existing dump dir (append coverage)")
    r.add_argument("test_args", nargs="*", help="extra args passed to `cargo test` (e.g. a test filter)")

    a = sub.add_parser("analyze", help="analyze an existing RLX_DUMP_KERNELS dump dir")
    a.add_argument("dump_dir")
    a.add_argument("-o", "--out", default=None, help="output dir (default <dump>/report)")
    a.add_argument("--arch", default=None, help="target SM (default: auto-detect via nvidia-smi)")
    a.add_argument("--cuda", default=os.environ.get("CUDA_HOME"), help="CUDA toolkit home")

    d = sub.add_parser("diff", help="diff two report.json files")
    d.add_argument("old")
    d.add_argument("new")

    args = ap.parse_args()
    if args.cmd == "run":
        return run_run(args)
    if args.cmd == "diff":
        return run_diff(args)
    if args.cmd == "analyze":
        return run_analyze(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
