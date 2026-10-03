# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ISA portability gate for the CPU backend (x86-64 and aarch64).

One rlx binary has to run on every host of its architecture: the AVX-512
server or M4 that builds it *and* the Atom box or Raspberry Pi someone
deploys it on. That only holds if above-baseline instructions appear
exclusively inside functions reached through a runtime CPU-feature check.
Nothing in `cargo build` enforces it, and the failure mode is a bare
`Illegal instruction` on hardware the author never sees — so this script
checks both halves:

  scan   Disassemble an ELF/Mach-O and attribute every above-baseline
         instruction to its enclosing symbol. A symbol that carries
         AVX/BMI/DotProd/FP16/SVE but is not runtime-gated is a finding.
         Also catches the other direction: a binary built with
         `-C target-cpu=native` has above-baseline code smeared across
         thousands of ordinary symbols (`Op::clone` included), which no
         runtime dispatch can save. Judged against the baseline of the
         target it was built for — see BASELINES; `--baseline` asks the
         cross-target question instead.

  atom   Run the suite on an AVX-less Atom. No Atom needed: Docker
         `--platform linux/amd64` gives a real x86-64 Linux toolchain (on
         Apple Silicon it is Rosetta-backed, so it builds at near-native
         speed), while `qemu-user-static -cpu Denverton` inside that
         container emulates a Goldmont Atom that traps AVX.

  arm    The same for ARMv8.0: `--platform linux/arm64` (native on Apple
         Silicon) plus `qemu-aarch64-static -cpu cortex-a53`, which has
         neither DotProd nor FP16 arithmetic — a Raspberry Pi 3/4.

Usage (from anywhere in the tree; `just check-isa` wraps it):

    python3 .../isa_portability.py scan target/release/my-binary ...
    python3 .../isa_portability.py atom
    python3 .../isa_portability.py arm --models cortex-a53,cortex-a72
    python3 .../isa_portability.py scan --baseline aarch64-v80 some-mac-binary

Exit code is non-zero on any finding, so it works as a gate.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """crates/backends/rlx-cpu/tools/isa_portability.py → repo root.

    Lazy: `atom` copies this file into the container as /tmp/isa.py, where
    the path has no 4th parent and only `scan` runs.
    """
    return Path(__file__).resolve().parents[4]


IMAGE = "rust:1-bookworm"
CONTAINER = "rlx-isa"

# ── What counts as "above baseline" is per-TARGET, not per-arch ───────────
#
# The baseline is whatever rustc assumes with no `-C target-cpu`, and it
# differs by target even within one architecture. `aarch64-apple-darwin`
# compiles for apple-m1 — v8.5, so FP16 arithmetic, DotProd, crypto and LSE
# atomics are all baseline and a `sdot` there is not a finding. The same
# instruction in an `aarch64-unknown-linux-gnu` binary IS one, because that
# target's baseline is ARMv8.0-A: a Cortex-A53 Pi would SIGILL on it.
#
# So scan against the baseline of the target you intend to RUN on. The format
# string picks a sensible default; `--baseline` overrides it, which is how you
# ask the useful cross-target question — "would this Mac build survive on a
# v8.0 box?" — with `--baseline aarch64-v80`.
#
# A tier matches when its mnemonic pattern matches AND its operand pattern is
# found in the full instruction text (either may be None = don't care), and
# its exclusion pattern does not match.
#
# ── x86-64 tiers, in the order an Atom loses them ─────────────────────────
#
# avx    VEX/EVEX-encoded (ymm/zmm operand or a `v`-prefixed mnemonic),
#        which includes F16C's vcvtph2ps. Absent on EVERY Atom through
#        Tremont — Bonnell, Silvermont, Goldmont, Tremont: D525, x5-Z8350,
#        J4125, N5105. Present only on Gracemont (N100/N200/N305) and later.
# bmi    andn/bextr/bzhi/mulx/shlx/shrx/sarx/rorx/pdep/pext.
#        Absent through Goldmont Plus.
# sse42  popcnt/crc32/ptest/round*/pcmpgtq/pmovzx*/pmulld/…
#        Absent on Bonnell only.
#
# Deliberately NOT flagged: tzcnt/lzcnt, which decode as `rep bsf`/`rep bsr`
# and execute correctly as bsf/bsr on pre-BMI parts; and pshufb/phadd/palignr
# (SSSE3) and pinsrw/pextrw (SSE2), present on every 64-bit Atom.
SYMBOL = re.compile(r"^[0-9a-f]+ <(.+)>:$")
WIDE_REG = re.compile(r"\b[yz]mm\d+\b")
VEX = re.compile(r"^v[a-z0-9]+$")
BMI = re.compile(r"^(andn|bextr|bzhi|mulx|shlx|shrx|sarx|rorx|pdep|pext)$")
SSE42 = re.compile(
    r"^(popcnt|crc32|ptest|roundp[sd]|rounds[sd]|pcmpgtq|pblendvb|blendvp[sd]"
    r"|blendp[sd]|pmovzx[bwd][wdq]|pmovsx[bwd][wdq]|pmulld|pminu[dw]|pmaxu[dw]"
    r"|pmins[bd]|pmaxs[bd]|pcmpistri|pcmpestri|pcmpistrm|pcmpestrm|movntdqa"
    r"|extractps|insertps|pinsr[bdq]|pextr[bdq]|packusdw|phminposuw|mpsadbw"
    r"|pcmpeqq|dpp[sd])$"
)

# ── aarch64 tiers ─────────────────────────────────────────────────────────
#
# Baseline ARMv8.0-A already has NEON and single/double FP, plus f16<->f32
# *conversion* (`fcvt`), so plain NEON and `fcvt` are never findings. What is
# optional and must be runtime-detected:
DOTPROD = re.compile(r"^(sdot|udot)$")  # FEAT_DotProd (v8.2 opt / v8.4 req)
I8MM = re.compile(r"^(smmla|ummla|usmmla|usdot|sudot)$")  # FEAT_I8MM
BF16 = re.compile(r"^(bfdot|bfmmla|bfcvt|bfcvtn2?|bfmlal[bt]?)$")  # FEAT_BF16
# FEAT_FP16 is half-precision ARITHMETIC. Conversion is baseline, so `fcvt*`
# is excluded; what marks it is an f-op on `.8h`/`.4h` vectors or `h` scalars.
FP16_OP = re.compile(
    r"^f(add|sub|mul|mulx|div|mla|mls|neg|abs|sqrt|max|min|maxnm|minnm|madd"
    r"|msub|nmul|nmadd|nmsub|cmeq|cmgt|cmge|cmle|cmlt|cmp|cmpe|rsqrte|rsqrts"
    r"|recpe|recps|rinta|rinti|rintm|rintn|rintp|rintx|rintz|acge|acgt)$"
)
FP16_REG = re.compile(r"\.[48]h\b|\bh\d+\b")
NOT_FCVT = re.compile(r"^fcvt")
SVE_OP = re.compile(r"^(ptrue|pfalse|whilel[ote]|whilelo|rdvl|cnt[bhwd]|inc[bhwd])$")
SVE_REG = re.compile(r"\bz\d+\b|\bp\d+(/[zm])?\b")
SME_OP = re.compile(r"^(smstart|smstop|[fbsu]mopa|[fbsu]mops|zero|mova|addha|addva)$")
SME_REG = re.compile(r"\bza\d*\b")
CRYPTO = re.compile(
    r"^(aese|aesd|aesmc|aesimc|sha1[chmps]|sha1su[01]|sha256h2?|sha256su[01]"
    r"|sha512[hsu]|sha512su[01]|pmull2?|eor3|bcax|xar|rax1|sm3[a-z]*|sm4[a-z]*)$"
)

# tier name → (mnemonic, operand, exclude)
_AARCH64_OPTIONAL = {
    "sve": (SVE_OP, None, None),
    "sve-reg": (None, SVE_REG, None),
    "sme": (SME_OP, SME_REG, None),
    "bf16": (BF16, None, None),
    "i8mm": (I8MM, None, None),
    "dotprod": (DOTPROD, None, None),
    "fp16": (FP16_OP, FP16_REG, NOT_FCVT),
    "crypto": (CRYPTO, None, None),
}

BASELINES: dict[str, dict] = {
    # rustc x86_64-*: SSE2. Nothing wider is guaranteed.
    "x86-64": {
        "avx": (VEX, None, None),
        "avx-reg": (None, WIDE_REG, None),
        "bmi": (BMI, None, None),
        "sse42": (SSE42, None, None),
    },
    # rustc aarch64-unknown-linux-gnu / -musl / -android: ARMv8.0-A.
    # A Raspberry Pi 3/4 (Cortex-A53/A72) is exactly this.
    "aarch64-v80": _AARCH64_OPTIONAL,
    # rustc aarch64-apple-*: apple-m1 (v8.5). FP16 arithmetic, DotProd,
    # crypto and LSE are baseline here; SVE/SME/BF16/I8MM are not.
    "aarch64-apple": {
        k: v
        for k, v in _AARCH64_OPTIONAL.items()
        if k in ("sve", "sve-reg", "sme", "bf16", "i8mm")
    },
}

# objdump's `file format` line → default baseline.
FORMATS = (
    ("elf64-x86-64", "x86-64"),
    ("elf32-i386", "x86-64"),
    ("mach-o 64-bit x86-64", "x86-64"),
    ("pei-x86-64", "x86-64"),
    ("mach-o arm64", "aarch64-apple"),
    ("mach-o 64-bit arm64", "aarch64-apple"),
    ("elf64-littleaarch64", "aarch64-v80"),
    ("elf64-bigaarch64", "aarch64-v80"),
)
FILE_FORMAT = re.compile(r"file format (\S+)")

# A symbol may legitimately carry above-baseline code when it is only ever
# reached through a runtime CPUID check. Name-based, because that is the
# convention every dispatched kernel in this crate (and in memchr, half,
# simd_adler32, zstd) already follows.
#
# `libm_math::arch::x86` is compiler-builtins' FMA layer, which does not use a
# feature-named symbol. Verified gated from the disassembly rather than taken
# on trust: `fma::initializer` calls `arch::x86::detect::load_x86_features`
# (cpuid) and stores whichever of `fma_with_fma` / `fma_with_fma4` /
# `fma_fallback` fits into the function pointer it then tail-calls.
GATED = re.compile(
    r"(_avx2?$|_avx\b|avx2|avx512|f16c|vnni|_bmi2|_sse4|_ssse3"
    r"|dotprod|i8mm|_fp16|fp16_|_bf16|_sve2?\b|_sme2?\b|_neon\b|neon_"
    r"|simd|detect|cpuid|is_x86_feature|is_aarch64_feature"
    r"|core_arch::(x86|aarch64|arm_shared)|libm_math::arch::(x86|aarch64)"
    # compiler-builtins' LSE atomics helpers: __aarch64_cas*/swp*/ldadd* are
    # selected by outline-atomics' own runtime check, not by the caller.
    r"|^__aarch64_(cas|swp|ldadd|ldclr|ldeor|ldset)"
    # RustCrypto `aes` (a rlx-ir dependency, for the AES-CTR RNG stream). Its
    # arch backends are named for the ISA, not for a feature, so they look
    # ungated. Verified in aes 0.9.1 rather than assumed: `lib.rs` selects
    # `mod armv8; mod autodetect; pub use autodetect::*` on aarch64, and
    # `autodetect.rs` holds an `arch::features::aes::InitToken`, branching on
    # `init_get()` / `token.get()` with the `soft` backend as the fallback.
    # A hit from any OTHER crate's `armv8`/`ni` module needs the same check
    # before it earns a line here.
    r"|^aes::(armv8|ni|x86)::)",
    re.I,
)


def find_objdump() -> list[str] | None:
    """Locate a disassembler: binutils, LLVM, or rustup's llvm-tools."""
    for cand in ("objdump", "llvm-objdump", "gobjdump"):
        if path := shutil.which(cand):
            return [path]
    try:
        sysroot = subprocess.run(
            ["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    for hit in Path(sysroot).rglob("llvm-objdump*"):
        if hit.is_file() and os.access(hit, os.X_OK):
            return [str(hit)]
    return None


def tier(insn: str, tiers: dict) -> str | None:
    op = insn.split()[0] if insn else ""
    op = op.lstrip("{").rstrip("}")  # strip EVEX decorators
    # Apple's objdump emits Apple-style AArch64 syntax, which hangs the vector
    # arrangement off the MNEMONIC (`sdot.4s v6, v4, v1`) where GNU style puts
    # it on the registers (`sdot v6.4s, v4.16b, v1.16b`). Normalise so one
    # mnemonic pattern covers both; the operand patterns search the full
    # instruction text, so `.4h` is still visible to the FP16 rule either way.
    op = op.split(".", 1)[0]
    for name, (mnemonic, operand, exclude) in tiers.items():
        if exclude is not None and exclude.match(op):
            continue
        if mnemonic is not None and not mnemonic.match(op):
            continue
        if operand is not None and not operand.search(insn):
            continue
        # `-reg` variants exist only so a register-only rule can share a
        # report line with its mnemonic rule.
        return name.removesuffix("-reg")
    return None


def auto_baseline(objdump: list[str], path: Path) -> str:
    """The baseline implied by the file's own format (no override)."""
    out = subprocess.run(
        objdump + ["-f", str(path)], capture_output=True, text=True
    ).stdout.lower()
    return next((b for f, b in FORMATS if f in out), "unknown")


def scan_one(objdump: list[str], path: Path, baseline: str | None) -> tuple[str, dict]:
    out = subprocess.run(
        objdump + ["-d", "--demangle", "--no-show-raw-insn", str(path)],
        capture_output=True,
        text=True,
    ).stdout
    if baseline is None:
        fmt = (m.group(1).lower() if (m := FILE_FORMAT.search(out)) else "") or ""
        head = out[:4000].lower()
        baseline = next((b for f, b in FORMATS if f in fmt or f in head), "unknown")
    tiers = BASELINES.get(baseline, {})
    cur = "?"
    hits: dict = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    for line in out.splitlines():
        if m := SYMBOL.match(line):
            cur = m.group(1)
            continue
        if "\t" not in line:
            continue
        insn = line.split("\t", 1)[1].strip()
        if t := tier(insn, tiers):
            hits[t][cur][insn.split()[0]] += 1
    return baseline, hits


def cmd_scan(paths: list[str], top: int = 12, baseline: str | None = None) -> int:
    objdump = find_objdump()
    if objdump is None:
        print(
            "isa-portability: no objdump found. Install binutils, or "
            "`rustup component add llvm-tools`.",
            file=sys.stderr,
        )
        return 2
    rc = 0
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"isa-portability: not a file: {p}", file=sys.stderr)
            rc = 2
            continue
        detected, hits = scan_one(objdump, path, baseline)
        if detected == "unknown" or not BASELINES.get(detected):
            print(
                f"===== {path.name}\n      no ISA baseline table for this target "
                f"({detected}) — nothing checked",
            )
            continue
        totals = {t: sum(sum(o.values()) for o in s.values()) for t, s in hits.items()}
        findings = [
            (t, sum(ops.values()), sym, ops)
            for t, syms in hits.items()
            for sym, ops in syms.items()
            if not GATED.search(sym)
        ]
        findings.sort(key=lambda f: -f[1])
        print(f"===== {path.name}   baseline: {detected}")
        if baseline is not None and detected != auto_baseline(objdump, path):
            # Judging a binary against a baseline it was not built for answers
            # "what would this code need?", not "is this build correct?" — a
            # real build for the stricter target may gate, soften or drop the
            # same code (a dependency whose SIMD backend is compile-time
            # enabled on one target and runtime-detected on another will look
            # ungated here and be fine there). Treat hits as leads; the
            # authoritative check is scanning a binary built for that target.
            print(
                "      NOTE: overridden baseline — findings below may be "
                "cross-target artifacts, not defects in this build",
            )
        print(f"      above-baseline instructions: {totals or '{}'}")
        print(f"      ungated symbols: {len(findings)}")
        for t, n, sym, ops in findings[:top]:
            ops_s = " ".join(f"{o}x{c}" for o, c in ops.most_common(5))
            print(f"  [{t}] {n:6d}  {sym}")
            print(f"            {ops_s}")
        if len(findings) > top:
            print(f"  … {len(findings) - top} more")
        if findings:
            rc = 1
            arm = detected.startswith("aarch64")
            if sum(totals.values()) > 20_000:
                cpu = "x86-64-v2/v3" if not arm else "a specific -mcpu"
                where = (
                    "any pre-Gracemont Atom"
                    if not arm
                    else "an ARMv8.0 part (Cortex-A53/A72, Pi 3/4)"
                )
                print(
                    f"  → above-baseline code is smeared across ordinary symbols: this "
                    f"binary was built with `-C target-cpu=native` (or {cpu}) and will "
                    f"SIGILL on {where}. Rebuild with RUSTFLAGS unset.",
                )
            else:
                detect = (
                    "is_x86_feature_detected!"
                    if not arm
                    else "is_aarch64_feature_detected!"
                )
                print(
                    f"  → each symbol above carries above-baseline instructions but "
                    f"is not runtime-gated. Hoist the body into a "
                    f"`#[target_feature(...)]` fn called behind `{detect}`.",
                )
    return rc


def docker() -> str | None:
    exe = shutil.which("docker")
    if exe is None:
        return None
    if subprocess.run([exe, "info"], capture_output=True).returncode != 0:
        return None
    return exe


TARGETS = {
    # subcommand → (docker platform, qemu-user binary, scan baseline,
    #               default CPU models, what they emulate)
    "atom": (
        "linux/amd64", "qemu-x86_64-static", "x86-64", ["Denverton"],
        "Denverton=Goldmont Atom, Snowridge=Tremont Atom, Haswell=control",
    ),
    # Docker linux/arm64 is NATIVE on Apple Silicon, so this builds at full
    # speed; cortex-a53/a72 are ARMv8.0 (no DotProd, no FP16 arithmetic),
    # i.e. a Raspberry Pi 3/4. `max` is the everything-enabled control.
    "arm": (
        "linux/arm64", "qemu-aarch64-static", "aarch64-v80", ["cortex-a53"],
        "cortex-a53/a72=ARMv8.0 (Pi 3/4), neoverse-n1=v8.2, max=control",
    ),
}


def cmd_emulate(
    which: str, models: list[str], keep: bool, test_args: list[str]
) -> int:
    platform, qemu, baseline, _, _ = TARGETS[which]
    exe = docker()
    if exe is None:
        print(
            f"isa-portability: Docker is not available, so the emulated-{which} run "
            "is skipped. `scan` covers the static half with no Docker.",
            file=sys.stderr,
        )
        return 2

    def dk(*args: str, **kw) -> subprocess.CompletedProcess:
        return subprocess.run([exe, *args], **kw)

    # rlx-js depends on a sibling ../quickjs-rs checkout; cargo resolves the
    # whole workspace, so the mount has to be there even to build rlx-cpu.
    repo = repo_root()
    sibling = repo.parent / "quickjs-rs"
    # Named volume for CARGO_TARGET_DIR so a second run is incremental.
    mounts = ["-v", f"{repo}:/src", "-v", f"rlx-isa-target-{which}:/build"]
    if sibling.is_dir():
        mounts += ["-v", f"{sibling}:/quickjs-rs"]

    container = f"{CONTAINER}-{which}"
    dk("rm", "-f", container, capture_output=True)
    print(f"[1/4] starting {IMAGE} ({platform}) …")
    if dk(
        "run", "-d", "--name", container, "--platform", platform, *mounts,
        "-w", "/src", "-e", "CARGO_TARGET_DIR=/build/target",
        IMAGE, "sleep", "infinity",
    ).returncode:
        return 2
    try:
        print("[2/4] installing qemu-user-static + binutils + OpenBLAS …")
        if dk(
            "exec", container, "sh", "-c",
            "apt-get update -qq && apt-get install -y -qq "
            "qemu-user-static binutils libopenblas-dev python3 >/dev/null",
        ).returncode:
            return 2

        print(f"[3/4] building rlx-cpu tests inside {platform} …")
        build = dk(
            "exec", container, "sh", "-c",
            "cargo test -p rlx-cpu --release --no-run --locked 2>&1",
            capture_output=True, text=True,
        )
        bins = re.findall(r"\((/build/target/\S+)\)", build.stdout)
        if build.returncode or not bins:
            print(build.stdout[-4000:], file=sys.stderr)
            return 2

        rc = 0
        print(f"[4/4] running {len(bins)} binaries under qemu: {', '.join(models)}")
        dk("cp", str(Path(__file__).resolve()), f"{container}:/tmp/isa.py",
           capture_output=True)
        for b in bins:
            name = Path(b).name.rsplit("-", 1)[0]
            scan = dk(
                "exec", container, "python3", "/tmp/isa.py", "scan", b,
                "--baseline", baseline,
                capture_output=True, text=True,
            )
            sys.stdout.write(scan.stdout)
            rc = rc or (1 if scan.returncode == 1 else 0)
            for model in models:
                run = dk(
                    "exec", container, "sh", "-c",
                    f"{qemu} -cpu {model} {b} "
                    f"--test-threads=1 {' '.join(test_args)} 2>&1 "
                    "| grep -v \"TCG doesn't support\" | tail -3",
                    capture_output=True, text=True,
                )
                tail = " ".join(run.stdout.split())[-110:]
                ok = "Illegal instruction" not in run.stdout and run.returncode == 0
                print(f"  {'ok  ' if ok else 'FAIL'} {name:<28} -cpu {model:<10} {tail}")
                if not ok:
                    rc = 1
                    if "Illegal instruction" in run.stdout:
                        print(
                            f"  → SIGILL on an emulated {model}. Some above-baseline "
                            "instruction executed without a CPUID check; the scan "
                            "above names the symbol.",
                        )
        return rc
    finally:
        if keep:
            print(f"container kept: docker exec -it {container} bash")
        else:
            dk("rm", "-f", container, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="isa_portability", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("scan", help="attribute above-baseline instructions to symbols")
    s.add_argument("paths", nargs="+")
    s.add_argument("--top", type=int, default=12, help="max findings to print per file")
    s.add_argument(
        "--baseline", choices=sorted(BASELINES), default=None,
        help="ISA baseline to judge against. Default: inferred from the file "
             "format. Set it explicitly to ask a cross-target question, e.g. "
             "--baseline aarch64-v80 on a Mac build to see what a Cortex-A53 "
             "would reject.",
    )
    for name, (_, _, _, defaults, models_help) in TARGETS.items():
        e = sub.add_parser(
            name, help=f"build + run the suite on an emulated {name} CPU"
        )
        e.add_argument(
            "--models", default=",".join(defaults),
            help=f"comma-separated qemu CPU models ({models_help}). "
                 f"Default: {','.join(defaults)}",
        )
        e.add_argument("--keep", action="store_true",
                       help="leave the container running")
        e.add_argument("test_args", nargs="*", help="passed to each test binary")
    args = ap.parse_args()

    if args.cmd == "scan":
        return cmd_scan(args.paths, args.top, args.baseline)
    which = args.cmd or "atom"
    if which in TARGETS:
        default = ",".join(TARGETS[which][3])
        models = [m for m in getattr(args, "models", default).split(",") if m]
        return cmd_emulate(which, models or TARGETS[which][3],
                           getattr(args, "keep", False),
                           getattr(args, "test_args", []))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
