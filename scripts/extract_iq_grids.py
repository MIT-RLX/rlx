#!/usr/bin/env python3
"""Extract IQ-family grid LUTs from llama.cpp's ggml-common.h.

Regenerates ``rlx-gguf/src/iq_grids.rs`` byte-for-byte from upstream.
Run when bumping the pinned llama.cpp source.

Usage:
    extract_iq_grids.py <path/to/ggml-common.h>     # writes to stdout
    extract_iq_grids.py                              # uses DEFAULT_HEADER

Then pipe to the target file:

    extract_iq_grids.py path/to/ggml-common.h > rlx-gguf/src/iq_grids.rs

Exit code is non-zero if any expected table is missing or its element
count disagrees with the declared length — both indicate the upstream
header moved beyond what this extractor knows about.
"""
import re
import sys
from pathlib import Path

DEFAULT_HEADER = (
    "/Users/Shared/rlx-models/.eagle3-bench/llama-cpp-b9606"
    "/ggml/src/ggml-common.h"
)

# (table_name, ggml dtype, declared length, Rust grouping per line)
# A length of None means "read from #define NGRID_IQ1S".
TABLES = [
    ("kmask_iq2xs", "uint8_t", 8, 4),
    ("ksigns_iq2xs", "uint8_t", 128, 16),
    ("iq2xxs_grid", "uint64_t", 256, 4),
    ("iq2xs_grid", "uint64_t", 512, 4),
    ("iq2s_grid", "uint64_t", 1024, 4),
    ("iq3xxs_grid", "uint32_t", 256, 8),
    ("iq3s_grid", "uint32_t", 512, 8),
    ("iq1s_grid", "uint64_t", None, 4),
    ("kvalues_iq4nl", "int8_t", 16, 8),
]

GGML_TO_RUST = {
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "int8_t": "i8",
}


def extract_table(src, name, dtype):
    pat = re.compile(
        r"GGML_TABLE_BEGIN\(\s*"
        + re.escape(dtype)
        + r"\s*,\s*"
        + re.escape(name)
        + r"\s*,[^)]+\)\s*(.*?)\s*GGML_TABLE_END\(\)",
        re.DOTALL,
    )
    m = pat.search(src)
    if not m:
        raise SystemExit(f"table {name} not found")
    body = m.group(1)
    body = re.sub(r"//.*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return [x.strip() for x in body.split(",") if x.strip()]


def rust_array(name, rust_ty, items, group):
    lines = [f"pub static {name.upper()}: [{rust_ty}; {len(items)}] = ["]
    for i in range(0, len(items), group):
        lines.append("    " + ", ".join(items[i : i + group]) + ",")
    lines.append("];")
    return "\n".join(lines)


def main():
    header_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HEADER)
    src = header_path.read_text()
    m = re.search(r"#define\s+NGRID_IQ1S\s+(\d+)", src)
    ngrid_iq1s = int(m.group(1)) if m else 2048

    print("// Auto-generated from llama.cpp ggml-common.h. DO NOT EDIT BY HAND.")
    print("// Regenerate with `scripts/extract_iq_grids.py <ggml-common.h>`.")
    print("// LUTs for IQ-family dequant. Layout matches llama.cpp byte-for-byte.")
    print()
    print("#![allow(clippy::unreadable_literal)]\n")

    for name, ggml_ty, length, group in TABLES:
        if length is None and name == "iq1s_grid":
            length = ngrid_iq1s
        items = extract_table(src, name, ggml_ty)
        if length is not None and len(items) != length:
            raise SystemExit(
                f"{name}: expected {length} entries, found {len(items)}"
            )
        rust_ty = GGML_TO_RUST[ggml_ty]
        print(rust_array(name, rust_ty, items, group))
        print()


if __name__ == "__main__":
    main()
