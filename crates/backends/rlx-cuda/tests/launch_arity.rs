// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Every `launch_builder` site must pass as many arguments as its kernel declares.
//!
//! `cuLaunchKernel` reads exactly as many pointers as the compiled kernel
//! expects and never sees how many the caller supplied. Too few reads past the
//! end of the argument array; too many silently drops the tail. Neither is a
//! compile error and neither reliably faults — the usual symptom is a kernel
//! that completes having written nothing, or having written garbage. This repo
//! has already shipped that bug once on the HIP side, where the fix note reads:
//! *"Passing 5 u32-offset params — as this did — makes hipModuleLaunchKernel
//! read a 6th arg past the array and misread the offsets, so a >4 GB arena
//! overflows u32 → SIGSEGV / garbage."*
//!
//! ROCm now checks this at launch, because its `launch` takes a slice and the
//! count is available. CUDA goes through cudarc's typed builder across ~157
//! sites, where there is no single place to count — so it is checked here
//! statically instead: resolve each launcher back to its kernel, count the
//! `.arg(` calls, and compare against the `__global__` signature.
//!
//! Being a source lint, this needs no GPU and costs nothing at runtime. Sites
//! it cannot resolve are *reported*, not silently passed — a coverage number
//! that quietly shrinks would make this look like protection it isn't.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `accessor_fn_name -> entry`, from `kernel_cache!(STATIC, accessor, SRC, "entry")`
/// and its `_arch` variant. The invocations wrap across lines, so commas are
/// collected from the balanced argument list rather than a single line.
fn accessor_to_entry(src: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for macro_name in ["kernel_cache!", "kernel_cache_arch!"] {
        let mut from = 0usize;
        while let Some(pos) = src[from..].find(macro_name) {
            let start = from + pos + macro_name.len();
            let bytes = src.as_bytes();
            let Some(open) = src[start..].find('(').map(|i| start + i) else {
                break;
            };
            let mut depth = 0i32;
            let mut end = None;
            for (i, &b) in bytes.iter().enumerate().skip(open) {
                match b {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(end) = end else { break };
            let args: Vec<&str> = src[open + 1..end].split(',').map(str::trim).collect();
            if args.len() >= 4 {
                let accessor = args[1].to_string();
                let entry = args[3].trim_matches('"').to_string();
                if !accessor.is_empty() && !entry.is_empty() {
                    map.insert(accessor, entry);
                }
            }
            from = end;
        }
    }
    map
}

/// `entry -> declared parameter count`, over every shared `.cu`.
fn entry_arity() -> HashMap<String, usize> {
    let mut out = HashMap::new();
    let dir = manifest().join("../rlx-gpu-kernels/kernels");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "cu")
            && let Ok(text) = std::fs::read_to_string(&p)
        {
            let mut from = 0usize;
            while let Some(g) = text[from..].find("__global__") {
                let g = from + g;
                let Some(paren) = text[g..].find('(').map(|i| g + i) else {
                    break;
                };
                let name = text[g..paren]
                    .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty()
                    && let Some(n) = rlx_gpu_kernels::declared_param_count(&text, &name)
                {
                    out.insert(name, n);
                }
                from = g + "__global__".len();
            }
        }
    }
    out
}

#[test]
fn launch_sites_pass_the_declared_number_of_arguments() {
    let kernels_rs = std::fs::read_to_string(manifest().join("src/kernels/mod.rs"))
        .expect("read rlx-cuda kernels/mod.rs");
    let accessors = accessor_to_entry(&kernels_rs);
    let arities_map = entry_arity();
    assert!(
        accessors.len() > 20,
        "resolved only {} kernel accessors — the macro shape changed and this \
         lint is no longer reading it",
        accessors.len()
    );
    assert!(
        arities_map.len() > 50,
        "found only {} kernel signatures — the .cu scrape is out of date",
        arities_map.len()
    );

    let mut files = Vec::new();
    rust_sources(&manifest().join("src"), &mut files);

    let (mut checked, mut skipped) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        // var -> declared arity. A rebinding whose kernel cannot be resolved
        // *removes* the entry rather than leaving the previous one in place:
        // `let kernel = if small_m { a_kernel(..) } else { b_kernel(..) };`
        // spans lines, and treating a stale binding as current made this lint
        // compare an argument list against an unrelated kernel — four
        // confident, wrong findings on the first run.
        let mut bound: HashMap<String, usize> = HashMap::new();
        for (i, line) in lines.iter().enumerate() {
            if let Some(let_pos) = line.find("let ")
                && let Some(eq) = line.find(" = ")
                && eq > let_pos
            {
                let var = line[let_pos + 4..eq]
                    .trim()
                    .trim_start_matches("mut ")
                    .trim()
                    .to_string();
                if !var.is_empty() && var.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    // Collect every `*_kernel(` candidate in this binding,
                    // which may span lines until the terminating `;`.
                    let mut cands: Vec<String> = Vec::new();
                    for l in lines.iter().skip(i).take(12) {
                        let mut rest = *l;
                        while let Some(c) = rest.find("_kernel(") {
                            let seg = &rest[..c + "_kernel".len()];
                            let name = seg
                                .rsplit(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                                .next()
                                .unwrap_or("")
                                .to_string();
                            if !name.is_empty() {
                                cands.push(name);
                            }
                            rest = &rest[c + "_kernel(".len()..];
                        }
                        if l.contains(';') {
                            break;
                        }
                    }
                    let mut arities: Vec<usize> = cands
                        .iter()
                        .filter_map(|a| accessors.get(a))
                        .filter_map(|e| arities_map.get(e).copied())
                        .collect();
                    arities.dedup();
                    // Only usable when every candidate agrees; otherwise the
                    // site is genuinely ambiguous and is reported as skipped.
                    // Exactly one distinct arity across all candidates: a
                    // conditional pick between kernels of the same shape is
                    // still checkable. Anything else is genuinely ambiguous.
                    if arities.len() == 1 {
                        bound.insert(var, arities[0]);
                    } else {
                        bound.remove(&var);
                    }
                }
            }

            let Some(lb) = line.find("launch_builder(&") else {
                continue;
            };
            let rest = &line[lb + "launch_builder(&".len()..];
            let Some(dot) = rest.find(".function") else {
                skipped += 1;
                continue;
            };
            let var = rest[..dot].trim().to_string();
            // Unresolved: conditional selection with differing arities, or a
            // kernel whose source is generated into OUT_DIR and not on disk.
            let Some(&want) = bound.get(&var) else {
                skipped += 1;
                continue;
            };

            // Count `.arg(` from here until the matching `.launch(`.
            let mut args = 0usize;
            let mut found_launch = false;
            for l in lines.iter().skip(i).take(80) {
                args += l.matches(".arg(").count();
                if l.contains(".launch(") || l.contains(".launch_cooperative(") {
                    found_launch = true;
                    break;
                }
            }
            if !found_launch {
                skipped += 1;
                continue;
            }
            checked += 1;
            if args != want {
                bad.push(format!(
                    "{}:{}: `{var}` kernel declares {want} parameter(s) but this launch passes {args}",
                    f.file_name().unwrap().to_string_lossy(),
                    i + 1
                ));
            }
        }
    }

    println!("launch-arity lint: checked {checked} site(s), skipped {skipped} unresolved");
    assert!(
        checked >= 40,
        "only {checked} launch sites resolved (skipped {skipped}) — the lint is not \
         covering enough to be meaningful; fix resolution rather than lower this"
    );
    assert!(
        bad.is_empty(),
        "launch argument count does not match the kernel signature:\n  {}",
        bad.join("\n  ")
    );
}
