// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile an XDNA NPU overlay (`aie.mlir → .xclbin + insts.bin`) **without
//! Python**.
//!
//! The real MLIR-AIE compiler is a native ELF binary (`mlir_aie/bin/aiecc`,
//! ~212 MB, no `libpython`); `aiecc.py` is just a thin Python shim that execs
//! it. So driving overlay generation from rlx is a plain `std::process` call to
//! that binary — verified to run with `python3` blocked. Uses Peano (the
//! `llvm-aie` core compiler), not Vitis/Chess (`--no-xchesscc`), so it needs no
//! proprietary toolchain.

use std::path::Path;
use std::process::Command;

use crate::XdnaError;

/// Inputs for a Python-free overlay compile.
#[derive(Debug, Clone)]
pub struct OverlaySpec<'a> {
    /// The native `aiecc` binary (`.../mlir_aie/bin/aiecc`, NOT `aiecc.py`).
    pub aiecc: &'a str,
    /// The Peano install dir (`.../llvm-aie`) — the AIE core compiler.
    pub peano: &'a str,
    /// Input AIE MLIR (the IRON design's `aie.mlir`, or one rlx emits).
    pub mlir: &'a str,
    /// Scratch dir for intermediates.
    pub tmpdir: &'a str,
    /// Output `.xclbin` path.
    pub out_xclbin: &'a str,
    /// Output `insts_*.bin` path.
    pub out_insts: &'a str,
}

/// Collapse repeated `%name = aie.tile(c, r)` declarations onto one SSA value.
///
/// AIE rows are fixed by hardware, so several logical roles legitimately land on
/// the same physical tile — an attention design has `shim_q`, `shim_kv` and
/// `shim_out`, and on a 1-column device all three ARE tile (0,0). Declaring it
/// three times parses and even builds an xclbin, but the objectfifo allocator
/// then treats them as three tiles and the design returns wrong data (attention
/// came back at max-rel-err 1.0 — the DMA channel assignment collides). One
/// declaration, three references, is both what the hardware is and what the
/// allocator needs.
fn dedupe_tile_decls(mlir: &str) -> String {
    use std::collections::HashMap;
    let mut canon: HashMap<(String, String), String> = HashMap::new();
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut kept: Vec<String> = Vec::new();

    for line in mlir.lines() {
        let t = line.trim();
        let parsed = t
            .strip_prefix('%')
            .and_then(|r| r.split_once(" = aie.tile("))
            .and_then(|(name, rest)| {
                let inner = rest.strip_suffix(')')?;
                let (c, r) = inner.split_once(',')?;
                Some((name.to_string(), c.trim().to_string(), r.trim().to_string()))
            });
        match parsed {
            Some((name, c, r)) => match canon.get(&(c.clone(), r.clone())) {
                // Already declared this physical tile — drop the line, alias the name.
                Some(first) => {
                    alias.insert(name, first.clone());
                }
                None => {
                    canon.insert((c, r), name);
                    kept.push(line.to_string());
                }
            },
            None => kept.push(line.to_string()),
        }
    }
    let mut out = kept.join("\n");
    if mlir.ends_with('\n') {
        out.push('\n');
    }
    // Longest first: `%shim_q` must not be rewritten by a rule for `%shim`.
    let mut names: Vec<&String> = alias.keys().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.len()));
    for name in names {
        let to = &alias[name];
        out = out.replace(&format!("%{name}"), &format!("%{to}"));
    }
    out
}

/// AIE-dialect syntax generation the installed mlir-aie parses.
///
/// rlx's emitters write the original (≤ 1.3.x) syntax; [`upgrade_aie_dialect`]
/// rewrites it forward at compile time, so one emitter serves every release.
/// Two breaking changes, verified against the upstream `AIEOps.td` / tests at
/// each tag (issues #10, #12):
///
/// * **1.4.0** — `aie.dma_bd` moved to a keyword assembly format:
///   `(%a : T, 0, N, [<size = s, stride = t>, …])` →
///   `(%a : T offset = 0 len = N sizes = [s, …] strides = [t, …])`.
/// * **1.4.3** — `!aie.objectfifosubview` was removed: `objectfifo.acquire`
///   returns the `memref`s directly and `objectfifo.subview.access` is gone.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum AieDialect {
    /// ≤ 1.3.x: positional `dma_bd`, subview-typed acquire.
    Legacy,
    /// 1.4.0 – 1.4.2: keyword `dma_bd`, subview-typed acquire.
    DmaBdKeywords,
    /// ≥ 1.4.3: keyword `dma_bd`, memref-typed acquire.
    NoSubview,
}

impl AieDialect {
    fn from_version(v: (u32, u32, u32)) -> Self {
        if v < (1, 4, 0) {
            AieDialect::Legacy
        } else if v < (1, 4, 3) {
            AieDialect::DmaBdKeywords
        } else {
            AieDialect::NoSubview
        }
    }
}

/// [`parse_version`] for sibling modules.
pub(crate) fn parse_version_pub(s: &str) -> Option<(u32, u32, u32)> {
    parse_version(s)
}

/// Parse a `major.minor[.patch]` prefix (`"1.4.4"`, `"1.4.5a1"`, `"1.4"`).
fn parse_version(s: &str) -> Option<(u32, u32, u32)> {
    let mut it = s.split('.').map(|p| {
        let digits: String = p.chars().take_while(char::is_ascii_digit).collect();
        digits.parse::<u32>().ok()
    });
    let major = it.next()??;
    let minor = it.next()??;
    let patch = it.next().flatten().unwrap_or(0);
    Some((major, minor, patch))
}

/// The mlir-aie wheel version owning `aiecc`, from the `mlir_aie*-X.Y.Z.dist-info`
/// next to the package. Covers both layouts seen in the wild: AIECC pointing
/// inside the package (`<site-packages>/mlir_aie/bin/aiecc`) and at a venv's
/// `bin/aiecc` (`<venv>/lib/pythonX.Y/site-packages/…`). `None` for a source
/// build, which has no dist-info.
pub fn mlir_aie_version(aiecc: &str) -> Option<(u32, u32, u32)> {
    let scan = |dir: &Path| -> Option<(u32, u32, u32)> {
        std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".dist-info")?;
            let (dist, ver) = stem.split_once('-')?;
            (dist == "mlir_aie" || dist.starts_with("mlir_aie_"))
                .then(|| parse_version(ver))
                .flatten()
        })
    };
    let aiecc = Path::new(aiecc);
    for dir in aiecc.ancestors().skip(1).take(4) {
        if let Some(v) = scan(dir) {
            return Some(v);
        }
        for lib in ["lib", "lib64"] {
            let Ok(rd) = std::fs::read_dir(dir.join(lib)) else {
                continue;
            };
            for py in rd.flatten() {
                if let Some(v) = scan(&py.path().join("site-packages")) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Which dialect generation to emit for this `aiecc`.
///
/// `RLX_XDNA_AIE_DIALECT=<version>` (e.g. `1.3`, `1.4.0`, `1.4.4`) overrides;
/// otherwise the installed wheel's version decides. A source build has no
/// version on disk, so the CLI generation is the fallback: the declarative
/// driver ships with ≥ 1.4.x releases (and current main), the `--aie-generate-*`
/// driver with the legacy dialect.
fn aie_dialect(aiecc: &str, caps: &AieccCaps) -> AieDialect {
    if let Some(v) = rlx_ir::env::var("RLX_XDNA_AIE_DIALECT").and_then(|s| parse_version(&s)) {
        return AieDialect::from_version(v);
    }
    if let Some(v) = mlir_aie_version(aiecc) {
        return AieDialect::from_version(v);
    }
    match caps.cli {
        AieccCli::Declarative => AieDialect::NoSubview,
        AieccCli::Generate => AieDialect::Legacy,
    }
}

/// Rewrite rlx's legacy-syntax AIE MLIR into `dialect`. See [`AieDialect`].
fn upgrade_aie_dialect(mlir: &str, dialect: AieDialect) -> String {
    if dialect == AieDialect::Legacy {
        return mlir.to_string();
    }
    let mut lines: Vec<String> = mlir
        .lines()
        .map(|l| upgrade_dma_bd(l).unwrap_or_else(|| l.to_string()))
        .collect();
    if dialect >= AieDialect::NoSubview {
        lines = drop_objectfifo_subviews(lines);
    }
    let mut out = lines.join("\n");
    if mlir.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Index just past the `>` closing the `<` at `open`, or `None` if unbalanced.
fn angle_close(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s[open..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Positional `aie.dma_bd(%a : T, off, len[, [<size = s, stride = t>, …]])` →
/// keyword form. `None` when the line holds no positional `dma_bd`.
fn upgrade_dma_bd(line: &str) -> Option<String> {
    const OP: &str = "aie.dma_bd(";
    let start = line.find(OP)? + OP.len();
    let rest = &line[start..];
    let ty_open = rest.find("memref<")? + "memref".len();
    let ty_end = angle_close(rest, ty_open)?;
    let args = rest[ty_end..].strip_prefix(", ")?;
    // The positional operands end at the first `)` outside `[…]` / `<…>`.
    let mut depth = 0i32;
    let close = args.char_indices().find_map(|(i, c)| {
        match c {
            '[' | '<' => depth += 1,
            ']' | '>' => depth -= 1,
            ')' if depth == 0 => return Some(i),
            _ => {}
        }
        None
    })?;
    let mut parts = args[..close].splitn(3, ", ");
    let off = parts.next()?.trim();
    let len = parts.next()?.trim();
    let mut kw = format!(" offset = {off} len = {len}");
    if let Some(dims) = parts.next() {
        let field = |key: &str| -> Option<Vec<&str>> {
            dims.split(key)
                .skip(1)
                .map(|s| s.split([',', '>']).next().map(str::trim))
                .collect()
        };
        let sizes = field("size = ")?;
        let strides = field("stride = ")?;
        if sizes.is_empty() || sizes.len() != strides.len() {
            return None;
        }
        kw.push_str(&format!(
            " sizes = [{}] strides = [{}]",
            sizes.join(", "),
            strides.join(", ")
        ));
    }
    Some(format!(
        "{}{}{kw}{}",
        &line[..start],
        &rest[..ty_end],
        &args[close..]
    ))
}

/// Fold each `%x = aie.objectfifo.subview.access %sv[0] : … -> T` into the
/// acquire that defined `%sv`: the acquire is renamed to `%x` and retyped to
/// the bare `T`. rlx only acquires one object at a time, so index 0 is the
/// whole subview; anything else is left for aiecc to reject loudly.
fn drop_objectfifo_subviews(lines: Vec<String>) -> Vec<String> {
    use std::collections::HashMap;
    const SUBVIEW: &str = "!aie.objectfifosubview<";
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut acquired: HashMap<String, usize> = HashMap::new();
    for line in lines {
        let t = line.trim_start();
        let Some((lhs, rhs)) = t.strip_prefix('%').and_then(|r| r.split_once(" = ")) else {
            out.push(line);
            continue;
        };
        if rhs.starts_with("aie.objectfifo.acquire ")
            && let Some(at) = line.find(SUBVIEW)
            && let Some(end) = angle_close(&line, at + SUBVIEW.len() - 1)
        {
            let inner = &line[at + SUBVIEW.len()..end - 1];
            acquired.insert(lhs.to_string(), out.len());
            out.push(format!("{}{inner}{}", &line[..at], &line[end..]));
            continue;
        }
        if let Some(access) = rhs.strip_prefix("aie.objectfifo.subview.access %")
            && let Some((sv, idx)) = access.split_once('[')
            && idx.starts_with("0]")
            && let Some(&def) = acquired.get(sv)
        {
            out[def] = out[def].replacen(&format!("%{sv} = "), &format!("%{lhs} = "), 1);
            continue;
        }
        out.push(line);
    }
    out
}

/// Which `aiecc` command-line generation we are driving.
///
/// mlir-aie replaced `aiecc`'s CLI: the Python-era driver took "generate this"
/// toggles plus `--no-*` opt-outs, while the current native **declarative
/// driver** (SHA `95b3d1ccc0b`, Sep 2026) builds a compilation graph and you
/// REQUEST outputs by edge name. Every bare flag we used to pass disappeared in
/// that change — reported as issue #2, where `aiecc --help` from a current
/// install accepts none of `--no-xchesscc`, `--no-xbridge`,
/// `--aie-generate-xclbin`, `--aie-generate-npu-insts` or `--no-compile-host`.
///
/// Probed rather than pinned, because both generations are in the wild: a hard
/// switch would break anyone on an older mlir-aie, and the four `name=value`
/// flags we pass (`--tmpdir`, `--peano`, `--xclbin-name`, `--npu-insts-name`)
/// are valid in both, so only the bare flags need to differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AieccCli {
    /// Pre-Sep-2026: `--aie-generate-*` toggles, `--no-xchesscc`/`--no-xbridge`
    /// opt-outs, `--no-compile-host`.
    Generate,
    /// Declarative driver: outputs requested via `--get`, and
    /// xchesscc/xbridge became opt-IN (so omitting them is Peano-only).
    Declarative,
}

/// What this `aiecc` accepts, probed once from its own `--help`.
#[derive(Clone, Debug)]
struct AieccCaps {
    cli: AieccCli,
    /// `--get-<name>` shorthands the binary actually registers. Discovered, not
    /// assumed: only `--get-xclbin` is named in the help prose, and guessing the
    /// edge name for the instruction stream would trade one wrong flag for
    /// another.
    get_shorthands: Vec<String>,
    /// The binary still offers `--no-xchesscc` / `--no-xbridge`. 1.4.0 is
    /// declarative yet defaults to the Chess toolchain, so without them it
    /// fails at `chesslinked_{0}.ll` on a Peano-only host; 1.4.1+ dropped them.
    chess_opt_out: bool,
}

/// Ask `aiecc` what it supports.
///
/// Falls back to [`AieccCli::Generate`] when `--help` cannot be run or says
/// nothing recognisable, which keeps the historical behaviour for anyone whose
/// build worked before this probe existed.
fn probe_aiecc(aiecc: &str) -> AieccCaps {
    let help = Command::new(aiecc)
        .arg("--help")
        .output()
        .ok()
        .map(|o| {
            let mut t = String::from_utf8_lossy(&o.stdout).into_owned();
            t.push_str(&String::from_utf8_lossy(&o.stderr));
            t
        })
        .unwrap_or_default();

    // `--aie-generate-xclbin` is the clearest legacy marker; `--get` is the
    // declarative one. Check the legacy marker first so a driver that somehow
    // offers both keeps the behaviour that is known to work.
    classify_aiecc_help(&help)
}

/// Classify an `aiecc --help` dump. Split out of [`probe_aiecc`] so the flag
/// selection is testable on any host — choosing the wrong CLI generation is a
/// pure string decision, and it should not take an NPU to catch it.
fn classify_aiecc_help(help: &str) -> AieccCaps {
    // `--aie-generate-xclbin` is the clearest legacy marker; `--get` is the
    // declarative one. Check the legacy marker first so a driver that somehow
    // offers both keeps the behaviour that is known to work. An unreadable or
    // empty `--help` also lands on Generate, preserving historical behaviour.
    let cli = if help.contains("--aie-generate-xclbin") {
        AieccCli::Generate
    } else if help.contains("--get") {
        AieccCli::Declarative
    } else {
        AieccCli::Generate
    };

    let mut get_shorthands: Vec<String> = Vec::new();
    for line in help.lines() {
        let t = line.trim_start();
        if !t.starts_with("--get-") {
            continue;
        }
        let name: String = t
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if name.len() > "--get-".len() && !get_shorthands.contains(&name) {
            get_shorthands.push(name);
        }
    }
    AieccCaps {
        cli,
        get_shorthands,
        chess_opt_out: help.contains("--no-xchesscc"),
    }
}

/// The output requests to pass a declarative `aiecc`.
///
/// `RLX_XDNA_AIECC_GET` overrides the whole list (comma-separated edge names),
/// which is the escape hatch for a newer mlir-aie that renames an edge: the
/// instruction-stream edge name is NOT discoverable from `--help` (only
/// `--get-xclbin` is named there), and `--emit-dot` is what prints the real
/// graph. Without the override we request what we can prove and let the
/// existing "did the file appear" check report the rest.
fn declarative_get_args(caps: &AieccCaps) -> Vec<String> {
    if let Some(list) = rlx_ir::env::var("RLX_XDNA_AIECC_GET") {
        let names: Vec<&str> = list
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if !names.is_empty() {
            return vec![format!("--get={}", names.join(","))];
        }
    }
    // Prefer shorthands the binary actually registers.
    let mut args: Vec<String> = Vec::new();
    for want in ["xclbin", "npu-insts", "npu_insts", "insts"] {
        let short = format!("--get-{want}");
        if caps.get_shorthands.contains(&short) {
            args.push(short);
        }
    }
    if args.is_empty() {
        // The shorthands exist even where `--help` does not list them as options.
        // Measured on the NPU rig against every 1.4.x release: `--get-xclbin
        // --get-npu-insts` works on 1.4.0-1.4.4; `--get=xclbin` is rejected by
        // 1.4.0-1.4.3 ("unknown output 'xclbin'"); `--get=<output path>` by all of
        // them ("not part of this build").
        args.push("--get-xclbin".to_string());
        args.push("--get-npu-insts".to_string());
    }
    args
}

/// How to ask `aiecc` for the xclbin and instruction stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Request {
    /// Pre-declarative driver: `--aie-generate-xclbin --aie-generate-npu-insts`.
    Generate,
    /// Declarative: `--get-xclbin --get-npu-insts` (1.4.0+, listed in `--help` or not).
    Shorthands,
    /// Declarative: `--get=xclbin,npu-insts` (some builds accept only this).
    GetNames,
}

impl Request {
    fn tag(self) -> &'static str {
        match self {
            Request::Generate => "generate",
            Request::Shorthands => "shorthands",
            Request::GetNames => "get-names",
        }
    }
    fn from_tag(t: &str) -> Option<Self> {
        [Request::Generate, Request::Shorthands, Request::GetNames]
            .into_iter()
            .find(|r| r.tag() == t)
    }
}

impl AieDialect {
    fn tag(self) -> &'static str {
        match self {
            AieDialect::Legacy => "legacy",
            AieDialect::DmaBdKeywords => "dma-bd-keywords",
            AieDialect::NoSubview => "no-subview",
        }
    }
    fn from_tag(t: &str) -> Option<Self> {
        [
            AieDialect::Legacy,
            AieDialect::DmaBdKeywords,
            AieDialect::NoSubview,
        ]
        .into_iter()
        .find(|d| d.tag() == t)
    }
}

/// What one `aiecc` accepts: which AIE-MLIR syntax and how outputs are
/// requested, plus a Peano that compiles with it when the configured one
/// does not. Predicted from the wheel version and `--help`, then **confirmed
/// by compiling a tiny design**, trying the other combinations if the
/// prediction fails — so a release that changes one of these known points is
/// handled without a code change — and cached per toolchain on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Recipe {
    dialect: AieDialect,
    request: Request,
    /// A Peano that worked where the configured one did not (RLX-XDNA-010).
    peano: Option<String>,
}

/// `aiecc`'s identity for the recipe cache: path, size and mtime, so an
/// upgrade in place is re-probed.
fn fingerprint(aiecc: &str) -> String {
    let meta = std::fs::metadata(aiecc).ok();
    let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{aiecc}|{len}|{mtime}")
}

fn recipe_cache_path() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .map(|d| d.join("rlx-xdna/recipes.tsv"))
}

fn load_recipe(fp: &str) -> Option<Recipe> {
    let text = std::fs::read_to_string(recipe_cache_path()?).ok()?;
    text.lines().find_map(|l| {
        let c: Vec<&str> = l.split('\t').collect();
        (c.len() >= 3 && c[0] == fp).then(|| Recipe {
            dialect: AieDialect::from_tag(c[1]).unwrap_or(AieDialect::Legacy),
            request: Request::from_tag(c[2]).unwrap_or(Request::Generate),
            peano: c.get(3).filter(|p| !p.is_empty()).map(|p| p.to_string()),
        })
    })
}

fn store_recipe(fp: &str, r: &Recipe) {
    let Some(path) = recipe_cache_path() else {
        return;
    };
    let mut lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.starts_with(&format!("{fp}\t")))
        .map(String::from)
        .collect();
    lines.push(format!(
        "{fp}\t{}\t{}\t{}",
        r.dialect.tag(),
        r.request.tag(),
        r.peano.as_deref().unwrap_or("")
    ));
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, lines.join("\n") + "\n");
}

/// The prediction, from the wheel version and `--help`.
fn predicted_recipe(aiecc: &str, caps: &AieccCaps) -> Recipe {
    Recipe {
        dialect: aie_dialect(aiecc, caps),
        request: match caps.cli {
            AieccCli::Generate => Request::Generate,
            AieccCli::Declarative => Request::Shorthands,
        },
        peano: None,
    }
}

/// Every combination to try, the prediction first.
fn ladder(pred: &Recipe, forced_dialect: bool) -> Vec<Recipe> {
    let dialects: Vec<AieDialect> = if forced_dialect {
        vec![pred.dialect]
    } else {
        let mut d = vec![pred.dialect];
        for x in [
            AieDialect::NoSubview,
            AieDialect::DmaBdKeywords,
            AieDialect::Legacy,
        ] {
            if !d.contains(&x) {
                d.push(x);
            }
        }
        d
    };
    let mut requests = vec![pred.request];
    for x in [Request::Shorthands, Request::Generate, Request::GetNames] {
        if !requests.contains(&x) {
            requests.push(x);
        }
    }
    let mut out = Vec::new();
    for &request in &requests {
        for &dialect in &dialects {
            out.push(Recipe {
                dialect,
                request,
                peano: None,
            });
        }
    }
    out
}

/// The recipe for this toolchain: cached, else found by compiling the
/// passthrough design with each candidate. Falls back to the prediction (so
/// the real compile reports the real error) when nothing compiles.
fn recipe_for(spec: &OverlaySpec, caps: &AieccCaps) -> Recipe {
    let fp = fingerprint(spec.aiecc);
    let forced = rlx_ir::env::var("RLX_XDNA_AIE_DIALECT").is_some();
    if !forced && let Some(r) = load_recipe(&fp) {
        return r;
    }
    let pred = predicted_recipe(spec.aiecc, caps);
    let dir = std::env::temp_dir().join(format!("rlx_xdna_recipe_{}", std::process::id()));
    // A design that uses BOTH constructs the releases changed — a compute core
    // that acquires from an objectfifo, and runtime `dma_bd`s. The passthrough
    // design has no core, so every dialect parses it and it cannot tell 1.4.1
    // from 1.4.3 (measured: it confirmed the wrong syntax for 1.4.1).
    let src = crate::aie::emit_eltwise(1024, 1024, crate::aie::Eltwise::Relu);
    for cand in ladder(&pred, forced) {
        let _ = std::fs::remove_dir_all(&dir);
        if std::fs::create_dir_all(dir.join("build")).is_err() {
            break;
        }
        let mlir = dir.join("probe.mlir");
        if std::fs::write(&mlir, upgrade_aie_dialect(&src, cand.dialect)).is_err() {
            break;
        }
        let (m, b, x, i) = (
            mlir.to_string_lossy().into_owned(),
            dir.join("build").to_string_lossy().into_owned(),
            dir.join("p.xclbin").to_string_lossy().into_owned(),
            dir.join("p_insts.bin").to_string_lossy().into_owned(),
        );
        let trial = OverlaySpec {
            aiecc: spec.aiecc,
            peano: spec.peano,
            mlir: &m,
            tmpdir: &b,
            out_xclbin: &x,
            out_insts: &i,
        };
        if run_aiecc(&trial, None, caps, cand.request).is_ok() {
            let _ = std::fs::remove_dir_all(&dir);
            if cand != pred {
                eprintln!(
                    "[rlx-xdna] {} needs {} syntax with {} output requests (not the predicted \
                     {}/{}); remembered in {}",
                    spec.aiecc,
                    cand.dialect.tag(),
                    cand.request.tag(),
                    pred.dialect.tag(),
                    pred.request.tag(),
                    recipe_cache_path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                );
            }
            if !forced {
                store_recipe(&fp, &cand);
            }
            return cand;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    pred
}

/// A failure another Peano build may not have (RLX-XDNA-010).
fn peano_signature(err: &str) -> bool {
    [
        "expected value token",
        "ran out of registers",
        "Error running Peano",
    ]
    .iter()
    .any(|s| err.contains(s))
}

/// Whether an aiecc failure is Peano's `llc` itself crashing or refusing an
/// instruction (an assertion, `LLVM ERROR: unable to legalize …`) — a compiler
/// limitation on that design, not a broken install. Another Peano build rarely
/// helps; a different kernel formulation does.
pub fn llc_crash(err: &str) -> bool {
    err.contains("llc' failed")
        && ["Assertion", "LLVM ERROR", "Aborted"]
            .iter()
            .any(|s| err.contains(s))
}

/// Compile `spec.mlir` to `spec.out_xclbin` + `spec.out_insts` by invoking the
/// **native** `aiecc` binary (no Python in the loop). Returns the two output
/// paths on success.
pub fn compile_overlay(spec: &OverlaySpec) -> Result<(String, String), XdnaError> {
    compile_overlay_linked(spec, &[])
}

/// Like [`compile_overlay`] but additionally makes each path in `link_objs` (a
/// pre-compiled AIE-core `.o`, e.g. the Peano-built `aie::mmul` microkernel)
/// available to aiecc's `link_with` resolution by copying it into the tmpdir under
/// its basename — so an emitted `func.func private @k(...) attributes {link_with =
/// "k.o"}` links against it. This is the seam for C++-microkernel cores (task #25).
pub fn compile_overlay_linked(
    spec: &OverlaySpec,
    link_objs: &[&str],
) -> Result<(String, String), XdnaError> {
    if !Path::new(spec.aiecc).exists() {
        return Err(XdnaError(format!(
            "native aiecc binary not found at {} (point at mlir_aie/bin/aiecc, not aiecc.py)",
            spec.aiecc
        )));
    }
    // Each attempt gets a CLEAN tmpdir. aiecc leaves a `.prj` tree and partial
    // objects behind on failure, and re-running into that directory makes the
    // retry fail the same way regardless of the flags — the -O0 rung below
    // compiles fine by hand and still failed here until this reset existed.
    let stage = |dir: &str| -> Result<(), XdnaError> {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).map_err(|e| XdnaError(format!("create tmpdir {dir}: {e}")))?;
        for obj in link_objs {
            let base = Path::new(obj)
                .file_name()
                .ok_or_else(|| XdnaError(format!("bad link obj path {obj}")))?;
            let dst = Path::new(dir).join(base);
            std::fs::copy(obj, &dst)
                .map_err(|e| XdnaError(format!("copy link obj {obj} → {}: {e}", dst.display())))?;
        }
        Ok(())
    };

    // Try the default optimisation level, then -O1, then -O0.
    //
    // At -O2 LLVM's own loop vectorizer re-widens a deliberately SCALAR core
    // loop into 16-lane vector arithmetic, and AIE2 has no datapath for some of
    // it — `LLVM ERROR: unable to legalize instruction: %98:_(<16 x s32>) =
    // G_MUL`. These are INT8/BF16 MAC tiles; a 32-bit vector multiply is not a
    // thing they can do. -O1 leaves the loop scalar and the same kernel builds
    // and runs. Retrying is better than pinning -O1 everywhere: the ops that do
    // vectorize keep their -O2 codegen.
    //
    // -O0 is the last rung, and it is about SIZE, not legality. An AIE2 core
    // has 64 KB of data memory but only **16 KB of program memory**, and an
    // optimised kernel can simply not fit: the attention core builds a 19,456 B
    // `.text` at -O1/-O2/-O3 and 11,184 B at -O0. aiecc reports that overflow as
    // `ValueError: Failed to generate cdo because:` with an EMPTY reason, which
    // is why it reads like a mystery — check `llvm-size -A` on
    // `<tmpdir>/main_core_0_2.elf` against 16 KB before assuming anything else.
    // Slower code that fits beats faster code that cannot be placed.
    // Probe once, not per attempt: `--help` on a ~212 MB binary is cheap but the
    // retry loop runs it up to three times, and the answer cannot change mid-compile.
    let caps = probe_aiecc(spec.aiecc);
    let recipe = recipe_for(spec, &caps);
    // Read before the first `stage`: callers may keep the source inside
    // `tmpdir`, which `stage` wipes (issue #9).
    let src = std::fs::read_to_string(spec.mlir)
        .map_err(|e| XdnaError(format!("read {}: {e}", spec.mlir)))?;
    let src = upgrade_aie_dialect(&dedupe_tile_decls(&src), recipe.dialect);
    let attempt = |peano: &str| -> Result<(String, String), XdnaError> {
        let mut last_err = None;
        for opt in [None, Some("-O1"), Some("-O0")] {
            stage(spec.tmpdir)?;
            let deduped = Path::new(spec.tmpdir).join("deduped.mlir");
            std::fs::write(&deduped, &src)
                .map_err(|e| XdnaError(format!("write {}: {e}", deduped.display())))?;
            let mut spec_d = spec.clone();
            let deduped_s = deduped.to_string_lossy().into_owned();
            spec_d.mlir = &deduped_s;
            spec_d.peano = peano;
            match run_aiecc(&spec_d, opt, &caps, recipe.request) {
                Ok(v) => return Ok(v),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.expect("at least one attempt"))
    };
    // A Peano that already rescued this toolchain is tried first.
    let first = recipe
        .peano
        .as_deref()
        .filter(|p| Path::new(p).exists())
        .unwrap_or(spec.peano);
    let err = match attempt(first) {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    if !peano_signature(&err.0) {
        return Err(err);
    }
    // The failure looks like a Peano / mlir-aie mismatch: try every other
    // Peano build installed, pinned build first, and remember one that works.
    let host = crate::toolchain::Host::real();
    let version = mlir_aie_version(spec.aiecc);
    let pinned = version.and_then(crate::toolchain::pinned_peano);
    let mut others = crate::toolchain::peanos(&host);
    others.sort_by_key(|p| {
        std::cmp::Reverse(
            pinned.is_some() && p.version.as_deref().and_then(|v| v.split('+').next()) == pinned,
        )
    });
    for p in others {
        let dir = p.dir.to_string_lossy().into_owned();
        if dir == first || dir == spec.peano {
            continue;
        }
        if let Ok(v) = attempt(&dir) {
            eprintln!(
                "[rlx-xdna] compiled with Peano {} after {} failed (RLX-XDNA-010); remembered \
                 for {}",
                p.version.as_deref().unwrap_or(&dir),
                first,
                spec.aiecc
            );
            store_recipe(
                &fingerprint(spec.aiecc),
                &Recipe {
                    peano: Some(dir),
                    ..recipe.clone()
                },
            );
            return Ok(v);
        }
    }
    Err(err)
}

/// One `aiecc` invocation. `opt` optionally forces an AIE-core optimisation
/// level; the link objects are already staged in `spec.tmpdir` by the caller.
fn run_aiecc(
    spec: &OverlaySpec<'_>,
    opt: Option<&str>,
    caps: &AieccCaps,
    request: Request,
) -> Result<(String, String), XdnaError> {
    // Delete the outputs first. Success is judged by "did these files appear",
    // and they live OUTSIDE tmpdir, so a stale artifact from an earlier build
    // makes a FAILED compile report success and the caller then runs the
    // previous kernel. That produced a bogus PASS during debugging: aiecc died
    // on CDO generation, the old xclbin was still on disk, and the example
    // happily executed it.
    let _ = std::fs::remove_file(spec.out_xclbin);
    let _ = std::fs::remove_file(spec.out_insts);
    let mut extra: Vec<String> = Vec::new();
    if let Some(o) = opt {
        // `-O <uint>` is an ordinary value option: mlir-aie 1.3.0/1.3.1 parse a
        // glued `-O1` as grouped single-letter flags and reject it ("may not
        // occur within a group"), so the level always goes as its own argument.
        extra.push("-O".to_string());
        extra.push(o.trim_start_matches("-O").to_string());
    }
    // Bare flags differ by CLI generation; the `name=value` ones below are
    // accepted by both.
    match request {
        Request::Generate => {
            // Peano, not Vitis/Chess. BOTH flags are required: `--no-xchesscc`
            // alone still routes core-ELF linking through `xchesscc_wrapper`,
            // which execs `xchesscc` from an AMD Vitis install that a
            // peano-only host does not have —
            //   `xchesscc_wrapper: line 53: xchesscc: command not found`
            // and aiecc exits 127 after having compiled everything else.
            extra.push("--no-xchesscc".to_string());
            extra.push("--no-xbridge".to_string());
            extra.push("--aie-generate-xclbin".to_string());
            extra.push("--aie-generate-npu-insts".to_string());
            extra.push("--no-compile-host".to_string());
        }
        Request::Shorthands | Request::GetNames => {
            // No `--no-*` opt-outs here: xchesscc/xbridge became opt-IN, so
            // omitting them is exactly the Peano-only path the comment above
            // describes. Nothing is built that was not requested either, which
            // is what `--no-compile-host` used to buy.
            if caps.chess_opt_out {
                extra.push("--no-xchesscc".to_string());
                extra.push("--no-xbridge".to_string());
            }
            if request == Request::GetNames {
                extra.push("--get=xclbin,npu-insts".to_string());
            } else {
                extra.extend(declarative_get_args(caps));
            }
        }
    }
    let output = Command::new(spec.aiecc)
        .args(&extra)
        .args([
            &format!("--tmpdir={}", spec.tmpdir),
            &format!("--peano={}", spec.peano),
            &format!("--xclbin-name={}", spec.out_xclbin),
            &format!("--npu-insts-name={}", spec.out_insts),
            spec.mlir,
        ])
        .output()
        .map_err(|e| XdnaError(format!("spawn aiecc: {e}")))?;
    // Captured, not streamed: aiecc prints pages per kernel. The tail goes into
    // the error (where the actual failure is); `RLX_VERBOSE=1` shows it all.
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    if rlx_ir::env::flag("RLX_VERBOSE") {
        eprint!("{log}");
    }
    if !output.status.success() {
        return Err(XdnaError(format!(
            "aiecc exited with {} compiling {}:\n{}",
            output.status,
            spec.mlir,
            log_tail(&log, 12)
        )));
    }
    for out in [spec.out_xclbin, spec.out_insts] {
        if !Path::new(out).exists() {
            // On the declarative driver an artifact can go missing simply
            // because we never asked for it: outputs are requested by graph
            // edge name, and the instruction-stream edge is not discoverable
            // from `--help`. Say so, instead of leaving it as "aiecc did not
            // produce" with no next step.
            if caps.cli == AieccCli::Declarative {
                return Err(XdnaError(format!(
                    "aiecc did not produce {out} — this driver requests outputs by graph \
                     edge name and rlx asked for [{}]. Run `aiecc --emit-dot <mlir>` to list \
                     the real edge names, then set RLX_XDNA_AIECC_GET=<comma,separated> to \
                     override the request list.",
                    declarative_get_args(caps).join(" ")
                )));
            }
            return Err(XdnaError(format!("aiecc did not produce {out}")));
        }
    }
    Ok((spec.out_xclbin.to_string(), spec.out_insts.to_string()))
}

/// Compile the vendor `aie::mmul` int8 microkernel to an AIE-core object `out_o`
/// for a square `d×d×d` tile (DIM_M=DIM_K=DIM_N=`d`, i8→i32, 4×8×8 subtiles), via
/// Peano `clang++`. `clangxx` = `<peano>/bin/clang++`, `include` =
/// `<mlir_aie>/include`. The `.o` (symbols `matmul_i8_i32` + `zero_i32`) is what
/// [`compile_overlay_linked`] links against an emitted microkernel overlay.
/// Idempotent-safe (overwrites). Cheap (~1s). See [`mm_kernel_source`] for the
/// mlir-aie versions it reads.
pub fn build_mm_kernel(
    clangxx: &str,
    include: &str,
    d: usize,
    out_o: &str,
) -> Result<(), XdnaError> {
    build_mm_kernel_combo(
        clangxx,
        include,
        d,
        out_o,
        "int8, i8, int32, i32, 4, 8, 8",
        "int32",
    )
}

/// bf16 twin of [`build_mm_kernel`]: the `bf16_f32` combo (bf16 in, f32 out,
/// 4×8×4 subtiles), symbols `matmul_bf16_f32` + `zero_f32`, for
/// [`crate::aie::emit_matmul_microkernel_bf16`]. `d` must divide by 8.
pub fn build_mm_kernel_bf16(
    clangxx: &str,
    include: &str,
    d: usize,
    out_o: &str,
) -> Result<(), XdnaError> {
    build_mm_kernel_combo(
        clangxx,
        include,
        d,
        out_o,
        "bfloat16, bf16, float, f32, 4, 8, 4",
        "float",
    )
}

/// The vendor matmul kernel source under `include`. mlir-aie ≤ v1.4.3 ships it
/// as `aie_kernels/aie2/mm.cc`; v1.4.4 moved it to `aie_kernels/linalg/mm.cc`
/// (which picks `mm_aie2.h` from `__AIE_ARCH__`).
pub fn mm_kernel_source(include: &str) -> Result<String, XdnaError> {
    let candidates = [
        format!("{include}/aie_kernels/aie2/mm.cc"),
        format!("{include}/aie_kernels/linalg/mm.cc"),
    ];
    candidates
        .iter()
        .find(|p| Path::new(p).exists())
        .cloned()
        .ok_or_else(|| {
            XdnaError(format!(
                "matmul kernel source not found (looked for {}) — bad mlir_aie include dir? \
                 Set RLX_XDNA_AIE_INCLUDE to <mlir_aie>/include",
                candidates.join(", ")
            ))
        })
}

/// `combo` is one mlir-aie `combos(X)` entry (`ctype_in, in, ctype_out, out,
/// r, s, t`); `ctype_out` is its C accumulator type, for the `zero_<out>`
/// fallback.
fn build_mm_kernel_combo(
    clangxx: &str,
    include: &str,
    d: usize,
    out_o: &str,
    combo: &str,
    ctype_out: &str,
) -> Result<(), XdnaError> {
    let src = mm_kernel_source(include)?;
    let out_ty = combo.split(',').nth(3).unwrap_or_default().trim();
    let wrap = format!("{out_o}.wrap.cc");
    std::fs::write(&wrap, mm_kernel_wrapper(&src, d, combo, ctype_out, out_ty))
        .map_err(|e| XdnaError(format!("write kernel wrapper {wrap}: {e}")))?;
    let status = Command::new(clangxx)
        .args([
            "-O2",
            "-std=c++20",
            "--target=aie2-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-Wno-missing-template-arg-list-after-template-kw",
            "-DNDEBUG",
            &format!("-I{include}"),
            "-D__AIE_API_AIE_ADF_HPP__",
            "-c",
            &wrap,
            "-o",
            out_o,
        ])
        .status()
        .map_err(|e| XdnaError(format!("spawn {clangxx}: {e}")))?;
    if !status.success() {
        return Err(XdnaError(format!(
            "Peano clang++ failed compiling {src} (DIM={d}, combo {combo})"
        )));
    }
    if !Path::new(out_o).exists() {
        return Err(XdnaError(format!("kernel object {out_o} not produced")));
    }
    Ok(())
}

/// The translation unit compiled for one kernel: tile dims + the single combo,
/// then the vendor source. mlir-aie ≤ v1.4.3 defines `zero_<out>` itself (via
/// its `zero_vectorized_c_func` macro); v1.4.4 moved zeroing to a separate
/// `zero.cc` with one unsuffixed `zero` symbol, so the overlay's `zero_<out>`
/// is supplied here when the macro is absent.
fn mm_kernel_wrapper(src: &str, d: usize, combo: &str, ctype_out: &str, out_ty: &str) -> String {
    format!(
        "#define DIM_M {d}\n#define DIM_K {d}\n#define DIM_N {d}\n\
         #define combos(X) X({combo})\n#include \"{src}\"\n\
         #ifndef zero_vectorized_c_func\n\
         extern \"C\" void zero_{out_ty}({ctype_out} *__restrict c) {{\n\
         \x20 constexpr int L = 64 / sizeof({ctype_out});\n\
         \x20 for (int i = 0; i < DIM_M * DIM_N; i += L)\n\
         \x20   aie::store_v(c + i, aie::zeros<{ctype_out}, L>());\n\
         }}\n\
         #endif\n"
    )
}

/// The last `n` meaningful lines of an aiecc log. The declarative driver
/// redraws a `(k/N) step` progress line with `\r`, which piles up as one long
/// "line" ahead of the real error — split on `\r` too and drop that noise.
fn log_tail(log: &str, n: usize) -> String {
    let progress = |l: &str| {
        l.strip_prefix('(')
            .and_then(|r| r.split_once(')'))
            .is_some_and(|(k, _)| {
                k.split_once('/').is_some_and(|(a, b)| {
                    !a.is_empty()
                        && !b.is_empty()
                        && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_digit())
                })
            })
    };
    // An LLVM crash appends its backtrace (`#N 0x… frame`), the pass stack
    // (`N.\tRunning pass …`) and boilerplate after the assertion that matters.
    let crash_noise = |l: &str| {
        let digits_then = |l: &str, sep: &str| {
            l.split_once(sep)
                .is_some_and(|(d, _)| !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()))
        };
        l.strip_prefix('#').is_some_and(|r| digits_then(r, " "))
            || digits_then(l, ".\t")
            || l.starts_with("PLEASE submit a bug report")
            || l.starts_with("Stack dump:")
    };
    let lines: Vec<&str> = log
        .split(['\n', '\r'])
        .map(str::trim_end)
        .filter(|l| {
            let t = l.trim_start();
            !t.is_empty() && !progress(t) && !crash_noise(t)
        })
        .collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {

    #[test]
    fn log_tail_drops_progress_redraws() {
        let log = "\r(0/48) input.mlir\r(1/48)           \r(20/48) [0/1] objects_{0}.o\n\
                   LLVM ERROR: unable to legalize instruction: G_MUL\n\
                   aiecc: pipeline failed\n";
        assert_eq!(
            log_tail(log, 12),
            "LLVM ERROR: unable to legalize instruction: G_MUL\naiecc: pipeline failed"
        );
        assert_eq!(log_tail("a\nb\nc\n", 2), "b\nc");
        let crash = "llc: AIELegalizerHelper.cpp:217: Assertion `EltSize != 64' failed.\n\
                     PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/\n\
                     Stack dump:\n\
                     0.\tProgram arguments: llc -O2\n\
                     1.\tRunning pass 'Legalizer' on function '@core_0_2'\n\
                     #0 0x00007f llvm::sys::PrintStackTrace (libLLVM.so+0x1)\n\
                     #27 0x0000 _start (llc+0x2)\n\
                     aiecc: 'llc' failed: Aborted (core dumped)\n\
                     aiecc: pipeline failed\n";
        let t = log_tail(crash, 3);
        assert!(
            t.starts_with("llc: AIELegalizerHelper.cpp:217: Assertion"),
            "{t}"
        );
        assert!(llc_crash(&t), "{t}");
        assert!(!llc_crash("aiecc: 'opt' failed: expected value token"));
    }
    use super::*;

    /// The exact line aiecc 1.4.4 rejected in issue #10, and the keyword form
    /// upstream's 1.4.4 tests use (`test/dialect/AIEX/dma_task_iteration.mlir`).
    #[test]
    fn dma_bd_is_rewritten_to_the_1_4_keyword_format() {
        let legacy = "        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}";
        assert_eq!(
            upgrade_dma_bd(legacy).unwrap(),
            "        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}"
        );
        assert_eq!(
            upgrade_dma_bd("aie.dma_bd(%arg2 : memref<64x32xi8>, 512, 2048)").unwrap(),
            "aie.dma_bd(%arg2 : memref<64x32xi8> offset = 512 len = 2048)"
        );
        // Already keyword-form, or not a dma_bd at all: untouched.
        assert!(upgrade_dma_bd("aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)").is_none());
        assert!(upgrade_dma_bd("aie.end").is_none());
    }

    /// Issue #12: 1.4.3 removed `!aie.objectfifosubview`; the acquire yields
    /// the memref and `subview.access` disappears (upstream `cpp_basic.mlir`).
    #[test]
    fn objectfifo_subviews_fold_into_the_acquire() {
        let legacy = "\
      %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<1024xf32>>
      %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<1024xf32>> -> memref<1024xf32>
      %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<4x8xi32>>
      %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<4x8xi32>> -> memref<4x8xi32>
      aie.objectfifo.release @in0(Consume, 1)
";
        assert_eq!(
            upgrade_aie_dialect(legacy, AieDialect::NoSubview),
            "\
      %in = aie.objectfifo.acquire @in0(Consume, 1) : memref<1024xf32>
      %out = aie.objectfifo.acquire @out0(Produce, 1) : memref<4x8xi32>
      aie.objectfifo.release @in0(Consume, 1)
"
        );
        // 1.4.0–1.4.2 still have subviews.
        assert_eq!(
            upgrade_aie_dialect(legacy, AieDialect::DmaBdKeywords),
            legacy
        );
    }

    /// Every emitter output, upgraded to the newest dialect, must be free of
    /// both removed syntaxes — a new emitter shape the rewrite misses fails here
    /// instead of on a user's aiecc.
    #[test]
    fn emitted_designs_carry_no_removed_syntax_after_upgrade() {
        use crate::aie::*;
        let designs = [
            emit_passthrough(1024, 1024),
            emit_eltwise_multicol(4096, 1024, 4, &[Eltwise::Relu]),
            emit_relu_f32(1024, 256),
            emit_relu_bf16(1024, 256),
            emit_matmul(32, 32, 32),
            emit_matmul_microkernel(64, 2, 4, "mm.o"),
            emit_softmax(4, 64),
            emit_attention(16, 32, 1, 0.17, true),
            emit_gather(4, 16, 8, 6),
        ];
        for d in &designs {
            let up = upgrade_aie_dialect(&dedupe_tile_decls(d), AieDialect::NoSubview);
            assert!(!up.contains("objectfifosubview"), "{up}");
            assert!(!up.contains("subview.access"), "{up}");
            for l in up.lines().filter(|l| l.contains("aie.dma_bd(")) {
                assert!(l.contains(" offset = ") && l.contains(" len = "), "{l}");
            }
            // Legacy hosts get the emitter output unchanged.
            assert_eq!(upgrade_aie_dialect(d, AieDialect::Legacy), *d);
        }
    }

    #[test]
    fn dialect_follows_the_mlir_aie_version() {
        assert_eq!(AieDialect::from_version((1, 3, 4)), AieDialect::Legacy);
        assert_eq!(
            AieDialect::from_version((1, 4, 0)),
            AieDialect::DmaBdKeywords
        );
        assert_eq!(
            AieDialect::from_version((1, 4, 2)),
            AieDialect::DmaBdKeywords
        );
        assert_eq!(AieDialect::from_version((1, 4, 3)), AieDialect::NoSubview);
        assert_eq!(AieDialect::from_version((1, 5, 0)), AieDialect::NoSubview);
        assert_eq!(parse_version("1.4.5a1"), Some((1, 4, 5)));
        assert_eq!(parse_version("1.4"), Some((1, 4, 0)));
        assert_eq!(parse_version("x"), None);
    }

    /// Both install layouts: AIECC inside the package (issue #10's reporter)
    /// and a venv `bin/aiecc` (the amd rig).
    #[test]
    fn mlir_aie_version_is_read_from_the_wheel_dist_info() {
        let root = std::env::temp_dir().join(format!("rlx_aie_ver_{}", std::process::id()));
        let site = root.join("venv/lib/python3.14/site-packages");
        std::fs::create_dir_all(site.join("mlir_aie/bin")).unwrap();
        std::fs::create_dir_all(site.join("mlir_aie-1.4.4.dist-info")).unwrap();
        std::fs::create_dir_all(root.join("venv/bin")).unwrap();
        let pkg = site.join("mlir_aie/bin/aiecc");
        let venv = root.join("venv/bin/aiecc");
        assert_eq!(mlir_aie_version(pkg.to_str().unwrap()), Some((1, 4, 4)));
        assert_eq!(mlir_aie_version(venv.to_str().unwrap()), Some((1, 4, 4)));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn mm_kernel_wrapper_selects_one_combo_and_backfills_zero() {
        let w = mm_kernel_wrapper(
            "/x/mm.cc",
            32,
            "bfloat16, bf16, float, f32, 4, 8, 4",
            "float",
            "f32",
        );
        assert!(w.starts_with("#define DIM_M 32\n#define DIM_K 32\n#define DIM_N 32\n"));
        assert!(w.contains(
            "#define combos(X) X(bfloat16, bf16, float, f32, 4, 8, 4)\n#include \"/x/mm.cc\""
        ));
        // Fallback only when the vendor source did not define its own zero_<out>.
        let fallback = &w[w.find("#ifndef zero_vectorized_c_func").expect("guarded")..];
        assert!(fallback.contains("extern \"C\" void zero_f32(float *__restrict c)"));
        assert!(fallback.contains("aie::zeros<float, L>()"));
        assert!(fallback.trim_end().ends_with("#endif"));
    }

    #[test]
    fn mm_kernel_source_accepts_both_mlir_aie_layouts() {
        let root = std::env::temp_dir().join(format!("rlx_mm_src_{}", std::process::id()));
        let inc = root.to_str().unwrap();
        assert!(mm_kernel_source(inc).is_err());
        let linalg = root.join("aie_kernels/linalg");
        std::fs::create_dir_all(&linalg).unwrap();
        std::fs::write(linalg.join("mm.cc"), "").unwrap();
        assert!(
            mm_kernel_source(inc)
                .unwrap()
                .ends_with("aie_kernels/linalg/mm.cc")
        );
        let aie2 = root.join("aie_kernels/aie2");
        std::fs::create_dir_all(&aie2).unwrap();
        std::fs::write(aie2.join("mm.cc"), "").unwrap();
        assert!(
            mm_kernel_source(inc)
                .unwrap()
                .ends_with("aie_kernels/aie2/mm.cc")
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Real `aiecc --help` lines from the current declarative driver, as
    /// attached to issue #2 (mlir-aie SHA `95b3d1ccc0b`, built Sep 11 2026).
    /// Verbatim on purpose: the point of the probe is to agree with the actual
    /// tool, and paraphrasing the fixture would let it agree with a paraphrase.
    const DECLARATIVE_HELP: &str = "\
OVERVIEW: aiecc declarative driver

USAGE: aiecc [options] <input mlir> [-- <host cc args>...]

OPTIONS:
  --get=<name>               - Request graph output(s) by edge name (repeatable / comma-separated); named artifacts also have --get-<name> shorthands (e.g. --get-xclbin)
  --npu-insts-name=<string>  - Output NPU insts filename template (use {0} for multi-device)
  --peano=<string>           - Peano install dir
  --tmpdir=<string>          - Intermediate workdir (default: <input>.prj in cwd)
  --xbridge                  - Link cores with the Chess toolchain (xbridge/BCF) instead of the default
  --xchesscc                 - Compile cores with the Chess toolchain (xchesscc) instead of the default
  --xclbin-name=<string>     - Output xclbin filename template (use {0} for multi-device)
";

    /// Shape of the pre-Sep-2026 driver we used to target exclusively.
    const LEGACY_HELP: &str = "\
OPTIONS:
  --aie-generate-xclbin      - Generate xclbin
  --aie-generate-npu-insts   - Generate NPU instructions
  --no-xchesscc              - Do not use xchesscc
  --no-xbridge               - Do not use xbridge
  --no-compile-host          - Do not compile the host program
  --xclbin-name=<string>     - Output xclbin
";

    /// Without listed shorthands the unlisted-but-real ones are used: the only
    /// form every 1.4.x release accepts (see `declarative_get_args`).
    #[test]
    fn outputs_use_the_shorthands_every_1_4_release_accepts() {
        let caps = classify_aiecc_help(DECLARATIVE_HELP);
        assert_eq!(
            declarative_get_args(&caps),
            vec!["--get-xclbin".to_string(), "--get-npu-insts".to_string()]
        );
    }

    /// The prediction goes first, then every other syntax × request pair, once.
    #[test]
    fn the_ladder_tries_the_prediction_first_and_every_pair_once() {
        let pred = Recipe {
            dialect: AieDialect::DmaBdKeywords,
            request: Request::Shorthands,
            peano: None,
        };
        let l = ladder(&pred, false);
        assert_eq!(l[0], pred);
        assert_eq!(l.len(), 9);
        let mut uniq = l.clone();
        uniq.dedup();
        assert_eq!(uniq.len(), 9);
        // A forced dialect only varies the request.
        assert!(
            ladder(&pred, true)
                .iter()
                .all(|r| r.dialect == AieDialect::DmaBdKeywords)
        );
    }

    #[test]
    fn recipes_round_trip_through_the_cache() {
        let dir = std::env::temp_dir().join(format!("rlx_recipe_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _g = rlx_ir::env::scoped("RLX_XDNA_AIE_DIALECT", "");
        // SAFETY: test-local cache root; read back below on this thread.
        unsafe { std::env::set_var("XDG_CACHE_HOME", &dir) };
        let r = Recipe {
            dialect: AieDialect::NoSubview,
            request: Request::GetNames,
            peano: Some("/p/llvm-aie".into()),
        };
        store_recipe("/x/aiecc|1|2", &r);
        store_recipe(
            "/y/aiecc|1|2",
            &Recipe {
                peano: None,
                ..r.clone()
            },
        );
        assert_eq!(load_recipe("/x/aiecc|1|2"), Some(r.clone()));
        assert_eq!(load_recipe("/y/aiecc|1|2").unwrap().peano, None);
        assert_eq!(load_recipe("/z/aiecc|1|2"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peano_mismatch_signatures_are_recognised() {
        assert!(peano_signature(
            "opt: x.ll:88:25: error: expected value token"
        ));
        assert!(peano_signature(
            "error: ran out of registers during register allocation"
        ));
        assert!(!peano_signature("aiecc: --get: unknown output 'xclbin'"));
    }

    /// 1.4.0: declarative, but Chess is still the default and opted out of.
    #[test]
    fn a_transitional_driver_keeps_the_chess_opt_outs() {
        let help = format!("{DECLARATIVE_HELP}  --no-xchesscc  - Do not use xchesscc\n");
        let caps = classify_aiecc_help(&help);
        assert_eq!(caps.cli, AieccCli::Declarative);
        assert!(caps.chess_opt_out);
        assert!(!classify_aiecc_help(DECLARATIVE_HELP).chess_opt_out);
    }

    #[test]
    fn current_driver_is_detected_as_declarative() {
        let caps = classify_aiecc_help(DECLARATIVE_HELP);
        assert_eq!(caps.cli, AieccCli::Declarative);
    }

    #[test]
    fn older_driver_is_detected_as_generate() {
        let caps = classify_aiecc_help(LEGACY_HELP);
        assert_eq!(caps.cli, AieccCli::Generate);
    }

    /// An `aiecc` we cannot interrogate must keep the behaviour that worked
    /// before the probe existed, not fall into the newer flag set.
    #[test]
    fn unreadable_help_falls_back_to_the_historical_flags() {
        assert_eq!(classify_aiecc_help("").cli, AieccCli::Generate);
        assert_eq!(classify_aiecc_help("garbage").cli, AieccCli::Generate);
    }

    /// The regression this guards: every bare flag rlx used to pass is absent
    /// from the current driver. If someone re-adds one unconditionally, the
    /// Declarative branch must not be the place it lands.
    #[test]
    fn none_of_the_legacy_bare_flags_exist_in_the_current_driver() {
        for f in [
            "--no-xchesscc",
            "--no-xbridge",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
        ] {
            assert!(
                !DECLARATIVE_HELP.contains(f),
                "{f} unexpectedly present in the declarative driver fixture"
            );
        }
        // …while the `name=value` flags we keep passing in BOTH modes are there.
        for f in [
            "--tmpdir=",
            "--peano=",
            "--xclbin-name=",
            "--npu-insts-name=",
        ] {
            assert!(DECLARATIVE_HELP.contains(f), "{f} missing from the fixture");
        }
    }

    /// Chess became opt-IN, so the Peano-only path is "say nothing" — the
    /// declarative request list must carry no `--no-*` opt-outs at all.
    #[test]
    fn declarative_requests_outputs_and_opts_out_of_nothing() {
        let caps = classify_aiecc_help(DECLARATIVE_HELP);
        let args = declarative_get_args(&caps);
        assert!(
            args.iter().all(|a| a.starts_with("--get")),
            "expected only output requests, got {args:?}"
        );
        assert!(
            args.iter().any(|a| a.contains("xclbin")),
            "xclbin was never requested: {args:?}"
        );
    }

    /// When the binary registers `--get-<name>` shorthands as real options we
    /// use them, rather than guessing edge names.
    #[test]
    fn get_shorthands_are_discovered_not_assumed() {
        let help = format!(
            "{DECLARATIVE_HELP}  --get-xclbin  - Request the xclbin\n  --get-npu-insts  - Request the NPU instruction stream\n"
        );
        let caps = classify_aiecc_help(&help);
        assert!(caps.get_shorthands.contains(&"--get-xclbin".to_string()));
        assert!(caps.get_shorthands.contains(&"--get-npu-insts".to_string()));
        let args = declarative_get_args(&caps);
        assert!(args.contains(&"--get-xclbin".to_string()), "{args:?}");
        assert!(args.contains(&"--get-npu-insts".to_string()), "{args:?}");
    }
}
