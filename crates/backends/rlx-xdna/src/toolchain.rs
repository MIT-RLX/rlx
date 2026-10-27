// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Find the mlir-aie toolchain (`aiecc`, Peano, the `aie_kernels` include tree)
//! instead of making the user set three env vars.
//!
//! `AIECC` / `PEANO` / `RLX_XDNA_AIE_INCLUDE` still win when set. Otherwise the
//! usual install places are searched: `aiecc` on `PATH`, `~/.local` and venvs
//! under `$HOME` (one or two levels deep — `~/aie144`, `~/mlir-aie/ironenv`),
//! the system Python's dist-packages, and `/opt`. When several mlir-aie
//! installs exist, the newest wheel version wins. Peano (`llvm-aie`) is taken
//! from the same site-packages when it is there, else from any other.
//!
//! Every lookup goes through a [`Host`], so tests can point it at a temp tree
//! that looks like any machine.

use std::path::{Path, PathBuf};

/// Where to look: a filesystem root, `$HOME`, `$PATH` and the env vars that
/// override discovery. [`Host::real`] reads the running process.
#[derive(Debug, Clone, Default)]
pub struct Host {
    pub root: PathBuf,
    pub home: Option<PathBuf>,
    pub path: Vec<PathBuf>,
    pub env: Vec<(String, String)>,
}

impl Host {
    pub fn real() -> Self {
        Host {
            root: PathBuf::from("/"),
            home: std::env::var_os("HOME").map(PathBuf::from),
            path: std::env::var_os("PATH")
                .map(|p| std::env::split_paths(&p).collect())
                .unwrap_or_default(),
            env: std::env::vars().collect(),
        }
    }

    fn var(&self, k: &str) -> Option<&str> {
        self.env
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
    }

    /// `p` under this host's root (absolute paths are re-rooted).
    fn at(&self, p: &Path) -> PathBuf {
        match p.strip_prefix("/") {
            Ok(rel) => self.root.join(rel),
            Err(_) => self.root.join(p),
        }
    }
}

/// How a toolchain path was obtained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// From `AIECC` / `PEANO`.
    Env,
    /// Found by searching; the site-packages it came from.
    Discovered(PathBuf),
}

/// A resolved mlir-aie toolchain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toolchain {
    pub aiecc: PathBuf,
    pub peano: PathBuf,
    /// `mlir_aie/include` (holds `aie_kernels/` and `aie_api/`), when found.
    pub include: Option<PathBuf>,
    /// The mlir-aie wheel version, when a `dist-info` records it.
    pub version: Option<(u32, u32, u32)>,
    /// The llvm-aie (Peano) build, e.g. `22.0.0.2026092101`, when recorded.
    pub peano_version: Option<String>,
    pub source: Source,
}

/// The llvm-aie build each mlir-aie release pins in its
/// `utils/peano-requirements.txt` (1.3.4 onward; earlier releases pin none).
pub const PEANO_PINS: &[((u32, u32, u32), &str)] = &[
    ((1, 3, 4), "21.0.0.2026062301"),
    ((1, 4, 0), "21.0.0.2026072001"),
    ((1, 4, 1), "21.0.0.2026080301"),
    ((1, 4, 2), "21.0.0.2026080301"),
    ((1, 4, 3), "22.0.0.2026090701"),
    ((1, 4, 4), "22.0.0.2026092101"),
];

impl Toolchain {
    /// When mlir-aie pins a Peano build and a different one is installed:
    /// `(pinned, installed)`. A mismatch is not always fatal (1.4.4 ran on the
    /// 1.4.0 Peano), but LLVM IR from one can fail to parse in the other's
    /// `opt` (`expected value token`) — known issue RLX-XDNA-010.
    pub fn peano_mismatch(&self) -> Option<(&'static str, String)> {
        let v = self.version?;
        let pinned = PEANO_PINS.iter().find(|(r, _)| *r == v)?.1;
        let installed = self.peano_version.as_deref()?;
        (installed.split('+').next() != Some(pinned)).then(|| (pinned, installed.to_string()))
    }
}

impl Toolchain {
    pub fn aiecc_str(&self) -> String {
        self.aiecc.to_string_lossy().into_owned()
    }
    pub fn peano_str(&self) -> String {
        self.peano.to_string_lossy().into_owned()
    }
}

/// The mlir-aie release `cargo rlx setup xdna` installs by default: the newest
/// one that passes the full NPU suite in the version matrix.
pub const RECOMMENDED: (u32, u32, u32) = (1, 4, 4);

/// Releases measured to fail part of the NPU suite with every published Peano
/// (known issue RLX-XDNA-011). Discovery ranks them below working releases.
pub const KNOWN_BAD: &[(u32, u32, u32)] = &[(1, 3, 0), (1, 3, 1), (1, 3, 3)];

/// The Peano build `version` pins, if it pins one.
pub fn pinned_peano(version: (u32, u32, u32)) -> Option<&'static str> {
    PEANO_PINS
        .iter()
        .find(|(v, _)| *v == version)
        .map(|(_, p)| *p)
}

/// The toolchain for this process: resolved once, cached.
pub fn toolchain() -> Result<Toolchain, String> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Result<Toolchain, String>> = OnceLock::new();
    CACHED.get_or_init(|| resolve(&Host::real())).clone()
}

/// Where `cargo rlx setup xdna` installs toolchains:
/// `$XDG_DATA_HOME/rlx/toolchains` (default `~/.local/share/rlx/toolchains`).
pub fn managed_dir(host: &Host) -> Option<PathBuf> {
    host.var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| host.home.as_ref().map(|h| h.join(".local/share")))
        .map(|d| d.join("rlx/toolchains"))
}

/// Resolve on `host`: env vars first, then discovery and pairing. `Err` is a
/// readable report of where rlx looked, what it saw, and the fix.
pub fn resolve(host: &Host) -> Result<Toolchain, String> {
    let env_path = |k: &str| {
        host.var(k)
            .map(PathBuf::from)
            .filter(|p| host.at(p).exists())
    };
    let found = discover(host);
    let (aiecc, source) = match host.var("AIECC") {
        Some(v) => {
            let p = PathBuf::from(v);
            if !host.at(&p).exists() {
                return Err(format!(
                    "AIECC={v} does not exist. Unset it to let rlx find mlir-aie, or point it at \
                     <site-packages>/mlir_aie/bin/aiecc"
                ));
            }
            (p, Source::Env)
        }
        None => match best_install(&found) {
            Some(i) => (i.aiecc.clone(), Source::Discovered(i.site.clone())),
            None => return Err(not_found_report(host)),
        },
    };
    // Follow a symlinked aiecc to its install, so the version, include tree and
    // sibling Peano are found from where it really lives.
    let real = canonical(host, &aiecc);
    let site = site_packages_of(host, &aiecc).or_else(|| site_packages_of(host, &real));
    let version = site.as_ref().and_then(|s| dist_info_version(&host.at(s)));
    // A release measured to miscompile is refused, not ranked: on 1.3.0
    // softmax and attention ran and returned wrong numbers (rel err 1.0).
    if let Some(v) = version
        && KNOWN_BAD.contains(&v)
        && host
            .var("RLX_XDNA_ALLOW_KNOWN_BAD")
            .is_none_or(|x| x == "0")
    {
        let mut good: Vec<&Install> = found
            .iter()
            .filter(|i| i.version.is_some_and(|w| !KNOWN_BAD.contains(&w)))
            .collect();
        good.sort_by_key(|i| rank(i));
        let better: Vec<String> = good
            .into_iter()
            .take(3)
            .map(|i| {
                let (a, b, c) = i.version.unwrap_or_default();
                format!("{a}.{b}.{c} at {}", i.aiecc.display())
            })
            .collect();
        let (a, b, c) = v;
        let (ra, rb, rc) = RECOMMENDED;
        return Err(format!(
            "mlir-aie {a}.{b}.{c} cannot compile every rlx kernel correctly (known issue \
             RLX-XDNA-011: wrong results or failed compiles on the NPU). Fix: `cargo rlx setup \
             xdna` (installs {ra}.{rb}.{rc}){}. RLX_XDNA_ALLOW_KNOWN_BAD=1 uses it anyway.",
            if better.is_empty() {
                String::new()
            } else {
                format!(
                    ", or use a working release already here: {}",
                    better.join("; ")
                )
            }
        ));
    }
    let peano = match host.var("PEANO") {
        Some(v) => {
            let p = PathBuf::from(v);
            if !host.at(&p).exists() {
                return Err(format!(
                    "PEANO={v} does not exist. Unset it to let rlx find llvm-aie, or point it at \
                     <site-packages>/llvm-aie"
                ));
            }
            p
        }
        None => match pick_peano(host, site.as_deref(), version) {
            Some(p) => p,
            None => {
                let pin = version
                    .and_then(pinned_peano)
                    .map(|b| format!("llvm-aie=={b}"))
                    .unwrap_or_else(|| "llvm-aie".into());
                return Err(format!(
                    "found aiecc at {} but no Peano (llvm-aie) anywhere rlx looks. Fix: \
                     `cargo rlx setup xdna`, or install the build this mlir-aie pins next to \
                     it: `pip install {pin} -f \
                     https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly`, or \
                     set PEANO to an llvm-aie directory",
                    aiecc.display()
                ));
            }
        },
    };
    let include = env_path("RLX_XDNA_AIE_INCLUDE").or_else(|| {
        site.as_ref()
            .map(|s| s.join("mlir_aie/include"))
            .filter(|p| host.at(p).join("aie_kernels").is_dir())
            .or_else(|| {
                [&aiecc, &real].into_iter().find_map(|a| {
                    a.parent()
                        .and_then(|b| b.parent())
                        .map(|m| m.join("include"))
                        .filter(|p| host.at(p).join("aie_kernels").is_dir())
                })
            })
    });
    let peano_version = peano
        .parent()
        .and_then(|s| dist_info_raw(&host.at(s), "llvm_aie"));
    Ok(Toolchain {
        aiecc,
        peano,
        include,
        version,
        peano_version,
        source,
    })
}

/// One discovered mlir-aie install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Install {
    pub aiecc: PathBuf,
    /// The site-packages (or install prefix, for a source build) it lives in.
    pub site: PathBuf,
    pub version: Option<(u32, u32, u32)>,
    /// Where discovery found it (`setup`, `venv`, `PATH`, `~/.local`, …).
    pub origin: &'static str,
}

/// One discovered Peano (llvm-aie) install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeanoInstall {
    pub dir: PathBuf,
    /// The llvm-aie build, e.g. `22.0.0.2026092101+0006955e`.
    pub version: Option<String>,
}

/// Every mlir-aie install rlx can see on `host`, in the order it would pick
/// them (see [`best_install`]).
pub fn installs(host: &Host) -> Vec<Install> {
    let mut v = discover(host);
    v.sort_by_key(rank);
    v
}

/// Working releases before known-bad ones, newest first; an unversioned
/// (source) install after every versioned working one.
fn rank(i: &Install) -> (bool, std::cmp::Reverse<Option<(u32, u32, u32)>>) {
    let bad = i.version.is_some_and(|v| KNOWN_BAD.contains(&v));
    (bad, std::cmp::Reverse(i.version))
}

fn best_install(found: &[Install]) -> Option<&Install> {
    found.iter().min_by_key(|i| rank(i))
}

/// Every Peano install rlx can see on `host`.
pub fn peanos(host: &Host) -> Vec<PeanoInstall> {
    let mut out: Vec<PeanoInstall> = Vec::new();
    for (site, _) in search_sites(host) {
        let dir = site.join("llvm-aie");
        if host.at(&dir.join("bin")).is_dir() && !out.iter().any(|p| p.dir == dir) {
            out.push(PeanoInstall {
                version: dist_info_raw(&host.at(&site), "llvm_aie"),
                dir,
            });
        }
    }
    out
}

/// The Peano for an mlir-aie in `site` at `version`: the build that release
/// pins, from any install; else the one next to it; else the newest.
fn pick_peano(
    host: &Host,
    site: Option<&Path>,
    version: Option<(u32, u32, u32)>,
) -> Option<PathBuf> {
    let all = peanos(host);
    if let Some(pin) = version.and_then(pinned_peano)
        && let Some(p) = all
            .iter()
            .find(|p| p.version.as_deref().and_then(|v| v.split('+').next()) == Some(pin))
    {
        return Some(p.dir.clone());
    }
    if let Some(s) = site {
        let sibling = s.join("llvm-aie");
        if host.at(&sibling.join("bin")).is_dir() {
            return Some(sibling);
        }
    }
    all.into_iter()
        .max_by(|a, b| a.version.cmp(&b.version))
        .map(|p| p.dir)
}

/// Python package directories to search, highest priority first, each with
/// where it came from.
fn search_sites(host: &Host) -> Vec<(PathBuf, &'static str)> {
    let mut out: Vec<(PathBuf, &'static str)> = Vec::new();
    // Deduplicate by real path: venvs symlink lib64 → lib, which listed every
    // install twice.
    let add = |prefix: &Path, origin: &'static str, out: &mut Vec<(PathBuf, &'static str)>| {
        for s in python_site_dirs(host, prefix) {
            let real = canonical(host, &s);
            if !out
                .iter()
                .any(|(p, _)| *p == s || canonical(host, p) == real)
            {
                out.push((s, origin));
            }
        }
    };
    if let Some(m) = managed_dir(host) {
        for d in list_dirs(host, &m) {
            add(&d, "cargo rlx setup", &mut out);
        }
    }
    if let Some(d) = host.var("RLX_XDNA_TOOLCHAIN_DIR") {
        add(Path::new(d), "RLX_XDNA_TOOLCHAIN_DIR", &mut out);
    }
    for k in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Some(d) = host.var(k) {
            add(Path::new(d), "active env", &mut out);
        }
    }
    for dir in &host.path {
        // `<venv>/bin/aiecc` → `<venv>/lib/pythonX.Y/site-packages`.
        if host.at(&dir.join("aiecc")).exists()
            && let Some(venv) = dir.parent()
        {
            add(venv, "PATH", &mut out);
        }
    }
    if let Some(home) = &host.home {
        add(&home.join(".local"), "~/.local", &mut out);
        for d1 in list_dirs(host, home) {
            add(&d1, "venv", &mut out);
            for d2 in list_dirs(host, &d1) {
                add(&d2, "venv", &mut out);
            }
        }
        for conda in [
            "miniconda3",
            "anaconda3",
            "miniforge3",
            "mambaforge",
            ".conda",
        ] {
            for env in list_dirs(host, &home.join(conda).join("envs")) {
                add(&env, "conda env", &mut out);
            }
        }
        for v in list_dirs(host, &home.join(".pyenv/versions")) {
            add(&v, "pyenv", &mut out);
        }
    }
    for base in ["/usr", "/usr/local"] {
        add(Path::new(base), "system", &mut out);
    }
    for d in list_dirs(host, Path::new("/opt")) {
        add(&d, "/opt", &mut out);
    }
    out
}

/// Every mlir-aie install on `host`, in search order.
fn discover(host: &Host) -> Vec<Install> {
    let mut out: Vec<Install> = Vec::new();
    for (site, origin) in search_sites(host) {
        let aiecc = site.join("mlir_aie/bin/aiecc");
        if host.at(&aiecc).exists() && !out.iter().any(|i| i.site == site) {
            out.push(Install {
                version: dist_info_version(&host.at(&site)),
                aiecc,
                site,
                origin,
            });
        }
    }
    // A source build: `<prefix>/bin/aiecc` beside `<prefix>/include/aie_kernels`.
    for dir in &host.path {
        if let Some(prefix) = dir.parent()
            && host.at(&dir.join("aiecc")).exists()
            && host.at(&prefix.join("include/aie_kernels")).is_dir()
            && !out.iter().any(|i| i.site == prefix)
        {
            out.push(Install {
                aiecc: dir.join("aiecc"),
                site: prefix.to_path_buf(),
                version: None,
                origin: "PATH (source build)",
            });
        }
    }
    out
}

/// `<prefix>/lib{,64}/python*/{site,dist}-packages`.
fn python_site_dirs(host: &Host, prefix: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for lib in ["lib", "lib64"] {
        let libdir = prefix.join(lib);
        for py in list_dirs(host, &libdir) {
            let name = py.file_name().map(|n| n.to_string_lossy().into_owned());
            if name.is_some_and(|n| n.starts_with("python")) {
                for pk in ["site-packages", "dist-packages"] {
                    let d = py.join(pk);
                    if host.at(&d).is_dir() {
                        out.push(d);
                    }
                }
            }
        }
    }
    out
}

/// Sub-directories of `dir` (host-rooted), as host paths. Hidden ones are
/// skipped (callers name `.local`, `.pyenv`, `.conda` explicitly).
fn list_dirs(host: &Host, dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(host.at(dir)) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| dir.join(e.file_name()))
        .collect();
    v.sort();
    v
}

/// The site-packages holding `aiecc`: the package layout
/// (`<site>/mlir_aie/bin/aiecc`) or a venv's `bin/aiecc`.
fn site_packages_of(host: &Host, aiecc: &Path) -> Option<PathBuf> {
    let pkg = aiecc.parent()?.parent()?;
    if pkg.file_name().is_some_and(|n| n == "mlir_aie") {
        return pkg.parent().map(Path::to_path_buf);
    }
    let venv = aiecc.parent()?.parent()?;
    python_site_dirs(host, venv)
        .into_iter()
        .find(|s| host.at(&s.join("mlir_aie")).is_dir())
}

/// `p` with symlinks resolved, as a host path (unchanged if it cannot be).
fn canonical(host: &Host, p: &Path) -> PathBuf {
    let Ok(real) = std::fs::canonicalize(host.at(p)) else {
        return p.to_path_buf();
    };
    let root = std::fs::canonicalize(&host.root).unwrap_or_else(|_| host.root.clone());
    match real.strip_prefix(&root) {
        Ok(rel) => Path::new("/").join(rel),
        Err(_) => p.to_path_buf(),
    }
}

/// The raw version string of `<dist>-<version>.dist-info` in `site`.
fn dist_info_raw(site: &Path, dist: &str) -> Option<String> {
    std::fs::read_dir(site).ok()?.flatten().find_map(|e| {
        let name = e.file_name().into_string().ok()?;
        let stem = name.strip_suffix(".dist-info")?;
        let (d, ver) = stem.split_once('-')?;
        (d == dist).then(|| ver.to_string())
    })
}

fn dist_info_version(site: &Path) -> Option<(u32, u32, u32)> {
    std::fs::read_dir(site).ok()?.flatten().find_map(|e| {
        let name = e.file_name().into_string().ok()?;
        let stem = name.strip_suffix(".dist-info")?;
        let (dist, ver) = stem.split_once('-')?;
        (dist == "mlir_aie" || dist.starts_with("mlir_aie_"))
            .then(|| crate::compile::parse_version_pub(ver))
            .flatten()
    })
}

/// No usable mlir-aie: say where rlx looked, what it did see, and the fix.
fn not_found_report(host: &Host) -> String {
    let sites = search_sites(host);
    let mut origins: Vec<&str> = sites.iter().map(|(_, o)| *o).collect();
    origins.dedup();
    let mut seen: Vec<String> = Vec::new();
    for (site, _) in &sites {
        if host.at(&site.join("mlir_aie/bin/aiecc.py")).exists()
            && !host.at(&site.join("mlir_aie/bin/aiecc")).exists()
        {
            let ver = dist_info_version(&host.at(site))
                .map(|(a, b, c)| format!(" {a}.{b}.{c}"))
                .unwrap_or_default();
            seen.push(format!(
                "mlir-aie{ver} at {} ships only aiecc.py — releases before 1.3 have no native \
                 aiecc (known issue RLX-XDNA-009)",
                site.display()
            ));
        }
    }
    for p in peanos(host) {
        seen.push(format!(
            "Peano {} at {} (usable once mlir-aie is installed)",
            p.version.as_deref().unwrap_or("(unknown build)"),
            p.dir.display()
        ));
    }
    let (a, b, c) = RECOMMENDED;
    let pin = pinned_peano(RECOMMENDED).unwrap_or("");
    let mut msg = format!(
        "no usable mlir-aie toolchain found (known issue RLX-XDNA-007).\n  searched {} \
         Python package directories ({}) and PATH",
        sites.len(),
        if origins.is_empty() {
            "none exist".to_string()
        } else {
            origins.join(", ")
        }
    );
    for s in &seen {
        msg.push_str(&format!("\n  saw: {s}"));
    }
    msg.push_str(&format!(
        "\n  fix: `cargo rlx setup xdna` installs mlir-aie {a}.{b}.{c} with its pinned Peano \
         into ~/.local/share/rlx/toolchains (no root needed)\n  \
         or by hand, into any venv (found automatically): `python3 -m venv ~/mlir-aie && \
         ~/mlir-aie/bin/pip install mlir_aie=={a}.{b}.{c} -f \
         https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v{a}.{b}.{c} && \
         ~/mlir-aie/bin/pip install llvm-aie=={pin} -f \
         https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly`\n  \
         or point rlx at an install: AIECC=<site-packages>/mlir_aie/bin/aiecc \
         PEANO=<site-packages>/llvm-aie, or RLX_XDNA_TOOLCHAIN_DIR=<venv or prefix>"
    ));
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rlx_xdna_tc_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A pip install of `mlir_aie` (+ optionally `llvm-aie`) into `site`.
    fn pip_install(root: &Path, site: &str, version: &str, peano: bool) {
        let s = root.join(site.trim_start_matches('/'));
        std::fs::create_dir_all(s.join("mlir_aie/bin")).unwrap();
        std::fs::write(s.join("mlir_aie/bin/aiecc"), "").unwrap();
        std::fs::create_dir_all(s.join("mlir_aie/include/aie_kernels")).unwrap();
        std::fs::create_dir_all(s.join(format!("mlir_aie-{version}.dist-info"))).unwrap();
        if peano {
            std::fs::create_dir_all(s.join("llvm-aie/bin")).unwrap();
        }
    }

    fn host(root: &Path) -> Host {
        Host {
            root: root.to_path_buf(),
            home: Some(PathBuf::from("/home/u")),
            path: vec![PathBuf::from("/usr/bin")],
            env: Vec::new(),
        }
    }

    #[test]
    fn finds_a_user_pip_install_with_its_peano_and_include() {
        let r = tmp("user");
        pip_install(
            &r,
            "/home/u/.local/lib/python3.14/site-packages",
            "1.4.4",
            true,
        );
        let tc = resolve(&host(&r)).unwrap();
        let site = PathBuf::from("/home/u/.local/lib/python3.14/site-packages");
        assert_eq!(tc.aiecc, site.join("mlir_aie/bin/aiecc"));
        assert_eq!(tc.peano, site.join("llvm-aie"));
        assert_eq!(tc.include, Some(site.join("mlir_aie/include")));
        assert_eq!(tc.version, Some((1, 4, 4)));
        assert_eq!(tc.source, Source::Discovered(site));
    }

    /// The amd rig's shape: the newest mlir-aie in one venv, Peano only in another.
    #[test]
    fn prefers_the_newest_venv_and_borrows_peano_from_another() {
        let r = tmp("venvs");
        pip_install(
            &r,
            "/home/u/mlir-aie/ironenv/lib/python3.14/site-packages",
            "1.3.4",
            true,
        );
        pip_install(
            &r,
            "/home/u/aie144/lib/python3.14/site-packages",
            "1.4.4",
            false,
        );
        let tc = resolve(&host(&r)).unwrap();
        assert_eq!(tc.version, Some((1, 4, 4)));
        assert!(tc.aiecc.starts_with("/home/u/aie144"));
        assert_eq!(
            tc.peano,
            PathBuf::from("/home/u/mlir-aie/ironenv/lib/python3.14/site-packages/llvm-aie")
        );
    }

    #[test]
    fn env_vars_override_discovery_and_bad_ones_are_named() {
        let r = tmp("env");
        pip_install(
            &r,
            "/home/u/.local/lib/python3.14/site-packages",
            "1.4.4",
            true,
        );
        pip_install(&r, "/opt/old/lib/python3.12/site-packages", "1.3.4", true);
        let mut h = host(&r);
        h.env = vec![(
            "AIECC".into(),
            "/opt/old/lib/python3.12/site-packages/mlir_aie/bin/aiecc".into(),
        )];
        let tc = resolve(&h).unwrap();
        assert_eq!(tc.source, Source::Env);
        assert_eq!(tc.version, Some((1, 3, 4)));
        h.env = vec![("AIECC".into(), "/nope/aiecc".into())];
        assert!(
            resolve(&h)
                .unwrap_err()
                .contains("AIECC=/nope/aiecc does not exist")
        );
    }

    #[test]
    fn a_pre_1_3_install_is_named_not_reported_missing() {
        let r = tmp("old");
        let s = r.join("home/u/old/lib/python3.12/site-packages/mlir_aie/bin");
        std::fs::create_dir_all(&s).unwrap();
        std::fs::write(s.join("aiecc.py"), "").unwrap();
        let e = resolve(&host(&r)).unwrap_err();
        assert!(
            e.contains("ships only aiecc.py") && e.contains("RLX-XDNA-009"),
            "{e}"
        );
    }

    #[test]
    fn a_peano_other_than_the_pinned_build_is_flagged() {
        let r = tmp("pin");
        pip_install(
            &r,
            "/home/u/.local/lib/python3.14/site-packages",
            "1.4.4",
            true,
        );
        let site = r.join("home/u/.local/lib/python3.14/site-packages");
        std::fs::create_dir_all(site.join("llvm_aie-21.0.0.2026072001+ce8c0f8f.dist-info"))
            .unwrap();
        let tc = resolve(&host(&r)).unwrap();
        assert_eq!(
            tc.peano_mismatch(),
            Some((
                "22.0.0.2026092101",
                "21.0.0.2026072001+ce8c0f8f".to_string()
            ))
        );
        std::fs::remove_dir_all(site.join("llvm_aie-21.0.0.2026072001+ce8c0f8f.dist-info"))
            .unwrap();
        std::fs::create_dir_all(site.join("llvm_aie-22.0.0.2026092101+0006955e.dist-info"))
            .unwrap();
        assert_eq!(resolve(&host(&r)).unwrap().peano_mismatch(), None);
    }

    #[test]
    fn nothing_installed_says_how_to_install() {
        let r = tmp("none");
        let e = resolve(&host(&r)).unwrap_err();
        assert!(
            e.contains("cargo rlx setup xdna") && e.contains("RLX-XDNA-007"),
            "{e}"
        );
        assert!(e.contains("llvm-aie==22.0.0.2026092101"), "{e}");
    }

    /// A stray Peano is reported as seen, so the user knows half is there.
    #[test]
    fn not_found_names_what_it_did_see() {
        let r = tmp("seen");
        std::fs::create_dir_all(r.join("home/u/.local/lib/python3.14/site-packages/llvm-aie/bin"))
            .unwrap();
        let e = resolve(&host(&r)).unwrap_err();
        assert!(e.contains("saw: Peano"), "{e}");
    }

    #[test]
    fn the_setup_dir_wins_and_known_bad_releases_rank_last() {
        let r = tmp("rank");
        pip_install(
            &r,
            "/home/u/old/lib/python3.14/site-packages",
            "1.3.3",
            true,
        );
        pip_install(
            &r,
            "/home/u/mid/lib/python3.14/site-packages",
            "1.3.4",
            true,
        );
        let tc = resolve(&host(&r)).unwrap();
        assert_eq!(
            tc.version,
            Some((1, 3, 4)),
            "1.3.3 is known bad, 1.3.4 must win"
        );
        pip_install(
            &r,
            "/home/u/.local/share/rlx/toolchains/mlir-aie-1.4.4/lib/python3.14/site-packages",
            "1.4.4",
            true,
        );
        let all = installs(&host(&r));
        assert_eq!(all[0].version, Some((1, 4, 4)));
        assert_eq!(all[0].origin, "cargo rlx setup");
        assert_eq!(all.last().unwrap().version, Some((1, 3, 3)));
    }

    /// The pinned Peano is used even when a different one sits next to aiecc.
    #[test]
    fn the_pinned_peano_is_paired_from_any_install() {
        let r = tmp("pair");
        pip_install(
            &r,
            "/home/u/aie/lib/python3.14/site-packages",
            "1.4.4",
            true,
        );
        let sib = r.join("home/u/aie/lib/python3.14/site-packages");
        std::fs::create_dir_all(sib.join("llvm_aie-21.0.0.2026072001+ce8c0f8f.dist-info")).unwrap();
        let other = r.join("home/u/peano/lib/python3.14/site-packages");
        std::fs::create_dir_all(other.join("llvm-aie/bin")).unwrap();
        std::fs::create_dir_all(other.join("llvm_aie-22.0.0.2026092101+0006955e.dist-info"))
            .unwrap();
        let tc = resolve(&host(&r)).unwrap();
        assert_eq!(
            tc.peano,
            PathBuf::from("/home/u/peano/lib/python3.14/site-packages/llvm-aie")
        );
        assert_eq!(tc.peano_mismatch(), None);
    }

    #[test]
    fn conda_envs_and_source_builds_are_found() {
        let r = tmp("conda");
        pip_install(
            &r,
            "/home/u/miniforge3/envs/npu/lib/python3.12/site-packages",
            "1.4.2",
            true,
        );
        assert_eq!(resolve(&host(&r)).unwrap().version, Some((1, 4, 2)));

        let r = tmp("srcbuild");
        std::fs::create_dir_all(r.join("opt/mlir-aie/install/bin")).unwrap();
        std::fs::write(r.join("opt/mlir-aie/install/bin/aiecc"), "").unwrap();
        std::fs::create_dir_all(r.join("opt/mlir-aie/install/include/aie_kernels")).unwrap();
        std::fs::create_dir_all(r.join("home/u/.local/lib/python3.14/site-packages/llvm-aie/bin"))
            .unwrap();
        let mut h = host(&r);
        h.path = vec![PathBuf::from("/opt/mlir-aie/install/bin")];
        let tc = resolve(&h).unwrap();
        assert_eq!(tc.aiecc, PathBuf::from("/opt/mlir-aie/install/bin/aiecc"));
        assert_eq!(
            tc.include,
            Some(PathBuf::from("/opt/mlir-aie/install/include"))
        );
    }

    /// A known-bad release is refused even when AIECC names it, with the
    /// working installs on the machine offered instead.
    #[test]
    fn a_known_bad_release_is_refused_unless_allowed() {
        let r = tmp("bad");
        pip_install(
            &r,
            "/home/u/old/lib/python3.14/site-packages",
            "1.3.0",
            true,
        );
        pip_install(
            &r,
            "/home/u/good/lib/python3.14/site-packages",
            "1.4.4",
            true,
        );
        let mut h = host(&r);
        h.env = vec![(
            "AIECC".into(),
            "/home/u/old/lib/python3.14/site-packages/mlir_aie/bin/aiecc".into(),
        )];
        let e = resolve(&h).unwrap_err();
        assert!(
            e.contains("RLX-XDNA-011") && e.contains("1.4.4 at /home/u/good"),
            "{e}"
        );
        h.env.push(("RLX_XDNA_ALLOW_KNOWN_BAD".into(), "1".into()));
        assert_eq!(resolve(&h).unwrap().version, Some((1, 3, 0)));
    }

    #[cfg(unix)]
    #[test]
    fn a_lib64_symlink_does_not_list_an_install_twice() {
        let r = tmp("lib64");
        pip_install(&r, "/home/u/v/lib/python3.14/site-packages", "1.4.4", true);
        std::os::unix::fs::symlink(r.join("home/u/v/lib"), r.join("home/u/v/lib64")).unwrap();
        assert_eq!(installs(&host(&r)).len(), 1);
    }

    /// AIECC pointing at a symlink still finds the real install's version and include.
    #[test]
    fn a_symlinked_aiecc_resolves_to_its_install() {
        let r = tmp("symlink");
        pip_install(
            &r,
            "/home/u/aie/lib/python3.14/site-packages",
            "1.4.1",
            true,
        );
        std::fs::create_dir_all(r.join("home/u/bin")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            r.join("home/u/aie/lib/python3.14/site-packages/mlir_aie/bin/aiecc"),
            r.join("home/u/bin/aiecc"),
        )
        .unwrap();
        let mut h = host(&r);
        h.env = vec![("AIECC".into(), "/home/u/bin/aiecc".into())];
        let tc = resolve(&h).unwrap();
        assert_eq!(tc.version, Some((1, 4, 1)));
        assert!(tc.include.is_some());
    }

    #[test]
    fn aiecc_without_peano_says_so() {
        let r = tmp("nopeano");
        pip_install(
            &r,
            "/home/u/.local/lib/python3.14/site-packages",
            "1.4.4",
            false,
        );
        let e = resolve(&host(&r)).unwrap_err();
        assert!(e.contains("no Peano"), "{e}");
    }
}
