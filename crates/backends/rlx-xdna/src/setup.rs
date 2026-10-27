// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo rlx setup xdna`: install the mlir-aie toolchain rlx needs, in one
//! command, without root.
//!
//! A matrix-tested mlir-aie release and the Peano (llvm-aie) build it pins go
//! into a fresh venv under [`crate::toolchain::managed_dir`], which discovery
//! searches first — so after setup nothing has to be configured. What needs
//! root (the XRT runtime) is not installed; the exact commands are returned
//! instead, for the distro detected.

use std::path::{Path, PathBuf};

use crate::toolchain::{Host, PEANO_PINS, RECOMMENDED, managed_dir, pinned_peano};

/// What `setup` will do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub version: (u32, u32, u32),
    /// The venv the toolchain goes into.
    pub dir: PathBuf,
    /// Commands to run, in order (argv each).
    pub commands: Vec<Vec<String>>,
    /// The Peano build installed with it.
    pub peano: &'static str,
}

/// Versions `setup` can install: those whose Peano pin rlx knows, minus the
/// releases measured to fail the NPU suite ([`crate::toolchain::KNOWN_BAD`]),
/// newest first.
pub fn installable() -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = PEANO_PINS
        .iter()
        .map(|(v, _)| *v)
        .filter(|v| !crate::toolchain::KNOWN_BAD.contains(v))
        .collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

/// The plan for `version` (default [`RECOMMENDED`]) on `host`, using
/// `python` (default `python3`).
pub fn plan(
    host: &Host,
    version: Option<(u32, u32, u32)>,
    python: Option<&str>,
) -> Result<Plan, String> {
    let version = version.unwrap_or(RECOMMENDED);
    let peano = pinned_peano(version).ok_or_else(|| {
        let list: Vec<String> = installable()
            .iter()
            .map(|(a, b, c)| format!("{a}.{b}.{c}"))
            .collect();
        format!(
            "rlx does not know the Peano build mlir-aie {}.{}.{} pins; installable: {}",
            version.0,
            version.1,
            version.2,
            list.join(", ")
        )
    })?;
    let root = managed_dir(host).ok_or("no $HOME or $XDG_DATA_HOME to install into")?;
    let (a, b, c) = version;
    let dir = root.join(format!("mlir-aie-{a}.{b}.{c}"));
    let pip = dir.join("bin/pip").to_string_lossy().into_owned();
    let d = dir.to_string_lossy().into_owned();
    let py = python.unwrap_or("python3").to_string();
    let commands = vec![
        vec![py, "-m".into(), "venv".into(), d],
        vec![
            pip.clone(),
            "install".into(),
            "--upgrade".into(),
            "pip".into(),
        ],
        vec![
            pip.clone(),
            "install".into(),
            format!("mlir_aie=={a}.{b}.{c}"),
            "-f".into(),
            format!("https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v{a}.{b}.{c}"),
        ],
        vec![
            pip,
            "install".into(),
            format!("llvm-aie=={peano}"),
            "-f".into(),
            "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly".into(),
        ],
    ];
    Ok(Plan {
        version,
        dir,
        commands,
        peano,
    })
}

/// mlir-aie and llvm-aie publish wheels for Linux x86-64 only (XDNA NPUs are
/// on Ryzen AI laptops); elsewhere `setup` says so instead of failing in pip.
pub fn platform_supported() -> Result<(), String> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok(())
    } else {
        Err(format!(
            "mlir-aie publishes toolchains for Linux x86-64 only; this is {}-{}. Run \
             `cargo rlx setup xdna` on the Ryzen AI machine itself",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
    }
}

/// Commands that put the XRT runtime (and the headers rlx's shim builds
/// against) on this distro. Root is needed; `setup` prints, never runs them.
pub fn xrt_instructions(root: &Path) -> Vec<String> {
    let os = std::fs::read_to_string(root.join("etc/os-release")).unwrap_or_default();
    let field = |k: &str| {
        os.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(|v| v.trim_matches('"').to_string())
            .unwrap_or_default()
    };
    let id = field("ID");
    let like = field("ID_LIKE");
    if id == "ubuntu" || id == "debian" || like.contains("ubuntu") || like.contains("debian") {
        vec![
            "sudo apt install libxrt2 libxrt-npu2 libxrt-utils-npu libxrt-dev".into(),
            "# newer XRT (2.25) on Ubuntu: sudo add-apt-repository ppa:amd-team/xrt && sudo apt update"
                .into(),
            "# without root: apt-get download those packages, dpkg-deb -x each into one directory, \
             and set RLX_XDNA_XRT_LIB to its usr/lib/x86_64-linux-gnu"
                .into(),
        ]
    } else if id == "fedora" || like.contains("fedora") {
        vec!["sudo dnf install xrt xrt-devel".into()]
    } else if id == "arch" || like.contains("arch") {
        vec!["install xrt and xrt-plugin-amdxdna from the AUR".into()]
    } else {
        vec![
            "install AMD XRT with the amdxdna plugin and headers for your distro \
             (https://github.com/amd/xdna-driver)"
                .into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> Host {
        Host {
            root: PathBuf::from("/"),
            home: Some(PathBuf::from("/home/u")),
            path: vec![],
            env: vec![],
        }
    }

    #[test]
    fn default_plan_installs_the_recommended_release_with_its_pinned_peano() {
        let p = plan(&host(), None, None).unwrap();
        assert_eq!(p.version, RECOMMENDED);
        assert_eq!(
            p.dir,
            PathBuf::from("/home/u/.local/share/rlx/toolchains/mlir-aie-1.4.4")
        );
        let flat: Vec<String> = p.commands.iter().map(|c| c.join(" ")).collect();
        assert!(
            flat[0]
                .starts_with("python3 -m venv /home/u/.local/share/rlx/toolchains/mlir-aie-1.4.4")
        );
        assert!(flat[2].contains("mlir_aie==1.4.4") && flat[2].contains("expanded_assets/v1.4.4"));
        assert!(flat[3].contains("llvm-aie==22.0.0.2026092101"));
    }

    /// The setup directory is where discovery looks first.
    #[test]
    fn setup_installs_where_discovery_looks_first() {
        let p = plan(&host(), Some((1, 4, 2)), Some("python3.14")).unwrap();
        assert!(p.dir.starts_with(managed_dir(&host()).unwrap()));
        assert_eq!(p.peano, "21.0.0.2026080301");
        assert!(p.commands[0][0] == "python3.14");
    }

    #[test]
    fn unknown_or_broken_releases_are_refused_with_the_list() {
        let e = plan(&host(), Some((1, 3, 3)), None).unwrap_err();
        assert!(e.contains("installable: 1.4.4"), "{e}");
        assert!(!installable().contains(&(1, 3, 3)));
    }

    #[test]
    fn xrt_instructions_follow_the_distro() {
        let r = std::env::temp_dir().join(format!("rlx_setup_os_{}", std::process::id()));
        std::fs::create_dir_all(r.join("etc")).unwrap();
        std::fs::write(
            r.join("etc/os-release"),
            "ID=ubuntu\nVERSION_ID=\"26.04\"\n",
        )
        .unwrap();
        assert!(xrt_instructions(&r)[0].contains("apt install libxrt2"));
        std::fs::write(r.join("etc/os-release"), "ID=fedora\n").unwrap();
        assert!(xrt_instructions(&r)[0].contains("dnf"));
        let _ = std::fs::remove_dir_all(&r);
    }
}
