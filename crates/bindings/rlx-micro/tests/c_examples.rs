// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The C and C++ examples, against `librlx_micro.a`: built and run on the host
//! (`cc` / `c++`), then — when `arm-none-eabi-gcc` and `qemu-system-arm` are
//! installed and the `thumbv7em-none-eabihf` target is — cross-built as
//! bare-metal Cortex-M4 firmware and run in QEMU's `mps2-an386`, output
//! through semihosting. Each prints the same transcript.

use std::path::{Path, PathBuf};
use std::process::Command;

const EXPECT: &[&str] = &[
    concat!("rlx-micro ", env!("CARGO_PKG_VERSION")),
    "mnist: digit 7 (label 7)",
    "dense: 1.00 4.00",
];

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn has(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Build the staticlib for `target` (None = host) into its own target dir, so
/// the build does not wait on the workspace's lock; returns the archive.
fn staticlib(target: Option<&str>) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rlx-micro");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args([
        "build",
        "-q",
        "--release",
        "-p",
        "rlx-micro",
        "--target-dir",
    ])
    .arg(&dir);
    if let Some(t) = target {
        cmd.args(["--target", t]);
    }
    let st = cmd.status().expect("cargo");
    assert!(st.success(), "building rlx-micro for {target:?}");
    let mut lib = dir;
    if let Some(t) = target {
        lib.push(t);
    }
    lib.join("release").join("librlx_micro.a")
}

fn check(what: &str, out: std::process::Output) {
    // QEMU sends semihosting output to stderr; a host program to stdout.
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{what} failed ({}):\n{text}",
        out.status
    );
    for line in EXPECT {
        assert!(text.contains(line), "{what}: missing '{line}' in:\n{text}");
    }
}

fn compile(cc: &str, args: &[&str], out: &Path) {
    let o = Command::new(cc)
        .args(args)
        .arg("-o")
        .arg(out)
        .current_dir(crate_dir())
        .output()
        .expect(cc);
    assert!(
        o.status.success(),
        "{cc} {args:?}:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

#[cfg(not(windows))]
#[test]
fn c_and_cpp_examples_run_on_the_host() {
    if !has("cc") || !has("c++") {
        eprintln!("skipping: no cc / c++");
        return;
    }
    let lib = staticlib(None);
    let lib = lib.to_str().unwrap();
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let port = tmp.join("rlx-micro-port_host.o");
    compile(
        "cc",
        &["-std=c11", "-Iexamples", "-c", "examples/port_host.c"],
        &port,
    );
    let sys: &[&str] = if cfg!(target_os = "linux") {
        &["-lpthread", "-ldl", "-lm"]
    } else {
        &[]
    };
    for (cc, std, src) in [
        ("cc", "-std=c11", "examples/mnist.c"),
        ("c++", "-std=c++17", "examples/mnist.cpp"),
    ] {
        let exe = tmp.join(format!(
            "rlx-micro-{}",
            Path::new(src).extension().unwrap().to_str().unwrap()
        ));
        let mut args = vec![
            std,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Iinclude",
            "-Iexamples",
        ];
        args.extend([src, port.to_str().unwrap()]);
        args.push(lib);
        args.extend(sys);
        compile(cc, &args, &exe);
        check(src, Command::new(&exe).output().unwrap());
    }
}

#[test]
fn c_and_cpp_examples_run_bare_metal_on_a_cortex_m4() {
    const TARGET: &str = "thumbv7em-none-eabihf";
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .is_ok_and(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .any(|l| l == TARGET)
        });
    if !(has("arm-none-eabi-gcc") && has("qemu-system-arm") && installed) {
        eprintln!(
            "skipping: needs arm-none-eabi-gcc, qemu-system-arm and `rustup target add {TARGET}`"
        );
        return;
    }
    let lib = staticlib(Some(TARGET));
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rlx-micro-mcu");
    std::fs::create_dir_all(&tmp).unwrap();
    let o = |n: &str| tmp.join(n).to_str().unwrap().to_string();
    let cpu = [
        "-mcpu=cortex-m4",
        "-mthumb",
        "-mfloat-abi=hard",
        "-mfpu=fpv4-sp-d16",
    ];
    let cflags = [
        &cpu[..],
        &[
            "-ffreestanding",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wall",
            "-Wextra",
            "-Werror",
        ],
        &["-Iinclude", "-Iexamples", "-c"],
    ]
    .concat();
    let obj = |cc: &str, extra: &[&str], src: &str, out: &str| {
        let mut a = cflags.clone();
        a.extend(extra);
        a.push(src);
        compile(cc, &a, Path::new(out));
    };
    obj(
        "arm-none-eabi-gcc",
        &["-std=c11"],
        "examples/mcu/startup.c",
        &o("startup.o"),
    );
    obj(
        "arm-none-eabi-gcc",
        &["-std=c11"],
        "examples/mcu/semihost.c",
        &o("semihost.o"),
    );
    obj(
        "arm-none-eabi-gcc",
        &["-std=c11"],
        "examples/mnist.c",
        &o("mnist_c.o"),
    );
    obj(
        "arm-none-eabi-g++",
        &[
            "-std=c++17",
            "-fno-exceptions",
            "-fno-rtti",
            "-fno-threadsafe-statics",
        ],
        "examples/mnist.cpp",
        &o("mnist_cpp.o"),
    );
    for e in ["c", "cpp"] {
        let elf = o(&format!("mnist_{e}.elf"));
        let mut a = cpu.to_vec();
        let (m, s, h) = (o(&format!("mnist_{e}.o")), o("startup.o"), o("semihost.o"));
        a.extend([
            "-nostdlib",
            "-Wl,--gc-sections",
            "-Wl,-z,noexecstack",
            "-Texamples/mcu/link.ld",
        ]);
        a.extend([
            m.as_str(),
            s.as_str(),
            h.as_str(),
            lib.to_str().unwrap(),
            "-lgcc",
        ]);
        compile("arm-none-eabi-gcc", &a, Path::new(&elf));
        let run = Command::new("qemu-system-arm")
            .args([
                "-M",
                "mps2-an386",
                "-nographic",
                "-monitor",
                "none",
                "-serial",
                "none",
            ])
            .args([
                "-semihosting-config",
                "enable=on,target=native",
                "-kernel",
                &elf,
            ])
            .output()
            .unwrap();
        check(&format!("mnist.{e} on Cortex-M4 (QEMU)"), run);
    }
}
