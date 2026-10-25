// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generated FFI-language bindings — Go, Java, Ruby, Julia, Zig, C#, C++,
//! Node, Kotlin, Lua and the rest listed in `crates/bindings/README.md` —
//! against the library cargo just built: each one's test suite, and its five
//! examples (inference, training, loading, distributed, backends), which must
//! print the same answers as every other binding's.
//!
//! A language runs natively when its toolchain is on the machine; otherwise,
//! with `RLX_TEST_DOCKER=1`, in its official container image against a Linux
//! build of the library in the `rlx-linux-target` volume (`just
//! linux-capi-volume`); otherwise it is skipped with a note.

use std::path::{Path, PathBuf};
use std::process::Command;

mod support;
use support::{plugin_lib, target_dir};

fn bindings() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn lib_file() -> PathBuf {
    let name = if cfg!(target_os = "macos") {
        "librlx_capi.dylib"
    } else if cfg!(windows) {
        "rlx_capi.dll"
    } else {
        "librlx_capi.so"
    };
    target_dir().join(name)
}

/// The first of `candidates` that runs (`--version`, `version` or `-v`), if any.
fn tool(candidates: &[&str]) -> Option<String> {
    candidates
        .iter()
        .find(|c| {
            ["--version", "version", "-v"].iter().any(|flag| {
                Command::new(c)
                    .arg(flag)
                    .output()
                    .is_ok_and(|o| o.status.success())
            })
        })
        .map(|c| c.to_string())
}

/// The plugin in the `rlx-linux-target` volume, for container runs.
const LINUX_PLUGIN: &str = "RLX_MODEL_PLUGINS=/target/debug/librlx_model_plugin.so";

/// Lines each example must print — the same in every language.
const EXPECT: &[(&str, &[&str])] = &[
    ("inference", &["compiled on: cpu", "probs sum: 1.0000"]),
    (
        "training",
        &[
            "trainer w: 1.999 -0.999 0.498",
            "checkpoint: 3 tensors",
            "manual loss: 0.000000 w: 2.000 -1.000 0.500",
        ],
    ),
    (
        "loading",
        &["gguf architecture: demo", "q8 max|Δ|: 0.0039", "argmax: 1"],
    ),
    (
        "distributed",
        &[
            "rank 0 w: 1.500 -2.000 0.500",
            "rank 1 w agrees: true",
            "rank 2 w agrees: true",
        ],
    ),
    (
        "backends",
        &["fastest for graph: cpu", "router ran on: cpu"],
    ),
    (
        "models",
        &[
            "model: affine (graph) on cpu",
            "model run: 2.000 3.000",
            "plugin model: count (text-generation)",
            "model text: defg",
            "model ids: 3 4 5",
            "chat: defg",
            "voice: 5 samples at 16000 Hz",
            "heard: hello",
            "embed: 5 2 3 0",
        ],
    ),
];

/// A C compiler: `CC`, else `cc`.
fn cc() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    tool(&[&cc])
}

/// `pkg-config --cflags` of the first package found.
fn pkg_cflags(packages: &[&str]) -> Option<Vec<String>> {
    packages.iter().find_map(|p| {
        let out = Command::new("pkg-config")
            .args(["--cflags", p])
            .output()
            .ok()?;
        out.status.success().then(|| {
            String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .map(str::to_string)
                .collect()
        })
    })
}

/// Link flags for a loadable module against librlx_capi in `target_dir()`.
fn module_link_args() -> Vec<String> {
    let lib = target_dir();
    let mut args = vec![
        format!("-L{}", lib.display()),
        "-lrlx_capi".to_string(),
        format!("-Wl,-rpath,{}", lib.display()),
    ];
    if cfg!(target_os = "macos") {
        args.splice(
            0..0,
            ["-bundle", "-undefined", "dynamic_lookup"].map(String::from),
        );
    } else {
        args.insert(0, "-shared".to_string());
    }
    args
}

/// `n` loopback `host:port` addresses that were free a moment ago.
fn loopback_peers(n: usize) -> String {
    let listeners: Vec<_> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").expect("bind"))
        .collect();
    listeners
        .iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// The distributed example as one process per rank, for languages without
/// threads: `cmd(rank, world, peers)` builds a rank; rank 0's output returned.
fn run_ranks(what: &str, cmd: impl Fn(&str, &str, &str) -> Command) -> String {
    let peers = loopback_peers(3);
    let mut workers: Vec<_> = (1..3)
        .map(|r| {
            cmd(&r.to_string(), "3", &peers)
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap_or_else(|e| panic!("{what} rank {r}: {e}"))
        })
        .collect();
    let text = run(what, &mut cmd("0", "3", &peers));
    for w in &mut workers {
        assert!(
            w.wait().expect("worker").success(),
            "{what}: a worker failed"
        );
    }
    text
}

/// Whether languages without a local toolchain run in containers.
fn docker_enabled() -> bool {
    ["RLX_TEST_DOCKER", "RLX_TEST_DOTNET_DOCKER"]
        .iter()
        .any(|v| rlx_ir::env::read(v).is_ok_and(|v| v == "1"))
}

/// Run a package's `script` (POSIX sh, in the package directory, with
/// `RLX_CAPI_LIB` set): natively when one of `tools` is installed, else in
/// `image` when [`docker_enabled`]; `None` (skipped) otherwise.
fn native_or_docker(
    lang: &str,
    tools: &[&str],
    image: &str,
    dir: &Path,
    script: &str,
) -> Option<String> {
    let script = format!("set -e\n{script}");
    if tool(tools).is_some() {
        return Some(run(
            lang,
            Command::new("sh")
                .args(["-c", &script])
                .current_dir(dir)
                .env("RLX_CAPI_LIB", lib_file())
                .env("RLX_CAPI_LIB_DIR", target_dir()),
        ));
    }
    if !docker_enabled() {
        eprintln!(
            "skip: no {} (set RLX_TEST_DOCKER=1 to use {image})",
            tools[0]
        );
        return None;
    }
    Some(run(
        lang,
        Command::new("docker")
            .args(["run", "--rm", "-v"])
            .arg(format!("{}:/pkg", dir.display()))
            .args([
                "-v",
                "rlx-linux-target:/target:ro",
                "-e",
                "RLX_CAPI_LIB=/target/debug/librlx_capi.so",
                "-e",
                "RLX_CAPI_LIB_DIR=/target/debug",
                "-e",
                LINUX_PLUGIN,
                "-w",
                "/pkg",
                image,
                "sh",
                "-c",
                &script,
            ]),
    ))
}

/// The examples' loop for a [`native_or_docker`] script: `== <example>`
/// before each one's output; `cmd` runs `$e`.
fn each_example(cmd: &str) -> String {
    let names: Vec<&str> = EXPECT.iter().map(|(e, _)| *e).collect();
    format!(
        "for e in {}; do echo \"== $e\"; {cmd}; done",
        names.join(" ")
    )
}

/// [`each_example`] for a language whose mesh ranks are processes: the
/// distributed example runs as three, `rank` with `$r` (rank), `$w` (world)
/// and `$P` (comma-separated loopback peers) set.
fn each_example_ranks(single: &str, rank: &str) -> String {
    let names: Vec<&str> = EXPECT.iter().map(|(e, _)| *e).collect();
    format!(
        "for e in {names}; do echo \"== $e\"\n\
         if [ \"$e\" = distributed ]; then\n\
         \x20 b=$((20000 + $$ % 30000)); P=127.0.0.1:$b,127.0.0.1:$((b+1)),127.0.0.1:$((b+2)); w=3\n\
         \x20 r=1; {rank} >/dev/null & p1=$!\n\
         \x20 r=2; {rank} >/dev/null & p2=$!\n\
         \x20 r=0; {rank}\n\
         \x20 wait $p1; wait $p2\n\
         else {single}; fi; done",
        names = names.join(" ")
    )
}

/// Check a [`each_example`] transcript: the test summary before the first
/// example contains `passed`, and each example printed its lines.
fn check_transcript(lang: &str, text: &str, passed: &str) {
    let (tests, rest) = text.split_once("\n== ").unwrap_or((text, ""));
    assert!(
        tests.contains(passed),
        "{lang} tests: no `{passed}` in:\n{tests}"
    );
    for section in format!("\n== {rest}").split("\n== ").skip(1) {
        let (example, out) = section.split_once('\n').unwrap_or((section, ""));
        expect(lang, example.trim(), out);
    }
    for (example, _) in EXPECT {
        assert!(
            text.contains(&format!("== {example}")),
            "{lang}: example {example} did not run:\n{text}"
        );
    }
}

fn run(what: &str, cmd: &mut Command) -> String {
    // Every language finds the reference plugin the same way: the variable
    // the runtime reads at the first model call.
    cmd.env("RLX_MODEL_PLUGINS", plugin_lib());
    let out = cmd.output().unwrap_or_else(|e| panic!("{what}: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{what} failed:\n{text}");
    text
}

fn expect(lang: &str, example: &str, text: &str) {
    let wants = EXPECT.iter().find(|(e, _)| *e == example).unwrap().1;
    let lower = text.to_lowercase();
    for want in wants {
        assert!(
            lower.contains(&want.to_lowercase()),
            "{lang} {example}: missing `{want}` in:\n{text}"
        );
    }
}

#[test]
fn go_binding_tests_and_examples() {
    let Some(go) = tool(&["go"]) else {
        eprintln!("skip: no go toolchain");
        return;
    };
    let dir = bindings().join("rlx-go");
    let ld = format!("-L{0} -lrlx_capi -Wl,-rpath,{0}", target_dir().display());
    run(
        "go test",
        Command::new(&go)
            .args(["test", "./..."])
            .current_dir(&dir)
            .env("CGO_LDFLAGS", &ld),
    );
    for (example, _) in EXPECT {
        let text = run(
            &format!("go run {example}"),
            Command::new(&go)
                .args(["run", &format!("./examples/{example}")])
                .current_dir(&dir)
                .env("CGO_LDFLAGS", &ld),
        );
        expect("go", example, &text);
    }
}

#[test]
fn java_binding_tests_and_examples() {
    // `JAVA_HOME` first: a keg-only or side-by-side JDK is often not on PATH.
    let from_home = std::env::var("JAVA_HOME")
        .map(|h| format!("{h}/bin/javac"))
        .unwrap_or_default();
    let Some(javac) = tool(&[from_home.as_str(), "javac"]) else {
        eprintln!("skip: no JDK");
        return;
    };
    let java = javac.replace("javac", "java");
    let dir = bindings().join("rlx-java");
    let classes = std::env::temp_dir().join(format!("rlx-java-classes-{}", std::process::id()));
    let mut sources = Vec::new();
    for sub in [
        "src/main/java/org/rlx",
        "src/test/java/org/rlx",
        "examples/org/rlx/examples",
    ] {
        for e in std::fs::read_dir(dir.join(sub)).unwrap() {
            sources.push(e.unwrap().path());
        }
    }
    run(
        "javac",
        Command::new(&javac).arg("-d").arg(&classes).args(&sources),
    );
    let java_cmd = |main: &str, args: &[&str]| {
        let mut c = Command::new(&java);
        c.arg("--enable-native-access=ALL-UNNAMED")
            .arg(format!("-Drlx.capi.lib={}", lib_file().display()))
            .arg("-cp")
            .arg(&classes)
            .arg(main)
            .args(args);
        c
    };
    let text = run("RLXTest", &mut java_cmd("org.rlx.RLXTest", &[]));
    assert!(text.contains("checks passed"), "{text}");
    for (example, _) in EXPECT {
        let mut name = example.to_string();
        name[..1].make_ascii_uppercase();
        let text = run(
            example,
            &mut java_cmd(&format!("org.rlx.examples.{name}"), &[]),
        );
        expect("java", example, &text);
    }
    std::fs::remove_dir_all(&classes).ok();
}

#[test]
fn ruby_binding_tests_and_examples() {
    let Some(ruby) = tool(&["ruby"]) else {
        eprintln!("skip: no ruby");
        return;
    };
    let dir = bindings().join("rlx-ruby");
    let ruby_cmd = |script: PathBuf| {
        let mut c = Command::new(&ruby);
        c.arg("-I")
            .arg(dir.join("lib"))
            .arg(script)
            .env("RLX_CAPI_LIB", lib_file());
        c
    };
    let text = run("ruby tests", &mut ruby_cmd(dir.join("test/test_rlx.rb")));
    assert!(text.contains("0 failures, 0 errors"), "{text}");
    for (example, _) in EXPECT {
        let text = run(
            example,
            &mut ruby_cmd(dir.join(format!("examples/{example}.rb"))),
        );
        expect("ruby", example, &text);
    }
}

#[test]
fn julia_binding_tests_and_examples() {
    let Some(julia) = tool(&["julia"]) else {
        eprintln!("skip: no julia");
        return;
    };
    let dir = bindings().join("rlx-julia");
    let julia_cmd = |script: PathBuf| {
        let mut c = Command::new(&julia);
        c.arg("-t")
            .arg("3")
            .arg(format!("--project={}", dir.display()))
            .arg(script)
            .env("RLX_CAPI_LIB", lib_file());
        c
    };
    run("julia tests", &mut julia_cmd(dir.join("test/runtests.jl")));
    for (example, _) in EXPECT {
        let text = run(
            example,
            &mut julia_cmd(dir.join(format!("examples/{example}.jl"))),
        );
        expect("julia", example, &text);
    }
}

#[test]
fn zig_binding_tests_and_examples() {
    let Some(zig) = tool(&["zig"]) else {
        eprintln!("skip: no zig");
        return;
    };
    let dir = bindings().join("rlx-zig");
    let lib_dir = format!("-Dlib-dir={}", target_dir().display());
    run(
        "zig build test",
        Command::new(&zig)
            .args(["build", "test", &lib_dir])
            .current_dir(&dir),
    );
    for (example, _) in EXPECT {
        let text = run(
            example,
            Command::new(&zig)
                .args(["build", example, &lib_dir])
                .current_dir(&dir),
        );
        expect("zig", example, &text);
    }
}

#[test]
fn csharp_binding_tests_and_examples() {
    let dir = bindings().join("rlx-csharp");
    let docker = docker_enabled();
    let dotnet = tool(&["dotnet"]);
    let in_container = |workdir: &str, script: &str| {
        let mut c = Command::new("docker");
        c.args(["run", "--rm", "-v"])
            .arg(format!("{}:/src", dir.display()))
            .args([
                "-v",
                "rlx-linux-target:/target:ro",
                "-e",
                "RLX_CAPI_LIB=/target/debug/librlx_capi.so",
                "-e",
                LINUX_PLUGIN,
                "-w",
                workdir,
                "mcr.microsoft.com/dotnet/sdk:8.0",
                "bash",
                "-c",
                script,
            ]);
        c
    };
    let (tests, examples): (Box<dyn Fn() -> Command>, Box<dyn Fn(&str) -> Command>) =
        match (dotnet, docker) {
            (Some(dotnet), _) => {
                let d1 = dotnet.clone();
                let dir1 = dir.clone();
                let dir2 = dir.clone();
                (
                    Box::new(move || {
                        let mut c = Command::new(&d1);
                        c.args(["run"])
                            .current_dir(dir1.join("tests/Rlx.Tests"))
                            .env("RLX_CAPI_LIB", lib_file());
                        c
                    }),
                    Box::new(move |e: &str| {
                        let mut c = Command::new(&dotnet);
                        c.args(["run", "--", e])
                            .current_dir(dir2.join("examples/Rlx.Examples"))
                            .env("RLX_CAPI_LIB", lib_file());
                        c
                    }),
                )
            }
            (None, true) => (
                Box::new(move || in_container("/src/tests/Rlx.Tests", "dotnet run -v q")),
                Box::new(move |e: &str| {
                    in_container(
                        "/src/examples/Rlx.Examples",
                        &format!("dotnet run -v q -- {e}"),
                    )
                }),
            ),
            (None, false) => {
                eprintln!("skip: no dotnet (set RLX_TEST_DOCKER=1 to use the .NET SDK container)");
                return;
            }
        };
    let text = run("dotnet tests", &mut tests());
    assert!(text.contains("checks passed"), "{text}");
    for (example, _) in EXPECT {
        let text = run(example, &mut examples(example));
        expect("c#", example, &text);
    }
}

#[test]
fn cpp_binding_tests_and_examples() {
    let Some(cmake) = tool(&["cmake"]) else {
        eprintln!("skip: no cmake");
        return;
    };
    let dir = bindings().join("rlx-cpp");
    let build = target_dir().join("rlx-cpp");
    run(
        "cmake configure",
        Command::new(&cmake)
            .arg("-S")
            .arg(&dir)
            .arg("-B")
            .arg(&build)
            .arg(format!("-DRLX_CAPI_LIB_DIR={}", target_dir().display())),
    );
    run(
        "cmake build",
        Command::new(&cmake).arg("--build").arg(&build).arg("-j4"),
    );
    let text = run("c++ tests", &mut Command::new(build.join("test_rlx")));
    assert!(text.contains("checks passed"), "{text}");
    for (example, _) in EXPECT {
        let text = run(example, &mut Command::new(build.join(example)));
        expect("c++", example, &text);
    }
}

#[test]
fn node_binding_tests_and_examples() {
    let (Some(node), Some(npm)) = (tool(&["node"]), tool(&["npm"])) else {
        eprintln!("skip: no node/npm");
        return;
    };
    let dir = bindings().join("rlx-node");
    if !dir.join("node_modules/koffi").exists() {
        run(
            "npm install",
            Command::new(&npm)
                .args(["install", "--no-audit", "--no-fund"])
                .current_dir(&dir),
        );
    }
    run(
        "npm test",
        Command::new(&npm)
            .arg("test")
            .current_dir(&dir)
            .env("RLX_CAPI_LIB", lib_file()),
    );
    for (example, _) in EXPECT {
        let text = run(
            example,
            Command::new(&node)
                .arg(format!("examples/{example}.js"))
                .current_dir(&dir)
                .env("RLX_CAPI_LIB", lib_file()),
        );
        expect("node", example, &text);
    }
}

#[test]
fn kotlin_binding_tests_and_examples() {
    let (Some(gradle), Some(cc)) = (tool(&["gradle"]), cc()) else {
        eprintln!("skip: no gradle / C compiler");
        return;
    };
    // The JNI headers: `JAVA_HOME`, else the home of the `java` on PATH.
    let java_home = std::env::var_os("JAVA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            let out = Command::new("java")
                .args(["-XshowSettings:properties", "-version"])
                .output()
                .ok()?;
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .find_map(|l| l.trim().strip_prefix("java.home = ").map(PathBuf::from))
        });
    let Some(java_home) = java_home.filter(|h| h.join("include/jni.h").exists()) else {
        eprintln!("skip: no JDK (set JAVA_HOME)");
        return;
    };
    let include = java_home.join("include");
    let os_include = std::fs::read_dir(&include)
        .expect("JDK include dir")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("jni_md.h").exists())
        .expect("jni_md.h");
    let dir = bindings().join("rlx-kotlin");
    let out = target_dir().join("rlx-kotlin");
    std::fs::create_dir_all(&out).unwrap();
    let jni = out.join(if cfg!(target_os = "macos") {
        "librlx_jni.dylib"
    } else {
        "librlx_jni.so"
    });
    let lib = target_dir();
    run(
        "jni build",
        Command::new(&cc)
            .args(["-shared", "-fPIC", "-O1"])
            .arg(format!("-I{}", include.display()))
            .arg(format!("-I{}", os_include.display()))
            .arg(format!(
                "-I{}",
                bindings().join("rlx-capi/include").display()
            ))
            .arg(dir.join("src/main/c/rlx_jni.c"))
            .arg(dir.join("src/main/c/rlx_jni_support.c"))
            .arg(format!("-L{}", lib.display()))
            .arg("-lrlx_capi")
            .arg(format!("-Wl,-rpath,{}", lib.display()))
            .arg("-o")
            .arg(&jni),
    );
    let prop = format!("-Prlx.jni.lib={}", jni.display());
    let text = run(
        "gradle rlxTest",
        Command::new(&gradle)
            .args(["-q", "rlxTest", &prop])
            .current_dir(&dir),
    );
    assert!(text.contains("checks passed"), "{text}");
    for (example, _) in EXPECT {
        let text = run(
            example,
            Command::new(&gradle)
                .args(["-q", "example", &format!("-Pname={example}"), &prop])
                .current_dir(&dir),
        );
        expect("kotlin", example, &text);
    }
}

#[test]
fn lua_binding_tests_and_examples() {
    let (Some(lua), Some(cc)) = (tool(&["lua", "lua5.5", "lua5.4"]), cc()) else {
        eprintln!("skip: no lua / C compiler");
        return;
    };
    let Some(cflags) = std::env::var("LUA_INCDIR")
        .ok()
        .map(|d| vec![format!("-I{d}")])
        .or_else(|| pkg_cflags(&["lua", "lua5.5", "lua5.4", "lua-5.4"]))
    else {
        eprintln!("skip: no Lua headers (set LUA_INCDIR)");
        return;
    };
    let dir = bindings().join("rlx-lua");
    let out = target_dir().join("rlx-lua");
    std::fs::create_dir_all(&out).unwrap();
    run(
        "lua module build",
        Command::new(&cc)
            .args(["-fPIC", "-O1"])
            .args(&cflags)
            .arg(format!(
                "-I{}",
                bindings().join("rlx-capi/include").display()
            ))
            .arg(dir.join("src/rlx_core.c"))
            .arg(dir.join("src/rlx_lua_support.c"))
            .args(module_link_args())
            .arg("-o")
            .arg(out.join("rlx_core.so")),
    );
    let lua_cmd = |script: &str| {
        let mut c = Command::new(&lua);
        c.arg(dir.join(script))
            .current_dir(&dir)
            .env("LUA_PATH", format!("{}/?.lua;;", dir.display()))
            .env("LUA_CPATH", format!("{}/?.so;;", out.display()));
        c
    };
    let text = run("lua tests", &mut lua_cmd("test/test_rlx.lua"));
    assert!(text.contains("checks passed"), "{text}");
    for (example, _) in EXPECT {
        let script = format!("examples/{example}.lua");
        let text = if *example == "distributed" {
            run_ranks(example, |r, w, p| {
                let mut c = lua_cmd(&script);
                c.args([r, w, p]);
                c
            })
        } else {
            run(example, &mut lua_cmd(&script))
        };
        expect("lua", example, &text);
    }
}

#[test]
fn octave_binding_tests_and_examples() {
    let (Some(octave), Some(mkoctfile)) = (tool(&["octave"]), tool(&["mkoctfile"])) else {
        eprintln!("skip: no octave / mkoctfile");
        return;
    };
    let dir = bindings().join("rlx-matlab");
    let out = target_dir().join("rlx-matlab");
    std::fs::create_dir_all(&out).unwrap();
    let lib = target_dir();
    run(
        "mex build",
        Command::new(&mkoctfile)
            .arg("--mex")
            .arg(format!(
                "-I{}",
                bindings().join("rlx-capi/include").display()
            ))
            .arg(dir.join("src/rlx_mex.c"))
            .arg(dir.join("src/rlx_mex_support.c"))
            .arg(format!("-L{}", lib.display()))
            .arg("-lrlx_capi")
            .arg(format!("-Wl,-rpath,{}", lib.display()))
            .arg("-o")
            .arg(out.join("rlx_mex.mex"))
            .current_dir(&out),
    );
    let path = [
        out.clone(),
        dir.clone(),
        dir.join("test"),
        dir.join("examples"),
    ]
    .iter()
    .map(|p| format!("'{}'", p.display()))
    .collect::<Vec<_>>()
    .join(", ");
    let eval = |call: &str| {
        let mut c = Command::new(&octave);
        c.args(["--no-gui", "--no-window-system", "-q", "--eval"])
            .arg(format!("addpath({path}); {call}"))
            .current_dir(&dir);
        c
    };
    let text = run("octave tests", &mut eval("test_rlx()"));
    assert!(text.contains("checks passed"), "{text}");
    for (example, _) in EXPECT {
        let text = if *example == "distributed" {
            run_ranks(example, |r, w, p| {
                eval(&format!("example_distributed({r}, {w}, '{p}')"))
            })
        } else {
            run(example, &mut eval(&format!("example_{example}()")))
        };
        expect("octave", example, &text);
    }
}

#[test]
fn dart_binding_tests_and_examples() {
    let dir = bindings().join("rlx-dart");
    let script = format!(
        "dart pub get >/dev/null\ndart test\n{}",
        each_example("dart run example/$e.dart")
    );
    if let Some(text) = native_or_docker("dart", &["dart"], "dart:stable", &dir, &script) {
        check_transcript("dart", &text, "All tests passed");
    }
}

#[test]
fn r_binding_tests_and_examples() {
    let dir = bindings().join("rlx-r");
    let script = format!(
        "Rscript -e 'quit(status = !requireNamespace(\"jsonlite\", quietly = TRUE))' 2>/dev/null || \
         {{ apt-get -qq update && apt-get -qq install -y r-cran-jsonlite; }} >/dev/null 2>&1\n\
         t=$(mktemp -d); cp -r . \"$t/rlx\"; mkdir \"$t/lib\"\n\
         R CMD INSTALL -l \"$t/lib\" \"$t/rlx\" >\"$t/install.log\" 2>&1 || {{ cat \"$t/install.log\"; exit 1; }}\n\
         export R_LIBS=\"$t/lib\"\n\
         Rscript tests/test_rlx.R\n{}",
        each_example_ranks(
            "Rscript inst/examples/$e.R",
            "Rscript inst/examples/distributed.R $r $w $P"
        )
    );
    if let Some(text) = native_or_docker("r", &["Rscript"], "r-base:latest", &dir, &script) {
        check_transcript("r", &text, "checks passed");
    }
}

#[test]
fn php_binding_tests_and_examples() {
    let dir = bindings().join("rlx-php");
    let php = "php -d ffi.enable=1";
    let script = format!(
        "php -m | grep -q FFI || {{ apt-get -qq update && apt-get -qq install -y libffi-dev && \
         docker-php-ext-install ffi; }} >/dev/null 2>&1\n\
         {php} tests/test_rlx.php\n{}",
        each_example_ranks(
            &format!("{php} examples/$e.php"),
            &format!("{php} examples/distributed.php $r $w $P")
        )
    );
    if let Some(text) = native_or_docker("php", &["php"], "php:8.4-cli", &dir, &script) {
        check_transcript("php", &text, "checks passed");
    }
}

#[test]
fn nim_binding_tests_and_examples() {
    let dir = bindings().join("rlx-nim");
    let script = format!(
        "o=$(mktemp -d)\n\
         export LD_LIBRARY_PATH=\"$RLX_CAPI_LIB_DIR\" DYLD_LIBRARY_PATH=\"$RLX_CAPI_LIB_DIR\"\n\
         N=\"nim c --hints:off -d:release --path:src --outdir:$o\"\n\
         $N -r tests/test_rlx.nim\n\
         for e in inference training loading backends distributed models; do $N examples/$e.nim >/dev/null; done\n{}",
        each_example_ranks("\"$o/$e\"", "\"$o/distributed\" $r $w $P")
    );
    if let Some(text) = native_or_docker("nim", &["nim"], "nimlang/nim:latest", &dir, &script) {
        check_transcript("nim", &text, "checks passed");
    }
}

#[test]
fn d_binding_tests_and_examples() {
    let dir = bindings().join("rlx-d");
    let script = format!(
        "command -v ldc2 >/dev/null || {{ apt-get -qq update && apt-get -qq install -y ldc gcc; }} >/dev/null 2>&1\n\
         o=$(mktemp -d)\n\
         L=\"-L-L$RLX_CAPI_LIB_DIR -L-lrlx_capi -L-rpath=$RLX_CAPI_LIB_DIR\"\n\
         ldc2 -i -Isource -od=$o/obj -of=$o/test_rlx tests/test_rlx.d $L && $o/test_rlx\n\
         for e in inference training loading backends distributed models; do ldc2 -i -Isource -od=$o/obj -of=$o/$e examples/$e.d $L; done\n{}",
        each_example_ranks("\"$o/$e\"", "\"$o/distributed\" $r $w $P")
    );
    if let Some(text) = native_or_docker("d", &["ldc2"], "debian:trixie", &dir, &script) {
        check_transcript("d", &text, "checks passed");
    }
}

#[test]
fn v_binding_tests_and_examples() {
    let dir = bindings().join("rlx-v");
    // The official V image is musl; the Linux library is glibc, so in a
    // container V is built on Debian.
    let script = format!(
        "command -v v >/dev/null || {{ apt-get -qq update && apt-get -qq install -y git make gcc libc6-dev && \
         git clone -q --depth 1 https://github.com/vlang/v /opt/v && make -C /opt/v && ln -s /opt/v/v /usr/local/bin/v; }} >/dev/null 2>&1\n\
         o=$(mktemp -d); mkdir \"$o/mods\"; ln -s \"$PWD\" \"$o/mods/rlx\"; export VMODULES=\"$o/mods\"\n\
         F=\"-cflags -L$RLX_CAPI_LIB_DIR -cflags -Wl,-rpath,$RLX_CAPI_LIB_DIR\"\n\
         v $F -o \"$o/test_rlx\" tests/test_rlx.v && \"$o/test_rlx\"\n\
         for e in inference training loading backends distributed models; do v $F -o \"$o/$e\" examples/$e.v; done\n{}",
        each_example_ranks("\"$o/$e\"", "\"$o/distributed\" $r $w $P")
    );
    if let Some(text) = native_or_docker("v", &["v"], "debian:trixie", &dir, &script) {
        check_transcript("v", &text, "checks passed");
    }
}

#[test]
fn ocaml_binding_tests_and_examples() {
    let dir = bindings().join("rlx-ocaml");
    let script = format!(
        "command -v opam >/dev/null && eval \"$(opam env 2>/dev/null)\"\n\
         o=$(mktemp -d); cp -r src \"$o/src\"\n\
         (cd \"$o/src\" && ocamlopt -I +unix -I +threads -c support.ml generated.ml rlx.ml && \
         ocamlopt -ccopt -I. -c rlx_stubs.c rlx_ml_support.c)\n\
         build() {{ ocamlopt -I +unix -I +threads unix.cmxa threads.cmxa -I \"$o/src\" \"$o/src/support.cmx\" \
         \"$o/src/generated.cmx\" \"$o/src/rlx.cmx\" \"$o/src/rlx_stubs.o\" \"$o/src/rlx_ml_support.o\" \"$1\" \
         -cclib \"-L$RLX_CAPI_LIB_DIR -lrlx_capi -Wl,-rpath,$RLX_CAPI_LIB_DIR\" -o \"$2\"; }}\n\
         cp test/test_rlx.ml examples/*.ml \"$o/\"\n\
         build \"$o/test_rlx.ml\" \"$o/test_rlx\" && \"$o/test_rlx\"\n\
         for e in inference training loading backends distributed models; do build \"$o/$e.ml\" \"$o/$e\"; done\n{}",
        each_example_ranks("\"$o/$e\"", "\"$o/distributed\" $r $w $P")
    );
    if let Some(text) = native_or_docker(
        "ocaml",
        &["ocamlopt"],
        "ocaml/opam:debian-ocaml-5.3",
        &dir,
        &script,
    ) {
        check_transcript("ocaml", &text, "checks passed");
    }
}

#[test]
fn elixir_binding_tests_and_examples() {
    let dir = bindings().join("rlx-elixir");
    // The ranks of the distributed example are BEAM processes in one VM.
    let script = format!(
        "o=$(mktemp -d); cp -r . \"$o/p\"; cd \"$o/p\"\n\
         make RLX_CAPI_LIB_DIR=\"$RLX_CAPI_LIB_DIR\" >/dev/null\n\
         mkdir -p ebin && elixirc -o ebin lib/rlx/*.ex\n\
         export RLX_NIF=\"$o/p/priv/rlx_nif\"\n\
         elixir -pa ebin test/test_rlx.exs\n{}",
        each_example("elixir -pa ebin examples/$e.exs")
    );
    if let Some(text) = native_or_docker("elixir", &["elixir"], "elixir:latest", &dir, &script) {
        check_transcript("elixir", &text, "checks passed");
    }
}

#[test]
fn odin_binding_tests_and_examples() {
    let dir = bindings().join("rlx-odin");
    // In a container: the pinned Odin release for its architecture (Odin
    // links with clang on Linux).
    let script = format!(
        "command -v odin >/dev/null || {{ apt-get -qq update && apt-get -qq install -y curl ca-certificates clang && \
         a=$(uname -m | sed 's/aarch64/arm64/; s/x86_64/amd64/') && \
         curl -sL https://github.com/odin-lang/Odin/releases/download/dev-2026-10/odin-linux-$a-dev-2026-10.tar.gz | tar -xz -C /opt && \
         ln -s /opt/odin-*/odin /usr/local/bin/odin; }} >/dev/null 2>&1\n\
         o=$(mktemp -d)\n\
         L=\"-extra-linker-flags:-L$RLX_CAPI_LIB_DIR -Wl,-rpath,$RLX_CAPI_LIB_DIR\"\n\
         odin build tests -collection:rlx=\"$PWD\" \"$L\" -out:\"$o/test_rlx\" && \"$o/test_rlx\"\n\
         for e in inference training loading backends distributed models; do \
         odin build examples/$e -collection:rlx=\"$PWD\" \"$L\" -out:\"$o/$e\"; done\n{}",
        each_example("\"$o/$e\"")
    );
    if let Some(text) = native_or_docker("odin", &["odin"], "debian:trixie", &dir, &script) {
        check_transcript("odin", &text, "checks passed");
    }
}

#[test]
fn haskell_binding_tests_and_examples() {
    let dir = bindings().join("rlx-haskell");
    // -O0: the generated module is large, and these runs check behavior.
    let script = format!(
        "o=$(mktemp -d); cp -r . \"$o/p\"; cd \"$o/p\"\n\
         cabal update -v0 >/dev/null 2>&1 || true\n\
         B=\"--extra-lib-dirs=$RLX_CAPI_LIB_DIR --ghc-options=-O0 --enable-tests\"\n\
         cabal build -v0 all $B > \"$o/build.log\" 2>&1 || {{ tail -40 \"$o/build.log\"; exit 1; }}\n\
         export LD_LIBRARY_PATH=\"$RLX_CAPI_LIB_DIR\" DYLD_LIBRARY_PATH=\"$RLX_CAPI_LIB_DIR\"\n\
         \"$(cabal list-bin test-rlx $B)\"\n\
         X=\"$(cabal list-bin rlx-example $B)\"\n{}",
        each_example_ranks("\"$X\" $e", "\"$X\" distributed $r $w $P")
    );
    if let Some(text) = native_or_docker("haskell", &["cabal"], "haskell:9", &dir, &script) {
        check_transcript("haskell", &text, "checks passed");
    }
}
