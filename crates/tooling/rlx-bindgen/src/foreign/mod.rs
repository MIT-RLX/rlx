// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bindings for a foreign C library, from its API header alone.
//!
//! rlx's own bindings come from rlx's tables; a library that is not rlx
//! has only its header. This reads one (`header`: the structures, the
//! callback types and the functions marked by the library's export macro,
//! their comments as docs) into a typed list, and an emitter per language
//! marshals by C type, one rule per type, so every language gets every
//! function: Python (ctypes), Julia (`ccall`), C# (P/Invoke), Go (cgo, its
//! declarations generated from the list, a trampoline per callback type),
//! Ruby (Fiddle), and through a plain C header of the list, Zig
//! (`@cImport`) and Swift (a module map). First written for libngspice
//! (`sharedspice.h`).
//!
//! ```text
//! rlx-bindgen foreign --header sharedspice.h --lib ngspice --name NgSpice \
//!     --export IMPEXP --define XSPICE --strip ngSpice_ --strip ng --out bindings
//! ```

pub mod cheader;
pub mod csharp;
pub mod golang;
pub mod header;
pub mod julia;
pub mod python;
pub mod ruby;
pub mod swift;
pub mod zig;

use std::path::PathBuf;

/// What a library is called, and how its header is read
#[derive(Clone, Debug)]
pub struct Config {
    /// The library's name: `lib<lib>.so`, `lib<lib>.dylib`, `<lib>.dll`,
    /// `-l<lib>`, and the Go package
    pub lib: String,
    /// The module (Julia, Ruby), namespace (C#) name: `NgSpice`
    pub name: String,
    /// The environment variable naming the library's path
    pub env: String,
    /// The header's file name, for the generated files' notes
    pub header_name: String,
    /// The macro in front of each exported function (`IMPEXP`)
    pub export: String,
    /// The macros taken as defined in `#ifdef`s
    pub defines: Vec<String>,
    /// The function name prefixes the wrappers drop (`ngSpice_`, `ng`)
    pub strip: Vec<String>,
}

impl Config {
    /// The emitters' own text with the library's names: written for
    /// libngspice, and every other library's names put in its place
    pub fn fill(&self, text: &str) -> String {
        text.replace("cargo run -p ngspice-bindgen", "rlx-bindgen foreign")
            .replace(
                "NGSPICE_SHARED_H",
                &format!("{}_SHARED_H", self.lib.to_uppercase()),
            )
            .replace("ngspice-bindgen", "rlx-bindgen foreign")
            .replace("NgSpiceShared", &format!("{}Shared", self.name))
            .replace("NgSpice", &self.name)
            .replace("NGSPICE_LIBRARY", &self.env)
            .replace("libngspice", &format!("lib{}", self.lib))
            .replace("sharedspice.h", &self.header_name)
            .replace("ngspice_shared", &format!("{}_shared", self.lib))
            .replace("ngspice", &self.lib)
    }

    /// A wrapper's name: the prefix dropped, snake case
    /// (`ngGet_Vec_Info` → `get_vec_info`)
    pub fn snake(&self, name: &str) -> String {
        let mut n = name;
        for p in &self.strip {
            if let Some(r) = n.strip_prefix(p.as_str()) {
                n = r;
                break;
            }
        }
        let mut o = String::new();
        let mut prev_lower = false;
        for c in n.chars() {
            if c.is_ascii_uppercase() {
                if prev_lower {
                    o.push('_');
                }
                o.push(c.to_ascii_lowercase());
            } else {
                o.push(c);
            }
            prev_lower = c.is_ascii_lowercase() || c.is_ascii_digit();
        }
        o
    }
}

/// The bindings of the library whose header is `text`: (path under the
/// output directory, text)
pub fn generate(text: &str, cfg: &Config) -> Vec<(PathBuf, String)> {
    let h = header::read(text, &cfg.export, &cfg.defines);
    let lib = &cfg.lib;
    vec![
        (
            PathBuf::from(format!("python/{lib}_shared.py")),
            python::emit(&h, cfg),
        ),
        (
            PathBuf::from(format!("julia/{}Shared.jl", cfg.name)),
            julia::emit(&h, cfg),
        ),
        (
            PathBuf::from(format!("csharp/{}.cs", cfg.name)),
            csharp::emit(&h, cfg),
        ),
        (
            PathBuf::from(format!("go/{lib}/{lib}.go")),
            golang::emit(&h, cfg),
        ),
        (PathBuf::from(format!("ruby/{lib}.rb")), ruby::emit(&h, cfg)),
        (
            PathBuf::from(format!("c/{lib}_shared.h")),
            cheader::emit(&h, cfg),
        ),
        (PathBuf::from(format!("zig/{lib}.zig")), zig::emit(&h, cfg)),
        (
            PathBuf::from(format!("swift/{}.swift", cfg.name)),
            swift::emit(&h, cfg),
        ),
        (
            PathBuf::from("swift/module.modulemap"),
            swift::modulemap(cfg),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = r#"
#define DEMO_API
typedef struct pt { double x; double y; } pt, *ppt;
/* a progress report */
typedef int (Progress)(char*, int, void*);
#ifdef WITH_EXTRA
/* only with the extra */
DEMO_API int demo_extra(void);
#endif
/* start: the callback gets each message */
DEMO_API int demo_init(Progress* cb, void* user);
/* the names of everything */
DEMO_API char** demo_names(void);
/* a point by name */
DEMO_API ppt demo_point(char* name);
"#;

    fn cfg(defines: &[&str]) -> Config {
        Config {
            lib: "demo".into(),
            name: "Demo".into(),
            env: "DEMO_LIBRARY".into(),
            header_name: "demo.h".into(),
            export: "DEMO_API".into(),
            defines: defines.iter().map(|s| s.to_string()).collect(),
            strip: vec!["demo_".into()],
        }
    }

    #[test]
    fn reads_the_header() {
        let h = header::read(HEADER, "DEMO_API", &[]);
        assert_eq!(h.structs.len(), 1);
        assert_eq!(h.callbacks.len(), 1);
        let names: Vec<&str> = h.funcs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["demo_init", "demo_names", "demo_point"]);
        assert_eq!(
            h.funcs[2].ret,
            header::Ty {
                base: "pt".into(),
                ptr: 1,
                konst: false
            }
        );
        assert_eq!(h.funcs[0].doc, "start: the callback gets each message");
        let h = header::read(HEADER, "DEMO_API", &["WITH_EXTRA".into()]);
        assert_eq!(h.funcs.len(), 4);
    }

    #[test]
    fn every_language_names_the_library() {
        for (path, text) in generate(HEADER, &cfg(&[])) {
            assert!(
                !text.contains("ngspice") && !text.contains("NgSpice"),
                "{}",
                path.display()
            );
            assert!(
                text.contains("demo") || text.contains("Demo"),
                "{}",
                path.display()
            );
            // (the module map names the header, not the functions)
            assert!(
                path.ends_with("module.modulemap")
                    || text.contains("demo_point")
                    || text.contains("DemoPoint"),
                "{}: every function",
                path.display()
            );
        }
    }

    #[test]
    fn wrapper_names() {
        let c = cfg(&[]);
        assert_eq!(c.snake("demo_init"), "init");
        let mut c2 = c.clone();
        c2.strip = vec!["ngSpice_".into(), "ng".into()];
        assert_eq!(c2.snake("ngGet_Vec_Info"), "get_vec_info");
        assert_eq!(c2.snake("ngCM_Input_Path"), "cm_input_path");
    }
}
